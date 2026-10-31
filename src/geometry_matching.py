"""Rotation-aware matching of simulated docking patterns; see notebook 03."""

import numpy as np
from numba import njit
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial import cKDTree


def load_designs(info):
    """Read saved pattern IDs/layouts and build their grid in camera pixels."""
    metadata = next((item for item in info if "patterns" in item and "settings" in item), None)
    if metadata is None:
        raise ValueError("Simulation designs are missing; regenerate the picks with notebook 02.")
    ids = np.array([int(key) for key in metadata["patterns"]])
    masks = [np.array([list(row) for row in rows]) == "1"
             for rows in metadata["patterns"].values()]
    # Center smaller layouts on a common canvas; padding represents empty sites.
    shape = np.maximum(3, np.max([mask.shape for mask in masks], axis=0))
    patterns = np.zeros((len(masks), *shape), dtype=bool)
    for pattern, mask in zip(patterns, masks):
        r, c = (shape - mask.shape) // 2
        pattern[r:r+mask.shape[0], c:c+mask.shape[1]] = mask
    pixel_size = next(item["Pixelsize"] for item in reversed(info) if "Pixelsize" in item)
    pitch = metadata["settings"]["spacing_nm"] / pixel_size
    rows, cols = np.indices(shape)
    grid = np.column_stack((cols.ravel()-(shape[1]-1)/2,
                            rows.ravel()-(shape[0]-1)/2)) * pitch
    return ids, patterns, grid


def rotation(angle):
    """Row-vector rotation: positive angle is from +x toward +y."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s], [-s, c]])


@njit(cache=True)
def _expectation(xy, variance, grid, weights, area):
    """Gaussian site responsibilities plus uniform background, in linear time in N."""
    n, k = len(xy), len(grid)
    responsibilities = np.empty((n, k+1))
    log_likelihood = 0.0
    for i in range(n):
        total = weights[k] / area
        responsibilities[i, k] = total
        for j in range(k):
            distance2 = (xy[i, 0]-grid[j, 0])**2 + (xy[i, 1]-grid[j, 1])**2
            value = weights[j] * np.exp(-distance2/(2*variance[i])) / (2*np.pi*variance[i])
            responsibilities[i, j] = value
            total += value
        total = max(total, 1e-300)
        responsibilities[i] /= total
        log_likelihood += np.log(total)
    return responsibilities, log_likelihood / n


def _fit_grid(xy, precision2, grid, pose, radius, max_iter, tolerance):
    """EM: infer site weights, weighted rigid pose and additional Gaussian width."""
    angle, center = pose[2], np.array(pose[:2])
    weights = np.r_[np.full(len(grid), 0.9/len(grid)), 0.1]
    extra, previous, converged = 0.025, -np.inf, False
    for iteration in range(max_iter):
        transformed = grid @ rotation(angle) + center
        variance = precision2 + extra**2
        resp, likelihood = _expectation(xy, variance, transformed, weights, np.pi*radius**2)
        if iteration > 5 and abs(likelihood-previous) < tolerance:
            converged = True
            break
        previous = likelihood
        weights = np.maximum(resp.mean(0), 1e-12)
        weights /= weights.sum()
        signal = resp[:, :-1]
        a = (signal / variance[:, None]).sum(0)
        b = (signal / variance[:, None]).T @ xy
        grid_mean, data_mean = a @ grid / a.sum(), b.sum(0) / a.sum()
        u, _, vt = np.linalg.svd((grid-grid_mean).T @ (b-a[:, None]*data_mean))
        rot = u @ np.diag([1, np.linalg.det(u @ vt)]) @ vt
        angle = np.arctan2(rot[0, 1], rot[0, 0])
        center = data_mean-grid_mean @ rot
        distance2 = ((xy[:, None, :] - (grid @ rot+center))**2).sum(2)
        mass, residual = signal.sum(1), (signal*distance2).sum(1)
        def width_loss(width):
            v = precision2 + width**2
            return np.sum(mass*np.log(v) + residual/(2*v))
        extra = minimize_scalar(width_loss, bounds=(0.0001, 0.08), method="bounded").x
    transformed = grid @ rotation(angle) + center
    resp, likelihood = _expectation(xy, precision2+extra**2, transformed, weights, np.pi*radius**2)
    return dict(center=center, angle=angle, weights=weights, extra_sigma=extra,
                log_likelihood=likelihood, converged=converged, iterations=iteration+1)


def _seeds(xy, precision2, grid, shape, angle_step, n_starts):
    """Search lattice angle/phase, then refine nearest-site alignment before EM."""
    tree = cKDTree(grid)
    lattice = grid.reshape(*shape, 2)
    basis = np.array([lattice[1, 0]-grid[0], lattice[0, 1]-grid[0]])
    pitch = min(np.linalg.norm(basis, axis=1))
    origin = xy.mean(0)
    weight = 1/(precision2+0.025**2)
    def loss(pose):
        rot = rotation(pose[2])
        distances, ids = tree.query((xy-pose[:2]) @ rot.T)
        delta = xy-(grid[ids] @ rot+pose[:2])
        keep = distances < pitch
        derivative = grid[ids] @ np.array([[-np.sin(pose[2]), np.cos(pose[2])],
                                          [-np.cos(pose[2]), -np.sin(pose[2])]])
        weighted = delta * (weight*keep)[:, None]
        gradient = -2*np.r_[weighted.mean(0), np.mean(np.sum(weighted*derivative, axis=1))]
        return np.mean(np.minimum(distances**2, pitch**2)*weight), gradient
    candidates = []
    for angle in np.deg2rad(np.arange(0, 180, angle_step)):
        for dr in (-0.5, 0, 0.5):
            for dc in (-0.5, 0, 0.5):
                center = origin + np.array([dr, dc]) @ basis @ rotation(angle)
                pose = np.r_[center, angle]
                candidates.append((loss(pose)[0], pose))
    starts = []
    for _, pose in sorted(candidates, key=lambda item: item[0]):
        if any(abs(np.angle(np.exp(2j*(pose[2]-p[2]))))/2 < np.deg2rad(12)
               and np.linalg.norm(pose[:2]-p[:2]) < pitch*0.7 for p in starts):
            continue
        fitted = minimize(loss, pose, jac=True, method="L-BFGS-B",
                          bounds=[(origin[0]-0.2, origin[0]+0.2),
                                  (origin[1]-0.2, origin[1]+0.2), (None, None)],
                          options={"maxiter": 50}).x
        starts.append(fitted)
        if len(starts) == n_starts:
            break
    return starts


def _matches(weights, patterns, threshold_fraction, n_fit):
    """Signed-intensity template scores; zero-padded edge shifts, never wrapping."""
    height, width = patterns.shape[1:]
    image = weights.reshape(height, width)
    bright = np.sort(image.ravel())[-10:].mean()
    threshold = threshold_fraction*bright
    populated = image > 1/n_fit
    baseline = image[populated].mean() if populated.any() else image.mean()
    candidates = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            shifted = np.zeros_like(image)
            r0, r1 = max(0, dr), min(height, height+dr)
            c0, c1 = max(0, dc), min(width, width+dc)
            shifted[r0:r1, c0:c1] = image[r0-dr:r1-dr, c0-dc:c1-dc]
            if image.sum()-shifted.sum() > 0.01*image.sum():
                continue
            for flip in (False, True):
                evidence = shifted[::-1, ::-1] if flip else shifted
                centered = evidence-baseline
                scores = np.sum((2*patterns.astype(float)-1)*centered, axis=(1, 2))
                scores /= max(np.abs(centered).sum(), 1e-12)
                observed = evidence > threshold
                for barcode, score in enumerate(scores):
                    candidates.append(dict(barcode=barcode, score=score, flip=flip, dr=dr, dc=dc,
                                           hamming=int(np.sum(observed != patterns[barcode])),
                                           signal=evidence.copy(), observed=observed.copy()))
    per_class = [max((m for m in candidates if m["barcode"] == k), key=lambda m: m["score"])
                 for k in range(len(patterns))]
    return sorted(per_class, key=lambda m: m["score"], reverse=True)


def equivalent_templates(patterns, grid, template):
    """Find designs identical up to translation and quarter-turn rotation."""
    reference = grid[patterns[template].ravel()]
    reference = reference-reference.mean(0)
    tree, equivalent = cKDTree(reference), []
    for index, pattern in enumerate(patterns):
        sites = grid[pattern.ravel()]
        if len(sites) == len(reference):
            sites = sites-sites.mean(0)
            if any(np.max(tree.query(sites @ rotation(angle))[0]) < 1e-8
                   for angle in np.arange(4)*np.pi/2):
                equivalent.append(index)
    return equivalent


def match_particle(locs, patterns, grid, *, class_ids=None, pick_center=None, radius=1.2,
                   max_fit_locs=2000, angle_step=10, n_starts=4, max_iter=150,
                   tolerance=1e-6, threshold_fraction=0.25, max_hamming=6,
                   min_gap=0.05, max_background=0.35, max_rms_pitch=0.8, seed=7):
    """Fit a picked particle. Scores/acceptance are diagnostics, not probabilities."""
    xy = locs[["x", "y"]].to_numpy(dtype=float)
    precision = locs[["lpx", "lpy"]].to_numpy(dtype=float)
    precision2 = np.mean(precision**2, axis=1)
    valid = np.isfinite(xy).all(1) & np.isfinite(precision).all(1) & (precision > 0).all(1)
    xy, precision2 = xy[valid], precision2[valid]
    if len(xy) < 50:
        raise ValueError("Fewer than 50 finite localizations with positive uncertainty.")
    origin = xy.mean(0) if pick_center is None else np.asarray(pick_center, dtype=float)
    xy -= origin
    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(len(xy), min(len(xy), max_fit_locs), replace=False))
    x, p = xy[sample], precision2[sample]
    coarse = np.linspace(0, len(x)-1, min(400, len(x)), dtype=int)
    starts = _seeds(x[coarse], p[coarse], grid, patterns.shape[1:], angle_step, n_starts)
    fits = [_fit_grid(x, p, grid, pose, radius, max_iter, tolerance) for pose in starts]
    fitted = max(fits, key=lambda fit: fit["log_likelihood"])
    matches = _matches(fitted["weights"][:-1], patterns, threshold_fraction, len(sample))
    best, runner_up = matches[0], matches[min(1, len(matches)-1)]
    class_ids = np.arange(len(patterns)) if class_ids is None else np.asarray(class_ids)
    equivalent = equivalent_templates(patterns, grid, best["barcode"])
    rot = rotation(fitted["angle"])
    lattice = grid.reshape(*patterns.shape[1:], 2)
    basis = np.array([lattice[1, 0]-grid[0], lattice[0, 1]-grid[0]])
    center = fitted["center"] - np.array([best["dr"], best["dc"]]) @ basis @ rot
    canonical_grid = grid @ rot + center
    angle = fitted["angle"]
    if best["flip"]:
        canonical_grid = canonical_grid[::-1]
        angle += np.pi
    distances = cKDTree(grid @ rot+fitted["center"]).query(xy)[0]
    resp, _ = _expectation(xy, precision2+fitted["extra_sigma"]**2,
                           grid @ rot+fitted["center"], fitted["weights"], np.pi*radius**2)
    signal_weight = 1-resp[:, -1]
    rms = np.sqrt(np.average(distances**2, weights=signal_weight))
    pitch = min(np.linalg.norm(basis, axis=1))
    gap = best["score"]-runner_up["score"] if len(matches) > 1 else np.inf
    crop_margin = radius-np.linalg.norm(canonical_grid[patterns[best["barcode"]].ravel()], axis=1).max()
    checks = {"not_converged": not fitted["converged"], "poor_grid_fit": rms/pitch > max_rms_pitch,
              "high_background": fitted["weights"][-1] > max_background,
              "too_many_bit_mismatches": best["hamming"] > max_hamming,
              "ambiguous_class": gap < min_gap or len(equivalent) > 1,
              "pattern_at_crop_edge": crop_margin < 0.02}
    reason = ";".join(k for k, failed in checks.items() if failed)
    summary = dict(n_locs=len(locs), n_valid=len(xy), n_fit=len(sample),
                   best_digit=int(class_ids[best["barcode"]]),
                   digit_id=int(class_ids[best["barcode"]]) if not reason else -1,
                   equivalent_digits="/".join(str(class_ids[k]) for k in equivalent),
                   runner_up=int(class_ids[runner_up["barcode"]]) if len(matches) > 1 else -1,
                   score=best["score"], score_gap=gap,
                   hamming=best["hamming"], background_fraction=fitted["weights"][-1],
                   rms_pitch=rms/pitch, extra_sigma_px=fitted["extra_sigma"],
                   center_x=center[0]+origin[0], center_y=center[1]+origin[1],
                   angle_deg=np.rad2deg(angle) % 360, crop_margin_px=crop_margin,
                   iterations=fitted["iterations"], accepted=not reason, reason=reason or "accepted")
    for candidate in matches:
        candidate["digit"] = int(class_ids[candidate["barcode"]])
    return dict(summary=summary, template_index=best["barcode"], grid_xy=canonical_grid+origin,
                signal=best["signal"], observed=best["observed"], matches=matches)
