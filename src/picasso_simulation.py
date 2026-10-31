"""Run notebook-defined docking patterns through native Picasso Simulate and Localize."""

from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import sys

import h5py
import numpy as np
import pandas as pd
from picasso import io, lib, localize, simulate
from scipy.spatial import cKDTree

TESTED_VERSIONS = {
    "picassosr": "0.11.3", "numpy": "2.5.4", "scipy": "1.18.1",
    "numba": "0.68.1", "h5py": "3.16.0", "pandas": "2.3.3",
    "PyYAML": "6.0.3", "matplotlib": "3.11.2",
}


@dataclass(frozen=True)
class SimulationSettings:
    """Acquisition and localization settings, supplied explicitly by the notebook."""
    frames: int = 40000
    image_size: int = 64
    particles_per_digit: int = 3
    spacing_nm: float = 20.0
    pixel_size_nm: float = 130.0
    integration_time_ms: float = 100.0
    mean_dark_ms: float = 54054.0
    mean_bright_ms: float = 280.0
    photon_rate_per_ms: float = 23.0
    photon_rate_std_per_ms: float = 0.0
    photon_budget: float = 1500000.0
    psf_sigma_px: float = 1.06
    background_photons_per_pixel: float = 78.0
    random_orientation: bool = False
    incorporation: float = 1.0
    seed: int = 20261031
    box_size: int = 7
    min_net_gradient: float = 500.0
    example_frames: int = 2000

    def validate(self):
        """Reject invalid acquisition values before calling Picasso."""
        for key in ("frames", "image_size", "particles_per_digit", "box_size", "example_frames"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.frames < 100 or self.box_size < 5 or self.box_size % 2 == 0:
            raise ValueError("Use at least 100 frames and an odd box_size >= 5")
        positive = ("spacing_nm", "pixel_size_nm", "integration_time_ms", "photon_rate_per_ms",
                    "photon_budget", "psf_sigma_px", "mean_dark_ms", "mean_bright_ms")
        for key in positive + ("photon_rate_std_per_ms", "background_photons_per_pixel", "min_net_gradient"):
            value = getattr(self, key)
            if not np.isfinite(value) or (value <= 0 if key in positive else value < 0):
                raise ValueError(f"Invalid {key}: {value}")
        if min(self.mean_dark_ms, self.mean_bright_ms) < 1 or not 0 < self.incorporation <= 1:
            raise ValueError("Mean times must be >= 1 ms and incorporation in (0, 1]")
        if type(self.random_orientation) is not bool:
            raise ValueError("random_orientation must be True or False")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")


def pattern_coordinates(rows, spacing_nm=20.0):
    """Convert a nonempty rectangular 0/1 pattern to centered nm coordinates (+y up)."""
    try:
        mask = np.asarray([list(row) for row in rows], dtype=str)
    except (ValueError, TypeError) as error:
        raise ValueError("A pattern must contain equally long rows of 0 and 1") from error
    if mask.ndim != 2 or not mask.size or not np.isin(mask, ("0", "1")).all() or not (mask == "1").any():
        raise ValueError("Use a rectangular binary pattern with at least one docking site (1)")
    row, col = np.nonzero(mask == "1")
    return np.column_stack((col - (mask.shape[1] - 1)/2, (mask.shape[0] - 1)/2 - row)) * spacing_nm


def _prepare_patterns(patterns, settings):
    """Validate class IDs and size the camera margin and particle crops from the designs."""
    if not isinstance(patterns, dict) or not patterns:
        raise ValueError("patterns must be a nonempty dictionary: integer ID -> binary rows")
    if any(type(key) is not int or not 0 <= key < 2**63 for key in patterns):
        raise ValueError("Pattern IDs must be nonnegative integers; -1 is reserved for unassigned fits")
    coordinates = {key: pattern_coordinates(rows, settings.spacing_nm) for key, rows in patterns.items()}
    xy = np.concatenate(list(coordinates.values()))
    radius = np.ceil(np.linalg.norm(xy, axis=1).max()/settings.spacing_nm) * settings.spacing_nm/settings.pixel_size_nm + 0.3
    extent = np.linalg.norm(xy, axis=1).max() if settings.random_orientation else np.abs(xy).max()
    margin = max(6.0, settings.box_size/2 + extent/settings.pixel_size_nm + 1)
    side = int(np.ceil(np.sqrt(len(patterns) * settings.particles_per_digit)))
    if settings.image_size <= 2*margin or (side > 1 and
            (settings.image_size - 2*margin)/(side - 1) < max(settings.box_size + 1, 2*radius)):
        raise ValueError("Increase image_size or reduce pattern count, size or particles_per_digit")
    return coordinates, margin, radius


def _place_structures(coordinates, settings, margin):
    """Use native rotation/incorporation; retain each designed site's presence and pose."""
    copies = settings.particles_per_digit
    centers = simulate.generatePositions(len(coordinates)*copies, settings.image_size, margin, arrangement=0)
    centers = centers[np.random.permutation(len(centers))]
    structures, particle_rows, template_rows, designed_rows = [], [], [], []
    template_ids, design_ids = [], []
    for index, (digit, xy) in enumerate(coordinates.items()):
        # Temporary sequence tags track sites through native incorporation; reset to 1 before imaging.
        defined = simulate.defineStructure(xy[:, 0], xy[:, 1], np.arange(1, len(xy) + 1), np.zeros(len(xy)),
                                          settings.pixel_size_nm, mean=False)
        start = index*copies
        subset = centers[start:start + copies]
        for offset, (cx, cy) in enumerate(subset):
            rotated = simulate.rotateStructure(defined) if settings.random_orientation else defined
            angle = _rotation_angle(defined, rotated) if settings.random_orientation else 0.0
            # Rotation is already native; orientation=0 here avoids rotating twice.
            placed = simulate.prepareStructures(rotated, subset[offset:offset + 1], orientation=0, number=1,
                                                 incorporation=settings.incorporation, exchange=0)
            kept = placed[2].astype(int) - 1
            present = np.isin(np.arange(len(xy)), kept)
            design_start, pid = len(designed_rows), start + offset
            world = rotated[:2].T + [cx, cy]
            designed_rows.extend((design_start + site, pid, digit, site, x, y, x - 0.5,
                                  settings.image_size - y - 0.5, bool(present[site]))
                                 for site, (x, y) in enumerate(world))
            template_ids.extend(kept)
            design_ids.extend(design_start + kept)
            placed[2], placed[3] = 1, pid
            structures.append(placed)
            particle_rows.append((pid, digit, cx, cy, cx - 0.5, settings.image_size - cy - 0.5,
                                  angle, (-angle) % 360, len(xy), len(kept), len(xy) - len(kept)))
        template_rows.extend((digit, site, x, y, 1, 0) for site, (x, y) in enumerate(xy))
    structures = np.concatenate(structures, axis=1)
    if structures.shape[1] == 0:
        raise ValueError("Picasso removed every docking site; increase incorporation")
    particles = pd.DataFrame(particle_rows, columns=["particle_id", "digit", "simulate_center_x_px",
        "simulate_center_y_px", "camera_center_x_px", "camera_center_y_px", "orientation_degrees",
        "camera_orientation_degrees", "designed_sites", "incorporated_sites", "missing_sites"])
    designed = pd.DataFrame(designed_rows, columns=["design_site_id", "particle_id", "digit", "template_site_id",
        "simulate_x_px", "simulate_y_px", "camera_x_px", "camera_y_px", "incorporated"])
    templates = pd.DataFrame(template_rows, columns=["digit", "template_site_id", "x_nm", "y_nm", "exchange_label", "z_nm"])
    particle_ids = structures[3].astype(int)
    sites = pd.DataFrame({"site_id": np.arange(len(particle_ids)), "template_site_id": template_ids,
        "design_site_id": design_ids, "particle_id": particle_ids,
        "digit": particles.digit.to_numpy()[particle_ids], "simulate_x_px": structures[0],
        "simulate_y_px": structures[1], "camera_x_px": structures[0] - 0.5,
        "camera_y_px": settings.image_size - structures[1] - 0.5, "exchange_label": structures[2].astype(int)})
    return structures, {"particles": particles, "sites": sites, "designed_sites": designed, "digit_templates": templates}


def _rotation_angle(original, rotated):
    """Read the native rotation from a tagged site; a centered single point has no observable angle."""
    site = np.linalg.norm(original[:2], axis=0).argmax()
    if np.linalg.norm(original[:2, site]) == 0:
        return np.nan
    angle = np.arctan2(rotated[1, site], rotated[0, site]) - np.arctan2(original[1, site], original[0, site])
    return float(np.degrees(angle) % 360)


def _render_movie(structures, settings):
    """Generate native blinking traces, sample the PSF and add native Poisson background."""
    photons = np.zeros((structures.shape[1], settings.frames), dtype=np.int32)
    for site in range(len(photons)):
        photons[site], _, _ = simulate.distphotons(
            structures, settings.integration_time_ms, settings.frames, settings.mean_dark_ms,
            settings.mean_bright_ms, settings.photon_rate_per_ms, settings.photon_rate_std_per_ms, settings.photon_budget)
    print(f"Picasso generated photon traces for {len(photons)} docking sites.")
    movie = np.empty((settings.frames, settings.image_size, settings.image_size), dtype="<u2")
    saturated = 0
    for frame in range(settings.frames):
        image = simulate.convertMovie(frame, photons, structures, settings.image_size, settings.frames,
            settings.psf_sigma_px, settings.photon_rate_per_ms, settings.background_photons_per_pixel,
            np.sqrt(settings.background_photons_per_pixel), False, [], [])
        image = simulate.noisy_p(image, settings.background_photons_per_pixel)
        saturated += int(np.count_nonzero(image >= 65535))
        movie[frame] = simulate.check_type(image)
        if (frame + 1) % max(1, settings.frames//8) == 0:
            print(f"Picasso rendered {frame + 1:,}/{settings.frames:,} frames.")
    return movie, photons, saturated


def _localize_movie(movie, temporary, settings):
    """Fit a temporary raw movie with Picasso, without passing ground-truth positions."""
    raw = temporary / "digits.raw"
    metadata = {"Generated by": f"Picasso v{version('picassosr')} Simulate (notebook)",
        "Byte Order": "<", "Data Type": "uint16", "Camera": "Simulation", "Frames": settings.frames,
        "Width": settings.image_size, "Height": settings.image_size, "Pixelsize": settings.pixel_size_nm,
        "Camera.Integration Time": settings.integration_time_ms, "Simulation settings": asdict(settings)}
    simulate.saveMovie(str(raw), movie, metadata)
    loaded, info = io.load_movie(str(raw))
    fitted, info = localize.localize(loaded, movie_info=info, fitting_method="gausslq", threaded=False,
        camera_info={"Baseline": 0, "Sensitivity": 1.0, "Gain": 1, "Qe": 1, "Pixelsize": settings.pixel_size_nm},
        identification_parameters={"Box Size": settings.box_size, "Min. Net Gradient": settings.min_net_gradient})
    fitted = lib.ensure_sanity(fitted, info).sort_values("frame", kind="stable").reset_index(drop=True)
    if fitted.empty:
        raise ValueError("No valid localizations; increase frames/brightness or review the detection threshold")
    if not fitted.frame.between(0, settings.frames - 1).all():
        raise ValueError("Picasso returned frames outside the acquisition")
    return fitted, info


def _label_particles(locs, truth, photons, frames, radius):
    """Associate fits with known particles for evaluation; leave ambiguous site/event origins unassigned."""
    particles, sites = truth["particles"], truth["sites"]
    distance, nearest = cKDTree(particles[["camera_center_x_px", "camera_center_y_px"]]).query(locs[["x", "y"]])
    within = distance <= radius
    active = np.zeros((len(particles), frames), dtype=np.int16)
    for pid in particles.particle_id:
        active[pid] = (photons[sites.particle_id.to_numpy() == pid] > 0).sum(axis=0)
    return pd.DataFrame({"localization_row": np.arange(len(locs)),
        "particle_id": np.where(within, nearest, -1),
        "digit": np.where(within, particles.digit.to_numpy()[nearest], -1),
        "distance_to_true_center_px": distance,
        "emitting_sites_in_associated_particle": np.where(within, active[nearest, locs.frame.to_numpy().astype(int)], -1)})


def _summarize(locs, truth, settings, saturated, requested):
    """Report acquisition/detection counts and precision, without claiming classification accuracy."""
    labels = truth["localization_labels"]
    return {"engine": "picasso.simulate -> raw movie -> picasso.localize.localize",
        "frames": settings.frames, "image_shape": [settings.image_size, settings.image_size],
        "particles": len(truth["particles"]), "particles_per_digit": settings.particles_per_digit,
        "random_orientation": settings.random_orientation, "incorporation_probability": settings.incorporation,
        "requested_digits": list(requested), "detected_digits": sorted(int(x) for x in labels.digit.unique() if x >= 0),
        "designed_docking_sites": len(truth["designed_sites"]),
        "incorporated_docking_sites": len(truth["sites"]), "saved_localizations": len(locs),
        "missing_docking_sites": len(truth["designed_sites"]) - len(truth["sites"]),
        "realized_incorporation": len(truth["sites"])/len(truth["designed_sites"]),
        "particles_with_missing_sites": int((truth["particles"].missing_sites > 0).sum()),
        "detected_frame_min": int(locs.frame.min()), "detected_frame_max": int(locs.frame.max()),
        "unassigned_localizations": int((labels.particle_id < 0).sum()),
        "associated_fits_with_multiple_emitting_sites": int((labels.emitting_sites_in_associated_particle > 1).sum()),
        "saturated_camera_pixels": saturated, "median_photons": float(locs.photons.median()),
        "median_lpx_px": float(locs.lpx.median()), "median_lpx_nm": float(locs.lpx.median()*settings.pixel_size_nm),
        "example_frames_in_memory": min(settings.example_frames, settings.frames),
        "scope": "Native rigid 2D rotation/site loss; physical settings are assumptions; no classification accuracy measured."}


def _plain(value):
    """Convert NumPy scalars and arrays to portable YAML/JSON values."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value.item() if isinstance(value, np.generic) else value


def _save_result(temporary, output_dir, locs, info, truth, settings, patterns, versions, summary):
    """Write the compact HDF5/YAML pair and verify it with Picasso before replacing saved outputs."""
    path = temporary / "digits_locs.hdf5"
    parameters = {"settings": asdict(settings), "software_versions": versions, "patterns": patterns}
    summary["picasso_reload_identical"] = True
    with h5py.File(path, "w") as handle:
        record = locs.to_records(index=False)
        handle.create_dataset("locs", data=record, compression="gzip", shuffle=True)
        group = handle.create_group("truth")
        for name, table in truth.items():
            group.create_dataset(name, data=table.to_records(index=False), compression="gzip", shuffle=True)
        group.attrs["note"] = ("designed_sites includes absent sites; sites includes only incorporated emitters. "
            "Angles use native +y up; camera angles account for the y flip. Centered single-point angles are NaN. "
            "Known class labels record input identity; multi-emitter fits have no unique site/event assignment.")
        for key, value in parameters.items():
            handle.attrs[f"{key}_json"] = json.dumps(value, sort_keys=key != "patterns")
        handle.attrs["validation_summary_json"] = json.dumps(summary, sort_keys=True)
        handle.attrs["localizations_sha256"] = hashlib.sha256(record.tobytes()).hexdigest()
    io.save_info(str(path.with_suffix(".yaml")), _plain(info + [parameters]))
    reloaded, reloaded_info = io.load_locs(str(path))
    pd.testing.assert_frame_equal(locs, reloaded.reset_index(drop=True), check_dtype=False)
    if lib.get_from_metadata(reloaded_info, "Frames") != settings.frames:
        raise ValueError("Saved acquisition frame count differs from the simulation")
    for suffix in (".hdf5", ".yaml"):
        path.with_suffix(suffix).replace(output_dir / f"digits_locs{suffix}")


def run_simulation(output_dir, settings, patterns):
    """Simulate notebook-defined integer classes; return a summary and an in-memory movie preview."""
    settings.validate()
    versions = {name: version(name) for name in TESTED_VERSIONS}
    if sys.version_info[:2] != (3, 12) or versions != TESTED_VERSIONS:
        raise RuntimeError("Use Python 3.12 and the pinned requirements.txt to reproduce this dataset")
    coordinates, margin, radius = _prepare_patterns(patterns, settings)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = np.random.get_state()
    try:
        np.random.seed(settings.seed)
        structures, truth = _place_structures(coordinates, settings, margin)
        movie, photons, saturated = _render_movie(structures, settings)
    finally:
        np.random.set_state(state)
    preview = movie[:min(settings.example_frames, settings.frames)].copy()
    with TemporaryDirectory(prefix=".picasso-", dir=output_dir) as temporary:
        temporary = Path(temporary)
        print("Localizing the new raw movie with Picasso Localize.")
        locs, info = _localize_movie(movie, temporary, settings)
        del movie
        truth["localization_labels"] = _label_particles(locs, truth, photons, settings.frames, radius)
        summary = _summarize(locs, truth, settings, saturated, coordinates)
        normalized = {key: ["".join(str(int(bit)) for bit in row) for row in rows] for key, rows in patterns.items()}
        _save_result(temporary, output_dir, locs, info, truth, settings, normalized, versions, summary)
    print(f"Saved {len(locs):,} native Picasso localizations and embedded ground truth.")
    return summary, preview
