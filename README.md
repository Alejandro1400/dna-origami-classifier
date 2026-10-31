# DNA-origami classifier

Inspect DNA-PAINT localizations, pick individual origami, and compare methods of origami classification with geometry-based classification and nanoTRON.

## Data

- **Simulated:** the complete workflow uses known designs and labels for evaluation.
- **Real:** inspect and pick experimental Picasso localizations (you need to use `.hdf5` + matching `.yaml`, including pixel-size metadata). The classification notebooks needs adaptation and validation for experimental data.

## Project structure

```text
notebooks/          # Analysis notebooks
src/                # Simulation and geometry-fitting functions
data/raw/           # Experimental localization files
data/simulated/     # Generated localizations and ground truth
data/derived/       # Particle picks, classifications and trained models
outputs/            # Exported figures
requirements*.txt   # Python dependencies
```

## Notebooks

| Notebook | Purpose |
|---|---|
| [00 · Simulate](notebooks/00_simulate_digits_picasso.ipynb) | Generate patterned DNA-PAINT movies and localize them with Picasso. |
| [01 · Load and render](notebooks/01_load_and_render.ipynb) | Inspect localization quality and images. |
| [02 · Find particles](notebooks/02_find_particles.ipynb) | Compare manual, HDBSCAN and Picasso Addon picking. |
| [03 · Geometry matching](notebooks/03_geometry_matching.ipynb) | Fit position and rotation, then compare known patterns. |
| [04 · nanoTRON](notebooks/04_nanotron_classification.ipynb) | Train on an independent simulation and evaluate the original picks. |

## Run

From the project root, with **Python 3.12** and Git installed:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps -r requirements-autopicking.txt
.venv/bin/jupyter lab notebooks/
```

Select the `.venv` Python kernel and **Run All**. Follow **00 → 01 → 02**, then **03 and/or 04**. For real data, start at **01**. Input paths, parameters and method explanations are in each notebook.

## References

- [Picasso / DNA-PAINT — Schnitzbauer et al. (2017)](https://doi.org/10.1038/nprot.2017.024).
- [dNAM — Dickinson et al. (2021)](https://doi.org/10.1038/s41467-021-22277-y); [LocMoFit — Wu et al. (2023)](https://doi.org/10.1038/s41592-022-01676-z).
- [nanoTRON — Auer et al. (2020)](https://doi.org/10.1093/bioinformatics/btaa154).
- [HDBSCAN documentation](https://scikit-learn.org/1.7/modules/generated/sklearn.cluster.HDBSCAN.html); [Picasso Addon source](https://github.com/schwille-paint/picasso_addon).
