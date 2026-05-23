# Baseline ML Outputs

This directory contains baseline machine-learning outputs built from `outputs/fof_outcomes.csv`.

Scope:

- One row per simulation.
- Predict FoF-derived fragment statistics only.
- No raw HDF5 access during training.
- No use of `fragment_catalog.csv` in this first baseline.

Targets:

- `fragment_count_min_particles`
- `largest_fragment_particle_count`
- `largest_fragment_mass_kg` when present and non-empty

Explicitly excluded:

- `fragment_mass_fraction` when it is constant and therefore non-informative
- any claim about moon formation, bound debris, or orbital capture

Artifacts:

- `tables/model_metrics.csv`
- `tables/feature_importance.csv`
- `tables/dataset_summaries.csv`
- `plots/<model_name>/` for actual-vs-predicted and residual plots grouped by model
- `models/` for serialized baseline models
- `ml_summary.txt` for a plain-text overview
- `model_diagnostics/` for:
  - train-vs-test overfitting checks
  - grouped residual analysis by periapsis, mass, velocity, spin axis, and FoF linking length
  - feature-importance stability checks across datasets and feature sets
  - with-vs-without `fof_linking_length` robustness comparisons
