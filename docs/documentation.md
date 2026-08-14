# IRP Results Summary

## Aim

This repository analyses SPH simulation outputs for Martian moon-formation scenarios with two linked goals:

- quantify how impact/setup parameters affect fragmentation outcomes
- test how well those outcomes can be predicted from simulation metadata with lightweight ML models

The outcome tables in this repo are useful **screening and comparison products**. They are not direct proof of disk formation or moon formation.

## Delivered data products

The main tracked result tables are:

- `outputs/manifest.csv`: parsed simulation metadata
- `outputs/fof_outcomes.csv`: FoF-derived fragmentation outcomes
- `outputs/bound_outcomes.csv`: bound vs unbound post-processed outcomes
- `outputs/hdf5_schema_summary.csv`: compact schema audit for sampled HDF5 files

## Dataset coverage

Current tracked coverage:

- `489` manifest rows in the extracted simulation set
- `489` FoF outcome rows, with `475` simulations producing fragments
- `407` successful bound-outcome rows
- `208,155` fragment rows in the FoF outcome extraction
- `189,664` fragment rows in the bound/unbound extraction

Bound/unbound split in the successful bound extraction:

- bound fragments: `39,479` (`20.8%`)
- unbound fragments: `150,185` (`79.2%`)

At run level:

- `182` runs have zero bound mass fraction
- `225` runs have mixed bound/unbound mass
- `168` runs are entirely unbound

## Main scientific results

### Fragmentation trends

- Periapsis is a dominant control on fragmentation severity.
- FoF linking length changes some raw fragmentation counts, so it must be treated as a post-processing sensitivity parameter rather than a purely physical variable.
- The size of the dominant remnant is more predictable than the full fragment-count distribution.

### Bound-retention trends

- Bound mass fraction varies strongly across periapsis and velocity.
- The eccentricity proxy is an important organising variable for retained mass: higher eccentricity is associated with weaker retention.
- Spin has a visible effect in some views, but it is not the main first-order control compared with periapsis and encounter conditions.

## ML results

Grouped validation is done by `physical_file` to reduce leakage across related runs.

### Best fragmentation regressions

- `fragment_count_min_particles`: gradient boosting on the `clean_subset`, `R² = 0.872`
- `largest_fragment_mass_kg`: gradient boosting on the `full` dataset with FoF length, `R² = 0.885`
- `largest_fragment_particle_count`: gradient boosting on the `full` dataset with FoF length, `R² = 0.902`

Interpretation:

- fragment count is the hardest major fragmentation target, but still reasonably predictable on the controlled subset
- largest-fragment size, especially particle count, is the strongest fragmentation ML result in the repo

### Best bound-retention regressions

- `bound_mass_fraction`: Random Forest used for the evaluated dashboard prototype, grouped `R² = 0.8971`, `MAE = 0.01839`, `RMSE = 0.02984`
- `bound_mass_fraction` subsequent model-comparison result: two-stage CatBoost hurdle, grouped `R² = 0.9483`, `MAE = 0.01217`, `RMSE = 0.02115`
- `largest_bound_fragment_mass_kg`: random forest on all successful runs with FoF length, `R² = 0.824`
- `average_bound_fragment_mass_kg`: random forest with FoF length, `R² = 0.569`
- `bound_fragment_count`: random forest with FoF length, `R² = 0.496`

Interpretation:

- continuous retained mass is predicted substantially better by the deployed hurdle surrogate than by the previous random-forest benchmark
- the visible `BMF >= 10%` retention screen remains a transparent threshold applied to the continuous deployed BMF prediction
- fragment-count style bound targets are materially harder than retained-mass targets

### Best bound-retention classifications

- `bound_mass_fraction_ge_0_1`: shown in the dashboard as a threshold on the deployed continuous BMF prediction, not as a separate deployed classifier
- `has_any_bound_mass`: also derived from the deployed continuous BMF prediction in the dashboard view
- archived classification experiments remain useful benchmarks, but they are not the deployed public-facing path

Interpretation:

- coarse decision boundaries such as “any bound mass” and “BMF >= 10%” are still useful screening views
- the deployment now keeps those decisions transparent by deriving them from one continuous BMF surrogate rather than switching model families

## Physics-structured surrogate upgrade

The repository now includes a dedicated next-phase surrogate workflow under:

- `scripts/train_physics_structured_surrogate.py`
- `ml/physics_structured_surrogate/`
- `model_training.ipynb`

This upgraded phase keeps the scientific framing unchanged:

- the surrogate is a fast in-domain screening model trained on SPH-derived outcomes
- it is not a replacement for SPH
- `bound_mass_fraction` remains the primary research target
- fragmentation targets remain secondary diagnostics

### Baseline reproduction

The upgraded pipeline first reproduces the grouped-validation baseline with `GroupKFold` grouped by `physical_file`.

Confirmed baseline reference:

- `bound_mass_fraction`, random forest, `with_fof_linking_length`: `R² = 0.8970`, `MAE = 0.01848`, `RMSE = 0.02986`

The baseline stage now writes matched outputs for both:

- `with_fof_linking_length`
- `without_fof_linking_length`

so that later FoF comparisons and tuning summaries use identical grouped folds.

### Compact tuning result

The first tuning pass uses a compact grouped-CV search on the primary target only, `bound_mass_fraction`.

Result:

- tuning did **not** justify promotion over the baseline random forest
- best compact tuned RF with FoF reached `R² = 0.8983`, but the gain over baseline was too small to satisfy the conservative promotion rule
- `promotion_summary.csv` therefore keeps `baseline RF` for the tuning-only decision

This is intentional: the workflow does not promote a tuned model just because mean `R²` increases slightly.

### With-FoF vs without-FoF

The paired FoF comparison shows that FoF linking length still materially improves prediction for the current archive.

Current result:

- best predictive feature set: `with_fof_linking_length`
- `without_fof_linking_length` is more physically clean, but it is materially less accurate on `bound_mass_fraction`

This means the promoted predictive surrogate currently remains post-processing-aware.

### Physics-derived feature ablation

The new workflow adds deterministic, leakage-free physics-style features derived from setup-time quantities already present in the tracked tables.

Implemented physics-style features include:

- `encounter_eccentricity_proxy`
- `v_inf_squared`
- `periapsis_inverse`
- `angular_momentum_proxy`
- `spin_frequency_hr_inv`
- `has_spin`
- `particle_mass_proxy`
- `mass_resolution_interaction`

Current ablation result:

- `physics-feature RF`, `with_fof_linking_length`, `bound_mass_fraction`: `R² = 0.9225`, `MAE = 0.0179`

This result is **not** the deployed surrogate because that feature set included `largest_fragment_mass_fraction`, which is only known after the SPH outcome and therefore leaks post-simulation information.

### Dashboard prototype model and follow-up comparison

The Random Forest was used for the evaluated dashboard prototype.

- deployed dashboard model: `Random Forest`
- target: `bound_mass_fraction`
- feature set: `with_fof_linking_length`
- grouped validation: by `physical_file`
- grouped-CV BMF score: `R² = 0.897127`
- grouped-CV BMF MAE: `0.018394` (`1.8394` percentage points)
- grouped-CV BMF RMSE: `0.029844`

Subsequent model comparison showed that a two-stage CatBoost hurdle model improved grouped held-out `R²` from `0.897` to `0.948` and reduced MAE from `1.84` to `1.22` percentage points.

This means:

- **Scientific/modelling finding:** BMF is better represented as two stages, first whether any material remains bound and then how much is retained, than as one continuous prediction problem.
- **Future-development finding:** the two-stage CatBoost hurdle model is the preferred candidate to replace the RF dashboard model once it is packaged, tested, and connected properly.

The hurdle architecture used in that comparison was:

1. CatBoost classifier for `BMF > 0`
2. CatBoost regressor trained only on positive-BMF rows
3. final prediction `predicted_bmf = P(BMF > 0) × predicted_positive_bmf`
4. final value clipped to `[0, 1]`

Supporting files:

- deployed dashboard prototype:
  `ml/bound_outcomes/models/all_successful_runs__with_fof_linking_length__bound_mass_fraction__random_forest_regressor.pkl`
- subsequent CatBoost comparison:
  `ml/triage/bmf_hurdle_bundle.pkl`
  `ml/triage/bmf_hurdle_metrics.json`
  `ml/triage/bmf_hurdle_oof_predictions.csv`
  `ml/triage/bmf_hurdle_local_diagnostics.csv`
  `ml/triage/bmf_hurdle_controlled_slices.csv`
  `ml/triage/bmf_hurdle_mass_19p5_check.csv`

### Trust rules and caution zones

The upgraded surrogate now writes per-prediction trust flags and screening recommendations.

Trust logic is based on:

- whether the query lies inside the sampled training range
- whether it is near the sampled edge
- whether it falls in a sparse parameter bin
- model spread between tree families
- whether predicted `bound_mass_fraction` is borderline around the `0.10` threshold

Current trust inputs in the deployed dashboard:

- whether the query lies inside the sampled training range
- whether it is near the sampled edge
- nearby independent SPH run count from the local diagnostics table
- local grouped held-out absolute error
- whether predicted `bound_mass_fraction` is borderline around the `0.10` threshold
- disagreement between the deployed Random Forest prediction and the gradient-boosting benchmark

Low-confidence / SPH-required cases remain those that are:

- extrapolative
- near the sampled edge of parameter space
- sparse
- borderline in retained mass
- dependent on detailed fragment, orbital, or debris evolution

### Diagnostics and outputs

The upgraded phase now produces:

- controlled periapsis / velocity / mass / spin slices from the deployed hurdle bundle
- parameter coverage heatmaps
- coverage-vs-error heatmaps
- target-transform comparison tables for the secondary targets
- a dedicated model card and notebook stub for the new surrogate phase

Key outputs are under:

- `ml/physics_structured_surrogate/tables/`
- `ml/physics_structured_surrogate/plots/`

## What the notebook delivers

`model_training.ipynb` is the compact demonstration notebook for the bound-retention and fragmentation modelling workflow. It now summarises:

- the default bound-mass regression task
- supporting bound-retention targets
- three fragmentation targets:
  - `n_fragments`
  - `largest_fragment_mass_kg`
  - `largest_fragment_particle_count`
- parameter-space interpretation tables
- outcome support ranges and extrapolation checks for new setups

## Recommended evidence to show

For presentation or review, the most useful tracked outputs are listed in:

- [important_plots_and_tables.md](/Users/nny124/irp/docs/important_plots_and_tables.md)

That file is the index of plots and tables kept in Git and reused in slides.

## Limits

- FoF outcomes are proxy fragmentation descriptors, not a full physical debris-orbit solution.
- Bound-retention outputs are post-processed and should not be over-interpreted as direct moon-formation proof.
- ML outputs are best used for screening, ranking, and regime mapping, not as a replacement for SPH in new or edge-of-domain cases.
- The physics-structured surrogate is still strongest as an in-domain screening model. SPH remains required for extrapolated, sparse, borderline, or detailed-physics cases.
