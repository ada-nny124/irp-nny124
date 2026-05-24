# Interpretation Analysis

This file consolidates the current interpretation of the extracted FoF outcomes, EDA outputs, baseline ML results, and ML diagnostics for the Martian-moons tidal-disruption project.

Scope limits:

- The extraction and ML stages currently describe **FoF-derived fragment statistics only**.
- They do **not** yet provide validated measurements of bound debris mass, orbital capture, disk mass, or moon formation.
- Any comparison to Kegerreis et al. (2024) should therefore be read as a comparison in **qualitative control hierarchy**, not as a direct reproduction of the paper's capture or disk-mass results.

## Dataset Status

- Outcome rows extracted: `489 / 489`
- Fragment rows extracted: `208,155`
- Simulations with at least one fragment row: `475`
- Extraction errors recorded: `0`
- Rows with mass metrics available: `489`

Outcome summary values:

- `fragment_count_min_particles`
  - median: `255`
  - mean: `425.675`
  - min/max: `0` / `6441`
- `largest_fragment_particle_count`
  - median: `259,351`
  - mean: `873,854.292`
  - min/max: `0` / `3,922,894`
- `largest_fragment_mass_kg`
  - median: `6.311e18 kg`
  - mean: `2.413e19 kg`
  - max: `3.888e20 kg`
- `fof_linking_length`
  - median: `0.004`
  - mean: `0.00360`
- Recommended clean subset:
  - `timestep == 90000`
  - `resolution_code == n65`
  - `fof_linking_length == 0.004`
  - rows: `166`

## Table Inventory

### Raw-data EDA tables

| File | Meaning |
| --- | --- |
| `coverage_mass_vs_periapsis.csv` | Sampling density across mass and periapsis combinations |
| `coverage_mass_vs_resolution.csv` | Sampling density across mass and numerical resolution |
| `coverage_periapsis_vs_velocity.csv` | Sampling density across periapsis and encounter speed |
| `dataset_overview.csv` | Top-level manifest and schema coverage summary |
| `parameter_counts.csv` | Counts of categorical parameter values |
| `parameter_summary_stats.csv` | Numeric summary of parsed manifest parameters |
| `schema_available_fields.csv` | Which HDF5 fields appear in sampled files |
| `schema_dataset_paths.csv` | Sampled HDF5 dataset-path inventory |

From `dataset_overview.csv`, `parameter_counts.csv`, and the coverage matrices, it can be seen that the dataset covers multiple masses, periapses, velocities, resolutions, and spins, but not with perfectly even density. Therefore any later physical or ML interpretation must be read against possible sampling imbalance, especially where a parameter combination appears in only a few cells.

### Outcome EDA tables

| File | Meaning |
| --- | --- |
| `clean_physical_subset_summary.csv` | Controlled subset summary for the recommended `timestep=90000`, `resolution=n65`, `fof_linking_length=0.004` slice |
| `grouped_outcome_means.csv` | Mean FoF outcomes across grouped parameter combinations |
| `outcome_dataset_overview.csv` | Extraction completeness and high-level counts |
| `outcome_summary_stats.csv` | Global summary statistics for FoF targets |

From `outcome_dataset_overview.csv`, it can be seen that the extracted table is complete at `489/489` simulations with `0` extraction errors. From `outcome_summary_stats.csv`, it can be seen that `fragment_count_min_particles` spans `0` to `6441` with median `255`, and `largest_fragment_particle_count` spans `0` to `3,922,894` with median `259,351`. Therefore the outcome space is broad enough to support meaningful modelling, but the targets are also highly skewed and nontrivial.

### Baseline ML tables

| File | Meaning |
| --- | --- |
| `dataset_summaries.csv` | Definitions and row counts for `full` and `clean_subset` |
| `feature_importance.csv` | Coefficient, built-in, and permutation importance values |
| `model_metrics.csv` | Train/test `MAE`, `RMSE`, `R2`, and gap values for all runs |

From `model_metrics.csv`, it can be seen that the best full-dataset models reach `R2 = 0.902` for `largest_fragment_particle_count`, `R2 = 0.885` for `largest_fragment_mass_kg`, and `R2 = 0.737` for `fragment_count_min_particles` when `fof_linking_length` is included. Therefore the baseline ML is already useful for some FoF-derived targets, but not equally strong across all targets.

### ML diagnostics tables

| File | Meaning |
| --- | --- |
| `feature_stability_summary.csv` | Stability of top-ranked features across datasets and feature-set variants |
| `fof_linking_length_comparison.csv` | Best-model performance with vs without `fof_linking_length` |
| `overfit_summary.csv` | Direct train-vs-test gap audit |
| `prediction_bias_summary.csv` | Bias on low-actual and high-actual cases |
| `prediction_records.csv` | Row-level actual, predicted, residual, and feature values |
| `residual_group_stats.csv` | Residual summaries by periapsis, mass, velocity, spin, and linking length |
| `target_difficulty_summary.csv` | Best-target comparison and difficulty ranking |

From `fof_linking_length_comparison.csv`, it can be seen that removing `fof_linking_length` on the full dataset drops `R2` by `0.050` for fragment count, `0.159` for largest fragment particle count, and `0.228` for largest fragment mass. Therefore the current baseline is learning a meaningful amount of FoF post-processing behavior in addition to physical structure. From `prediction_bias_summary.csv`, it can also be seen that extreme high-actual cases are often underpredicted, especially for the largest-fragment targets.

## Plot Inventory

### Raw-data EDA plots

| File | Meaning |
| --- | --- |
| `count_by_fof_linking_length.png` | Frequency of FoF linking lengths |
| `count_by_mass.png` | Frequency of mass bins |
| `count_by_periapsis.png` | Frequency of periapsis bins |
| `count_by_resolution.png` | Frequency of resolution bins |
| `count_by_spin.png` | Frequency of spin configurations |
| `count_by_timestep.png` | Frequency of timesteps |
| `count_by_velocity.png` | Frequency of velocity bins |
| `file_size_distribution.png` | HDF5 file-size spread |
| `heatmap_mass_vs_periapsis_count.png` | Coverage across mass and periapsis |
| `heatmap_mass_vs_resolution_count.png` | Coverage across mass and resolution |
| `heatmap_periapsis_vs_velocity_count.png` | Coverage across periapsis and velocity |

From the raw-data coverage plots, it can be seen where the sampled design space is dense and where it is thin. Therefore later physical and ML conclusions are more trustworthy in the well-sampled regions than in edge-case combinations with few examples.

### Outcome EDA plots

| File | Meaning |
| --- | --- |
| `distribution_fragment_count.png` | Distribution of the fragment-count target |
| `distribution_fragment_mass_fraction.png` | Distribution of `fragment_mass_fraction` |
| `distribution_largest_fragment_mass_kg.png` | Distribution of largest-fragment mass |
| `distribution_largest_fragment_particle_count.png` | Distribution of largest-fragment particle count |
| `fragment_count_vs_fof_linking_length.png` | Fragment-count dependence on FoF linking length |
| `fragment_count_vs_mass.png` | Fragment-count dependence on mass |
| `fragment_count_vs_periapsis.png` | Fragment-count dependence on periapsis |
| `fragment_count_vs_velocity.png` | Fragment-count dependence on velocity |
| `fragment_mass_fraction_vs_periapsis.png` | Mass-fraction behavior across periapsis |
| `heatmap_mean_fragment_count_mass_vs_periapsis.png` | Mean fragment count over mass-periapsis bins |
| `heatmap_mean_fragment_count_periapsis_vs_velocity.png` | Mean fragment count over periapsis-velocity bins |
| `largest_fragment_mass_vs_mass.png` | Largest-fragment mass across mass families |
| `largest_fragment_particles_vs_fof_linking_length.png` | Largest-fragment size vs FoF linking length |
| `largest_fragment_particles_vs_periapsis.png` | Largest-fragment size vs periapsis |

From the outcome plots, it can be seen that `fragment_mass_fraction` is not a useful target because it is effectively constant, while fragment count and largest-fragment metrics show broad and structured variation. Therefore the project was right to focus ML on `fragment_count_min_particles`, `largest_fragment_particle_count`, and `largest_fragment_mass_kg` instead.

### Baseline ML plots

| Plot type / file pattern | Meaning |
| --- | --- |
| `{dataset}__{feature_set}__{target}__{model}__actual_vs_predicted.png` | Fit-quality plot showing how close predictions stay to the diagonal |
| `{dataset}__{feature_set}__{target}__{model}__residuals.png` | Residual-structure plot showing systematic error patterns |

Filename pattern:

- `{dataset}__{feature_set}__{target}__{model}__actual_vs_predicted.png`
- `{dataset}__{feature_set}__{target}__{model}__residuals.png`

Interpretation for all `actual_vs_predicted` plots:

- Tight alignment with the diagonal indicates useful predictive skill.
- Flattening at the extremes indicates regression toward the mean.
- Strong vertical compression indicates underfitting.

Interpretation for all `residuals` plots:

- A centered horizontal cloud around zero is healthy.
- Curvature, fan shapes, or asymmetry indicate systematic bias or missing nonlinear structure.

From the baseline ML plots, it can be seen whether a model is only regressing toward the mean or is actually tracking outcome variation. Therefore these plots are the quickest visual screen for whether a model is worth interpreting further.

### ML diagnostics plots

| Plot type / file pattern | Meaning |
| --- | --- |
| `{dataset}__{feature_set}__{target}__{model}__residuals_by_periapsis_Rm.png` | Failure pattern across encounter distance |
| `{dataset}__{feature_set}__{target}__{model}__residuals_by_mass_log10_kg.png` | Failure pattern across mass family |
| `{dataset}__{feature_set}__{target}__{model}__residuals_by_v_inf_kms.png` | Failure pattern across encounter speed |
| `{dataset}__{feature_set}__{target}__{model}__residuals_by_spin_axis.png` | Failure pattern across spin orientation |
| `{dataset}__{feature_set}__{target}__{model}__residuals_by_fof_linking_length.png` | Failure pattern across FoF grouping choice |

Filename pattern:

- `{dataset}__{feature_set}__{target}__{model}__residuals_by_{feature}.png`

Interpretation for grouped residual plots:

- `residuals_by_periapsis_Rm`
  - Positive residuals at small periapsis imply underprediction of the most disruptive close encounters.
- `residuals_by_mass_log10_kg`
  - Monotonic drift implies the model does not scale cleanly across mass families.
- `residuals_by_v_inf_kms`
  - Structured offsets imply unresolved encounter-speed effects.
- `residuals_by_spin_axis`
  - Large between-axis differences imply the model treats some spin orientations as systematically harder.
- `residuals_by_fof_linking_length`
  - Strong separation by linking length suggests dependence on FoF grouping choices rather than only physical controls.

From the diagnostics plots and grouped residual summaries, it can be seen that the largest errors cluster in specific periapsis regimes and, in some runs, around particular FoF linking lengths. Therefore the best models are capturing a meaningful signal, but they are not uniformly reliable across all parts of parameter space.

## ML Model Inventory

Active model families:

| Model | What it covers | What it means |
| --- | --- | --- |
| `DummyRegressor` | Reference baseline for every dataset, feature set, and target | Predicts the mean only; establishes the minimum useful bar |
| `Ridge` | Linear baseline for every dataset, feature set, and target | Tests whether additive linear structure is enough |
| `RandomForestRegressor` | Nonlinear tree ensemble for every dataset, feature set, and target | Captures interactions and thresholds, but can overfit |
| `GradientBoostingRegressor` | Boosted nonlinear ensemble for every dataset, feature set, and target | Often strongest on structured tabular targets, but still biased at extremes |

Current active filename pattern for serialized models:

| Pattern | Coverage |
| --- | --- |
| `ml/models/{dataset}__{feature_set}__{target}__{model}.pkl` | `2` datasets × `2` feature sets × `3` targets × `4` model families = `48` active model combinations |

## What Each Model Means

- `DummyRegressor`
  - Purpose: minimum baseline that predicts the mean target value.
  - Interpretation: if a real model barely beats this, the learned signal is weak.
- `Ridge`
  - Purpose: regularized linear baseline.
  - Interpretation: tests whether the target is mostly captured by additive linear structure in the engineered features.
- `RandomForestRegressor`
  - Purpose: nonlinear ensemble with flexible interactions.
  - Interpretation: good for capturing threshold-like or mixed nonlinear behavior, but can overfit if train/test gaps become large.
- `GradientBoostingRegressor`
  - Purpose: staged nonlinear ensemble that often gives the strongest tabular performance.
  - Interpretation: useful for capturing structured nonlinear response, but still vulnerable to bias at the most extreme target values.

## Feature Importance and Comparison to Kegerreis et al. (2024)

Kegerreis et al. argue that the dominant physical controls on disruptive partial capture are:

- periapsis distance
- encounter speed
- spin direction and spin magnitude

with periapsis as the strongest overall driver.

### Quantitative feature-importance results from the current baseline

Best models on the `full` dataset with `fof_linking_length` included:

- `fragment_count_min_particles` | `random_forest`
  - `mass_log10_kg = 637.414`
  - `fof_linking_length = 625.357`
  - `periapsis_Rm = 128.685`
  - `spin_axis = 78.506`
  - `particle_log10 = 33.315`
- `largest_fragment_mass_kg` | `gradient_boosting`
  - `periapsis_Rm = 2.010e19`
  - `fof_linking_length = 9.695e18`
  - `mass_log10_kg = 7.320e18`
  - `spin_axis = 4.928e18`
  - `particle_log10 = 1.765e18`
- `largest_fragment_particle_count` | `gradient_boosting`
  - `periapsis_Rm = 766,305.926`
  - `spin_axis = 284,746.792`
  - `fof_linking_length = 167,301.121`
  - `particle_log10 = 124,362.202`
  - `mass_log10_kg = 118,930.259`

Best models on the `clean_subset` with `fof_linking_length` included:

- `fragment_count_min_particles` | `gradient_boosting`
  - `spin_axis = 310.632`
  - `periapsis_Rm = 185.824`
  - `spin_period_hr = 101.855`
  - `v_inf_kms = 21.666`
- `largest_fragment_mass_kg` | `gradient_boosting`
  - `periapsis_Rm = 2.700e19`
  - `spin_axis = 8.325e18`
  - `spin_period_hr = 3.951e18`
  - `v_inf_kms = 1.150e18`
- `largest_fragment_particle_count` | `random_forest`
  - `periapsis_Rm = 750,320.289`
  - `spin_axis = 259,494.881`
  - `spin_period_hr = 110,691.775`
  - `v_inf_kms = 49,980.954`

### Interpretation

There is substantial qualitative agreement with the Kegerreis control hierarchy:

- `periapsis_Rm` is consistently one of the strongest or the strongest features for the largest-fragment targets and remains highly important in the clean subset.
- `spin_axis` and `spin_period_hr` are repeatedly important, especially in the clean subset, which is directionally consistent with Kegerreis’ emphasis on spin alignment and magnitude.
- `v_inf_kms` matters, but in the current FoF-only baseline it is usually weaker than periapsis and spin.

However, the baseline also shows two important deviations:

- `fof_linking_length` becomes highly important on the full dataset, especially for `largest_fragment_mass_kg` and `largest_fragment_particle_count`.
- `mass_log10_kg` is also highly influential in some full-dataset models.

This means the present ML system is learning a mixture of:

- physical controls such as periapsis, velocity, and spin
- post-processing structure from the FoF grouping choice
- scaling effects associated with mass and resolution

So the current feature-importance story is **partly physically plausible**, but not yet clean enough to claim that it isolates only the physical disruption hierarchy described by Kegerreis et al.

## Quantitative Answers to the Research Questions

### 1. How do orbital and physical parameters influence tidal disruption outcomes?

Current answer: **partially answered for FoF-derived fragment outcomes**.

Quantitative evidence:

- Full-dataset best-test `R2` values with `fof_linking_length` included:
  - `largest_fragment_particle_count`: `0.902`
  - `largest_fragment_mass_kg`: `0.885`
  - `fragment_count_min_particles`: `0.737`
- Clean-subset best-test `R2` values with `fof_linking_length` included:
  - `fragment_count_min_particles`: `0.872`
  - `largest_fragment_particle_count`: `0.830`
  - `largest_fragment_mass_kg`: `0.816`

Interpretation:

- The extracted FoF outcomes are clearly structured enough that orbital and physical parameters carry predictive signal.
- Periapsis is repeatedly one of the strongest features.
- Spin axis and spin period also matter strongly in the clean subset.
- Velocity contributes, but is usually secondary to periapsis and spin in the current baseline.

So this question is **answered at the level of FoF-derived fragment statistics**, but not yet at the deeper level of capture or disk formation.

### 2. Which parameters most strongly control fragment formation and bound debris mass?

Current answer: **partially answered for fragment formation, not answered for bound debris mass**.

For fragment formation:

- Best full-dataset model for `fragment_count_min_particles`:
  - `random_forest`
  - `test_MAE = 131.521`
  - `test_RMSE = 277.270`
  - `test_R2 = 0.737`
- Without `fof_linking_length`, best full-dataset fragment-count performance drops to:
  - `test_MAE = 206.454`
  - `test_R2 = 0.687`

Interpretation:

- Fragment formation is strongly influenced by a combination of physical parameters and FoF grouping settings.
- In the clean subset, the strongest physical signals are periapsis and spin-related variables.
- In the full dataset, `fof_linking_length` is influential enough that it materially changes fragment-count performance.

For bound debris mass:

- No validated `bound_debris_mass` target has been extracted.
- No ML target in the current workflow measures bound debris mass directly.

So the second half of this question is **not yet answered**.

### 3. Can a machine learning model reliably predict disruption outcomes across parameter space?

Current answer: **yes for some FoF-derived targets, but only partially and not yet for the full physical science outcomes**.

Best current quantitative results:

- Full | with `fof_linking_length`
  - `largest_fragment_particle_count` | `gradient_boosting`
    - `test_MAE = 208,553.958`
    - `test_RMSE = 375,799.135`
    - `test_R2 = 0.902`
  - `largest_fragment_mass_kg` | `gradient_boosting`
    - `test_MAE = 7.155e18 kg`
    - `test_RMSE = 1.218e19 kg`
    - `test_R2 = 0.885`
  - `fragment_count_min_particles` | `random_forest`
    - `test_MAE = 131.521`
    - `test_RMSE = 277.270`
    - `test_R2 = 0.737`

Key limitations:

- Removing `fof_linking_length` causes a substantial performance drop on the full dataset:
  - fragment count: `R2` drops by `0.050`
  - largest fragment particle count: `R2` drops by `0.159`
  - largest fragment mass: `R2` drops by `0.228`
- High-actual cases are often underpredicted:
  - Full | with `fof_linking_length` | `largest_fragment_mass_kg` | `gradient_boosting`
    - `mean_residual_high_actual = 1.252e19`
    - `underpredict_rate_high_actual = 0.960`
  - Full | with `fof_linking_length` | `largest_fragment_particle_count` | `gradient_boosting`
    - `mean_residual_high_actual = 298,043.098`
    - `underpredict_rate_high_actual = 0.640`

Interpretation:

- The model is already reliably useful for predicting some **FoF-derived fragment statistics**, especially the largest-fragment targets.
- Reliability is weaker for fragment count and for extreme cases.
- The dependence on `fof_linking_length` shows that some of the current predictive power comes from FoF grouping behavior, not purely from physical controls.

So the answer is **yes, but only for the FoF-derived proxy outcomes currently extracted, not yet for bound debris, capture, or proto-disk formation outcomes**.

## Conclusion

This project has successfully built:

- a complete FoF outcome extraction pipeline
- raw-data and outcome-level EDA
- a working baseline ML pipeline
- an ML diagnostics layer for importance, bias, overfitting, and robustness

The strongest current results are:

- the dataset is complete at `489` simulations with `0` extraction errors
- `largest_fragment_particle_count` and `largest_fragment_mass_kg` are the easiest targets to predict
- `periapsis_Rm` is consistently a dominant feature
- spin-related variables are important, especially in the cleaner subset
- `fof_linking_length` materially affects full-dataset ML performance and therefore still acts as a major non-physical influence

What is still missing is exactly the part needed to fully answer the original planetary-science questions:

- validated bound debris mass
- capture metrics
- disk mass or circularised mass in the moon-forming region

So the current state of the project is:

> The computational workflow is now strong and the FoF-derived fragmentation analysis is quantitatively useful, but the project has not yet reached the stage where it can fully answer the physical questions about bound debris or proto-satellite formation in the way Kegerreis et al. discuss.
