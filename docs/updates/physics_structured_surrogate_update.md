# Physics-Structured Surrogate Update

Generated: July 17, 2026

## Summary

This report summarises the new physics-structured surrogate phase added under `ml/physics_structured_surrogate/`.

The scientific framing is unchanged:

- the surrogate is a fast in-domain screening model trained on SPH-derived outcomes
- it is not a replacement for SPH
- `bound_mass_fraction` is the primary target
- fragmentation targets remain secondary diagnostics

## What Changed

The new workflow is driven by:

- `scripts/train_physics_structured_surrogate.py`
- `model_training.ipynb`
- `ml/physics_structured_surrogate/`

It adds:

- grouped baseline reproduction using `GroupKFold` by `physical_file`
- compact hyperparameter tuning for the primary target
- with-FoF versus without-FoF comparison
- deterministic physics-derived features
- secondary-target transform comparisons
- per-prediction trust flags
- representative slice diagnostics
- coverage and error diagnostics
- a model-card style summary

## Baseline Reproduction

The grouped-CV baseline was reproduced successfully.

Reference result:

- `bound_mass_fraction`, random forest, `with_fof_linking_length`
  - `R² = 0.8971`
  - `MAE = 0.0184`
  - `RMSE = 0.0298`

This matches the expected baseline reference and confirms that the new pipeline preserved the original grouped-validation behavior.

## Compact Tuning Result

The first tuning pass used a compact search on the primary target only, `bound_mass_fraction`.

Best compact tuned candidate:

- random forest
- `n_estimators = 500`
- `max_depth = None`
- `min_samples_leaf = 2`
- `max_features = 0.8`

Best compact tuned result:

- `R² = 0.8983`
- `MAE = 0.0186`

Promotion decision:

- tuning **did not** justify promotion
- gains over baseline were too small
- the conservative rule therefore kept `baseline RF` for the tuning-only decision

## With-FoF vs Without-FoF

The paired FoF comparison shows that `with_fof_linking_length` remains materially stronger than the more physically clean `without_fof_linking_length` setting.

Decision:

- best predictive model: `with_fof_linking_length`
- best more-physical fallback: `without_fof_linking_length` was not close enough to replace the predictive choice

Current interpretation:

- FoF linking length still behaves as useful predictive information in the present archive
- this makes the current promoted surrogate post-processing-aware rather than purely encounter-parameter-only

## Physics-Feature Ablation

The workflow adds deterministic physics-style features derived from setup-time quantities already present in the tracked tables.

Implemented features include:

- `encounter_eccentricity_proxy`
- `v_inf_squared`
- `periapsis_inverse`
- `angular_momentum_proxy`
- `spin_frequency_hr_inv`
- `has_spin`
- `particle_mass_proxy`
- `mass_resolution_interaction`

Best ablation result:

- `physics-feature RF`
- feature set: `with_fof_linking_length`
- `R² = 0.9225`
- `MAE = 0.0179`

This materially outperformed the reproduced baseline and became the current promoted model.

## Promoted Model

Current promoted surrogate:

- model: `physics-feature RF`
- target: `bound_mass_fraction`
- feature set: `with_fof_linking_length`
- grouped-CV BMF `R² = 0.9225`
- grouped-CV BMF `MAE = 0.0179`

Promotion reason:

- physics-feature ablation materially improved BMF

## Trust Package

The new trust layer writes per-prediction flags for:

- in-range vs extrapolative use
- near-edge conditions
- sparse-bin status
- model spread
- fold spread
- borderline BMF region
- final confidence recommendation

Current trust summary:

- high-confidence screening rows: `40`
- medium-confidence screening rows: `254`
- low-confidence / SPH-required rows: `113`
- spread threshold: `0.009678`

Operational interpretation:

- use the surrogate most confidently inside dense, in-range regions and away from the `BMF = 0.10` threshold
- defer to SPH for sparse, edge-of-domain, extrapolative, or borderline cases

## Secondary-Target Transform Checks

The current transform comparison does not support replacing the raw targets for the main secondary diagnostics.

Notable outcomes:

- `n_fragments`: raw target outperformed `log1p`
- `largest_fragment_mass_kg`: raw target outperformed `log1p`
- `largest_fragment_particle_count`: raw target outperformed `log1p`
- `largest_fragment_mass_fraction` remained well-behaved and predictive in both forms

This keeps the current interpretation simple:

- BMF remains the strongest primary target
- fragmentation-style targets are still materially noisier

## Diagnostics

### BMF slice: baseline vs promoted

This figure shows the representative controlled slice:

- mass = `A2000`
- resolution = `n65`
- velocity = `v00`
- spin = `s030z`
- timestep = `90000`
- FoF linking length = `0.004`
- varying periapsis

![BMF slice](../ml/physics_structured_surrogate/plots/bmf_slice_baseline_vs_promoted.png)

### Fragment count slice

![Fragment count slice](../ml/physics_structured_surrogate/plots/fragment_count_slice_baseline_vs_promoted.png)

### Largest fragment mass slice

![Largest fragment mass slice](../ml/physics_structured_surrogate/plots/largest_fragment_mass_slice_baseline_vs_promoted.png)

### Largest fragment particle count slice

![Largest fragment particle count slice](../ml/physics_structured_surrogate/plots/largest_fragment_particle_count_slice_baseline_vs_promoted.png)

### Parameter coverage

Coverage remains uneven and concentrated in a limited subset of parameter space.

![Coverage heatmaps](../ml/physics_structured_surrogate/plots/parameter_coverage_heatmaps.png)

Current coverage summary:

- occupied `mass_log10_kg x periapsis_Rm` bins: `38 / 105`
- occupied `periapsis_Rm x v_inf_kms` bins: `45 / 165`

### Coverage versus error

![Coverage vs error heatmaps](../ml/physics_structured_surrogate/plots/coverage_vs_error_heatmaps.png)

Current quantitative summary:

- mean error in dense bins: `0.01836`
- mean error in sparse bins: `0.01446`
- worst mass-periapsis error bin: `(19.5, 1.6)`
- worst periapsis-velocity error bin: `(2.0, 1.0)`

## Main Takeaways

1. The new pipeline reproduces the original grouped baseline correctly.
2. Compact tuning alone was not enough to justify replacing the baseline RF.
3. Physics-derived features produced the main improvement and promoted the current `physics-feature RF`.
4. The current best predictive surrogate still includes FoF information.
5. Trust is conditional and must still be tied to domain coverage and decision proximity.
6. SPH remains necessary for extrapolated, sparse, borderline, or detailed-physics cases.

## Key Files

- Driver: `scripts/train_physics_structured_surrogate.py`
- Notebook: `model_training.ipynb`
- Model card: `ml/physics_structured_surrogate/model_card.md`
- Baseline metrics: `ml/physics_structured_surrogate/tables/baseline_metrics.csv`
- Tuning search: `ml/physics_structured_surrogate/tables/tuning_search_results.csv`
- Promotion summary: `ml/physics_structured_surrogate/tables/promotion_summary.csv`
- FoF comparison: `ml/physics_structured_surrogate/tables/with_vs_without_fof_promotion.csv`
- Trust flags: `ml/physics_structured_surrogate/tables/predictions_with_trust_flags.csv`
- Coverage summary: `ml/physics_structured_surrogate/tables/coverage_error_summary.csv`
