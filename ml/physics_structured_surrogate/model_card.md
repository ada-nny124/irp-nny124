# Physics-Structured Surrogate Model Card

The surrogate is not a replacement for SPH. It is a fast in-domain screening model trained on SPH-derived outcomes.

- Promoted model name: `physics-feature RF`
- Primary target: `bound_mass_fraction`
- Feature set: `with_fof_linking_length`
- Physics-derived features included: `True`
- Promotion reason: physics-feature ablation materially improved BMF
- Grouped-CV BMF R^2: 0.9225
- Grouped-CV BMF MAE: 0.0179
- Trust spread threshold: 0.0097
- High-confidence predictions: 40
- Medium-confidence predictions: 254
- Low-confidence / SPH required: 113
- Coverage summary file: `ml/physics_structured_surrogate/tables/coverage_error_summary.csv`

## Caution zones
- outside the training range
- near the sampled edge of parameter space
- sparse coverage bins
- borderline BMF around 0.10
- cases needing detailed debris, orbit, or eccentricity evolution

## Future work
- expand the SPH archive in sparse regions
- validate promoted predictions against newly run SPH cases
- test stronger physics-aware proxies before considering neural methods
