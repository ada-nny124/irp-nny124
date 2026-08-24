# IRP Results Summary

This repository analyses SPH simulation outputs for Martian moon-formation scenarios with two linked goals:

- quantify how impact/setup parameters affect fragmentation outcomes
- test how well those outcomes can be predicted from simulation metadata with lightweight ML models

The outcome tables in this repo are useful screening (understanding predictive tidal disruption for cetain scenarios) and further deciding tool on whether or not an SPH run is needed.

## Delivered data products

The main tracked result tables are the extraction outputs directly processed from the SPH simulations:

- extraction_outputs/tables/manifest.csv: parsed simulation metadata
- extraction_outputs/tables/fof_outcomes.csv: FoF-derived fragmentation outcomes
- extraction_outputs/tables/bound_outcomes.csv: bound vs unbound post-processed outcomes

## Dataset coverage

Data used for this research:

- 489 manifest rows in the extracted simulation set
- 489 FoF outcome rows, with 475 simulations producing fragments
- 407 successful bound-outcome rows
- 208,155 fragment rows in the FoF outcome extraction
- 189,664 fragment rows in the bound/unbound extraction

Bound/unbound split in the successful bound extraction:

- bound fragments: 39,479 (20.8%)
- unbound fragments: 150,185 (79.2%)

At run level:

- 182 runs have zero bound mass fraction
- 225 runs have mixed bound/unbound mass
- 168 runs are entirely unbound

## Main scientific results (just the summary, complete analysis explained in the report)

### Fragmentation trends

- Periapsis is a dominant control on fragmentation severity.
- FoF linking length changes some raw fragmentation counts, so it must be treated as a post-processing sensitivity parameter rather than a purely physical variable.
- The size of the dominant remnant is more predictable than the full fragment-count distribution.

### Bound-retention trends

- Bound mass fraction varies strongly across periapsis and velocity.
- The eccentricity proxy is an important organising variable for retained mass: higher eccentricity is associated with weaker retention.
- Spin has a visible effect in some views, but it is not the main first-order control compared with periapsis and encounter conditions.

## ML results

Grouped validation is done by physical_file to reduce leakage across related runs.

All active model files used by the current repo are included in the repository under ml/triage/ (csv and json files, pkl models only the ones used in demo, others can be reproduced through the script in model_training_scripts if needed) and ml/trainingartifacts/ (this one includes the pkl files so anyone who clone my repo can immediately run my demo without further run). The main deployed bound-mass model is the tuned Gradient Boosting BMF artifact at ml/trainingartifacts/tuned_gradient_boosting/main_bmf_tuned_gradient_boosting.pkl.

### Best fragmentation regressions

- fragment_count_min_particles: gradient boosting on the clean_subset, R² = 0.872
- largest_fragment_mass_kg: gradient boosting on the full dataset with FoF length, R² = 0.885
- largest_fragment_particle_count: gradient boosting on the full dataset with FoF length, R² = 0.902

Interpretation:

- fragment count is the hardest major fragmentation target, but still reasonably predictable on the controlled subset
- largest-fragment size, especially particle count, is the strongest fragmentation ML result in the repo

### Best bound-retention regressions

- bound_mass_fraction: tuned Gradient Boosting used as the current main deployed BMF model, grouped R² = 0.9217, MAE = 0.0159, RMSE = 0.0260
- largest_bound_fragment_mass_kg: random forest on all successful runs with FoF length, R² = 0.824
- average_bound_fragment_mass_kg: random forest with FoF length, R² = 0.569
- bound_fragment_count: random forest with FoF length, R² = 0.496

Interpretation:

- continuous retained mass in the active demo/report path is predicted with the tuned Gradient Boosting surrogate
- the visible BMF >= 10% retention screen remains a transparent threshold applied to the continuous deployed BMF prediction
- fragment-count style bound targets are materially harder than retained-mass targets

### Best bound-retention classifications

- bound_mass_fraction_ge_0_1: shown in the dashboard as a threshold on the deployed continuous BMF prediction, not as a separate deployed classifier
- has_any_bound_mass: also derived from the deployed continuous BMF prediction in the dashboard view
- archived classification experiments remain useful benchmarks, but they are not the deployed public-facing path

Interpretation:

- coarse decision boundaries such as “any bound mass” and “BMF >= 10%” are still useful screening views
- the deployment now keeps those decisions transparent by deriving them from one continuous BMF surrogate rather than switching model families

### Dashboard prototype model 

The current active dashboard/report BMF path uses tuned Gradient Boosting.

- deployed dashboard BMF model: tuned Gradient Boosting
- artifact: ml/trainingartifacts/tuned_gradient_boosting/main_bmf_tuned_gradient_boosting.pkl
- target: bound_mass_fraction
- grouped validation: by physical_file
- grouped-CV BMF score: R² = 0.9217
- grouped-CV BMF MAE: 0.0159 (1.59 percentage points)
- grouped-CV BMF RMSE: 0.0260

The active triage/demo path keeps this tuned BMF artifact as the main continuous bound-mass model.

### Trust rules and caution zones

Trust logic is based on:

- whether the query lies inside the sampled training range
- whether it is near the sampled edge
- whether it falls in a sparse parameter bin
- model spread between tree families
- whether predicted bound_mass_fraction is borderline around the 0.10 threshold

Current trust inputs in the deployed dashboard:

- whether the query lies inside the sampled training range
- whether it is near the sampled edge
- nearby independent SPH run count from the local diagnostics table
- local grouped held-out absolute error
- whether predicted bound_mass_fraction is borderline around the 0.10 threshold
- disagreement between the deployed tuned-Gradient-Boosting prediction and a secondary Random Forest benchmark

Low-confidence / SPH-required cases remain those that are:

- extrapolative
- near the sampled edge of parameter space
- sparse
- borderline in retained mass
- dependent on detailed fragment, orbital, or debris evolution