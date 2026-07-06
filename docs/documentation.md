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

- `bound_mass_fraction`: random forest on all successful runs with FoF length, `R² = 0.897`
- `largest_bound_fragment_mass_kg`: random forest on all successful runs with FoF length, `R² = 0.824`
- `average_bound_fragment_mass_kg`: random forest with FoF length, `R² = 0.569`
- `bound_fragment_count`: random forest with FoF length, `R² = 0.496`

Interpretation:

- continuous retained mass is predicted well
- fragment-count style bound targets are materially harder than retained-mass targets

### Best bound-retention classifications

- `bound_mass_fraction_ge_0_1`: gradient boosting with FoF length, balanced accuracy `0.962`, ROC AUC `0.990`
- `has_any_bound_mass`: gradient boosting with FoF length, balanced accuracy `0.927`, ROC AUC `0.975`

Interpretation:

- coarse decision boundaries such as “any bound mass” and “BMF >= 10%” are strongly learnable from the available metadata
- these are useful screening targets even when finer physical detail still requires SPH

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
