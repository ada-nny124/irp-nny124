# Bound vs Unbound EDA

This EDA uses:

- `outputs/fragment_orbital_catalog.csv` as the main fragment-level table.
- `outputs/bound_outcomes.csv` for run-level aggregation.
- `outputs/bound_unbound_extraction_log.csv` for QA and coverage only.

The reproducible script is [scripts/eda_bound_eda.py](/Users/nny124/irp/scripts/eda_bound_eda.py). It writes artifacts under [eda/bound_eda](/Users/nny124/irp/eda/bound_eda).

## Executive Read

The main result is that `is_bound` is not just correlated with `specific_energy_J_kg`; it is exactly determined by its sign in the extracted catalog. In [energy_sign_crosstab.csv](/Users/nny124/irp/eda/bound_eda/tables/energy_sign_crosstab.csv), all 39,479 bound fragments have negative specific energy and all 150,185 unbound fragments have positive specific energy.

That means:

1. Training on `specific_energy_J_kg` makes the classification task effectively trivial and physically definitional.
2. `com_speed_m_s` and `com_r_m` are also highly informative because they are upstream ingredients of orbital energy, so they are likely to behave like near-leakage features.
3. The more interesting EDA is run-level: when do simulations produce a larger bound mass fraction at all?

## Coverage

From [dataset_overview.csv](/Users/nny124/irp/eda/bound_eda/tables/dataset_overview.csv):

| Metric | Value |
| --- | ---: |
| Fragment rows | 189,664 |
| Run rows in `bound_outcomes.csv` | 407 |
| QA log rows | 489 |
| Successful QA rows | 407 |
| Bound fragments | 39,479 |
| Unbound fragments | 150,185 |
| Bound fragment share | 20.82% |
| Runs with zero bound mass fraction | 182 |
| Runs with mixed bound mass fraction | 225 |
| Runs with `unbound_mass_fraction = 1` | 168 |

QA interpretation from [extraction_status_summary.csv](/Users/nny124/irp/eda/bound_eda/tables/extraction_status_summary.csv):

- 393 rows are normal successes with mass and GM sourced from metadata/header.
- 14 rows are `success_no_fragments`.
- 81 rows are `missing_physical_file`.
- 1 row is `error`.

## Plots To Look At

If only four plots are worth inspecting, these are the ones:

1. [specific_energy_signed_log10_by_class.png](/Users/nny124/irp/eda/bound_eda/plots/specific_energy_signed_log10_by_class.png)
Result: the class split happens exactly at zero signed energy. This is the most important plot in the whole EDA.

2. [radius_vs_speed_by_class.png](/Users/nny124/irp/eda/bound_eda/plots/radius_vs_speed_by_class.png)
Result: bound fragments cluster at lower COM speeds and somewhat larger radii; unbound fragments sit at substantially higher speeds. This is the physical geometry behind the energy split.

3. [bound_mass_fraction_heatmap_periapsis_velocity.png](/Users/nny124/irp/eda/bound_eda/plots/bound_mass_fraction_heatmap_periapsis_velocity.png)
Result: higher bound mass fraction concentrates at lower `velocity_code` and tighter `periapsis_code`. Faster encounters trend toward mostly or entirely unbound outcomes.

4. [bound_mass_fraction_vs_fof_linking_length.png](/Users/nny124/irp/eda/bound_eda/plots/bound_mass_fraction_vs_fof_linking_length.png)
Result: linking length does move the aggregate bound fraction, but this is secondary to the physical parameters and some linking-length values have very small sample counts.

Supporting distribution plots:

- [fragment_class_balance.png](/Users/nny124/irp/eda/bound_eda/plots/fragment_class_balance.png)
- [fragment_mass_by_class.png](/Users/nny124/irp/eda/bound_eda/plots/fragment_mass_by_class.png)
- [fragment_particle_count_by_class.png](/Users/nny124/irp/eda/bound_eda/plots/fragment_particle_count_by_class.png)

## Why The Label Looks This Way

From [fragment_class_summary.csv](/Users/nny124/irp/eda/bound_eda/tables/fragment_class_summary.csv):

| Metric | Unbound median | Bound median | Interpretation |
| --- | ---: | ---: | --- |
| `fragment_particle_count` | 74 | 80 | Class separation is not driven by count alone. |
| `fragment_mass_kg` | 1.93e15 | 2.82e15 | Bound fragments are somewhat heavier in the median, but mass is not the core separator. |
| `com_r_m` | 2.215e8 | 2.449e8 | Bound fragments tend to sit farther out in this extracted state. |
| `com_speed_m_s` | 920.24 | 454.78 | Speed is a major separator. |
| `specific_energy_J_kg` | 2.31e5 | -7.13e4 | The sign flips exactly across classes. |

The threshold behavior is easiest to see in [threshold_edge_fragments.csv](/Users/nny124/irp/eda/bound_eda/tables/threshold_edge_fragments.csv):

| `fof_file` | `group_id` | `com_speed_m_s` | `specific_energy_J_kg` | `is_bound` |
| --- | ---: | ---: | ---: | --- |
| `Ma_xp_A2100_n70_r16_v00_90000_fof_0.0074_0000.hdf5` | 420 | 559.47 | 5.36 | `False` |
| `Ma_xp_A2100_n70_r16_v00_90000_fof_0.0044_0000.hdf5` | 438 | 559.47 | 5.36 | `False` |
| `Ma_xp_A2000_s047x_n65_r12_v00_90000_fof_0.0040_0000.hdf5` | 1545 | 601.64 | 10.44 | `False` |
| `Ma_xp_A2000_s030z_n65_r22_v00_90000_fof_0.0040_0000.hdf5` | 129 | 601.64 | -17.59 | `True` |
| `Ma_xp_A2000_n65_r16_v00_90000_fof_0.0020_0000.hdf5` | 561 | 603.58 | 19.40 | `False` |
| `Ma_xp_A2000_s036y_n65_r16_v06_90000_fof_0.0040_0000.hdf5` | 945 | 607.72 | -26.22 | `True` |

Interpretation: the nearest examples to the decision boundary have energies extremely close to zero, and tiny changes in the speed/radius balance are enough to flip the class. That is why the energy plot is the single best explanation plot.

## Run-Level Behavior

The strongest run-level trends from [parameter_bound_summary.csv](/Users/nny124/irp/eda/bound_eda/tables/parameter_bound_summary.csv):

| Parameter | Value | Runs | Mean bound mass fraction | Zero-bound run share |
| --- | --- | ---: | ---: | ---: |
| `mass_code` | `A2050` | 6 | 0.2445 | 0.0000 |
| `mass_code` | `A2100` | 6 | 0.2357 | 0.0000 |
| `mass_code` | `A1800` | 10 | 0.0000 | 1.0000 |
| `mass_code` | `A1850` | 8 | 0.0000 | 1.0000 |
| `resolution_code` | `n70` | 13 | 0.2225 | 0.0000 |
| `resolution_code` | `n60` | 23 | 0.0237 | 0.8261 |
| `periapsis_code` | `r11` | 12 | 0.1863 | 0.0833 |
| `periapsis_code` | `r13` | 13 | 0.1731 | 0.1538 |
| `periapsis_code` | `r24` | 12 | 0.0119 | 0.8333 |
| `velocity_code` | `v02` | 20 | 0.1080 | 0.3000 |
| `velocity_code` | `v10` | 13 | 0.0040 | 0.6923 |
| `velocity_code` | `v12` | 3 | 0.0000 | 1.0000 |

Main interpretation:

- Higher-mass runs are much more likely to retain bound material.
- Tighter periapsis tends to increase retained bound mass.
- Higher velocity strongly suppresses bound retention.
- Resolution matters, but the `n70` advantage is partly confounded by which physical cases exist at that resolution.
- Linking length changes the measured outcome, but the physical parameters dominate the story.

## Important Sample Run Tables

High bound-retention examples from [top_bound_mass_fraction_runs.csv](/Users/nny124/irp/eda/bound_eda/tables/top_bound_mass_fraction_runs.csv):

| `fof_file` | `mass_code` | `periapsis_code` | `velocity_code` | `fof_linking_length` | `bound_mass_fraction` |
| --- | --- | --- | --- | ---: | ---: |
| `Ma_xp_A2000_s030z_n65_r12_v00_90000_fof_0.0040_0000.hdf5` | `A2000` | `r12` | `v00` | 0.0040 | 0.275659 |
| `Ma_xp_A2000_s030z_n65_r11_v00_90000_fof_0.0040_0000.hdf5` | `A2000` | `r11` | `v00` | 0.0040 | 0.273952 |
| `Ma_xp_A2050_n70_r12_v00_90000_fof_0.0050_0000.hdf5` | `A2050` | `r12` | `v00` | 0.0050 | 0.271159 |
| `Ma_xp_A2050_n70_r12_v00_90000_fof_0.0040_0000.hdf5` | `A2050` | `r12` | `v00` | 0.0040 | 0.271140 |
| `Ma_xp_A2100_n70_r12_v00_90000_fof_0.0044_0000.hdf5` | `A2100` | `r12` | `v00` | 0.0044 | 0.258791 |

Why these matter: even the best-retention runs keep only about 26% to 28% of mass in bound fragments, and the largest unbound fragment is still larger than the largest bound fragment in these cases. So this dataset is not split into "mostly bound" versus "mostly unbound" runs; it is mostly "entirely unbound" versus "partially bound but still unbound-dominated."

Zero-bound examples from [zero_bound_mass_fraction_runs.csv](/Users/nny124/irp/eda/bound_eda/tables/zero_bound_mass_fraction_runs.csv):

| `fof_file` | `mass_code` | `periapsis_code` | `velocity_code` | `fof_linking_length` | `n_fragments` |
| --- | --- | --- | --- | ---: | ---: |
| `Ma_xp_A1800_n60_r12_v00_90000_fof_0.0013_0000.hdf5` | `A1800` | `r12` | `v00` | 0.0013 | 265 |
| `Ma_xp_A1800_n60_r12_v00_90000_fof_0.0014_0000.hdf5` | `A1800` | `r12` | `v00` | 0.0014 | 265 |
| `Ma_xp_A1800_n60_r16_v00_90000_fof_0.0013_0000.hdf5` | `A1800` | `r16` | `v00` | 0.0013 | 88 |
| `Ma_xp_A1800_n60_r16_v00_90000_fof_0.0014_0000.hdf5` | `A1800` | `r16` | `v00` | 0.0014 | 88 |
| `Ma_xp_A1800_n60_r12_v00_90000_fof_0.0020_0000.hdf5` | `A1800` | `r12` | `v00` | 0.0020 | 265 |

Why these matter: low-mass `A1800` and `A1850` cases are fully zero-bound across the available successful runs, which is why `mass_code` is such a strong run-level grouping variable.

## ML Interpretation

For fragment-level classification:

- `specific_energy_J_kg` should be treated as a label-defining feature, not a normal predictor.
- `com_speed_m_s` and `com_r_m` should also be treated carefully because they encode the same orbital-energy story.
- If the goal is a non-trivial classifier, build at least one feature set that excludes `specific_energy_J_kg` and likely excludes direct orbital-state leakage variables.

For run-level modeling:

- `bound_mass_fraction` is a more interesting target than fragment-level `is_bound`.
- The run-level signal appears to be driven mainly by `mass_code`, `periapsis_code`, and `velocity_code`, with `fof_linking_length` as a secondary analysis/control variable.

## Artifacts

- Script: [scripts/eda_bound_eda.py](/Users/nny124/irp/scripts/eda_bound_eda.py)
- EDA folder: [eda/bound_eda](/Users/nny124/irp/eda/bound_eda)
- Overview table: [dataset_overview.csv](/Users/nny124/irp/eda/bound_eda/tables/dataset_overview.csv)
- Parameter summary: [parameter_bound_summary.csv](/Users/nny124/irp/eda/bound_eda/tables/parameter_bound_summary.csv)
- Threshold examples: [threshold_edge_fragments.csv](/Users/nny124/irp/eda/bound_eda/tables/threshold_edge_fragments.csv)
