# Important Plots and Tables

This file lists the plots and tables kept in Git and used in slides.

## Tier 1: core figures used in slides

| Plot | Why it matters | Location |
| --- | --- | --- |
| Dataset coverage heatmap | Shows uneven sampling and justifies the clean subset. Used in the data coverage slide. | `eda/plots/heatmap_mass_vs_periapsis_count.png` |
| Fragment count vs periapsis | Main fragmentation trend: close encounters enable stronger disruption. | `eda/plots/fragment_count_vs_periapsis.png` |
| Largest fragment vs FoF linking length | Raw scatter only. It mixes many physical scenarios, so use it as a confounding check rather than a clean FoF-sensitivity result. See the matched-scenario diagnostics in `eda/tables/fof_linking_length_scenario_summary.csv`. | `eda/plots/largest_fragment_particles_vs_fof_linking_length.png` |
| Largest fragment vs FoF linking length, single matched scenario | Controlled FoF-sensitivity check within one fixed physical scenario. | `eda/plots/largest_fragment_particles_vs_fof_single_scenario.png` |
| Largest fragment vs FoF linking length, matched scenarios | Multi-line comparison across several fixed physical scenarios. | `eda/plots/largest_fragment_particles_vs_fof_multiscenario.png` |
| Largest fragment fraction vs FoF linking length, matched scenarios | Normalized version that reduces total-particle-count scale effects across scenarios. | `eda/plots/largest_fragment_fraction_vs_fof_multiscenario.png` |
| Bound mass fraction heatmap | Main retention overview across periapsis and velocity. | `eda/plots/bound_mass_fraction_heatmap_periapsis_velocity.png` |
| Bound mass fraction vs periapsis, full | Main retained-mass trend used for the bound-retention story. | `eda/plots/bmf_vs_periapsis_full.png` |
| Bound mass fraction vs periapsis, spin effect | Supports the spin interpretation without making it the main claim. | `eda/plots/bmf_vs_periapsis_spin_effect.png` |
| Matched-family spin spread and low/high periapsis panels | Cleanest spin-regime figure: each point is one matched family and the bottom panels show raw BMF separation for a common spin trio. | `report/figures/spin_argument_matched_families.svg` |
| Eccentricity vs bound mass fraction | Main eccentricity result: retention collapses at higher eccentricity proxy. | `eda/plots/eccentricity_vs_bound_mass_fraction.png` |
| Best fragment-count regression plot | Representative ML fit for the hardest fragmentation target that still works reasonably well. | `ml/plots/report_reference/fragmentation/full__with_fof_linking_length__fragment_count_min_particles__random_forest__actual_vs_predicted.png` |
| Best largest-fragment-particle regression plot | Strongest evidence that dominant-remnant size is predictable. | `ml/plots/report_reference/fragmentation/full__with_fof_linking_length__largest_fragment_particle_count__gradient_boosting__actual_vs_predicted.png` |
| Best largest-fragment-mass regression plot | Main regression result for physically interpretable fragment mass. | `ml/plots/report_reference/fragmentation/full__with_fof_linking_length__largest_fragment_mass_kg__gradient_boosting__actual_vs_predicted.png` |
| Combined ROC panel | Best compact appendix view for bound-retention classifiers. | `ml/plots/report_reference/bound/roc_combined_four_targets.png` |

## Tier 1: core tables used in slides

| Table | Why it matters | Location |
| --- | --- | --- |
| Dataset overview | Source for run counts and overall coverage statements. | `eda/tables/dataset_overview.csv` |
| Coverage by mass and periapsis | Numeric backing for the coverage heatmap. | `eda/tables/coverage_mass_vs_periapsis.csv` |
| Parameter summary stats | Quick reference for dominant ranges and skew. | `eda/tables/parameter_summary_stats.csv` |
| Outcome dataset overview | Confirms extracted outcome completeness. | `eda/tables/outcome_dataset_overview.csv` |
| Outcome summary stats | Source for fragment target ranges and skew. | `eda/tables/outcome_summary_stats.csv` |
| FoF scenario summary | Shows how many physical scenarios are mixed in the raw FoF scatter and which ones have repeated FoF sweeps. | `eda/tables/fof_linking_length_scenario_summary.csv` |
| Matched FoF scenarios | Restricted table of scenarios with more than one FoF linking length. | `eda/tables/fof_linking_length_matched_scenarios.csv` |
| Clean subset summary | Justifies the controlled subset used in interpretation. | `eda/tables/clean_physical_subset_summary.csv` |
| Bound dataset overview | Source for bound-retention coverage and extraction status. | `eda/tables/dataset_overview.csv` |
| Parameter bound summary | Supports the observed parameter effects table in the deck. | `eda/tables/parameter_bound_summary.csv` |
| Fragment-model metrics | Main slide table for fragmentation ML performance. | `ml/tables/model_metrics.csv` |
| Fragment dataset summaries | Defines `full` vs `clean_subset` for ML tables. | `ml/tables/dataset_summaries.csv` |
| BMF model comparison | Main report comparison of baseline, tuned, physics-feature, and exploratory models. | `report-table-figure/tables/tableA2_used_in_report.csv` |
| BMF failure case | Sparse-support held-out example used in the report. | `report-table-figure/tables/table2_used_in_report.csv` |

## Tier 2: keep as core pipeline inputs

| File | Why it matters |
| --- | --- |
| `extraction_outputs/tables/manifest.csv` | Parsed simulation metadata table |
| `extraction_outputs/tables/fof_outcomes.csv` | Main FoF outcome table |
| `extraction_outputs/tables/bound_outcomes.csv` | Main bound-retention outcome table |
| `extraction_outputs/tables/hdf5_schema_summary.csv` | Compact extraction/schema audit |

## Everything else

Generated files outside this list should be ignored by default unless they become necessary for a report, paper figure, or reproducibility check.
