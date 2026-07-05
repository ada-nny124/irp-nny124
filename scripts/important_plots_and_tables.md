# Important Plots and Tables

This file lists the plots and tables worth keeping in Git. Priority is based on the 26 June 2026 slide deck.

## Tier 1: core figures used in slides

| Plot | Why it matters | Location |
| --- | --- | --- |
| Dataset coverage heatmap | Shows uneven sampling and justifies the clean subset. Used in the data coverage slide. | `eda/raw_data_overview/plots/heatmap_mass_vs_periapsis_count.png` |
| Fragment count vs periapsis | Main fragmentation trend: close encounters enable stronger disruption. | `eda/outcome_eda/plots/fragment_count_vs_periapsis.png` |
| Largest fragment vs FoF linking length | Separates physical signal from post-processing sensitivity. | `eda/outcome_eda/plots/largest_fragment_particles_vs_fof_linking_length.png` |
| Bound mass fraction heatmap | Main retention overview across periapsis and velocity. | `eda/bound_eda/plots/bound_mass_fraction_heatmap_periapsis_velocity.png` |
| Bound mass fraction vs periapsis, full | Main retained-mass trend used for the bound-retention story. | `eda/bound_eda/plots/kegerreis_bmf_vs_periapsis_full.png` |
| Bound mass fraction vs periapsis, spin effect | Supports the spin interpretation without making it the main claim. | `eda/bound_eda/plots/kegerreis_bmf_vs_periapsis_spin_effect.png` |
| Eccentricity vs bound mass fraction | Main eccentricity result: retention collapses at higher eccentricity proxy. | `eda/eccentricity_eda/plots/eccentricity_vs_bound_mass_fraction.png` |
| Best fragment-count regression plot | Representative ML fit for the hardest fragmentation target that still works reasonably well. | `ml/plots/random_forest/with_fof_linking_length/full__with_fof_linking_length__fragment_count_min_particles__random_forest__actual_vs_predicted.png` |
| Best largest-fragment-particle regression plot | Strongest evidence that dominant-remnant size is predictable. | `ml/plots/gradient_boosting/with_fof_linking_length/full__with_fof_linking_length__largest_fragment_particle_count__gradient_boosting__actual_vs_predicted.png` |
| Best largest-fragment-mass regression plot | Main regression result for physically interpretable fragment mass. | `ml/plots/gradient_boosting/with_fof_linking_length/full__with_fof_linking_length__largest_fragment_mass_kg__gradient_boosting__actual_vs_predicted.png` |
| Combined ROC panel | Best compact appendix view for bound-retention classifiers. | `ml/bound_outcomes/plots/roc_four_targets/roc_combined_four_targets.png` |

## Tier 1: core tables used in slides

| Table | Why it matters | Location |
| --- | --- | --- |
| Dataset overview | Source for run counts and overall coverage statements. | `eda/raw_data_overview/tables/dataset_overview.csv` |
| Coverage by mass and periapsis | Numeric backing for the coverage heatmap. | `eda/raw_data_overview/tables/coverage_mass_vs_periapsis.csv` |
| Parameter summary stats | Quick reference for dominant ranges and skew. | `eda/raw_data_overview/tables/parameter_summary_stats.csv` |
| Outcome dataset overview | Confirms extracted outcome completeness. | `eda/outcome_eda/tables/outcome_dataset_overview.csv` |
| Outcome summary stats | Source for fragment target ranges and skew. | `eda/outcome_eda/tables/outcome_summary_stats.csv` |
| Clean subset summary | Justifies the controlled subset used in interpretation. | `eda/outcome_eda/tables/clean_physical_subset_summary.csv` |
| Bound dataset overview | Source for bound-retention coverage and extraction status. | `eda/bound_eda/tables/dataset_overview.csv` |
| Parameter bound summary | Supports the observed parameter effects table in the deck. | `eda/bound_eda/tables/parameter_bound_summary.csv` |
| Fragment-model metrics | Main slide table for fragmentation ML performance. | `ml/tables/model_metrics.csv` |
| Fragment dataset summaries | Defines `full` vs `clean_subset` for ML tables. | `ml/tables/dataset_summaries.csv` |
| Bound classification metrics | Main slide table for bound-retention classification. | `ml/bound_outcomes/tables/classification_metrics.csv` |
| Bound regression metrics | Main bound-retention regression reference. | `ml/bound_outcomes/tables/regression_metrics.csv` |
| FoF length comparison | Supports the claim that performance remains meaningful without FoF length. | `ml/model_diagnostics/tables/fof_linking_length_comparison.csv` |
| Overfit summary | Short diagnostics table for train-test gap checks. | `ml/model_diagnostics/tables/overfit_summary.csv` |

## Tier 2: keep as core pipeline inputs

| File | Why it matters |
| --- | --- |
| `outputs/manifest.csv` | Parsed simulation metadata table |
| `outputs/fof_outcomes.csv` | Main FoF outcome table |
| `outputs/bound_outcomes.csv` | Main bound-retention outcome table |
| `outputs/hdf5_schema_summary.csv` | Compact extraction/schema audit |

## Everything else

Generated files outside this list should be ignored by default unless they become necessary for a report, paper figure, or reproducibility check.
