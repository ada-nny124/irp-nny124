# Bound Outcome ML

This report covers the first run-level ML pass on [outputs/bound_outcomes.csv](/Users/nny124/irp/outputs/bound_outcomes.csv).

The training script is [scripts/train_bound_models.py](/Users/nny124/irp/scripts/train_bound_models.py). It writes artifacts under [ml/bound_outcomes](/Users/nny124/irp/ml/bound_outcomes).

## Scope

This is initial ML, not fragment-level bound classification.

The workflow models two run-level targets:

1. `has_any_bound_mass`
This is a binary classification task: does a successful FoF run retain any bound mass at all?

2. `bound_mass_fraction`
This is a regression task: how much of the run mass ends up in bound fragments?

The evaluation uses grouped folds by `physical_file`. That means different FoF linking-length variants from the same physical simulation stay in the same fold and do not leak across train and test.

## Datasets

From [dataset_summaries.csv](/Users/nny124/irp/ml/bound_outcomes/tables/dataset_summaries.csv):

| Dataset | Rows | Unique physical files | Meaning |
| --- | ---: | ---: | --- |
| `all_successful_runs` | 407 | 279 | All successful bound-outcome rows |
| `positive_bound_runs` | 225 | 160 | Only rows with `bound_mass_fraction > 0` |

The positive-only subset is only used for regression. Classification is only meaningful on `all_successful_runs`.

## Best Classification Result

From [classification_metrics.csv](/Users/nny124/irp/ml/bound_outcomes/tables/classification_metrics.csv):

| Dataset | Feature set | Best model | Balanced accuracy | F1 | ROC AUC |
| --- | --- | --- | ---: | ---: | ---: |
| `all_successful_runs` | `with_fof_linking_length` | `gradient_boosting_classifier` | 0.9272 | 0.9360 | 0.9754 |
| `all_successful_runs` | `without_fof_linking_length` | `gradient_boosting_classifier` | 0.9139 | 0.9254 | 0.9634 |

Interpretation:

- The binary question is strongly learnable from run metadata.
- `fof_linking_length` helps, but the improvement is modest.
- The physical setup appears to carry most of the signal.

Plots to inspect:

- [all_successful_runs__with_fof_linking_length__has_any_bound_mass__gradient_boosting_classifier__confusion_matrix.png](/Users/nny124/irp/ml/bound_outcomes/plots/gradient_boosting_classifier/with_fof_linking_length/all_successful_runs__with_fof_linking_length__has_any_bound_mass__gradient_boosting_classifier__confusion_matrix.png)
- [all_successful_runs__with_fof_linking_length__has_any_bound_mass__gradient_boosting_classifier__roc_curve.png](/Users/nny124/irp/ml/bound_outcomes/plots/gradient_boosting_classifier/with_fof_linking_length/all_successful_runs__with_fof_linking_length__has_any_bound_mass__gradient_boosting_classifier__roc_curve.png)
- [all_successful_runs__without_fof_linking_length__has_any_bound_mass__gradient_boosting_classifier__roc_curve.png](/Users/nny124/irp/ml/bound_outcomes/plots/gradient_boosting_classifier/without_fof_linking_length/all_successful_runs__without_fof_linking_length__has_any_bound_mass__gradient_boosting_classifier__roc_curve.png)

What these show:

- The confusion matrix shows that the model is not just learning the majority class.
- The ROC curves show that the rank ordering between zero-bound and nonzero-bound runs is very strong even without linking length.

## Best Regression Result

From [regression_metrics.csv](/Users/nny124/irp/ml/bound_outcomes/tables/regression_metrics.csv):

| Dataset | Feature set | Best model | MAE | RMSE | R2 |
| --- | --- | --- | ---: | ---: | ---: |
| `all_successful_runs` | `with_fof_linking_length` | `random_forest_regressor` | 0.0184 | 0.0298 | 0.8971 |
| `all_successful_runs` | `without_fof_linking_length` | `random_forest_regressor` | 0.0208 | 0.0369 | 0.8426 |
| `positive_bound_runs` | `with_fof_linking_length` | `gradient_boosting_regressor` | 0.0193 | 0.0280 | 0.8793 |
| `positive_bound_runs` | `without_fof_linking_length` | `gradient_boosting_regressor` | 0.0195 | 0.0284 | 0.8764 |

Interpretation:

- The retained-mass fraction is harder than the binary gate, but still highly learnable.
- On all successful runs, the model explains about 90% of the variance in `bound_mass_fraction`.
- On positive-only runs, performance stays strong, which means the model is not only learning the zero-versus-nonzero split.
- `fof_linking_length` improves the all-run regression more than it improves the positive-only regression.

Plots to inspect:

- [all_successful_runs__with_fof_linking_length__bound_mass_fraction__random_forest_regressor__actual_vs_predicted.png](/Users/nny124/irp/ml/bound_outcomes/plots/random_forest_regressor/with_fof_linking_length/all_successful_runs__with_fof_linking_length__bound_mass_fraction__random_forest_regressor__actual_vs_predicted.png)
- [all_successful_runs__with_fof_linking_length__bound_mass_fraction__random_forest_regressor__residuals.png](/Users/nny124/irp/ml/bound_outcomes/plots/random_forest_regressor/with_fof_linking_length/all_successful_runs__with_fof_linking_length__bound_mass_fraction__random_forest_regressor__residuals.png)
- [positive_bound_runs__with_fof_linking_length__bound_mass_fraction__gradient_boosting_regressor__actual_vs_predicted.png](/Users/nny124/irp/ml/bound_outcomes/plots/gradient_boosting_regressor/with_fof_linking_length/positive_bound_runs__with_fof_linking_length__bound_mass_fraction__gradient_boosting_regressor__actual_vs_predicted.png)
- [positive_bound_runs__with_fof_linking_length__bound_mass_fraction__gradient_boosting_regressor__residuals.png](/Users/nny124/irp/ml/bound_outcomes/plots/gradient_boosting_regressor/with_fof_linking_length/positive_bound_runs__with_fof_linking_length__bound_mass_fraction__gradient_boosting_regressor__residuals.png)

What these show:

- The all-run scatter plot shows the regressor tracks the full zero-to-positive range well.
- The positive-only scatter plot checks whether the model still works once the trivial zero-mass cases are removed.
- The residual plots are the quickest way to see whether high-retention runs are being systematically underpredicted.

## Practical Read

The current run-level metadata is already enough for a useful first model.

The binary task is very strong, and the regression task is also strong enough to justify deeper model development. This is a better next-step ML problem than fragment-level `is_bound`, because fragment-level `is_bound` is effectively defined by `specific_energy_J_kg`.

## What Next

The next modeling step should be one of these:

1. Add model interpretation tables.
This means permutation importance, grouped residual summaries, and parameter-wise calibration checks.

2. Add stricter leakage checks.
This means testing alternate group definitions and possibly holding out whole parameter families.

3. Add richer run-level targets.
Good candidates are `bound_fragment_count`, `largest_bound_fragment_mass_kg`, and thresholded versions such as `bound_mass_fraction >= 0.1`.

The fragment-level bound classifier should only be revisited with leakage-restricted feature sets. It is not the highest-value next model from this dataset.
