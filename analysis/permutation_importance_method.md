# Permutation Importance: How It Works

Permutation importance measures how much a trained model depends on each input column at evaluation time.

## Procedure used in this repo

1. Fit the promoted random-forest surrogate on the full feature matrix.
2. Evaluate the fitted model on the same design matrix with the normal scoring metric, which here is `R^2`.
3. Pick one feature column.
4. Randomly shuffle only that column across rows.
5. Re-evaluate the model without retraining it.
6. Record how much the score falls relative to the unshuffled baseline.
7. Repeat the shuffle several times and average the score drops.

The reported importance is therefore:

`importance(feature) = mean(baseline_score - shuffled_feature_score)`

## Interpretation

- A large positive importance means the model loses predictive skill when that feature is destroyed, so the model was relying on that column.
- A near-zero importance means the fitted model can recover similar performance from the remaining columns.
- A low importance does not automatically mean the feature is physically unimportant. It can also mean the same information is already encoded in correlated features.

## Why this matters here

The promoted surrogate contains several intentionally correlated physics-style variables, for example:

- `v_inf_kms`
- `v_inf_squared`
- `encounter_eccentricity_proxy`
- `angular_momentum_proxy`
- `periapsis_Rm`
- `periapsis_inverse`

In that setting, permutation importance should be read as "marginal usefulness after the other columns are already present", not as a pure ranking of underlying physics.
