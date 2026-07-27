# Interpreting the Permutation-Importance Ranking

This note explains three questions raised by the promoted bound-mass-fraction surrogate:

1. Why does velocity look relatively unimportant?
2. Why can `v_inf_kms^2` rank above `v_inf_kms`, and why can `periapsis_inverse` rank near `periapsis_Rm`?
3. Does that pattern suggest overfitting?

## Short answer

Not by itself. In the current promoted feature set, several columns intentionally encode overlapping physics. Permutation importance then measures the *extra marginal information* in one column after the other correlated columns are already present.

That means:

- a low-ranked velocity term does not mean velocity is physically unimportant
- it usually means other columns already carry much of the same signal
- paired transforms such as `periapsis_Rm` and `1 / periapsis_Rm` can both appear important because each helps the trees split the same underlying trend in a different numerical form

## Why velocity is not near the top

The promoted model does not only see raw velocity. It also sees:

- `v_inf_squared`
- `encounter_eccentricity_proxy`
- `angular_momentum_proxy`

All three reuse `v_inf_kms`, and two of them also combine it with periapsis.

The dataset itself is also uneven in velocity coverage. Out of 407 rows:

- `v_inf = 0.0 km/s`: 292 rows
- `v_inf = 0.2 km/s`: 20 rows
- `v_inf = 0.4 km/s`: 21 rows
- `v_inf >= 0.6 km/s`: only 74 rows total

So the model sees a strong low-velocity concentration and several derived columns that already summarize the speed regime. Under that setup, shuffling `v_inf_kms` alone does not destroy as much information as you might expect from the physics alone.

## Why `v_inf_squared` can beat `v_inf_kms`

That is physically plausible rather than suspicious.

- Kinetic-energy-like scaling depends on velocity squared.
- The eccentricity proxy already depends on `r_p * v_inf^2 / mu`.
- Tree models split on thresholds, so a squared term can sometimes separate the low-speed and fast-flyby regimes more cleanly than the raw linear term.

In the current importance table, `v_inf_squared` is only slightly above `v_inf_kms`, which is consistent with both columns encoding nearly the same information.

## Why `periapsis_Rm` and `periapsis_inverse` both matter

This is also expected.

- `periapsis_Rm` is the direct orbital-distance variable.
- `periapsis_inverse` turns “closer to Mars” into a larger number, which often lines up better with tidal-strength intuition.

The two columns are almost perfect monotonic transforms of one another. Their sample correlation in this dataset is `-0.9695`.

Because the model is nonlinear, the inverse transform can still help it place more useful thresholds near the low-periapsis regime where disruption changes rapidly.

## Quick overfitting check

I checked the same grouped-cross-validation workflow while removing correlated columns one at a time from the input-only promoted random forest.

Key grouped-CV `R^2` results:

- All input-only promoted features: `0.8671`
- Drop `v_inf_kms`: `0.8695`
- Drop `v_inf_squared`: `0.8701`
- Drop both velocity terms: `0.8684`
- Drop `periapsis_Rm`: `0.8586`
- Drop `periapsis_inverse`: `0.8579`
- Drop both periapsis terms: `0.3862`

Interpretation:

- Removing one velocity representation hardly changes performance because the remaining correlated velocity-derived columns still carry the signal.
- Removing one periapsis representation hurts a little, and removing both hurts a lot.
- That is the signature of feature redundancy plus strong periapsis dependence, not a clean sign of overfitting.

## Conclusion

The current ranking is best read as:

- periapsis is structurally important
- velocity is still important physically, but its importance is spread across several correlated features
- `v_inf_kms` versus `v_inf_squared` and `periapsis_Rm` versus `periapsis_inverse` reflects alternate encodings of the same physics
- the ranking alone does not justify calling this overfitting

The more careful criticism is not "overfitting" but "importance dilution under correlated engineered features".
