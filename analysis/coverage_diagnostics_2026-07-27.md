# Coverage Diagnostics Notes

Date: 2026-07-27

This note records the follow-up checks on the surrogate coverage and error maps.

## Updated figures

- Main coverage map: `ml/physics_structured_surrogate/plots/parameter_coverage_heatmaps.png`
- Main coverage vs error map: `ml/physics_structured_surrogate/plots/coverage_vs_error_heatmaps.png`
- Extended pairwise coverage/error map: `ml/physics_structured_surrogate/plots/extended_pairwise_coverage_error_heatmaps.png`
- Slide asset with the same coverage styling: `report/figures/model_trust_parameter_support.png`

Color convention:

- blue scale = coverage count
- red scale = mean absolute error
- green = no data in that projected bin
- pale blue = explicit zero count in coverage panels

## Why does the error heatmap have empty cells?

Because the error map is not a full prediction grid.

It is built from out-of-fold prediction residuals on the observed SPH rows only. So a blank cell in the error panel means:

- there was no SPH case in that projected 2D bin
- therefore there is no held-out error value to aggregate there

It does **not** mean the model cannot produce a prediction there. The screening model can still predict an input in that region, but there is no direct held-out SPH residual for that exact projected cell.

For the current mass-periapsis view:

- coverage bins with support and error present: `38`
- coverage bins with support but missing error: `0`
- bins with zero support and therefore missing error: `67`

So every supported mass-periapsis bin already has an error value. The empty red cells are purely unsupported projection bins.

## Does the model interpolate at mass = 19.5?

Yes, but this is an important nuance: `mass_log10_kg = 19.5` is already present in the archive. So this is not a pure unseen-mass interpolation test.

The archive contains `10` BMF rows at mass `19.5`, all at:

- `v_inf_kms = 0.0`
- `periapsis_Rm = 1.2` or `1.6`

Those cases are inside the global training range, but they sit in a sparse low-periapsis edge slice.

Grouped held-out performance on the `19.5` cases is poor:

- mass `19.5`, periapsis `1.2`: actual BMF `0.0913`, predicted `0.0134`, mean absolute error `0.0779`
- mass `19.5`, periapsis `1.6`: actual BMF `0.0002`, predicted `0.0827`, mean absolute error `0.0825`

Neighbouring masses behave much better on the same `v_inf=0` and periapsis slice:

- mass `19.0`, periapsis `1.2`: error `0.0135`
- mass `19.0`, periapsis `1.6`: error `0.0089`
- mass `20.0`, periapsis `1.2`: error `0.0242`
- mass `20.0`, periapsis `1.6`: error `0.0199`

So the current issue is not that the model fails everywhere near `19.5`. It is that the exact `19.5` low-periapsis zero-velocity slice behaves differently from the neighbouring masses and is only weakly constrained by the archive.

## Are near-zero BMF cases easier?

On average, yes.

- actual `BMF < 0.01`: mean absolute error `0.0152`
- actual `BMF >= 0.01`: mean absolute error `0.0203`

So some low-error regions are partly explained by many cases sitting near zero retained mass. Those are often easier to predict because the target is close to a floor.

That said, zero-BMF alone does not explain everything. The `19.5, periapsis=1.6, v_inf=0` slice is nearly zero in truth and is still one of the worst regions, so sparse corner behavior matters too.

## Weird / high-error regions

Worst projected bins in the current diagnostics:

- mass vs periapsis: `(19.5, 1.6)` with mean absolute error about `0.0825`
- mass vs velocity: `(19.5, 0.0)` with mean absolute error about `0.0802`
- mass vs spin axis: `(19.5, none)` with mean absolute error about `0.0802`

Other notable high-error cases include:

- some low-periapsis `mass=20.0` spin-specific runs
- a few `spin_axis = mz` zero-velocity cases

These are not broad failures across the archive. They are local slices where the target changes sharply and the archive is thin.

## Why trust or not trust smooth behavior outside training data?

Smoothness alone is not enough.

For tree ensembles, smooth-looking curves outside the observed domain can simply come from averaging terminal-node values. That is a visual property of the model, not direct evidence of physical correctness.

The defensible trust statement is:

- trust most when the query lies in a supported region with dense nearby SPH cases and low held-out error
- be cautious near parameter-space edges even if the prediction curve looks smooth
- do not treat unsupported green regions as validated interpolation

## Why make more coverage maps?

Because a single mass-periapsis projection can hide sparsity in other dimensions.

The extended pairwise diagnostic figure now adds:

- mass vs velocity
- mass vs FoF linking length
- periapsis vs velocity
- mass vs spin axis

This helps distinguish:

- regions that look supported in mass-periapsis but are sparse in velocity or spin
- regions where the error is really tied to a hidden parameter slice rather than the displayed 2D plane
