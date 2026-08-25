# Approximate parent-normalized bound-fraction diagnostic

This diagnostic leaves the project target unchanged and adds a separate approximate paper-comparison path.

## Definition

`approx_f_bnd_parent = bound_mass_kg / target_mass_kg`

## Why this is closer to the paper's `f_bnd`

- It uses the original asteroid mass in the denominator rather than resolved fragment mass.
- That makes it closer in normalization to the paper's bound fraction than the current project metric `bound_mass_fraction`.

## Why it is still not identical

- The numerator still comes from saved resolved-FoF bound mass only.
- Unresolved/background material cannot be recovered from the saved extraction outputs.
- This is not equivalent to paper `f_capt`.
- Exact `f_capt` cannot be reconstructed from saved outputs without raw particle/orbit HDF5 data.

## Paper-reference subset used here

- Mass: `10^20 kg` (`A2000`)
- Spin: no spin
- Resolution: `n65`
- FoF filter applied for de-duplication: yes, 0.004 only
- Saved comparison rows in subset: 38

## Numerical comparison against paper-reference rows

- Matched scenarios: 27
- Mean absolute difference vs paper `f_bnd`: 0.2146
- Max absolute difference vs paper `f_bnd`: 0.3296
- Mean signed difference (`approx_f_bnd_parent - paper_f_bnd`): -0.2146
- Range of (`bound_mass_fraction - approx_f_bnd_parent`) on matched rows: 0.0000 to 0.0003

## Source checks

- Diagnostic CSV rows: 407
- `approx_f_bnd_parent` non-null rows: 407
- Saved `fof_outcomes.csv` has capture-specific columns needed for exact `f_capt` reconstruction: no
