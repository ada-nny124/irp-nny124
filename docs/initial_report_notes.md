# Initial Report Notes

- Dataset downloaded and verified: 489 files, about 203G total.
- All currently identified files appear to be FoF HDF5 snapshots.
- Main parameter coverage:
  - `A1900` and `A2000` are the densest mass codes.
  - `n65` dominates the resolution coverage.
  - `r12`, `r16`, and `r20` are the densest periapsis values.
  - `v00` dominates the velocity coverage.
  - Spin sweeps include no-spin, `s030`, `s047`, `s086`, and `s170`.
  - Timestep `90000` dominates the outputs.

## Current Plan

1. Manifest generation from filenames.
2. HDF5 schema inspection on sample files.
3. EDA on parameter coverage.
4. Extract physical outcomes from FoF data.
5. Simple parameter analysis.
6. Baseline ML prediction.

## Questions For Supervisor

1. Are these 489 files supposed to be FoF-only, or should I also have initial/final non-FoF snapshots?
2. Which timestep should be considered the main final state?
3. Which outcome variable should I prioritise first: bound mass, fragment count, largest remnant mass, disk mass, or orbital debris?
4. Should FoF linking length be treated as an input parameter or fixed preprocessing choice?
5. Does each FoF file contain coordinates, velocities, masses, particle IDs, and group IDs?
