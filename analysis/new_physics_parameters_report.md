# New Physics-Defined Parameters

Date: 2026-07-27

This note records the new derived parameters added to the physics-structured surrogate and what happened after rerunning the feature-ablation stage.

## Added parameters

1. `asteroid_radius_km`
   Defined from target mass assuming a fixed rocky bulk density of `2700 kg m^-3`:
   `R = (3 M / 4 pi rho)^(1/3)`.

2. `time_within_2_mars_radii_hr`
   Approximate total encounter time spent inside `2.0 R_Mars`, computed from periapsis and `v_inf` with a two-body flyby timing model.

3. `time_within_tidal_disruption_hr`
   Approximate total encounter time spent inside an estimated tidal-disruption radius.
   I used a fluid Roche-style threshold
   `r_tidal = 2.44 R_Mars (rho_Mars / rho_asteroid)^(1/3)`,
   which gives `2.7661 R_Mars` for `rho_Mars = 3933.5 kg m^-3` and `rho_asteroid = 2700 kg m^-3`.

## Result summary

The new parameters were not neutral. They improved some promoted-model variants and strongly changed the permutation-importance ranking.

Grouped-CV `R^2` changes for the physics-feature ablation:

- `physics_with_fof`, gradient boosting: `0.9197 -> 0.9377`
- `physics_with_fof`, random forest: `0.9225 -> 0.9227`
- `physics_without_fof`, gradient boosting: `0.9191 -> 0.9316`
- `physics_without_fof`, random forest: `0.9126 -> 0.9071`

So the new features help most for gradient boosting, are nearly neutral for the FoF-aware random forest, and slightly hurt the FoF-free random forest.

## Importance ranking

For the current `physics_with_fof` random forest, the new features rank as:

- `time_within_tidal_disruption_hr`: `0.1492`
- `asteroid_radius_km`: `0.0713`
- `time_within_2_mars_radii_hr`: `0.0340`

That makes `time_within_tidal_disruption_hr` the highest-ranked new input and the second-highest feature overall after the outcome-derived `largest_fragment_mass_fraction`.

## Interpretation

`time_within_tidal_disruption_hr` appears useful because it combines two ingredients the model already cares about:

- how close the encounter gets to Mars
- how long the body remains in the strongest tidal regime

That is more physically targeted than raw periapsis alone, so it is reasonable that the feature carries extra signal.

`time_within_2_mars_radii_hr` helps less. A fixed `2 R_Mars` threshold is a coarser heuristic than the tidal-disruption threshold, so it is less aligned with the disruption physics.

`asteroid_radius_km` is physically interpretable, but here it is largely a monotonic transform of mass under a fixed-density assumption. That means it is not independent new information in the strictest sense. Its usefulness is mostly that the transformed scale is easier for some models to split on than `mass_log10_kg` alone.

## Bottom line

The new physics-defined parameters are worth keeping.

- `time_within_tidal_disruption_hr` adds a strong and physically sensible signal.
- `time_within_2_mars_radii_hr` adds weaker but still nonzero information.
- `asteroid_radius_km` is interpretable, but it should be described as a mass-derived transform, not a wholly independent physical observable.
