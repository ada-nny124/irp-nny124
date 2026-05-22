# Outcome EDA Interpretation

## Data coverage

- The outcome table is complete: `489/489` simulations were processed.
- No extraction errors were recorded.
- The fragment catalog contains `208,155` fragment rows across `475` simulations.
- Mass-based fields are populated for all outcome rows, so mass-scaled summaries are available everywhere.

## Main patterns

- `fof_linking_length` is the strongest non-physical control in the current outputs. At `0.0001`, the extractor finds no valid fragments; around `0.002` to `0.005`, fragment counts and largest-fragment size change substantially.
- Median behavior across the full table is `255` fragments per simulation and a median largest fragment of `259,351` particles, but the spread is wide:
  - `fragment_count_min_particles` ranges from `0` to `6,441`
  - `largest_fragment_particle_count` ranges from `0` to `3,922,894`
- Periapsis shows a strong structural trend. Low-to-mid periapsis cases tend to produce more fragments, while larger-periapsis cases produce fewer but larger dominant fragments.
- In the recommended comparison subset (`timestep == 90000`, `resolution_code == n65`, `fof_linking_length == 0.004`, excluding special cases), this pattern remains visible:
  - `r12`: mean fragment count `559.7`, mean largest fragment `417,780.9`
  - `r16`: mean fragment count `632.8`, mean largest fragment `800,729.2`
  - `r24`: mean fragment count `162.3`, mean largest fragment `2,249,856.0`
  - `r28`: mean fragment count `1.5`, mean largest fragment `2,848,443.0`
- Velocity matters, but the relationship is weaker and less monotonic than periapsis in the current grouped summaries. Within the clean subset, `v00` and `v06` show higher fragment counts than several intermediate velocities, so velocity effects likely interact with other parameters rather than acting alone.
- Mass scaling is strong. The highest-mass families produce both more fragments and larger dominant remnants on average. For example, `A2100` cases average `2,516.7` fragments with a mean largest fragment of `2,356,490` particles.

## Important caveat

- `fragment_mass_fraction` is not currently informative. It is `1.0` for every row where it is defined, which means it does not distinguish between fragmentation outcomes in this extracted table.
- Practically, that means the most useful ML or sensitivity targets right now are:
  - `fragment_count_min_particles`
  - `largest_fragment_particle_count`
  - potentially `largest_fragment_mass_kg`
- `fragment_mass_fraction` should be excluded from modelling until the denominator and fragment-selection logic are revised to produce variation.

## Recommended next step

- For downstream modelling or physics comparison, start with the clean subset used by the EDA script:
  - `timestep == 90000`
  - `resolution_code == n65`
  - `fof_linking_length == 0.004`
  - exclude special-case runs
- Treat `fof_linking_length` as a controlled preprocessing parameter, not a physical predictor, unless the modelling goal is explicitly to quantify FoF sensitivity.
