# FoF Extraction Notes

## Scope

This pipeline extracts only conservative FoF-level outcomes that are directly supported by each HDF5 file:

- filename-derived simulation parameters in `outputs/manifest.csv`
- sampled HDF5 schema metadata in `outputs/hdf5_schema_summary.csv`
- one row per simulation in `outputs/fof_outcomes.csv`
- one row per FoF fragment in `outputs/fragment_catalog.csv`
- extraction failures in `outputs/extraction_errors.csv`, only when errors occur

It does not attempt bound-debris, orbital capture, disk mass, or moon-forming debris science. Those require validated positions, velocities, units, frame conventions, and energy calculations.

## Safe First-Stage Outcomes

For first-pass EDA and model training, the safest target table is `outputs/fof_outcomes.csv`. It contains one row per FoF snapshot file. The most defensible columns are:

- filename parameters: `mass_code`, `spin_code`, `resolution_code`, `periapsis_code`, `velocity_code`, `timestep`, `fof_linking_length`
- `n_fof_groups`: unique FoF group IDs after excluding any configured sentinel IDs
- `fragment_count_min_particles`: number of groups with at least `min_particles`
- `largest_fragment_particle_count`: particle-count proxy for the largest surviving fragment
- `largest_fragment_mass_kg`, `total_fragment_mass_kg`, `fragment_mass_fraction`: mass-based outcomes only when particle masses and mass units are present

## Real Mass Metrics vs Proxies

Mass-based outcomes are only populated when the HDF5 file includes both:

- `PartType0/Masses` (or equivalent gas-particle masses)
- unit metadata that can convert particle masses to kilograms

When that information is available:

- `mass_metrics_available = True`
- `mass_unit = kg`
- `largest_fragment_mass_kg`, `total_fragment_mass_kg`, and `fragment_mass_fraction` are physically scaled from HDF5 mass fields

Particle-count fields are still proxies:

- `largest_fragment_particle_count`
- `particle_count` and `particle_fraction_of_snapshot` in `fragment_catalog.csv`

Those should not be described as mass unless you explicitly choose to use them as resolution-dependent proxies.

## Current Group Filters

The extraction script supports:

- `--min-particles 20` to define what counts as a fragment in summary metrics
- `--exclude-group-id -1` to drop the common unassigned/background FoF sentinel from group counts

In addition, the extractor reads `Parameters.attrs["FOF:group_id_default"]` when present and excludes that HDF5-defined default group ID automatically. `n_fof_groups` counts all non-excluded FoF IDs. `fragment_count_min_particles` applies the particle threshold on top of that.

## HDF5 Schema Summary

`outputs/hdf5_schema_summary.csv` is sample-only. It records dataset paths, shapes, dtypes, and attribute previews for the requested number of files without loading full arrays into memory.

## Running on Imperial HPC

The scripts assume the dataset already exists on the cluster filesystem under `$EPHEMERAL/martian_moons_data`.

Login-shell test:

```bash
module load tools/prod h5py/3.12.1-foss-2024a
python3 scripts/extract_fof_outcomes.py \
  --data-dir "$EPHEMERAL/martian_moons_data" \
  --outputs-dir outputs \
  --limit 3 \
  --schema-samples 3 \
  --min-particles 20 \
  --exclude-group-id -1
```

Full extraction through Slurm:

```bash
sbatch scripts/run_extract_fof.slurm
squeue -u "$USER"
tail -f logs/irp_fof_extract_*.out
```

## Copying Outputs Locally

Only small CSV outputs should be committed. If an output is not committed, copy it from HPC with:

```bash
rsync -avz nny124@login.cx3.hpc.ic.ac.uk:~/irp-nny124/outputs/ ./outputs/
```
