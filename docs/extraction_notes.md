# FoF Extraction Notes

This is the first real HDF5 extraction stage for the Martian-moons dataset. It is designed to run on Imperial HPC, next to the downloaded FoF snapshot files, without copying the raw HDF5 data into the repository.

## Scope

- Extract filename-level parameters into `outputs/manifest.csv`.
- Inspect a sample of HDF5 schemas into `outputs/hdf5_schema_summary.csv`.
- Extract conservative FoF group statistics into:
  - `outputs/fof_outcomes.csv`
  - `outputs/fragment_catalog.csv`
- Write `outputs/extraction_errors.csv` only if errors occur.

## What The Current Extractor Computes

The current extractor only computes FoF group statistics that can be obtained directly from fields present in the HDF5 files:

- `fragment_count_min_particles`
- `largest_fragment_particle_count`
- `largest_fragment_mass`, only if a particle mass dataset is found
- `total_fragment_mass`, only if a particle mass dataset is found
- `fragment_mass_fraction`, only if a particle mass dataset is found

## What It Does Not Yet Compute

This stage does not yet calculate:

- bound mass
- orbital capture
- disk mass
- debris orbital classification

Those require validated physics and field-level interpretation beyond simple FoF extraction.

## Interpreting Mass-Like Outputs

- If `mass_dataset` is populated and `mass_source` is `particle_masses`, then mass-like outputs come from an actual particle-mass field.
- If `mass_dataset` is blank and `mass_source` is `particle_count_proxy`, then `largest_fragment_mass` and `total_fragment_mass` are only particle-count proxies and must not be presented as physical mass.

## Quality Checks

- Inspect `outputs/extraction_errors.csv` before trusting downstream analysis.
- Review `outputs/hdf5_schema_summary.csv` to confirm which datasets are actually present.
- Treat `fragment_count_min_particles` and `largest_fragment_particle_count` as the safest first targets for early sensitivity analysis.

## HPC Run Outline

Test on a few files first:

```bash
python scripts/extract_fof_outcomes.py \
  --data-dir "$EPHEMERAL/martian_moons_data" \
  --outputs-dir outputs \
  --limit 3 \
  --schema-samples 3 \
  --min-particles 20 \
  --exclude-group-id -1
```

Then submit the full batch run:

```bash
sbatch scripts/run_extract_fof.slurm
squeue -u "$USER"
tail -f logs/irp_fof_extract_*.out
```
