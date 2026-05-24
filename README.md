# IRP FoF Extraction, EDA, and Baseline ML Pipeline

This repository supports an MSc IRP on tidal disruption during close encounters with Mars. The current repository is no longer a local-first scaffold. It now contains:

- FoF outcome extraction scripts for HDF5 simulation outputs
- before-extraction and after-extraction EDA workflows
- a working baseline ML pipeline for FoF-derived fragment statistics

The ML scope is intentionally limited to FoF-derived fragment statistics. It does not claim to predict moon formation, bound debris, orbital capture, or other orbital outcomes.

## Current Workflow

1. Build or refresh `outputs/manifest.csv` from FoF filenames.
2. Inspect sampled HDF5 schema when needed.
3. Extract FoF outcome tables into `outputs/`.
4. Run EDA on raw coverage and extracted FoF outcomes.
5. Train baseline regressors from `outputs/fof_outcomes.csv`.
6. Review ML diagnostics, including feature importance, residual behavior, train-vs-test gaps, and the effect of including `fof_linking_length`.

## Repo Structure

```text
scripts/
  make_manifest.py
  inspect_hdf5_schema.py
  extract_fof_outcomes.py
  eda_raw_data_overview.py
  eda_outcome_eda.py
  train_baseline_models.py
configs/
  paths.example.yaml
  ssh.example.yaml
docs/
eda/
ml/
outputs/
logs/
archived/
```

## Main Scripts

Create a manifest from FoF filenames:

```bash
python scripts/make_manifest.py --data-dir /path/to/hdf5_dir --output outputs/manifest.csv
```

Inspect sampled HDF5 structure:

```bash
python scripts/inspect_hdf5_schema.py \
  --data-dir /path/to/hdf5_dir \
  --output outputs/hdf5_schema_summary.csv \
  --limit 3
```

Extract FoF outcomes:

```bash
python scripts/extract_fof_outcomes.py \
  --data-dir /path/to/hdf5_dir \
  --outputs-dir outputs \
  --schema-samples 3 \
  --min-particles 20 \
  --exclude-group-id -1
```

Run raw-data coverage EDA:

```bash
python scripts/eda_raw_data_overview.py \
  --manifest outputs/manifest.csv \
  --schema outputs/hdf5_schema_summary.csv \
  --eda-dir eda/raw_data_overview
```

Run outcome EDA:

```bash
python scripts/eda_outcome_eda.py \
  --outcomes outputs/fof_outcomes.csv \
  --fragments outputs/fragment_catalog.csv \
  --errors outputs/extraction_errors.csv \
  --eda-dir eda/outcome_eda
```

Train baseline ML and diagnostics:

```bash
python scripts/train_baseline_models.py \
  --dataset outputs/fof_outcomes.csv \
  --ml-dir ml
```

## ML Scope

The baseline ML pipeline uses one row per simulation from `outputs/fof_outcomes.csv`.

Targets:

- `fragment_count_min_particles`
- `largest_fragment_particle_count`
- `largest_fragment_mass_kg` when present and valid

Excluded as a target:

- `fragment_mass_fraction` when it is constant and therefore non-informative

Models:

- `DummyRegressor`
- `Ridge`
- `RandomForestRegressor`
- `GradientBoostingRegressor`

Diagnostics:

- feature importance and permutation importance
- residual analysis by periapsis, mass, velocity, spin axis, and FoF linking length
- train-vs-test overfitting checks
- target difficulty comparison
- stability checks across datasets and feature sets
- with-vs-without `fof_linking_length` comparisons

## Output Directories

- `outputs/`: extracted CSV tables used by EDA and ML
- `eda/raw_data_overview/`: coverage summaries before full outcome extraction
- `eda/outcome_eda/`: summaries for extracted FoF outcomes
- `ml/`: baseline ML metrics, plots, and diagnostics

The consolidated narrative interpretation now lives in the root [interpretation_analysis.md](/Users/nny124/irp/interpretation_analysis.md:1).

## Safe Configuration

Do not commit secrets, passwords, or personal machine-specific configuration.

- Copy `configs/paths.example.yaml` to `configs/paths.yaml` for local machine-specific paths.
- Keep real SSH access in `~/.ssh/config`.
- HDF5 inputs, generated outputs, generated plots, generated tables, and serialized ML models are ignored by Git.

Recommended SSH config:

```sshconfig
Host imperial-hpc
    HostName login.cx3.hpc.ic.ac.uk
    User nny124
    IdentityFile ~/.ssh/id_ed25519
    ForwardAgent yes
```

## Data Safety

- Do not commit `*.hdf5`, `*.h5`, generated CSV outputs, generated plots, or serialized model artifacts.
- Do not hardcode passwords in code, config, notebooks, or shell scripts.
- Treat FoF linking length as a post-processing control parameter when interpreting ML behavior.
