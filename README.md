# IRP Local-First Pipeline

This repository supports an MSc IRP on tidal disruption of asteroids during close encounters with Mars. The immediate goal is a local-first workflow that extracts structured metadata from simulation filenames, performs basic EDA, and sets up the path toward later HDF5 outcome extraction and baseline ML.

## Current Focus

- Work locally first from filename-level metadata.
- Avoid downloading or copying the full HPC dataset.
- Keep SSH access key-based and configuration-driven.
- Defer scientific outcome modeling until real targets are extracted from HDF5 content.

## Repo Structure

```text
scripts/
  make_manifest.py
  inspect_hdf5_schema.py
  eda_from_manifest.py
  baseline_ml.py
configs/
  paths.example.yaml
  ssh.example.yaml
outputs/
plots/
docs/
```

## Safe Configuration

Do not commit secrets, passwords, or personal machine-specific config.

- Copy `configs/paths.example.yaml` to `configs/paths.yaml` for local use if needed.
- Copy `configs/ssh.example.yaml` to `configs/ssh.yaml` only for notes; keep real access in `~/.ssh/config`.
- `configs/paths.yaml`, `configs/ssh.yaml`, `.env`, HDF5 data, generated CSVs, and plots are ignored by Git.

Recommended SSH config:

```sshconfig
Host imperial-hpc
    HostName login.cx3.hpc.ic.ac.uk
    User nny124
    IdentityFile ~/.ssh/id_ed25519
    ForwardAgent yes
```

Generate and install a key:

```bash
ssh-keygen -t ed25519 -C "nny124@imperial-hpc"
ssh-copy-id imperial-hpc
```

If `ssh-copy-id` is unavailable, copy `~/.ssh/id_ed25519.pub` into `~/.ssh/authorized_keys` on the cluster manually.

## Local-First Workflow

1. Build a manifest from filenames only.
2. Run EDA on the manifest.
3. Inspect a sample HDF5 schema when local or HPC file access is stable.
4. Add outcome extraction before any real ML training.

## Manifest Creation

From a local text file of filenames:

```bash
python scripts/make_manifest.py --from-file filenames.txt --output outputs/manifest.csv
```

From a local directory containing HDF5 files:

```bash
python scripts/make_manifest.py --from-dir path/to/hdf5_dir --output outputs/manifest.csv
```

From the HPC directory via an SSH alias:

```bash
python scripts/make_manifest.py \
  --ssh-host imperial-hpc \
  --remote-dir '$EPHEMERAL/martian_moons_data' \
  --output outputs/manifest.csv
```

This only lists filenames remotely. It does not download the 203G dataset.

## EDA

Run EDA locally from the manifest:

```bash
python scripts/eda_from_manifest.py --manifest outputs/manifest.csv
```

This prints counts by parameter, writes summary CSVs into `outputs/`, and saves simple coverage plots into `plots/`.

## HDF5 Schema Inspection

Inspect one sample file without loading full arrays:

```bash
python scripts/inspect_hdf5_schema.py --file path/to/sample.hdf5
```

This prints top-level groups, recursively lists datasets, records shapes and dtypes, and writes `outputs/hdf5_schema_summary.csv`.

## Baseline ML Placeholder

The baseline ML script currently builds manifest-derived features only:

```bash
python scripts/baseline_ml.py --manifest outputs/manifest.csv
```

If no real physical targets are available, it stops with:

```text
No physical outcome target available yet. Run outcome extraction first.
```

## Data Safety

- Do not commit `*.hdf5`, `*.h5`, generated outputs, plots, or secrets.
- Do not hardcode passwords in code, config, notebooks, or shell scripts.
- Prefer SSH keys and `~/.ssh/config` so the workflow does not prompt repeatedly for passwords.
