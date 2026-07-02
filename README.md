# IRP Repository

Parameter sensitivity and ML prediction of tidal-disruption outcomes in SPH simulations of Martian moon formation.

## What is here

- `scripts/`: extraction, EDA, and ML pipelines
- `outputs/`: core extracted tables used by the analysis
- `eda/`: selected EDA plots and summary tables kept for review
- `ml/`: selected model metrics and representative plots kept for review
- `src/irp_triage/`: lightweight triage prediction code
- `important_plots_and_tables.md`: slide-backed figure/table index

## Core data products kept in Git

- `outputs/manifest.csv`
- `outputs/fof_outcomes.csv`
- `outputs/bound_outcomes.csv`
- `outputs/hdf5_schema_summary.csv`

These are the smallest useful tables for checking the extraction and analysis pipeline without rerunning the full project.

## Main scripts

- `scripts/make_manifest.py`
- `scripts/extract_fof_outcomes.py`
- `scripts/extract_bound_unbound_outcomes.py`
- `scripts/eda/eda_raw_data_overview.py`
- `scripts/eda/eda_outcome_eda.py`
- `scripts/eda/eda_bound_eda.py`
- `scripts/eda/eda_eccentricity.py`
- `scripts/train_baseline_models.py`
- `scripts/train_bound_models.py`
- `scripts/train_triage_models.py`

## Repo policy

- Keep only the plots and tables that support the slides or are needed to understand the pipeline.
- Ignore regenerated bulk outputs, model binaries, full diagnostics trees, and long auto-generated text summaries.
- Use `important_plots_and_tables.md` as the reference for what should stay tracked.
