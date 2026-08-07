# IRP Repository

Parameter sensitivity and ML prediction of tidal-disruption outcomes in SPH simulations of Martian moon formation.

## What is here

- `scripts/`: extraction, EDA, diagnostics, and ML pipelines
- `extraction_outputs/`: core extracted tables used by the analysis
- `eda/`: selected EDA plots and summary tables kept for review
- `ml/`: selected model metrics and representative plots kept for review
- `src/triage/`: builds features, loads saved models, predicts outcomes, and serves the local dashboard/API
- `docs/important_plots_and_tables.md`: figure/table index used in presentation slides

## Core data products kept in Git

- `extraction_outputs/manifest.csv`
- `extraction_outputs/fof_outcomes.csv`
- `extraction_outputs/bound_outcomes.csv`
- `extraction_outputs/hdf5_schema_summary.csv`

These are kept in Git because they are small enough and let a cloned repo run the API without re-running the full extraction pipeline.

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

ML training & results summarised in Notebook for demo: `model_training.ipynb`

## Installable dashboard / API

After cloning the repo, install the package in editable mode:

```bash
pip install -e .
```

Run the local dashboard/API:

```bash
mars-flyby-dashboard
```

This serves the local dashboard and JSON API on `http://127.0.0.1:8000` by default.

You can also choose a different bind address or port:

```bash
mars-flyby-dashboard --host 127.0.0.1 --port 8000
```

Equivalent alias:

```bash
mars-flyby-api
```

The package entrypoint delegates to the local dashboard server in `scripts/app.py`, so the cloned repository still provides the required models, tables, and HTML assets.

If you edit the dashboard template or server text, stop the running server and restart it before refreshing the browser.
