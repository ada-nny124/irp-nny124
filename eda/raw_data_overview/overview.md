# Raw Data Overview

This directory contains exploratory data analysis generated from:

- `outputs/manifest.csv`
- `outputs/hdf5_schema_summary.csv`

It does not use `outputs/fof_outcomes.csv` or `outputs/fragment_catalog.csv`.

## Contents

- `tables/`: dataset coverage tables, parameter counts, summary statistics, and sampled schema path summaries
- `plots/`: bar charts, coverage heatmaps, and file-size distribution plots
- `analysis_summary.txt`: textual summary of the current dataset and schema coverage

## Re-run

```bash
python scripts/eda_raw_data_overview.py \
  --manifest outputs/manifest.csv \
  --schema outputs/hdf5_schema_summary.csv \
  --eda-dir eda/raw_data_overview
```
