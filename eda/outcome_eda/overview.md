# Outcome EDA

This directory is reserved for outcome-level EDA after `outputs/fof_outcomes.csv` is complete.

The script will refuse full analysis when `fof_outcomes.csv` has far fewer than 489 simulation rows.

Generated outputs include:

- `tables/` for dataset overview and grouped outcome summaries
- `plots/` for fragment-count and mass-metric visualisations
- `analysis_summary.txt` for a plain-text interpretation and ML-readiness note

## Re-run

```bash
python scripts/eda_outcome_eda.py   --outcomes outputs/fof_outcomes.csv   --fragments outputs/fragment_catalog.csv   --errors outputs/extraction_errors.csv   --eda-dir eda/outcome_eda
```
