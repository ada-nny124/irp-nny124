# Bound EDA

Generated artifacts for fragment-level and run-level bound vs unbound analysis.

Outputs:

- `tables/` contains summary tables, parameter aggregates, and sample rows.
- `plots/` contains fragment-level and run-level visualisations.

## Re-run

```bash
python scripts/eda/eda_bound_eda.py   --fragments outputs/fragment_orbital_catalog.csv   --outcomes outputs/bound_outcomes.csv   --log outputs/bound_unbound_extraction_log.csv   --eda-dir eda/bound_eda
```
