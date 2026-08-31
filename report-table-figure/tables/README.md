# Eccentricity EDA

This folder contains plots and tables for how encounter eccentricity relates to fragmentation proxies.

Outputs:

- `tables/` contains threshold scans, summary statistics, and low-eccentricity edge cases.
- `plots/` contains the main eccentricity-versus-fragmentation visualisations.
- `analysis_summary.txt` gives a concise interpretation.

## Re-run

```bash
python eda/eccentricity_analysis/eda_eccentricity.py   --fof-outcomes extraction-outputs/tables/fof_outcomes.csv   --bound-outcomes extraction-outputs/tables/bound_outcomes.csv   --fragment-orbits outputs/fragment_orbital_catalog.csv   --eda-dir eda/eccentricity_eda
```
