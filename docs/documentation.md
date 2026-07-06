# Project Notes

## Scope

This repository studies two related outcome families from existing SPH runs:

- FoF fragmentation targets
- bound-retention proxy targets

The retained-mass quantities here are post-processed proxy outcomes. They are useful for comparing runs, but they are not direct proof of disk formation or moon formation.

## Current dataset

- `489` simulation outputs in the full extracted set
- `166` rows in the controlled clean subset
- grouped validation by `physical_file` for ML

## What stays tracked

- core extracted CSVs in `outputs/`
- slide-backed EDA plots and summary tables
- compact ML metrics tables
- a small number of representative ML plots

## What is ignored

- bulk generated plot trees
- model binaries
- triage output artifacts
- duplicate or exploratory tables
- long auto-generated text summaries

See `important_plots_and_tables.md` for the exact keep list.
