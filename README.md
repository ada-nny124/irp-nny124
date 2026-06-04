# IRP Repository

## Project

**Parameter Sensitivity Analysis and Machine Learning Prediction of Tidal Disruption Outcomes in SPH Simulations of Martian Moon Formation**

## Abstract

This project investigates how the initial conditions of asteroid tidal-disruption simulations control the production of debris relevant to the origin of Mars's moons. The immediate challenge is that the current workflow extracts Friends-of-Friends (FoF) fragment statistics, which are useful proxies for disruption severity but are not yet equivalent to physically bound moon-forming material. The project therefore combines three stages: structured extraction from SPH/HDF5 outputs, exploratory and statistical analysis of parameter sensitivity, and machine-learning models that test how well disruption outcomes can be predicted from simulation metadata. Initial results already show strong structure in the data, with clear dependence on periapsis and substantial sensitivity to FoF linking length. The main next step is to extend the extractor so that bound-versus-unbound outcomes can be measured directly from matching physical snapshots.

## What This Repo Have

1. Extraction
   Parse simulation filenames and HDF5 outputs into compact CSV tables such as outputs/manifest.csv, outputs/fof_outcomes.csv, and bound/unbound outcome summaries.

2. EDA
   Inspect parameter coverage and outcome trends, especially the effects of periapsis, mass, velocity, and FoF linking length on fragmentation and retained bound mass.

3. Machine Learning
   Train baseline models on one row per simulation to test how predictable disruption outcomes are from simulation metadata.

## Main Scripts

- scripts/make_manifest.py: build a manifest from simulation filenames
- scripts/inspect_hdf5_schema.py: inspect sampled HDF5 structure
- scripts/extract_fof_outcomes.py: extract FoF outcome tables
- scripts/extract_bound_unbound_outcomes.py: compute bound vs unbound run-level/fragment-level outputs
- scripts/eda_bound_eda.py: bound/unbound exploratory analysis
- scripts/train_baseline_models.py: baseline ML for FoF outcomes
- scripts/train_bound_models.py: run-level ML for bound outcomes

## Repository Layout

- outputs/: extracted CSV outputs used by analysis
- scripts/: extraction, EDA, and ML scripts
- documentation.md: interpretation and project notes
- deliverables/: project-plan and submission-facing material

## Notes

- Large generated artifacts are mostly kept out of Git.
- Some compact tables and representative plots are retained for review.
