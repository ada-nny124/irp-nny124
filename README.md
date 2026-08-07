# IRP Repository

Machine-learning screening of asteroid tidal disruption during close encounters with Mars, motivated by possible formation pathways for Phobos and Deimos.

## Abstract

This paper investigates asteroid tidal disruption during close encounters with Mars, a process in which tidal forces can fragment an incoming asteroid and leave part of the resulting debris gravitationally bound. Such retained material may be relevant to proposed formation pathways for Mars’ moons: Phobos and Deimos. However, exploring the encounter parameter space with high-fidelity smoothed-particle hydrodynamics (SPH) simulations is computationally expensive. To support more efficient exploration, SPH particle outputs are post-processed into fragmentation diagnostics and bound mass fraction (BMF), and machine-learning surrogates are trained using leakage-safe grouped cross-validation. Physics-derived features, advanced boosting models, regime-aware approaches, and two-stage hurdle architectures are evaluated alongside tree-based baselines. The baseline Random Forest achieves out-of-fold R2=0.8971, improving to R2=0.9225 with physics-derived features. The best overall model, an NGBoost hurdle architecture, reaches R2=0.9485 and RMSE=0.0211, while a CatBoost hurdle gives the lowest MAE of 0.0122. Controlled parameter slices and coverage-error diagnostics show that predictive reliability depends strongly on local SPH support. The resulting surrogate is therefore suitable for rapid in-domain screening and prioritisation, while sparse, edge, or physically detailed cases should still be evaluated with full SPH simulation.

## Repository Guide

- `src/triage/`
  The packaged local API and dashboard code. This is the user-facing screening interface.

- `scripts/`
  The main active extraction, training, packaging, and dashboard-serving scripts.

- `scripts/eda/`
  The main active EDA scripts used to produce retained EDA outputs.

- `archived/`
  Older, one-off, or non-core scripts and materials kept for reference rather than day-to-day use.

- `ml/`
  Trained model artifacts, selected metrics, packaged runtime inputs, and model-specific outputs.

- `eda/`
  Saved exploratory data analysis outputs, tables, and plots.

- `docs/`
  Written project documentation, figure references, and supporting notes.

- `extraction_outputs/`
  Canonical extracted SPH-derived tables used as the main analysis and API inputs.

- `report/`
  Report-ready notes and figures for presentation or interpretation.

- `deliverables/`
  Submitted project outputs and formal deliverable files.

- `logbook/`
  Project logbook and meeting record material.

- `title/`
  Title-page and repository title configuration material.

- `configs/`
  Example configuration files and local path templates.

## Local API / Dashboard

Clone the repository, install the package, and run the local screening interface:

```bash
pip install -e .
mars-flyby-dashboard
```

This starts the packaged local API/dashboard on `http://127.0.0.1:8000` by default.
