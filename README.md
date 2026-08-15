# IRP Repository

Machine-learning screening of asteroid tidal disruption during close encounters with Mars, motivated by possible formation pathways for Phobos and Deimos.

## Abstract

Close encounters between asteroids and Mars can tidally disrupt the asteroid and leave some debris bound to the planet, a process that has been proposed as one possible pathway toward the formation of Mars’s moons, Phobos and Deimos. Smoothed-particle hydrodynamics (SPH) models this process in detail, but testing every input combination is computationally expensive. In this project, SPH outputs are processed into fragmentation measures, and machine learning models are then used to predict bound mass fraction (BMF) from the simulation inputs. The models are evaluated using grouped cross-validation, and the main model is a Random Forest using physics-derived features calculated from the raw simulation inputs, achieving a grouped cross-validated R2 of 0.9225, MAE of 0.0179, and RMSE of 0.0259. In controlled tests, predicted BMF decreases from about 0.25 at 1.2 Mars radii as periapsis increases, while the available SPH cases fall to approximately zero beyond about 2.2 Mars radii. Increasing encounter speed produces a clearer decrease from about 0.25 at 0 km/s to approximately zero above 1 km/s. Periapsis is the strongest overall physical input, while the effects of spin and velocity depend on the encounter conditions, meaning that different physical inputs can become more influential in different parts of the sampled parameter range. In matched cases, changing spin produces BMF differences of about 0.06-0.09 below 1.5 Mars radii and up to about 0.24 at larger periapsis.. A sparsely sampled 1019.5 kg case gives held-out errors near 0.08 BMF despite lying within the numerical training ranges, showing that being within the input range does not guarantee a reliable prediction. The model is therefore useful for rapidly screening well-supported encounter conditions and identifying parameter interactions that warrant further SPH study, but it does not replace SPH or establish physical causation.

## Repository Guide

- `src/triage/`
  The packaged local API and dashboard code. This is the user-facing screening interface.

- `scripts/`
  The main tracked automation scripts for extraction, training, packaging, and dashboard serving.

- `ml/`
  Tracked model assets, packaged runtime inputs, and selected benchmark or surrogate materials that support the deployed workflow.

- `docs/`
  Written project documentation, figure references, and supporting notes.

- `extraction_outputs/`
  Canonical SPH-derived tables that feed the downstream modelling and screening pipeline.

- `deliverables/`
  Submitted project outputs and formal deliverable files.

- `logbook/`
  Project logbook and meeting record material.

- `title/`
  Title-page and repository title configuration material.

- `configs/`
  Example configuration files and local path templates.

Ignored local outputs such as exploratory plots, report assets, archived materials, and scratch diagnostics are intentionally excluded from this guide because they are not part of the tracked repo surface.

## Local API / Dashboard

Clone the repository, install the package, and run the local screening interface:

```bash
pip install -e .
mars-flyby-dashboard
```

This starts the packaged local API/dashboard on `http://127.0.0.1:8000` by default.
