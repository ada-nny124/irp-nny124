# IRP Repository

Origins of Mars's Moons: Physics-Structured ML Using Asteroid Tidal-Disruption SPH Simulations

## Abstract

One theory for the formation of Mars's moons, Phobos and Deimos, is that close collisions between asteroids and Mars can tidally damage the asteroid and leave some debris bonded to the planet. This process is thoroughly modelled by smoothed-particle hydrodynamics (SPH), but testing every possible combination of inputs is computationally costly. In this study, bound mass fraction (BMF) is predicted from the simulation inputs using machine learning models after SPH outputs are converted into fragmentation measurements. The primary model, a tuned Gradient Boosting regressor using the original simulation inputs, achieves a grouped cross-validated R² of 0.9234, MAE of 0.0159, and RMSE of 0.0257. Adding physics-derived features gives no meaningful extra improvement, so the simpler tuned raw-input model is retained. The models are assessed using grouped cross-validation. In a controlled set, predicted BMF generally decreases from roughly 0.27 at 1.2 Mars radii to approximately zero by about 2.5 Mars radii as periapsis increases. A similarly clear decline occurs as encounter speed increases, from roughly 0.25 at low velocity to approximately zero by about 1.2 km/s. While the impacts of spin and velocity depend on the encounter conditions, periapsis is the strongest overall physical input. As a result, distinct physical inputs may become more significant in different regions of the measured parameter range. Changes in spin in matched cases result in BMF discrepancies of up to 0.24 at bigger periapsis and between 0.06 and 0.09 below 1.5 Mars radii. Despite falling within the numerical training ranges, a sparsely sampled 1019.5 kg instance yields held-out errors close to 0.08 BMF, demonstrating that being within the input range does not ensure a trustworthy prediction. Therefore, the model does not replace SPH or prove physical causality, but it is helpful for quickly screening well-supported encounter circumstances and highlighting parameter correlations that call for additional SPH research.

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
