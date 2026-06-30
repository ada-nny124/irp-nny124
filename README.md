# IRP Repository

## Project

**Parameter Sensitivity Analysis and Machine Learning Prediction of Tidal Disruption Outcomes in SPH Simulations of Martian Moon Formation**

## Abstract

This project investigates how the initial conditions of asteroid tidal-disruption simulations control the production of debris relevant to the origin of Mars's moons. The immediate challenge is that the current workflow extracts Friends-of-Friends (FoF) fragment statistics, which are useful proxies for disruption severity but are not yet equivalent to physically bound moon-forming material. The project therefore combines three stages: structured extraction from SPH/HDF5 outputs, exploratory and statistical analysis of parameter sensitivity, and machine-learning models that test how well disruption outcomes can be predicted from simulation metadata. Initial results already show strong structure in the data, with clear dependence on periapsis and substantial sensitivity to FoF linking length. The main next step is to extend the extractor so that bound-versus-unbound outcomes can be measured directly from matching physical snapshots.

## What This Repo Have

1. Extraction: Parse simulation filenames and HDF5 outputs into compact CSV tables such as outputs/manifest.csv, outputs/fof_outcomes.csv, and bound/unbound outcome summaries.

2. EDA: Inspect parameter coverage and outcome trends, especially the effects of periapsis, mass, velocity, and FoF linking length on fragmentation and retained bound mass.

3. Machine Learning: Train models using one row per simulation to evaluate how predictable disruption outcomes are from simulation metadata. Compare several baseline model families, including linear models and tree-based methods, using grouped cross-validation to avoid leakage between related simulation runs. Benchmark model performance across the main FoF and bound-aware targets, report the best-performing models, and analyse where predictions succeed or fail through residual and feature-importance checks.

For bound-retention work, the repository now treats these as the main regression targets:

- `bound_mass_fraction`
- `bound_fragment_count`
- `largest_bound_fragment_mass_kg`
- `average_bound_fragment_mass_kg`

Threshold labels such as `has_any_bound_mass` and `bound_mass_fraction_ge_0_1` are still retained for analysis and classifier diagnostics, but they are no longer the primary bound-retention prediction targets for triage.


## Main Scripts

- scripts/make_manifest.py: build a manifest from simulation filenames
- scripts/inspect_hdf5_schema.py: inspect sampled HDF5 structure
- scripts/extract_fof_outcomes.py: extract FoF outcome tables
- scripts/extract_bound_unbound_outcomes.py: compute bound vs unbound run-level/fragment-level outputs
- scripts/eda/eda_bound_eda.py: bound/unbound exploratory analysis
- scripts/train_baseline_models.py: baseline ML for FoF outcomes
- scripts/train_bound_models.py: run-level ML for bound outcomes
- scripts/train_triage_models.py: train a decision-support surrogate for FoF-derived fragmentation proxies

## Repository Layout

- outputs/: extracted CSV outputs used by analysis
- scripts/: extraction, EDA, and ML scripts
- documentation.md: interpretation and project notes
- deliverables/: project-plan and submission-facing material

## Notes

- Large generated artifacts are mostly kept out of Git.
- Some compact tables and representative plots are retained for review.

## SPH Fragmentation Triage Tool

The repository now includes a lightweight local triage demo that uses the extracted `outputs/fof_outcomes.csv` table to triage new simulation proposals. The tool predicts FoF-derived proxy outcomes rather than physically bound moon-forming material.

Disclaimer:
“This tool predicts FoF-derived fragmentation proxy outcomes. It does not replace SPH and does not directly validate long-term capture, disk mass, or moon formation.”

Use it to prioritise which new simulations deserve expensive SPH follow-up.

### Train the triage models

```bash
python scripts/train_triage_models.py
```

This writes local artifacts under `ml/triage/`:

- `fragmentation_classifier.pkl`
- `fragmentation_regressor.pkl`
- `metrics.json`
- `training_domain.json`

### Run the local demo from an editable template

```bash
python scripts/run_triage_demo.py
```

Edit [templates/triage_case_template.json](/Users/nny124/irp/templates/triage_case_template.json:1) and rerun the command. The script prints the prediction summary to the terminal and saves the full results table to `outputs/triage_demo_predictions.csv`.

If the model files are missing, the runner shows a message telling you to run `python scripts/train_triage_models.py`.

### Template fields

- `mass_log10_kg`
- `mass_code`
- `periapsis_Rm`
- `v_inf_kms`
- `spin_period_hr`
- `spin_axis`
- `resolution_code`
- `resolution_value`
- `timestep`
- `fof_linking_length`
- `has_explicit_spin`

### Example JSON template

```json
[
  {
    "case_name": "baseline_in_domain_case",
    "mass_log10_kg": 18.0,
    "mass_code": "A1800",
    "periapsis_Rm": 1.2,
    "v_inf_kms": 0.8,
    "spin_period_hr": 3.0,
    "spin_axis": "z",
    "resolution_code": "n60",
    "resolution_value": 60,
    "timestep": 90000,
    "fof_linking_length": 0.002,
    "has_explicit_spin": true
  }
]
```

### Optional web wrapper later

The same prediction core is also reusable from `app.py` if you later want a simple browser UI, but the local file-based runner is the primary demo path now.

### Output fields

- Fragmentation probability
- Predicted largest fragment mass
- Severity class
- Domain status: `in_domain`, `near_edge`, or `out_of_domain`
- SPH recommendation
- Short explanation of the recommendation
