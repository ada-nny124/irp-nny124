# AI Usage Disclosure

This document records how generative AI was used in the development of this IRP
repository. It is part of the AI acknowledgement required by the Academic
Integrity Declaration. All AI-influenced material listed below was reviewed,
modified, tested, and verified by me before inclusion, and I take full
responsibility for the submitted work.

## Tools used

| Tool | Provider | Version | Service URL | How it was used |
| --- | --- | --- | --- | --- |
| Codex | OpenAI | 5.4 | https://chatgpt.com/codex | Code generation, iterative development of training/analysis scripts, refactoring, debugging, documentation and commit messages |

## Summary

Generative AI was used throughout the project to produce initial implementation
drafts and exploratory data analysis (EDA) code, and was consulted iteratively
for further development: debugging, refactoring, visualisation and analysis
scripts, tests, project documentation (README, update reports, notebooks), and
commit messages. AI was not used to write the Project Plan or the Final Report,
or to generate any figure, caption, or table explanation appearing in those
documents. Scientific decisions, interpretations, and final wording of the
report are my own.

## AI-influenced code

Each entry describes what AI generated or changed. "Initial implementation"
means the first working draft of that part was written by AI and subsequently
reviewed, modified and tested by me.

### archived/cleanup_junk/scripts/train_bound_models.py

- Filename parser and feature engineering (parse_simulation_filename, run-level feature table): initial implementation.
- Preprocessing utilities (median impute, one-hot encoding, scaling) and model factories: initial implementation.
- Grouped-fold helper (grouping by physical_file): initial implementation, then modified by me.
- Binary has_any_bound_mass classification task with GroupKFold, confusion-matrix/ROC outputs: initial implementation.
- Continuous bound_mass_fraction regression task (ridge/RF/gradient boosting), actual-vs-predicted and residual plots: initial implementation.
- CLI wiring and metrics/report outputs: initial implementation.
- Later changes: target-spec generalisation, average_bound_fragment_mass target, script regrouping (June 2026), AI-assisted edits, reviewed by me.

### eda/scripts/train_physics_structured_surrogate.py

- Canonical dataset construction (physics-derived feature frame): initial implementation.
- Grouped baseline evaluation (random forest and gradient boosting with scaled preprocessors): initial implementation.
- Compact hyperparameter tuning (grid search over n_estimators, max_depth, learning_rate via itertools.product): initial implementation.
- Reusable grouped model-config evaluator: initial implementation.
- Deterministic physics feature layer (eccentricity proxy, v_inf_squared, angular-momentum proxy, etc.): initial implementation, physics choices specified by me.
- FoF ablation and comparison stages: initial implementation.
- Secondary-target transform comparison stage: initial implementation.
- Promoted-model selection logic (determine_promoted_model, promotion only on meaningful R2 gain, reason records): initial implementation, promotion criteria set by me.
- Per-prediction trust flags (ensemble spread vs training domain): initial implementation.
- Representative-slice and coverage/error diagnostics: initial implementation.
- Packaging outputs and staged CLI dispatcher: initial implementation.
- Later changes: Mars-proximity physics features (July 2026), coverage-diagnostics EDA, notebook consolidation fix (August 2026), AI-assisted edits, reviewed by me.

### scripts/train_model_optimization_candidates.py, archived/cleanup_junk/scripts/train_baseline_models.py, archived/cleanup_junk/scripts/train_triage_models.py

- Training and optimisation-candidate evaluation scripts: initial implementation, subsequently edited (import fixes, candidate evaluation) with AI assistance and reviewed by me.

### src/triage/ (packaged API and dashboard)

- Hurdle-model bundle loading and BmfPrediction helpers: initial implementation.
- Feature handling and leaky-feature exclusion: initial implementation, leak list checked by me.
- Decision/screening rules and coverage checks: initial implementation, rules defined by me.
- Prediction pipeline (predict.py): initial implementation.
- Local dashboard/API server (server.py, templates): initial implementation.
- Unit tests (including scaffolds): AI-generated, extended and verified by me.

### scripts/app.py, scripts/run_triage_demo.py

- Dashboard entrypoint and demo runner: initial implementation.

### Extraction and data-processing scripts

- extraction_outputs/scripts/extract_fof_outcomes.py, extraction_outputs/scripts/extract_bound_unbound_outcomes.py: HDF5 schema inspection, FoF filename/group-id variant handling, bound/unbound outcome extraction. Initial implementation, with manual debugging by me against HPC outputs.
- extraction_outputs/scripts/make_manifest.py, extraction_outputs/scripts/inspect_hdf5_schema.py, extraction_outputs/scripts/deduplicate_outcomes.py: manifest building, schema audit, duplicate-outcome handling. Initial implementation, verified against extraction outputs by me.

### EDA, plotting and analysis scripts

- eda/scripts/ (large batch of exploratory, plotting, surrogate-analysis, and presentation-support scripts): initial implementations were AI-assisted. This includes raw-data overview, outcome EDA, bound-mass EDA, eccentricity EDA, retained-mass plots, confusion/ROC plots, regime-aware/global-importance plots, spin-importance plots, interpolation/trust diagnostics, Kegerreis-style figure regeneration, and presentation/report support scripts. Plot choices, filters, thresholds, figure selection, and scientific framing were specified and verified by me.
- archived/eda/scripts/analyze_dense_region_velocity_trend.py, archived/eda/scripts/analyze_sparse_region_velocity_contrast.py: matched dense/sparse-region velocity-sweep analysis (matched-group monotonicity checks, support counts, markdown summaries). Initial implementation; analysis questions and thresholds set by me.

### report-table-figure/

- Numbered figure/table build scripts and reproducibility packaging were assembled with AI assistance from existing project scripts and outputs. All the choice of analyses, figures and tables to produce, their design, and their inclusion in the report were my own decisions.
- The build scripts are AI-assisted packaging/reproducibility code.
- The *_used_in_report.csv files and copied report figure assets are exact submitted artifacts generated or assembled from my pipeline; they are not AI-authored scientific prose or interpretation.

### model_training.ipynb

- Section structure, markdown narration and cell implementations written with AI assistance across the project; modelling targets, results and interpretation checked by me.

### Packaging and configuration

- pyproject.toml, configs/*.example.yaml: initial drafts assisted by AI, adjusted by me.
- .gitignore: AI-assisted cleanup of ignore rules for temporary files and generated artifacts, reviewed and adjusted by me.

## AI-influenced documentation

| File | What AI did |
| --- | --- |
| README.md | Repository guide structure and phrasing drafted with AI; abstract text is my own report wording. |
| docs/ai_usage.md | AI-usage disclosure drafted and revised with AI assistance from my own record of tool usage; final wording and responsibility statement checked by me. |
| docs/documentation.md | Structure and drafting of the results summary with AI, from my own results and notes; corrected by me. |

## Commit messages

A portion of commit messages (including several short feat:/fix:/docs:
messages and a number of two-paragraph bodies) were drafted by AI and
edited/reviewed by me before committing. They accurately describe the changes
in each commit.

## Statement

I have reviewed, tested, understood and can explain, modify and reproduce the
AI-assisted code in this repository without AI assistance. AI influence is
described here at file and component level, you can also see the
commit history.
