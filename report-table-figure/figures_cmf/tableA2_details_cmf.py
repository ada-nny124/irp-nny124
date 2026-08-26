#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmf_runtime import aliased_bound_dataset, patched_text_labels

FIGURES_DIR = ROOT / "report-table-figure" / "figures"
OUTPUT_ROOT_ENV = os.environ.get("CMF_OUTPUT_ROOT")
OUTPUT_BASE = Path(OUTPUT_ROOT_ENV).resolve() if OUTPUT_ROOT_ENV else ROOT
OUTPUT_FIGURES = OUTPUT_BASE / "report-table-figure" / "figures_cmf"
CORRECTED_BOUND = ROOT / "extraction-outputs_corrected_bmf" / "tables" / "bound_outcomes.csv"
CORRECTED_ARTIFACTS = OUTPUT_BASE / "ml" / "trainingartifacts_cmf"


def ensure_corrected_bound() -> None:
    if not CORRECTED_BOUND.exists():
        raise FileNotFoundError(f"Missing corrected bound_outcomes.csv at {CORRECTED_BOUND}")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ensure_corrected_bound()
    module = load_module("tableA2_details_cmf_runtime", FIGURES_DIR / "tableA2_details.py")
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    with aliased_bound_dataset(CORRECTED_BOUND, "captured_mass_fraction") as aliased_bound:
        module.SOURCE_PATH = aliased_bound
        module.FIG_PATH = OUTPUT_FIGURES / "tableA2_details_cmf.png"
        module.FOLDS_PATH = CORRECTED_ARTIFACTS / "tuned_physics_gradient_boosting" / "grouped_cv_fold_assignments.csv"
        metrics = module.evaluate_feature_contribution(module.load_frame())
        with patched_text_labels():
            module.remake_plot(module.FIG_PATH, metrics)
        print(module.FIG_PATH)


if __name__ == "__main__":
    main()
