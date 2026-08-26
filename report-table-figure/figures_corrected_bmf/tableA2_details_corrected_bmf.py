#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "report-table-figure" / "figures"
OUTPUT_FIGURES = ROOT / "report-table-figure" / "figures_corrected_bmf"
CORRECTED_BOUND = ROOT / "extraction-outputs_corrected_bmf" / "tables" / "bound_outcomes.csv"
CORRECTED_ARTIFACTS = ROOT / "ml" / "trainingartifacts_corrected_bmf"


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
    module = load_module("tableA2_details_corrected_bmf_runtime", FIGURES_DIR / "tableA2_details.py")
    module.SOURCE_PATH = CORRECTED_BOUND
    module.FIG_PATH = OUTPUT_FIGURES / "tableA2_details_corrected_bmf.png"
    module.FOLDS_PATH = CORRECTED_ARTIFACTS / "tuned_physics_gradient_boosting" / "grouped_cv_fold_assignments.csv"
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    module.main()


if __name__ == "__main__":
    main()
