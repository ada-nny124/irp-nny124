#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "report-table-figure" / "figures"
OUTPUT_ROOT_ENV = os.environ.get("CORRECTED_BMF_OUTPUT_ROOT")
OUTPUT_BASE = Path(OUTPUT_ROOT_ENV).resolve() if OUTPUT_ROOT_ENV else ROOT
OUTPUT_FIGURES = OUTPUT_BASE / "report-table-figure" / "figures_corrected_bmf"
CORRECTED_BOUND = ROOT / "extraction-outputs_corrected_bmf" / "tables" / "bound_outcomes.csv"
CORRECTED_ARTIFACTS = OUTPUT_BASE / "ml" / "trainingartifacts_corrected_bmf"


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
    module = load_module("figure5_build_corrected_bmf_runtime", FIGURES_DIR / "figure5_build.py")
    module.BOUND_SOURCE = CORRECTED_BOUND
    module.PREDICTIONS_SOURCE = CORRECTED_ARTIFACTS / "tuned_gradient_boosting" / "main_bmf_tuned_gradient_boosting_oof_predictions.csv"
    module.OUTPUT_PATH = OUTPUT_FIGURES / "figure5_used_in_report_corrected_bmf.png"
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    module.main()


if __name__ == "__main__":
    main()
