#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmf_runtime import aliased_bound_dataset, aliased_oof_predictions, patched_text_labels

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
    module = load_module("figure5_build_cmf_runtime", FIGURES_DIR / "figure5_build.py")
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    with aliased_bound_dataset(CORRECTED_BOUND, "captured_mass_fraction") as aliased_bound:
        with aliased_oof_predictions(CORRECTED_ARTIFACTS / "tuned_gradient_boosting" / "main_cmf_tuned_gradient_boosting_oof_predictions.csv") as aliased_oof:
            module.BOUND_SOURCE = aliased_bound
            module.PREDICTIONS_SOURCE = aliased_oof
            module.OUTPUT_PATH = OUTPUT_FIGURES / "figure5_used_in_report_cmf.png"
            with patched_text_labels():
                module.main()


if __name__ == "__main__":
    main()
