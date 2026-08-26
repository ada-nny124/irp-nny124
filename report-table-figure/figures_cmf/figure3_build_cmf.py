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


def ensure_corrected_bound() -> None:
    if not CORRECTED_BOUND.exists():
        raise FileNotFoundError(f"Missing corrected bound_outcomes.csv at {CORRECTED_BOUND}")


def cmf_ylim_upper() -> float:
    import pandas as pd
    frame = pd.read_csv(CORRECTED_BOUND, low_memory=False)
    values = pd.to_numeric(frame["captured_mass_fraction"], errors="coerce").dropna()
    ymax = max(0.55, float(values.max()) * 1.08)
    return min(0.65, round(ymax / 0.05) * 0.05)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ensure_corrected_bound()
    module = load_module("figure3_build_cmf_runtime", FIGURES_DIR / "figure3_build.py")
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    with aliased_bound_dataset(CORRECTED_BOUND, "captured_mass_fraction") as aliased_bound:
        module.DATASET_PATH = aliased_bound
        module.OUTPUT_PATH = OUTPUT_FIGURES / "figure3_used_in_report_cmf.png"
        module.BMF_YMAX_OVERRIDE = cmf_ylim_upper()
        with patched_text_labels():
            module.main()


if __name__ == "__main__":
    main()
