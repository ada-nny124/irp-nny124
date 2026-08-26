#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "report-table-figure" / "figures"
OUTPUT_FIGURES = ROOT / "report-table-figure" / "figures_corrected_bmf"
CORRECTED_BOUND = ROOT / "extraction-outputs_corrected_bmf" / "tables" / "bound_outcomes.csv"


def ensure_corrected_bound() -> None:
    if not CORRECTED_BOUND.exists():
        raise FileNotFoundError(f"Missing corrected bound_outcomes.csv at {CORRECTED_BOUND}")


def corrected_bmf_ylim_upper() -> float:
    frame = pd.read_csv(CORRECTED_BOUND, low_memory=False)
    values = pd.to_numeric(frame["bound_mass_fraction"], errors="coerce").dropna()
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
    module = load_module("figureA1_build_corrected_bmf_runtime", FIGURES_DIR / "figureA1_build.py")
    module.DATASET_PATH = CORRECTED_BOUND
    module.OUTPUT_PATH = OUTPUT_FIGURES / "figureA1_used_in_report_corrected_bmf.png"
    module.BMF_YMAX_OVERRIDE = corrected_bmf_ylim_upper()
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    module.main()


if __name__ == "__main__":
    main()
