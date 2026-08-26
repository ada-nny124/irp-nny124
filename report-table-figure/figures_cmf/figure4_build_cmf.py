#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmf_runtime import aliased_bound_dataset, patched_text_labels

FIGURES_DIR = ROOT / "report-table-figure" / "figures"
OUTPUT_ROOT_ENV = os.environ.get("CMF_OUTPUT_ROOT")
OUTPUT_BASE = Path(OUTPUT_ROOT_ENV).resolve() if OUTPUT_ROOT_ENV else ROOT
OUTPUT_FIGURES = OUTPUT_BASE / "report-table-figure" / "figures_cmf"
CORRECTED_BOUND = ROOT / "extraction-outputs_corrected_bmf" / "tables" / "bound_outcomes.csv"


def corrected_figure4_ylim_upper(bound_table: Path) -> float:
    rows = list(pd.read_csv(bound_table, low_memory=False).to_dict(orient="records"))
    spin_order = ("no spin", "3h z", "4.7h z")
    velocity_order = (0.0, 0.2, 0.4, 0.6, 0.8)
    grouped: dict[tuple[float, float], dict[str, float]] = {}
    vel_grouped: dict[tuple[str, float], dict[float, float]] = {}
    for row in rows:
        if row["mass_code"] != "A2000" or row["resolution_code"] != "n65" or str(row["timestep"]) != "90000" or float(row["fof_linking_length"]) != 0.004:
            continue
        spin_code_raw = row.get("spin_code")
        spin_code = "" if spin_code_raw is None or str(spin_code_raw).lower() == "nan" else str(spin_code_raw)
        spin = "no spin" if not spin_code else f"{float(spin_code[1:4]) / 10.0:g}h {spin_code[4:] or 'none'}"
        periapsis = float(str(row["periapsis_code"])[1:]) / 10.0
        velocity = float(str(row["velocity_code"])[1:]) / 10.0
        bmf = float(row["captured_mass_fraction"])
        grouped.setdefault((velocity, periapsis), {})[spin] = bmf
        vel_grouped.setdefault((spin, periapsis), {})[velocity] = bmf
    deltas = []
    for (_, _), sm in grouped.items():
        if set(spin_order).issubset(sm):
            vals = [sm[s] for s in spin_order]
            deltas.append(max(vals) - min(vals))
    for (_, _), vm in vel_grouped.items():
        if len(vm) >= 3:
            vals = list(vm.values())
            deltas.append(max(vals) - min(vals))
    max_delta = max(deltas) if deltas else 0.30
    ymax = max(0.35, max_delta * 1.08)
    return min(0.50, round(ymax / 0.05) * 0.05)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    module = load_module("figure4_build_cmf_runtime", FIGURES_DIR / "figure4_build.py")
    import sys
    previous_argv = sys.argv[:]
    try:
        with aliased_bound_dataset(CORRECTED_BOUND, "captured_mass_fraction") as aliased_bound:
            module.YMAX_OVERRIDE = corrected_figure4_ylim_upper(CORRECTED_BOUND)
            sys.argv = [
                str(FIGURES_DIR / "figure4_build.py"),
                "--bound-table",
                str(aliased_bound),
                "--png-out",
                str(OUTPUT_FIGURES / "figure4_used_in_report_cmf.png"),
            ]
            with patched_text_labels():
                module.main()
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
