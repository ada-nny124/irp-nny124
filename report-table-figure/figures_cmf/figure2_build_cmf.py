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

OUTPUT_ROOT_ENV = os.environ.get("CMF_OUTPUT_ROOT")
OUTPUT_BASE = Path(OUTPUT_ROOT_ENV).resolve() if OUTPUT_ROOT_ENV else ROOT
OUTPUT_FIGURES = OUTPUT_BASE / "report-table-figure" / "figures_cmf"
FIGURES_DIR = ROOT / "report-table-figure" / "figures"
CORRECTED_BOUND = ROOT / "extraction-outputs_corrected_bmf" / "tables" / "bound_outcomes.csv"
CORRECTED_FOF = ROOT / "extraction-outputs_corrected_bmf" / "tables" / "fof_outcomes.csv"


def ensure_corrected_fof() -> None:
    if not CORRECTED_FOF.exists():
        raise FileNotFoundError(f"Missing corrected fof_outcomes.csv at {CORRECTED_FOF}")


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
    ensure_corrected_fof()
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    module = load_module("figure2_build_cmf_runtime", FIGURES_DIR / "figure2_build.py")
    module.BOUND_FRACTION_YMAX_OVERRIDE = cmf_ylim_upper()
    temp_root = OUTPUT_FIGURES
    import types
    with aliased_bound_dataset(CORRECTED_BOUND, "captured_mass_fraction") as aliased_bound:
        args = types.SimpleNamespace(
            fof_outcomes=str(CORRECTED_FOF),
            bound_outcomes=str(aliased_bound),
            fragment_orbits="outputs/fragment_orbital_catalog.csv",
            eda_dir=None,
            tables_dir=str(temp_root),
            plots_dir=str(temp_root),
        )
        _tables_dir, plots_dir = module.resolve_output_dirs(args)
        fof_outcomes = module.load_csv(Path(args.fof_outcomes))
        overlapping_columns = [
            "bound_mass_fraction",
            "bound_fragment_count",
            "largest_bound_fragment_mass_kg",
            "unbound_mass_fraction",
        ]
        fof_outcomes = fof_outcomes.drop(columns=[c for c in overlapping_columns if c in fof_outcomes.columns])
        bound_outcomes = module.load_csv(Path(args.bound_outcomes))
        run_frame = module.prepare_run_frame(fof_outcomes, bound_outcomes)
        with patched_text_labels():
            module.plot_bound_mass(run_frame, plots_dir)
        source = plots_dir / "eccentricity_vs_bound_mass_fraction.png"
        corrected = OUTPUT_FIGURES / "figure2_used_in_report_cmf.png"
        if source.exists() and source != corrected:
            source.replace(corrected)
        if not corrected.exists():
            raise FileNotFoundError(f"Expected CMF Figure 2 output was not created at {corrected}")


if __name__ == "__main__":
    main()
