#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FIGURES = ROOT / "report-table-figure" / "figures_corrected_bmf"
FIGURES_DIR = ROOT / "report-table-figure" / "figures"
CORRECTED_BOUND = ROOT / "extraction-outputs_corrected_bmf" / "tables" / "bound_outcomes.csv"
CORRECTED_FOF = ROOT / "extraction-outputs_corrected_bmf" / "tables" / "fof_outcomes.csv"


def ensure_corrected_fof() -> None:
    if not CORRECTED_FOF.exists():
        raise FileNotFoundError(f"Missing corrected fof_outcomes.csv at {CORRECTED_FOF}")


def corrected_bmf_ylim_upper() -> float:
    import pandas as pd
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
    ensure_corrected_fof()
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    module = load_module("figure2_build_corrected_bmf_runtime", FIGURES_DIR / "figure2_build.py")
    module.BOUND_FRACTION_YMAX_OVERRIDE = corrected_bmf_ylim_upper()
    temp_root = OUTPUT_FIGURES
    import types
    args = types.SimpleNamespace(
        fof_outcomes=str(CORRECTED_FOF),
        bound_outcomes=str(CORRECTED_BOUND),
        fragment_orbits="outputs/fragment_orbital_catalog.csv",
        eda_dir=None,
        tables_dir=str(temp_root),
        plots_dir=str(temp_root),
    )
    tables_dir, plots_dir = module.resolve_output_dirs(args)
    fof_outcomes = module.load_csv(Path(args.fof_outcomes))
    bound_outcomes = module.load_csv(CORRECTED_BOUND)
    run_frame = module.prepare_run_frame(fof_outcomes, bound_outcomes)
    module.plot_bound_mass(run_frame, plots_dir)
    source = plots_dir / "eccentricity_vs_bound_mass_fraction.png"
    corrected = OUTPUT_FIGURES / "figure2_used_in_report_corrected_bmf.png"
    if source.exists() and source != corrected:
        source.replace(corrected)


if __name__ == "__main__":
    main()
