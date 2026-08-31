from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MARS_MU_KM3_S2 = 4.282837e4
MARS_RADIUS_KM = 3389.5
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT_ENV = os.environ.get("PIPELINE_OUTPUT_ROOT")
OUTPUT_BASE = Path(OUTPUT_ROOT_ENV).resolve() if OUTPUT_ROOT_ENV else SCRIPT_DIR.parents[1]
DEFAULT_PLOTS_DIR = OUTPUT_BASE / "report-table-figure" / "figures"
BOUND_FRACTION_YMAX_OVERRIDE: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fof-outcomes",
        default="extraction-outputs/tables/fof_outcomes.csv",
        help="Path to fof_outcomes.csv",
    )
    parser.add_argument(
        "--bound-outcomes",
        default="extraction-outputs/tables/bound_outcomes.csv",
        help="Path to bound_outcomes.csv",
    )
    parser.add_argument(
        "--fragment-orbits",
        default="outputs/fragment_orbital_catalog.csv",
        help="Optional path to fragment_orbital_catalog.csv or equivalent orbital fragment catalog",
    )
    parser.add_argument(
        "--eda-dir",
        default=None,
        help="Deprecated compatibility option. If set, the script uses <eda-dir>/tables and <eda-dir>/plots.",
    )
    parser.add_argument("--tables-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--plots-dir", default=str(DEFAULT_PLOTS_DIR), help="Directory for PNG outputs")
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.eda_dir:
        base_dir = Path(args.eda_dir)
        plots_dir = base_dir / "plots"
    else:
        plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def parse_code_numeric(frame: pd.DataFrame, column: str, prefix: str, scale: float) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    extracted = frame[column].astype(str).str.extract(rf"{prefix}(\d+)")[0]
    return pd.to_numeric(extracted, errors="coerce") / scale


def scaled_from_value(frame: pd.DataFrame, column: str, scale: float) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce") / scale


def periapsis_rm(frame: pd.DataFrame) -> pd.Series:
    values = scaled_from_value(frame, "periapsis_value", 10.0)
    return values if values.notna().any() else parse_code_numeric(frame, "periapsis_code", "r", 10.0)


def velocity_kms(frame: pd.DataFrame) -> pd.Series:
    values = scaled_from_value(frame, "velocity_value", 10.0)
    return values if values.notna().any() else parse_code_numeric(frame, "velocity_code", "v", 10.0)


def mass_log10_kg(frame: pd.DataFrame) -> pd.Series:
    values = scaled_from_value(frame, "mass_value", 100.0)
    return values if values.notna().any() else parse_code_numeric(frame, "mass_code", "A", 100.0)


def eccentricity_proxy(periapsis_rm_values: pd.Series, velocity_kms_values: pd.Series) -> pd.Series:
    periapsis_km = periapsis_rm_values * MARS_RADIUS_KM
    with np.errstate(divide="ignore", invalid="ignore"):
        proxy = 1.0 + (periapsis_km * np.square(velocity_kms_values)) / MARS_MU_KM3_S2
    return proxy.replace([np.inf, -np.inf], np.nan)


def prepare_run_frame(fof_outcomes: pd.DataFrame, bound_outcomes: pd.DataFrame | None) -> pd.DataFrame:
    frame = fof_outcomes.copy()
    frame["run_key"] = frame.get("filename", pd.Series(index=frame.index, dtype="object")).astype(str)
    frame["mass_log10_kg"] = mass_log10_kg(frame)
    frame["periapsis_Rm"] = periapsis_rm(frame)
    frame["v_inf_kms"] = velocity_kms(frame)
    frame["eccentricity_proxy"] = eccentricity_proxy(frame["periapsis_Rm"], frame["v_inf_kms"])

    frame["fragment_count_min_particles"] = pd.to_numeric(frame.get("fragment_count_min_particles"), errors="coerce").fillna(0)
    frame["is_fragmented_proxy"] = frame["fragment_count_min_particles"] > 1

    total_mass = pd.to_numeric(frame.get("total_particle_mass_kg"), errors="coerce")
    largest_mass = pd.to_numeric(frame.get("largest_fragment_mass_kg"), errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["dispersed_mass_fraction"] = 1.0 - (largest_mass / total_mass)
    frame["dispersed_mass_fraction"] = frame["dispersed_mass_fraction"].clip(lower=0.0, upper=1.0)
    frame["moderate_fragmentation"] = frame["dispersed_mass_fraction"] >= 0.1
    frame["strong_fragmentation"] = frame["dispersed_mass_fraction"] >= 0.4

    if bound_outcomes is not None and not bound_outcomes.empty:
        bound = bound_outcomes.copy()
        bound["run_key"] = bound.get("fof_file", pd.Series(index=bound.index, dtype="object")).astype(str)
        if "bound_mass_fraction" in bound.columns:
            bound["bound_mass_fraction_from_outcomes"] = pd.to_numeric(bound["bound_mass_fraction"], errors="coerce")
        keep_columns = [
            "run_key",
            "bound_fragment_count",
            "largest_bound_fragment_mass_kg",
            "unbound_mass_fraction",
            "bound_mass_fraction_from_outcomes",
        ]
        keep_columns = [column for column in keep_columns if column in bound.columns]
        frame = frame.merge(bound[keep_columns], on="run_key", how="left")
        frame["bound_mass_fraction"] = pd.to_numeric(frame.get("bound_mass_fraction_from_outcomes"), errors="coerce")
        frame["bound_fragment_count"] = pd.to_numeric(frame.get("bound_fragment_count"), errors="coerce")
        frame["bmf_ge_0p1"] = frame["bound_mass_fraction"] >= 0.1
    else:
        frame["bmf_ge_0p1"] = False
    return frame


def normalize_fragment_orbits(fragment_orbits: pd.DataFrame) -> pd.DataFrame:
    frame = fragment_orbits.copy()
    if "eccentricity" in frame.columns:
        frame["eccentricity"] = pd.to_numeric(frame["eccentricity"], errors="coerce")
    elif "eccentricity_e" in frame.columns:
        frame["eccentricity"] = pd.to_numeric(frame["eccentricity_e"], errors="coerce")
    else:
        raise ValueError("No eccentricity column found in fragment orbital catalog.")

    if "fof_file" in frame.columns:
        frame["run_key"] = frame["fof_file"].astype(str)
    elif "filename" in frame.columns:
        frame["run_key"] = frame["filename"].astype(str)
    elif "sim_id" in frame.columns:
        frame["run_key"] = frame["sim_id"].astype(str)
    else:
        frame["run_key"] = pd.Series([f"row_{idx}" for idx in frame.index], index=frame.index)

    if "fragment_mass_kg" in frame.columns:
        frame["fragment_mass_kg"] = pd.to_numeric(frame["fragment_mass_kg"], errors="coerce")
    else:
        frame["fragment_mass_kg"] = np.nan

    if "is_bound" in frame.columns:
        frame["is_bound"] = frame["is_bound"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
    else:
        frame["is_bound"] = frame["eccentricity"] < 1.0
    return frame


def plot_bound_mass(run_frame: pd.DataFrame, plots_dir: Path) -> None:
    if "bound_mass_fraction" not in run_frame.columns or run_frame["bound_mass_fraction"].dropna().empty:
        return
    frame = run_frame.dropna(subset=["eccentricity_proxy", "bound_mass_fraction"]).copy()
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.scatter(frame["eccentricity_proxy"], frame["bound_mass_fraction"], alpha=0.75, color="#2c7fb8", edgecolors="none")
    ax.set_xlabel("Encounter eccentricity")
    ax.set_ylabel("Mass fraction")
    ax.set_title("Mass retention versus encounter eccentricity")
    ymax = BOUND_FRACTION_YMAX_OVERRIDE
    if ymax is None:
        ymax = max(0.30, float(frame["bound_mass_fraction"].max()) * 1.08)
    ax.set_ylim(0.0, ymax)
    fig.tight_layout()
    fig.savefig(plots_dir / "figure2_used_in_report.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plots_dir = resolve_output_dir(args)

    fof_outcomes = load_csv(Path(args.fof_outcomes))
    bound_outcomes_path = Path(args.bound_outcomes)
    bound_outcomes = load_csv(bound_outcomes_path) if bound_outcomes_path.exists() else None

    run_frame = prepare_run_frame(fof_outcomes, bound_outcomes)

    fragment_orbits_path = Path(args.fragment_orbits)
    fragment_frame = None
    if fragment_orbits_path.exists():
        fragment_frame = normalize_fragment_orbits(load_csv(fragment_orbits_path))

    plot_bound_mass(run_frame, plots_dir)


if __name__ == "__main__":
    main()
