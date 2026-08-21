#!/usr/bin/env python3
"""EDA for encounter eccentricity versus fragmentation outcomes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

MPLCONFIGDIR = Path(".tmp/matplotlib")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR.resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MARS_MU_KM3_S2 = 4.282837e4
MARS_RADIUS_KM = 3389.5


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
    parser.add_argument("--eda-dir", default="eda", help="Output directory for EDA artifacts")
    return parser.parse_args()


def ensure_dirs(base_dir: Path) -> tuple[Path, Path]:
    tables_dir = base_dir / "tables"
    plots_dir = base_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, plots_dir


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
        keep_columns = [
            "run_key",
            "bound_mass_fraction",
            "bound_fragment_count",
            "largest_bound_fragment_mass_kg",
            "unbound_mass_fraction",
        ]
        keep_columns = [column for column in keep_columns if column in bound.columns]
        frame = frame.merge(bound[keep_columns], on="run_key", how="left")
        frame["bound_mass_fraction"] = pd.to_numeric(frame.get("bound_mass_fraction"), errors="coerce")
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


def write_dataset_overview(run_frame: pd.DataFrame, fragment_frame: pd.DataFrame | None, tables_dir: Path) -> pd.DataFrame:
    overview = pd.DataFrame(
        [
            {
                "run_rows": int(len(run_frame)),
                "fragment_orbit_rows": int(len(fragment_frame)) if fragment_frame is not None else 0,
                "eccentricity_proxy_min": float(run_frame["eccentricity_proxy"].min()),
                "eccentricity_proxy_median": float(run_frame["eccentricity_proxy"].median()),
                "eccentricity_proxy_max": float(run_frame["eccentricity_proxy"].max()),
                "fragmented_share": float(run_frame["is_fragmented_proxy"].mean()),
                "moderate_fragmentation_share": float(run_frame["moderate_fragmentation"].mean()),
                "strong_fragmentation_share": float(run_frame["strong_fragmentation"].mean()),
                "bmf_ge_0p1_share": float(run_frame["bmf_ge_0p1"].mean()),
            }
        ]
    )
    overview.to_csv(tables_dir / "dataset_overview.csv", index=False)
    return overview


def write_fragmentation_summary(run_frame: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, mask in [
        ("all_runs", pd.Series(True, index=run_frame.index)),
        ("fragmented_proxy", run_frame["is_fragmented_proxy"]),
        ("moderate_fragmentation", run_frame["moderate_fragmentation"]),
        ("strong_fragmentation", run_frame["strong_fragmentation"]),
        ("bmf_ge_0p1", run_frame["bmf_ge_0p1"]),
    ]:
        subset = run_frame.loc[mask, "eccentricity_proxy"].dropna()
        if subset.empty:
            continue
        rows.append(
            {
                "subset": label,
                "count": int(len(subset)),
                "min_eccentricity_proxy": float(subset.min()),
                "p25_eccentricity_proxy": float(subset.quantile(0.25)),
                "median_eccentricity_proxy": float(subset.median()),
                "mean_eccentricity_proxy": float(subset.mean()),
                "p75_eccentricity_proxy": float(subset.quantile(0.75)),
                "max_eccentricity_proxy": float(subset.max()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(tables_dir / "fragmentation_summary_by_eccentricity.csv", index=False)
    return summary


def write_threshold_scan(run_frame: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    eccentricity_values = run_frame["eccentricity_proxy"].dropna()
    thresholds = np.unique(np.round(np.linspace(eccentricity_values.min(), eccentricity_values.max(), 20), 3))
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        subset = run_frame[run_frame["eccentricity_proxy"] <= threshold]
        if subset.empty:
            continue
        rows.append(
            {
                "eccentricity_threshold": float(threshold),
                "run_count": int(len(subset)),
                "fragmented_share": float(subset["is_fragmented_proxy"].mean()),
                "moderate_fragmentation_share": float(subset["moderate_fragmentation"].mean()),
                "strong_fragmentation_share": float(subset["strong_fragmentation"].mean()),
                "bmf_ge_0p1_share": float(subset["bmf_ge_0p1"].mean()),
            }
        )
    threshold_df = pd.DataFrame(rows)
    threshold_df.to_csv(tables_dir / "eccentricity_threshold_scan.csv", index=False)
    return threshold_df


def write_low_eccentricity_cases(run_frame: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    columns = [
        "run_key",
        "mass_log10_kg",
        "periapsis_Rm",
        "v_inf_kms",
        "eccentricity_proxy",
        "fragment_count_min_particles",
        "dispersed_mass_fraction",
        "bound_mass_fraction",
    ]
    available_columns = [column for column in columns if column in run_frame.columns]
    low_e = (
        run_frame[run_frame["strong_fragmentation"]]
        .sort_values(["eccentricity_proxy", "dispersed_mass_fraction"], ascending=[True, False])
        .loc[:, available_columns]
        .head(20)
    )
    low_e.to_csv(tables_dir / "lowest_eccentricity_strong_fragmentation_cases.csv", index=False)
    return low_e


def write_fragment_orbit_summary(fragment_frame: pd.DataFrame | None, tables_dir: Path) -> pd.DataFrame | None:
    if fragment_frame is None or fragment_frame.empty:
        return None
    rows = []
    for label, mask in [
        ("all_fragments", pd.Series(True, index=fragment_frame.index)),
        ("bound_fragments", fragment_frame["is_bound"]),
        ("unbound_fragments", ~fragment_frame["is_bound"]),
        ("low_e_bound_fragments_e_lt_0p2", fragment_frame["is_bound"] & (fragment_frame["eccentricity"] < 0.2)),
    ]:
        subset = fragment_frame.loc[mask, "eccentricity"].dropna()
        if subset.empty:
            continue
        rows.append(
            {
                "subset": label,
                "count": int(len(subset)),
                "min_eccentricity": float(subset.min()),
                "median_eccentricity": float(subset.median()),
                "mean_eccentricity": float(subset.mean()),
                "max_eccentricity": float(subset.max()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(tables_dir / "fragment_orbit_eccentricity_summary.csv", index=False)
    return summary


def plot_scatter(run_frame: pd.DataFrame, plots_dir: Path) -> None:
    frame = run_frame.dropna(subset=["eccentricity_proxy", "dispersed_mass_fraction"]).copy()
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    scatter = ax.scatter(
        frame["eccentricity_proxy"],
        frame["dispersed_mass_fraction"],
        c=frame["fragment_count_min_particles"],
        cmap="inferno",
        alpha=0.8,
        edgecolors="none",
    )
    ax.set_xlabel("Encounter eccentricity proxy")
    ax.set_ylabel("Dispersed mass fraction")
    ax.set_title("Fragmentation severity versus encounter eccentricity proxy")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Fragment count (min-particles proxy)")
    fig.tight_layout()
    fig.savefig(plots_dir / "eccentricity_vs_dispersed_mass_fraction.png", dpi=180)
    plt.close(fig)


def plot_box(run_frame: pd.DataFrame, plots_dir: Path) -> None:
    groups = [
        ("non_fragmented", run_frame.loc[~run_frame["is_fragmented_proxy"], "eccentricity_proxy"].dropna()),
        ("fragmented", run_frame.loc[run_frame["is_fragmented_proxy"], "eccentricity_proxy"].dropna()),
        ("strong_fragmentation", run_frame.loc[run_frame["strong_fragmentation"], "eccentricity_proxy"].dropna()),
    ]
    groups = [(label, values) for label, values in groups if not values.empty]
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.boxplot([values.to_numpy() for _, values in groups], tick_labels=[label for label, _ in groups], patch_artist=True)
    ax.set_ylabel("Encounter eccentricity proxy")
    ax.set_title("Eccentricity proxy distribution by fragmentation class")
    fig.tight_layout()
    fig.savefig(plots_dir / "eccentricity_boxplot_by_fragmentation_class.png", dpi=180)
    plt.close(fig)


def plot_threshold_scan(threshold_df: pd.DataFrame, plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(threshold_df["eccentricity_threshold"], threshold_df["fragmented_share"], label="Any fragmentation", linewidth=2)
    ax.plot(threshold_df["eccentricity_threshold"], threshold_df["moderate_fragmentation_share"], label="Moderate fragmentation", linewidth=2)
    ax.plot(threshold_df["eccentricity_threshold"], threshold_df["strong_fragmentation_share"], label="Strong fragmentation", linewidth=2)
    ax.plot(threshold_df["eccentricity_threshold"], threshold_df["bmf_ge_0p1_share"], label="BMF >= 10%", linewidth=2)
    ax.set_xlabel("Upper eccentricity threshold")
    ax.set_ylabel("Share of runs below threshold")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("How disruption prevalence changes across low-eccentricity regimes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "eccentricity_threshold_scan.png", dpi=180)
    plt.close(fig)


def plot_bound_mass(run_frame: pd.DataFrame, plots_dir: Path) -> None:
    if "bound_mass_fraction" not in run_frame.columns or run_frame["bound_mass_fraction"].dropna().empty:
        return
    frame = run_frame.dropna(subset=["eccentricity_proxy", "bound_mass_fraction"]).copy()
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.scatter(frame["eccentricity_proxy"], frame["bound_mass_fraction"], alpha=0.75, color="#2c7fb8", edgecolors="none")
    ax.set_xlabel("Encounter eccentricity proxy")
    ax.set_ylabel("Bound mass fraction")
    ax.set_title("Bound retention versus encounter eccentricity proxy")
    fig.tight_layout()
    fig.savefig(plots_dir / "eccentricity_vs_bound_mass_fraction.png", dpi=180)
    plt.close(fig)


def write_summary(run_frame: pd.DataFrame, fragment_frame: pd.DataFrame | None, base_dir: Path) -> None:
    def safe_min(mask: pd.Series) -> float | None:
        subset = run_frame.loc[mask, "eccentricity_proxy"].dropna()
        return float(subset.min()) if not subset.empty else None

    any_frag_min = safe_min(run_frame["is_fragmented_proxy"])
    strong_frag_min = safe_min(run_frame["strong_fragmentation"])
    bmf_min = safe_min(run_frame["bmf_ge_0p1"])

    lines = [
        "Eccentricity EDA summary",
        "",
        "Important interpretation:",
        "This script treats eccentricity in two different ways.",
        "1. The main analysis uses an encounter eccentricity proxy derived from periapsis and v_inf. This is the pre-impact orbital regime and is the correct quantity for asking how eccentricity relates to disruption.",
        "2. If a fragment orbital catalog exists, the script also summarises post-encounter fragment eccentricities. Those are outcomes, not the input driver.",
        "",
        f"Lowest encounter eccentricity proxy with any fragmentation proxy: {any_frag_min:.3f}" if any_frag_min is not None else "No fragmented runs found.",
        f"Lowest encounter eccentricity proxy with strong fragmentation: {strong_frag_min:.3f}" if strong_frag_min is not None else "No strong-fragmentation runs found.",
        f"Lowest encounter eccentricity proxy with bound_mass_fraction >= 0.1: {bmf_min:.3f}" if bmf_min is not None else "No BMF >= 0.1 runs found.",
        "",
        "Caveat:",
        "In the current dataset, the minimum observed encounter eccentricity proxy is already the parabolic edge (about 1.0), so this EDA can identify the lowest sampled regime that disrupts, but it cannot prove a sharper threshold below the sampled minimum.",
    ]
    if fragment_frame is None:
        lines.extend(
            [
                "",
                "Fragment-orbit note:",
                "No fragment orbital catalog was available at the default path, so post-encounter eccentricity summaries were skipped.",
            ]
        )
    (base_dir / "analysis_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(base_dir: Path) -> None:
    content = """# Eccentricity EDA

This folder contains plots and tables for how encounter eccentricity relates to fragmentation proxies.

Outputs:

- `tables/` contains threshold scans, summary statistics, and low-eccentricity edge cases.
- `plots/` contains the main eccentricity-versus-fragmentation visualisations.
- `analysis_summary.txt` gives a concise interpretation.

## Re-run

```bash
python eda/scripts/eda_eccentricity.py \
  --fof-outcomes outputs/fof_outcomes.csv \
  --bound-outcomes outputs/bound_outcomes.csv \
  --fragment-orbits outputs/fragment_orbital_catalog.csv \
  --eda-dir eda/eccentricity_eda
```
"""
    (base_dir / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_dir = Path(args.eda_dir)
    tables_dir, plots_dir = ensure_dirs(base_dir)

    fof_outcomes = load_csv(Path(args.fof_outcomes))
    bound_outcomes_path = Path(args.bound_outcomes)
    bound_outcomes = load_csv(bound_outcomes_path) if bound_outcomes_path.exists() else None

    run_frame = prepare_run_frame(fof_outcomes, bound_outcomes)

    fragment_orbits_path = Path(args.fragment_orbits)
    fragment_frame = None
    if fragment_orbits_path.exists():
        fragment_frame = normalize_fragment_orbits(load_csv(fragment_orbits_path))

    write_dataset_overview(run_frame, fragment_frame, tables_dir)
    threshold_df = write_threshold_scan(run_frame, tables_dir)
    write_fragmentation_summary(run_frame, tables_dir)
    write_low_eccentricity_cases(run_frame, tables_dir)
    write_fragment_orbit_summary(fragment_frame, tables_dir)

    plot_scatter(run_frame, plots_dir)
    plot_box(run_frame, plots_dir)
    plot_threshold_scan(threshold_df, plots_dir)
    plot_bound_mass(run_frame, plots_dir)

    write_summary(run_frame, fragment_frame, base_dir)
    write_readme(base_dir)


if __name__ == "__main__":
    main()
