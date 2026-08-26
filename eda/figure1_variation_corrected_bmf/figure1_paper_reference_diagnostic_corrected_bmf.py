#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = ROOT / "extraction-outputs_corrected_bmf" / "tables" / "bound_outcomes.csv"
OUTPUT_DIR = ROOT / "report-table-figure" / "figures_corrected_bmf"

LINEAR_PLOT = OUTPUT_DIR / "figure1_paper_reference_strict_linear_corrected_bmf.png"
LOG_PLOT = OUTPUT_DIR / "figure1_paper_reference_strict_log_corrected_bmf.png"
COMPARISON_CSV = OUTPUT_DIR / "figure1_paper_reference_strict_comparison_corrected_bmf.csv"
MIXING_AUDIT_CSV = OUTPUT_DIR / "figure1_current_mass20_panel_mixing_audit_corrected_bmf.csv"

VELOCITY_COLORS = {
    0.0: "#1f77b4",
    0.2: "#2ca02c",
    0.4: "#ffbf00",
    0.6: "#ff7f0e",
}

PAPER_TABLE2_ROWS = [
    {"periapsis_Rm": 1.1, "v_inf_kms": 0.0, "paper_f_bnd": 0.526, "paper_f_capt": 0.479},
    {"periapsis_Rm": 1.2, "v_inf_kms": 0.0, "paper_f_bnd": 0.516, "paper_f_capt": 0.444},
    {"periapsis_Rm": 1.3, "v_inf_kms": 0.0, "paper_f_bnd": 0.516, "paper_f_capt": 0.424},
    {"periapsis_Rm": 1.4, "v_inf_kms": 0.0, "paper_f_bnd": 0.513, "paper_f_capt": 0.442},
    {"periapsis_Rm": 1.5, "v_inf_kms": 0.0, "paper_f_bnd": 0.434, "paper_f_capt": 0.428},
    {"periapsis_Rm": 1.6, "v_inf_kms": 0.0, "paper_f_bnd": 0.466, "paper_f_capt": 0.368},
    {"periapsis_Rm": 1.7, "v_inf_kms": 0.0, "paper_f_bnd": 0.388, "paper_f_capt": 0.336},
    {"periapsis_Rm": 1.8, "v_inf_kms": 0.0, "paper_f_bnd": 0.238, "paper_f_capt": 0.235},
    {"periapsis_Rm": 1.9, "v_inf_kms": 0.0, "paper_f_bnd": 0.181, "paper_f_capt": 0.181},
    {"periapsis_Rm": 2.0, "v_inf_kms": 0.0, "paper_f_bnd": 0.000, "paper_f_capt": 0.000},
    {"periapsis_Rm": 2.2, "v_inf_kms": 0.0, "paper_f_bnd": 0.000, "paper_f_capt": 0.000},
    {"periapsis_Rm": 2.4, "v_inf_kms": 0.0, "paper_f_bnd": 0.000, "paper_f_capt": 0.000},
    {"periapsis_Rm": 1.2, "v_inf_kms": 0.2, "paper_f_bnd": 0.493, "paper_f_capt": 0.441},
    {"periapsis_Rm": 1.4, "v_inf_kms": 0.2, "paper_f_bnd": 0.452, "paper_f_capt": 0.405},
    {"periapsis_Rm": 1.6, "v_inf_kms": 0.2, "paper_f_bnd": 0.392, "paper_f_capt": 0.352},
    {"periapsis_Rm": 1.8, "v_inf_kms": 0.2, "paper_f_bnd": 0.237, "paper_f_capt": 0.188},
    {"periapsis_Rm": 2.0, "v_inf_kms": 0.2, "paper_f_bnd": 0.121, "paper_f_capt": 0.000},
    {"periapsis_Rm": 1.2, "v_inf_kms": 0.4, "paper_f_bnd": 0.426, "paper_f_capt": 0.344},
    {"periapsis_Rm": 1.4, "v_inf_kms": 0.4, "paper_f_bnd": 0.380, "paper_f_capt": 0.330},
    {"periapsis_Rm": 1.6, "v_inf_kms": 0.4, "paper_f_bnd": 0.334, "paper_f_capt": 0.298},
    {"periapsis_Rm": 1.8, "v_inf_kms": 0.4, "paper_f_bnd": 0.183, "paper_f_capt": 0.168},
    {"periapsis_Rm": 2.0, "v_inf_kms": 0.4, "paper_f_bnd": 0.000, "paper_f_capt": 0.000},
    {"periapsis_Rm": 1.2, "v_inf_kms": 0.6, "paper_f_bnd": 0.395, "paper_f_capt": 0.278},
    {"periapsis_Rm": 1.4, "v_inf_kms": 0.6, "paper_f_bnd": 0.321, "paper_f_capt": 0.243},
    {"periapsis_Rm": 1.6, "v_inf_kms": 0.6, "paper_f_bnd": 0.253, "paper_f_capt": 0.213},
    {"periapsis_Rm": 1.8, "v_inf_kms": 0.6, "paper_f_bnd": 0.112, "paper_f_capt": 0.087},
    {"periapsis_Rm": 2.0, "v_inf_kms": 0.6, "paper_f_bnd": 0.000, "paper_f_capt": 0.000},
]


def parse_numeric_code(series: pd.Series, pattern: str, scale: float = 1.0) -> pd.Series:
    extracted = series.fillna("").astype(str).str.extract(pattern)[0]
    return pd.to_numeric(extracted, errors="coerce") / scale


def load_frame() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE_PATH, low_memory=False)
    frame["periapsis_Rm"] = parse_numeric_code(frame["periapsis_code"], r"r(\d+)", 10.0)
    frame["v_inf_kms"] = parse_numeric_code(frame["velocity_code"], r"v(\d+)", 10.0)
    spin_code = frame["spin_code"].fillna("").astype(str)
    frame["spin_orientation"] = "no_spin"
    frame.loc[spin_code.str.contains("mz"), "spin_orientation"] = "retrograde_z"
    frame.loc[spin_code.str.contains("x") | spin_code.str.contains("y"), "spin_orientation"] = "equatorial"
    frame.loc[spin_code.str.contains("z") & ~spin_code.str.contains("mz"), "spin_orientation"] = "prograde_z"
    frame["captured_mass_fraction"] = pd.to_numeric(frame["captured_mass_fraction"], errors="coerce")
    frame["bound_mass_fraction"] = pd.to_numeric(frame["bound_mass_fraction"], errors="coerce")
    frame["fof_linking_length"] = pd.to_numeric(frame["fof_linking_length"], errors="coerce")
    return frame


def build_mixing_audit(frame: pd.DataFrame) -> pd.DataFrame:
    panel = frame.loc[frame["mass_code"] == "A2000"].copy()
    return (
        panel.groupby(["periapsis_Rm", "v_inf_kms", "spin_orientation"], as_index=False)
        .agg(
            rows=("fof_file", "size"),
            physical_files=("physical_file", "nunique"),
            resolutions=("resolution_code", "nunique"),
            fof_lengths=("fof_linking_length", "nunique"),
            spin_codes=("spin_code", "nunique"),
        )
        .sort_values(["v_inf_kms", "spin_orientation", "periapsis_Rm"])
    )


def build_strict_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    paper = pd.DataFrame(PAPER_TABLE2_ROWS)
    no_spin = frame["spin_orientation"] == "no_spin"
    strict = frame.loc[
        (frame["mass_code"] == "A2000")
        & (frame["resolution_code"] == "n65")
        & np.isclose(frame["fof_linking_length"], 0.0040)
        & no_spin
    ].copy()
    strict = strict[
        [
            "periapsis_Rm",
            "v_inf_kms",
            "captured_mass_fraction",
            "bound_mass_fraction",
            "physical_file",
            "fof_file",
        ]
    ]
    merged = paper.merge(strict, on=["periapsis_Rm", "v_inf_kms"], how="left")
    merged["captured_minus_paper_f_capt"] = merged["captured_mass_fraction"] - merged["paper_f_capt"]
    merged["bound_minus_paper_f_bnd"] = merged["bound_mass_fraction"] - merged["paper_f_bnd"]
    return merged.sort_values(["v_inf_kms", "periapsis_Rm"])


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#e6e6e6", linewidth=0.8, alpha=0.8)
    ax.set_facecolor("white")


def plot_comparison(comparison: pd.DataFrame, output_path: Path, *, log_scale: bool) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=220)
    floor = 1e-4
    for velocity, subset in comparison.groupby("v_inf_kms", sort=True):
        color = VELOCITY_COLORS.get(float(velocity), "#333333")
        subset = subset.sort_values("periapsis_Rm")
        archive_values = subset["captured_mass_fraction"].clip(lower=floor) if log_scale else subset["captured_mass_fraction"]
        paper_values = subset["paper_f_capt"].clip(lower=floor) if log_scale else subset["paper_f_capt"]
        ax.plot(
            subset["periapsis_Rm"],
            paper_values,
            color=color,
            linestyle="--",
            linewidth=1.8,
            marker="o",
            markersize=4.0,
            label=f"Paper, v∞={velocity:g}",
        )
        ax.plot(
            subset["periapsis_Rm"],
            archive_values,
            color=color,
            linestyle="-",
            linewidth=1.8,
            marker="s",
            markersize=3.8,
            label=f"Archive, v∞={velocity:g}",
        )
    ax.set_xlabel(r"Periapsis ($R_{\mathrm{Mars}}$)")
    ax.set_ylabel("Captured mass fraction")
    ax.set_title(r"Strict $10^{20}$ kg, no-spin, n65, FoF = 0.004 comparison")
    ax.set_xlim(1.1, 2.4)
    ax.set_xticks([1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4])
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(floor, 0.7)
        ax.text(
            0.02,
            0.02,
            "Zeros shown at 10^-4 for display on log scale only.",
            transform=ax.transAxes,
            fontsize=8,
            ha="left",
            va="bottom",
        )
    else:
        ax.set_ylim(0.0, 0.55)
    style_axes(ax)
    ax.legend(loc="upper right", fontsize=8, frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    mixing_audit = build_mixing_audit(frame)
    mixing_audit.to_csv(MIXING_AUDIT_CSV, index=False)
    comparison = build_strict_comparison(frame)
    comparison.to_csv(COMPARISON_CSV, index=False)
    plot_comparison(comparison, LINEAR_PLOT, log_scale=False)
    plot_comparison(comparison, LOG_PLOT, log_scale=True)
    print(LINEAR_PLOT)
    print(LOG_PLOT)
    print(COMPARISON_CSV)
    print(MIXING_AUDIT_CSV)


if __name__ == "__main__":
    main()
