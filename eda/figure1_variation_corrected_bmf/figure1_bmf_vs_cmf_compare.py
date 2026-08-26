#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = ROOT / "extraction-outputs_corrected_bmf" / "tables" / "bound_outcomes.csv"
OUTPUT_PATH = ROOT / "report-table-figure" / "figures_corrected_bmf" / "figure1_bmf_vs_cmf_compare.png"

PERI_RANGE = (1.1, 3.0)
PERI_TICKS = [1.1, 1.3, 1.5, 1.7, 1.9, 2.2, 2.6, 3.0]
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]
VELOCITY_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
VELOCITY_COLORS = {
    0.0: "#1f77b4",
    0.2: "#2ca02c",
    0.4: "#ffbf00",
    0.6: "#ff7f0e",
    0.8: "#ff4d4d",
    1.0: "#c266ff",
    1.2: "#7f7f7f",
    1.4: "#bcbd22",
    1.6: "#17becf",
}
SPIN_MARKERS = {
    "no_spin": "o",
    "equatorial": "s",
    "prograde_z": "^",
    "retrograde_z": "D",
}
SPIN_LINESTYLES = {
    "no_spin": "solid",
    "equatorial": (0, (6, 2)),
    "prograde_z": (0, (2, 2)),
    "retrograde_z": (0, (9, 3, 2, 3)),
}
SPIN_LABELS = {
    "no_spin": "No spin",
    "equatorial": "Equatorial spin",
    "prograde_z": "Prograde z-spin",
    "retrograde_z": "Retrograde z-spin",
}


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
    for column in ["bound_mass_fraction", "captured_mass_fraction", "fof_linking_length"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#e6e6e6", linewidth=0.8, alpha=0.8)
    ax.set_facecolor("white")


def add_grouped_legends(ax: plt.Axes) -> None:
    velocity_handles = [
        Line2D([0], [0], color=VELOCITY_COLORS[v], linewidth=2.0, label=f"{v:g}")
        for v in VELOCITY_VALUES
        if v in VELOCITY_COLORS
    ]
    spin_handles = [
        Line2D(
            [0],
            [0],
            color="#666666",
            linestyle=SPIN_LINESTYLES[name],
            marker=SPIN_MARKERS[name],
            markerfacecolor="white",
            markeredgecolor="#666666",
            markersize=6,
            linewidth=1.6,
            label=SPIN_LABELS[name],
        )
        for name in ["no_spin", "equatorial", "prograde_z", "retrograde_z"]
    ]
    leg1 = ax.legend(
        handles=velocity_handles,
        loc="upper right",
        frameon=True,
        title="v∞ (km s$^{-1}$)",
        title_fontsize=9,
        fontsize=8,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=spin_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.40),
        frameon=True,
        title="spin",
        title_fontsize=9,
        fontsize=8,
    )


def add_exact_legends(ax: plt.Axes, velocities: list[float]) -> None:
    velocity_handles = [
        Line2D([0], [0], color=VELOCITY_COLORS[v], linewidth=2.0, marker="o", markersize=4.5, label=f"{v:g}")
        for v in velocities
    ]
    leg1 = ax.legend(
        handles=velocity_handles,
        loc="upper right",
        frameon=True,
        title="v∞ (km s$^{-1}$)",
        title_fontsize=9,
        fontsize=8,
    )
    ax.add_artist(leg1)
    metric_handles = [
        Line2D([0], [0], color="#222222", linewidth=1.8, marker="o", markersize=4.5, label="BMF"),
        Line2D([0], [0], color="#222222", linewidth=1.8, linestyle="--", marker="s", markersize=4.2, label="CMF"),
    ]
    ax.legend(
        handles=metric_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.40),
        frameon=True,
        title="metric",
        title_fontsize=9,
        fontsize=8,
    )


def grouped_metric_panel(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    panel = frame.loc[
        frame["periapsis_Rm"].between(*PERI_RANGE, inclusive="both")
        & frame[metric].notna()
        & frame["v_inf_kms"].notna()
    ].copy()
    return (
        panel.groupby(["periapsis_Rm", "v_inf_kms", "spin_orientation"], as_index=False)
        .agg(metric_median=(metric, "median"))
        .sort_values(["v_inf_kms", "spin_orientation", "periapsis_Rm"])
    )


def draw_grouped_panel(ax: plt.Axes, grouped: pd.DataFrame, title: str, y_max: float) -> None:
    for (velocity, spin_name), subset in grouped.groupby(["v_inf_kms", "spin_orientation"], sort=True):
        color = VELOCITY_COLORS.get(float(velocity), "#333333")
        marker = SPIN_MARKERS.get(spin_name, "o")
        linestyle = SPIN_LINESTYLES.get(spin_name, "solid")
        if len(subset) > 1:
            ax.plot(
                subset["periapsis_Rm"],
                subset["metric_median"],
                color=color,
                linewidth=1.6,
                linestyle=linestyle,
                alpha=0.9,
                zorder=2,
            )
        ax.scatter(
            subset["periapsis_Rm"],
            subset["metric_median"],
            color=[color],
            marker=marker,
            s=30,
            linewidths=0.45,
            edgecolors="white",
            alpha=0.95,
            zorder=3,
        )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(r"Periapsis ($R_{\mathrm{Mars}}$)", fontsize=11)
    ax.set_ylabel("Mass Fraction")
    ax.set_xlim(*PERI_RANGE)
    ax.set_xticks(PERI_TICKS)
    ax.set_ylim(0.0, y_max)
    style_axes(ax)
    add_grouped_legends(ax)


def exact_subset(frame: pd.DataFrame, mass_code: str, *, resolution_code: str, fof_linking_length: float) -> pd.DataFrame:
    return frame.loc[
        (frame["mass_code"] == mass_code)
        & (frame["resolution_code"] == resolution_code)
        & np.isclose(frame["fof_linking_length"], fof_linking_length)
        & (frame["spin_orientation"] == "no_spin")
    ].copy()


def draw_exact_compare_panel(ax: plt.Axes, subset: pd.DataFrame, title: str, y_max: float) -> None:
    subset = subset.sort_values(["v_inf_kms", "periapsis_Rm"])
    velocities = sorted(float(v) for v in subset["v_inf_kms"].dropna().unique())
    for velocity, rows in subset.groupby("v_inf_kms", sort=True):
        color = VELOCITY_COLORS.get(float(velocity), "#333333")
        ax.plot(
            rows["periapsis_Rm"],
            rows["bound_mass_fraction"],
            color=color,
            linewidth=1.9,
            marker="o",
            markersize=4.6,
        )
        ax.plot(
            rows["periapsis_Rm"],
            rows["captured_mass_fraction"],
            color=color,
            linewidth=1.8,
            linestyle="--",
            marker="s",
            markersize=4.0,
        )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(r"Periapsis ($R_{\mathrm{Mars}}$)", fontsize=11)
    ax.set_ylabel("Mass Fraction")
    ax.set_xlim(*PERI_RANGE)
    ax.set_xticks(PERI_TICKS)
    ax.set_ylim(0.0, y_max)
    style_axes(ax)
    add_exact_legends(ax, velocities)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    grouped_bmf = grouped_metric_panel(frame, "bound_mass_fraction")
    grouped_cmf = grouped_metric_panel(frame, "captured_mass_fraction")
    exact_mass19 = exact_subset(frame, "A1900", resolution_code="n65", fof_linking_length=0.0020)
    exact_mass20 = exact_subset(frame, "A2000", resolution_code="n65", fof_linking_length=0.0040)
    y_max = max(
        0.55,
        float(grouped_bmf["metric_median"].max()) * 1.08 if not grouped_bmf.empty else 0.0,
        float(grouped_cmf["metric_median"].max()) * 1.08 if not grouped_cmf.empty else 0.0,
        float(exact_mass19["bound_mass_fraction"].max()) * 1.08 if not exact_mass19.empty else 0.0,
        float(exact_mass20["bound_mass_fraction"].max()) * 1.08 if not exact_mass20.empty else 0.0,
    )
    y_max = min(0.65, y_max)

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.2), dpi=220)
    draw_grouped_panel(axes[0, 0], grouped_bmf, "BMF overview", y_max)
    draw_grouped_panel(axes[0, 1], grouped_cmf, "CMF overview", y_max)
    draw_exact_compare_panel(axes[1, 0], exact_mass19, r"BMF vs CMF ($10^{19}$ kg)", y_max)
    draw_exact_compare_panel(axes[1, 1], exact_mass20, r"BMF vs CMF ($10^{20}$ kg)", y_max)

    for ax, panel_label in zip(axes.flat, PANEL_LABELS):
        ax.text(0.0, 1.03, panel_label, transform=ax.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold")

    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.98), h_pad=2.5, w_pad=2.1)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
