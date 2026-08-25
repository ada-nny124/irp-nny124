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
SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "extraction-outputs" / "tables" / "bound_outcomes.csv"
OUTPUT_PATH = SCRIPT_DIR / "figure1_used_in_report.png"

PERI_RANGE = (1.1, 3.0)
PERI_TICKS = [1.1, 1.3, 1.5, 1.7, 1.9, 2.2, 2.6, 3.0]
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
VELOCITY_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 2.0]
VELOCITY_COLORS = {
    0.0: "#1f77b4",
    0.2: "#2ca02c",
    0.4: "#ffbf00",
    0.6: "#ff7f0e",
    0.8: "#ff4d4d",
    1.0: "#c266ff",
    1.2: "#7f7f7f",
    1.4: "#bcbd22",
    1.5: "#9c755f",
    1.6: "#17becf",
    2.0: "#111111",
}
MASS_LABELS = {
    "A1900": r"10^{19}",
    "A2000": r"10^{20}",
}
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]


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
    return frame


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#e6e6e6", linewidth=0.8, alpha=0.8)
    ax.set_facecolor("white")


def draw_panel_a(ax: plt.Axes, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = frame.loc[
        frame["periapsis_Rm"].between(*PERI_RANGE, inclusive="both")
        & frame["n_fragments"].notna()
        & frame["v_inf_kms"].notna()
    ].copy()
    panel["n_fragments"] = pd.to_numeric(panel["n_fragments"], errors="coerce")
    panel = panel.dropna(subset=["n_fragments", "periapsis_Rm", "v_inf_kms"])

    ax.scatter(
        panel["periapsis_Rm"],
        panel["n_fragments"],
        color="#f28e2b",
        marker="o",
        s=26,
        alpha=0.75,
        linewidths=0.25,
        edgecolors="white",
        zorder=2,
    )

    median_by_peri = (
        panel.groupby("periapsis_Rm", as_index=False)["n_fragments"]
        .median()
        .sort_values("periapsis_Rm")
        .rename(columns={"n_fragments": "median_fragment_count"})
    )
    ax.set_title("Fragment count vs periapsis", fontsize=12)
    ax.set_xlabel(r"Periapsis ($R_{\mathrm{Mars}}$)", fontsize=11)
    ax.set_ylabel("Number of FoF fragments")
    ax.set_xlim(*PERI_RANGE)
    ax.set_xticks(PERI_TICKS)
    ax.tick_params(axis="x", labelsize=9, pad=4)
    style_axes(ax)
    return panel, median_by_peri


def build_grouped_bmf_panel(frame: pd.DataFrame, mass_code: str | None = None) -> pd.DataFrame:
    panel = frame.loc[
        frame["periapsis_Rm"].between(*PERI_RANGE, inclusive="both")
        & frame["bound_mass_fraction"].notna()
        & frame["v_inf_kms"].notna()
    ].copy()
    if mass_code is not None:
        panel = panel.loc[panel["mass_code"] == mass_code].copy()
    panel["bound_mass_fraction"] = pd.to_numeric(panel["bound_mass_fraction"], errors="coerce")
    panel = panel.dropna(subset=["bound_mass_fraction", "periapsis_Rm", "v_inf_kms"])

    return (
        panel.groupby(["periapsis_Rm", "v_inf_kms", "spin_orientation"], as_index=False)
        .agg(
            bound_mass_fraction_median=("bound_mass_fraction", "median"),
            raw_row_count=("bound_mass_fraction", "size"),
        )
        .sort_values(["v_inf_kms", "spin_orientation", "periapsis_Rm"])
    )


def draw_panel_b(
    ax: plt.Axes,
    grouped: pd.DataFrame,
    title: str,
    *,
    y_max: float,
    show_legends: bool = False,
) -> None:
    for (velocity, spin_name), subset in grouped.groupby(["v_inf_kms", "spin_orientation"], sort=True):
        color = VELOCITY_COLORS[float(velocity)]
        marker = SPIN_MARKERS.get(spin_name, "o")
        linestyle = SPIN_LINESTYLES.get(spin_name, "solid")
        if len(subset) > 1:
            ax.plot(
                subset["periapsis_Rm"],
                subset["bound_mass_fraction_median"],
                color=color,
                linewidth=1.6,
                linestyle=linestyle,
                alpha=0.9,
                zorder=2,
            )
        ax.scatter(
            subset["periapsis_Rm"],
            subset["bound_mass_fraction_median"],
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
    ax.set_ylabel("Bound Mass Fraction")
    ax.set_xlim(*PERI_RANGE)
    ax.set_xticks(PERI_TICKS)
    ax.tick_params(axis="x", labelsize=9, pad=4)
    ax.set_ylim(0.0, y_max)
    style_axes(ax)
    if show_legends:
        add_legends(ax)


def add_legends(ax: plt.Axes) -> None:
    velocity_handles = [
        Line2D([0], [0], color=VELOCITY_COLORS[v], linewidth=2.0, label=f"{v:g}")
        for v in VELOCITY_VALUES
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
        bbox_to_anchor=(1.0, 0.38),
        frameon=True,
        title="spin",
        title_fontsize=9,
        fontsize=8,
    )


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    panel_b_all_groups = build_grouped_bmf_panel(frame)
    panel_b_mass19_groups = build_grouped_bmf_panel(frame, mass_code="A1900")
    panel_b_mass20_groups = build_grouped_bmf_panel(frame, mass_code="A2000")
    shared_bmf_ymax = max(
        0.32,
        panel_b_all_groups["bound_mass_fraction_median"].max() * 1.08,
        panel_b_mass19_groups["bound_mass_fraction_median"].max() * 1.08 if not panel_b_mass19_groups.empty else 0.0,
        panel_b_mass20_groups["bound_mass_fraction_median"].max() * 1.08 if not panel_b_mass20_groups.empty else 0.0,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.2), dpi=220)
    axes_flat = axes.flat
    first_ax = next(axes_flat)
    draw_panel_a(first_ax, frame)
    draw_panel_b(
        next(axes_flat),
        panel_b_all_groups,
        "Bound Mass Fraction vs Periapsis",
        y_max=shared_bmf_ymax,
        show_legends=True,
    )
    draw_panel_b(
        next(axes_flat),
        panel_b_mass19_groups,
        r"Bound Mass Fraction vs Periapsis ($10^{19}$ kg only)",
        y_max=shared_bmf_ymax,
        show_legends=True,
    )
    draw_panel_b(
        next(axes_flat),
        panel_b_mass20_groups,
        r"Bound Mass Fraction vs Periapsis ($10^{20}$ kg only)",
        y_max=shared_bmf_ymax,
        show_legends=True,
    )

    for ax, panel_label in zip(axes.flat, PANEL_LABELS):
        ax.text(0.0, 1.03, panel_label, transform=ax.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold")

    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.98), h_pad=2.4, w_pad=2.0)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
