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


ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = ROOT / "extraction_outputs" / "bound_outcomes.csv"
FIG_DIR = ROOT / "report" / "figures"
NOTES_PATH = ROOT / "report" / "figure1_regeneration_notes.md"

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


def draw_panel_b(ax: plt.Axes, frame: pd.DataFrame) -> pd.DataFrame:
    panel = frame.loc[
        frame["periapsis_Rm"].between(*PERI_RANGE, inclusive="both")
        & frame["bound_mass_fraction"].notna()
        & frame["v_inf_kms"].notna()
    ].copy()
    panel["bound_mass_fraction"] = pd.to_numeric(panel["bound_mass_fraction"], errors="coerce")
    panel = panel.dropna(subset=["bound_mass_fraction", "periapsis_Rm", "v_inf_kms"])

    grouped = (
        panel.groupby(["periapsis_Rm", "v_inf_kms", "spin_orientation"], as_index=False)
        .agg(
            bound_mass_fraction_median=("bound_mass_fraction", "median"),
            raw_row_count=("bound_mass_fraction", "size"),
        )
        .sort_values(["v_inf_kms", "spin_orientation", "periapsis_Rm"])
    )

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

    ax.set_title("Bound Mass Fraction vs Periapsis", fontsize=12)
    ax.set_xlabel(r"Periapsis ($R_{\mathrm{Mars}}$)", fontsize=11)
    ax.set_ylabel("Bound Mass Fraction")
    ax.set_xlim(*PERI_RANGE)
    ax.set_xticks(PERI_TICKS)
    ax.tick_params(axis="x", labelsize=9, pad=4)
    ax.set_ylim(0.0, max(0.32, grouped["bound_mass_fraction_median"].max() * 1.08))
    style_axes(ax)
    return grouped


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


def write_notes(panel_a_points: pd.DataFrame, panel_b_groups: pd.DataFrame) -> None:
    text = "\n".join(
        [
            "# Figure 1 Regeneration Notes",
            "",
            "## Source file",
            f"- `{SOURCE_PATH.relative_to(ROOT)}`",
            "",
            "## Derived source columns",
            "- `periapsis_Rm` from `periapsis_code` using `r11 -> 1.1`, ..., `r30 -> 3.0`",
            "- `v_inf_kms` from `velocity_code` using `v00 -> 0.0`, ..., `v20 -> 2.0`",
            "- `spin_orientation` from `spin_code` with categories `no_spin`, `equatorial`, `prograde_z`, `retrograde_z`",
            "",
            "## Panel 1a columns used",
            "- `periapsis_code -> periapsis_Rm`",
            "- `n_fragments`",
            "- `velocity_code -> v_inf_kms`",
            "- `spin_code -> spin_orientation`",
            "",
            "## Panel 1a filters",
            "- kept rows with non-null `periapsis_Rm`, `n_fragments`, and `v_inf_kms`",
            "- kept periapsis range `1.1 <= periapsis_Rm <= 3.0`",
            "- no additional mass / velocity / spin filtering; all available cases included",
            f"- plotted raw FoF rows: `{len(panel_a_points):,}`",
            "",
            "## Panel 1b columns used",
            "- `periapsis_code -> periapsis_Rm`",
            "- `bound_mass_fraction`",
            "- `velocity_code -> v_inf_kms`",
            "- `spin_code -> spin_orientation`",
            "",
            "## Panel 1b filters",
            "- kept rows with non-null `periapsis_Rm`, `bound_mass_fraction`, and `v_inf_kms`",
            "- kept periapsis range `1.1 <= periapsis_Rm <= 3.0`",
            "- no additional mass / velocity / spin filtering; all available cases included",
            "- grouped medians by `(periapsis_Rm, v_inf_kms, spin_orientation)` from raw FoF rows",
            f"- plotted grouped median cells: `{len(panel_b_groups):,}`",
            "",
            "## Interpretation note",
            "- Connecting and trend lines are visual guides across sampled periapsis bins only; they must not be interpreted as physical trajectories.",
        ]
    )
    NOTES_PATH.write_text(text + "\n", encoding="utf-8")


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in [".png", ".svg", ".pdf"]:
        fig.savefig(FIG_DIR / f"{stem}{suffix}", bbox_inches="tight")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_frame()

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), dpi=220)
    panel_a_points, panel_a_medians = draw_panel_a(axes[0], frame)
    panel_b_groups = draw_panel_b(axes[1], frame)
    add_legends(axes[1])
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.98))

    save_figure(fig, "figure1_periapsis_regenerated")
    plt.close(fig)

    panel_fig_a, panel_ax_a = plt.subplots(1, 1, figsize=(7.0, 5.6), dpi=220)
    draw_panel_a(panel_ax_a, frame)
    panel_fig_a.tight_layout()
    save_figure(panel_fig_a, "figure1a_fragment_count_vs_periapsis_regenerated")
    plt.close(panel_fig_a)

    panel_fig_b, panel_ax_b = plt.subplots(1, 1, figsize=(7.4, 5.6), dpi=220)
    draw_panel_b(panel_ax_b, frame)
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
    leg1 = panel_ax_b.legend(handles=velocity_handles, loc="upper right", frameon=True, title="v∞ (km s$^{-1}$)")
    panel_ax_b.add_artist(leg1)
    panel_ax_b.legend(handles=spin_handles, loc="upper right", bbox_to_anchor=(1.0, 0.38), frameon=True, title="spin")
    panel_fig_b.tight_layout()
    save_figure(panel_fig_b, "figure1b_bmf_vs_periapsis_regenerated")
    plt.close(panel_fig_b)

    write_notes(panel_a_points, panel_b_groups)


if __name__ == "__main__":
    main()
