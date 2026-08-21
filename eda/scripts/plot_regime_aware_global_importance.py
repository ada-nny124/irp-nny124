#!/usr/bin/env python3
"""Create a simpler archive-wide importance figure with a regime-dependent caveat.

This intentionally replaces a raw global permutation-importance story with a
parameter-family grouped-ablation summary that is easier to explain and harder
to over-interpret.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_PNG = Path("eda/plots/physics_feature_importance_inputs_only.png")
OUT_SVG = Path("eda/plots/physics_feature_importance_inputs_only.svg")

# Source: ml/physics_structured_surrogate/reports/physical_parameter_importance_report.md
GLOBAL_DELTA_R2 = {
    "Periapsis": 0.5042,
    "Velocity": 0.2788,
    "Spin": 0.2579,
    "Mass": 0.0910,
    "FoF linking length": 0.0595,
}


def build_plot() -> plt.Figure:
    labels = list(GLOBAL_DELTA_R2.keys())
    values = [GLOBAL_DELTA_R2[label] for label in labels]
    y = np.arange(len(labels))

    plt.style.use("default")
    fig = plt.figure(figsize=(11.4, 6.8), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.85, 1.0], wspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    ax_note = fig.add_subplot(gs[0, 1])

    bar_colors = ["#3557A8", "#4F78C8", "#7092D8", "#96AFE6", "#C7D6F4"]
    bars = ax.barh(y, values, color=bar_colors, edgecolor="none", height=0.66)
    ax.invert_yaxis()

    ax.set_xlim(0, 0.56)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12, color="#203247")
    ax.set_xlabel("Grouped held-out performance loss when the family is removed (|ΔR²|)", fontsize=11, color="#42576C")
    ax.set_title(
        "Archive-wide parameter-family importance is a global average",
        loc="left",
        fontsize=17,
        fontweight="bold",
        color="#17283A",
        pad=16,
    )
    ax.text(
        0,
        1.02,
        "Grouped family ablations for the promoted BMF surrogate. Higher bars mean the model loses more skill when that family is removed.",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#60758D",
        va="bottom",
    )

    ax.grid(axis="x", color="#DFE7EF", linewidth=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#C8D4E0")
    ax.tick_params(axis="x", colors="#60758D", labelsize=11)
    ax.tick_params(axis="y", length=0)

    for rect, value in zip(bars, values):
        ax.text(
            value + 0.01,
            rect.get_y() + rect.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=11,
            color="#3557A8",
            fontweight="bold",
        )

    ax_note.axis("off")
    ax_note.set_xlim(0, 1)
    ax_note.set_ylim(0, 1)
    panel = plt.Rectangle((0.02, 0.09), 0.96, 0.82, facecolor="#F6F9FD", edgecolor="#D7E2EE", linewidth=1.2)
    ax_note.add_patch(panel)

    ax_note.text(0.08, 0.86, "How to read this", fontsize=13.5, fontweight="bold", color="#17283A")
    note_lines = [
        "Across the archive, periapsis is the strongest first-order control on disruption severity and retained debris.",
        "Velocity and spin also matter strongly once correlated orbit proxies are removed with them.",
        "Mass matters, but more conditionally than periapsis, velocity, or spin in this grouped archive-wide average.",
    ]
    y_cursor = 0.77
    for line in note_lines:
        ax_note.text(0.08, y_cursor, line, fontsize=10.8, color="#42576C", wrap=True, va="top")
        y_cursor -= 0.14

    ax_note.text(0.08, 0.37, "Why the ranking changes across regimes", fontsize=13.5, fontweight="bold", color="#17283A")
    caveat_lines = [
        "This figure averages over all sampled cases.",
        "A parameter that looks moderate globally can dominate within a restricted slice.",
        "Example: matched higher-periapsis slices show a larger spin-driven split than lower-periapsis slices, so spin can matter more locally than the global bar alone suggests.",
        "Use this plot for archive-wide intuition, not as a universal physics ranking for every encounter.",
    ]
    y_cursor = 0.30
    for line in caveat_lines:
        ax_note.text(0.08, y_cursor, line, fontsize=10.6, color="#42576C", wrap=True, va="top")
        y_cursor -= 0.11

    fig.text(
        0.065,
        0.035,
        "Global average over the sampled archive. Family scores overlap and are not additive.",
        fontsize=10,
        color="#60758D",
    )
    return fig


def main() -> None:
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig = build_plot()
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_SVG, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_SVG}")


if __name__ == "__main__":
    main()
