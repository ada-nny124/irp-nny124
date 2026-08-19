from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIG_PATH = ROOT / "report-table-figure" / "figures" / "figureA1_used_in_report.png"
FEATURE_CONTRIBUTION_PATH = ROOT / "ml" / "trainingartifacts" / "physics_rf" / "main_bmf_physics_rf_feature_contribution.json"


def load_metrics(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def remake_plot(output_path: Path = FIG_PATH, metrics: dict[str, object] | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = metrics or load_metrics(FEATURE_CONTRIBUTION_PATH)

    top_metrics = metrics["top"]
    bottom_metrics = metrics["bottom"]

    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        plt.style.use("classic")

    fig, axes = plt.subplots(2, 1, figsize=(10, 11), gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0]
    top_labels = ["Raw", "Raw + simple", "Raw + physics", "All"]
    top_values = [float(top_metrics[label]) for label in top_labels]
    top_colors = ["#d9d9d9", "#33a02c", "#ff7f00", "#e31a1c"]
    top_x = np.arange(len(top_labels))
    top_bars = ax.bar(top_x, top_values, color=top_colors, edgecolor="none")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(top_x)
    ax.set_xticklabels(top_labels, rotation=0, ha="center", fontsize=8)
    ax.tick_params(axis="x", which="major", pad=8)
    ax.set_ylabel(r"Change in grouped held-out $R^2$ vs raw baseline")
    ax.set_title("Contribution of Physics-Derived Features")
    ax.set_ylim(0.0, max(0.025, max(top_values) * 1.1 if top_values else 0.025))

    for rect, value, color in zip(top_bars, top_values, top_colors):
        cx = rect.get_x() + rect.get_width() / 2.0
        h = rect.get_height()
        text_color = "white" if color == "#33a02c" else "black"
        y = max(h * 0.5, 0.001)
        ax.text(cx, y, f"{value:+.3f}", ha="center", va="center", fontsize=10, fontweight="semibold", color=text_color)

    ax2 = axes[1]
    bottom_labels = ["Raw", r"v_inf^2", "1/r_p", "f_spin", "radius", "all simple"]
    bottom_values = [0.0] + [float(bottom_metrics[label]) for label in bottom_labels[1:]]
    bottom_x = np.arange(len(bottom_labels))
    bottom_bars = ax2.bar(bottom_x, bottom_values, color="#6c83b5", edgecolor="none")
    ax2.axhline(0, color="k", linewidth=0.8)
    ax2.set_xticks(bottom_x)
    ax2.set_xticklabels(bottom_labels, rotation=0, ha="center", fontsize=8)
    ax2.tick_params(axis="x", which="major", pad=8)
    ax2.set_ylabel(r"Change in grouped held-out $R^2$ vs raw baseline")
    ax2.set_title("Simple Transform Checks")
    spread = max(abs(min(bottom_values)), abs(max(bottom_values)), 0.01)
    ax2.set_ylim(min(-0.025, -spread * 1.2), max(0.01, spread * 1.2))

    for rect, value in zip(bottom_bars, bottom_values):
        cx = rect.get_x() + rect.get_width() / 2.0
        if value >= 0:
            y = max(value * 0.55, 0.0015)
            va = "center"
        else:
            y = value * 0.5
            va = "center"
        ax2.text(cx, y, f"{value:+.3f}", ha="center", va=va, fontsize=10, fontweight="semibold", color="white")

    plt.subplots_adjust(hspace=0.45, top=0.92)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    remake_plot()
    print(FIG_PATH)


if __name__ == "__main__":
    main()
