#!/usr/bin/env python3
"""Plot a Kegerreis-style bound-mass-fraction figure from a cleaned CSV."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt


BMF_FLOOR = 1e-4
VELOCITY_COLORS = {
    0.0: "#1565C0",
    0.2: "#2E7D32",
    0.4: "#558B2F",
    0.6: "#F9A825",
    0.8: "#EF6C00",
    1.0: "#B71C1C",
    1.2: "#6A1B9A",
    1.4: "#00838F",
    1.6: "#37474F",
    2.0: "#212121",
}
SPIN_LINESTYLES = {
    "no_spin": "-",
    "prograde_z": "--",
    "retrograde_z": "-.",
    "equatorial": ":",
    "other": (0, (3, 1, 1, 1)),
}
SPIN_LABELS = {
    "no_spin": "No spin",
    "prograde_z": "Prograde z-spin",
    "retrograde_z": "Retrograde z-spin",
    "equatorial": "Equatorial spin",
    "other": "Other spin",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cleaned-csv",
        default="outputs/plots/kegerreis_figure6_cleaned_dataset.csv",
    )
    parser.add_argument(
        "--plot-out",
        default="outputs/plots/kegerreis_figure6_bound_mass_fraction_vs_periapsis.png",
    )
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["periapsis_rm"] = float(row["periapsis_rm"])
            row["v_inf_kms"] = float(row["v_inf_kms"])
            row["bound_mass_fraction"] = float(row["bound_mass_fraction"])
            rows.append(row)
    return rows


def plot_rows(rows: List[Dict[str, object]], plot_out: Path) -> int:
    positive_rows = [row for row in rows if row["bound_mass_fraction"] > BMF_FLOOR]

    grouped: Dict[Tuple[str, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in positive_rows:
        key = (
            str(row["mass_code"]),
            str(row["velocity_code"]),
            str(row["spin_code"]),
            str(row["resolution_code"]),
            str(row["timestep"]),
            str(row["fof_linking_length"]),
        )
        grouped[key].append(row)

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    velocity_handles: Dict[float, mlines.Line2D] = {}
    spin_handles: Dict[str, mlines.Line2D] = {}

    for key, group in sorted(grouped.items()):
        group.sort(key=lambda row: row["periapsis_rm"])
        velocity = float(group[0]["v_inf_kms"])
        spin_orientation = str(group[0]["spin_orientation"])
        color = VELOCITY_COLORS.get(velocity, "#666666")
        linestyle = SPIN_LINESTYLES.get(spin_orientation, "-")
        alpha = 0.9 if len(group) > 1 else 0.75
        linewidth = 1.6 if spin_orientation == "no_spin" else 1.3

        x_values = [row["periapsis_rm"] for row in group]
        y_values = [row["bound_mass_fraction"] for row in group]

        if len(group) > 1:
            ax.plot(
                x_values,
                y_values,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
            )

        ax.scatter(
            x_values,
            y_values,
            color=color,
            s=18,
            alpha=alpha,
            edgecolors="none",
        )

        if velocity not in velocity_handles:
            velocity_handles[velocity] = mlines.Line2D(
                [], [], color=color, linestyle="-", linewidth=2, label=f"{velocity:.1f}"
            )
        if spin_orientation not in spin_handles:
            spin_handles[spin_orientation] = mlines.Line2D(
                [], [], color="black", linestyle=linestyle, linewidth=2, label=SPIN_LABELS[spin_orientation]
            )

    ax.set_yscale("log")
    ax.set_ylim(BMF_FLOOR * 0.7, 1.05)
    ax.set_xlabel(r"Periapsis ($R_M$)")
    ax.set_ylabel("Bound Mass Fraction")
    ax.set_title("Retained / Bound Mass Fraction vs Periapsis")
    ax.grid(True, which="both", linestyle="--", alpha=0.2)
    ax.annotate(
        "Zero-BMF rows omitted for log scale",
        xy=(0.01, 0.01),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#666666",
    )

    if positive_rows:
        ax.set_xlim(
            min(row["periapsis_rm"] for row in positive_rows) - 0.05,
            max(row["periapsis_rm"] for row in positive_rows) + 0.05,
        )

    velocity_legend = ax.legend(
        handles=[velocity_handles[key] for key in sorted(velocity_handles)],
        title=r"$v_\infty$ (km s$^{-1}$)",
        loc="lower left",
        fontsize=8.5,
        title_fontsize=9,
        framealpha=0.92,
    )
    ax.add_artist(velocity_legend)

    ax.legend(
        handles=[spin_handles[key] for key in SPIN_LABELS if key in spin_handles],
        title="Spin orientation",
        loc="upper right",
        fontsize=8.5,
        title_fontsize=9,
        framealpha=0.92,
    )

    fig.tight_layout()
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return len(positive_rows)


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.cleaned_csv))
    plotted_rows = plot_rows(rows, Path(args.plot_out))
    print(f"Plot rows used: {plotted_rows}")
    print(f"Plot saved: {args.plot_out}")


if __name__ == "__main__":
    main()
