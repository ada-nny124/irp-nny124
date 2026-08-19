from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = ROOT / "extraction_outputs" / "bound_outcomes.csv"
OUTPUT_DIR = ROOT / "eda" / "bound_eda" / "plots"
OUTPUT_PATH = OUTPUT_DIR / "figure1_mass20_only_temp.png"
MASS_CODE = "A2000"
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


def parse_code(value: str, prefix: str, scale: float) -> float | None:
    if not value or not value.startswith(prefix):
        return None
    digits = value[len(prefix) :]
    if not digits.isdigit():
        return None
    return int(digits) / scale


def parse_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def spin_orientation(spin_code: str) -> str:
    code = spin_code or ""
    if "mz" in code:
        return "retrograde_z"
    if "x" in code or "y" in code:
        return "equatorial"
    if "z" in code and "mz" not in code:
        return "prograde_z"
    return "no_spin"


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with SOURCE_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            if raw.get("mass_code") != MASS_CODE:
                continue
            periapsis = parse_code(raw.get("periapsis_code", ""), "r", 10.0)
            velocity = parse_code(raw.get("velocity_code", ""), "v", 10.0)
            n_fragments = parse_float(raw.get("n_fragments", ""))
            bound_mass_fraction = parse_float(raw.get("bound_mass_fraction", ""))
            rows.append(
                {
                    "periapsis_Rm": periapsis,
                    "v_inf_kms": velocity,
                    "n_fragments": n_fragments,
                    "bound_mass_fraction": bound_mass_fraction,
                    "spin_orientation": spin_orientation(raw.get("spin_code", "")),
                }
            )
    return rows


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#e6e6e6", linewidth=0.8, alpha=0.8)
    ax.set_facecolor("white")


def in_peri_range(value: float | None) -> bool:
    return value is not None and PERI_RANGE[0] <= value <= PERI_RANGE[1]


def draw_panel_a(ax: plt.Axes, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    panel = [
        row
        for row in rows
        if in_peri_range(row["periapsis_Rm"])
        and row["n_fragments"] is not None
        and row["v_inf_kms"] is not None
    ]

    ax.scatter(
        [row["periapsis_Rm"] for row in panel],
        [row["n_fragments"] for row in panel],
        color="#f28e2b",
        marker="o",
        s=26,
        alpha=0.75,
        linewidths=0.25,
        edgecolors="white",
        zorder=2,
    )

    ax.set_title(r"Fragment count vs periapsis ($10^{20}$ kg only)", fontsize=12)
    ax.set_xlabel(r"Periapsis ($R_{\mathrm{Mars}}$)", fontsize=11)
    ax.set_ylabel("Number of FoF fragments")
    ax.set_xlim(*PERI_RANGE)
    ax.set_xticks(PERI_TICKS)
    ax.tick_params(axis="x", labelsize=9, pad=4)
    style_axes(ax)
    return panel


def build_grouped_panel_b(rows: list[dict[str, object]]) -> list[dict[str, float | str | int]]:
    panel = [
        row
        for row in rows
        if in_peri_range(row["periapsis_Rm"])
        and row["bound_mass_fraction"] is not None
        and row["v_inf_kms"] is not None
    ]
    grouped_values: dict[tuple[float, float, str], list[float]] = defaultdict(list)
    for row in panel:
        key = (float(row["periapsis_Rm"]), float(row["v_inf_kms"]), str(row["spin_orientation"]))
        grouped_values[key].append(float(row["bound_mass_fraction"]))

    grouped_rows: list[dict[str, float | str | int]] = []
    for (periapsis, velocity, spin_name), values in grouped_values.items():
        grouped_rows.append(
            {
                "periapsis_Rm": periapsis,
                "v_inf_kms": velocity,
                "spin_orientation": spin_name,
                "bound_mass_fraction_median": float(np.median(np.asarray(values, dtype=float))),
                "raw_row_count": len(values),
            }
        )
    grouped_rows.sort(key=lambda row: (row["v_inf_kms"], row["spin_orientation"], row["periapsis_Rm"]))
    return grouped_rows


def draw_panel_b(ax: plt.Axes, grouped_rows: list[dict[str, float | str | int]]) -> None:
    series: dict[tuple[float, str], list[dict[str, float | str | int]]] = defaultdict(list)
    for row in grouped_rows:
        series[(float(row["v_inf_kms"]), str(row["spin_orientation"]))].append(row)

    for (velocity, spin_name), subset in sorted(series.items()):
        subset.sort(key=lambda row: float(row["periapsis_Rm"]))
        color = VELOCITY_COLORS[velocity]
        marker = SPIN_MARKERS.get(spin_name, "o")
        linestyle = SPIN_LINESTYLES.get(spin_name, "solid")
        x = [float(row["periapsis_Rm"]) for row in subset]
        y = [float(row["bound_mass_fraction_median"]) for row in subset]
        if len(subset) > 1:
            ax.plot(
                x,
                y,
                color=color,
                linewidth=1.6,
                linestyle=linestyle,
                alpha=0.9,
                zorder=2,
            )
        ax.scatter(
            x,
            y,
            color=[color],
            marker=marker,
            s=30,
            linewidths=0.45,
            edgecolors="white",
            alpha=0.95,
            zorder=3,
        )

    ax.set_title(r"Bound Mass Fraction vs Periapsis ($10^{20}$ kg only)", fontsize=12)
    ax.set_xlabel(r"Periapsis ($R_{\mathrm{Mars}}$)", fontsize=11)
    ax.set_ylabel("Bound Mass Fraction")
    ax.set_xlim(*PERI_RANGE)
    ax.set_xticks(PERI_TICKS)
    ax.tick_params(axis="x", labelsize=9, pad=4)
    ymax = max((float(row["bound_mass_fraction_median"]) for row in grouped_rows), default=0.3)
    ax.set_ylim(0.0, max(0.32, ymax * 1.08))
    style_axes(ax)


def add_legends(ax: plt.Axes, grouped_rows: list[dict[str, float | str | int]]) -> None:
    velocities = sorted({float(row["v_inf_kms"]) for row in grouped_rows})
    velocity_handles = [
        Line2D([0], [0], color=VELOCITY_COLORS[v], linewidth=2.0, label=f"{v:g}")
        for v in velocities
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), dpi=220)
    panel_a_rows = draw_panel_a(axes[0], rows)
    grouped_rows = build_grouped_panel_b(rows)
    draw_panel_b(axes[1], grouped_rows)
    add_legends(axes[1], grouped_rows)
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.98))
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {OUTPUT_PATH}")
    print(f"Mass filter: {MASS_CODE}")
    print(f"Panel A rows: {len(panel_a_rows)}")
    print(f"Panel B grouped cells: {len(grouped_rows)}")


if __name__ == "__main__":
    main()
