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
OUTPUT_PATH = OUTPUT_DIR / "figure6_kegerreis_subset_temp.png"

PERI_RANGE = (1.1, 3.0)
PERI_TICKS = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8]
VALID_VELOCITIES = {"v00", "v02", "v04", "v06", "v08"}
TARGET_FOF_LINKING_LENGTH = "0.004"
TARGET_TIMESTEP = "90000"

VELOCITY_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8]
VELOCITY_COLORS = {
    0.0: "#1f77b4",
    0.2: "#2ca02c",
    0.4: "#bcbd22",
    0.6: "#ff7f0e",
    0.8: "#ff4d4d",
}
SERIES_STYLES = {
    "0": "solid",
    "0.5": (0, (6, 4)),
    "1": "dashdot",
    "differentiated": (0, (1.5, 3.5)),
}
SERIES_LABELS = {
    "0": "0",
    "0.5": "0.5",
    "1": "1",
    "differentiated": "Differentiated",
}


def parse_numeric_code(value: str, prefix: str, scale: float) -> float | None:
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


def classify_paper_series(row: dict[str, str]) -> str | None:
    mass_code = row.get("mass_code", "")
    spin_code = row.get("spin_code", "")

    if mass_code == "A2000c30" and spin_code == "":
        return "differentiated"
    if mass_code != "A2000":
        return None
    if spin_code == "":
        return "0"
    if spin_code == "s047z":
        return "0.5"
    if spin_code == "s030z":
        return "1"
    return None


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with SOURCE_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            series = classify_paper_series(raw)
            if series is None:
                continue
            if raw.get("velocity_code") not in VALID_VELOCITIES:
                continue
            if raw.get("fof_linking_length") != TARGET_FOF_LINKING_LENGTH:
                continue
            if raw.get("timestep") != TARGET_TIMESTEP:
                continue

            periapsis = parse_numeric_code(raw.get("periapsis_code", ""), "r", 10.0)
            velocity = parse_numeric_code(raw.get("velocity_code", ""), "v", 10.0)
            bmf = parse_float(raw.get("bound_mass_fraction", ""))
            if periapsis is None or velocity is None or bmf is None:
                continue

            rows.append(
                {
                    "periapsis_Rm": periapsis,
                    "v_inf_kms": velocity,
                    "bound_mass_fraction": bmf,
                    "series": series,
                }
            )
    return rows


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#e6e6e6", linewidth=0.8, alpha=0.8)
    ax.set_facecolor("white")


def build_grouped_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped_values: dict[tuple[float, float, str], list[float]] = defaultdict(list)
    for row in rows:
        periapsis = float(row["periapsis_Rm"])
        if not (PERI_RANGE[0] <= periapsis <= PERI_RANGE[1]):
            continue
        key = (periapsis, float(row["v_inf_kms"]), str(row["series"]))
        grouped_values[key].append(float(row["bound_mass_fraction"]))

    grouped_rows: list[dict[str, object]] = []
    for (periapsis, velocity, series), values in grouped_values.items():
        grouped_rows.append(
            {
                "periapsis_Rm": periapsis,
                "v_inf_kms": velocity,
                "series": series,
                "bound_mass_fraction_median": float(np.median(np.asarray(values, dtype=float))),
                "raw_row_count": len(values),
            }
        )
    grouped_rows.sort(key=lambda row: (row["v_inf_kms"], row["series"], row["periapsis_Rm"]))
    return grouped_rows


def draw_plot(ax: plt.Axes, grouped_rows: list[dict[str, object]]) -> None:
    by_series: dict[tuple[float, str], list[dict[str, object]]] = defaultdict(list)
    for row in grouped_rows:
        by_series[(float(row["v_inf_kms"]), str(row["series"]))].append(row)

    for (velocity, series), subset in sorted(by_series.items()):
        subset.sort(key=lambda row: float(row["periapsis_Rm"]))
        x = [float(row["periapsis_Rm"]) for row in subset]
        y = [float(row["bound_mass_fraction_median"]) for row in subset]
        ax.plot(
            x,
            y,
            color=VELOCITY_COLORS[velocity],
            linewidth=2.0,
            linestyle=SERIES_STYLES[series],
            alpha=0.95,
            zorder=2,
        )
        ax.scatter(
            x,
            y,
            color=[VELOCITY_COLORS[velocity]],
            s=28,
            linewidths=0.4,
            edgecolors="white",
            alpha=0.95,
            zorder=3,
        )

    ax.set_title(r"Bound Mass Fraction vs Periapsis (Kegerreis Figure 6 subset)", fontsize=13)
    ax.set_xlabel(r"Periapsis ($R_{\mathrm{Mars}}$)", fontsize=11)
    ax.set_ylabel("Bound Mass Fraction", fontsize=11)
    ax.set_xlim(*PERI_RANGE)
    ax.set_xticks(PERI_TICKS)
    ymax = max((float(row["bound_mass_fraction_median"]) for row in grouped_rows), default=0.3)
    ax.set_ylim(0.0, max(0.32, ymax * 1.08))
    style_axes(ax)


def add_legends(ax: plt.Axes) -> None:
    velocity_handles = [
        Line2D([0], [0], color=VELOCITY_COLORS[v], linewidth=2.0, label=f"{v:g}")
        for v in VELOCITY_VALUES
    ]
    spin_handles = [
        Line2D([0], [0], color="#666666", linewidth=1.8, linestyle=SERIES_STYLES[key], label=SERIES_LABELS[key])
        for key in ["0", "0.5", "1"]
    ]
    differentiated_handle = Line2D(
        [0],
        [0],
        color="#666666",
        linewidth=1.8,
        linestyle=SERIES_STYLES["differentiated"],
        label=SERIES_LABELS["differentiated"],
    )

    leg1 = ax.legend(
        handles=velocity_handles,
        loc="upper right",
        frameon=True,
        title=r"$v_\infty$ (km s$^{-1}$)",
        title_fontsize=9,
        fontsize=8,
    )
    ax.add_artist(leg1)
    leg2 = ax.legend(
        handles=spin_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.72),
        frameon=True,
        title=r"$L_z$ ($L_{\max}$)",
        title_fontsize=9,
        fontsize=8,
    )
    ax.add_artist(leg2)
    ax.legend(
        handles=[differentiated_handle],
        loc="upper left",
        frameon=True,
        fontsize=8,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    grouped_rows = build_grouped_rows(rows)

    fig, ax = plt.subplots(1, 1, figsize=(8.6, 5.8), dpi=220)
    draw_plot(ax, grouped_rows)
    add_legends(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {OUTPUT_PATH}")
    print("Subset settings:")
    print("- mass_code in {A2000, A2000c30}")
    print("- velocity_code in {v00, v02, v04, v06, v08}")
    print(f"- fof_linking_length == {TARGET_FOF_LINKING_LENGTH}")
    print(f"- timestep == {TARGET_TIMESTEP}")
    print("- series 0 := A2000 with no spin")
    print("- series 0.5 := A2000 with spin_code s047z")
    print("- series 1 := A2000 with spin_code s030z")
    print("- differentiated := A2000c30 with no spin")
    print(f"Grouped cells plotted: {len(grouped_rows)}")


if __name__ == "__main__":
    main()
