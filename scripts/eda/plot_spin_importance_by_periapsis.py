#!/usr/bin/env python3
"""Plot a focused spin-importance summary from the aggregated plotting table.

This figure is intentionally selective: it uses velocities where the current
aggregated table shows a small spin split at low periapsis and a larger split
at higher periapsis.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


FOCUS_VELOCITIES = (0.4, 0.6)
SPIN_A = "no_spin"
SPIN_B = "prograde_z"

COLORS = {
    0.4: "#B8B200",
    0.6: "#F39C12",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-table",
        default="outputs/kegerreis_style_bmf_plotting_table.csv",
    )
    parser.add_argument(
        "--plot-out",
        default="outputs/plots/spin_importance_vs_periapsis.png",
    )
    parser.add_argument(
        "--svg-out",
        default="outputs/plots/spin_importance_vs_periapsis.svg",
    )
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_series(rows: List[Dict[str, str]]) -> Dict[float, List[Tuple[float, float]]]:
    by_key: Dict[Tuple[float, float], Dict[str, float]] = defaultdict(dict)
    for row in rows:
        periapsis = float(row["periapsis"])
        velocity = float(row["v_inf_kms"])
        spin = row["spin_orientation"]
        bmf = float(row["bound_mass_fraction_median"])
        if velocity not in FOCUS_VELOCITIES:
            continue
        if spin not in {SPIN_A, SPIN_B}:
            continue
        by_key[(velocity, periapsis)][spin] = bmf

    series: Dict[float, List[Tuple[float, float]]] = defaultdict(list)
    for (velocity, periapsis), spins in sorted(by_key.items()):
        if SPIN_A in spins and SPIN_B in spins:
            delta = abs(spins[SPIN_B] - spins[SPIN_A])
            series[velocity].append((periapsis, delta))
    return series


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_svg(series: Dict[float, List[Tuple[float, float]]], out_path: Path) -> None:
    width = 1100
    height = 760
    left = 110
    right = 80
    top = 80
    bottom = 120
    plot_width = width - left - right
    plot_height = height - top - bottom

    all_points = [pt for pts in series.values() for pt in pts]
    if not all_points:
        raise ValueError("No matching paired spin rows found in the plotting table.")

    x_values = [x for x, _ in all_points]
    y_values = [y for _, y in all_points]
    x_min = min(x_values) - 0.05
    x_max = max(x_values) + 0.05
    y_min = 0.0
    y_max = max(0.1, max(y_values) * 1.25)

    def x_to_svg(value: float) -> float:
        return left + (value - x_min) * plot_width / (x_max - x_min)

    def y_to_svg(value: float) -> float:
        return top + (y_max - value) * plot_height / (y_max - y_min)

    svg: List[str] = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')

    x_ticks = sorted(set(x_values))
    y_ticks = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]

    for tick in x_ticks:
        x = x_to_svg(tick)
        svg.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#e5e5e5" stroke-width="1"/>')
        svg.append(f'<text x="{x:.2f}" y="{top + plot_height + 34}" font-size="22" text-anchor="middle" fill="#222">{tick:g}</text>')

    for tick in y_ticks:
        if tick > y_max:
            continue
        y = y_to_svg(tick)
        svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e5e5e5" stroke-width="1"/>')
        svg.append(f'<text x="{left - 16}" y="{y + 7:.2f}" font-size="22" text-anchor="end" fill="#222">{tick:.2f}</text>')

    svg.append(f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#111" stroke-width="2"/>')

    for velocity in sorted(series):
        pts = sorted(series[velocity])
        color = COLORS.get(velocity, "#666666")
        path = " ".join(
            [f"M {x_to_svg(pts[0][0]):.2f} {y_to_svg(pts[0][1]):.2f}"]
            + [f"L {x_to_svg(x):.2f} {y_to_svg(y):.2f}" for x, y in pts[1:]]
        )
        svg.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x_value, y_value in pts:
            x = x_to_svg(x_value)
            y = y_to_svg(y_value)
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="{color}" stroke="white" stroke-width="1.5"/>')

    title = "Spin Effect Is Stronger At Higher Periapsis"
    subtitle = "Absolute median BMF difference between no spin and prograde spin"
    svg.append(f'<text x="{width/2:.2f}" y="38" font-size="30" text-anchor="middle" fill="#111">{svg_escape(title)}</text>')
    svg.append(f'<text x="{width/2:.2f}" y="66" font-size="20" text-anchor="middle" fill="#555">{svg_escape(subtitle)}</text>')
    svg.append(f'<text x="{left + plot_width/2:.2f}" y="{height - 32}" font-size="28" text-anchor="middle" fill="#111">Periapsis (R♂)</text>')
    svg.append(f'<text x="34" y="{top + plot_height/2:.2f}" font-size="28" text-anchor="middle" fill="#111" transform="rotate(-90 34 {top + plot_height/2:.2f})">|Δ Bound Mass Fraction|</text>')

    legend_x = width - 240
    legend_y = 135
    svg.append(f'<rect x="{legend_x - 18}" y="{legend_y - 36}" width="190" height="126" fill="white" stroke="#cccccc" stroke-width="1"/>')
    svg.append(f'<text x="{legend_x}" y="{legend_y - 10}" font-size="22" fill="#111">v∞ (km s⁻¹)</text>')
    for idx, velocity in enumerate(sorted(series)):
        y = legend_y + idx * 34
        color = COLORS.get(velocity, "#666666")
        svg.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<circle cx="{legend_x + 14}" cy="{y}" r="5" fill="{color}" stroke="white" stroke-width="1"/>')
        svg.append(f'<text x="{legend_x + 42}" y="{y + 7}" font-size="20" fill="#111">{velocity:g}</text>')

    note = "Low periapsis points stay close; higher periapsis points separate more."
    svg.append(f'<text x="{left}" y="{height - 78}" font-size="20" text-anchor="start" fill="#444">{svg_escape(note)}</text>')

    svg.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.plot_table))
    series = build_series(rows)
    write_svg(series, Path(args.svg_out))
    print(f"Plot SVG: {args.svg_out}")
    for velocity in sorted(series):
        print(f"v={velocity:g}: {series[velocity]}")


if __name__ == "__main__":
    main()
