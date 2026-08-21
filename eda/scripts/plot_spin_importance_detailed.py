#!/usr/bin/env python3
"""Detailed spin-importance figure across velocity, periapsis, and bound mass fraction."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


PLOT_X_MIN = 1.05
PLOT_X_MAX = 2.05
PLOT_Y_MIN = 8e-3
PLOT_Y_MAX = 4e-1

SPIN_ORDER = ["no_spin", "prograde_z", "equatorial", "retrograde_z"]
SPIN_COLORS = {
    "no_spin": "#1f4ed8",
    "prograde_z": "#2e8b57",
    "equatorial": "#c2b000",
    "retrograde_z": "#e24a33",
}
SPIN_LABELS = {
    "no_spin": "No spin",
    "prograde_z": "Prograde z-spin",
    "equatorial": "Equatorial spin",
    "retrograde_z": "Retrograde z-spin",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-table",
        default="outputs/kegerreis_style_bmf_plotting_table.csv",
    )
    parser.add_argument(
        "--summary-out",
        default="outputs/spin_importance_detailed_summary.csv",
    )
    parser.add_argument(
        "--svg-out",
        default="outputs/plots/spin_importance_detailed_by_velocity.svg",
    )
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def group_rows(rows: List[Dict[str, str]]) -> Dict[float, Dict[str, List[Dict[str, float]]]]:
    grouped: Dict[float, Dict[str, List[Dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        velocity = float(row["v_inf_kms"])
        spin = row["spin_orientation"]
        periapsis = float(row["periapsis"])
        bmf = float(row["bound_mass_fraction_median"])
        row_count = int(row["row_count"])
        grouped[velocity][spin].append(
            {
                "periapsis": periapsis,
                "bound_mass_fraction": bmf,
                "row_count": row_count,
            }
        )
    for velocity in grouped:
        for spin in grouped[velocity]:
            grouped[velocity][spin].sort(key=lambda item: item["periapsis"])
    return grouped


def build_summary_rows(grouped: Dict[float, Dict[str, List[Dict[str, float]]]]) -> List[Dict[str, object]]:
    summary: List[Dict[str, object]] = []
    by_velocity_periapsis: Dict[Tuple[float, float], Dict[str, float]] = defaultdict(dict)
    for velocity, spin_map in grouped.items():
        for spin, pts in spin_map.items():
            for pt in pts:
                by_velocity_periapsis[(velocity, pt["periapsis"])][spin] = pt["bound_mass_fraction"]

    for (velocity, periapsis), spin_values in sorted(by_velocity_periapsis.items()):
        values = list(spin_values.values())
        summary.append(
            {
                "v_inf_kms": f"{velocity:g}",
                "periapsis": f"{periapsis:g}",
                "available_spins": ";".join(sorted(spin_values)),
                "spin_count": len(spin_values),
                "min_bound_mass_fraction": f"{min(values):.12g}",
                "max_bound_mass_fraction": f"{max(values):.12g}",
                "spin_spread": f"{(max(values) - min(values)):.12g}",
            }
        )
    return summary


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def radius_from_count(row_count: int) -> float:
    if row_count >= 20:
        return 7.0
    if row_count >= 10:
        return 6.0
    if row_count >= 4:
        return 5.0
    return 4.0


def write_svg(grouped: Dict[float, Dict[str, List[Dict[str, float]]]], out_path: Path) -> None:
    velocities = sorted(grouped)
    width = 1560
    height = 1180
    margin_left = 90
    margin_top = 110
    panel_w = 400
    panel_h = 320
    gap_x = 55
    gap_y = 85
    cols = 3

    log_y_min = math.log10(PLOT_Y_MIN)
    log_y_max = math.log10(PLOT_Y_MAX)

    def x_to_svg(panel_x: float, periapsis: float) -> float:
        return panel_x + (periapsis - PLOT_X_MIN) * panel_w / (PLOT_X_MAX - PLOT_X_MIN)

    def y_to_svg(panel_y: float, bmf: float) -> float:
        return panel_y + (log_y_max - math.log10(bmf)) * panel_h / (log_y_max - log_y_min)

    svg: List[str] = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')
    svg.append(f'<text x="{width/2:.2f}" y="40" font-size="32" text-anchor="middle" fill="#111">Spin Importance by Velocity and Periapsis</text>')
    svg.append(f'<text x="{width/2:.2f}" y="72" font-size="20" text-anchor="middle" fill="#555">Median bound mass fraction from the aggregated plotting table; marker size indicates rows aggregated</text>')

    x_ticks = [1.1, 1.2, 1.4, 1.6, 1.8, 2.0]
    y_ticks = [1e-2, 1e-1]

    for idx, velocity in enumerate(velocities):
        row = idx // cols
        col = idx % cols
        panel_x = margin_left + col * (panel_w + gap_x)
        panel_y = margin_top + row * (panel_h + gap_y)

        svg.append(f'<rect x="{panel_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#111" stroke-width="1.8"/>')
        svg.append(f'<text x="{panel_x + panel_w/2:.2f}" y="{panel_y - 16}" font-size="22" text-anchor="middle" fill="#111">v∞ = {velocity:g} km s⁻¹</text>')

        for tick in x_ticks:
            if tick < PLOT_X_MIN or tick > PLOT_X_MAX:
                continue
            x = x_to_svg(panel_x, tick)
            svg.append(f'<line x1="{x:.2f}" y1="{panel_y}" x2="{x:.2f}" y2="{panel_y + panel_h}" stroke="#e6e6e6" stroke-width="1"/>')
            svg.append(f'<text x="{x:.2f}" y="{panel_y + panel_h + 28}" font-size="18" text-anchor="middle" fill="#222">{tick:g}</text>')

        for tick in y_ticks:
            y = y_to_svg(panel_y, tick)
            svg.append(f'<line x1="{panel_x}" y1="{y:.2f}" x2="{panel_x + panel_w}" y2="{y:.2f}" stroke="#e6e6e6" stroke-width="1"/>')
            label = "1e-1" if tick == 1e-1 else "1e-2"
            svg.append(f'<text x="{panel_x - 12}" y="{y + 6:.2f}" font-size="18" text-anchor="end" fill="#222">{label}</text>')

        spin_map = grouped[velocity]
        for spin in SPIN_ORDER:
            points = spin_map.get(spin)
            if not points:
                continue
            color = SPIN_COLORS[spin]
            if len(points) >= 2:
                path = " ".join(
                    [f"M {x_to_svg(panel_x, points[0]['periapsis']):.2f} {y_to_svg(panel_y, points[0]['bound_mass_fraction']):.2f}"]
                    + [
                        f"L {x_to_svg(panel_x, point['periapsis']):.2f} {y_to_svg(panel_y, point['bound_mass_fraction']):.2f}"
                        for point in points[1:]
                    ]
                )
                svg.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            for point in points:
                x = x_to_svg(panel_x, point["periapsis"])
                y = y_to_svg(panel_y, point["bound_mass_fraction"])
                r = radius_from_count(point["row_count"])
                svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{color}" stroke="white" stroke-width="1.2"/>')

    svg.append(f'<text x="{width/2:.2f}" y="{height - 24}" font-size="28" text-anchor="middle" fill="#111">Periapsis (R♂)</text>')
    svg.append(f'<text x="26" y="{margin_top + panel_h + gap_y/2:.2f}" font-size="28" text-anchor="middle" fill="#111" transform="rotate(-90 26 {margin_top + panel_h + gap_y/2:.2f})">Bound Mass Fraction</text>')

    legend_x = width - 295
    legend_y = 850
    svg.append(f'<rect x="{legend_x - 18}" y="{legend_y - 38}" width="240" height="220" fill="white" stroke="#cccccc" stroke-width="1"/>')
    svg.append(f'<text x="{legend_x}" y="{legend_y - 10}" font-size="24" fill="#111">Spin state</text>')
    for idx, spin in enumerate([spin for spin in SPIN_ORDER if any(spin in grouped[v] for v in velocities)]):
        y = legend_y + idx * 34
        color = SPIN_COLORS[spin]
        svg.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<circle cx="{legend_x + 14}" cy="{y}" r="5" fill="{color}" stroke="white" stroke-width="1"/>')
        svg.append(f'<text x="{legend_x + 42}" y="{y + 7}" font-size="19" fill="#111">{svg_escape(SPIN_LABELS[spin])}</text>')

    size_y = legend_y + 150
    svg.append(f'<text x="{legend_x}" y="{size_y}" font-size="20" fill="#111">Marker size = rows aggregated</text>')
    size_specs = [(4.0, "1-3"), (5.0, "4-9"), (6.0, "10-19"), (7.0, "20+")]
    for idx, (radius, label) in enumerate(size_specs):
        x = legend_x + 12 + idx * 48
        y = size_y + 34
        svg.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="#888" stroke="white" stroke-width="1"/>')
        svg.append(f'<text x="{x}" y="{y + 24}" font-size="15" text-anchor="middle" fill="#444">{label}</text>')

    note = "Interpretation: compare vertical separation between spin curves within each velocity panel; larger separation implies stronger spin importance."
    svg.append(f'<text x="{margin_left}" y="{height - 70}" font-size="19" text-anchor="start" fill="#444">{svg_escape(note)}</text>')

    svg.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.plot_table))
    grouped = group_rows(rows)
    summary_rows = build_summary_rows(grouped)
    write_csv(Path(args.summary_out), summary_rows)
    write_svg(grouped, Path(args.svg_out))
    print(f"Summary CSV: {args.summary_out}")
    print(f"Plot SVG: {args.svg_out}")
    print(f"Velocities shown: {', '.join(str(v) for v in sorted(grouped))}")


if __name__ == "__main__":
    main()
