#!/usr/bin/env python3
"""Create a linear bound-mass-vs-periapsis plot using all raw data.

This version does not do balanced/resampled weighting. It aggregates raw FoF
rows directly into (periapsis, velocity, spin_orientation) cells and plots the
median bound mass fraction for each cell.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, List, Tuple


SPIN_LABELS = {
    "no_spin": "0",
    "prograde_z": "0.5",
    "equatorial": "1",
    "retrograde_z": "1",
}

SPIN_LINESTYLES = {
    "no_spin": None,
    "prograde_z": "9 5",
    "equatorial": "10 4 2 4",
    "retrograde_z": "2 5",
}

VELOCITY_COLORS = {
    0.0: "#1565C0",
    0.2: "#2E7D32",
    0.4: "#B8B200",
    0.6: "#F39C12",
    0.8: "#E53935",
    1.0: "#8E44AD",
}

PLOT_X_MIN = 1.05
PLOT_X_MAX = 2.05
PLOT_Y_MIN = 0.0
PLOT_Y_MAX = 0.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="outputs/bound_outcomes.csv")
    parser.add_argument("--svg-out", default="outputs/plots/bound_mass_fraction_vs_periapsis_linear_all data.svg")
    parser.add_argument("--csv-out", default="outputs/bound_mass_fraction_vs_periapsis_linear_all_data.csv")
    return parser.parse_args()


def parse_code_decimal(code: str, prefix: str) -> float | None:
    text = (code or "").strip()
    if not text.startswith(prefix):
        return None
    digits = text[len(prefix) :]
    if not digits.isdigit():
        return None
    return int(digits) / 10.0


def infer_spin_orientation(spin_code: str) -> str:
    code = (spin_code or "").strip()
    if not code:
        return "no_spin"
    suffix = code[4:] if len(code) > 4 else "z"
    if suffix == "z":
        return "prograde_z"
    if suffix == "mz":
        return "retrograde_z"
    if suffix in {"x", "mx", "y", "my"}:
        return "equatorial"
    return "equatorial"


def load_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            periapsis = parse_code_decimal(row["periapsis_code"], "r")
            velocity = parse_code_decimal(row["velocity_code"], "v")
            bmf = float(row["bound_mass_fraction"])
            if periapsis is None or velocity is None:
                continue
            if periapsis > 2.0:
                continue
            rows.append(
                {
                    "bound_mass_fraction": bmf,
                    "periapsis": periapsis,
                    "velocity": velocity,
                    "spin_orientation": infer_spin_orientation(str(row.get("spin_code", ""))),
                }
            )
    return rows


def aggregate_all_data(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[float, float, str], List[float]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["periapsis"]), float(row["velocity"]), str(row["spin_orientation"]))].append(
            float(row["bound_mass_fraction"])
        )

    out: List[Dict[str, object]] = []
    for (periapsis, velocity, spin_orientation), values in sorted(grouped.items()):
        out.append(
            {
                "periapsis": periapsis,
                "v_inf_kms": velocity,
                "spin_orientation": spin_orientation,
                "bound_mass_fraction_median": median(values),
                "raw_row_count": len(values),
            }
        )
    return out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "periapsis",
                "v_inf_kms",
                "spin_orientation",
                "bound_mass_fraction_median",
                "raw_row_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def svg_escape(text: object) -> str:
    value = str(text)
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_svg(rows: List[Dict[str, object]], out_path: Path) -> None:
    width, height = 1280, 860
    left, right, top, bottom = 110, 282, 44, 128
    plot_w, plot_h = width - left - right, height - top - bottom

    def x_to_svg(value: float) -> float:
        return left + (value - PLOT_X_MIN) * plot_w / (PLOT_X_MAX - PLOT_X_MIN)

    def y_to_svg(value: float) -> float:
        return top + (PLOT_Y_MAX - value) * plot_h / (PLOT_Y_MAX - PLOT_Y_MIN)

    grouped: Dict[Tuple[float, str], List[Dict[str, object]]] = defaultdict(list)
    velocities_seen: List[float] = []
    spins_seen: List[str] = []
    for row in rows:
        grouped[(float(row["v_inf_kms"]), str(row["spin_orientation"]))].append(row)
        if float(row["v_inf_kms"]) not in velocities_seen:
            velocities_seen.append(float(row["v_inf_kms"]))
        if str(row["spin_orientation"]) not in spins_seen:
            spins_seen.append(str(row["spin_orientation"]))
    velocities_seen.sort()
    spins_seen.sort(key=lambda item: list(SPIN_LABELS).index(item) if item in SPIN_LABELS else 999)

    svg: List[str] = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')
    svg.append(f'<text x="{width/2:.2f}" y="34" font-size="30" text-anchor="middle" fill="#111">Bound Mass Fraction vs Periapsis (All Data)</text>')
    svg.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#111" stroke-width="2"/>')

    for tick in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]:
        x = x_to_svg(tick)
        svg.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#e0e0e0" stroke-width="1"/>')
        svg.append(f'<text x="{x:.2f}" y="{top + plot_h + 36}" font-size="20" text-anchor="middle" fill="#222">{tick:g}</text>')
    for tick in [0.0, 0.1, 0.2, 0.3]:
        y = y_to_svg(tick)
        svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e0e0e0" stroke-width="1"/>')
        svg.append(f'<text x="{left - 16}" y="{y + 7:.2f}" font-size="20" text-anchor="end" fill="#222">{tick:g}</text>')

    for (velocity, spin_orientation), members in sorted(grouped.items()):
        pts = sorted(members, key=lambda item: float(item["periapsis"]))
        color = VELOCITY_COLORS.get(velocity, "#666666")
        dash = SPIN_LINESTYLES.get(spin_orientation)
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        if len(pts) >= 2:
            path_d = " ".join(
                [f"M {x_to_svg(float(pts[0]['periapsis'])):.2f} {y_to_svg(float(pts[0]['bound_mass_fraction_median'])):.2f}"]
                + [
                    f"L {x_to_svg(float(pt['periapsis'])):.2f} {y_to_svg(float(pt['bound_mass_fraction_median'])):.2f}"
                    for pt in pts[1:]
                ]
            )
            svg.append(f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2.5"{extra} opacity="0.95"/>')
        for pt in pts:
            x = x_to_svg(float(pt["periapsis"]))
            y = y_to_svg(float(pt["bound_mass_fraction_median"]))
            n = int(pt["raw_row_count"])
            r = 4.5 if n < 5 else 5.5 if n < 15 else 6.5
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{color}" stroke="white" stroke-width="1.2"/>')

    svg.append(f'<text x="{left + plot_w/2:.2f}" y="{height - 18}" font-size="26" text-anchor="middle" fill="#111">Periapsis (R♂)</text>')
    svg.append(f'<text x="30" y="{top + plot_h/2:.2f}" font-size="26" text-anchor="middle" fill="#111" transform="rotate(-90 30 {top + plot_h/2:.2f})">Bound Mass Fraction</text>')

    legend_x = width - 250
    legend_y = 120
    svg.append(f'<rect x="{legend_x - 16}" y="{legend_y - 34}" width="230" height="{36 * len(velocities_seen) + 28}" fill="white" stroke="#cccccc" stroke-width="1"/>')
    svg.append(f'<text x="{legend_x}" y="{legend_y - 10}" font-size="22" fill="#111">v∞ (km s⁻¹)</text>')
    for index, velocity in enumerate(velocities_seen):
        y = legend_y + index * 30
        color = VELOCITY_COLORS.get(velocity, "#666666")
        svg.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 26}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{legend_x + 40}" y="{y + 7}" font-size="19" fill="#111">{velocity:g}</text>')

    spin_legend_y = legend_y + 36 * len(velocities_seen) + 148
    svg.append(f'<rect x="{legend_x - 16}" y="{spin_legend_y - 28}" width="230" height="{32 * len(spins_seen) + 24}" fill="white" stroke="#cccccc" stroke-width="1"/>')
    svg.append(f'<text x="{legend_x}" y="{spin_legend_y - 8}" font-size="22" fill="#111">Lz (Lmax) - spin</text>')
    for index, spin_orientation in enumerate(spins_seen):
        y = spin_legend_y + index * 28
        dash = SPIN_LINESTYLES.get(spin_orientation)
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="#111" stroke-width="2.5"{extra}/>')
        svg.append(f'<text x="{legend_x + 40}" y="{y + 7}" font-size="18" fill="#111">{svg_escape(SPIN_LABELS.get(spin_orientation, spin_orientation))}</text>')

    note = "All-data aggregation: median bound mass fraction per periapsis/velocity/spin cell from raw FoF rows."
    svg.append(f'<text x="{left}" y="{height - 54}" font-size="18" text-anchor="start" fill="#555">{svg_escape(note)}</text>')
    svg.append("</svg>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.source))
    aggregated_rows = aggregate_all_data(rows)
    write_csv(Path(args.csv_out), aggregated_rows)
    build_svg(aggregated_rows, Path(args.svg_out))
    print(f"All-data CSV: {args.csv_out}")
    print(f"All-data SVG: {args.svg_out}")
    print(f"Raw rows used (periapsis <= 2.0): {len(rows)}")
    print(f"Aggregated cells: {len(aggregated_rows)}")


if __name__ == "__main__":
    main()
