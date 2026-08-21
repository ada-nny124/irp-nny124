#!/usr/bin/env python3
"""Create bound-mass summary graphs and heatmaps from the repo's bound outcomes table."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Sequence, Tuple


SOURCE_CSV = "extraction-outputs/tables/bound_outcomes.csv"
OUTPUT_DIR = "outputs/plots"

SPIN_ORDER = ["no_spin", "prograde_z", "equatorial", "retrograde_z"]
SPIN_LABELS = {
    "no_spin": "No spin",
    "prograde_z": "Prograde z-spin",
    "equatorial": "Equatorial spin",
    "retrograde_z": "Retrograde z-spin",
}
SPIN_COLORS = {
    "no_spin": "#1f4ed8",
    "prograde_z": "#2e8b57",
    "equatorial": "#b6a400",
    "retrograde_z": "#e24a33",
}
HEATMAP_COLORS = [
    "#f7fbff",
    "#d6e9f8",
    "#abd0ee",
    "#7fb6e3",
    "#5399d8",
    "#2f7ac7",
    "#1557a1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=SOURCE_CSV)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
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
            bmf = float(row["bound_mass_fraction"])
            periapsis = parse_code_decimal(row["periapsis_code"], "r")
            velocity = parse_code_decimal(row["velocity_code"], "v")
            if periapsis is None or velocity is None:
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


def svg_escape(text: object) -> str:
    value = str(text)
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def grouped_median(rows: Sequence[Dict[str, object]], key_fn) -> List[Tuple[object, float, int]]:
    grouped: Dict[object, List[float]] = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(float(row["bound_mass_fraction"]))
    items = []
    for key, values in grouped.items():
        items.append((key, median(values), len(values)))
    return sorted(items, key=lambda item: item[0])


def write_svg(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def line_plot_svg(
    data: Sequence[Tuple[float, float, int]],
    title: str,
    x_label: str,
    out_path: Path,
    *,
    color: str = "#1f4ed8",
) -> None:
    width, height = 1100, 760
    left, right, top, bottom = 110, 60, 70, 110
    plot_w, plot_h = width - left - right, height - top - bottom
    x_values = [x for x, _, _ in data]
    y_values = [y for _, y, _ in data]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = 0.0, max(0.3, max(y_values) * 1.15)

    def x_to_svg(value: float) -> float:
        return left + (value - x_min) * plot_w / (x_max - x_min if x_max != x_min else 1.0)

    def y_to_svg(value: float) -> float:
        return top + (y_max - value) * plot_h / (y_max - y_min if y_max != y_min else 1.0)

    svg: List[str] = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')
    svg.append(f'<text x="{width/2:.2f}" y="38" font-size="30" text-anchor="middle" fill="#111">{svg_escape(title)}</text>')
    svg.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#111" stroke-width="2"/>')

    for x, _, _ in data:
        xx = x_to_svg(x)
        svg.append(f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{top + plot_h}" stroke="#e5e5e5" stroke-width="1"/>')
        svg.append(f'<text x="{xx:.2f}" y="{top + plot_h + 34}" font-size="20" text-anchor="middle" fill="#222">{x:g}</text>')
    for tick in [0.0, 0.1, 0.2, 0.3]:
        if tick > y_max:
            continue
        yy = y_to_svg(tick)
        svg.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{left + plot_w}" y2="{yy:.2f}" stroke="#e5e5e5" stroke-width="1"/>')
        svg.append(f'<text x="{left - 14}" y="{yy + 7:.2f}" font-size="20" text-anchor="end" fill="#222">{tick:g}</text>')

    pts = [(x_to_svg(x), y_to_svg(y), n) for x, y, n in data]
    if pts:
        path_d = " ".join([f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"] + [f"L {x:.2f} {y:.2f}" for x, y, _ in pts[1:]])
        svg.append(f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, y, n in pts:
            r = 4 if n < 5 else 5 if n < 15 else 6
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{color}" stroke="white" stroke-width="1.2"/>')

    svg.append(f'<text x="{left + plot_w/2:.2f}" y="{height - 26}" font-size="26" text-anchor="middle" fill="#111">{svg_escape(x_label)}</text>')
    svg.append(f'<text x="32" y="{top + plot_h/2:.2f}" font-size="26" text-anchor="middle" fill="#111" transform="rotate(-90 32 {top + plot_h/2:.2f})">Bound Mass Fraction</text>')
    write_svg(out_path, "\n".join(svg + ["</svg>"]))


def bar_plot_svg(
    data: Sequence[Tuple[str, float, int]],
    title: str,
    x_label: str,
    out_path: Path,
) -> None:
    width, height = 1100, 760
    left, right, top, bottom = 110, 60, 70, 130
    plot_w, plot_h = width - left - right, height - top - bottom
    y_max = max(0.3, max(y for _, y, _ in data) * 1.15)
    bar_w = plot_w / max(len(data), 1) * 0.56

    def y_to_svg(value: float) -> float:
        return top + (y_max - value) * plot_h / (y_max if y_max else 1.0)

    svg: List[str] = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')
    svg.append(f'<text x="{width/2:.2f}" y="38" font-size="30" text-anchor="middle" fill="#111">{svg_escape(title)}</text>')
    svg.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#111" stroke-width="2"/>')

    for tick in [0.0, 0.1, 0.2, 0.3]:
        if tick > y_max:
            continue
        yy = y_to_svg(tick)
        svg.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{left + plot_w}" y2="{yy:.2f}" stroke="#e5e5e5" stroke-width="1"/>')
        svg.append(f'<text x="{left - 14}" y="{yy + 7:.2f}" font-size="20" text-anchor="end" fill="#222">{tick:g}</text>')

    step = plot_w / max(len(data), 1)
    for idx, (label, value, count) in enumerate(data):
        cx = left + step * (idx + 0.5)
        x = cx - bar_w / 2
        y = y_to_svg(value)
        color = SPIN_COLORS.get(label, "#4b5563")
        svg.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{top + plot_h - y:.2f}" fill="{color}" opacity="0.9"/>')
        svg.append(f'<text x="{cx:.2f}" y="{top + plot_h + 40}" font-size="18" text-anchor="middle" fill="#222">{svg_escape(SPIN_LABELS.get(label, label))}</text>')
        svg.append(f'<text x="{cx:.2f}" y="{y - 10:.2f}" font-size="15" text-anchor="middle" fill="#444">n={count}</text>')

    svg.append(f'<text x="{left + plot_w/2:.2f}" y="{height - 30}" font-size="26" text-anchor="middle" fill="#111">{svg_escape(x_label)}</text>')
    svg.append(f'<text x="32" y="{top + plot_h/2:.2f}" font-size="26" text-anchor="middle" fill="#111" transform="rotate(-90 32 {top + plot_h/2:.2f})">Bound Mass Fraction</text>')
    write_svg(out_path, "\n".join(svg + ["</svg>"]))


def color_for_value(value: float, vmin: float, vmax: float) -> str:
    if vmax <= vmin:
        return HEATMAP_COLORS[-1]
    frac = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    idx = min(len(HEATMAP_COLORS) - 1, int(frac * (len(HEATMAP_COLORS) - 1) + 0.5))
    return HEATMAP_COLORS[idx]


def heatmap_svg(
    matrix: Dict[Tuple[object, object], float],
    x_keys: Sequence[object],
    y_keys: Sequence[object],
    title: str,
    x_label: str,
    y_label: str,
    out_path: Path,
    *,
    x_formatter=str,
    y_formatter=str,
) -> None:
    width, height = 1100, 840
    left, right, top, bottom = 150, 110, 80, 130
    plot_w, plot_h = width - left - right, height - top - bottom
    cell_w = plot_w / max(len(x_keys), 1)
    cell_h = plot_h / max(len(y_keys), 1)
    values = list(matrix.values())
    vmin, vmax = min(values), max(values)

    svg: List[str] = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')
    svg.append(f'<text x="{width/2:.2f}" y="40" font-size="30" text-anchor="middle" fill="#111">{svg_escape(title)}</text>')

    for yi, y_key in enumerate(y_keys):
        for xi, x_key in enumerate(x_keys):
            x = left + xi * cell_w
            y = top + yi * cell_h
            value = matrix.get((x_key, y_key))
            fill = "#f3f4f6" if value is None else color_for_value(value, vmin, vmax)
            svg.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w:.2f}" height="{cell_h:.2f}" fill="{fill}" stroke="white" stroke-width="1.5"/>')
            if value is not None:
                svg.append(f'<text x="{x + cell_w/2:.2f}" y="{y + cell_h/2 + 7:.2f}" font-size="16" text-anchor="middle" fill="#111">{value:.3f}</text>')

    svg.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#111" stroke-width="2"/>')
    for xi, x_key in enumerate(x_keys):
        x = left + (xi + 0.5) * cell_w
        svg.append(f'<text x="{x:.2f}" y="{top + plot_h + 34}" font-size="18" text-anchor="middle" fill="#222">{svg_escape(x_formatter(x_key))}</text>')
    for yi, y_key in enumerate(y_keys):
        y = top + (yi + 0.5) * cell_h + 6
        svg.append(f'<text x="{left - 12}" y="{y:.2f}" font-size="18" text-anchor="end" fill="#222">{svg_escape(y_formatter(y_key))}</text>')

    svg.append(f'<text x="{left + plot_w/2:.2f}" y="{height - 28}" font-size="26" text-anchor="middle" fill="#111">{svg_escape(x_label)}</text>')
    svg.append(f'<text x="34" y="{top + plot_h/2:.2f}" font-size="26" text-anchor="middle" fill="#111" transform="rotate(-90 34 {top + plot_h/2:.2f})">{svg_escape(y_label)}</text>')
    write_svg(out_path, "\n".join(svg + ["</svg>"]))


def aggregate_heatmap(rows: Sequence[Dict[str, object]], x_key: str, y_key: str) -> Tuple[Dict[Tuple[object, object], float], List[object], List[object]]:
    grouped: Dict[Tuple[object, object], List[float]] = defaultdict(list)
    x_keys, y_keys = set(), set()
    for row in rows:
        x_val = row[x_key]
        y_val = row[y_key]
        grouped[(x_val, y_val)].append(float(row["bound_mass_fraction"]))
        x_keys.add(x_val)
        y_keys.add(y_val)
    matrix = {(x, y): median(values) for (x, y), values in grouped.items()}
    return matrix, sorted(x_keys), sorted(y_keys)


def panel_heatmap_svg(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    spins_present = [spin for spin in SPIN_ORDER if any(row["spin_orientation"] == spin for row in rows)]
    width, height = 1500, 980
    cols = 2
    rows_n = math.ceil(len(spins_present) / cols)
    panel_w, panel_h = 560, 320
    margin_left, margin_top, gap_x, gap_y = 110, 100, 70, 90

    velocities = sorted({row["velocity"] for row in rows})
    periapses = sorted({row["periapsis"] for row in rows})

    grouped: Dict[str, Dict[Tuple[float, float], List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["spin_orientation"])][(float(row["velocity"]), float(row["periapsis"]))].append(float(row["bound_mass_fraction"]))

    all_values = [median(vals) for spin in grouped.values() for vals in spin.values()]
    vmin, vmax = min(all_values), max(all_values)

    svg: List[str] = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')
    svg.append(f'<text x="{width/2:.2f}" y="42" font-size="30" text-anchor="middle" fill="#111">Bound Mass Fraction vs Periapsis vs Velocity by Spin</text>')

    for idx, spin in enumerate(spins_present):
        row_idx, col_idx = divmod(idx, cols)
        left = margin_left + col_idx * (panel_w + gap_x)
        top = margin_top + row_idx * (panel_h + gap_y)
        cell_w = panel_w / len(periapses)
        cell_h = panel_h / len(velocities)

        svg.append(f'<text x="{left + panel_w/2:.2f}" y="{top - 14}" font-size="22" text-anchor="middle" fill="#111">{svg_escape(SPIN_LABELS[spin])}</text>')
        for yi, vel in enumerate(velocities):
            for xi, peri in enumerate(periapses):
                x = left + xi * cell_w
                y = top + yi * cell_h
                values = grouped[spin].get((vel, peri))
                fill = "#f3f4f6" if not values else color_for_value(median(values), vmin, vmax)
                svg.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w:.2f}" height="{cell_h:.2f}" fill="{fill}" stroke="white" stroke-width="1.2"/>')
                if values:
                    svg.append(f'<text x="{x + cell_w/2:.2f}" y="{y + cell_h/2 + 6:.2f}" font-size="14" text-anchor="middle" fill="#111">{median(values):.3f}</text>')
        svg.append(f'<rect x="{left}" y="{top}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#111" stroke-width="1.8"/>')
        for xi, peri in enumerate(periapses):
            x = left + (xi + 0.5) * cell_w
            svg.append(f'<text x="{x:.2f}" y="{top + panel_h + 24}" font-size="14" text-anchor="middle" fill="#222">{peri:g}</text>')
        for yi, vel in enumerate(velocities):
            y = top + (yi + 0.5) * cell_h + 5
            svg.append(f'<text x="{left - 10}" y="{y:.2f}" font-size="14" text-anchor="end" fill="#222">{vel:g}</text>')

    svg.append(f'<text x="{margin_left + panel_w:.2f}" y="{height - 22}" font-size="24" text-anchor="middle" fill="#111">Periapsis (R♂)</text>')
    svg.append(f'<text x="34" y="{margin_top + panel_h:.2f}" font-size="24" text-anchor="middle" fill="#111" transform="rotate(-90 34 {margin_top + panel_h:.2f})">Velocity at infinity (km/s)</text>')
    write_svg(out_path, "\n".join(svg + ["</svg>"]))


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.source))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    peri_data = grouped_median(rows, lambda row: float(row["periapsis"]))
    spin_data = grouped_median(rows, lambda row: str(row["spin_orientation"]))
    vel_data = grouped_median(rows, lambda row: float(row["velocity"]))

    line_plot_svg(
        peri_data,
        "Bound Mass Fraction vs Periapsis",
        "Periapsis (R♂)",
        output_dir / "bound_mass_vs_periapsis.svg",
        color="#1f4ed8",
    )
    bar_plot_svg(
        spin_data,
        "Bound Mass Fraction vs Spin",
        "Spin orientation",
        output_dir / "bound_mass_vs_spin.svg",
    )
    line_plot_svg(
        vel_data,
        "Bound Mass Fraction vs Velocity",
        "Velocity at infinity (km/s)",
        output_dir / "bound_mass_vs_velocity.svg",
        color="#2e8b57",
    )

    matrix, x_keys, y_keys = aggregate_heatmap(rows, "spin_orientation", "periapsis")
    heatmap_svg(
        matrix,
        x_keys,
        y_keys,
        "Bound Mass Fraction vs Periapsis vs Spin",
        "Spin orientation",
        "Periapsis (R♂)",
        output_dir / "bound_mass_vs_periapsis_vs_spin.svg",
        x_formatter=lambda key: SPIN_LABELS.get(str(key), str(key)),
        y_formatter=lambda key: f"{key:g}",
    )

    matrix, x_keys, y_keys = aggregate_heatmap(rows, "velocity", "spin_orientation")
    heatmap_svg(
        matrix,
        x_keys,
        y_keys,
        "Bound Mass Fraction vs Spin vs Velocity",
        "Velocity at infinity (km/s)",
        "Spin orientation",
        output_dir / "bound_mass_vs_spin_vs_velocity.svg",
        x_formatter=lambda key: f"{key:g}",
        y_formatter=lambda key: SPIN_LABELS.get(str(key), str(key)),
    )

    matrix, x_keys, y_keys = aggregate_heatmap(rows, "velocity", "periapsis")
    heatmap_svg(
        matrix,
        x_keys,
        y_keys,
        "Bound Mass Fraction vs Periapsis vs Velocity",
        "Velocity at infinity (km/s)",
        "Periapsis (R♂)",
        output_dir / "bound_mass_vs_periapsis_vs_velocity.svg",
        x_formatter=lambda key: f"{key:g}",
        y_formatter=lambda key: f"{key:g}",
    )

    panel_heatmap_svg(rows, output_dir / "bound_mass_vs_periapsis_vs_velocity_vs_spin.svg")

    print(f"Created plots in {output_dir}")


if __name__ == "__main__":
    main()
