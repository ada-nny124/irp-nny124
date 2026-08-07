#!/usr/bin/env python3
"""Create line-graph variations of bound mass vs periapsis / velocity / spin."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


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
VELOCITY_COLORS = {
    0.0: "#1f4ed8",
    0.2: "#2e8b57",
    0.4: "#b6a400",
    0.6: "#f39c12",
    0.8: "#e24a33",
    1.0: "#8e44ad",
}
VELOCITY_LINESTYLES = {
    0.0: None,
    0.2: "10 4",
    0.4: "10 3 2 3",
    0.6: "3 4",
    0.8: "14 4",
    1.0: "14 4 2 4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-table",
        default="outputs/kegerreis_style_bmf_plotting_table.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/plots",
    )
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "periapsis": float(row["periapsis"]),
                    "v_inf_kms": float(row["v_inf_kms"]),
                    "spin_orientation": row["spin_orientation"],
                    "bound_mass_fraction_median": float(row["bound_mass_fraction_median"]),
                    "row_count": int(row["row_count"]),
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


def radius_for_count(row_count: int) -> float:
    if row_count >= 20:
        return 6.5
    if row_count >= 10:
        return 5.5
    if row_count >= 4:
        return 4.8
    return 4.0


def render_multi_panel(
    panels: Sequence[Tuple[str, List[Tuple[str, str, List[Dict[str, object]]]]]],
    out_path: Path,
    *,
    legend_title: str,
) -> None:
    cols = 2
    rows_n = math.ceil(len(panels) / cols)
    panel_w = 520
    panel_h = 300
    gap_x = 70
    gap_y = 90
    margin_left = 95
    margin_top = 95
    width = margin_left * 2 + cols * panel_w + (cols - 1) * gap_x + 260
    height = margin_top + rows_n * panel_h + (rows_n - 1) * gap_y + 130

    all_points = [
        pt
        for _, lines in panels
        for _, _, pts in lines
        for pt in pts
    ]
    x_values = [float(pt["periapsis"]) for pt in all_points]
    y_values = [float(pt["bound_mass_fraction_median"]) for pt in all_points]
    x_min, x_max = min(x_values) - 0.05, max(x_values) + 0.05
    y_min, y_max = 0.0, max(0.30, max(y_values) * 1.12)

    def x_to_svg(panel_x: float, value: float) -> float:
        return panel_x + (value - x_min) * panel_w / (x_max - x_min if x_max != x_min else 1.0)

    def y_to_svg(panel_y: float, value: float) -> float:
        return panel_y + (y_max - value) * panel_h / (y_max - y_min if y_max != y_min else 1.0)

    svg: List[str] = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')

    for idx, (panel_title, lines) in enumerate(panels):
        row_idx, col_idx = divmod(idx, cols)
        panel_x = margin_left + col_idx * (panel_w + gap_x)
        panel_y = margin_top + row_idx * (panel_h + gap_y)
        svg.append(f'<rect x="{panel_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#111" stroke-width="1.8"/>')
        svg.append(f'<text x="{panel_x + panel_w/2:.2f}" y="{panel_y - 18}" font-size="22" text-anchor="middle" fill="#111">{svg_escape(panel_title)}</text>')

        for tick in [1.1, 1.2, 1.4, 1.6, 1.8, 2.0]:
            if tick < x_min or tick > x_max:
                continue
            x = x_to_svg(panel_x, tick)
            svg.append(f'<line x1="{x:.2f}" y1="{panel_y}" x2="{x:.2f}" y2="{panel_y + panel_h}" stroke="#e6e6e6" stroke-width="1"/>')
            svg.append(f'<text x="{x:.2f}" y="{panel_y + panel_h + 28}" font-size="16" text-anchor="middle" fill="#222">{tick:g}</text>')

        for tick in [0.0, 0.1, 0.2, 0.3]:
            if tick > y_max:
                continue
            y = y_to_svg(panel_y, tick)
            svg.append(f'<line x1="{panel_x}" y1="{y:.2f}" x2="{panel_x + panel_w}" y2="{y:.2f}" stroke="#e6e6e6" stroke-width="1"/>')
            svg.append(f'<text x="{panel_x - 12}" y="{y + 6:.2f}" font-size="16" text-anchor="end" fill="#222">{tick:g}</text>')

        for label, color, pts in lines:
            if len(pts) >= 2:
                dash = VELOCITY_LINESTYLES.get(float(label), None) if legend_title.startswith("Velocity") else None
                extra = f' stroke-dasharray="{dash}"' if dash else ""
                path = " ".join(
                    [f"M {x_to_svg(panel_x, float(pts[0]['periapsis'])):.2f} {y_to_svg(panel_y, float(pts[0]['bound_mass_fraction_median'])):.2f}"]
                    + [
                        f"L {x_to_svg(panel_x, float(pt['periapsis'])):.2f} {y_to_svg(panel_y, float(pt['bound_mass_fraction_median'])):.2f}"
                        for pt in pts[1:]
                    ]
                )
                svg.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.6"{extra}/>')
            for pt in pts:
                x = x_to_svg(panel_x, float(pt["periapsis"]))
                y = y_to_svg(panel_y, float(pt["bound_mass_fraction_median"]))
                r = radius_for_count(int(pt["row_count"]))
                svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{color}" stroke="white" stroke-width="1.2"/>')

    svg.append(f'<text x="{margin_left + panel_w:.2f}" y="{height - 18}" font-size="26" text-anchor="middle" fill="#111">Periapsis (R♂)</text>')
    svg.append(f'<text x="30" y="{margin_top + panel_h:.2f}" font-size="26" text-anchor="middle" fill="#111" transform="rotate(-90 30 {margin_top + panel_h:.2f})">Bound Mass Fraction</text>')

    legend_x = width - 220
    legend_y = 120
    legend_items = []
    if legend_title.startswith("Velocity"):
        for velocity in sorted({float(label) for _, lines in panels for label, _, _ in lines}):
            legend_items.append((f"{velocity:g}", VELOCITY_COLORS.get(velocity, "#666666"), VELOCITY_LINESTYLES.get(velocity)))
    else:
        for spin in [spin for spin in SPIN_ORDER if any(label == spin for _, lines in panels for label, _, _ in lines)]:
            legend_items.append((SPIN_LABELS[spin], SPIN_COLORS[spin], None))

    svg.append(f'<rect x="{legend_x - 18}" y="{legend_y - 34}" width="190" height="{42 * len(legend_items) + 26}" fill="white" stroke="#cccccc" stroke-width="1"/>')
    svg.append(f'<text x="{legend_x}" y="{legend_y - 10}" font-size="22" fill="#111">{svg_escape(legend_title)}</text>')
    for idx, (label, color, dash) in enumerate(legend_items):
        y = legend_y + idx * 34
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{color}" stroke-width="3"{extra}/>')
        svg.append(f'<circle cx="{legend_x + 14}" cy="{y}" r="5" fill="{color}" stroke="white" stroke-width="1"/>')
        svg.append(f'<text x="{legend_x + 42}" y="{y + 7}" font-size="18" fill="#111">{svg_escape(label)}</text>')

    svg.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg), encoding="utf-8")


def make_spin_panels(rows: Sequence[Dict[str, object]]) -> List[Tuple[str, List[Tuple[str, str, List[Dict[str, object]]]]]]:
    grouped: Dict[str, Dict[float, List[Dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["spin_orientation"])][float(row["v_inf_kms"])].append(row)
    panels = []
    for spin in [spin for spin in SPIN_ORDER if spin in grouped]:
        lines = []
        for velocity in sorted(grouped[spin]):
            pts = sorted(grouped[spin][velocity], key=lambda item: float(item["periapsis"]))
            lines.append((str(velocity), VELOCITY_COLORS.get(velocity, "#666666"), pts))
        panels.append((SPIN_LABELS.get(spin, spin), lines))
    return panels


def make_velocity_panels(rows: Sequence[Dict[str, object]]) -> List[Tuple[str, List[Tuple[str, str, List[Dict[str, object]]]]]]:
    grouped: Dict[float, Dict[str, List[Dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[float(row["v_inf_kms"])][str(row["spin_orientation"])].append(row)
    panels = []
    for velocity in sorted(grouped):
        lines = []
        for spin in [spin for spin in SPIN_ORDER if spin in grouped[velocity]]:
            pts = sorted(grouped[velocity][spin], key=lambda item: float(item["periapsis"]))
            lines.append((spin, SPIN_COLORS.get(spin, "#666666"), pts))
        panels.append((f"v∞ = {velocity:g} km s⁻¹", lines))
    return panels


def make_overlay_lines(rows: Sequence[Dict[str, object]]) -> List[Tuple[str, str, List[Dict[str, object]]]]:
    grouped: Dict[Tuple[str, float], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["spin_orientation"]), float(row["v_inf_kms"]))].append(row)
    lines = []
    for (spin, velocity), pts in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        pts_sorted = sorted(pts, key=lambda item: float(item["periapsis"]))
        label = f"{SPIN_LABELS.get(spin, spin)} | v={velocity:g}"
        color = SPIN_COLORS.get(spin, "#666666")
        lines.append((label, color, pts_sorted))
    return [("All spin / velocity combinations", lines)]


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.plot_table))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spin_panels = make_spin_panels(rows)
    render_multi_panel(
        spin_panels,
        out_dir / "bound_mass_vs_periapsis_velocity_by_spin_lines.svg",
        legend_title="Velocity (km s⁻¹)",
    )

    velocity_panels = make_velocity_panels(rows)
    render_multi_panel(
        velocity_panels,
        out_dir / "bound_mass_vs_periapsis_spin_by_velocity_lines.svg",
        legend_title="Spin",
    )

    overlay_panel = make_overlay_lines(rows)
    render_multi_panel(
        overlay_panel,
        out_dir / "bound_mass_vs_periapsis_velocity_spin_overlay_lines.svg",
        legend_title="Spin",
    )

    print(f"Created line-graph variations in {out_dir}")


if __name__ == "__main__":
    main()
