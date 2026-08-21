#!/usr/bin/env python3
"""Build a matched-family spin figure directly from the bound-outcomes table.

The figure has three parts:

1. A family-level spread plot:
   x = periapsis
   y = max(BMF across spins) - min(BMF across spins)
   point colour = velocity
   point size = number of available spin states in that matched family

2. A low-periapsis panel showing raw BMF across the common spin trio
   (`no spin`, `3h z`, `4.7h z`) while holding the rest of the setup fixed.

3. A higher-periapsis panel with the same common spin trio.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


COMMON_SPIN_TRIO = ("no spin", "3h z", "4.7h z")
DEFAULT_LOW_PERIAPSIS_MAX = 1.5
DEFAULT_HIGH_PERIAPSIS_MIN = 1.6

VELOCITY_COLORS = {
    0.0: "#1f4e79",
    0.2: "#2a6f97",
    0.4: "#1b998b",
    0.6: "#c77d00",
    0.8: "#d95f02",
    1.0: "#b23a48",
    1.2: "#8e5ea2",
    1.4: "#6a4c93",
    1.6: "#7a3e2b",
    2.0: "#4d4d4d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bound-table",
        default="extraction-outputs/tables/bound_outcomes.csv",
        help="Path to the extracted bound-outcomes CSV.",
    )
    parser.add_argument(
        "--svg-out",
        default="eda/plots/spin_argument_matched_families.svg",
        help="Output SVG path.",
    )
    parser.add_argument(
        "--summary-out",
        default="eda/tables/spin_argument_matched_families.csv",
        help="Output CSV summary path.",
    )
    parser.add_argument(
        "--low-periapsis-max",
        type=float,
        default=DEFAULT_LOW_PERIAPSIS_MAX,
        help="Upper periapsis bound for the low-periapsis common-trio panel.",
    )
    parser.add_argument(
        "--high-periapsis-min",
        type=float,
        default=DEFAULT_HIGH_PERIAPSIS_MIN,
        help="Lower periapsis bound for the high-periapsis common-trio panel.",
    )
    return parser.parse_args()


def parse_spin_label(spin_code: str) -> str:
    if not spin_code:
        return "no spin"
    period_hr = float(spin_code[1:4]) / 10.0
    axis = spin_code[4:] or "none"
    return f"{period_hr:g}h {axis}"


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        row["periapsis_Rm"] = f"{float(row['periapsis_code'][1:]) / 10.0:.1f}"
        row["v_inf_kms"] = f"{float(row['velocity_code'][1:]) / 10.0:.1f}"
        row["spin_label"] = parse_spin_label(row["spin_code"])
        row["bmf"] = row["bound_mass_fraction"]
    return rows


def build_family_records(rows: Iterable[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, ...], Dict[str, object]] = {}

    for row in rows:
        family_key = (
            row["mass_code"],
            row["velocity_code"],
            row["periapsis_code"],
            row["resolution_code"],
            row["timestep"],
            row["fof_linking_length"],
        )
        record = grouped.setdefault(
            family_key,
            {
                "family_key": family_key,
                "mass_code": row["mass_code"],
                "velocity_code": row["velocity_code"],
                "v_inf_kms": float(row["v_inf_kms"]),
                "periapsis_code": row["periapsis_code"],
                "periapsis_Rm": float(row["periapsis_Rm"]),
                "resolution_code": row["resolution_code"],
                "timestep": row["timestep"],
                "fof_linking_length": row["fof_linking_length"],
                "spin_to_bmf": {},
                "physical_files": set(),
            },
        )
        spin_to_bmf = record["spin_to_bmf"]
        assert isinstance(spin_to_bmf, dict)
        spin_to_bmf[row["spin_label"]] = float(row["bmf"])
        physical_files = record["physical_files"]
        assert isinstance(physical_files, set)
        physical_files.add(row["physical_file"])

    families: List[Dict[str, object]] = []
    for index, record in enumerate(grouped.values(), start=1):
        spin_to_bmf = record["spin_to_bmf"]
        assert isinstance(spin_to_bmf, dict)
        physical_files = record["physical_files"]
        assert isinstance(physical_files, set)
        if len(spin_to_bmf) < 2 or len(physical_files) < 2:
            continue
        delta_bmf = max(spin_to_bmf.values()) - min(spin_to_bmf.values())
        record["family_id"] = f"F{index:03d}"
        record["delta_bmf_spin"] = delta_bmf
        record["n_spin_states"] = len(spin_to_bmf)
        record["spin_labels"] = sorted(spin_to_bmf)
        record["physical_files"] = sorted(physical_files)
        families.append(record)
    families.sort(key=lambda item: (float(item["periapsis_Rm"]), float(item["v_inf_kms"]), str(item["mass_code"])))
    return families


def trio_subset(
    families: Iterable[Dict[str, object]],
    *,
    low_periapsis_max: float,
    high_periapsis_min: float,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    low: List[Dict[str, object]] = []
    high: List[Dict[str, object]] = []
    for family in families:
        spin_to_bmf = family["spin_to_bmf"]
        assert isinstance(spin_to_bmf, dict)
        if not set(COMMON_SPIN_TRIO).issubset(spin_to_bmf):
            continue
        periapsis = float(family["periapsis_Rm"])
        if periapsis <= low_periapsis_max:
            low.append(family)
        if periapsis >= high_periapsis_min:
            high.append(family)
    return low, high


def write_summary_csv(
    families: Iterable[Dict[str, object]],
    out_path: Path,
    *,
    low_periapsis_max: float,
    high_periapsis_min: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "family_id",
            "mass_code",
            "periapsis_Rm",
            "v_inf_kms",
            "resolution_code",
            "timestep",
            "fof_linking_length",
            "n_spin_states",
            "delta_bmf_spin",
            "spin_labels",
            "common_trio_complete",
            "low_periapsis_panel",
            "high_periapsis_panel",
            "physical_files",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for family in families:
            spin_to_bmf = family["spin_to_bmf"]
            assert isinstance(spin_to_bmf, dict)
            writer.writerow(
                {
                    "family_id": family["family_id"],
                    "mass_code": family["mass_code"],
                    "periapsis_Rm": f"{float(family['periapsis_Rm']):.1f}",
                    "v_inf_kms": f"{float(family['v_inf_kms']):.1f}",
                    "resolution_code": family["resolution_code"],
                    "timestep": family["timestep"],
                    "fof_linking_length": family["fof_linking_length"],
                    "n_spin_states": int(family["n_spin_states"]),
                    "delta_bmf_spin": f"{float(family['delta_bmf_spin']):.6f}",
                    "spin_labels": "; ".join(sorted(spin_to_bmf)),
                    "common_trio_complete": int(set(COMMON_SPIN_TRIO).issubset(spin_to_bmf)),
                    "low_periapsis_panel": int(
                        set(COMMON_SPIN_TRIO).issubset(spin_to_bmf) and float(family["periapsis_Rm"]) <= low_periapsis_max
                    ),
                    "high_periapsis_panel": int(
                        set(COMMON_SPIN_TRIO).issubset(spin_to_bmf) and float(family["periapsis_Rm"]) >= high_periapsis_min
                    ),
                    "physical_files": "; ".join(family["physical_files"]),
                }
            )


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    return f"rgba({red},{green},{blue},{alpha:.3f})"


def circle_radius(n_spin_states: int) -> float:
    return 4.0 + min(8.0, math.sqrt(max(0, n_spin_states - 1)) * 2.2)


def velocity_color(velocity: float) -> str:
    return VELOCITY_COLORS.get(velocity, "#5c677d")


def write_svg(
    families: List[Dict[str, object]],
    low_trio: List[Dict[str, object]],
    high_trio: List[Dict[str, object]],
    out_path: Path,
    *,
    low_periapsis_max: float,
    high_periapsis_min: float,
) -> None:
    if not families:
        raise ValueError("No matched multi-spin families found in the bound outcomes table.")

    width = 1560
    height = 1180
    margin_left = 90
    margin_right = 50
    margin_top = 92
    spread_top = 150
    spread_height = 470
    trio_top = 745
    trio_height = 310
    gutter = 60
    panel_width = (width - margin_left - margin_right - gutter) / 2

    peri_values = [float(family["periapsis_Rm"]) for family in families]
    delta_values = [float(family["delta_bmf_spin"]) for family in families]
    bmf_values = [
        bmf
        for family in [*low_trio, *high_trio]
        for bmf in family["spin_to_bmf"].values()
        if isinstance(family["spin_to_bmf"], dict)
    ]

    peri_min = min(peri_values) - 0.08
    peri_max = max(peri_values) + 0.08
    delta_max = max(0.30, max(delta_values) * 1.10)
    bmf_min = 0.0
    bmf_max = max(0.30, max(bmf_values) * 1.10 if bmf_values else 0.30)

    def spread_x(value: float) -> float:
        return margin_left + (value - peri_min) * (width - margin_left - margin_right) / (peri_max - peri_min)

    def spread_y(value: float) -> float:
        return spread_top + (delta_max - value) * spread_height / delta_max

    def trio_x(panel_index: int, spin_index: int) -> float:
        panel_left = margin_left + panel_index * (panel_width + gutter)
        return panel_left + (spin_index + 0.5) * panel_width / len(COMMON_SPIN_TRIO)

    def trio_y(value: float) -> float:
        return trio_top + (bmf_max - value) * trio_height / (bmf_max - bmf_min)

    svg: List[str] = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fcfcfb"/>')

    svg.append(
        f'<text x="{width / 2:.1f}" y="44" text-anchor="middle" font-size="30" fill="#111">'
        "Matched-Family Spin Effect On Bound Mass Fraction"
        "</text>"
    )
    subtitle = (
        "Matched families hold mass, velocity, periapsis, resolution, timestep, and FoF setting fixed. "
        "Only spin changes."
    )
    svg.append(
        f'<text x="{width / 2:.1f}" y="72" text-anchor="middle" font-size="18" fill="#555">{svg_escape(subtitle)}</text>'
    )

    for boundary, label in (
        (low_periapsis_max, f"low panel max = {low_periapsis_max:.1f}"),
        (high_periapsis_min, f"high panel min = {high_periapsis_min:.1f}"),
    ):
        x = spread_x(boundary)
        svg.append(f'<line x1="{x:.2f}" y1="{spread_top}" x2="{x:.2f}" y2="{spread_top + spread_height}" stroke="#d7d7d7" stroke-dasharray="8 6" stroke-width="2"/>')
        svg.append(f'<text x="{x + 6:.2f}" y="{spread_top + 24}" font-size="16" fill="#666">{svg_escape(label)}</text>')

    for tick in sorted({round(value, 1) for value in peri_values}):
        x = spread_x(tick)
        svg.append(f'<line x1="{x:.2f}" y1="{spread_top}" x2="{x:.2f}" y2="{spread_top + spread_height}" stroke="#ececec" stroke-width="1"/>')
        svg.append(f'<text x="{x:.2f}" y="{spread_top + spread_height + 28}" text-anchor="middle" font-size="18" fill="#222">{tick:.1f}</text>')

    for tick in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        if tick > delta_max + 1e-9:
            continue
        y = spread_y(tick)
        svg.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#ececec" stroke-width="1"/>')
        svg.append(f'<text x="{margin_left - 12}" y="{y + 6:.2f}" text-anchor="end" font-size="18" fill="#222">{tick:.2f}</text>')

    svg.append(
        f'<rect x="{margin_left}" y="{spread_top}" width="{width - margin_left - margin_right}" height="{spread_height}" fill="none" stroke="#111" stroke-width="2"/>'
    )
    svg.append(
        f'<text x="{width / 2:.1f}" y="{spread_top + spread_height + 66}" text-anchor="middle" font-size="24" fill="#111">Periapsis (R_Mars)</text>'
    )
    spread_axis_y = spread_top + spread_height / 2
    svg.append(
        f'<text x="30" y="{spread_axis_y:.2f}" text-anchor="middle" font-size="22" fill="#111" transform="rotate(-90 30 {spread_axis_y:.2f})">'
        "ΔBMF_spin = max(BMF across spins) - min(BMF across spins)"
        "</text>"
    )

    for family in families:
        periapsis = float(family["periapsis_Rm"])
        delta_bmf = float(family["delta_bmf_spin"])
        velocity = float(family["v_inf_kms"])
        n_spin_states = int(family["n_spin_states"])
        x = spread_x(periapsis)
        y = spread_y(delta_bmf)
        color = velocity_color(velocity)
        radius = circle_radius(n_spin_states)
        svg.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{rgba(color, 0.70)}" stroke="{color}" stroke-width="1.6">'
            f'<title>{svg_escape(str(family["family_id"]))}: peri={periapsis:.1f}, v={velocity:.1f}, spins={n_spin_states}, ΔBMF={delta_bmf:.3f}</title>'
            "</circle>"
        )

    svg.append(
        f'<text x="{margin_left}" y="{spread_top - 18}" font-size="24" fill="#111">A. Spread across all matched spin families ({len(families)} families)</text>'
    )

    for panel_index, panel_families in enumerate([low_trio, high_trio]):
        panel_left = margin_left + panel_index * (panel_width + gutter)
        panel_title = (
            f"B. Low-periapsis common spin trio (periapsis ≤ {low_periapsis_max:.1f}; {len(low_trio)} families)"
            if panel_index == 0
            else f"C. Higher-periapsis common spin trio (periapsis ≥ {high_periapsis_min:.1f}; {len(high_trio)} families)"
        )
        svg.append(f'<text x="{panel_left}" y="{trio_top - 24}" font-size="22" fill="#111">{svg_escape(panel_title)}</text>')

        for tick in [0.0, 0.1, 0.2, 0.3]:
            if tick > bmf_max + 1e-9:
                continue
            y = trio_y(tick)
            svg.append(f'<line x1="{panel_left}" y1="{y:.2f}" x2="{panel_left + panel_width}" y2="{y:.2f}" stroke="#ececec" stroke-width="1"/>')
            if panel_index == 0:
                svg.append(f'<text x="{panel_left - 12}" y="{y + 6:.2f}" text-anchor="end" font-size="18" fill="#222">{tick:.1f}</text>')

        for spin_index, spin_label in enumerate(COMMON_SPIN_TRIO):
            x = trio_x(panel_index, spin_index)
            svg.append(f'<line x1="{x:.2f}" y1="{trio_top}" x2="{x:.2f}" y2="{trio_top + trio_height}" stroke="#ececec" stroke-width="1"/>')
            svg.append(f'<text x="{x:.2f}" y="{trio_top + trio_height + 28}" text-anchor="middle" font-size="18" fill="#222">{svg_escape(spin_label)}</text>')

        svg.append(f'<rect x="{panel_left}" y="{trio_top}" width="{panel_width}" height="{trio_height}" fill="none" stroke="#111" stroke-width="2"/>')

        for family in panel_families:
            spin_to_bmf = family["spin_to_bmf"]
            assert isinstance(spin_to_bmf, dict)
            velocity = float(family["v_inf_kms"])
            color = velocity_color(velocity)
            points = []
            for spin_index, spin_label in enumerate(COMMON_SPIN_TRIO):
                x = trio_x(panel_index, spin_index)
                y = trio_y(float(spin_to_bmf[spin_label]))
                points.append((x, y))
            path = " ".join(
                [f"M {points[0][0]:.2f} {points[0][1]:.2f}"] + [f"L {x:.2f} {y:.2f}" for x, y in points[1:]]
            )
            svg.append(f'<path d="{path}" fill="none" stroke="{rgba(color, 0.45)}" stroke-width="2.2"/>')
            for x, y in points:
                svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.8" fill="{color}" stroke="white" stroke-width="1.2"/>')

    trio_axis_x = margin_left + panel_width + gutter / 2
    svg.append(f'<text x="{trio_axis_x:.2f}" y="{height - 34}" text-anchor="middle" font-size="24" fill="#111">Spin state</text>')
    trio_axis_y = trio_top + trio_height / 2
    svg.append(
        f'<text x="30" y="{trio_axis_y:.2f}" text-anchor="middle" font-size="22" fill="#111" transform="rotate(-90 30 {trio_axis_y:.2f})">Bound mass fraction</text>'
    )

    legend_x = width - 280
    legend_y = 160
    legend_height = 220
    svg.append(f'<rect x="{legend_x}" y="{legend_y}" width="230" height="{legend_height}" fill="white" stroke="#cfcfcf" stroke-width="1"/>')
    svg.append(f'<text x="{legend_x + 16}" y="{legend_y + 28}" font-size="20" fill="#111">Velocity colour (km s^-1)</text>')
    for index, velocity in enumerate(sorted({float(family["v_inf_kms"]) for family in families})):
        y = legend_y + 58 + index * 24
        color = velocity_color(velocity)
        svg.append(f'<line x1="{legend_x + 16}" y1="{y}" x2="{legend_x + 42}" y2="{y}" stroke="{color}" stroke-width="4"/>')
        svg.append(f'<circle cx="{legend_x + 29}" cy="{y}" r="5" fill="{color}" stroke="white" stroke-width="1"/>')
        svg.append(f'<text x="{legend_x + 54}" y="{y + 6}" font-size="18" fill="#111">{velocity:.1f}</text>')

    size_legend_y = legend_y + legend_height - 46
    svg.append(f'<text x="{legend_x + 16}" y="{size_legend_y - 14}" font-size="18" fill="#111">Point size = number of spin states</text>')
    for offset, count in enumerate([2, 3, 6, 16]):
        x = legend_x + 38 + offset * 46
        radius = circle_radius(count)
        svg.append(f'<circle cx="{x:.2f}" cy="{size_legend_y:.2f}" r="{radius:.2f}" fill="{rgba("#5c677d", 0.40)}" stroke="#5c677d" stroke-width="1.2"/>')
        svg.append(f'<text x="{x:.2f}" y="{size_legend_y + 28:.2f}" text-anchor="middle" font-size="16" fill="#333">{count}</text>')

    note = (
        "Panels B and C use only families with the shared trio "
        "`no spin`, `3h z`, `4.7h z`; Panel A uses every matched family with at least two spin states."
    )
    svg.append(f'<text x="{margin_left}" y="{height - 12}" font-size="16" fill="#555">{svg_escape(note)}</text>')

    svg.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.bound_table))
    families = build_family_records(rows)
    low_trio, high_trio = trio_subset(
        families,
        low_periapsis_max=args.low_periapsis_max,
        high_periapsis_min=args.high_periapsis_min,
    )
    write_summary_csv(
        families,
        Path(args.summary_out),
        low_periapsis_max=args.low_periapsis_max,
        high_periapsis_min=args.high_periapsis_min,
    )
    write_svg(
        families,
        low_trio,
        high_trio,
        Path(args.svg_out),
        low_periapsis_max=args.low_periapsis_max,
        high_periapsis_min=args.high_periapsis_min,
    )

    print(f"Matched families: {len(families)}")
    print(f"Low-periapsis common-trio families: {len(low_trio)}")
    print(f"Higher-periapsis common-trio families: {len(high_trio)}")
    print(f"Summary CSV: {args.summary_out}")
    print(f"Figure SVG: {args.svg_out}")


if __name__ == "__main__":
    main()
