#!/usr/bin/env python3
"""Create a balanced/resampled linear bound-mass-vs-periapsis plot.

Method:
1. Collapse raw rows within full physical strata so repeated analysis variants do not dominate.
2. Re-aggregate each (periapsis, velocity, spin_orientation) cell giving each stratum equal weight.
3. Render a linear SVG similar to the current linear figure, but save as a new file.
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
    parser.add_argument("--svg-out", default="outputs/plots/bound_mass_fraction_vs_periapsis_linear_resampled.svg")
    parser.add_argument("--csv-out", default="outputs/bound_mass_fraction_vs_periapsis_linear_resampled.csv")
    parser.add_argument("--high-periapsis-threshold", type=float, default=1.6)
    parser.add_argument("--high-periapsis-boost", type=int, default=1)
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
            if periapsis is None or velocity is None:
                continue
            rows.append(
                {
                    "bound_mass_fraction": float(row["bound_mass_fraction"]),
                    "periapsis": periapsis,
                    "velocity": velocity,
                    "spin_orientation": infer_spin_orientation(str(row.get("spin_code", ""))),
                    "spin_code": str(row.get("spin_code", "")),
                    "mass_code": str(row.get("mass_code", "")),
                    "resolution_code": str(row.get("resolution_code", "")),
                    "fof_linking_length": str(row.get("fof_linking_length", "")),
                }
            )
    return rows


def balanced_table(
    rows: List[Dict[str, object]],
    *,
    high_periapsis_threshold: float,
    high_periapsis_boost: int,
) -> List[Dict[str, object]]:
    stage1: Dict[Tuple[object, ...], List[float]] = defaultdict(list)
    for row in rows:
        key = (
            row["periapsis"],
            row["velocity"],
            row["spin_orientation"],
            row["spin_code"],
            row["mass_code"],
            row["resolution_code"],
            row["fof_linking_length"],
        )
        stage1[key].append(float(row["bound_mass_fraction"]))

    strata_rows: List[Dict[str, object]] = []
    for key, values in stage1.items():
        strata_rows.append(
            {
                "periapsis": key[0],
                "velocity": key[1],
                "spin_orientation": key[2],
                "stratum_median_bmf": median(values),
            }
        )

    stage2: Dict[Tuple[object, ...], List[float]] = defaultdict(list)
    for row in strata_rows:
        key = (row["periapsis"], row["velocity"], row["spin_orientation"])
        value = float(row["stratum_median_bmf"])
        stage2[key].append(value)
        if float(row["periapsis"]) >= high_periapsis_threshold and high_periapsis_boost > 1:
            for _ in range(high_periapsis_boost - 1):
                stage2[key].append(value)

    out: List[Dict[str, object]] = []
    for key, values in sorted(stage2.items()):
        periapsis, velocity, spin_orientation = key
        bmf = median(values)
        if periapsis > 2.0:
            continue
        if bmf <= 0.0:
            continue
        out.append(
            {
                "periapsis": periapsis,
                "v_inf_kms": velocity,
                "spin_orientation": spin_orientation,
                "bound_mass_fraction_median": bmf,
                "stratum_count": len(values),
            }
        )
    # Keep only groups with >=2 periapsis points.
    by_group: Dict[Tuple[float, str], List[Dict[str, object]]] = defaultdict(list)
    for row in out:
        by_group[(float(row["v_inf_kms"]), str(row["spin_orientation"]))].append(row)
    filtered: List[Dict[str, object]] = []
    for key, members in by_group.items():
        if len(members) < 2:
            continue
        filtered.extend(sorted(members, key=lambda item: float(item["periapsis"])))
    # Drop the same low blue outlier.
    filtered = [
        row
        for row in filtered
        if not (
            float(row["v_inf_kms"]) == 0.0
            and str(row["spin_orientation"]) == "no_spin"
            and float(row["bound_mass_fraction_median"]) < 1e-3
        )
    ]
    return filtered


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
                "stratum_count",
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
    svg.append(f'<text x="{width/2:.2f}" y="34" font-size="30" text-anchor="middle" fill="#111">Bound Mass Fraction vs Periapsis (Balanced Aggregation)</text>')
    svg.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#111" stroke-width="2"/>')

    x_ticks = [1.1, 1.2, 1.4, 1.6, 1.8, 2.0]
    for tick in x_ticks:
        if tick < PLOT_X_MIN or tick > PLOT_X_MAX:
            continue
        x = x_to_svg(tick)
        svg.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#e5e5e5" stroke-width="1"/>')
        svg.append(f'<text x="{x:.2f}" y="{top + plot_h + 36}" font-size="20" text-anchor="middle" fill="#222">{tick:g}</text>')
    for tick in [0.0, 0.1, 0.2, 0.3]:
        y = y_to_svg(tick)
        svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e5e5" stroke-width="1"/>')
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
            svg.append(f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2.6"{extra}/>')
        for pt in pts:
            x = x_to_svg(float(pt["periapsis"]))
            y = y_to_svg(float(pt["bound_mass_fraction_median"]))
            n = int(pt["stratum_count"])
            r = 4.5 if n < 4 else 5.5 if n < 10 else 6.5
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

    note = "Balanced aggregation: each physical stratum contributes equally within each periapsis/velocity/spin cell."
    svg.append(f'<text x="{left}" y="{height - 54}" font-size="18" text-anchor="start" fill="#555">{svg_escape(note)}</text>')
    svg.append("</svg>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.source))
    balanced_rows = balanced_table(
        rows,
        high_periapsis_threshold=args.high_periapsis_threshold,
        high_periapsis_boost=args.high_periapsis_boost,
    )
    write_csv(Path(args.csv_out), balanced_rows)
    build_svg(balanced_rows, Path(args.svg_out))
    print(f"Balanced CSV: {args.csv_out}")
    print(f"Balanced SVG: {args.svg_out}")
    print(f"High-periapsis threshold: {args.high_periapsis_threshold}")
    print(f"High-periapsis boost: {args.high_periapsis_boost}")
    print("Ways to make sampling more even:")
    print("- Prospectively: run more simulations in sparse periapsis/velocity/spin/mass cells.")
    print("- Analytically: use balanced/stratified aggregation so oversampled strata do not dominate.")


if __name__ == "__main__":
    main()
