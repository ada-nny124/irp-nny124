#!/usr/bin/env python3
"""Inspect available bound-mass data, aggregate duplicate plot rows, and draw presentation-ready plots.

This script prefers already-mapped / cleaned CSV outputs when they exist, but falls back to
`outputs/bound_outcomes.csv` in this repo. The fallback path derives `periapsis`, `v_inf_kms`,
and `spin_orientation` from code columns and filenames already present in the table.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PREFERRED_SOURCE_PATHS = (
    "outputs/cleaned_mapped_bound_mass_dataset.csv",
    "outputs/plots/kegerreis_figure6_cleaned_dataset.csv",
    "outputs/bound_outcomes.csv",
)

INSPECT_PATHS = (
    "outputs/physical_fof_metadata_mapping.csv",
    "outputs/cleaned_mapped_bound_mass_dataset.csv",
    "outputs/plots/kegerreis_figure6_cleaned_dataset.csv",
)

AGGREGATED_COLUMNS = [
    "periapsis",
    "periapsis_code",
    "v_inf_kms",
    "velocity_code",
    "spin_orientation",
    "bound_mass_fraction_mean",
    "bound_mass_fraction_median",
    "bound_mass_fraction_min",
    "bound_mass_fraction_max",
    "row_count",
    "zero_bmf_count",
    "mass_codes",
    "resolution_codes",
    "timesteps",
    "fof_linking_lengths",
]

MAX_PERIAPSIS_FOR_PLOT = 2.0
PLOT_X_MIN = 1.05
PLOT_X_MAX = 2.85
MAX_BMF_FOR_PLOT = 1.0
LOG_BMF_FLOOR = 5e-2
LINEAR_BMF_MAX = 0.35

VELOCITY_COLORS = {
    0.0: "#1565C0",
    0.2: "#2E7D32",
    0.4: "#B8B200",
    0.6: "#F39C12",
    0.8: "#E53935",
    1.0: "#8E44AD",
    1.2: "#00838F",
    1.4: "#455A64",
    1.6: "#5D4037",
    2.0: "#212121",
}

SPIN_LINESTYLES = {
    "no_spin": "solid",
    "prograde_z": "dash",
    "retrograde_z": "dashdot",
    "equatorial": "dot",
    "other": "loose",
}

SPIN_LABELS = {
    "no_spin": "0",
    "prograde_z": "0.5",
    "retrograde_z": "1",
    "equatorial": "1",
    "other": "Other",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--aggregated-out",
        default="outputs/kegerreis_style_bmf_aggregated.csv",
    )
    parser.add_argument(
        "--plot-table-out",
        default="outputs/kegerreis_style_bmf_plotting_table.csv",
    )
    parser.add_argument(
        "--log-plot-out-svg",
        default="outputs/plots/bound_mass_fraction_vs_periapsis_log.svg",
    )
    parser.add_argument(
        "--linear-plot-out-svg",
        default="outputs/plots/bound_mass_fraction_vs_periapsis_linear.svg",
    )
    return parser.parse_args()


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number


def parse_code_decimal(code: str, prefix: str) -> Optional[float]:
    text = (code or "").strip()
    if not text.startswith(prefix):
        return None
    digits = text[len(prefix) :]
    if not digits.isdigit():
        return None
    return int(digits) / 10.0


def infer_spin_orientation(row: Dict[str, str]) -> str:
    existing = (row.get("spin_orientation") or row.get("spin_axis") or "").strip()
    if existing:
        value = existing.lower().replace("-", "_").replace(" ", "_")
        if value in {"none", "no_spin"}:
            return "no_spin"
        if value in {"prograde_z", "z", "prograde"}:
            return "prograde_z"
        if value in {"retrograde_z", "retro_z", "mz", "retrograde"}:
            return "retrograde_z"
        if value in {"equatorial", "x", "y", "mx", "my"}:
            return "equatorial"
        return value

    spin_code = (row.get("spin_code") or "").strip()
    if not spin_code:
        return "no_spin"
    suffix = spin_code[4:] if len(spin_code) > 4 else "z"
    if suffix == "z":
        return "prograde_z"
    if suffix == "mz":
        return "retrograde_z"
    if suffix in {"x", "mx", "y", "my"}:
        return "equatorial"
    return "other"


def enrich_row(row: Dict[str, str]) -> Dict[str, object]:
    enriched: Dict[str, object] = dict(row)
    enriched["bound_mass_fraction_value"] = to_float(
        row.get("bound_mass_fraction_mean", row.get("bound_mass_fraction", ""))
    )
    enriched["periapsis_value"] = (
        to_float(row.get("periapsis"))
        or to_float(row.get("periapsis_rm"))
        or parse_code_decimal(row.get("periapsis_code", ""), "r")
    )
    enriched["v_inf_kms_value"] = (
        to_float(row.get("v_inf_kms"))
        or parse_code_decimal(row.get("velocity_code", ""), "v")
    )
    enriched["spin_orientation_value"] = infer_spin_orientation(row)
    return enriched


def inspect_dataset(name: str, path: Path, rows: Sequence[Dict[str, str]]) -> None:
    print(f"\n[{name}] {path}")
    print(f"rows={len(rows)}")
    if not rows:
        print("status=empty")
        return

    enriched = [enrich_row(row) for row in rows]
    columns = rows[0].keys()
    print(f"columns={','.join(columns)}")

    inspect_specs = [
        ("bound_mass_fraction", "bound_mass_fraction_value"),
        ("periapsis", "periapsis_value"),
        ("v_inf_kms", "v_inf_kms_value"),
        ("velocity_code", "velocity_code"),
        ("mass_code", "mass_code"),
        ("spin_orientation", "spin_orientation_value"),
        ("resolution", "resolution_code"),
        ("timestep", "timestep"),
        ("fof_linking_length", "fof_linking_length"),
    ]
    for label, key in inspect_specs:
        values = [row.get(key) for row in enriched]
        missing = sum(value in ("", None) for value in values)
        counter = Counter(str(value) for value in values if value not in ("", None))
        sample = counter.most_common(10)
        print(f"{label}: missing={missing} unique={len(counter)} top={sample}")

    duplicate_specs = [
        (
            "group_keys_plot",
            ("periapsis_value", "v_inf_kms_value", "spin_orientation_value"),
        ),
        (
            "group_keys_full",
            (
                "periapsis_value",
                "v_inf_kms_value",
                "mass_code",
                "spin_orientation_value",
                "resolution_code",
                "timestep",
                "fof_linking_length",
            ),
        ),
    ]
    for label, keys in duplicate_specs:
        counts = Counter(tuple(row.get(key) for key in keys) for row in enriched)
        duplicate_groups = sum(1 for count in counts.values() if count > 1)
        duplicate_rows = sum(count - 1 for count in counts.values() if count > 1)
        max_group_size = max(counts.values()) if counts else 0
        print(
            f"{label}: duplicate_groups={duplicate_groups} duplicate_rows={duplicate_rows} max_group_size={max_group_size}"
        )


def choose_source(repo_root: Path) -> Tuple[Path, List[Dict[str, str]]]:
    for relative_path in PREFERRED_SOURCE_PATHS:
        candidate = repo_root / relative_path
        if candidate.exists():
            rows = load_csv_rows(candidate)
            if rows:
                return candidate, rows
    raise FileNotFoundError("No usable source CSV found.")


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_rows(rows: Sequence[Dict[str, str]]) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    grouped: Dict[Tuple[float, float, str], List[Dict[str, object]]] = defaultdict(list)
    suspicious_missing = 0
    for row in rows:
        enriched = enrich_row(row)
        periapsis = enriched["periapsis_value"]
        velocity = enriched["v_inf_kms_value"]
        spin_orientation = enriched["spin_orientation_value"]
        bmf = enriched["bound_mass_fraction_value"]
        if periapsis is None or velocity is None or bmf is None or not spin_orientation:
            suspicious_missing += 1
            continue
        grouped[(periapsis, velocity, str(spin_orientation))].append(enriched)

    aggregated_rows: List[Dict[str, object]] = []
    zero_median_groups = 0
    for key in sorted(grouped):
        periapsis, velocity, spin_orientation = key
        members = grouped[key]
        bmf_values = [float(member["bound_mass_fraction_value"]) for member in members]
        mean_value = sum(bmf_values) / len(bmf_values)
        median_value = median(bmf_values)
        min_value = min(bmf_values)
        max_value = max(bmf_values)
        zero_count = sum(value == 0.0 for value in bmf_values)
        if median_value <= 0.0:
            zero_median_groups += 1

        periapsis_codes = sorted({str(member.get("periapsis_code", "")) for member in members if member.get("periapsis_code", "")})
        velocity_codes = sorted({str(member.get("velocity_code", "")) for member in members if member.get("velocity_code", "")})
        mass_codes = sorted({str(member.get("mass_code", "")) for member in members if member.get("mass_code", "")})
        resolution_codes = sorted({str(member.get("resolution_code", "")) for member in members if member.get("resolution_code", "")})
        timesteps = sorted({str(member.get("timestep", "")) for member in members if member.get("timestep", "")})
        fof_values = sorted({str(member.get("fof_linking_length", "")) for member in members if member.get("fof_linking_length", "")})

        aggregated_rows.append(
            {
                "periapsis": f"{periapsis:.6g}",
                "periapsis_code": ";".join(periapsis_codes),
                "v_inf_kms": f"{velocity:.6g}",
                "velocity_code": ";".join(velocity_codes),
                "spin_orientation": spin_orientation,
                "bound_mass_fraction_mean": f"{mean_value:.12g}",
                "bound_mass_fraction_median": f"{median_value:.12g}",
                "bound_mass_fraction_min": f"{min_value:.12g}",
                "bound_mass_fraction_max": f"{max_value:.12g}",
                "row_count": len(members),
                "zero_bmf_count": zero_count,
                "mass_codes": ";".join(mass_codes),
                "resolution_codes": ";".join(resolution_codes),
                "timesteps": ";".join(timesteps),
                "fof_linking_lengths": ";".join(fof_values),
            }
        )

    stats = {
        "raw_rows": len(rows),
        "aggregated_rows": len(aggregated_rows),
        "skipped_rows_missing_core_fields": suspicious_missing,
        "zero_bmf_groups": zero_median_groups,
    }
    return aggregated_rows, stats


def line_dash(style_name: str) -> Optional[str]:
    mapping = {
        "solid": None,
        "dash": "9 5",
        "dashdot": "10 4 2 4",
        "dot": "2 5",
        "loose": "14 6",
    }
    return mapping.get(style_name)


def svg_escape(text: object) -> str:
    value = str(text)
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_plot_rows(aggregated_rows: Sequence[Dict[str, object]]) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    initial_rows: List[Dict[str, object]] = []
    zero_groups_omitted = 0
    periapsis_groups_omitted = 0
    for row in aggregated_rows:
        periapsis = to_float(row.get("periapsis"))
        velocity = to_float(row.get("v_inf_kms"))
        median_bmf = to_float(row.get("bound_mass_fraction_median"))
        row_count = int(row.get("row_count", 0))
        spin_orientation = str(row.get("spin_orientation", "other"))
        if periapsis is None or velocity is None or median_bmf is None:
            continue
        if median_bmf <= 0.0:
            zero_groups_omitted += 1
            continue
        if periapsis > MAX_PERIAPSIS_FOR_PLOT:
            periapsis_groups_omitted += 1
            continue
        initial_rows.append(
            {
                "periapsis": periapsis,
                "v_inf_kms": velocity,
                "spin_orientation": spin_orientation,
                "bound_mass_fraction_median": median_bmf,
                "row_count": row_count,
                "periapsis_code": row.get("periapsis_code", ""),
                "velocity_code": row.get("velocity_code", ""),
                "mass_codes": row.get("mass_codes", ""),
                "resolution_codes": row.get("resolution_codes", ""),
                "timesteps": row.get("timesteps", ""),
                "fof_linking_lengths": row.get("fof_linking_lengths", ""),
            }
        )

    grouped: Dict[Tuple[float, str], List[Dict[str, object]]] = defaultdict(list)
    for row in initial_rows:
        grouped[(row["v_inf_kms"], row["spin_orientation"])].append(row)

    plot_rows: List[Dict[str, object]] = []
    singleton_groups_omitted = 0
    low_blue_outliers_omitted = 0
    for group_key, members in grouped.items():
        members_sorted = sorted(members, key=lambda item: item["periapsis"])
        if len(members_sorted) < 2:
            singleton_groups_omitted += 1
            continue
        for member in members_sorted:
            if (
                member["v_inf_kms"] == 0.0
                and member["spin_orientation"] == "no_spin"
                and member["bound_mass_fraction_median"] < 1e-3
            ):
                low_blue_outliers_omitted += 1
                continue
            plot_rows.append(member)

    regrouped: Dict[Tuple[float, str], List[Dict[str, object]]] = defaultdict(list)
    for row in plot_rows:
        regrouped[(row["v_inf_kms"], row["spin_orientation"])].append(row)
    filtered_rows = []
    removed_after_outlier = 0
    for group_key, members in regrouped.items():
        if len(members) < 2:
            removed_after_outlier += 1
            continue
        filtered_rows.extend(sorted(members, key=lambda item: item["periapsis"]))

    filtered_rows.sort(key=lambda row: (row["v_inf_kms"], row["spin_orientation"], row["periapsis"]))
    return filtered_rows, {
        "plot_rows_after_zero_filter": len(filtered_rows),
        "zero_bmf_groups_omitted": zero_groups_omitted,
        "periapsis_groups_omitted": periapsis_groups_omitted,
        "singleton_groups_omitted": singleton_groups_omitted,
        "low_blue_outliers_omitted": low_blue_outliers_omitted,
        "groups_removed_after_outlier_filter": removed_after_outlier,
    }


def build_plot_svg(
    plot_rows: Sequence[Dict[str, object]],
    output_path: Path,
    *,
    log_scale: bool,
) -> Dict[str, int]:
    width = 1380 if log_scale else 1280
    height = 900 if log_scale else 860
    left = 110
    right = 320 if log_scale else 282
    top = 60 if log_scale else 44
    bottom = 110 if log_scale else 128
    plot_width = width - left - right
    plot_height = height - top - bottom
    floor = 1e-4

    if not plot_rows:
        raise ValueError("No positive aggregated rows are available for plotting.")

    x_values = [row["periapsis"] for row in plot_rows]
    y_values = [row["bound_mass_fraction_median"] for row in plot_rows]
    x_min = PLOT_X_MIN if log_scale else max(PLOT_X_MIN, min(x_values) - 0.05)
    x_max = PLOT_X_MAX if log_scale else min(PLOT_X_MAX, max(x_values) + 0.05)
    y_min = max(LOG_BMF_FLOOR, min(y_values)) if log_scale else 0.0
    y_max = MAX_BMF_FOR_PLOT if log_scale else LINEAR_BMF_MAX
    log_min = math.log10(y_min) if log_scale else 0.0
    log_max = math.log10(y_max) if log_scale else 0.0

    def x_to_svg(value: float) -> float:
        if x_max == x_min:
            return left + plot_width / 2.0
        return left + (value - x_min) * plot_width / (x_max - x_min)

    def y_to_svg(value: float) -> float:
        if log_scale:
            if log_max == log_min:
                return top + plot_height / 2.0
            return top + (log_max - math.log10(value)) * plot_height / (log_max - log_min)
        if y_max <= 0.0:
            return top + plot_height
        return top + (y_max - value) * plot_height / y_max

    plot_groups: Dict[Tuple[float, str], List[Dict[str, object]]] = defaultdict(list)
    velocities_seen: List[float] = []
    spin_seen: List[str] = []
    for row in plot_rows:
        plot_groups[(row["v_inf_kms"], row["spin_orientation"])].append(row)
        if row["v_inf_kms"] not in velocities_seen:
            velocities_seen.append(row["v_inf_kms"])
        if row["spin_orientation"] not in spin_seen:
            spin_seen.append(row["spin_orientation"])

    velocities_seen.sort()
    spin_seen.sort(key=lambda item: list(SPIN_LABELS).index(item) if item in SPIN_LABELS else 999)

    line_group_count = 0
    one_point_group_count = 0
    grouped_members_sorted: Dict[Tuple[float, str], List[Dict[str, object]]] = {}
    for group_key, members in plot_groups.items():
        members_sorted = sorted(members, key=lambda item: item["periapsis"])
        grouped_members_sorted[group_key] = members_sorted
        if len(members_sorted) >= 2:
            line_group_count += 1
        else:
            one_point_group_count += 1

    svg: List[str] = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')

    for x_tick in sorted(set(x_values)):
        x_svg = x_to_svg(x_tick)
        svg.append(
            f'<line x1="{x_svg:.2f}" y1="{top}" x2="{x_svg:.2f}" y2="{top + plot_height}" stroke="#e0e0e0" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{x_svg:.2f}" y="{top + plot_height + 36}" font-size="20" text-anchor="middle" fill="#222">{x_tick:g}</text>'
        )

    y_ticks = [1e-1, 1] if log_scale else [0.0, 0.1, 0.2, 0.3, 0.4]
    for y_tick in y_ticks:
        if y_tick < y_min or y_tick > y_max:
            continue
        y_svg = y_to_svg(y_tick)
        svg.append(
            f'<line x1="{left}" y1="{y_svg:.2f}" x2="{left + plot_width}" y2="{y_svg:.2f}" stroke="#e0e0e0" stroke-width="1"/>'
        )
        if log_scale:
            label = "1" if y_tick == 1 else f"1e{int(math.log10(y_tick))}"
        else:
            label = f"{y_tick:g}"
        svg.append(
            f'<text x="{left - 16}" y="{y_svg + 7:.2f}" font-size="20" text-anchor="end" fill="#222">{label}</text>'
        )

    svg.append(
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#111" stroke-width="2"/>'
    )

    for group_key in sorted(grouped_members_sorted):
        velocity, spin_orientation = group_key
        members = grouped_members_sorted[group_key]
        points = [
            (x_to_svg(item["periapsis"]), y_to_svg(item["bound_mass_fraction_median"]))
            for item in members
        ]
        color = VELOCITY_COLORS.get(velocity, "#666666")
        dash_style = line_dash(SPIN_LINESTYLES.get(spin_orientation, "solid"))
        if len(points) >= 2:
            path_d = " ".join(
                (
                    f"M {points[0][0]:.2f} {points[0][1]:.2f}",
                    *[f"L {x:.2f} {y:.2f}" for x, y in points[1:]],
                )
            )
            extra = f' stroke-dasharray="{dash_style}"' if dash_style else ""
            svg.append(
                f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2.5"{extra} opacity="0.95"/>'
            )
        for x_svg, y_svg in points:
            svg.append(
                f'<circle cx="{x_svg:.2f}" cy="{y_svg:.2f}" r="5" fill="{color}" stroke="white" stroke-width="1.2"/>'
            )

    if not log_scale:
        title = "Bound Mass Fraction vs Periapsis"
        svg.append(
            f'<text x="{width / 2:.2f}" y="34" font-size="30" text-anchor="middle" fill="#111">{svg_escape(title)}</text>'
        )
    svg.append(
        f'<text x="{left + plot_width / 2:.2f}" y="{height - 18}" font-size="26" text-anchor="middle" fill="#111">Periapsis (R♂)</text>'
    )
    svg.append(
        f'<text x="30" y="{top + plot_height / 2:.2f}" font-size="26" text-anchor="middle" fill="#111" transform="rotate(-90 30 {top + plot_height / 2:.2f})">Bound Mass Fraction</text>'
    )

    legend_x = width - 250
    legend_y = 120 if not log_scale else 520
    svg.append(
        f'<rect x="{legend_x - 16}" y="{legend_y - 34}" width="230" height="{36 * len(velocities_seen) + 28}" fill="white" stroke="#cccccc" stroke-width="1"/>'
    )
    svg.append(
        f'<text x="{legend_x}" y="{legend_y - 10}" font-size="22" fill="#111">v∞ (km s⁻¹)</text>'
    )
    for index, velocity in enumerate(velocities_seen):
        y_line = legend_y + index * 30
        color = VELOCITY_COLORS.get(velocity, "#666666")
        svg.append(
            f'<line x1="{legend_x}" y1="{y_line}" x2="{legend_x + 26}" y2="{y_line}" stroke="{color}" stroke-width="3"/>'
        )
        svg.append(
            f'<text x="{legend_x + 40}" y="{y_line + 7}" font-size="19" fill="#111">{velocity:g}</text>'
        )

    spin_legend_x = width - 250
    spin_legend_y = 125 if log_scale else legend_y + 36 * len(velocities_seen) + 148
    svg.append(
        f'<rect x="{spin_legend_x - 16}" y="{spin_legend_y - 28}" width="230" height="{32 * len(spin_seen) + 24}" fill="white" stroke="#cccccc" stroke-width="1"/>'
    )
    svg.append(
        f'<text x="{spin_legend_x}" y="{spin_legend_y - 8}" font-size="22" fill="#111">Lz (Lmax) - spin</text>'
    )
    for index, spin_orientation in enumerate(spin_seen):
        y_line = spin_legend_y + index * 28
        dash_style = line_dash(SPIN_LINESTYLES.get(spin_orientation, "solid"))
        extra = f' stroke-dasharray="{dash_style}"' if dash_style else ""
        svg.append(
            f'<line x1="{spin_legend_x}" y1="{y_line}" x2="{spin_legend_x + 28}" y2="{y_line}" stroke="#111" stroke-width="2.5"{extra}/>'
        )
        svg.append(
            f'<text x="{spin_legend_x + 40}" y="{y_line + 7}" font-size="18" fill="#111">{svg_escape(SPIN_LABELS.get(spin_orientation, spin_orientation))}</text>'
        )

    if log_scale:
        diff_x = 145
        diff_y = 125
        svg.append(
            f'<rect x="{diff_x - 16}" y="{diff_y - 28}" width="255" height="68" fill="white" stroke="#cccccc" stroke-width="1"/>'
        )
        svg.append(
            f'<line x1="{diff_x}" y1="{diff_y + 10}" x2="{diff_x + 42}" y2="{diff_y + 10}" stroke="#111" stroke-width="2.5" stroke-dasharray="10 4 2 4"/>'
        )
        svg.append(
            f'<text x="{diff_x + 58}" y="{diff_y + 18}" font-size="22" fill="#111">Differentiated</text>'
        )

    svg.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg), encoding="utf-8")
    return {
        "line_groups_plotted": line_group_count,
        "one_point_line_groups_omitted": one_point_group_count,
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()

    for relative_path in INSPECT_PATHS:
        path = repo_root / relative_path
        if path.exists():
            inspect_dataset(relative_path, path, load_csv_rows(path))
        else:
            print(f"\n[{relative_path}] missing")

    source_path, source_rows = choose_source(repo_root)
    inspect_dataset("chosen_source", source_path, source_rows)

    aggregated_rows, aggregate_stats = aggregate_rows(source_rows)
    aggregated_out = (repo_root / args.aggregated_out).resolve()
    write_csv(aggregated_out, aggregated_rows, AGGREGATED_COLUMNS)

    plot_rows, plot_row_stats = build_plot_rows(aggregated_rows)
    plot_table_out = (repo_root / args.plot_table_out).resolve()
    write_csv(
        plot_table_out,
        plot_rows,
        [
            "periapsis",
            "v_inf_kms",
            "spin_orientation",
            "bound_mass_fraction_median",
            "row_count",
            "periapsis_code",
            "velocity_code",
            "mass_codes",
            "resolution_codes",
            "timesteps",
            "fof_linking_lengths",
        ],
    )

    log_plot_out = (repo_root / args.log_plot_out_svg).resolve()
    linear_plot_out = (repo_root / args.linear_plot_out_svg).resolve()
    log_plot_stats = build_plot_svg(plot_rows, log_plot_out, log_scale=True)
    linear_plot_stats = build_plot_svg(plot_rows, linear_plot_out, log_scale=False)

    print("\n[aggregation_summary]")
    for key, value in aggregate_stats.items():
        print(f"{key}={value}")
    print(f"aggregated_csv={aggregated_out}")
    print(f"plot_table_csv={plot_table_out}")
    print(f"log_plot_svg={log_plot_out}")
    print(f"linear_plot_svg={linear_plot_out}")
    for key, value in plot_row_stats.items():
        print(f"{key}={value}")
    for key, value in log_plot_stats.items():
        print(f"log_{key}={value}")
    for key, value in linear_plot_stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
