#!/usr/bin/env python3
"""Assess whether sparse Figure 5 cells support matched SPH velocity sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FIXED_GROUP_COLUMNS = [
    "mass_log10_kg",
    "periapsis_Rm",
    "has_explicit_spin",
    "spin_axis",
    "spin_period_hr",
    "resolution_value",
    "timestep",
    "fof_linking_length",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("ml/triage/bmf_hurdle_oof_predictions.csv"),
        help="Observed SPH BMF table.",
    )
    parser.add_argument(
        "--sparse-threshold",
        type=int,
        default=2,
        help="Maximum unique physical simulations in a mass-periapsis cell to count as low support.",
    )
    parser.add_argument(
        "--output-cells",
        type=Path,
        default=Path("report/tables/sparse_region_velocity_candidate_cells.csv"),
        help="Output table of objectively sparse mass-periapsis cells.",
    )
    parser.add_argument(
        "--output-points",
        type=Path,
        default=Path("report/tables/sparse_region_velocity_points.csv"),
        help="Long-form sparse-region matched-point table.",
    )
    parser.add_argument(
        "--output-groups",
        type=Path,
        default=Path("report/tables/sparse_region_velocity_groups.csv"),
        help="Sparse-region matched-group summary table.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("report/tables/sparse_region_velocity_summary.md"),
        help="Markdown summary report.",
    )
    parser.add_argument(
        "--dense-groups",
        type=Path,
        default=Path("report/tables/dense_region_velocity_trend_groups.csv"),
        help="Dense-region matched-group summary table for contrast.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def spin_label(row: pd.Series) -> str:
    if not bool(row["has_explicit_spin"]):
        return "no spin"
    if pd.isna(row["spin_period_hr"]):
        return f"{row['spin_axis']}, P=NA"
    return f"{row['spin_axis']}, P={row['spin_period_hr']:.1f} h"


def group_label(row: pd.Series) -> str:
    return f"mass={row['mass_log10_kg']:.1f}, q={row['periapsis_Rm']:.1f} Rm, {spin_label(row)}"


def build_sparse_cells(df: pd.DataFrame, sparse_threshold: int) -> pd.DataFrame:
    physical = df[["physical_file", "mass_log10_kg", "periapsis_Rm"]].drop_duplicates("physical_file")
    cells = (
        physical.groupby(["mass_log10_kg", "periapsis_Rm"], dropna=False)
        .agg(unique_physical_sims=("physical_file", "nunique"))
        .reset_index()
        .sort_values(["mass_log10_kg", "periapsis_Rm"])
        .reset_index(drop=True)
    )
    return cells[cells["unique_physical_sims"] <= sparse_threshold].reset_index(drop=True)


def build_sparse_points(df: pd.DataFrame, sparse_cells: pd.DataFrame) -> pd.DataFrame:
    sparse = df.merge(sparse_cells, on=["mass_log10_kg", "periapsis_Rm"], how="inner")
    point_table = (
        sparse.groupby(FIXED_GROUP_COLUMNS + ["v_inf_kms"], dropna=False)
        .agg(
            observed_bmf=("actual_bmf", "mean"),
            unique_physical_files=("physical_file", "nunique"),
            cell_unique_physical_sims=("unique_physical_sims", "max"),
        )
        .reset_index()
        .sort_values(["mass_log10_kg", "periapsis_Rm", "spin_axis", "spin_period_hr", "v_inf_kms"])
        .reset_index(drop=True)
    )
    if not point_table.empty:
        point_table["group_label"] = point_table.apply(group_label, axis=1)
        point_table["observed_bmf_percent"] = point_table["observed_bmf"] * 100.0
    return point_table


def classify_direction(values: np.ndarray, tolerance: float = 1e-12) -> tuple[str, int]:
    diffs = np.diff(values)
    increases = int(np.sum(diffs > tolerance))
    decreases = int(np.sum(diffs < -tolerance))
    if len(diffs) == 0:
        return "no comparison", increases
    if np.all(diffs <= tolerance):
        return "monotonic decrease", increases
    if increases <= 1 and values[-1] <= values[0] + tolerance and decreases >= 1:
        return "mostly decreases", increases
    return "no clear direction", increases


def build_sparse_groups(point_table: pd.DataFrame) -> pd.DataFrame:
    group_rows: list[dict[str, object]] = []
    for fixed_values, group in point_table.groupby(FIXED_GROUP_COLUMNS, dropna=False, sort=False):
        ordered = group.sort_values("v_inf_kms").reset_index(drop=True)
        velocities = ordered["v_inf_kms"].to_numpy(dtype=float)
        bmfs = ordered["observed_bmf"].to_numpy(dtype=float)
        direction, increase_count = classify_direction(bmfs)
        row = {column: value for column, value in zip(FIXED_GROUP_COLUMNS, fixed_values)}
        row.update(
            {
                "group_label": group_label(ordered.iloc[0]),
                "cell_unique_physical_sims": int(ordered["cell_unique_physical_sims"].max()),
                "n_velocities": int(len(ordered)),
                "velocity_values": ",".join(f"{value:.1f}" for value in velocities),
                "observed_bmf_percent_values": ",".join(f"{value * 100.0:.2f}" for value in bmfs),
                "support_counts": ",".join(str(int(value)) for value in ordered["unique_physical_files"]),
                "direction_class": direction,
                "increase_count": increase_count,
            }
        )
        group_rows.append(row)
    columns = FIXED_GROUP_COLUMNS + [
        "group_label",
        "cell_unique_physical_sims",
        "n_velocities",
        "velocity_values",
        "observed_bmf_percent_values",
        "support_counts",
        "direction_class",
        "increase_count",
    ]
    return pd.DataFrame(group_rows, columns=columns)


def write_summary(
    sparse_cells: pd.DataFrame,
    sparse_groups: pd.DataFrame,
    dense_group_count: int,
    dense_repeatable_count: int,
    output_path: Path,
    sparse_threshold: int,
) -> None:
    ensure_parent(output_path)
    any_comparison = sparse_groups[sparse_groups["n_velocities"] >= 2].copy()
    meaningful_comparison = sparse_groups[sparse_groups["n_velocities"] >= 3].copy()

    lines = [
        "# Sparse-region velocity contrast summary",
        "",
        "This report defines low support objectively from Figure 5 mass-periapsis cells using unique physical simulation counts.",
        "",
        f"- Sparse-cell rule: `unique_physical_sims <= {sparse_threshold}`.",
        f"- Sparse mass-periapsis cells found: `{len(sparse_cells)}`.",
        f"- Sparse matched groups with at least 2 distinct velocity values: `{len(any_comparison)}`.",
        f"- Sparse matched groups with at least 3 distinct velocity values: `{len(meaningful_comparison)}`.",
        f"- Dense matched groups from the comparison analysis: `{dense_group_count}`.",
        f"- Dense groups that are monotonic or mostly monotonic decreasing: `{dense_repeatable_count}/{dense_group_count}`.",
        "",
        "## Sparse cells with enough data for any matched velocity comparison",
        "",
    ]

    if any_comparison.empty:
        lines.extend(
            [
                "None.",
                "",
                "Once mass, periapsis, spin state/orientation, spin period, resolution, timestep, and FoF linking length are held fixed, the objectively sparse Figure 5 cells contain no matched velocity sweep at all. Every sparse candidate configuration has only one sampled `v_inf_kms` value.",
                "",
                "## Dense vs sparse support",
                "",
                f"The dense region supports `{dense_group_count}` matched velocity families and `{dense_repeatable_count}` of them show a monotonic or mostly monotonic BMF decrease with velocity. The sparse region supports `0` matched velocity families, so it cannot independently test whether the same velocity-BMF relationship repeats there. The practical contrast is therefore strong: the dense region provides a repeatable velocity-BMF relationship, while the sparse region does not provide enough matched SPH support to evaluate one.",
            ]
        )
    else:
        for _, row in any_comparison.iterrows():
            lines.extend(
                [
                    f"### {row['group_label']}",
                    "",
                    f"- Distinct velocities: `{int(row['n_velocities'])}`.",
                    f"- Velocity values: `{row['velocity_values']}` km/s.",
                    f"- Observed BMF values: `{row['observed_bmf_percent_values']}` percent.",
                    f"- Unique physical simulations per point: `{row['support_counts']}`.",
                    f"- Direction: `{row['direction_class']}`.",
                    "",
                ]
            )

    output_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    sparse_cells = build_sparse_cells(df, sparse_threshold=args.sparse_threshold)
    sparse_points = build_sparse_points(df, sparse_cells)
    sparse_groups = build_sparse_groups(sparse_points)

    dense_groups = pd.read_csv(args.dense_groups) if args.dense_groups.exists() else pd.DataFrame()
    dense_group_count = int(len(dense_groups))
    dense_repeatable_count = int(
        dense_groups.get("mostly_monotonic_decrease", pd.Series(dtype=bool)).fillna(False).sum()
    )

    ensure_parent(args.output_cells)
    sparse_cells.to_csv(args.output_cells, index=False)
    ensure_parent(args.output_points)
    sparse_points.to_csv(args.output_points, index=False)
    ensure_parent(args.output_groups)
    sparse_groups.to_csv(args.output_groups, index=False)
    write_summary(
        sparse_cells=sparse_cells,
        sparse_groups=sparse_groups,
        dense_group_count=dense_group_count,
        dense_repeatable_count=dense_repeatable_count,
        output_path=args.output_summary,
        sparse_threshold=args.sparse_threshold,
    )

    print(f"Wrote sparse-cell table: {args.output_cells}")
    print(f"Wrote sparse-point table: {args.output_points}")
    print(f"Wrote sparse-group table: {args.output_groups}")
    print(f"Wrote summary: {args.output_summary}")


if __name__ == "__main__":
    main()
