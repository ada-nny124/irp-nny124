from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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

POINT_COLUMNS = FIXED_GROUP_COLUMNS + ["v_inf_kms"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("ml/triage/bmf_hurdle_oof_predictions.csv"),
        help="Observed SPH BMF table.",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("report/figures/dense_region_velocity_trend.png"),
        help="Output figure path.",
    )
    parser.add_argument(
        "--output-points",
        type=Path,
        default=Path("report/tables/dense_region_velocity_trend_points.csv"),
        help="Long-form matched-point table.",
    )
    parser.add_argument(
        "--output-groups",
        type=Path,
        default=Path("report/tables/dense_region_velocity_trend_groups.csv"),
        help="Per-group summary table.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("report/tables/dense_region_velocity_trend_summary.md"),
        help="Markdown summary report.",
    )
    parser.add_argument(
        "--dense-mass",
        type=float,
        default=20.0,
        help="Dense-region mass_log10_kg to analyze.",
    )
    parser.add_argument(
        "--max-periapsis",
        type=float,
        default=1.6,
        help="Upper periapsis_Rm bound for the dense region.",
    )
    parser.add_argument(
        "--min-velocities",
        type=int,
        default=3,
        help="Minimum number of distinct velocity values for a matched group.",
    )
    parser.add_argument(
        "--figure-title",
        type=str,
        default="Observed SPH BMF vs encounter velocity in the dense Figure 5 region",
        help="Figure title.",
    )
    parser.add_argument(
        "--figure-subtitle",
        type=str,
        default=None,
        help="Optional subtitle. If omitted, a default subtitle is generated from the filters.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-12,
        help="Numerical tolerance for monotonic checks.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def spin_label(row: pd.Series) -> str:
    if not bool(row["has_explicit_spin"]):
        return "no spin"
    period = row["spin_period_hr"]
    if pd.isna(period):
        return f"{row['spin_axis']}, P=NA"
    return f"{row['spin_axis']}, P={period:.1f} h"


def group_label(row: pd.Series) -> str:
    return f"q={row['periapsis_Rm']:.1f} Rm, {spin_label(row)}"


def make_unique_group_labels(group_table: pd.DataFrame) -> pd.DataFrame:
    if group_table.empty:
        return group_table
    counts = group_table["group_label"].value_counts()
    duplicate_labels = counts[counts > 1].index
    if len(duplicate_labels) == 0:
        return group_table
    updated = group_table.copy()
    mask = updated["group_label"].isin(duplicate_labels)
    updated.loc[mask, "group_label"] = updated.loc[mask].apply(
        lambda row: f"{row['group_label']}, FoF={row['fof_linking_length']:.4f}",
        axis=1,
    )
    return updated


def mostly_monotonic(values: np.ndarray, tolerance: float) -> tuple[bool, int]:
    if len(values) <= 1:
        return False, 0
    diffs = np.diff(values)
    increase_count = int(np.sum(diffs > tolerance))
    net_drop = values[-1] <= values[0] + tolerance
    return bool(increase_count <= 1 and net_drop), increase_count


def build_point_table(df: pd.DataFrame, dense_mass: float, max_periapsis: float) -> pd.DataFrame:
    dense = df[
        np.isclose(df["mass_log10_kg"], dense_mass)
        & (pd.to_numeric(df["periapsis_Rm"], errors="coerce") <= max_periapsis)
    ].copy()
    point_table = (
        dense.groupby(POINT_COLUMNS, dropna=False)
        .agg(
            observed_bmf=("actual_bmf", "mean"),
            unique_physical_files=("physical_file", "nunique"),
            physical_files=("physical_file", lambda s: "|".join(sorted(set(map(str, s))))),
        )
        .reset_index()
        .sort_values(["periapsis_Rm", "spin_axis", "spin_period_hr", "v_inf_kms"])
        .reset_index(drop=True)
    )
    point_table["group_label"] = point_table.apply(group_label, axis=1)
    point_table["spin_label"] = point_table.apply(spin_label, axis=1)
    point_table["observed_bmf_percent"] = point_table["observed_bmf"] * 100.0
    return point_table


def build_group_table(point_table: pd.DataFrame, min_velocities: int, tolerance: float) -> pd.DataFrame:
    group_rows: list[dict[str, object]] = []
    for fixed_values, group in point_table.groupby(FIXED_GROUP_COLUMNS, dropna=False, sort=False):
        ordered = group.sort_values("v_inf_kms").reset_index(drop=True)
        if ordered["v_inf_kms"].nunique() < min_velocities:
            continue
        velocities = ordered["v_inf_kms"].to_numpy(dtype=float)
        bmfs = ordered["observed_bmf"].to_numpy(dtype=float)
        supports = ordered["unique_physical_files"].to_numpy(dtype=int)
        monotonic = bool(len(bmfs) > 1 and np.all(np.diff(bmfs) <= tolerance))
        mostly, increase_count = mostly_monotonic(bmfs, tolerance=tolerance)
        row = {column: value for column, value in zip(FIXED_GROUP_COLUMNS, fixed_values)}
        row.update(
            {
                "group_label": group_label(ordered.iloc[0]),
                "n_velocities": int(len(ordered)),
                "velocity_values": ",".join(f"{value:.1f}" for value in velocities),
                "observed_bmf_values": ",".join(f"{value:.6f}" for value in bmfs),
                "observed_bmf_percent_values": ",".join(f"{value * 100.0:.2f}" for value in bmfs),
                "support_counts": ",".join(str(value) for value in supports),
                "total_unique_physical_files": int(np.sum(supports)),
                "monotonic_decrease": monotonic,
                "mostly_monotonic_decrease": mostly,
                "increase_count": increase_count,
                "first_bmf": float(bmfs[0]),
                "last_bmf": float(bmfs[-1]),
                "net_bmf_change": float(bmfs[-1] - bmfs[0]),
            }
        )
        group_rows.append(row)
    group_table = pd.DataFrame(group_rows)
    if group_table.empty:
        return group_table
    group_table = group_table.sort_values(
        ["periapsis_Rm", "has_explicit_spin", "spin_axis", "spin_period_hr"]
    ).reset_index(drop=True)
    return make_unique_group_labels(group_table)


def render_plot(
    group_table: pd.DataFrame,
    point_table: pd.DataFrame,
    output_path: Path,
    figure_title: str,
    figure_subtitle: str | None,
) -> None:
    ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)
    cmap = plt.get_cmap("tab10")

    ordered_labels = group_table["group_label"].tolist()
    label_to_color = {label: cmap(idx % 10) for idx, label in enumerate(ordered_labels)}

    for _, group_row in group_table.iterrows():
        mask = (
            np.isclose(point_table["mass_log10_kg"], group_row["mass_log10_kg"])
            & np.isclose(point_table["periapsis_Rm"], group_row["periapsis_Rm"])
            & (point_table["has_explicit_spin"] == group_row["has_explicit_spin"])
            & (point_table["spin_axis"] == group_row["spin_axis"])
            & (
                point_table["spin_period_hr"].fillna(-9999.0)
                == (-9999.0 if pd.isna(group_row["spin_period_hr"]) else group_row["spin_period_hr"])
            )
            & np.isclose(point_table["resolution_value"], group_row["resolution_value"])
            & np.isclose(point_table["timestep"], group_row["timestep"])
            & np.isclose(point_table["fof_linking_length"], group_row["fof_linking_length"])
        )
        group_points = point_table.loc[mask].sort_values("v_inf_kms")
        color = label_to_color[group_row["group_label"]]
        yvals = group_points["observed_bmf_percent"].to_numpy()
        ax.plot(
            group_points["v_inf_kms"],
            yvals,
            marker="o",
            linewidth=2.0,
            markersize=6,
            color=color,
            label=group_row["group_label"],
        )
        for _, point in group_points.iterrows():
            ax.annotate(
                f"n={int(point['unique_physical_files'])}",
                (point["v_inf_kms"], point["observed_bmf_percent"]),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
                color=color,
            )

    mass_values = sorted(group_table["mass_log10_kg"].unique())
    resolution_values = sorted(group_table["resolution_value"].unique())
    fof_values = sorted(group_table["fof_linking_length"].unique())
    timestep_values = sorted(group_table["timestep"].unique())
    fig.suptitle(figure_title, fontsize=17)
    subtitle = figure_subtitle or (
        f"Matched groups only; fixed mass={mass_values[0]:.1f}, "
        f"periapsis <= {max(point_table['periapsis_Rm']):.1f} Rm, "
        f"resolution={resolution_values[0]:.0f}, timestep={timestep_values[0]:.0f}, "
        f"FoF={fof_values[0]:.3f}"
    )
    ax.set_title(subtitle, fontsize=10, pad=12)
    ax.set_xlabel("v_inf (km/s)")
    ax.set_ylabel("Observed BMF (%)")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    if point_table["v_inf_kms"].nunique() <= 1:
        ax.set_xlim(-0.05, 1.65)
        ax.set_xticks(np.arange(0.0, 1.61, 0.2))
    else:
        ax.set_xlim(left=-0.05)
    ax.set_ylim(bottom=-1.0)
    ax.legend(title="Matched fixed conditions", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_summary(
    group_table: pd.DataFrame,
    output_path: Path,
    point_table: pd.DataFrame,
    min_velocities: int,
    dense_mass: float,
    max_periapsis: float,
) -> None:
    ensure_parent(output_path)
    monotonic_count = int(group_table["monotonic_decrease"].sum())
    mostly_count = int(group_table["mostly_monotonic_decrease"].sum())
    net_decrease_count = int((group_table["net_bmf_change"] < 0.0).sum())
    shared_resolution = ", ".join(str(int(value)) for value in sorted(group_table["resolution_value"].unique()))
    shared_timestep = ", ".join(str(int(value)) for value in sorted(group_table["timestep"].unique()))
    shared_fof = ", ".join(f"{value:.3f}" for value in sorted(group_table["fof_linking_length"].unique()))

    lines = [
        "# Dense-region velocity trend summary",
        "",
        "This report uses observed SPH bound mass fraction only from matched groups in the selected Figure 5 subset.",
        "",
        f"- Subset filter: `mass_log10_kg = {dense_mass:.1f}` and `periapsis_Rm <= {max_periapsis:.1f}`.",
        f"- Matching dimensions held fixed: `{', '.join(FIXED_GROUP_COLUMNS)}`.",
        f"- Included groups: at least `{min_velocities}` distinct `v_inf_kms` values.",
        f"- Shared retained settings across included groups: `resolution_value={shared_resolution}`, `timestep={shared_timestep}`, `fof_linking_length={shared_fof}`.",
        f"- Matched groups included: `{len(group_table)}`.",
        f"- Strictly monotonic decreasing groups: `{monotonic_count}/{len(group_table)}`.",
        f"- Mostly monotonic decreasing groups: `{mostly_count}/{len(group_table)}`.",
        f"- Net first-to-last BMF decrease: `{net_decrease_count}/{len(group_table)}` groups.",
        "",
        "## Matched groups",
        "",
    ]

    for _, row in group_table.iterrows():
        monotonic_text = "yes" if bool(row["monotonic_decrease"]) else "no"
        mostly_text = "yes" if bool(row["mostly_monotonic_decrease"]) else "no"
        lines.extend(
            [
                f"### {row['group_label']}",
                "",
                f"- Fixed conditions: `mass={row['mass_log10_kg']:.1f}`, `periapsis={row['periapsis_Rm']:.1f}`, `has_explicit_spin={bool(row['has_explicit_spin'])}`, `spin_axis={row['spin_axis']}`, `spin_period_hr={row['spin_period_hr']}`, `resolution_value={row['resolution_value']}`, `timestep={int(row['timestep'])}`, `fof_linking_length={row['fof_linking_length']:.3f}`.",
                f"- Velocity values: `{row['velocity_values']}` km/s.",
                f"- Observed BMF values: `{row['observed_bmf_percent_values']}` percent.",
                f"- Unique physical_file support: `{row['support_counts']}`.",
                f"- Monotonic decrease: `{monotonic_text}`. Mostly monotonic decrease: `{mostly_text}`. Increases along the line: `{int(row['increase_count'])}`. Net BMF change: `{row['net_bmf_change'] * 100.0:.2f}` percentage points.",
                "",
            ]
        )

    output_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    point_table = build_point_table(df, dense_mass=args.dense_mass, max_periapsis=args.max_periapsis)
    group_table = build_group_table(
        point_table,
        min_velocities=args.min_velocities,
        tolerance=args.tolerance,
    )
    if group_table.empty:
        raise SystemExit("No dense-region matched groups met the minimum velocity-count requirement.")

    ensure_parent(args.output_points)
    point_table.to_csv(args.output_points, index=False)
    ensure_parent(args.output_groups)
    group_table.to_csv(args.output_groups, index=False)
    render_plot(
        group_table,
        point_table,
        args.output_figure,
        figure_title=args.figure_title,
        figure_subtitle=args.figure_subtitle,
    )
    write_summary(
        group_table,
        args.output_summary,
        point_table,
        min_velocities=args.min_velocities,
        dense_mass=args.dense_mass,
        max_periapsis=args.max_periapsis,
    )

    print(f"Wrote figure: {args.output_figure}")
    print(f"Wrote point table: {args.output_points}")
    print(f"Wrote group table: {args.output_groups}")
    print(f"Wrote summary: {args.output_summary}")


if __name__ == "__main__":
    main()
