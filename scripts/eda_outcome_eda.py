#!/usr/bin/env python3
"""Outcome EDA for full FoF outcome tables after extraction is complete."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EXPECTED_SIMULATION_ROWS = 489
BOUND_METRIC_COLUMNS = [
    "bound_particle_fraction",
    "bound_mass_fraction",
    "bound_fragment_count_min_particles",
    "largest_bound_fragment_mass_kg",
    "largest_bound_fragment_particle_count",
    "mean_bound_fragment_periapsis_Rm",
    "mean_bound_fragment_apoapsis_Rm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run outcome EDA using FoF outcomes and fragment catalogs."
    )
    parser.add_argument("--outcomes", required=True, help="Path to outputs/fof_outcomes.csv")
    parser.add_argument("--fragments", required=True, help="Path to outputs/fragment_catalog.csv")
    parser.add_argument(
        "--errors",
        default="outputs/extraction_errors.csv",
        help="Optional path to extraction_errors.csv",
    )
    parser.add_argument("--eda-dir", required=True, help="Output directory for EDA artifacts")
    return parser.parse_args()


def ensure_dirs(base_dir: Path):
    tables_dir = base_dir / "tables"
    plots_dir = base_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, plots_dir


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def has_meaningful_mass_metrics(outcomes: pd.DataFrame) -> bool:
    if "mass_metrics_available" not in outcomes.columns:
        return False
    available = outcomes["mass_metrics_available"].fillna(False).astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    if not available.any():
        return False
    mass_columns = ["largest_fragment_mass_kg", "fragment_mass_fraction"]
    return any(numeric_series(outcomes, column).notna().any() for column in mass_columns)


def has_bound_metrics(outcomes: pd.DataFrame) -> bool:
    if "bound_metrics_available" not in outcomes.columns:
        return False
    available = outcomes["bound_metrics_available"].fillna(False).astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    if not available.any():
        return False
    return any(
        numeric_series(outcomes, column).notna().any()
        for column in BOUND_METRIC_COLUMNS
        if column in outcomes.columns
    )


def write_readme(base_dir: Path) -> None:
    content = """# Outcome EDA

This directory is reserved for outcome-level EDA after `outputs/fof_outcomes.csv` is complete.

The script supports both full-study runs and smaller local validation subsets.

Generated outputs include:

- `tables/` for dataset overview and grouped outcome summaries
- `plots/` for fragment-count and mass-metric visualisations
- `analysis_summary.txt` for a plain-text interpretation and ML-readiness note

## Re-run

```bash
python scripts/eda_outcome_eda.py \
  --outcomes outputs/fof_outcomes.csv \
  --fragments outputs/fragment_catalog.csv \
  --errors outputs/extraction_errors.csv \
  --eda-dir eda/outcome_eda
```
"""
    (base_dir / "README.md").write_text(content, encoding="utf-8")


def write_dataset_overview(
    outcomes: pd.DataFrame, fragments: pd.DataFrame, errors: pd.DataFrame, tables_dir: Path
) -> pd.DataFrame:
    overview = pd.DataFrame(
        [
            {
                "outcome_rows": int(len(outcomes)),
                "expected_outcome_rows": EXPECTED_SIMULATION_ROWS,
                "outcomes_complete": bool(len(outcomes) >= EXPECTED_SIMULATION_ROWS),
                "fragment_rows": int(len(fragments)),
                "simulations_with_fragments": int(fragments["simulation_id"].nunique(dropna=True)),
                "extraction_error_rows": int(len(errors)),
                "simulations_with_errors": int(
                    errors["simulation_id"].nunique(dropna=True) if "simulation_id" in errors.columns else 0
                ),
                "mass_metrics_available_rows": int(
                    outcomes["mass_metrics_available"]
                    .fillna(False)
                    .astype(str)
                    .str.lower()
                    .isin({"true", "1", "yes"})
                    .sum()
                    if "mass_metrics_available" in outcomes.columns
                    else 0
                ),
            }
        ]
    )
    overview.to_csv(tables_dir / "outcome_dataset_overview.csv", index=False)
    return overview


def write_summary_stats(outcomes: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    metrics = [
        "particle_count_total",
        "n_fof_groups",
        "fragment_count_min_particles",
        "largest_fragment_particle_count",
        "largest_fragment_mass_kg",
        "total_fragment_mass_kg",
        "fragment_mass_fraction",
        "bound_particle_fraction",
        "bound_mass_fraction",
        "bound_fragment_count_min_particles",
        "largest_bound_fragment_mass_kg",
        "largest_bound_fragment_particle_count",
        "mean_bound_fragment_periapsis_Rm",
        "mean_bound_fragment_apoapsis_Rm",
        "fof_linking_length",
        "timestep",
    ]
    rows = []
    for metric in metrics:
        values = numeric_series(outcomes, metric).dropna()
        rows.append(
            {
                "metric": metric,
                "count": int(values.count()),
                "min": values.min() if not values.empty else pd.NA,
                "median": values.median() if not values.empty else pd.NA,
                "mean": values.mean() if not values.empty else pd.NA,
                "max": values.max() if not values.empty else pd.NA,
                "std": values.std() if len(values) > 1 else pd.NA,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(tables_dir / "outcome_summary_stats.csv", index=False)
    return summary


def write_grouped_means(outcomes: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    group_columns = [
        "mass_code",
        "resolution_code",
        "periapsis_code",
        "velocity_code",
        "timestep",
        "fof_linking_length",
    ]
    metric_columns = [
        column
        for column in [
            "fragment_count_min_particles",
            "largest_fragment_particle_count",
            "largest_fragment_mass_kg",
            "fragment_mass_fraction",
            "bound_particle_fraction",
            "bound_mass_fraction",
            "bound_fragment_count_min_particles",
            "largest_bound_fragment_mass_kg",
            "largest_bound_fragment_particle_count",
        ]
        if column in outcomes.columns
    ]
    grouped = (
        outcomes.groupby(group_columns, dropna=False)[metric_columns]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(group_columns)
    )
    grouped.to_csv(tables_dir / "grouped_outcome_means.csv", index=False)
    return grouped


def write_clean_subset_summary(outcomes: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    resolution_mode = outcomes["resolution_code"].mode(dropna=True)
    linking_mode = outcomes["fof_linking_length"].mode(dropna=True)
    resolution_value = resolution_mode.iloc[0] if not resolution_mode.empty else pd.NA
    linking_value = linking_mode.iloc[0] if not linking_mode.empty else pd.NA

    subset = outcomes.copy()
    subset = subset[numeric_series(subset, "timestep") == 90000]
    if resolution_value is not pd.NA:
        subset = subset[subset["resolution_code"] == resolution_value]
    if linking_value is not pd.NA:
        subset = subset[subset["fof_linking_length"] == linking_value]
    if "special_case_code" in subset.columns:
        subset = subset[subset["special_case_code"].fillna("") == ""]

    summary = pd.DataFrame(
        [
            {
                "recommended_timestep": 90000,
                "most_common_resolution_code": resolution_value,
                "most_common_fof_linking_length": linking_value,
                "exclude_special_cases": True,
                "subset_rows": int(len(subset)),
                "subset_fraction": float(len(subset) / len(outcomes)) if len(outcomes) else 0.0,
                "mean_fragment_count_min_particles": numeric_series(
                    subset, "fragment_count_min_particles"
                ).mean(),
                "mean_largest_fragment_particle_count": numeric_series(
                    subset, "largest_fragment_particle_count"
                ).mean(),
                "mean_largest_fragment_mass_kg": numeric_series(subset, "largest_fragment_mass_kg").mean(),
                "mean_fragment_mass_fraction": numeric_series(subset, "fragment_mass_fraction").mean(),
                "mean_bound_mass_fraction": numeric_series(subset, "bound_mass_fraction").mean(),
                "mean_bound_fragment_count_min_particles": numeric_series(
                    subset, "bound_fragment_count_min_particles"
                ).mean(),
                "mean_largest_bound_fragment_mass_kg": numeric_series(
                    subset, "largest_bound_fragment_mass_kg"
                ).mean(),
            }
        ]
    )
    summary.to_csv(tables_dir / "clean_physical_subset_summary.csv", index=False)
    return summary


def write_fragment_population_tables(fragments: pd.DataFrame, tables_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    masses = numeric_series(fragments, "fragment_mass_kg").dropna()
    mass_summary = pd.DataFrame(
        [
            {
                "fragment_rows_with_mass": int(masses.count()),
                "min_fragment_mass_kg": masses.min() if not masses.empty else pd.NA,
                "median_fragment_mass_kg": masses.median() if not masses.empty else pd.NA,
                "mean_fragment_mass_kg": masses.mean() if not masses.empty else pd.NA,
                "p90_fragment_mass_kg": masses.quantile(0.9) if not masses.empty else pd.NA,
                "p99_fragment_mass_kg": masses.quantile(0.99) if not masses.empty else pd.NA,
                "max_fragment_mass_kg": masses.max() if not masses.empty else pd.NA,
            }
        ]
    )
    mass_summary.to_csv(tables_dir / "fragment_mass_distribution_summary.csv", index=False)

    ranked = fragments.copy()
    ranked["fragment_mass_kg_num"] = numeric_series(ranked, "fragment_mass_kg")
    ranked = ranked.dropna(subset=["fragment_mass_kg_num"])
    ranked = ranked[ranked["fragment_mass_kg_num"] > 0].copy()
    ranked["rank"] = ranked.groupby("simulation_id")["fragment_mass_kg_num"].rank(method="first", ascending=False)
    ranked["total_fragment_mass_kg"] = ranked.groupby("simulation_id")["fragment_mass_kg_num"].transform("sum")
    ranked["cumulative_mass_fraction"] = (
        ranked.sort_values(["simulation_id", "rank"])
        .groupby("simulation_id")["fragment_mass_kg_num"]
        .cumsum()
        / ranked.sort_values(["simulation_id", "rank"]).groupby("simulation_id")["total_fragment_mass_kg"].transform("first")
    ).sort_index()
    rank_summary = (
        ranked[ranked["rank"] <= 20]
        .groupby("rank")["cumulative_mass_fraction"]
        .agg(["mean", "median", "min", "max"])
        .reset_index()
    )
    rank_summary.columns = ["rank", "mean_cumulative_mass_fraction", "median_cumulative_mass_fraction", "min_cumulative_mass_fraction", "max_cumulative_mass_fraction"]
    rank_summary.to_csv(tables_dir / "fragment_rank_cumulative_mass_summary.csv", index=False)
    return mass_summary, rank_summary


def save_histogram(series: pd.Series, title: str, xlabel: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(series.dropna(), bins=20, color="#4C78A8", edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_scatter(
    x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str, output_path: Path
) -> None:
    clean = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(clean["x"], clean["y"], alpha=0.7, color="#F58518", edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_heatmap(table: pd.DataFrame, title: str, xlabel: str, ylabel: str, output_path: Path) -> None:
    fig_width = max(8, len(table.columns) * 0.8)
    fig_height = max(6, len(table.index) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(table.to_numpy(), aspect="auto", cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels([str(value) for value in table.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels([str(value) for value in table.index])
    fig.colorbar(image, ax=ax, label="Mean fragment count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_fragment_population_plots(outcomes: pd.DataFrame, fragments: pd.DataFrame, plots_dir: Path) -> list[str]:
    generated: list[str] = []

    fragment_masses = numeric_series(fragments, "fragment_mass_kg").dropna()
    if not fragment_masses.empty:
        save_histogram(
            fragment_masses,
            "Distribution of fragment mass",
            "Fragment mass (kg)",
            plots_dir / "distribution_fragment_mass_kg.png",
        )
        generated.append("distribution_fragment_mass_kg.png")

        ccdf = (
            pd.DataFrame({"fragment_mass_kg": fragment_masses})
            .sort_values("fragment_mass_kg", ascending=False)
            .reset_index(drop=True)
        )
        ccdf["n_fragments_ge_mass"] = range(1, len(ccdf) + 1)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.step(ccdf["fragment_mass_kg"], ccdf["n_fragments_ge_mass"], where="post", color="#4C78A8")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("Cumulative number of fragments above mass")
        ax.set_xlabel("Fragment mass (kg)")
        ax.set_ylabel("N(>= mass)")
        fig.tight_layout()
        fig.savefig(plots_dir / "cumulative_fragment_count_above_mass.png", dpi=150)
        plt.close(fig)
        generated.append("cumulative_fragment_count_above_mass.png")

    ranked = fragments.copy()
    ranked["fragment_mass_kg_num"] = numeric_series(ranked, "fragment_mass_kg")
    ranked = ranked.dropna(subset=["fragment_mass_kg_num"])
    ranked = ranked[ranked["fragment_mass_kg_num"] > 0].copy()
    if not ranked.empty:
        ranked = ranked.sort_values(["simulation_id", "fragment_mass_kg_num"], ascending=[True, False])
        ranked["rank"] = ranked.groupby("simulation_id").cumcount() + 1
        ranked["total_fragment_mass_kg"] = ranked.groupby("simulation_id")["fragment_mass_kg_num"].transform("sum")
        ranked["cumulative_mass_fraction"] = (
            ranked.groupby("simulation_id")["fragment_mass_kg_num"].cumsum() / ranked["total_fragment_mass_kg"]
        )
        rank_curve = ranked[ranked["rank"] <= 20].groupby("rank", as_index=False)["cumulative_mass_fraction"].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(rank_curve["rank"], rank_curve["cumulative_mass_fraction"], color="#F58518", linewidth=2)
        ax.set_title("Cumulative mass fraction vs fragment rank")
        ax.set_xlabel("Fragment rank by mass")
        ax.set_ylabel("Mean cumulative mass fraction")
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(plots_dir / "cumulative_mass_fraction_vs_fragment_rank.png", dpi=150)
        plt.close(fig)
        generated.append("cumulative_mass_fraction_vs_fragment_rank.png")

    if {"largest_fragment_mass_kg", "total_particle_mass_kg", "periapsis_value"}.issubset(outcomes.columns):
        largest_fraction = pd.to_numeric(outcomes["largest_fragment_mass_kg"], errors="coerce") / pd.to_numeric(
            outcomes["total_particle_mass_kg"], errors="coerce"
        )
        save_scatter(
            numeric_series(outcomes, "periapsis_value"),
            largest_fraction,
            "Largest fragment mass fraction vs periapsis",
            "Periapsis (Rm)",
            "Largest fragment mass fraction",
            plots_dir / "largest_fragment_mass_fraction_vs_periapsis.png",
        )
        generated.append("largest_fragment_mass_fraction_vs_periapsis.png")

    return generated


def write_plots(outcomes: pd.DataFrame, plots_dir: Path):
    generated = []
    plot_metrics = [
        ("fragment_count_min_particles", "distribution_fragment_count.png", "Distribution of fragment count", "Fragment count"),
        (
            "largest_fragment_particle_count",
            "distribution_largest_fragment_particle_count.png",
            "Distribution of largest fragment particle count",
            "Largest fragment particle count",
        ),
    ]
    for column, filename, title, xlabel in plot_metrics:
        save_histogram(numeric_series(outcomes, column), title, xlabel, plots_dir / filename)
        generated.append(filename)

    if numeric_series(outcomes, "largest_fragment_mass_kg").notna().any():
        save_histogram(
            numeric_series(outcomes, "largest_fragment_mass_kg"),
            "Distribution of largest fragment mass",
            "Largest fragment mass (kg)",
            plots_dir / "distribution_largest_fragment_mass_kg.png",
        )
        generated.append("distribution_largest_fragment_mass_kg.png")

    if numeric_series(outcomes, "fragment_mass_fraction").notna().any():
        save_histogram(
            numeric_series(outcomes, "fragment_mass_fraction"),
            "Distribution of fragment mass fraction",
            "Fragment mass fraction",
            plots_dir / "distribution_fragment_mass_fraction.png",
        )
        generated.append("distribution_fragment_mass_fraction.png")

    if numeric_series(outcomes, "bound_mass_fraction").notna().any():
        save_histogram(
            numeric_series(outcomes, "bound_mass_fraction"),
            "Distribution of bound mass fraction",
            "Bound mass fraction",
            plots_dir / "distribution_bound_mass_fraction.png",
        )
        generated.append("distribution_bound_mass_fraction.png")

    if numeric_series(outcomes, "bound_fragment_count_min_particles").notna().any():
        save_histogram(
            numeric_series(outcomes, "bound_fragment_count_min_particles"),
            "Distribution of bound fragment count",
            "Bound fragment count",
            plots_dir / "distribution_bound_fragment_count.png",
        )
        generated.append("distribution_bound_fragment_count.png")

    scatter_specs = [
        ("periapsis_value", "fragment_count_min_particles", "fragment_count_vs_periapsis.png", "Fragment count vs periapsis", "Periapsis (Rm)", "Fragment count"),
        ("velocity_value", "fragment_count_min_particles", "fragment_count_vs_velocity.png", "Fragment count vs velocity", "Velocity (km/s)", "Fragment count"),
        ("mass_value", "fragment_count_min_particles", "fragment_count_vs_mass.png", "Fragment count vs mass", "Mass code value / 100", "Fragment count"),
        (
            "periapsis_value",
            "largest_fragment_particle_count",
            "largest_fragment_particles_vs_periapsis.png",
            "Largest fragment particles vs periapsis",
            "Periapsis (Rm)",
            "Largest fragment particle count",
        ),
        (
            "fof_linking_length",
            "fragment_count_min_particles",
            "fragment_count_vs_fof_linking_length.png",
            "Fragment count vs FoF linking length",
            "FoF linking length",
            "Fragment count",
        ),
        (
            "fof_linking_length",
            "largest_fragment_particle_count",
            "largest_fragment_particles_vs_fof_linking_length.png",
            "Largest fragment particles vs FoF linking length",
            "FoF linking length",
            "Largest fragment particle count",
        ),
    ]
    for x_col, y_col, filename, title, xlabel, ylabel in scatter_specs:
        save_scatter(
            numeric_series(outcomes, x_col),
            numeric_series(outcomes, y_col),
            title,
            xlabel,
            ylabel,
            plots_dir / filename,
        )
        generated.append(filename)

    if numeric_series(outcomes, "largest_fragment_mass_kg").notna().any():
        save_scatter(
            numeric_series(outcomes, "mass_value"),
            numeric_series(outcomes, "largest_fragment_mass_kg"),
            "Largest fragment mass vs mass",
            "Mass code value / 100",
            "Largest fragment mass (kg)",
            plots_dir / "largest_fragment_mass_vs_mass.png",
        )
        generated.append("largest_fragment_mass_vs_mass.png")

    if numeric_series(outcomes, "fragment_mass_fraction").notna().any():
        save_scatter(
            numeric_series(outcomes, "periapsis_value"),
            numeric_series(outcomes, "fragment_mass_fraction"),
            "Fragment mass fraction vs periapsis",
            "Periapsis (Rm)",
            "Fragment mass fraction",
            plots_dir / "fragment_mass_fraction_vs_periapsis.png",
        )
        generated.append("fragment_mass_fraction_vs_periapsis.png")

    if numeric_series(outcomes, "bound_mass_fraction").notna().any():
        save_scatter(
            numeric_series(outcomes, "periapsis_value"),
            numeric_series(outcomes, "bound_mass_fraction"),
            "Bound mass fraction vs periapsis",
            "Periapsis (Rm)",
            "Bound mass fraction",
            plots_dir / "bound_mass_fraction_vs_periapsis.png",
        )
        generated.append("bound_mass_fraction_vs_periapsis.png")

    if numeric_series(outcomes, "largest_bound_fragment_mass_kg").notna().any():
        save_scatter(
            numeric_series(outcomes, "periapsis_value"),
            numeric_series(outcomes, "largest_bound_fragment_mass_kg"),
            "Largest bound fragment mass vs periapsis",
            "Periapsis (Rm)",
            "Largest bound fragment mass (kg)",
            plots_dir / "largest_bound_fragment_mass_vs_periapsis.png",
        )
        generated.append("largest_bound_fragment_mass_vs_periapsis.png")

    heatmaps = [
        (
            ["mass_code", "periapsis_code"],
            "heatmap_mean_fragment_count_mass_vs_periapsis.png",
            "Mean fragment count: mass vs periapsis",
            "Periapsis code",
            "Mass code",
        ),
        (
            ["periapsis_code", "velocity_code"],
            "heatmap_mean_fragment_count_periapsis_vs_velocity.png",
            "Mean fragment count: periapsis vs velocity",
            "Velocity code",
            "Periapsis code",
        ),
    ]
    for group_cols, filename, title, xlabel, ylabel in heatmaps:
        table = outcomes.pivot_table(
            values="fragment_count_min_particles",
            index=group_cols[0],
            columns=group_cols[1],
            aggfunc="mean",
        ).sort_index().sort_index(axis=1)
        save_heatmap(table, title, xlabel, ylabel, plots_dir / filename)
        generated.append(filename)

    return generated


def write_analysis_summary(
    base_dir: Path,
    overview: pd.DataFrame,
    summary_stats: pd.DataFrame,
    clean_subset_summary: pd.DataFrame,
    errors: pd.DataFrame,
    mass_metrics_available: bool,
    bound_metrics_available: bool,
) -> None:
    overview_row = overview.iloc[0]
    clean_row = clean_subset_summary.iloc[0]
    fragment_summary = summary_stats.set_index("metric")
    lines = [
        "After-extraction EDA summary.",
        "",
        "Following the supervisor suggestion, this pass looks beyond fragment count and largest fragment toward the full fragment population.",
        "",
        f"Extraction completeness: {int(overview_row['outcome_rows'])}/{EXPECTED_SIMULATION_ROWS} outcome rows.",
        f"Simulations with extracted outcomes: {int(overview_row['outcome_rows'])}.",
        f"Extraction errors recorded: {int(overview_row['extraction_error_rows'])}.",
        "Outcome ranges and typical values:",
        f"- fragment_count_min_particles: median={fragment_summary.at['fragment_count_min_particles', 'median']}",
        f"- largest_fragment_particle_count: median={fragment_summary.at['largest_fragment_particle_count', 'median']}",
        f"- fragment_mass_fraction: median={fragment_summary.at['fragment_mass_fraction', 'median']}",
        "",
        f"Mass metrics available: {mass_metrics_available}.",
        f"Bound/unbound metrics available: {bound_metrics_available}.",
        "FoF linking length is a post-processing control and can dominate detected fragment counts.",
        "Bound mass fraction and bound-fragment metrics are the current bridge from FoF proxies to physical retention.",
        "Attempted bound-aware extraction note: fragment COM position and velocity are required for captured/bound metrics; sampled FoF-file velocities are zero, so FoF-only bound/captured extraction remains paused to avoid false metrics.",
        "",
        "Recommended clean subset:",
        f"- timestep == {int(clean_row['recommended_timestep'])}",
        f"- resolution_code == {clean_row['most_common_resolution_code']}",
        f"- fof_linking_length == {clean_row['most_common_fof_linking_length']}",
        "- exclude special cases when comparing physical trends",
        "",
        "ML readiness:",
        "- use one row per simulation from fof_outcomes.csv",
        "- suggested targets: fragment_count_min_particles, largest_fragment_particle_count",
        "- add largest_fragment_mass_kg and fragment_mass_fraction when mass metrics are available",
        "- prefer bound_mass_fraction, bound_fragment_count_min_particles, and largest_bound_fragment_mass_kg when bound metrics are available",
    ]
    if len(errors) and "error_message" in errors.columns:
        lines.extend(["", "Error sample:", f"- {errors['error_message'].iloc[0]}"])
    (base_dir / "analysis_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    outcomes_path = Path(args.outcomes)
    fragments_path = Path(args.fragments)
    errors_path = Path(args.errors)
    eda_dir = Path(args.eda_dir)
    ensure_dirs(eda_dir)
    write_readme(eda_dir)

    outcomes = load_csv(outcomes_path)
    fragments = load_csv(fragments_path)
    errors = load_csv(errors_path) if errors_path.exists() else pd.DataFrame()
    tables_dir, plots_dir = ensure_dirs(eda_dir)

    overview = write_dataset_overview(outcomes, fragments, errors, tables_dir)
    summary_stats = write_summary_stats(outcomes, tables_dir)
    write_grouped_means(outcomes, tables_dir)
    clean_subset_summary = write_clean_subset_summary(outcomes, tables_dir)
    write_fragment_population_tables(fragments, tables_dir)
    write_plots(outcomes, plots_dir)
    write_fragment_population_plots(outcomes, fragments, plots_dir)
    write_analysis_summary(
        eda_dir,
        overview,
        summary_stats,
        clean_subset_summary,
        errors,
        has_meaningful_mass_metrics(outcomes),
        has_bound_metrics(outcomes),
    )
    if len(outcomes) < EXPECTED_SIMULATION_ROWS:
        print(
            f"Warning: analyzed a partial outcome table with {len(outcomes)} rows; "
            "interpret results as local validation rather than full-study EDA."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
