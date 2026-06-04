#!/usr/bin/env python3
"""Raw-data overview for manifest and sampled HDF5 schema."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd


EXPECTED_SIMULATION_ROWS = 489


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run raw-data overview using manifest and sampled schema data."
    )
    parser.add_argument("--manifest", required=True, help="Path to outputs/manifest.csv")
    parser.add_argument("--schema", required=True, help="Path to outputs/hdf5_schema_summary.csv")
    parser.add_argument("--eda-dir", required=True, help="Output directory for EDA artifacts")
    parser.add_argument(
        "--report-dir",
        default="report/figures",
        help="Optional report figure output directory",
    )
    return parser.parse_args()


def ensure_dirs(base_dir: Path):
    tables_dir = base_dir / "tables"
    plots_dir = base_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, plots_dir


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def mass_log10_kg_from_value(series: pd.Series) -> pd.Series:
    return numeric_series(series) / 100.0


def periapsis_rm_from_value(series: pd.Series) -> pd.Series:
    return numeric_series(series) / 10.0


def velocity_kms_from_value(series: pd.Series) -> pd.Series:
    return numeric_series(series) / 10.0


def summarise_numeric_columns(manifest: pd.DataFrame) -> pd.DataFrame:
    summary_columns = {
        "mass_log10_kg": mass_log10_kg_from_value(manifest["mass_value"]),
        "particle_log10": numeric_series(manifest["resolution_value"]) / 10.0,
        "periapsis_Rm": periapsis_rm_from_value(manifest["periapsis_value"]),
        "v_inf_kms": velocity_kms_from_value(manifest["velocity_value"]),
        "spin_period_hr": numeric_series(manifest["spin_value"]) / 10.0,
        "fof_linking_length": numeric_series(manifest["fof_linking_length"]),
        "timestep": numeric_series(manifest["timestep"]),
        "file_size_bytes": numeric_series(manifest["file_size_bytes"]),
    }

    rows = []
    for metric, values in summary_columns.items():
        clean = values.dropna()
        rows.append(
            {
                "metric": metric,
                "count": int(clean.count()),
                "min": clean.min() if not clean.empty else pd.NA,
                "median": clean.median() if not clean.empty else pd.NA,
                "mean": clean.mean() if not clean.empty else pd.NA,
                "max": clean.max() if not clean.empty else pd.NA,
                "std": clean.std() if len(clean) > 1 else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def write_dataset_overview(manifest: pd.DataFrame, schema: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    file_sizes = numeric_series(manifest["file_size_bytes"])
    overview = pd.DataFrame(
        [
            {
                "manifest_simulation_rows": int(len(manifest)),
                "schema_rows": int(len(schema)),
                "unique_sampled_schema_files": int(schema["sample_filename"].nunique(dropna=True)),
                "total_file_size_bytes": int(file_sizes.sum()),
                "min_file_size_bytes": int(file_sizes.min()),
                "median_file_size_bytes": float(file_sizes.median()),
                "max_file_size_bytes": int(file_sizes.max()),
                "manifest_appears_complete": bool(len(manifest) == EXPECTED_SIMULATION_ROWS),
                "expected_manifest_rows": EXPECTED_SIMULATION_ROWS,
            }
        ]
    )
    overview.to_csv(tables_dir / "dataset_overview.csv", index=False)
    return overview


def write_parameter_counts(manifest: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    parameter_columns = [
        "mass_code",
        "resolution_code",
        "periapsis_code",
        "velocity_code",
        "spin_code",
        "spin_axis",
        "has_explicit_spin",
        "timestep",
        "fof_linking_length",
    ]

    rows = []
    for column in parameter_columns:
        values = manifest[column].fillna("(missing)")
        values = values.replace("", "(none)")
        counts = values.value_counts(dropna=False).sort_index()
        for value, count in counts.items():
            rows.append({"parameter": column, "value": value, "count": int(count)})

    parameter_counts = pd.DataFrame(rows)
    parameter_counts.to_csv(tables_dir / "parameter_counts.csv", index=False)
    return parameter_counts


def write_coverage_tables(manifest: pd.DataFrame, tables_dir: Path):
    coverage_specs = {
        "coverage_mass_vs_periapsis.csv": ("mass_code", "periapsis_code"),
        "coverage_periapsis_vs_velocity.csv": ("periapsis_code", "velocity_code"),
        "coverage_mass_vs_resolution.csv": ("mass_code", "resolution_code"),
    }

    tables = {}
    for filename, (rows, cols) in coverage_specs.items():
        table = pd.crosstab(manifest[rows], manifest[cols]).sort_index().sort_index(axis=1)
        table.to_csv(tables_dir / filename)
        tables[filename] = table
    return tables


def write_schema_tables(schema: pd.DataFrame, tables_dir: Path):
    path_series = schema["node_path"].fillna("")
    field_patterns = {
        "PartType0": r"(?:^|/)PartType0(?:$|/)",
        "FOFGroupIDs": r"(?:^|/)PartType0/FOFGroupIDs(?:$|/)",
        "Masses": r"(?:^|/)PartType0/Masses(?:$|/)",
        "Coordinates": r"(?:^|/)PartType0/Coordinates(?:$|/)",
        "Velocities": r"(?:^|/)PartType0/Velocities(?:$|/)",
        "ParticleIDs": r"(?:^|/)PartType0/ParticleIDs(?:$|/)",
        "Units": r"(?:^|/)Units(?:$|/)",
        "Parameters": r"(?:^|/)Parameters(?:$|/)",
    }

    availability_rows = []
    for field_name, pattern in field_patterns.items():
        matches = schema[path_series.str.contains(pattern, regex=True, na=False)]
        availability_rows.append(
            {
                "field_name": field_name,
                "present_in_sample_schema": bool(not matches.empty),
                "matching_schema_rows": int(len(matches)),
                "sample_files_with_match": int(matches["sample_filename"].nunique(dropna=True)),
            }
        )

    schema_fields = pd.DataFrame(availability_rows)
    schema_fields.to_csv(tables_dir / "schema_available_fields.csv", index=False)

    dataset_paths = (
        schema[["node_path", "node_type"]]
        .drop_duplicates()
        .sort_values(["node_path", "node_type"], na_position="last")
        .reset_index(drop=True)
    )
    dataset_paths.to_csv(tables_dir / "schema_dataset_paths.csv", index=False)
    return schema_fields, dataset_paths


def save_bar_plot(series: pd.Series, title: str, xlabel: str, ylabel: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [str(value) for value in series.index]
    ax.bar(labels, series.values, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_heatmap(table: pd.DataFrame, title: str, xlabel: str, ylabel: str, output_path: Path) -> None:
    fig_width = max(8, len(table.columns) * 0.7)
    fig_height = max(6, len(table.index) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(table.to_numpy(), aspect="auto", cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels([str(value) for value in table.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels([str(value) for value in table.index])
    fig.colorbar(image, ax=ax, label="Simulation count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_count_heatmap(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    log_scale: bool = False,
) -> None:
    clean = frame[[x_col, y_col]].copy()
    clean[x_col] = pd.to_numeric(clean[x_col], errors="coerce")
    clean[y_col] = pd.to_numeric(clean[y_col], errors="coerce")
    clean = clean.dropna()
    table = pd.crosstab(clean[y_col], clean[x_col]).sort_index().sort_index(axis=1)

    fig_width = max(8, len(table.columns) * 0.7)
    fig_height = max(6, len(table.index) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    values = table.to_numpy()
    if log_scale:
        positive = values[values > 0]
        norm = LogNorm(vmin=max(1, int(positive.min())), vmax=int(positive.max())) if positive.size else None
        image = ax.imshow(values, aspect="auto", cmap="Blues", norm=norm)
        colorbar_label = "Simulation count (log scale)"
    else:
        image = ax.imshow(values, aspect="auto", cmap="Blues")
        colorbar_label = "Simulation count"

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels([f"{value:.1f}" for value in table.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels([f"{value:.1f}" for value in table.index])
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_plots(manifest: pd.DataFrame, coverage_tables, plots_dir: Path, report_dir: Path | None = None):
    plot_specs = [
        ("mass_code", "count_by_mass.png", "Simulation count by mass", "Mass code"),
        ("resolution_code", "count_by_resolution.png", "Simulation count by resolution", "Resolution code"),
        ("periapsis_code", "count_by_periapsis.png", "Simulation count by periapsis", "Periapsis code"),
        ("velocity_code", "count_by_velocity.png", "Simulation count by velocity", "Velocity code"),
        ("spin_code", "count_by_spin.png", "Simulation count by spin", "Spin code"),
        ("timestep", "count_by_timestep.png", "Simulation count by timestep", "Timestep"),
        (
            "fof_linking_length",
            "count_by_fof_linking_length.png",
            "Simulation count by FoF linking length",
            "FoF linking length",
        ),
    ]

    generated_plots = []
    for column, filename, title, xlabel in plot_specs:
        values = manifest[column].fillna("(missing)").replace("", "(none)")
        counts = values.value_counts().sort_index()
        save_bar_plot(counts, title, xlabel, "Count", plots_dir / filename)
        generated_plots.append(filename)

    heatmap_specs = [
        (
            "coverage_mass_vs_periapsis.csv",
            "heatmap_mass_vs_periapsis_count.png",
            "Coverage heatmap: mass vs periapsis",
            "Periapsis code",
            "Mass code",
        ),
        (
            "coverage_periapsis_vs_velocity.csv",
            "heatmap_periapsis_vs_velocity_count.png",
            "Coverage heatmap: periapsis vs velocity",
            "Velocity code",
            "Periapsis code",
        ),
        (
            "coverage_mass_vs_resolution.csv",
            "heatmap_mass_vs_resolution_count.png",
            "Coverage heatmap: mass vs resolution",
            "Resolution code",
            "Mass code",
        ),
    ]

    for table_name, filename, title, xlabel, ylabel in heatmap_specs:
        save_heatmap(coverage_tables[table_name], title, xlabel, ylabel, plots_dir / filename)
        generated_plots.append(filename)

    physical_manifest = manifest.assign(
        mass_log10_kg=mass_log10_kg_from_value(manifest["mass_value"]),
        periapsis_Rm=periapsis_rm_from_value(manifest["periapsis_value"]),
        v_inf_kms=velocity_kms_from_value(manifest["velocity_value"]),
    )

    physical_heatmap_specs = [
        (
            "periapsis_Rm",
            "mass_log10_kg",
            "Coverage heatmap: mass vs periapsis",
            "Periapsis (Mars radii)",
            "Asteroid mass (log10 kg)",
            "heatmap_mass_vs_periapsis_count.png",
        ),
        (
            "periapsis_Rm",
            "mass_log10_kg",
            "Coverage heatmap: mass vs periapsis",
            "Periapsis (Mars radii)",
            "Asteroid mass (log10 kg)",
            "heatmap_mass_vs_periapsis_count_log.png",
        ),
        (
            "periapsis_Rm",
            "v_inf_kms",
            "Coverage heatmap: velocity vs periapsis",
            "Periapsis (Mars radii)",
            "Velocity at infinity (km/s)",
            "heatmap_velocity_vs_periapsis_count_log.png",
        ),
        (
            "v_inf_kms",
            "mass_log10_kg",
            "Coverage heatmap: mass vs velocity",
            "Velocity at infinity (km/s)",
            "Asteroid mass (log10 kg)",
            "heatmap_mass_vs_velocity_count_log.png",
        ),
    ]

    for x_col, y_col, title, xlabel, ylabel, filename in physical_heatmap_specs:
        save_count_heatmap(
            physical_manifest,
            x_col=x_col,
            y_col=y_col,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            output_path=plots_dir / filename,
            log_scale=True,
        )
        if filename not in generated_plots:
            generated_plots.append(filename)

    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        for _, _, title, xlabel, ylabel, filename in physical_heatmap_specs:
            if filename == "heatmap_mass_vs_periapsis_count.png":
                x_col = "periapsis_Rm"
                y_col = "mass_log10_kg"
            elif filename == "heatmap_velocity_vs_periapsis_count_log.png":
                x_col = "periapsis_Rm"
                y_col = "v_inf_kms"
            elif filename == "heatmap_mass_vs_velocity_count_log.png":
                x_col = "v_inf_kms"
                y_col = "mass_log10_kg"
            else:
                x_col = "periapsis_Rm"
                y_col = "mass_log10_kg"
            save_count_heatmap(
                physical_manifest,
                x_col=x_col,
                y_col=y_col,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                output_path=report_dir / filename,
                log_scale=True,
            )

    file_sizes_mb = numeric_series(manifest["file_size_bytes"]) / (1024.0 * 1024.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(file_sizes_mb.dropna(), bins=20, color="#72B7B2", edgecolor="black")
    ax.set_title("HDF5 file size distribution")
    ax.set_xlabel("File size (MiB)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(plots_dir / "file_size_distribution.png", dpi=150)
    plt.close(fig)
    generated_plots.append("file_size_distribution.png")

    return generated_plots


def write_readme(base_dir: Path) -> None:
    content = """# Raw Data Overview

This directory contains exploratory data analysis generated from:

- `outputs/manifest.csv`
- `outputs/hdf5_schema_summary.csv`

It does not use `outputs/fof_outcomes.csv` or `outputs/fragment_catalog.csv`.

## Contents

- `tables/`: dataset coverage tables, parameter counts, summary statistics, and sampled schema path summaries
- `plots/`: bar charts, coverage heatmaps, and file-size distribution plots
- `analysis_summary.txt`: textual summary of the current dataset and schema coverage

## Re-run

```bash
python scripts/eda_raw_data_overview.py \
  --manifest outputs/manifest.csv \
  --schema outputs/hdf5_schema_summary.csv \
  --eda-dir eda/raw_data_overview
```
"""
    (base_dir / "README.md").write_text(content, encoding="utf-8")


def summarise_top_counts(parameter_counts: pd.DataFrame, parameter: str, limit: int = 5) -> str:
    subset = parameter_counts[parameter_counts["parameter"] == parameter].sort_values(
        ["count", "value"], ascending=[False, True]
    )
    pairs = [f"{row.value} ({row.count})" for row in subset.head(limit).itertuples()]
    return ", ".join(pairs) if pairs else "none"


def write_analysis_summary(
    base_dir: Path,
    overview: pd.DataFrame,
    parameter_counts: pd.DataFrame,
    schema_fields: pd.DataFrame,
) -> None:
    overview_row = overview.iloc[0]
    manifest_complete = bool(overview_row["manifest_appears_complete"])
    field_summary = ", ".join(
        f"{row.field_name}={'yes' if row.present_in_sample_schema else 'no'}"
        for row in schema_fields.itertuples()
    )
    lines = [
        "Raw-data overview only.",
        "",
        f"Manifest completeness: {int(overview_row['manifest_simulation_rows'])} rows excluding header; "
        f"expected {EXPECTED_SIMULATION_ROWS}. Complete={manifest_complete}.",
        f"Sampled schema coverage: {int(overview_row['schema_rows'])} schema rows across "
        f"{int(overview_row['unique_sampled_schema_files'])} sampled files.",
        "fof_outcomes.csv is currently incomplete/test-only and is not used here.",
        "",
        "Parameter coverage summary:",
        f"- mass_code: {summarise_top_counts(parameter_counts, 'mass_code')}",
        f"- resolution_code: {summarise_top_counts(parameter_counts, 'resolution_code')}",
        f"- periapsis_code: {summarise_top_counts(parameter_counts, 'periapsis_code')}",
        f"- velocity_code: {summarise_top_counts(parameter_counts, 'velocity_code')}",
        f"- spin_code: {summarise_top_counts(parameter_counts, 'spin_code')}",
        f"- timestep: {summarise_top_counts(parameter_counts, 'timestep')}",
        f"- fof_linking_length: {summarise_top_counts(parameter_counts, 'fof_linking_length')}",
        "",
        "Schema availability summary:",
        f"- {field_summary}",
        "",
        "Interpretation:",
        "We can describe simulation parameter coverage and verify that sampled files expose the fields needed "
        "for FoF extraction and later analysis.",
        "We cannot use this stage for physical outcome EDA or ML targets because fof_outcomes.csv is not yet complete.",
        "",
        "Next step:",
        "Wait until outputs/fof_outcomes.csv reaches about 490 lines, then run scripts/eda_outcome_eda.py.",
    ]
    (base_dir / "analysis_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    schema_path = Path(args.schema)
    eda_dir = Path(args.eda_dir)
    report_dir = Path(args.report_dir) if args.report_dir else None
    tables_dir, plots_dir = ensure_dirs(eda_dir)

    manifest = load_csv(manifest_path)
    schema = load_csv(schema_path)

    overview = write_dataset_overview(manifest, schema, tables_dir)
    parameter_counts = write_parameter_counts(manifest, tables_dir)
    summarise_numeric_columns(manifest).to_csv(
        tables_dir / "parameter_summary_stats.csv", index=False
    )
    coverage_tables = write_coverage_tables(manifest, tables_dir)
    schema_fields, _ = write_schema_tables(schema, tables_dir)
    write_plots(manifest, coverage_tables, plots_dir, report_dir=report_dir)
    write_readme(eda_dir)
    write_analysis_summary(eda_dir, overview, parameter_counts, schema_fields)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
