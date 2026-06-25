#!/usr/bin/env python3
"""EDA for bound vs unbound fragment extraction outputs."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FRAGMENT_NUMERIC_COLUMNS = [
    "fragment_particle_count",
    "fragment_mass_kg",
    "com_r_m",
    "com_speed_m_s",
    "specific_energy_J_kg",
    "fof_linking_length",
]

RUN_PARAMETER_COLUMNS = [
    "mass_code",
    "resolution_code",
    "periapsis_code",
    "velocity_code",
    "fof_linking_length",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bound/unbound EDA.")
    parser.add_argument(
        "--fragments",
        default="outputs/fragment_orbital_catalog.csv",
        help="Path to fragment_orbital_catalog.csv",
    )
    parser.add_argument(
        "--outcomes",
        default="outputs/bound_outcomes.csv",
        help="Path to bound_outcomes.csv",
    )
    parser.add_argument(
        "--log",
        default="outputs/bound_unbound_extraction_log.csv",
        help="Path to bound_unbound_extraction_log.csv",
    )
    parser.add_argument(
        "--eda-dir",
        default="eda/bound_eda",
        help="Output directory for EDA artifacts",
    )
    return parser.parse_args()


def ensure_dirs(base_dir: Path):
    tables_dir = base_dir / "tables"
    plots_dir = base_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, plots_dir


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def write_readme(base_dir: Path) -> None:
    content = """# Bound EDA

Generated artifacts for fragment-level and run-level bound vs unbound analysis.

Outputs:

- `tables/` contains summary tables, parameter aggregates, and sample rows.
- `plots/` contains fragment-level and run-level visualisations.

## Re-run

```bash
python scripts/eda/eda_bound_eda.py \
  --fragments outputs/fragment_orbital_catalog.csv \
  --outcomes outputs/bound_outcomes.csv \
  --log outputs/bound_unbound_extraction_log.csv \
  --eda-dir eda/bound_eda
```
"""
    (base_dir / "README.md").write_text(content, encoding="utf-8")


def write_dataset_overview(
    fragments: pd.DataFrame, outcomes: pd.DataFrame, log_df: pd.DataFrame, tables_dir: Path
) -> pd.DataFrame:
    is_bound = bool_series(fragments["is_bound"])
    success_mask = log_df["status"].astype(str).str.startswith("success")
    overview = pd.DataFrame(
        [
            {
                "fragment_rows": int(len(fragments)),
                "run_rows": int(len(outcomes)),
                "extraction_log_rows": int(len(log_df)),
                "successful_log_rows": int(success_mask.sum()),
                "fragment_bound_count": int(is_bound.sum()),
                "fragment_unbound_count": int((~is_bound).sum()),
                "fragment_bound_share": float(is_bound.mean()),
                "fragment_unbound_share": float((~is_bound).mean()),
                "runs_zero_bound_mass_fraction": int((outcomes["bound_mass_fraction"] == 0).sum()),
                "runs_mixed_bound_mass_fraction": int(
                    ((outcomes["bound_mass_fraction"] > 0) & (outcomes["bound_mass_fraction"] < 1)).sum()
                ),
                "runs_all_unbound_mass_fraction": int((outcomes["unbound_mass_fraction"] == 1).sum()),
            }
        ]
    )
    overview.to_csv(tables_dir / "dataset_overview.csv", index=False)
    return overview


def write_fragment_class_summary(fragments: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    frame = fragments.copy()
    frame["is_bound"] = bool_series(frame["is_bound"])
    rows = []
    for metric in FRAGMENT_NUMERIC_COLUMNS:
        values = numeric_series(frame, metric)
        for label, subset in values.groupby(frame["is_bound"]):
            clean = subset.dropna()
            rows.append(
                {
                    "metric": metric,
                    "is_bound": bool(label),
                    "count": int(clean.count()),
                    "min": clean.min() if not clean.empty else pd.NA,
                    "p25": clean.quantile(0.25) if not clean.empty else pd.NA,
                    "median": clean.median() if not clean.empty else pd.NA,
                    "mean": clean.mean() if not clean.empty else pd.NA,
                    "p75": clean.quantile(0.75) if not clean.empty else pd.NA,
                    "max": clean.max() if not clean.empty else pd.NA,
                    "std": clean.std() if len(clean) > 1 else pd.NA,
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(tables_dir / "fragment_class_summary.csv", index=False)
    return summary


def write_energy_sign_crosstab(fragments: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    frame = fragments.copy()
    frame["is_bound"] = bool_series(frame["is_bound"])
    frame["energy_negative"] = numeric_series(frame, "specific_energy_J_kg") < 0
    crosstab = pd.crosstab(frame["energy_negative"], frame["is_bound"], margins=True)
    crosstab.to_csv(tables_dir / "energy_sign_crosstab.csv")
    return crosstab


def write_extraction_status_summary(log_df: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    summary = (
        log_df.groupby("status", dropna=False)
        .agg(
            rows=("status", "size"),
            readable_true=("readable", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
            matched_physical_true=(
                "matched_physical_file_found",
                lambda s: int(pd.Series(s).fillna(False).astype(bool).sum()),
            ),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    summary.to_csv(tables_dir / "extraction_status_summary.csv", index=False)
    return summary


def write_parameter_bound_summary(outcomes: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    frames = []
    for parameter in RUN_PARAMETER_COLUMNS:
        grouped = (
            outcomes.groupby(parameter, dropna=False)
            .agg(
                n_runs=("fof_file", "size"),
                mean_bound_mass_fraction=("bound_mass_fraction", "mean"),
                median_bound_mass_fraction=("bound_mass_fraction", "median"),
                mean_bound_fragment_count=("bound_fragment_count", "mean"),
                zero_bound_run_share=("bound_mass_fraction", lambda s: float((s == 0).mean())),
            )
            .reset_index()
            .rename(columns={parameter: "parameter_value"})
        )
        grouped.insert(0, "parameter", parameter)
        frames.append(grouped)
    summary = pd.concat(frames, ignore_index=True)
    summary.to_csv(tables_dir / "parameter_bound_summary.csv", index=False)
    return summary


def write_sample_tables(fragments: pd.DataFrame, outcomes: pd.DataFrame, tables_dir: Path) -> None:
    sample_columns = [
        "fof_file",
        "group_id",
        "fragment_particle_count",
        "fragment_mass_kg",
        "com_r_m",
        "com_speed_m_s",
        "specific_energy_J_kg",
        "is_bound",
    ]
    edge_fragments = (
        fragments.assign(abs_specific_energy=numeric_series(fragments, "specific_energy_J_kg").abs())
        .sort_values("abs_specific_energy")
        .loc[:, sample_columns + ["abs_specific_energy"]]
        .head(15)
    )
    edge_fragments.to_csv(tables_dir / "threshold_edge_fragments.csv", index=False)

    run_columns = [
        "fof_file",
        "mass_code",
        "resolution_code",
        "periapsis_code",
        "velocity_code",
        "fof_linking_length",
        "n_fragments",
        "bound_fragment_count",
        "bound_mass_fraction",
        "largest_bound_fragment_mass_kg",
        "largest_unbound_fragment_mass_kg",
    ]
    outcomes.sort_values("bound_mass_fraction", ascending=False).loc[:, run_columns].head(15).to_csv(
        tables_dir / "top_bound_mass_fraction_runs.csv", index=False
    )
    outcomes[outcomes["bound_mass_fraction"] == 0].loc[:, run_columns].head(15).to_csv(
        tables_dir / "zero_bound_mass_fraction_runs.csv", index=False
    )


def make_class_balance_plot(fragments: pd.DataFrame, plots_dir: Path) -> None:
    is_bound = bool_series(fragments["is_bound"])
    counts = pd.Series(
        {"Unbound": int((~is_bound).sum()), "Bound": int(is_bound.sum())},
        name="fragment_count",
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#c44e52", "#4c72b0"]
    counts.plot(kind="bar", ax=ax, color=colors)
    ax.set_title("Fragment Class Balance")
    ax.set_ylabel("Fragments")
    ax.set_xlabel("")
    for idx, value in enumerate(counts.values):
        ax.text(idx, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots_dir / "fragment_class_balance.png", dpi=200)
    plt.close(fig)


def make_energy_histogram(fragments: pd.DataFrame, plots_dir: Path) -> None:
    frame = fragments.copy()
    frame["is_bound"] = bool_series(frame["is_bound"])
    energy = numeric_series(frame, "specific_energy_J_kg")
    transformed = np.sign(energy) * np.log10(1 + energy.abs())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        transformed[~frame["is_bound"]],
        bins=80,
        alpha=0.7,
        label="Unbound",
        color="#c44e52",
    )
    ax.hist(
        transformed[frame["is_bound"]],
        bins=80,
        alpha=0.7,
        label="Bound",
        color="#4c72b0",
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Signed log10 Specific Energy by Class")
    ax.set_xlabel("sign(E) * log10(1 + |specific_energy_J_kg|)")
    ax.set_ylabel("Fragments")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "specific_energy_signed_log10_by_class.png", dpi=200)
    plt.close(fig)


def make_speed_radius_plot(fragments: pd.DataFrame, plots_dir: Path) -> None:
    frame = fragments.copy()
    frame["is_bound"] = bool_series(frame["is_bound"])
    sample = pd.concat(
        [
            frame[frame["is_bound"]].sample(n=min(12000, int(frame["is_bound"].sum())), random_state=42),
            frame[~frame["is_bound"]].sample(
                n=min(12000, int((~frame["is_bound"]).sum())), random_state=42
            ),
        ],
        ignore_index=True,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color in [(False, "#c44e52"), (True, "#4c72b0")]:
        subset = sample[sample["is_bound"] == label]
        ax.scatter(
            subset["com_r_m"],
            subset["com_speed_m_s"],
            s=7,
            alpha=0.2,
            c=color,
            label="Bound" if label else "Unbound",
            edgecolors="none",
        )
    ax.set_title("Radius vs Speed Sample by Class")
    ax.set_xlabel("COM radius (m)")
    ax.set_ylabel("COM speed (m/s)")
    ax.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "radius_vs_speed_by_class.png", dpi=200)
    plt.close(fig)


def make_fragment_metric_boxplots(fragments: pd.DataFrame, plots_dir: Path) -> None:
    frame = fragments.copy()
    frame["is_bound"] = bool_series(frame["is_bound"])
    labels = ["Unbound", "Bound"]
    configs = [
        ("fragment_mass_kg", "Fragment Mass by Class", "Mass (kg)", "fragment_mass_by_class.png"),
        (
            "fragment_particle_count",
            "Fragment Particle Count by Class",
            "Particle count",
            "fragment_particle_count_by_class.png",
        ),
    ]
    for metric, title, ylabel, filename in configs:
        data = [
            numeric_series(frame[~frame["is_bound"]], metric).dropna(),
            numeric_series(frame[frame["is_bound"]], metric).dropna(),
        ]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.boxplot(data, tick_labels=labels, showfliers=False)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(plots_dir / filename, dpi=200)
        plt.close(fig)


def make_run_level_plots(outcomes: pd.DataFrame, plots_dir: Path) -> None:
    heatmap = (
        outcomes.pivot_table(
            index="periapsis_code",
            columns="velocity_code",
            values="bound_mass_fraction",
            aggfunc="mean",
        )
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(heatmap.fillna(0).values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(heatmap.columns)))
    ax.set_xticklabels(heatmap.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index)
    ax.set_title("Mean Bound Mass Fraction by Periapsis and Velocity")
    ax.set_xlabel("velocity_code")
    ax.set_ylabel("periapsis_code")
    fig.colorbar(image, ax=ax, label="Mean bound mass fraction")
    fig.tight_layout()
    fig.savefig(plots_dir / "bound_mass_fraction_heatmap_periapsis_velocity.png", dpi=200)
    plt.close(fig)

    by_linking = (
        outcomes.groupby("fof_linking_length", dropna=False)["bound_mass_fraction"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .sort_values("fof_linking_length")
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(by_linking["fof_linking_length"], by_linking["mean"], marker="o", label="Mean")
    ax.plot(by_linking["fof_linking_length"], by_linking["median"], marker="s", label="Median")
    ax.set_title("Bound Mass Fraction vs FoF Linking Length")
    ax.set_xlabel("FoF linking length")
    ax.set_ylabel("Bound mass fraction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "bound_mass_fraction_vs_fof_linking_length.png", dpi=200)
    plt.close(fig)


def write_analysis_summary(
    fragments: pd.DataFrame, outcomes: pd.DataFrame, log_df: pd.DataFrame, base_dir: Path
) -> None:
    is_bound = bool_series(fragments["is_bound"])
    energy_negative = numeric_series(fragments, "specific_energy_J_kg") < 0
    exact_match = int((energy_negative == is_bound).sum())
    summary = "\n".join(
        [
            f"Fragment rows: {len(fragments):,}",
            f"Run rows: {len(outcomes):,}",
            f"Extraction log rows: {len(log_df):,}",
            f"Bound fragments: {int(is_bound.sum()):,} ({is_bound.mean():.2%})",
            f"Unbound fragments: {int((~is_bound).sum()):,} ({(~is_bound).mean():.2%})",
            f"Energy sign matches label for {exact_match:,} / {len(fragments):,} fragments.",
            f"Runs with zero bound mass fraction: {int((outcomes['bound_mass_fraction'] == 0).sum()):,}",
            f"Runs with mixed bound mass fraction: {int(((outcomes['bound_mass_fraction'] > 0) & (outcomes['bound_mass_fraction'] < 1)).sum()):,}",
        ]
    )
    (base_dir / "analysis_summary.txt").write_text(summary + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    eda_dir = Path(args.eda_dir)
    tables_dir, plots_dir = ensure_dirs(eda_dir)

    fragments = load_csv(Path(args.fragments))
    outcomes = load_csv(Path(args.outcomes))
    log_df = load_csv(Path(args.log))

    write_readme(eda_dir)
    write_dataset_overview(fragments, outcomes, log_df, tables_dir)
    write_fragment_class_summary(fragments, tables_dir)
    write_energy_sign_crosstab(fragments, tables_dir)
    write_extraction_status_summary(log_df, tables_dir)
    write_parameter_bound_summary(outcomes, tables_dir)
    write_sample_tables(fragments, outcomes, tables_dir)
    make_class_balance_plot(fragments, plots_dir)
    make_energy_histogram(fragments, plots_dir)
    make_speed_radius_plot(fragments, plots_dir)
    make_fragment_metric_boxplots(fragments, plots_dir)
    make_run_level_plots(outcomes, plots_dir)
    write_analysis_summary(fragments, outcomes, log_df, eda_dir)


if __name__ == "__main__":
    main()
