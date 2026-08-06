#!/usr/bin/env python3
"""Audit independence and summarize large held-out BMF errors."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from create_trustability_slide_asset import ROW_SPECS
from train_physics_structured_surrogate import add_physics_features, load_canonical_dataset


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "report"
FIGURES_DIR = REPORT_ROOT / "figures"
OUTPUT_MD = REPORT_ROOT / "high_error_case_analysis.md"
OUTPUT_PNG = FIGURES_DIR / "high_error_case_analysis.png"

DATASET_PATH = ROOT / "extraction_outputs" / "bound_outcomes.csv"
PREDICTIONS_PATH = ROOT / "ml" / "physics_structured_surrogate" / "tables" / "predictions_with_trust_flags.csv"

PRIMARY_TARGET = "bound_mass_fraction"
BORDERLINE_LOW = 0.0771
BORDERLINE_HIGH = 0.1229
SUBSTANTIAL_RETENTION_THRESHOLD = 0.10
FALSE_POSITIVE_ACTUAL_MAX = 0.01
FALSE_POSITIVE_PRED_MIN = 0.05
FALSE_NEGATIVE_ACTUAL_MIN = 0.05
FALSE_NEGATIVE_PRED_MAX = 0.03


def format_float(value: float | int | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "N/A"
    value = float(value)
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def load_frame() -> pd.DataFrame:
    return add_physics_features(load_canonical_dataset(DATASET_PATH))


def load_predictions() -> pd.DataFrame:
    predictions = pd.read_csv(PREDICTIONS_PATH)
    predictions = predictions[predictions["target"] == PRIMARY_TARGET].copy()
    predictions["abs_error"] = pd.to_numeric(predictions["residual"], errors="coerce").abs()
    predictions["predicted"] = pd.to_numeric(predictions["predicted"], errors="coerce")
    predictions["bound_mass_fraction"] = pd.to_numeric(predictions["bound_mass_fraction"], errors="coerce")
    return predictions


def merge_predictions(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    keep_columns = [
        "fof_file",
        "physical_file",
        "asteroid_radius_km",
        "mass_log10_kg",
        "periapsis_Rm",
        "v_inf_kms",
        "spin_period_hr",
        "bound_mass_fraction",
    ]
    keep = frame.loc[:, [column for column in keep_columns if column in frame.columns]].drop_duplicates(subset=["fof_file"])
    merged = predictions.merge(
        keep,
        on=["fof_file", "physical_file", "mass_log10_kg", "periapsis_Rm", "v_inf_kms", "spin_period_hr", "bound_mass_fraction"],
        how="left",
    )
    merged["failure_mode"] = merged.apply(classify_failure_mode, axis=1)
    return merged


def classify_failure_mode(row: pd.Series) -> str:
    actual = float(row["bound_mass_fraction"])
    predicted = float(row["predicted"])
    if actual <= FALSE_POSITIVE_ACTUAL_MAX and predicted >= FALSE_POSITIVE_PRED_MIN:
        return "False positive near zero"
    if actual >= FALSE_NEGATIVE_ACTUAL_MIN and predicted <= FALSE_NEGATIVE_PRED_MAX:
        return "False negative retained mass"
    if predicted > actual:
        return "Overprediction"
    return "Underprediction"


def explicit_row_label(row: pd.Series) -> str:
    return f"R{int(row['row_rank'])}"


def row_parameter_label(row: pd.Series) -> str:
    spin_text = (
        f"spin={format_float(row['spin_period_hr'], 1)} hr"
        if np.isfinite(pd.to_numeric(row["spin_period_hr"], errors="coerce"))
        else "spin=not available"
    )
    return (
        f"mass={format_float(row['mass_log10_kg'], 1)}, peri={format_float(row['periapsis_Rm'], 1)}, "
        f"v_inf={format_float(row['v_inf_kms'], 1)}, {spin_text}, FoF={format_float(row['fof_linking_length'], 4)}"
    )


def simulation_parameter_label(row: pd.Series) -> str:
    spin_text = (
        f"spin={format_float(row['spin_period_hr'], 1)} hr"
        if np.isfinite(pd.to_numeric(row["spin_period_hr"], errors="coerce"))
        else "spin=not available"
    )
    return (
        f"mass={format_float(row['mass_log10_kg'], 1)}, peri={format_float(row['periapsis_Rm'], 1)}, "
        f"v_inf={format_float(row['v_inf_kms'], 1)}, {spin_text}"
    )


def cell_subset(frame: pd.DataFrame, row_col: str, col_col: str, row_value: float, col_value: float) -> pd.DataFrame:
    row_series = pd.to_numeric(frame[row_col], errors="coerce")
    col_series = pd.to_numeric(frame[col_col], errors="coerce")
    return frame[np.isclose(row_series, float(row_value)) & np.isclose(col_series, float(col_value))].copy()


def build_projection_table(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in ROW_SPECS:
        row_col = str(spec["row_col"])
        col_col = str(spec["col_col"])
        row_values = sorted(pd.to_numeric(merged[row_col], errors="coerce").dropna().unique())
        col_values = sorted(pd.to_numeric(merged[col_col], errors="coerce").dropna().unique())
        for row_value in row_values:
            for col_value in col_values:
                subset = cell_subset(merged, row_col, col_col, row_value, col_value)
                if subset.empty:
                    continue
                per_sim = subset.groupby("physical_file", dropna=False).agg(sim_mae=("abs_error", "mean")).reset_index()
                rows.append(
                    {
                        "comparison": str(spec["title"]),
                        "row_col": row_col,
                        "col_col": col_col,
                        "row_value": float(row_value),
                        "col_value": float(col_value),
                        "row_count": int(len(subset)),
                        "unique_simulation_count": int(subset["physical_file"].nunique()),
                        "row_weighted_mae": float(subset["abs_error"].mean()),
                        "simulation_weighted_mae": float(per_sim["sim_mae"].mean()),
                        "max_abs_error": float(subset["abs_error"].max()),
                    }
                )
    projection = pd.DataFrame.from_records(rows)
    projection = projection.sort_values(
        ["simulation_weighted_mae", "unique_simulation_count", "row_weighted_mae", "max_abs_error"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)
    return projection


def build_prediction_rows(merged: pd.DataFrame) -> pd.DataFrame:
    top_rows = merged.sort_values("abs_error", ascending=False).copy().reset_index(drop=True)
    top_rows["row_rank"] = np.arange(1, len(top_rows) + 1)
    return top_rows


def build_unique_simulation_table(merged: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        merged.groupby(
            ["physical_file", "mass_log10_kg", "periapsis_Rm", "v_inf_kms", "spin_period_hr", "asteroid_radius_km"],
            dropna=False,
        )
        .agg(
            row_count=("abs_error", "size"),
            actual_mean=("bound_mass_fraction", "mean"),
            predicted_mean=("predicted", "mean"),
            simulation_weighted_mae=("abs_error", "mean"),
            max_row_error=("abs_error", "max"),
            fof_variants=("fof_linking_length", lambda s: ", ".join(sorted({format_float(v, 4) for v in pd.to_numeric(s, errors="coerce").dropna()}))),
            high_confidence_any=("high_confidence", "any"),
            borderline_any=("borderline_bmf", "any"),
        )
        .reset_index()
        .sort_values(["simulation_weighted_mae", "max_row_error"], ascending=False)
        .reset_index(drop=True)
    )
    grouped["failure_mode"] = grouped.apply(
        lambda row: classify_failure_mode(pd.Series({"bound_mass_fraction": row["actual_mean"], "predicted": row["predicted_mean"]})),
        axis=1,
    )
    grouped["simulation_rank"] = np.arange(1, len(grouped) + 1)
    return grouped


def build_local_mass_slice(merged: pd.DataFrame) -> pd.DataFrame:
    subset = merged[
        (np.isclose(pd.to_numeric(merged["v_inf_kms"], errors="coerce"), 0.0))
        & (pd.to_numeric(merged["spin_period_hr"], errors="coerce").isna())
        & (merged["periapsis_Rm"].isin([1.2, 1.6]))
        & (merged["mass_log10_kg"].isin([19.0, 19.5, 20.0, 20.5, 21.0]))
    ].copy()
    per_sim = (
        subset.groupby(["periapsis_Rm", "mass_log10_kg", "physical_file"], dropna=False)
        .agg(sim_mae=("abs_error", "mean"), actual=("bound_mass_fraction", "mean"), predicted=("predicted", "mean"), rows=("abs_error", "size"))
        .reset_index()
    )
    summary = (
        per_sim.groupby(["periapsis_Rm", "mass_log10_kg"], dropna=False)
        .agg(
            unique_simulation_count=("physical_file", "nunique"),
            row_count=("rows", "sum"),
            actual=("actual", "mean"),
            predicted=("predicted", "mean"),
            simulation_weighted_mae=("sim_mae", "mean"),
        )
        .reset_index()
        .sort_values(["periapsis_Rm", "mass_log10_kg"])
    )
    return summary


def overall_metrics(merged: pd.DataFrame) -> dict[str, float]:
    per_sim = merged.groupby("physical_file", dropna=False).agg(simulation_weighted_mae=("abs_error", "mean")).reset_index()
    return {
        "row_weighted_mae": float(merged["abs_error"].mean()),
        "simulation_weighted_mae": float(per_sim["simulation_weighted_mae"].mean()),
        "row_p90": float(merged["abs_error"].quantile(0.9)),
        "unique_simulation_count": int(merged["physical_file"].nunique()),
        "row_count": int(len(merged)),
    }


def duplicate_audit(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    config_counts = (
        merged.groupby(["physical_file", "fof_linking_length"], dropna=False)
        .size()
        .reset_index(name="row_count_for_config")
        .sort_values(["row_count_for_config", "physical_file"], ascending=[False, True])
    )
    duplicated_configs = config_counts[config_counts["row_count_for_config"] > 1].copy()

    exact_full_row_duplicates = int(merged.duplicated(keep=False).sum())
    substantive_duplicate_mask = merged.duplicated(subset=[c for c in merged.columns if c not in {"fof_file", "row_index"}], keep=False)
    substantive_duplicates = merged[substantive_duplicate_mask].copy()

    config_deduped = merged.drop_duplicates(subset=["physical_file", "fof_linking_length"]).copy()
    config_per_sim = config_deduped.groupby("physical_file", dropna=False).agg(simulation_weighted_mae=("abs_error", "mean")).reset_index()
    stats = {
        "exact_full_row_duplicates": exact_full_row_duplicates,
        "substantive_duplicate_rows": int(substantive_duplicate_mask.sum()),
        "raw_row_count": int(len(merged)),
        "unique_physical_fof_configs": int(len(config_deduped)),
        "config_dedup_row_weighted_mae": float(config_deduped["abs_error"].mean()),
        "config_dedup_simulation_weighted_mae": float(config_per_sim["simulation_weighted_mae"].mean()),
        "config_dedup_top12_unique_physical": int(config_deduped.sort_values("abs_error", ascending=False).head(12)["physical_file"].nunique()),
    }
    return duplicated_configs, substantive_duplicates, stats


def confidence_summary(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_summary = (
        merged.groupby("high_confidence", dropna=False)
        .agg(row_count=("abs_error", "size"), row_weighted_mae=("abs_error", "mean"), unique_simulation_count=("physical_file", "nunique"))
        .reset_index()
    )
    sim_summary = (
        merged.groupby(["physical_file", "high_confidence"], dropna=False)
        .agg(simulation_weighted_mae=("abs_error", "mean"))
        .reset_index()
        .groupby("high_confidence", dropna=False)
        .agg(unique_simulation_count=("physical_file", "nunique"), simulation_weighted_mae=("simulation_weighted_mae", "mean"))
        .reset_index()
    )
    return row_summary, sim_summary


def specific_cell_checks(merged: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("Mass × periapsis", "mass_log10_kg", "periapsis_Rm", 19.5, 1.2),
        ("Mass × periapsis", "mass_log10_kg", "periapsis_Rm", 19.5, 1.6),
        ("Radius × velocity", "asteroid_radius_km", "v_inf_kms", 140.879956, 0.0),
    ]
    rows: list[dict[str, object]] = []
    for comparison, row_col, col_col, row_value, col_value in checks:
        subset = cell_subset(merged, row_col, col_col, row_value, col_value)
        per_sim = subset.groupby("physical_file", dropna=False).agg(simulation_weighted_mae=("abs_error", "mean")).reset_index()
        rows.append(
            {
                "comparison": comparison,
                "row_value": float(row_value),
                "col_value": float(col_value),
                "row_count": int(len(subset)),
                "unique_simulation_count": int(subset["physical_file"].nunique()),
                "row_weighted_mae": float(subset["abs_error"].mean()),
                "simulation_weighted_mae": float(per_sim["simulation_weighted_mae"].mean()),
                "physical_files": ", ".join(sorted(subset["physical_file"].dropna().unique())),
            }
        )
    return pd.DataFrame.from_records(rows)


def plot_diagnostics(projection: pd.DataFrame, top_rows: pd.DataFrame, mass_slice: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14.4, 10.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.92, 1.28], hspace=0.34, wspace=0.34)

    ax_proj = fig.add_subplot(gs[0, 0])
    proj_plot = projection.head(8).iloc[::-1].copy()
    proj_labels = [
        f"{row['comparison']}\n{format_float(row['row_value'], 1)} × {format_float(row['col_value'], 1)}"
        for _, row in proj_plot.iterrows()
    ]
    bars = ax_proj.barh(proj_labels, proj_plot["simulation_weighted_mae"], color="#c62828", alpha=0.85)
    ax_proj.set_title("Highest-error parameter-space projections", fontsize=12.8, fontweight="bold")
    ax_proj.set_xlabel("Simulation-weighted mean absolute error")
    for bar, (_, row) in zip(bars, proj_plot.iterrows()):
        sparse = " sparse" if int(row["unique_simulation_count"]) in {1, 2} else ""
        ax_proj.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"unique={int(row['unique_simulation_count'])}, rows={int(row['row_count'])}{sparse}",
            va="center",
            fontsize=8.4,
        )
    ax_proj.grid(axis="x", alpha=0.22)
    ax_proj.text(
        0.0,
        1.015,
        "Ranked projections can overlap. A radius × velocity hotspot can be the same cases as a mass × periapsis hotspot.",
        transform=ax_proj.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.7,
        color="#555555",
    )

    ax_rows = fig.add_subplot(gs[0, 1])
    row_plot = top_rows.head(8).iloc[::-1].copy()
    row_plot["row_rank"] = np.arange(len(row_plot), 0, -1)
    y = np.arange(len(row_plot))
    ax_rows.hlines(y, row_plot["bound_mass_fraction"], row_plot["predicted"], color="#9e9e9e", linewidth=1.9)
    ax_rows.scatter(row_plot["bound_mass_fraction"], y, color="#1565c0", label="Actual BMF", zorder=3)
    ax_rows.scatter(row_plot["predicted"], y, color="#d32f2f", label="Predicted BMF", zorder=3)
    ax_rows.axvline(SUBSTANTIAL_RETENTION_THRESHOLD, color="#c7c7c7", linestyle="--", linewidth=1.2)
    ax_rows.set_yticks(y)
    ax_rows.set_yticklabels([f"R{rank}" for rank in range(8, 0, -1)], fontsize=9)
    ax_rows.set_xlabel("Bound mass fraction")
    ax_rows.set_title("Worst prediction rows: actual vs predicted", fontsize=12.8, fontweight="bold")
    ax_rows.legend(frameon=False, loc="lower right")
    ax_rows.grid(axis="x", alpha=0.22)
    ax_rows.text(
        0.0,
        1.015,
        "Complete row definitions are listed in the markdown table. `R1` is the single highest-error row.",
        transform=ax_rows.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.7,
        color="#555555",
    )

    ax_slice = fig.add_subplot(gs[1, :])
    count_lines: list[str] = []
    for periapsis, color in [(1.2, "#1e88e5"), (1.6, "#ef6c00")]:
        sub = mass_slice[np.isclose(mass_slice["periapsis_Rm"], periapsis)].sort_values("mass_log10_kg")
        ax_slice.plot(sub["mass_log10_kg"], sub["actual"], marker="o", color=color, linewidth=2.2, label=f"Actual, peri={periapsis}")
        ax_slice.plot(sub["mass_log10_kg"], sub["predicted"], marker="s", color=color, linestyle="--", linewidth=2.0, label=f"Predicted, peri={periapsis}")
        count_lines.append(
            f"peri={format_float(periapsis,1)}: "
            + ", ".join(
                [
                    f"{format_float(mass,1)} -> rows={int(rows)}, unique={int(unique)}"
                    for mass, rows, unique in zip(sub["mass_log10_kg"], sub["row_count"], sub["unique_simulation_count"])
                ]
            )
        )
    ax_slice.axvline(19.5, color="#616161", linestyle=":", linewidth=1.5)
    ax_slice.axhline(SUBSTANTIAL_RETENTION_THRESHOLD, color="#d0d0d0", linestyle="--", linewidth=1.1, zorder=0)
    ax_slice.set_title("Local zero-velocity, non-spinning slice around the mass=19.5 failure", fontsize=13, fontweight="bold")
    ax_slice.set_xlabel("log10 asteroid mass (kg)")
    ax_slice.set_ylabel("Bound mass fraction")
    ax_slice.grid(alpha=0.24)
    ax_slice.legend(frameon=False, ncol=2, loc="lower right")
    ax_slice.text(19.52, 0.282, "Failure slice: mass=19.5", fontsize=9.3, color="#555555", ha="left", va="top")
    ax_slice.text(0.01, 1.02, r"$v_{\infty} = 0$ km/s", transform=ax_slice.transAxes, ha="left", va="bottom", fontsize=9.3, color="#555555")

    pred_12 = float(mass_slice[np.isclose(mass_slice["periapsis_Rm"], 1.2) & np.isclose(mass_slice["mass_log10_kg"], 19.5)]["predicted"].iloc[0])
    pred_16 = float(mass_slice[np.isclose(mass_slice["periapsis_Rm"], 1.6) & np.isclose(mass_slice["mass_log10_kg"], 19.5)]["predicted"].iloc[0])
    ax_slice.annotate(
        "peri=1.2: underprediction",
        xy=(19.5, pred_12),
        xytext=(19.62, 0.048),
        arrowprops={"arrowstyle": "-", "color": "#1e88e5", "lw": 1.0},
        color="#1e88e5",
        fontsize=9.0,
    )
    ax_slice.annotate(
        "peri=1.6: overprediction",
        xy=(19.5, pred_16),
        xytext=(19.62, 0.104),
        arrowprops={"arrowstyle": "-", "color": "#ef6c00", "lw": 1.0},
        color="#ef6c00",
        fontsize=9.0,
    )
    ax_slice.text(
        0.01,
        0.02,
        "Row and unique simulation counts along the slice:\n" + "\n".join(count_lines),
        transform=ax_slice.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.1,
        color="#444444",
        bbox={"facecolor": "white", "edgecolor": "#d0d0d0", "alpha": 0.9, "boxstyle": "round,pad=0.25"},
    )

    fig.suptitle("Held-out BMF Error Diagnostics", fontsize=17.5, fontweight="bold", y=0.985)
    fig.text(
        0.05,
        0.02,
        "The largest errors are concentrated near outcome transitions. At mass=19.5, the surrogate smooths over a sharp "
        "periapsis-dependent change and reverses the local retained-mass ordering. Some ranked projections describe the "
        "same underlying cases.",
        ha="left",
        va="bottom",
        fontsize=9.8,
        color="#333333",
    )
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def markdown_table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ]


def write_report(
    merged: pd.DataFrame,
    projection: pd.DataFrame,
    top_rows: pd.DataFrame,
    unique_sims: pd.DataFrame,
    mass_slice: pd.DataFrame,
    cell_checks: pd.DataFrame,
    metrics: dict[str, float],
    conf_rows: pd.DataFrame,
    conf_sims: pd.DataFrame,
    duplicated_configs: pd.DataFrame,
    substantive_duplicates: pd.DataFrame,
    duplicate_stats: dict[str, float],
) -> None:
    top_12_rows = top_rows.head(12).copy()
    top_12_unique_count = int(top_12_rows["physical_file"].nunique())

    proj_rows = []
    for _, row in projection.head(8).iterrows():
        proj_rows.append(
            [
                str(row["comparison"]),
                format_float(row["row_value"], 1),
                format_float(row["col_value"], 1),
                str(int(row["unique_simulation_count"])),
                str(int(row["row_count"])),
                format_float(row["simulation_weighted_mae"], 4),
                format_float(row["row_weighted_mae"], 4),
            ]
        )

    top_row_rows = []
    for _, row in top_12_rows.iterrows():
        top_row_rows.append(
            [
                explicit_row_label(row),
                row["physical_file"],
                row_parameter_label(row),
                format_float(row["bound_mass_fraction"], 4),
                format_float(row["predicted"], 4),
                format_float(row["abs_error"], 4),
                row["failure_mode"],
            ]
        )

    unique_rows = []
    for _, row in unique_sims.head(12).iterrows():
        unique_rows.append(
            [
                f"S{int(row['simulation_rank'])}",
                row["physical_file"],
                simulation_parameter_label(row),
                str(int(row["row_count"])),
                row["fof_variants"],
                format_float(row["actual_mean"], 4),
                format_float(row["predicted_mean"], 4),
                format_float(row["simulation_weighted_mae"], 4),
                row["failure_mode"],
            ]
        )

    cell_rows = []
    for _, row in cell_checks.iterrows():
        cell_rows.append(
            [
                str(row["comparison"]),
                format_float(row["row_value"], 1),
                format_float(row["col_value"], 1),
                str(int(row["unique_simulation_count"])),
                str(int(row["row_count"])),
                format_float(row["simulation_weighted_mae"], 4),
                format_float(row["row_weighted_mae"], 4),
                row["physical_files"],
            ]
        )

    conf_row_rows = []
    for _, row in conf_rows.sort_values("high_confidence", ascending=False).iterrows():
        conf_row_rows.append(
            [
                "High confidence" if bool(row["high_confidence"]) else "Remaining rows",
                str(int(row["row_count"])),
                str(int(row["unique_simulation_count"])),
                format_float(row["row_weighted_mae"], 4),
            ]
        )

    conf_sim_rows = []
    for _, row in conf_sims.sort_values("high_confidence", ascending=False).iterrows():
        conf_sim_rows.append(
            [
                "High confidence" if bool(row["high_confidence"]) else "Remaining rows",
                str(int(row["unique_simulation_count"])),
                format_float(row["simulation_weighted_mae"], 4),
            ]
        )

    slice_rows = []
    for _, row in mass_slice.iterrows():
        slice_rows.append(
            [
                format_float(row["periapsis_Rm"], 1),
                format_float(row["mass_log10_kg"], 1),
                str(int(row["unique_simulation_count"])),
                str(int(row["row_count"])),
                format_float(row["actual"], 4),
                format_float(row["predicted"], 4),
                format_float(row["simulation_weighted_mae"], 4),
            ]
        )

    duplicate_rows = []
    for _, row in duplicated_configs.iterrows():
        subset = merged[
            (merged["physical_file"] == row["physical_file"])
            & np.isclose(pd.to_numeric(merged["fof_linking_length"], errors="coerce"), float(row["fof_linking_length"]))
        ].copy()
        varying = [column for column in merged.columns if subset[column].nunique(dropna=False) > 1]
        if varying == ["fof_file", "row_index"]:
            note = "Identifier-only duplicate: same values in all non-identifier columns."
        else:
            note = "Distinct within config by: " + ", ".join(varying)
        duplicate_rows.append(
            [
                row["physical_file"],
                format_float(row["fof_linking_length"], 4),
                str(int(row["row_count_for_config"])),
                note,
            ]
        )

    lines = [
        "# High-error case analysis for the BMF surrogate",
        "",
        f"Generated on July 30, 2026 from [`predictions_with_trust_flags.csv`](/Users/nny124/irp/ml/physics_structured_surrogate/tables/predictions_with_trust_flags.csv:1) and [`bound_outcomes.csv`](/Users/nny124/irp/extraction_outputs/bound_outcomes.csv:1).",
        "",
        f"![High-error diagnostics](figures/{OUTPUT_PNG.name})",
        "",
        "## Independence audit",
        f"- The held-out table has `{int(metrics['row_count'])}` prediction rows but only `{int(metrics['unique_simulation_count'])}` unique `physical_file` simulations.",
        f"- After collapsing to unique `physical_file + fof_linking_length` configurations, there are `{int(duplicate_stats['unique_physical_fof_configs'])}` configurations.",
        "- Repeated rows from the same `physical_file` are FoF linking-length variants. They probe post-processing sensitivity, not independent coverage of physical parameter space.",
        f"- Overall row-weighted MAE is `{format_float(metrics['row_weighted_mae'], 4)}`.",
        f"- Overall simulation-weighted MAE, computed by averaging within `physical_file` before averaging across simulations, is `{format_float(metrics['simulation_weighted_mae'], 4)}`.",
        f"- Among the 12 worst prediction rows, only `{top_12_unique_count}` unique physical simulations appear.",
        f"- Exact full-row duplicates across every column: `{int(duplicate_stats['exact_full_row_duplicates'])}` rows. Strict full-row duplication is therefore absent.",
        f"- Rows duplicated across all substantive columns except `fof_file` and `row_index`: `{int(duplicate_stats['substantive_duplicate_rows'])}` rows.",
        "",
        "## Duplicate configuration audit",
        *markdown_table(
            ["Physical file", "FoF linking length", "Rows with same physical-file/FoF key", "Audit result"],
            duplicate_rows,
        ),
        "",
        f"- The audited duplicate-config table explains why each `mass=19.5` simulation has five rows but only four unique FoF linking lengths: the numeric FoF value `0.002` appears twice, once as `_fof_0.0020_...` and once as `_fof_0.002_...`.",
        "- For `Ma_xp_A1950_n65_r16_v00_90000.hdf5`, those two `0.002` rows are substantively identical and differ only in `fof_file` formatting and `row_index`.",
        "- For `Ma_xp_A1950_n65_r12_v00_90000.hdf5`, the two `0.002` rows share the same numeric FoF key but differ in fragment counts and derived mass statistics, so they are not exact duplicates.",
        f"- Because there are no exact full-row duplicates, the main report keeps the raw rows for row-level summaries. For reference, configuration-deduplicated MAE would be `{format_float(duplicate_stats['config_dedup_row_weighted_mae'], 4)}` row-weighted and `{format_float(duplicate_stats['config_dedup_simulation_weighted_mae'], 4)}` simulation-weighted.",
        f"- On the configuration-deduplicated table, the top-12 rows span `{int(duplicate_stats['config_dedup_top12_unique_physical'])}` unique physical simulations instead of `{top_12_unique_count}` in the raw-row table.",
        "",
        "## Specific hotspot checks",
        *markdown_table(
            ["Comparison", "Row value", "Column value", "Unique SPH simulations", "FoF-derived rows", "Simulation-weighted MAE", "Row-weighted MAE", "Physical files"],
            cell_rows,
        ),
        "",
        "The `mass=19.5, peri=1.2` cell and the `mass=19.5, peri=1.6` cell each contain five FoF-derived rows from one unique physical simulation. "
        "The `radius≈140.9 km, v_inf=0` hotspot contains ten rows from two unique simulations: those same two `mass=19.5` cases viewed through a different projection.",
        "",
        "## Highest-error parameter-space projections",
        *markdown_table(
            ["Comparison", "Row value", "Column value", "Unique SPH simulations", "FoF-derived rows", "Simulation-weighted MAE", "Row-weighted MAE"],
            proj_rows,
        ),
        "",
        "These rankings are overlapping projections, not independent regions. A high bar in `radius × velocity` can refer to the same underlying simulations already counted in `mass × periapsis`.",
        "",
        "## Worst prediction rows",
        *markdown_table(
            ["Row ID", "Physical file", "Parameters", "Actual BMF", "Predicted BMF", "Absolute error", "Failure mode"],
            top_row_rows,
        ),
        "",
        "## Worst unique simulations",
        *markdown_table(
            ["Sim ID", "Physical file", "Parameters", "FoF-derived rows", "FoF linking lengths", "Mean actual BMF", "Mean predicted BMF", "Simulation-weighted MAE", "Failure mode"],
            unique_rows,
        ),
        "",
        "## Row definitions used in the figure",
        "- The upper-right panel shows the worst prediction rows as `R1` to `R8`.",
        "- Each `R` label maps to the full parameter row in the table above, including FoF linking length.",
        "",
        "## Boundary definitions",
        f"- `borderline_bmf` is defined in the trust-flag pipeline as `predicted` between `{format_float(BORDERLINE_LOW, 4)}` and `{format_float(BORDERLINE_HIGH, 4)}` inclusive. This is a band around the substantial-retention threshold `BMF = {format_float(SUBSTANTIAL_RETENTION_THRESHOLD, 2)}`.",
        f"- `False positive near zero` means actual BMF `<= {format_float(FALSE_POSITIVE_ACTUAL_MAX, 2)}` and predicted BMF `>= {format_float(FALSE_POSITIVE_PRED_MIN, 2)}`.",
        f"- `False negative retained mass` means actual BMF `>= {format_float(FALSE_NEGATIVE_ACTUAL_MIN, 2)}` and predicted BMF `<= {format_float(FALSE_NEGATIVE_PRED_MAX, 2)}`.",
        "- `Overprediction` means predicted BMF is greater than actual BMF after excluding the two special cases above.",
        "- `Underprediction` means predicted BMF is less than or equal to actual BMF after excluding the two special cases above.",
        "- The `borderline_bmf` flag refers to the `BMF = 0.10` substantial-retention threshold, not to the near-zero boundary.",
        "",
        "## Confidence-flag validation",
        *markdown_table(["Group", "Rows", "Unique SPH simulations", "Row-weighted MAE"], conf_row_rows),
        "",
        *markdown_table(["Group", "Unique SPH simulations", "Simulation-weighted MAE"], conf_sim_rows),
        "",
        "- The `high_confidence` flag in [`train_physics_structured_surrogate.py`](/Users/nny124/irp/scripts/train_physics_structured_surrogate.py:796) is based on training-range inclusion, edge status, sparse-bin status, model-spread threshold, and the predicted `borderline_bmf` band.",
        "- It does not use the true target, residual, or held-out error, so the comparison is not circular on that point.",
        "- Lower MAE in the high-confidence group is therefore an association with the screening rule, not proof of perfect calibration.",
        "",
        "## Local zero-velocity, non-spinning slice",
        *markdown_table(
            ["Periapsis", "log10 asteroid mass (kg)", "Unique SPH simulations", "FoF-derived rows", "Mean actual BMF", "Mean predicted BMF", "Simulation-weighted MAE"],
            slice_rows,
        ),
        "",
        "## Revised interpretation",
        "- The observed cases are consistent with a sharp local transition, but unique-simulation coverage is too sparse to distinguish physical regime behaviour from sampling limitations.",
        "- The present `mass=19.5` failure cluster is dominated by one unique simulation at periapsis `1.2` and one unique simulation at periapsis `1.6`, each repeated over FoF linking lengths.",
        "- Repeated FoF variants should therefore be interpreted as post-processing sensitivity checks, not as independent support for the physical parameter-space cell.",
        "- The radius–velocity hotspot is not a separate failure region; it is a re-projection of those same underlying simulations.",
        "",
        "## Practical conclusion",
        "- The previous wording about `reasonable support` from five rows was not justified once independence is audited at the `physical_file` level.",
        "- The strongest conclusion that remains is about sensitivity: these cases are unstable across retained-mass outcome boundaries and remain difficult for the surrogate.",
        "- If you want a follow-up, the next clean extension is to rebuild the broader trust figure so its blue support panels also use unique `physical_file` counts rather than raw FoF-derived row counts.",
    ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    frame = load_frame()
    merged = merge_predictions(frame, load_predictions())
    projection = build_projection_table(merged)
    top_rows = build_prediction_rows(merged)
    unique_sims = build_unique_simulation_table(merged)
    mass_slice = build_local_mass_slice(merged)
    cell_checks = specific_cell_checks(merged)
    metrics = overall_metrics(merged)
    conf_rows, conf_sims = confidence_summary(merged)
    duplicated_configs, substantive_duplicates, duplicate_stats = duplicate_audit(merged)
    plot_diagnostics(projection, top_rows, mass_slice)
    write_report(
        merged,
        projection,
        top_rows,
        unique_sims,
        mass_slice,
        cell_checks,
        metrics,
        conf_rows,
        conf_sims,
        duplicated_configs,
        substantive_duplicates,
        duplicate_stats,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
