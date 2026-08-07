#!/usr/bin/env python3
"""Create parameter-family and feature-engineering figures for the BMF surrogate."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_physics_structured_surrogate import (
    OUTPUT_ROOT,
    PLOTS_DIR,
    PRIMARY_TARGET,
    TABLES_DIR,
    add_physics_features,
    build_group_folds,
    determine_promoted_model,
    ensure_output_dirs,
    evaluate_model_config_oof,
    feature_columns_for_set,
    load_canonical_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = OUTPUT_ROOT / "reports"
FIG1_PATH = PLOTS_DIR / "original_physical_parameter_importance.png"
FIG2_PATH = PLOTS_DIR / "physics_derived_feature_contribution.png"
REPORT_PATH = REPORTS_DIR / "physical_parameter_importance_report.md"
FAMILY_RESULTS_PATH = TABLES_DIR / "physical_parameter_family_ablation.csv"
FEATURE_STAGE_RESULTS_PATH = TABLES_DIR / "feature_engineering_stage_comparison.csv"
SIMPLE_TRANSFORM_RESULTS_PATH = TABLES_DIR / "simple_transform_checks.csv"

OUTCOME_DERIVED_FEATURES = {"largest_fragment_mass_fraction"}
NON_INFERENCE_SAFE_FEATURES = {"largest_fragment_mass_fraction"}

RAW_STAGE_FEATURES = [
    "mass_log10_kg",
    "particle_log10",
    "periapsis_Rm",
    "v_inf_kms",
    "spin_period_hr",
    "spin_axis",
    "has_explicit_spin",
    "special_case_code",
    "fof_linking_length",
]

SIMPLE_TRANSFORM_FEATURES = [
    "v_inf_squared",
    "periapsis_inverse",
    "spin_frequency_hr_inv",
    "has_spin",
    "asteroid_radius_km",
]

COMPOSITE_FEATURES = [
    "encounter_eccentricity_proxy",
    "angular_momentum_proxy",
    "time_within_2_mars_radii_hr",
    "time_within_tidal_disruption_hr",
    "particle_mass_proxy",
    "mass_resolution_interaction",
]

FAMILY_SPECS = [
    {
        "family": "Periapsis",
        "raw_parameter": "periapsis",
        "features_to_remove": [
            "periapsis_Rm",
            "periapsis_inverse",
            "encounter_eccentricity_proxy",
            "angular_momentum_proxy",
            "time_within_2_mars_radii_hr",
            "time_within_tidal_disruption_hr",
        ],
        "note": "Removes the direct closest-approach variable and all engineered features that depend on periapsis.",
    },
    {
        "family": "Velocity",
        "raw_parameter": "v_inf",
        "features_to_remove": [
            "v_inf_kms",
            "v_inf_squared",
            "encounter_eccentricity_proxy",
            "angular_momentum_proxy",
            "time_within_2_mars_radii_hr",
            "time_within_tidal_disruption_hr",
        ],
        "note": "Removes encounter speed and every engineered feature that explicitly uses velocity.",
    },
    {
        "family": "Mass",
        "raw_parameter": "asteroid mass",
        "features_to_remove": [
            "mass_log10_kg",
            "asteroid_radius_km",
            "particle_mass_proxy",
            "mass_resolution_interaction",
        ],
        "note": "Mass overlaps with radius because radius is deterministically derived from mass in the current pipeline.",
    },
    {
        "family": "Radius",
        "raw_parameter": "asteroid radius",
        "features_to_remove": [
            "asteroid_radius_km",
        ],
        "note": "Radius is a deterministic mass-derived transform, so this family intentionally overlaps with mass.",
    },
    {
        "family": "Spin",
        "raw_parameter": "spin",
        "features_to_remove": [
            "spin_period_hr",
            "spin_axis",
            "has_explicit_spin",
            "has_spin",
            "spin_frequency_hr_inv",
        ],
        "note": "Removes spin magnitude, axis, and spin helper transforms.",
    },
    {
        "family": "Resolution",
        "raw_parameter": "resolution",
        "features_to_remove": [
            "particle_log10",
            "particle_mass_proxy",
            "mass_resolution_interaction",
        ],
        "note": "Resolution propagates into particle-mass and mass-resolution interaction features.",
    },
    {
        "family": "FoF Linking Length",
        "raw_parameter": "FoF linking length",
        "features_to_remove": [
            "fof_linking_length",
        ],
        "note": "Included because it is currently supplied at inference time in the demo, even though it is a post-processing control rather than an encounter parameter.",
    },
]


def evaluate_feature_set(
    frame: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    params: dict[str, object] | None,
    label: str,
) -> dict[str, object]:
    metrics, _, _ = evaluate_model_config_oof(
        frame,
        PRIMARY_TARGET,
        feature_columns,
        fold_assignments,
        model_name,
        params,
    )
    row = metrics.iloc[0].to_dict()
    row["label"] = label
    row["feature_columns"] = json.dumps(feature_columns)
    return row


def build_family_ablation_table(
    frame: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    promoted: dict[str, object],
) -> pd.DataFrame:
    full_features = feature_columns_for_set(str(promoted["feature_set"]), bool(promoted["include_physics_features"]))
    safe_full_features = [feature for feature in full_features if feature not in OUTCOME_DERIVED_FEATURES]
    baseline = evaluate_feature_set(
        frame,
        fold_assignments,
        safe_full_features,
        str(promoted["model_name"]),
        promoted["params"],
        "Full promoted model without outcome-derived features",
    )

    rows: list[dict[str, object]] = []
    for spec in FAMILY_SPECS:
        removed = [feature for feature in safe_full_features if feature in spec["features_to_remove"]]
        kept = [feature for feature in safe_full_features if feature not in spec["features_to_remove"]]
        result = evaluate_feature_set(
            frame,
            fold_assignments,
            kept,
            str(promoted["model_name"]),
            promoted["params"],
            spec["family"],
        )
        rows.append(
            {
                "family": spec["family"],
                "raw_parameter": spec["raw_parameter"],
                "note": spec["note"],
                "baseline_r2": baseline["r2"],
                "baseline_mae": baseline["mae"],
                "ablation_r2": result["r2"],
                "ablation_mae": result["mae"],
                "delta_r2": baseline["r2"] - result["r2"],
                "delta_mae": result["mae"] - baseline["mae"],
                "removed_features": json.dumps(removed),
                "retained_feature_count": len(kept),
            }
        )
    output = pd.DataFrame(rows).sort_values("delta_r2", ascending=False).reset_index(drop=True)
    output.to_csv(FAMILY_RESULTS_PATH, index=False)
    return output


def build_feature_engineering_tables(
    frame: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    promoted: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del promoted
    model_name = "random_forest"
    params = None

    stage_specs = [
        ("Raw physical inputs", RAW_STAGE_FEATURES),
        ("Raw + simple transforms", RAW_STAGE_FEATURES + SIMPLE_TRANSFORM_FEATURES),
        ("Raw + composite physics features", RAW_STAGE_FEATURES + COMPOSITE_FEATURES),
        ("Full inference-safe feature set", RAW_STAGE_FEATURES + SIMPLE_TRANSFORM_FEATURES + COMPOSITE_FEATURES),
        (
            "Full current feature set",
            RAW_STAGE_FEATURES + SIMPLE_TRANSFORM_FEATURES + COMPOSITE_FEATURES + ["largest_fragment_mass_fraction"],
        ),
    ]
    baseline_features = stage_specs[0][1]
    baseline = evaluate_feature_set(frame, fold_assignments, baseline_features, model_name, params, stage_specs[0][0])

    stage_rows: list[dict[str, object]] = []
    for label, features in stage_specs:
        result = evaluate_feature_set(frame, fold_assignments, features, model_name, params, label)
        stage_rows.append(
            {
                "stage": label,
                "r2": result["r2"],
                "mae": result["mae"],
                "delta_r2_vs_raw": result["r2"] - baseline["r2"],
                "delta_mae_vs_raw": baseline["mae"] - result["mae"],
                "includes_outcome_derived_feature": any(feature in NON_INFERENCE_SAFE_FEATURES for feature in features),
                "feature_columns": json.dumps(features),
            }
        )
    stage_frame = pd.DataFrame(stage_rows)
    stage_frame.to_csv(FEATURE_STAGE_RESULTS_PATH, index=False)

    transform_specs = [
        ("Raw only", RAW_STAGE_FEATURES),
        ("Raw + v_inf_squared", RAW_STAGE_FEATURES + ["v_inf_squared"]),
        ("Raw + periapsis_inverse", RAW_STAGE_FEATURES + ["periapsis_inverse"]),
        ("Raw + spin_frequency", RAW_STAGE_FEATURES + ["spin_frequency_hr_inv"]),
        ("Raw + asteroid_radius", RAW_STAGE_FEATURES + ["asteroid_radius_km"]),
        ("Raw + all simple transforms", RAW_STAGE_FEATURES + SIMPLE_TRANSFORM_FEATURES),
    ]
    simple_rows: list[dict[str, object]] = []
    for label, features in transform_specs:
        result = evaluate_feature_set(frame, fold_assignments, features, model_name, params, label)
        simple_rows.append(
            {
                "transform_case": label,
                "r2": result["r2"],
                "mae": result["mae"],
                "delta_r2_vs_raw": result["r2"] - baseline["r2"],
                "delta_mae_vs_raw": baseline["mae"] - result["mae"],
                "feature_columns": json.dumps(features),
            }
        )
    simple_frame = pd.DataFrame(simple_rows)
    simple_frame.to_csv(SIMPLE_TRANSFORM_RESULTS_PATH, index=False)
    return stage_frame, simple_frame


def plot_family_importance(family_frame: pd.DataFrame) -> None:
    plot_df = family_frame.sort_values("delta_r2", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    bars = ax.barh(plot_df["family"], plot_df["delta_r2"], color="#1f4e79")
    ax.set_xlabel("Drop in grouped held-out $R^2$")
    ax.set_ylabel("Original physical parameter family")
    ax.set_title("Importance of Original Physical Parameters", fontsize=15, fontweight="semibold")
    ax.text(
        0.0,
        1.02,
        "Loss in grouped held-out performance after removing each parameter family and its derived representations",
        transform=ax.transAxes,
        fontsize=10,
        color="#444444",
    )
    ax.grid(axis="x", alpha=0.25)
    x_pad = max(plot_df["delta_r2"].max() * 0.03, 0.005)
    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        ax.text(
            bar.get_width() + x_pad,
            bar.get_y() + bar.get_height() / 2.0,
            f"ΔMAE +{row['delta_mae']:.4f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#333333",
        )
    fig.tight_layout()
    fig.savefig(FIG1_PATH, dpi=300)
    plt.close(fig)


def plot_feature_engineering(stage_frame: pd.DataFrame, simple_frame: pd.DataFrame) -> None:
    stage_plot = stage_frame.copy()
    simple_plot = simple_frame.copy()
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 9.0), height_ratios=[1.35, 1.0])

    colors = ["#1f4e79", "#4f772d", "#c97a00", "#0b6e4f", "#b23a48"]
    axes[0].bar(stage_plot["stage"], stage_plot["delta_r2_vs_raw"], color=colors)
    axes[0].axhline(0.0, color="#555555", linewidth=1.0)
    axes[0].set_ylabel("Change in grouped held-out $R^2$ vs raw baseline")
    axes[0].set_title("Contribution of Physics-Derived Features", fontsize=15, fontweight="semibold")
    axes[0].text(
        0.0,
        1.02,
        "Grouped cross-validation comparison against the raw-input random-forest baseline",
        transform=axes[0].transAxes,
        fontsize=10,
        color="#444444",
    )
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y", alpha=0.25)
    for idx, row in stage_plot.iterrows():
        axes[0].text(
            idx,
            row["delta_r2_vs_raw"] + (0.002 if row["delta_r2_vs_raw"] >= 0 else -0.006),
            f"R² {row['r2']:.3f}\nMAE {row['mae']:.4f}",
            ha="center",
            va="bottom" if row["delta_r2_vs_raw"] >= 0 else "top",
            fontsize=9,
        )

    axes[1].bar(simple_plot["transform_case"], simple_plot["delta_r2_vs_raw"], color="#7a8dad")
    axes[1].axhline(0.0, color="#555555", linewidth=1.0)
    axes[1].set_ylabel("Change in grouped held-out $R^2$ vs raw baseline")
    axes[1].set_title("Simple Transform Checks", fontsize=12, fontweight="semibold")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.25)
    for idx, row in simple_plot.iterrows():
        axes[1].text(
            idx,
            row["delta_r2_vs_raw"] + (0.0015 if row["delta_r2_vs_raw"] >= 0 else -0.0045),
            f"{row['delta_r2_vs_raw']:+.3f}",
            ha="center",
            va="bottom" if row["delta_r2_vs_raw"] >= 0 else "top",
            fontsize=9,
        )

    fig.subplots_adjust(hspace=0.42, top=0.92, bottom=0.10)
    fig.savefig(FIG2_PATH, dpi=300)
    plt.close(fig)


def format_removed_features(value: str) -> str:
    features = json.loads(value)
    return ", ".join(features) if features else "none"


def build_report(
    promoted: dict[str, object],
    family_frame: pd.DataFrame,
    stage_frame: pd.DataFrame,
    simple_frame: pd.DataFrame,
) -> None:
    baseline_family_r2 = float(family_frame["baseline_r2"].iloc[0])
    baseline_family_mae = float(family_frame["baseline_mae"].iloc[0])
    full_current = stage_frame[stage_frame["stage"] == "Full current feature set"].iloc[0]
    full_safe = stage_frame[stage_frame["stage"] == "Full inference-safe feature set"].iloc[0]
    raw_stage = stage_frame[stage_frame["stage"] == "Raw physical inputs"].iloc[0]
    best_family = family_frame.sort_values("delta_r2", ascending=False).iloc[0]
    weakest_family = family_frame.sort_values("delta_r2", ascending=True).iloc[0]
    best_safe_stage = stage_frame[~stage_frame["includes_outcome_derived_feature"]].sort_values("r2", ascending=False).iloc[0]

    lines = [
        "# Physical Parameter Importance and Physics-Feature Contribution",
        "",
        f"- Date: `2026-08-05`",
        f"- Dataset: `extraction_outputs/bound_outcomes.csv`",
        f"- Figure 1 evaluation: grouped cross-validation by `physical_file` using the promoted `{promoted['model_name']}` BMF surrogate",
        "- Figure 2 evaluation: grouped cross-validation by `physical_file` using the Random Forest surrogate family for staged feature-set comparisons",
        f"- Figure 1: `{FIG1_PATH.as_posix()}`",
        f"- Figure 2: `{FIG2_PATH.as_posix()}`",
        "",
        "## Method",
        "",
        "Figure 1 uses grouped ablation rather than single-column permutation importance.",
        "For each original parameter family, the analysis removes the raw parameter and every engineered feature that carries the same information, retrains the model on the same grouped folds, and measures the held-out performance loss relative to the promoted feature set after excluding outcome-derived features.",
        "The family ablations overlap, so their scores must not be added together or interpreted as percentages.",
        "",
        "Figure 2 asks a different question: whether feature engineering helped the surrogate.",
        "It compares grouped held-out performance for raw physical inputs, raw inputs plus simple transforms, raw inputs plus composite physics proxies, the full inference-safe feature set, and the full current feature set including `largest_fragment_mass_fraction`.",
        "",
        "## Numerical Results",
        "",
        f"- Figure 1 baseline without outcome-derived features: `R² = {baseline_family_r2:.4f}`, `MAE = {baseline_family_mae:.4f}`",
        f"- Figure 2 raw-input baseline: `R² = {float(raw_stage['r2']):.4f}`, `MAE = {float(raw_stage['mae']):.4f}`",
        f"- Best inference-safe staged model: `{best_safe_stage['stage']}` with `R² = {float(best_safe_stage['r2']):.4f}`, `MAE = {float(best_safe_stage['mae']):.4f}`",
        f"- Full current feature set: `R² = {float(full_current['r2']):.4f}`, `MAE = {float(full_current['mae']):.4f}`",
        f"- Full inference-safe feature set: `R² = {float(full_safe['r2']):.4f}`, `MAE = {float(full_safe['mae']):.4f}`",
        "",
        "### Figure 1 parameter-family ablations",
        "",
    ]
    for _, row in family_frame.sort_values("delta_r2", ascending=False).iterrows():
        lines.append(
            f"- `{row['family']}`: `ΔR² = {-row['delta_r2']:.4f}`, `ΔMAE = {row['delta_mae']:+.4f}` after removing `{format_removed_features(row['removed_features'])}`"
        )
    lines.extend(
        [
            "",
            "### Figure 2 staged comparisons",
            "",
        ]
    )
    for _, row in stage_frame.iterrows():
        extra = " (includes outcome-derived feature)" if bool(row["includes_outcome_derived_feature"]) else ""
        lines.append(
            f"- `{row['stage']}`{extra}: `R² = {row['r2']:.4f}`, `MAE = {row['mae']:.4f}`, `ΔR² vs raw = {row['delta_r2_vs_raw']:+.4f}`"
        )
    lines.extend(
        [
            "",
            "### Simple transform checks",
            "",
        ]
    )
    for _, row in simple_frame.iterrows():
        lines.append(
            f"- `{row['transform_case']}`: `R² = {row['r2']:.4f}`, `MAE = {row['mae']:.4f}`, `ΔR² vs raw = {row['delta_r2_vs_raw']:+.4f}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"Figure 1 answers “what physics matters?” In this dataset, the largest grouped-CV degradation comes from removing `{best_family['family']}`, while the smallest comes from removing `{weakest_family['family']}`.",
            "This is a stronger claim than the old permutation-importance plot because each family removal deletes correlated engineered representations rather than leaving them behind as substitutes.",
            "If velocity no longer looks weak after family ablation, that means the older single-column plot was understating its role because `v_inf_kms`, `v_inf_squared`, `encounter_eccentricity_proxy`, and `angular_momentum_proxy` overlap.",
            "",
            "Figure 2 answers “did encoding more physics help the ML?”",
            "Simple one-to-one transforms should be treated as convenience encodings for the tree model, not as independent physical effects.",
            "Composite proxies are more meaningful if they improve grouped validation beyond the raw-input baseline.",
            "",
            "## `largest_fragment_mass_fraction` check",
            "",
            "- Classification: `largest_fragment_mass_fraction` is produced by SPH/FoF post-processing, not by the original simulation setup.",
            "- Construction: the training code computes it from `largest_fragment_mass_kg / target_mass_kg`, and `largest_fragment_mass_kg` comes from bound or unbound fragment masses in the extracted outcome table.",
            "- Figure 1 treatment: excluded, because it is not an original physical input.",
            "- Demo-pipeline compatibility: not compatible with a pure pre-SPH inference pipeline. It requires outcome information that is unavailable before running or post-processing the simulation.",
            f"- Performance impact in this analysis: the full current feature set gains `ΔR² = {float(full_current['r2'] - full_safe['r2']):+.4f}` relative to the inference-safe full set, so any gain should be described as outcome-assisted rather than input-only prediction.",
            "",
            "## What Changed Relative to the Old Permutation-Importance Interpretation",
            "",
            "- The old single-column permutation plot measured marginal usefulness after correlated columns stayed in the model.",
            "- The new Figure 1 uses grouped retraining ablations, so it is appropriate for discussing original parameter families.",
            "- The new Figure 2 separates the question of physical importance from the question of whether feature engineering improved validation performance.",
            "",
            "## Presentation-Ready Conclusions",
            "",
            f"- Figure 1: `{best_family['family']}` is the strongest original parameter family in grouped ablation, so the slide can answer “what physics matters?” without conflating raw inputs with engineered duplicates.",
            f"- Figure 2: the best inference-safe engineered stage is `{best_safe_stage['stage']}` at `R² = {float(best_safe_stage['r2']):.4f}`, while the current full set reaches `R² = {float(full_current['r2']):.4f}` only by including an outcome-derived feature that is not valid for a pure input-only demo.",
        ]
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_output_dirs()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = add_physics_features(load_canonical_dataset(ROOT / "extraction_outputs" / "bound_outcomes.csv"))
    promoted = determine_promoted_model(ROOT / "extraction_outputs" / "bound_outcomes.csv")
    fold_assignments_path = TABLES_DIR / "fold_assignments.csv"
    fold_assignments = pd.read_csv(fold_assignments_path) if fold_assignments_path.exists() else build_group_folds(frame, frame["physical_file"].astype(str))
    family_frame = build_family_ablation_table(frame, fold_assignments, promoted)
    stage_frame, simple_frame = build_feature_engineering_tables(frame, fold_assignments, promoted)
    plot_family_importance(family_frame)
    plot_feature_engineering(stage_frame, simple_frame)
    build_report(promoted, family_frame, stage_frame, simple_frame)


if __name__ == "__main__":
    main()
