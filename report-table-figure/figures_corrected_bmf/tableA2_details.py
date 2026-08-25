from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.model_training_scripts_corrected_bmf.helper_functions_ml import (
    PRIMARY_TARGET,
    add_physics_features,
    build_or_load_group_folds,
    evaluate_grouped_oof_regression,
    load_canonical_dataset,
)


SOURCE_PATH = ROOT / "extraction-outputs_corrected_bmf" / "tables" / "bound_outcomes.csv"
FIG_PATH = ROOT / "report-table-figure" / "figures_corrected_bmf" / "figureA1_used_in_report.png"
FOLDS_PATH = ROOT / "ml" / "trainingartifacts_corrected_bmf" / "tuned_physics_gradient_boosting" / "grouped_cv_fold_assignments.csv"

RAW_FEATURE_COLUMNS = [
    "mass_log10_kg",
    "periapsis_Rm",
    "v_inf_kms",
    "spin_period_hr",
    "spin_axis",
    "resolution_value",
    "fof_linking_length",
]
PHYSICS_FEATURE_COLUMNS = [
    "v_inf_squared",
    "periapsis_inverse",
    "angular_momentum_proxy",
    "spin_frequency_hr_inv",
    "asteroid_radius_km",
    "encounter_eccentricity_proxy",
    "time_within_2_mars_radii_hr",
    "time_within_tidal_disruption_hr",
]
SIMPLE_FEATURE_COLUMNS = [
    "v_inf_squared",
    "periapsis_inverse",
    "spin_frequency_hr_inv",
    "asteroid_radius_km",
]
COMBINED_PHYSICS_FEATURE_COLUMNS = [
    "angular_momentum_proxy",
    "encounter_eccentricity_proxy",
    "time_within_2_mars_radii_hr",
    "time_within_tidal_disruption_hr",
]
GB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.08,
    "max_depth": 3,
    "subsample": 0.8,
    "min_samples_leaf": 1,
    "random_state": 42,
}


def load_frame() -> pd.DataFrame:
    frame = load_canonical_dataset(SOURCE_PATH)
    frame = add_physics_features(frame.copy())
    frame = frame.loc[frame[PRIMARY_TARGET].notna()].copy()
    return frame


def evaluate_feature_contribution(frame: pd.DataFrame) -> dict[str, object]:
    folds = build_or_load_group_folds(frame, FOLDS_PATH)
    feature_sets = {
        "Raw": RAW_FEATURE_COLUMNS,
        "Raw + simple": RAW_FEATURE_COLUMNS + SIMPLE_FEATURE_COLUMNS,
        "Raw + physics": RAW_FEATURE_COLUMNS + COMBINED_PHYSICS_FEATURE_COLUMNS,
        "All": RAW_FEATURE_COLUMNS + PHYSICS_FEATURE_COLUMNS,
    }
    simple_checks = {
        r"v_inf^2": RAW_FEATURE_COLUMNS + ["v_inf_squared"],
        "1/r_p": RAW_FEATURE_COLUMNS + ["periapsis_inverse"],
        "f_spin": RAW_FEATURE_COLUMNS + ["spin_frequency_hr_inv"],
        "radius": RAW_FEATURE_COLUMNS + ["asteroid_radius_km"],
        "all simple": RAW_FEATURE_COLUMNS + SIMPLE_FEATURE_COLUMNS,
    }

    set_metrics: dict[str, dict[str, float | int]] = {}
    for label, columns in feature_sets.items():
        metrics, _ = evaluate_grouped_oof_regression(frame, columns, "gradient_boosting", GB_PARAMS, folds)
        set_metrics[label] = metrics

    simple_metrics: dict[str, dict[str, float | int]] = {}
    for label, columns in simple_checks.items():
        metrics, _ = evaluate_grouped_oof_regression(frame, columns, "gradient_boosting", GB_PARAMS, folds)
        simple_metrics[label] = metrics

    baseline_r2 = float(set_metrics["Raw"]["r2"])
    top = {label: float(metrics["r2"]) - baseline_r2 for label, metrics in set_metrics.items()}
    bottom = {label: float(metrics["r2"]) - baseline_r2 for label, metrics in simple_metrics.items()}
    return {
        "baseline_r2": baseline_r2,
        "top": top,
        "bottom": bottom,
        "top_absolute_r2": {label: float(metrics["r2"]) for label, metrics in set_metrics.items()},
        "bottom_absolute_r2": {label: float(metrics["r2"]) for label, metrics in simple_metrics.items()},
    }


def remake_plot(output_path: Path = FIG_PATH, metrics: dict[str, object] | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = metrics or evaluate_feature_contribution(load_frame())

    top_metrics = metrics["top"]
    bottom_metrics = metrics["bottom"]

    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        plt.style.use("classic")

    fig, axes = plt.subplots(2, 1, figsize=(10, 11), gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0]
    top_labels = ["Raw", "Raw + simple", "Raw + physics", "All"]
    top_values = [float(top_metrics[label]) for label in top_labels]
    top_colors = ["#d9d9d9", "#33a02c", "#ff7f00", "#e31a1c"]
    top_x = np.arange(len(top_labels))
    top_bars = ax.bar(top_x, top_values, color=top_colors, edgecolor="none")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(top_x)
    ax.set_xticklabels(top_labels, rotation=0, ha="center", fontsize=8)
    ax.tick_params(axis="x", which="major", pad=8)
    ax.set_ylabel(r"Change in grouped held-out $R^2$ vs raw baseline")
    ax.set_title("Contribution of Physics-Derived Features")
    ax.set_ylim(min(-0.025, min(top_values) * 1.2 if top_values else -0.025), max(0.025, max(top_values) * 1.1 if top_values else 0.025))

    for rect, value, color in zip(top_bars, top_values, top_colors):
        cx = rect.get_x() + rect.get_width() / 2.0
        y = value * 0.5 if abs(value) > 0.003 else value + (0.002 if value >= 0 else -0.002)
        va = "center" if abs(value) > 0.003 else ("bottom" if value >= 0 else "top")
        text_color = "white" if color in {"#33a02c", "#e31a1c"} and abs(value) > 0.003 else "black"
        ax.text(cx, y, f"{value:+.3f}", ha="center", va=va, fontsize=10, fontweight="semibold", color=text_color)

    ax2 = axes[1]
    bottom_labels = ["Raw", r"v_inf^2", "1/r_p", "f_spin", "radius", "all simple"]
    bottom_values = [0.0] + [float(bottom_metrics[label]) for label in bottom_labels[1:]]
    bottom_x = np.arange(len(bottom_labels))
    bottom_bars = ax2.bar(bottom_x, bottom_values, color="#6c83b5", edgecolor="none")
    ax2.axhline(0, color="k", linewidth=0.8)
    ax2.set_xticks(bottom_x)
    ax2.set_xticklabels(bottom_labels, rotation=0, ha="center", fontsize=8)
    ax2.tick_params(axis="x", which="major", pad=8)
    ax2.set_ylabel(r"Change in grouped held-out $R^2$ vs raw baseline")
    ax2.set_title("Simple Transform Checks")
    spread = max(abs(min(bottom_values)), abs(max(bottom_values)), 0.01)
    ax2.set_ylim(min(-0.025, -spread * 1.2), max(0.01, spread * 1.2))

    for rect, value in zip(bottom_bars, bottom_values):
        cx = rect.get_x() + rect.get_width() / 2.0
        if abs(value) > 0.003:
            y = value * 0.5
            va = "center"
            color = "white"
        else:
            y = value + (0.002 if value >= 0 else -0.002)
            va = "bottom" if value >= 0 else "top"
            color = "black"
        ax2.text(cx, y, f"{value:+.3f}", ha="center", va=va, fontsize=10, fontweight="semibold", color=color)

    fig.suptitle("Gradient Boosting Feature-Engineering Gains for BMF", fontsize=18, y=0.97)
    fig.text(
        0.5,
        0.035,
        "Scores use grouped held-out evaluation on the canonical 407-row BMF table with the tuned gradient boosting configuration.",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    plt.subplots_adjust(hspace=0.45, top=0.91, bottom=0.08)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    remake_plot()
    print(FIG_PATH)


if __name__ == "__main__":
    main()
