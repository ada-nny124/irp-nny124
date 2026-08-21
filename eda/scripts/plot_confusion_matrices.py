#!/usr/bin/env python3
"""Generate a 2x2 confusion-matrix summary for four bound-outcome targets.

This mirrors the combined ROC summary, but shows only the gradient boosting
models:
  1. GradientBoostingClassifier for has_any_bound_mass
  2. GradientBoostingClassifier for bound_mass_fraction_ge_0_1
  3. GradientBoostingRegressor thresholded at > 0 for bound_fragment_count
  4. GradientBoostingRegressor thresholded at > 0 for largest_bound_fragment_mass_kg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from sklearn.metrics import confusion_matrix


CLASSIFICATION_MODEL = "gradient_boosting_classifier"
REGRESSION_MODEL = "gradient_boosting_regressor"
CLASSIFICATION_TARGETS = [
    ("has_any_bound_mass", "Has Any Bound Mass", CLASSIFICATION_MODEL),
    ("bound_mass_fraction_ge_0_1", "Bound Mass Fraction ≥ 10%", CLASSIFICATION_MODEL),
]
DERIVED_REGRESSION_TARGETS = [
    ("bound_fragment_count", "Bound Fragment Count > 0", REGRESSION_MODEL, 0.0),
    ("largest_bound_fragment_mass_kg", "Largest Bound Fragment Mass > 0", REGRESSION_MODEL, 0.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        default="ml/bound_outcomes/tables/prediction_records.csv",
        help="Prediction records table written by train_bound_models.py",
    )
    parser.add_argument(
        "--metrics",
        default="ml/bound_outcomes/tables/classification_metrics.csv",
        help="Classification metrics table for choosing the feature set of native classifiers.",
    )
    parser.add_argument(
        "--out-dir",
        default="ml/bound_outcomes/plots/confusion_four_targets",
        help="Directory for the combined confusion-matrix figure.",
    )
    parser.add_argument(
        "--dataset",
        default="all_successful_runs",
        help="Dataset subset to plot.",
    )
    return parser.parse_args()


def best_feature_set(metrics: pd.DataFrame, target: str, model: str) -> str:
    subset = metrics[(metrics["target"] == target) & (metrics["model"] == model)].copy()
    if subset.empty:
        return "with_fof_linking_length"
    best = subset.sort_values(["roc_auc", "balanced_accuracy"], ascending=False).iloc[0]
    return str(best["feature_set"])


def parse_bool(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(bool)
    return series.map({"True": True, "False": False, True: True, False: False})


def build_native_matrix(
    records: pd.DataFrame,
    target: str,
    model: str,
    feature_set: str,
    dataset: str,
) -> np.ndarray:
    subset = records[
        (records["task"] == "classification")
        & (records["target"] == target)
        & (records["model"] == model)
        & (records["feature_set"] == feature_set)
        & (records["dataset"] == dataset)
    ].copy()
    if subset.empty:
        return np.zeros((2, 2), dtype=int)
    y_true = parse_bool(subset["actual"])
    y_pred = parse_bool(subset["predicted"])
    valid = y_true.notna() & y_pred.notna()
    if not valid.any():
        return np.zeros((2, 2), dtype=int)
    return confusion_matrix(y_true[valid].astype(bool), y_pred[valid].astype(bool), labels=[False, True])


def build_derived_matrix(
    records: pd.DataFrame,
    target: str,
    model: str,
    threshold: float,
    dataset: str,
) -> np.ndarray:
    subset = records[
        (records["task"] == "regression")
        & (records["target"] == target)
        & (records["model"] == model)
        & (records["dataset"] == dataset)
    ].copy()
    if subset.empty:
        return np.zeros((2, 2), dtype=int)
    actual = pd.to_numeric(subset["actual"], errors="coerce")
    predicted = pd.to_numeric(subset["predicted"], errors="coerce")
    valid = actual.notna() & predicted.notna()
    if not valid.any():
        return np.zeros((2, 2), dtype=int)
    y_true = actual[valid] > threshold
    y_pred = predicted[valid] > threshold
    return confusion_matrix(y_true, y_pred, labels=[False, True])


def draw_confusion_matrix(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    norm: Normalize,
) -> None:
    image = ax.imshow(matrix, cmap="Blues", norm=norm)
    threshold = norm.vmax / 2.0 if norm.vmax else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            text_color = "white" if value > threshold else "black"
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color=text_color,
                fontweight="semibold",
                fontsize=12,
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["False", "True"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["False", "True"])
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_aspect("equal")
    return image


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = pd.read_csv(args.predictions, low_memory=False)
    metrics = pd.read_csv(args.metrics, low_memory=False)

    panel_specs: list[tuple[str, str, str, str, np.ndarray]] = []
    for target, title, model in CLASSIFICATION_TARGETS:
        feature_set = best_feature_set(metrics, target, model)
        matrix = build_native_matrix(records, target, model, feature_set, args.dataset)
        panel_specs.append((target, title, model, feature_set, matrix))

    for target, title, model, threshold in DERIVED_REGRESSION_TARGETS:
        feature_set = "with_fof_linking_length"
        matrix = build_derived_matrix(records, target, model, threshold, args.dataset)
        panel_specs.append((target, title, model, feature_set, matrix))

    vmax = max(int(matrix.max()) for _, _, _, _, matrix in panel_specs) if panel_specs else 1
    norm = Normalize(vmin=0, vmax=max(vmax, 1))

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10))
    axes_flat = axes.flatten()
    images = []
    for idx, (target, title, model, feature_set, matrix) in enumerate(panel_specs):
        image = draw_confusion_matrix(axes_flat[idx], matrix, f"Target {idx + 1}: {title}", norm)
        images.append(image)

    cbar = fig.colorbar(images[0], ax=axes_flat.tolist(), fraction=0.028, pad=0.12)
    cbar.set_label("Count", fontsize=11)
    fig.suptitle("Confusion Matrices — Four Targets (Gradient Boosting)", fontsize=16, y=0.98)
    fig.subplots_adjust(left=0.07, right=0.82, bottom=0.07, top=0.90, wspace=0.32, hspace=0.45)

    png_path = out_dir / "confusion_combined_four_targets_gradient_boosting.png"
    svg_path = out_dir / "confusion_combined_four_targets_gradient_boosting.svg"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {svg_path}")


if __name__ == "__main__":
    main()
