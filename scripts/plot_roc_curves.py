#!/usr/bin/env python3
"""Generate ROC curves for 4 binary classification targets.

For each target we use the best-performing classifier from
ml/bound_outcomes/tables/classification_metrics.csv and the
corresponding probability scores from prediction_records.csv.

Targets covered (all binarised from the regression/continuous columns):
  1. bound_mass_fraction  > 0           (has_any_bound_mass)
  2. bound_fragment_count > 0           (binarised inline)
  3. bound/unbound > some_threshold     (binarised inline)
  4. largest_bound_fragment_mass_kg > 0 (binarised inline)

Each ROC plot shows all four classifiers (dummy, LR, RF, GBT) on the same
axis for that target so you can compare the family of models.

Run:
    python scripts/plot_roc_curves.py \
      --predictions ml/bound_outcomes/tables/prediction_records.csv \
      --metrics     ml/bound_outcomes/tables/classification_metrics.csv \
      --out-dir     ml/bound_outcomes/plots/roc_four_targets
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

# ── config ───────────────────────────────────────────────────────────────────

MODEL_COLORS = {
    "dummy_most_frequent":          "#999999",
    "logistic_regression":          "#1f77b4",
    "random_forest_classifier":     "#2ca02c",
    "gradient_boosting_classifier": "#d62728",
}

MODEL_LABELS = {
    "dummy_most_frequent":          "Dummy (most-frequent)",
    "logistic_regression":          "Logistic Regression",
    "random_forest_classifier":     "Random Forest",
    "gradient_boosting_classifier": "Gradient Boosting",
}

# For targets that exist only as regression in predictions, we binarise them:
# Each entry: (nice name, source column in predictions, threshold, threshold description)
DERIVED_BINARY_TARGETS = {
    "bound_fragment_count_gt0": (
        "Bound Fragment Count > 0",
        "bound_fragment_count",   # regression target column
        0.0,
        "count > 0",
    ),
    "largest_bound_mass_gt0": (
        "Largest Bound Fragment Mass > 0",
        "largest_bound_fragment_mass_kg",
        0.0,
        "mass > 0",
    ),
}

# ── helpers ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--predictions",
        default="ml/bound_outcomes/tables/prediction_records.csv",
    )
    p.add_argument(
        "--metrics",
        default="ml/bound_outcomes/tables/classification_metrics.csv",
    )
    p.add_argument(
        "--out-dir",
        default="ml/bound_outcomes/plots/roc_four_targets",
    )
    return p.parse_args()


def best_feature_set(metrics: pd.DataFrame, target: str) -> str:
    sub = metrics[metrics["target"] == target]
    if sub.empty:
        return "with_fof_linking_length"
    best = sub.sort_values(
        ["roc_auc", "balanced_accuracy"], ascending=False
    ).iloc[0]
    return str(best["feature_set"])


def roc_for_target(
    records: pd.DataFrame,
    target_col: str,
    target_name: str,
    feature_set: str,
    dataset: str,
    ax: plt.Axes,
    title: str,
) -> None:
    """Draw ROC curves for all models on a single axis."""
    sub = records[
        (records["task"] == "classification")
        & (records["target"] == target_col)
        & (records["feature_set"] == feature_set)
        & (records["dataset"] == dataset)
    ].copy()

    if sub.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=11)
        return

    ax.plot([0, 1], [0, 1], "--", color="#bbbbbb", linewidth=1, label="Random (AUC=0.50)")

    for model_name in ["dummy_most_frequent", "logistic_regression",
                       "random_forest_classifier", "gradient_boosting_classifier"]:
        msub = sub[sub["model"] == model_name].copy()
        if msub.empty:
            continue
        y_true = msub["actual"].astype(bool)
        y_score = pd.to_numeric(msub["score"], errors="coerce")
        valid = ~y_score.isna()
        y_true  = y_true[valid]
        y_score = y_score[valid]
        if y_true.nunique() < 2 or y_score.isna().all():
            continue
        try:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
        except Exception:
            continue
        color = MODEL_COLORS.get(model_name, "black")
        label = f"{MODEL_LABELS.get(model_name, model_name)} (AUC={auc:.3f})"
        ax.plot(fpr, tpr, color=color, linewidth=2.0, label=label)

    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.85)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)


def binarise_regression_target(records: pd.DataFrame, reg_col: str, threshold: float) -> pd.DataFrame:
    """From regression prediction rows, create pseudo-classification rows."""
    sub = records[
        (records["task"] == "regression")
        & (records["target"] == reg_col)
    ].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["actual"]    = (pd.to_numeric(sub["actual"], errors="coerce") > threshold).astype(bool)
    # Use predicted value as "score" (higher predicted = more likely positive)
    sub["score"]     = pd.to_numeric(sub["predicted"], errors="coerce")
    sub["task"]      = "classification"
    sub["target"]    = f"{reg_col}_gt{threshold}"
    return sub


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = pd.read_csv(args.predictions, low_memory=False)
    metrics = pd.read_csv(args.metrics, low_memory=False)

    # ── 1. has_any_bound_mass ─────────────────────────────────────────────────
    # ── 2. bound_mass_fraction_ge_0_1 ─────────────────────────────────────────
    # These exist as native classification targets
    native_cls_targets = [
        ("has_any_bound_mass",         "Has Any Bound Mass (BMF > 0)"),
        ("bound_mass_fraction_ge_0_1", "Bound Mass Fraction ≥ 10%"),
    ]

    # ── 3 & 4. derived from regression predictions ─────────────────────────
    derived_rows_bf  = binarise_regression_target(records, "bound_fragment_count", 0.0)
    derived_rows_lbm = binarise_regression_target(records, "largest_bound_fragment_mass_kg", 0.0)

    if not derived_rows_bf.empty:
        records = pd.concat([records, derived_rows_bf], ignore_index=True)
    if not derived_rows_lbm.empty:
        records = pd.concat([records, derived_rows_lbm], ignore_index=True)

    derived_cls_targets = [
        ("bound_fragment_count_gt0.0",           "Bound Fragment Count > 0"),
        ("largest_bound_fragment_mass_kg_gt0.0", "Largest Bound Fragment Mass > 0"),
    ]

    all_targets = [
        (t, name, "has_any_bound_mass" if "has_any" in t else t)
        for t, name in native_cls_targets
    ] + [
        (t, name, t)
        for t, name in derived_cls_targets
    ]

    # ─── Individual ROC plots (one per target) ─────────────────────────────
    for target_col, target_name, fs_lookup in all_targets:
        feature_set = best_feature_set(metrics, fs_lookup) if not ("_gt" in target_col) else "with_fof_linking_length"
        dataset     = "all_successful_runs"

        fig, ax = plt.subplots(figsize=(7, 5))
        roc_for_target(
            records, target_col, target_name, feature_set, dataset, ax,
            title=f"ROC — {target_name}",
        )
        fname = f"roc_{target_col.replace('/', '_').replace('.','p')}.png"
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=200)
        plt.close(fig)
        print(f"Saved {fname}")

    # ─── Combined 4-panel ROC figure ──────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes_flat = axes.flatten()
    titles = [
        "Target 1: Has Any Bound Mass",
        "Target 2: Bound Mass Fraction ≥ 10%",
        "Target 3: Bound Fragment Count > 0",
        "Target 4: Largest Bound Fragment Mass > 0",
    ]
    for ax_idx, ((target_col, target_name, fs_lookup), title) in enumerate(
        zip(all_targets, titles)
    ):
        feature_set = best_feature_set(metrics, fs_lookup) if not ("_gt" in target_col) else "with_fof_linking_length"
        roc_for_target(
            records, target_col, target_name, feature_set,
            "all_successful_runs", axes_flat[ax_idx], title=title,
        )

    fig.suptitle("ROC Curves — Four Classification Targets", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_combined_four_targets.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved roc_combined_four_targets.png")

    print(f"\nAll ROC plots saved to {out_dir}")


if __name__ == "__main__":
    main()
