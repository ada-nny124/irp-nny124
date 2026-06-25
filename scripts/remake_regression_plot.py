#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = ROOT / "ml" / "model_diagnostics" / "tables" / "prediction_records.csv"
METRICS_PATH = ROOT / "ml" / "tables" / "model_metrics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remake a regression actual-vs-predicted plot with annotations.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--feature-set", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def kg_tick_formatter(value: float, _: int) -> str:
    if np.isclose(value, 0.0):
        return "0"
    return f"{value:.1e}"


def main() -> None:
    args = parse_args()

    predictions = pd.read_csv(PREDICTIONS_PATH)
    metrics = pd.read_csv(METRICS_PATH)

    record_mask = (
        (predictions["dataset"] == args.dataset)
        & (predictions["feature_set"] == args.feature_set)
        & (predictions["target"] == args.target)
        & (predictions["model"] == args.model)
        & (predictions["split"] == "test")
    )
    records = predictions.loc[record_mask].copy()
    if records.empty:
        raise SystemExit("No matching test prediction records found.")

    metric_mask = (
        (metrics["dataset"] == args.dataset)
        & (metrics["feature_set"] == args.feature_set)
        & (metrics["target"] == args.target)
        & (metrics["model"] == args.model)
    )
    metric_row = metrics.loc[metric_mask]
    if metric_row.empty:
        raise SystemExit("No matching model metrics found.")
    metric_row = metric_row.iloc[0]

    actual = records["actual"].to_numpy(dtype=float)
    predicted = records["predicted"].to_numpy(dtype=float)

    correlation = float(np.corrcoef(actual, predicted)[0, 1])
    slope, intercept = np.polyfit(actual, predicted, 1)
    line_x = np.linspace(actual.min(), actual.max(), 200)
    line_y = slope * line_x + intercept

    lower = float(min(actual.min(), predicted.min()))
    upper = float(max(actual.max(), predicted.max()))
    padding = 0.04 * (upper - lower) if upper > lower else 1.0
    min_lim = lower - padding
    max_lim = upper + padding

    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    ax.scatter(actual, predicted, s=70, alpha=0.8, color="#2c7fb8", edgecolors="white", linewidths=0.4)
    ax.plot([min_lim, max_lim], [min_lim, max_lim], linestyle="--", color="#d62728", linewidth=2.0, label="Ideal fit")
    ax.plot(line_x, line_y, color="#1b4332", linewidth=2.0, label="Best-fit line")

    ax.set_xlim(min_lim, max_lim)
    ax.set_ylim(min_lim, max_lim)
    ax.set_title(args.title, fontsize=18, pad=14)
    ax.set_xlabel("Actual mass (kg)", fontsize=13)
    ax.set_ylabel("Predicted mass (kg)", fontsize=13)
    ax.grid(alpha=0.18)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FuncFormatter(kg_tick_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(kg_tick_formatter))

    stats_text = "\n".join(
        [
            f"Test n = {len(records)}",
            f"R = {correlation:.3f}",
            f"R^2 = {metric_row['test_r2']:.3f}",
            f"MAE = {metric_row['test_mae']:.3e} kg",
            f"RMSE = {metric_row['test_rmse']:.3e} kg",
            f"Fit: y = {slope:.3f}x + {intercept:.3e}",
        ]
    )
    ax.text(
        0.03,
        0.97,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#bdbdbd", "alpha": 0.95},
    )

    reference_text = "Axis ticks show kg directly\n1.0e20 kg = 100,000,000,000,000,000,000 kg"
    ax.text(
        0.97,
        0.03,
        reference_text,
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=10,
        color="#444444",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d9d9d9", "alpha": 0.9},
    )

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
