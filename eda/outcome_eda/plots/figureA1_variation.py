from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.train_helper import PRIMARY_TARGET, add_physics_features, load_canonical_dataset


SOURCE_PATH = ROOT / "extraction_outputs" / "bound_outcomes.csv"
FIG_PATH = ROOT / "report-table-figure" / "figures" / "figureA1_used_in_report.png"
PHYSICS_MODEL_PATH = ROOT / "ml" / "trainingartifacts" / "physics_rf" / "main_bmf_physics_rf.pkl"
RAW_MODEL_PATH = ROOT / "ml" / "trainingartifacts" / "raw_rf" / "main_bmf_raw_rf.pkl"


def load_bundle(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_frame() -> pd.DataFrame:
    frame = load_canonical_dataset(SOURCE_PATH)
    frame = add_physics_features(frame.copy())
    frame = frame.loc[frame[PRIMARY_TARGET].notna()].copy()
    return frame


def score_bundle(frame: pd.DataFrame, bundle: dict[str, object], label: str) -> dict[str, float | str]:
    feature_columns = list(bundle["feature_columns"])
    model = bundle["pipeline"]
    y_true = pd.to_numeric(frame[PRIMARY_TARGET], errors="coerce")
    y_pred = np.clip(model.predict(frame[feature_columns]), 0.0, 1.0)
    return {
        "label": label,
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def build_scores() -> pd.DataFrame:
    frame = load_frame()
    return pd.DataFrame(
        [
            score_bundle(frame, load_bundle(RAW_MODEL_PATH), "Raw features"),
            score_bundle(frame, load_bundle(PHYSICS_MODEL_PATH), "Raw + physics features"),
        ]
    )


def make_plot(scores: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.6))
    metric_specs = [
        ("r2", r"$R^2$", "#1f77b4", True),
        ("mae", "MAE", "#ff7f0e", False),
        ("rmse", "RMSE", "#2ca02c", False),
    ]
    x = np.arange(len(scores))

    for ax, (column, title, color, higher_is_better) in zip(axes, metric_specs):
        values = scores[column].to_numpy(dtype=float)
        bars = ax.bar(x, values, color=color, alpha=0.88, width=0.58)
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(scores["label"], rotation=0)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)

        value_span = max(values) - min(values)
        pad = max(value_span * 0.18, 0.002)
        if higher_is_better:
            ax.set_ylim(min(values) - pad * 0.4, max(values) + pad)
        else:
            ax.set_ylim(0.0, max(values) + pad)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + pad * 0.08,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    fig.suptitle("Raw vs physics-feature Random Forest BMF models", fontsize=18, y=0.98)
    fig.text(
        0.5,
        0.04,
        "Scores are computed on the canonical BMF table using the saved model bundles on the same 407 rows used for fitting.",
        ha="center",
        fontsize=10,
        color="#444444",
    )

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.02, 0.09, 0.98, 0.93))
    fig.savefig(FIG_PATH, dpi=220)
    plt.close(fig)


def main() -> None:
    scores = build_scores()
    make_plot(scores)
    print(FIG_PATH)


if __name__ == "__main__":
    main()
