#!/usr/bin/env python3
"""Train the main raw-feature Random Forest BMF model."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.train_helper import (
    PRIMARY_TARGET,
    add_physics_features,
    build_or_load_group_folds,
    evaluate_grouped_oof_regression,
    build_regression_pipeline,
    load_canonical_dataset,
)


DATASET_PATH = Path("extraction-outputs/tables/bound_outcomes.csv")
ARTIFACT_DIR = Path("ml/trainingartifacts/raw_rf")
MODEL_PATH = ARTIFACT_DIR / "main_bmf_raw_rf.pkl"
FOLDS_PATH = ARTIFACT_DIR / "grouped_cv_fold_assignments.csv"
OOF_PATH = ARTIFACT_DIR / "main_bmf_raw_rf_oof_predictions.csv"
METRICS_PATH = ARTIFACT_DIR / "main_bmf_raw_rf_metrics.json"

RAW_FEATURE_COLUMNS = [
    "mass_log10_kg",
    "periapsis_Rm",
    "v_inf_kms",
    "spin_period_hr",
    "spin_axis",
    "resolution_value",
    "fof_linking_length",
]

FEATURE_COLUMNS = RAW_FEATURE_COLUMNS

RF_PARAMS = {
    "n_estimators": 500,
    "max_features": 0.8,
    "min_samples_leaf": 2,
    "max_depth": None,
    "random_state": 42,
    "n_jobs": -1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Canonical bound-outcomes CSV used to train the model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MODEL_PATH,
        help="Destination .pkl path for the trained model bundle.",
    )
    return parser.parse_args()


def build_training_frame(dataset_path: Path) -> pd.DataFrame:
    raw = load_canonical_dataset(dataset_path)
    frame = add_physics_features(raw.copy())
    frame = frame.loc[frame[PRIMARY_TARGET].notna()].copy()
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")
    return frame


def train_model(frame: pd.DataFrame):
    pipeline = build_regression_pipeline(
        frame[FEATURE_COLUMNS],
        "random_forest",
        RF_PARAMS,
    )
    target = pd.to_numeric(frame[PRIMARY_TARGET], errors="coerce")
    return pipeline.fit(frame[FEATURE_COLUMNS], target)


def main() -> None:
    args = parse_args()
    frame = build_training_frame(args.dataset)
    folds = build_or_load_group_folds(frame, FOLDS_PATH)
    model = train_model(frame)
    grouped_cv_metrics, oof_predictions = evaluate_grouped_oof_regression(
        frame,
        FEATURE_COLUMNS,
        "random_forest",
        RF_PARAMS,
        folds,
    )

    bundle = {
        "model_name": "main_bmf_raw_random_forest",
        "target": PRIMARY_TARGET,
        "dataset_path": str(args.dataset),
        "feature_columns": FEATURE_COLUMNS,
        "raw_feature_columns": RAW_FEATURE_COLUMNS,
        "rf_params": RF_PARAMS,
        "training_row_count": int(len(frame)),
        "grouped_cv_metrics": grouped_cv_metrics,
        "fold_assignments_path": str(FOLDS_PATH),
        "oof_predictions_path": str(OOF_PATH),
        "metrics_path": str(METRICS_PATH),
        "pipeline": model,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(bundle, handle)
    oof_predictions.to_csv(OOF_PATH, index=False)
    METRICS_PATH.write_text(json.dumps(grouped_cv_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Saved model bundle to {args.output}")
    print(f"Training rows: {len(frame)}")
    print(f"Grouped-CV R2: {grouped_cv_metrics['r2']:.4f}")
    print(f"Grouped-CV MAE: {grouped_cv_metrics['mae']:.4f}")
    print(f"Grouped-CV RMSE: {grouped_cv_metrics['rmse']:.4f}")
    print("Features:")
    for column in FEATURE_COLUMNS:
        print(f"- {column}")


if __name__ == "__main__":
    main()
