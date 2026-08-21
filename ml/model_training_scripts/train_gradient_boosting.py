#!/usr/bin/env python3
"""Train the main raw-feature Gradient Boosting BMF model."""

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

from ml.model_training_scripts.helper_functions_ml import (
    PRIMARY_TARGET,
    add_physics_features,
    build_or_load_group_folds,
    build_regression_pipeline,
    evaluate_grouped_oof_regression,
    load_canonical_dataset,
)


DATASET_PATH = Path("extraction-outputs/tables/bound_outcomes.csv")
ARTIFACT_DIR = Path("ml/trainingartifacts/gradient_boosting")
MODEL_PATH = ARTIFACT_DIR / "main_bmf_gradient_boosting.pkl"
FOLDS_PATH = ARTIFACT_DIR / "grouped_cv_fold_assignments.csv"
OOF_PATH = ARTIFACT_DIR / "main_bmf_gradient_boosting_oof_predictions.csv"
METRICS_PATH = ARTIFACT_DIR / "main_bmf_gradient_boosting_metrics.json"

FEATURE_COLUMNS = [
    "mass_log10_kg",
    "periapsis_Rm",
    "v_inf_kms",
    "spin_period_hr",
    "spin_axis",
    "resolution_value",
    "fof_linking_length",
]

GB_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "max_depth": 3,
    "subsample": 1.0,
    "min_samples_leaf": 2,
    "random_state": 42,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    return parser.parse_args()


def build_training_frame(dataset_path: Path) -> pd.DataFrame:
    raw = load_canonical_dataset(dataset_path)
    frame = add_physics_features(raw.copy())
    frame = frame.loc[frame[PRIMARY_TARGET].notna()].copy()
    return frame


def train_model(frame: pd.DataFrame):
    pipeline = build_regression_pipeline(frame[FEATURE_COLUMNS], "gradient_boosting", GB_PARAMS)
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
        "gradient_boosting",
        GB_PARAMS,
        folds,
    )

    bundle = {
        "model_name": "main_bmf_gradient_boosting",
        "target": PRIMARY_TARGET,
        "dataset_path": str(args.dataset),
        "feature_columns": FEATURE_COLUMNS,
        "gb_params": GB_PARAMS,
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
    print(f"Grouped-CV MSE: {grouped_cv_metrics['mse']:.4f}")
    print(f"Grouped-CV RMSE: {grouped_cv_metrics['rmse']:.4f}")


if __name__ == "__main__":
    main()
