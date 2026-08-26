#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.model_training_scripts_corrected_bmf.common import (
    ARTIFACT_ROOT,
    COMBINED_PHYSICS_FEATURE_COLUMNS,
    CORRECTED_DATASET_PATH,
    PHYSICS_FEATURE_COLUMNS,
    RAW_FEATURE_COLUMNS,
    build_training_frame,
    evaluate_rf_feature_contribution,
    sidecars,
    train_and_save_regression_model,
)


MODEL_PATH = ARTIFACT_ROOT / "physics_rf" / "main_bmf_physics_rf.pkl"
FEATURE_COLUMNS = RAW_FEATURE_COLUMNS + PHYSICS_FEATURE_COLUMNS
RF_PARAMS = {
    "n_estimators": 500,
    "max_features": 0.8,
    "min_samples_leaf": 1,
    "max_depth": 10,
    "random_state": 42,
    "n_jobs": -1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=CORRECTED_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_and_save_regression_model(
        dataset_path=args.dataset,
        output_path=args.output,
        feature_columns=FEATURE_COLUMNS,
        model_name="random_forest",
        params=RF_PARAMS,
        bundle_name="main_bmf_physics_random_forest_corrected_bmf",
        extra_bundle_fields={
            "raw_feature_columns": RAW_FEATURE_COLUMNS,
            "physics_feature_columns": PHYSICS_FEATURE_COLUMNS,
        },
    )
    frame = build_training_frame(args.dataset)
    import pandas as pd
    folds = pd.read_csv(sidecars(args.output)["folds"])
    feature_contribution = evaluate_rf_feature_contribution(frame, folds, RF_PARAMS)
    sidecars(args.output)["feature_contribution"].write_text(
        json.dumps(feature_contribution, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved model bundle to {args.output}")
    print(f"Training rows: {len(result['frame'])}")
    print(f"Grouped-CV R2: {result['metrics']['r2']:.4f}")
    print(f"Grouped-CV MAE: {result['metrics']['mae']:.4f}")
    print(f"Grouped-CV RMSE: {result['metrics']['rmse']:.4f}")
    print(f"Feature contribution report: {sidecars(args.output)['feature_contribution']}")
    print("Features:")
    for column in FEATURE_COLUMNS:
        print(f"- {column}")


if __name__ == "__main__":
    main()
