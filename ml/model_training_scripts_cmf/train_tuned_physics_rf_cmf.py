#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.model_training_scripts_cmf.common import (
    ARTIFACT_ROOT,
    CORRECTED_DATASET_PATH,
    PHYSICS_FEATURE_COLUMNS,
    RAW_FEATURE_COLUMNS,
    train_and_save_regression_model,
)


MODEL_PATH = ARTIFACT_ROOT / "tuned_physics_rf" / "main_cmf_tuned_physics_rf.pkl"
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
        bundle_name="main_cmf_tuned_physics_random_forest_cmf",
        extra_bundle_fields={
            "raw_feature_columns": RAW_FEATURE_COLUMNS,
            "physics_feature_columns": PHYSICS_FEATURE_COLUMNS,
        },
    )
    print(f"Saved model bundle to {args.output}")
    print(f"Training rows: {len(result['frame'])}")
    print(f"Grouped-CV R2: {result['metrics']['r2']:.4f}")
    print(f"Grouped-CV MSE: {result['metrics']['mse']:.4f}")
    print(f"Grouped-CV RMSE: {result['metrics']['rmse']:.4f}")


if __name__ == "__main__":
    main()
