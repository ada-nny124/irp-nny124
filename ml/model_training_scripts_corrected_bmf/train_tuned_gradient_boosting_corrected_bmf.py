#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.model_training_scripts_corrected_bmf.common import (
    ARTIFACT_ROOT,
    CORRECTED_DATASET_PATH,
    RAW_FEATURE_COLUMNS,
    train_and_save_regression_model,
)


MODEL_PATH = ARTIFACT_ROOT / "tuned_gradient_boosting" / "main_bmf_tuned_gradient_boosting.pkl"
GB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.08,
    "max_depth": 3,
    "subsample": 0.8,
    "min_samples_leaf": 1,
    "random_state": 42,
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
        feature_columns=RAW_FEATURE_COLUMNS,
        model_name="gradient_boosting",
        params=GB_PARAMS,
        bundle_name="main_bmf_tuned_gradient_boosting_corrected_bmf",
    )
    print(f"Saved model bundle to {args.output}")
    print(f"Training rows: {len(result['frame'])}")
    print(f"Grouped-CV R2: {result['metrics']['r2']:.4f}")
    print(f"Grouped-CV MSE: {result['metrics']['mse']:.4f}")
    print(f"Grouped-CV RMSE: {result['metrics']['rmse']:.4f}")


if __name__ == "__main__":
    main()
