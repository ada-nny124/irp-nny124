#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.model_training_scripts.common import build_regression_pipeline, build_training_frame


MODEL_SPECS = [
    (
        "Raw GB",
        "raw_gb",
        "gradient_boosting",
        [
            "mass_log10_kg",
            "periapsis_Rm",
            "v_inf_kms",
            "spin_period_hr",
            "spin_axis",
            "resolution_value",
            "fof_linking_length",
        ],
        {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 1.0,
            "min_samples_leaf": 2,
            "random_state": 42,
        },
        "raw_gb_metrics.json",
    ),
    (
        "Tuned GB",
        "tuned_gb",
        "gradient_boosting",
        [
            "mass_log10_kg",
            "periapsis_Rm",
            "v_inf_kms",
            "spin_period_hr",
            "spin_axis",
            "resolution_value",
            "fof_linking_length",
        ],
        {
            "n_estimators": 500,
            "learning_rate": 0.08,
            "max_depth": 3,
            "subsample": 0.8,
            "min_samples_leaf": 1,
            "random_state": 42,
        },
        "tuned_gb_metrics.json",
    ),
    (
        "Raw RF",
        "raw_rf",
        "random_forest",
        [
            "mass_log10_kg",
            "periapsis_Rm",
            "v_inf_kms",
            "spin_period_hr",
            "spin_axis",
            "resolution_value",
            "fof_linking_length",
        ],
        {
            "n_estimators": 500,
            "max_features": 0.8,
            "min_samples_leaf": 2,
            "max_depth": None,
            "random_state": 42,
            "n_jobs": -1,
        },
        "raw_rf_metrics.json",
    ),
    (
        "Tuned RF",
        "tuned_rf",
        "random_forest",
        [
            "mass_log10_kg",
            "periapsis_Rm",
            "v_inf_kms",
            "spin_period_hr",
            "spin_axis",
            "resolution_value",
            "fof_linking_length",
        ],
        {
            "n_estimators": 500,
            "max_features": 0.8,
            "min_samples_leaf": 1,
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": -1,
        },
        "tuned_rf_metrics.json",
    ),
    (
        "Derived RF",
        "derived_rf",
        "random_forest",
        [
            "mass_log10_kg",
            "periapsis_Rm",
            "v_inf_kms",
            "spin_period_hr",
            "spin_axis",
            "resolution_value",
            "fof_linking_length",
            "v_inf_squared",
            "periapsis_inverse",
            "angular_momentum_proxy",
            "spin_frequency_hr_inv",
            "asteroid_radius_km",
            "encounter_eccentricity_proxy",
            "time_within_2_mars_radii_hr",
            "time_within_tidal_disruption_hr",
        ],
        {
            "n_estimators": 500,
            "max_features": 0.8,
            "min_samples_leaf": 1,
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": -1,
        },
        "derived_rf_metrics.json",
    ),
    (
        "Derived GB",
        "derived_gb",
        "gradient_boosting",
        [
            "mass_log10_kg",
            "periapsis_Rm",
            "v_inf_kms",
            "spin_period_hr",
            "spin_axis",
            "resolution_value",
            "fof_linking_length",
            "v_inf_squared",
            "periapsis_inverse",
            "angular_momentum_proxy",
            "spin_frequency_hr_inv",
            "asteroid_radius_km",
            "encounter_eccentricity_proxy",
            "time_within_2_mars_radii_hr",
            "time_within_tidal_disruption_hr",
        ],
        {
            "n_estimators": 500,
            "learning_rate": 0.08,
            "max_depth": 3,
            "subsample": 0.8,
            "min_samples_leaf": 1,
            "random_state": 42,
        },
        "derived_gb_metrics.json",
    ),
]

DEFAULT_DATASET = REPO_ROOT / "extraction-outputs" / "tables" / "bound_outcomes.csv"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "ml" / "trainingartifacts"
DEFAULT_OUTPUT = REPO_ROOT / "eda" / "tableA2_variation" / "tableA2_training_vs_oof.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Table A2 variation with training and OOF scores.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def build_rows(dataset_path: Path, artifact_root: Path) -> list[dict[str, object]]:
    frame = build_training_frame(dataset_path)
    rows: list[dict[str, object]] = []

    for label, artifact_dir_name, model_name, feature_columns, params, metrics_name in MODEL_SPECS:
        artifact_dir = artifact_root / artifact_dir_name
        oof_metrics = json.loads((artifact_dir / metrics_name).read_text(encoding="utf-8"))

        target_column = "bound_mass_fraction"
        target = pd.to_numeric(frame[target_column], errors="coerce")
        model = build_regression_pipeline(frame[feature_columns], model_name, params)
        model.fit(frame[feature_columns], target)
        predictions = np.clip(model.predict(frame[feature_columns]), 0.0, 1.0)

        train_r2 = float(r2_score(target, predictions))
        train_mae = float(mean_absolute_error(target, predictions))
        train_rmse = float(np.sqrt(mean_squared_error(target, predictions)))
        oof_r2 = float(oof_metrics["r2"])

        rows.append(
            {
                "Model": label,
                "Train R2": round(train_r2, 4),
                "OOF R2": round(oof_r2, 4),
                "R2 Gap": round(train_r2 - oof_r2, 4),
                "Train MAE": round(train_mae, 4),
                "OOF MAE": round(float(oof_metrics["mae"]), 4),
                "Train RMSE": round(train_rmse, 4),
                "OOF RMSE": round(float(oof_metrics["rmse"]), 4),
            }
        )

    return rows


def main() -> None:
    args = parse_args()
    table = pd.DataFrame(build_rows(args.dataset, args.artifact_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
