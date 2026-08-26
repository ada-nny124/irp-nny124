#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
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


CORRECTED_DATASET_PATH = REPO_ROOT / "extraction-outputs_corrected_bmf" / "tables" / "bound_outcomes.csv"
ARTIFACT_ROOT = REPO_ROOT / "ml" / "trainingartifacts_corrected_bmf"

RAW_FEATURE_COLUMNS = [
    "mass_log10_kg",
    "periapsis_Rm",
    "v_inf_kms",
    "spin_period_hr",
    "spin_axis",
    "resolution_value",
    "fof_linking_length",
]

PHYSICS_FEATURE_COLUMNS = [
    "v_inf_squared",
    "periapsis_inverse",
    "angular_momentum_proxy",
    "spin_frequency_hr_inv",
    "asteroid_radius_km",
    "encounter_eccentricity_proxy",
    "time_within_2_mars_radii_hr",
    "time_within_tidal_disruption_hr",
]

SIMPLE_FEATURE_COLUMNS = [
    "v_inf_squared",
    "periapsis_inverse",
    "spin_frequency_hr_inv",
    "asteroid_radius_km",
]

COMBINED_PHYSICS_FEATURE_COLUMNS = [
    "angular_momentum_proxy",
    "encounter_eccentricity_proxy",
    "time_within_2_mars_radii_hr",
    "time_within_tidal_disruption_hr",
]


def sidecars(output_path: Path) -> dict[str, Path]:
    artifact_dir = output_path.parent
    stem = output_path.stem
    return {
        "folds": artifact_dir / "grouped_cv_fold_assignments.csv",
        "oof": artifact_dir / f"{stem}_oof_predictions.csv",
        "metrics": artifact_dir / f"{stem}_metrics.json",
        "feature_contribution": artifact_dir / f"{stem}_feature_contribution.json",
    }


def require_corrected_dataset(dataset_path: Path) -> None:
    if dataset_path.exists():
        return
    raise FileNotFoundError(
        "Corrected-BMF dataset not found. Expected full corrected table at "
        f"{dataset_path}. The current extraction-outputs_corrected_bmf folder does not yet contain it."
    )


def build_training_frame(dataset_path: Path) -> pd.DataFrame:
    require_corrected_dataset(dataset_path)
    raw = load_canonical_dataset(dataset_path)
    if "bound_mass_fraction" not in raw.columns:
        raise KeyError("Corrected bound_outcomes.csv is missing bound_mass_fraction.")
    raw["bound_mass_fraction"] = pd.to_numeric(raw["bound_mass_fraction"], errors="coerce")
    if "bound_mass_kg" in raw.columns:
        raw["bound_mass_kg"] = pd.to_numeric(raw["bound_mass_kg"], errors="coerce")
    if "unbound_mass_fraction" in raw.columns:
        raw["unbound_mass_fraction"] = pd.to_numeric(raw["unbound_mass_fraction"], errors="coerce")
    if "unbound_mass_kg" in raw.columns:
        raw["unbound_mass_kg"] = pd.to_numeric(raw["unbound_mass_kg"], errors="coerce")
    frame = add_physics_features(raw.copy())
    frame = frame.loc[frame[PRIMARY_TARGET].notna()].copy()
    return frame


def train_and_save_regression_model(
    *,
    dataset_path: Path,
    output_path: Path,
    feature_columns: list[str],
    model_name: str,
    params: dict[str, object],
    bundle_name: str,
    extra_bundle_fields: dict[str, object] | None = None,
) -> dict[str, object]:
    frame = build_training_frame(dataset_path)
    paths = sidecars(output_path)
    folds = build_or_load_group_folds(frame, paths["folds"])
    pipeline = build_regression_pipeline(frame[feature_columns], model_name, params)
    target = pd.to_numeric(frame[PRIMARY_TARGET], errors="coerce")
    model = pipeline.fit(frame[feature_columns], target)
    grouped_cv_metrics, oof_predictions = evaluate_grouped_oof_regression(
        frame,
        feature_columns,
        model_name,
        params,
        folds,
    )

    bundle = {
        "model_name": bundle_name,
        "target": PRIMARY_TARGET,
        "dataset_path": str(dataset_path),
        "feature_columns": feature_columns,
        "training_row_count": int(len(frame)),
        "grouped_cv_metrics": grouped_cv_metrics,
        "fold_assignments_path": str(paths["folds"]),
        "oof_predictions_path": str(paths["oof"]),
        "metrics_path": str(paths["metrics"]),
        "pipeline": model,
    }
    if model_name == "gradient_boosting":
        bundle["gb_params"] = params
    elif model_name == "random_forest":
        bundle["rf_params"] = params
    if extra_bundle_fields:
        bundle.update(extra_bundle_fields)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(bundle, handle)
    oof_predictions.to_csv(paths["oof"], index=False)
    paths["metrics"].write_text(json.dumps(grouped_cv_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "frame": frame,
        "bundle": bundle,
        "metrics": grouped_cv_metrics,
        "paths": paths,
    }


def evaluate_rf_feature_contribution(frame: pd.DataFrame, folds: pd.DataFrame, rf_params: dict[str, object]) -> dict[str, object]:
    feature_sets = {
        "Raw": RAW_FEATURE_COLUMNS,
        "Raw + simple": RAW_FEATURE_COLUMNS + SIMPLE_FEATURE_COLUMNS,
        "Raw + physics": RAW_FEATURE_COLUMNS + COMBINED_PHYSICS_FEATURE_COLUMNS,
        "All": RAW_FEATURE_COLUMNS + PHYSICS_FEATURE_COLUMNS,
    }
    simple_checks = {
        r"v_inf^2": RAW_FEATURE_COLUMNS + ["v_inf_squared"],
        "1/r_p": RAW_FEATURE_COLUMNS + ["periapsis_inverse"],
        "f_spin": RAW_FEATURE_COLUMNS + ["spin_frequency_hr_inv"],
        "radius": RAW_FEATURE_COLUMNS + ["asteroid_radius_km"],
        "all simple": RAW_FEATURE_COLUMNS + SIMPLE_FEATURE_COLUMNS,
    }

    set_metrics: dict[str, dict[str, float | int]] = {}
    for label, columns in feature_sets.items():
        metrics, _ = evaluate_grouped_oof_regression(frame, columns, "random_forest", rf_params, folds)
        set_metrics[label] = metrics

    simple_metrics: dict[str, dict[str, float | int]] = {}
    for label, columns in simple_checks.items():
        metrics, _ = evaluate_grouped_oof_regression(frame, columns, "random_forest", rf_params, folds)
        simple_metrics[label] = metrics

    baseline_r2 = float(set_metrics["Raw"]["r2"])
    return {
        "baseline_r2": baseline_r2,
        "top": {label: float(metrics["r2"]) - baseline_r2 for label, metrics in set_metrics.items()},
        "bottom": {label: float(metrics["r2"]) - baseline_r2 for label, metrics in simple_metrics.items()},
        "top_absolute_r2": {label: float(metrics["r2"]) for label, metrics in set_metrics.items()},
        "bottom_absolute_r2": {label: float(metrics["r2"]) for label, metrics in simple_metrics.items()},
        "feature_sets": {label: columns for label, columns in feature_sets.items()},
        "simple_feature_sets": {label: columns for label, columns in simple_checks.items()},
    }
