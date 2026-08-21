#!/usr/bin/env python3
"""Train and package the deployed leakage-safe two-stage CatBoost hurdle BMF model."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from triage.bmf import LEAKY_FEATURES, build_training_domain_metadata, write_json
from train_model_optimization_candidates import RANDOM_STATE
from scripts.eda.train_physics_structured_surrogate import (
    add_physics_features,
    build_group_folds,
    build_preprocessor,
    feature_columns_for_set,
    load_canonical_dataset,
)


PRIMARY_TARGET = "bound_mass_fraction"
FEATURE_SET_NAME = "with_fof_linking_length"
MODEL_DIR = ROOT / "ml" / "triage"
BUNDLE_PATH = MODEL_DIR / "bmf_hurdle_bundle.pkl"
OOF_PATH = MODEL_DIR / "bmf_hurdle_oof_predictions.csv"
METRICS_PATH = MODEL_DIR / "bmf_hurdle_metrics.json"
LOCAL_DIAGNOSTICS_PATH = MODEL_DIR / "bmf_hurdle_local_diagnostics.csv"
SLICE_PATH = MODEL_DIR / "bmf_hurdle_controlled_slices.csv"
MASS_19P5_PATH = MODEL_DIR / "bmf_hurdle_mass_19p5_check.csv"
FOLDS_PATH = MODEL_DIR / "bmf_hurdle_fold_assignments.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "extraction_outputs" / "bound_outcomes.csv")
    return parser.parse_args()


def safe_feature_columns() -> list[str]:
    columns = feature_columns_for_set(FEATURE_SET_NAME, include_physics=True)
    return [column for column in columns if column not in LEAKY_FEATURES]


def categorical_columns(columns: list[str]) -> list[str]:
    return [column for column in columns if column in {"spin_axis", "special_case_code"}]


def make_catboost_classifier() -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=400,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=RANDOM_STATE,
        verbose=False,
    )


def make_catboost_regressor() -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=600,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=RANDOM_STATE,
        verbose=False,
    )


def make_rf_benchmark(X: pd.DataFrame) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(X, scaled=False)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def clip_bmf(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def evaluate_oof(
    frame: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    valid = frame[frame[PRIMARY_TARGET].notna()].copy()
    valid = valid.merge(fold_assignments[["row_index", "fold_index"]], left_index=True, right_on="row_index", how="left")
    X = valid[feature_columns].copy()
    y = pd.to_numeric(valid[PRIMARY_TARGET], errors="coerce")
    cat_columns = categorical_columns(feature_columns)
    cat_indices = [X.columns.get_loc(column) for column in cat_columns]

    positive_prob_oof = np.full(len(valid), np.nan)
    positive_estimate_oof = np.full(len(valid), np.nan)
    final_pred_oof = np.full(len(valid), np.nan)
    benchmark_pred_oof = np.full(len(valid), np.nan)
    fold_rows: list[dict[str, object]] = []

    for fold_index in sorted(valid["fold_index"].dropna().unique()):
        train_mask = valid["fold_index"] != fold_index
        test_mask = valid["fold_index"] == fold_index
        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]
        y_train = y.loc[train_mask]
        y_test = y.loc[test_mask]

        classifier = make_catboost_classifier()
        classifier.fit(X_train, (y_train > 0).astype(int), cat_features=cat_indices)
        positive_prob = np.asarray(classifier.predict_proba(X_test)[:, 1], dtype=float)

        regressor = make_catboost_regressor()
        positive_train_mask = y_train > 0
        regressor.fit(X_train.loc[positive_train_mask], y_train.loc[positive_train_mask], cat_features=cat_indices)
        positive_estimate = clip_bmf(regressor.predict(X_test))

        final_pred = clip_bmf(positive_prob * positive_estimate)

        benchmark = make_rf_benchmark(X_train)
        benchmark.fit(X_train, y_train)
        benchmark_pred = clip_bmf(benchmark.predict(X_test))

        idx = np.flatnonzero(test_mask.to_numpy())
        positive_prob_oof[idx] = positive_prob
        positive_estimate_oof[idx] = positive_estimate
        final_pred_oof[idx] = final_pred
        benchmark_pred_oof[idx] = benchmark_pred

        fold_rows.append(
            {
                "fold_index": int(fold_index),
                "rows": int(test_mask.sum()),
                "r2": float(r2_score(y_test, final_pred)),
                "mae": float(mean_absolute_error(y_test, final_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, final_pred))),
                "benchmark_r2": float(r2_score(y_test, benchmark_pred)),
                "benchmark_mae": float(mean_absolute_error(y_test, benchmark_pred)),
            }
        )

    oof = valid.copy()
    oof["actual_bmf"] = y.to_numpy()
    oof["predicted_positive_probability"] = positive_prob_oof
    oof["predicted_positive_bmf"] = positive_estimate_oof
    oof["predicted_bmf"] = final_pred_oof
    oof["benchmark_random_forest_bmf"] = benchmark_pred_oof
    oof["absolute_error"] = np.abs(oof["actual_bmf"] - oof["predicted_bmf"])
    oof["benchmark_disagreement"] = np.abs(oof["predicted_bmf"] - oof["benchmark_random_forest_bmf"])
    oof["predicted_bmf_ge_0p1"] = oof["predicted_bmf"] >= 0.1

    fold_frame = pd.DataFrame(fold_rows)
    metrics = {
        "model_name": "two-stage CatBoost hurdle",
        "bundle_id": "bmf_hurdle_catboost_v1",
        "feature_set": FEATURE_SET_NAME,
        "feature_columns": feature_columns,
        "categorical_columns": cat_columns,
        "grouping": "physical_file",
        "rows": int(len(oof)),
        "unique_physical_files": int(valid["physical_file"].nunique()),
        "grouped_cv_r2": float(r2_score(oof["actual_bmf"], oof["predicted_bmf"])),
        "grouped_cv_mae_fraction": float(mean_absolute_error(oof["actual_bmf"], oof["predicted_bmf"])),
        "grouped_cv_mae_percentage_points": float(mean_absolute_error(oof["actual_bmf"], oof["predicted_bmf"])) * 100.0,
        "grouped_cv_rmse": float(np.sqrt(mean_squared_error(oof["actual_bmf"], oof["predicted_bmf"]))),
        "zero_share": float((oof["actual_bmf"] == 0).mean()),
        "fold_metrics": fold_rows,
        "benchmark_model_name": "baseline_random_forest",
        "benchmark_grouped_cv_r2": float(r2_score(oof["actual_bmf"], oof["benchmark_random_forest_bmf"])),
        "benchmark_grouped_cv_mae_fraction": float(mean_absolute_error(oof["actual_bmf"], oof["benchmark_random_forest_bmf"])),
        "benchmark_grouped_cv_rmse": float(np.sqrt(mean_squared_error(oof["actual_bmf"], oof["benchmark_random_forest_bmf"]))),
        "benchmark_disagreement_mean_fraction": float(oof["benchmark_disagreement"].mean()),
        "benchmark_disagreement_mean_percentage_points": float(oof["benchmark_disagreement"].mean() * 100.0),
        "benchmark_disagreement_p75_fraction": float(oof["benchmark_disagreement"].quantile(0.75)),
        "benchmark_disagreement_p75_percentage_points": float(oof["benchmark_disagreement"].quantile(0.75) * 100.0),
    }
    return oof, metrics


def build_local_diagnostics(oof: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["mass_log10_kg", "periapsis_Rm", "v_inf_kms", "spin_axis", "has_explicit_spin"]
    local = (
        oof.groupby(group_cols, dropna=False)
        .agg(
            nearby_run_count=("physical_file", "nunique"),
            local_grouped_mae=("absolute_error", "mean"),
            local_grouped_rmse=("absolute_error", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            mean_predicted_bmf=("predicted_bmf", "mean"),
            mean_actual_bmf=("actual_bmf", "mean"),
            benchmark_disagreement_mean=("benchmark_disagreement", "mean"),
        )
        .reset_index()
    )
    sparse_threshold = float(local["nearby_run_count"].median()) if not local.empty else 0.0
    local["sparse_region_flag"] = local["nearby_run_count"] <= sparse_threshold
    local["local_grouped_mae_percentage_points"] = local["local_grouped_mae"] * 100.0
    local["benchmark_disagreement_percentage_points"] = local["benchmark_disagreement_mean"] * 100.0
    local["sparse_threshold"] = sparse_threshold
    return local


def build_mass_19p5_check(oof: pd.DataFrame) -> pd.DataFrame:
    subset = oof[np.isclose(oof["mass_log10_kg"], 19.5)].copy()
    subset["actual_bmf_percent"] = subset["actual_bmf"] * 100.0
    subset["predicted_bmf_percent"] = subset["predicted_bmf"] * 100.0
    subset["absolute_error_percentage_points"] = subset["absolute_error"] * 100.0
    return subset[
        [
            "physical_file",
            "periapsis_Rm",
            "v_inf_kms",
            "spin_axis",
            "fold_index",
            "actual_bmf_percent",
            "predicted_bmf_percent",
            "absolute_error_percentage_points",
            "benchmark_random_forest_bmf",
            "benchmark_disagreement",
        ]
    ].sort_values(["periapsis_Rm", "v_inf_kms", "spin_axis"])


def build_controlled_slices(frame: pd.DataFrame, bundle: dict[str, object], feature_columns: list[str]) -> pd.DataFrame:
    training = frame[frame[PRIMARY_TARGET].notna()].copy()
    representative = {
        "mass_log10_kg": 20.0,
        "periapsis_Rm": 2.0,
        "v_inf_kms": 0.0,
        "spin_period_hr": 3.0,
        "spin_axis": "z",
        "has_explicit_spin": True,
        "special_case_code": "none",
        "resolution_value": 65,
        "timestep": 90000,
        "fof_linking_length": 0.004,
    }
    value_sets = {
        "periapsis": sorted(training["periapsis_Rm"].dropna().unique()),
        "velocity": sorted(training["v_inf_kms"].dropna().unique()),
        "mass": sorted(training["mass_log10_kg"].dropna().unique()),
        "spin": sorted(training["spin_period_hr"].dropna().unique()),
    }
    rows: list[dict[str, object]] = []
    for dimension, values in value_sets.items():
        for value in values:
            row = representative.copy()
            if dimension == "periapsis":
                row["periapsis_Rm"] = float(value)
            elif dimension == "velocity":
                row["v_inf_kms"] = float(value)
            elif dimension == "mass":
                row["mass_log10_kg"] = float(value)
            elif dimension == "spin":
                row["spin_period_hr"] = float(value)
            frame_row = pd.DataFrame([row])
            frame_row["target_mass_kg"] = np.power(10.0, frame_row["mass_log10_kg"])
            frame_row["particle_log10"] = np.log10(pd.to_numeric(frame_row["resolution_value"], errors="coerce"))
            frame_row["has_spin"] = frame_row["has_explicit_spin"].astype(int)
            frame_row = add_physics_features(frame_row)
            from triage.bmf import predict_bmf_from_bundle  # local import to avoid circular import at script load
            pred = predict_bmf_from_bundle(bundle, frame_row[feature_columns])
            matched = training.loc[
                np.isclose(training["mass_log10_kg"], frame_row.iloc[0]["mass_log10_kg"])
                & np.isclose(training["periapsis_Rm"], frame_row.iloc[0]["periapsis_Rm"])
                & np.isclose(training["v_inf_kms"], frame_row.iloc[0]["v_inf_kms"])
                & np.isclose(training["spin_period_hr"], frame_row.iloc[0]["spin_period_hr"])
                & (training["spin_axis"].astype(str) == str(frame_row.iloc[0]["spin_axis"]))
                & (training["has_explicit_spin"].astype(bool) == bool(frame_row.iloc[0]["has_explicit_spin"]))
            ].copy()
            support_count = matched["physical_file"].nunique()
            observed_mean = float(matched[PRIMARY_TARGET].mean()) if not matched.empty else math.nan
            observed_min = float(matched[PRIMARY_TARGET].min()) if not matched.empty else math.nan
            observed_max = float(matched[PRIMARY_TARGET].max()) if not matched.empty else math.nan
            rows.append(
                {
                    "slice_dimension": dimension,
                    "slice_value": float(value),
                    "fixed_mass_log10_kg": float(frame_row.iloc[0]["mass_log10_kg"]),
                    "fixed_periapsis_Rm": float(frame_row.iloc[0]["periapsis_Rm"]),
                    "fixed_v_inf_kms": float(frame_row.iloc[0]["v_inf_kms"]),
                    "fixed_spin_period_hr": float(frame_row.iloc[0]["spin_period_hr"]),
                    "fixed_spin_axis": str(frame_row.iloc[0]["spin_axis"]),
                    "predicted_bmf": float(pred.final_prediction[0]),
                    "predicted_bmf_percent": float(pred.final_prediction[0] * 100.0),
                    "positive_probability": float(pred.positive_probability[0]),
                    "positive_only_estimate": float(pred.positive_estimate[0]),
                    "observed_mean_bmf": observed_mean,
                    "observed_mean_bmf_percent": observed_mean * 100.0 if math.isfinite(observed_mean) else math.nan,
                    "observed_min_bmf": observed_min,
                    "observed_max_bmf": observed_max,
                    "nearby_support_count": int(support_count),
                    "unsupported_flag": int(support_count == 0),
                }
            )
    return pd.DataFrame(rows)


def fit_final_bundle(frame: pd.DataFrame, feature_columns: list[str], metrics: dict[str, object]) -> dict[str, object]:
    training = frame[frame[PRIMARY_TARGET].notna()].copy()
    X = training[feature_columns].copy()
    y = pd.to_numeric(training[PRIMARY_TARGET], errors="coerce")
    cat_columns = categorical_columns(feature_columns)
    cat_indices = [X.columns.get_loc(column) for column in cat_columns]

    classifier = make_catboost_classifier()
    classifier.fit(X, (y > 0).astype(int), cat_features=cat_indices)

    regressor = make_catboost_regressor()
    positive_mask = y > 0
    regressor.fit(X.loc[positive_mask], y.loc[positive_mask], cat_features=cat_indices)

    benchmark = make_rf_benchmark(X)
    benchmark.fit(X, y)

    bundle = {
        "bundle_id": metrics["bundle_id"],
        "model_name": metrics["model_name"],
        "feature_set": FEATURE_SET_NAME,
        "feature_columns": feature_columns,
        "categorical_columns": cat_columns,
        "zero_vs_positive_classifier": classifier,
        "positive_only_regressor": regressor,
        "benchmark_random_forest": benchmark,
        "training_domain": build_training_domain_metadata(training, feature_columns, cat_columns),
        "metrics": metrics,
        "feature_construction": {
            "include_physics": True,
            "feature_set_name": FEATURE_SET_NAME,
            "forbidden_features": sorted(LEAKY_FEATURES),
            "notes": "Physics-derived features use setup-time quantities only. No post-simulation outcome features are allowed.",
        },
        "training_row_count": int(len(training)),
        "training_group_count": int(training["physical_file"].nunique()),
        "leakage_guard": {"forbidden_features": sorted(LEAKY_FEATURES)},
    }
    return bundle


def main() -> None:
    args = parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    frame = add_physics_features(load_canonical_dataset(args.dataset))
    feature_columns = safe_feature_columns()
    fold_assignments = build_group_folds(frame, frame["physical_file"].astype(str), FOLDS_PATH)
    oof, metrics = evaluate_oof(frame, fold_assignments, feature_columns)
    local_diagnostics = build_local_diagnostics(oof)
    bundle = fit_final_bundle(frame, feature_columns, metrics)
    slice_frame = build_controlled_slices(frame, bundle, feature_columns)
    mass_19p5 = build_mass_19p5_check(oof)

    oof.to_csv(OOF_PATH, index=False)
    local_diagnostics.to_csv(LOCAL_DIAGNOSTICS_PATH, index=False)
    slice_frame.to_csv(SLICE_PATH, index=False)
    mass_19p5.to_csv(MASS_19P5_PATH, index=False)
    with BUNDLE_PATH.open("wb") as handle:
        pickle.dump(bundle, handle)
    write_json(METRICS_PATH, metrics)
    print(json.dumps(
        {
            "bundle_path": str(BUNDLE_PATH),
            "grouped_cv_r2": metrics["grouped_cv_r2"],
            "grouped_cv_mae_fraction": metrics["grouped_cv_mae_fraction"],
            "grouped_cv_rmse": metrics["grouped_cv_rmse"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
