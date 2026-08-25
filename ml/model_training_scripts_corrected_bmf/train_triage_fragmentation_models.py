#!/usr/bin/env python3
"""Train the active demo fragmentation models used by triage."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from triage.features import (
    CATEGORICAL_FEATURE_COLUMNS,
    DOMAIN_CATEGORICAL_FEATURE_COLUMNS,
    DOMAIN_NUMERIC_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    add_derived_features,
    load_fof_data,
    prepare_features,
)


DATASET_PATH = REPO_ROOT / "extraction-outputs_corrected_bmf" / "tables" / "fof_outcomes.csv"
OUTPUT_DIR = REPO_ROOT / "ml" / "triage"
RANDOM_STATE = 42
N_SPLITS = 5
PHYSICAL_FILE_RE = re.compile(r"_fof_[0-9.]+_\d+\.hdf5$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def add_proxy_targets(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    fragment_count = pd.to_numeric(enriched["fragment_count_min_particles"], errors="coerce").fillna(0)
    enriched["is_fragmented_proxy"] = fragment_count > 1

    total_mass = pd.to_numeric(enriched.get("total_particle_mass_kg"), errors="coerce")
    largest_mass = pd.to_numeric(enriched.get("largest_fragment_mass_kg"), errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        dispersed_mass_fraction = 1.0 - (largest_mass / total_mass)
    dispersed_mass_fraction = dispersed_mass_fraction.clip(lower=0.0, upper=1.0)

    severity = pd.Series("no_fragmentation", index=enriched.index, dtype="object")
    fragmented_mask = enriched["is_fragmented_proxy"]
    if fragmented_mask.any() and dispersed_mass_fraction.notna().any():
        severity.loc[fragmented_mask & (dispersed_mass_fraction < 0.1)] = "weak_fragmentation"
        severity.loc[fragmented_mask & dispersed_mass_fraction.between(0.1, 0.4, inclusive="left")] = "moderate_fragmentation"
        severity.loc[fragmented_mask & (dispersed_mass_fraction >= 0.4)] = "strong_fragmentation"
    enriched["severity_class"] = severity
    return enriched


def build_group_labels(frame: pd.DataFrame) -> pd.Series:
    if "filename" not in frame.columns:
        return pd.Series([f"row_{idx}" for idx in range(len(frame))], index=frame.index)
    return frame["filename"].fillna("").astype(str).str.replace(PHYSICAL_FILE_RE, ".hdf5", regex=True)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC_FEATURE_COLUMNS),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURE_COLUMNS,
            ),
        ]
    )


def build_classifier_candidates() -> dict[str, Pipeline]:
    preprocessor = build_preprocessor()
    return {
        "random_forest": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
            ]
        ),
    }


def build_regressor_candidates() -> dict[str, Pipeline]:
    preprocessor = build_preprocessor()
    return {
        "random_forest": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
            ]
        ),
    }


def make_cv(groups: pd.Series):
    unique_groups = groups.nunique(dropna=True)
    if unique_groups >= 3:
        return GroupKFold(n_splits=min(N_SPLITS, unique_groups))
    return KFold(n_splits=min(3, len(groups)), shuffle=True, random_state=RANDOM_STATE)


def rmse(y_true: list[float], y_pred: list[float]) -> float:
    return math.sqrt(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2))


def evaluate_classifier(models: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> tuple[str, dict[str, float]]:
    cv = make_cv(groups)
    best_name = ""
    best_metrics: dict[str, float] = {}
    best_score = -np.inf

    for model_name, model in models.items():
        probabilities: list[float] = []
        labels: list[int] = []
        truths: list[int] = []
        for train_idx, test_idx in cv.split(X, y, groups if isinstance(cv, GroupKFold) else None):
            fitted = clone(model)
            fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
            fold_probs = fitted.predict_proba(X.iloc[test_idx])[:, 1]
            probabilities.extend(fold_probs.tolist())
            labels.extend((fold_probs >= 0.5).astype(int).tolist())
            truths.extend(y.iloc[test_idx].astype(int).tolist())

        metrics = {
            "roc_auc": float(roc_auc_score(truths, probabilities)),
            "accuracy": float(accuracy_score(truths, labels)),
            "f1": float(f1_score(truths, labels, zero_division=0)),
        }
        if metrics["roc_auc"] > best_score:
            best_name = model_name
            best_metrics = metrics
            best_score = metrics["roc_auc"]

    return best_name, best_metrics


def evaluate_regressor(models: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> tuple[str, dict[str, float]]:
    cv = make_cv(groups)
    best_name = ""
    best_metrics: dict[str, float] = {}
    best_score = -np.inf

    for model_name, model in models.items():
        predictions: list[float] = []
        truths: list[float] = []
        for train_idx, test_idx in cv.split(X, y, groups if isinstance(cv, GroupKFold) else None):
            fitted = clone(model)
            fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
            fold_preds = fitted.predict(X.iloc[test_idx])
            predictions.extend(np.asarray(fold_preds, dtype=float).tolist())
            truths.extend(y.iloc[test_idx].astype(float).tolist())

        metrics = {
            "mae": float(mean_absolute_error(truths, predictions)),
            "r2": float(r2_score(truths, predictions)),
        }
        if metrics["r2"] > best_score:
            best_name = model_name
            best_metrics = metrics
            best_score = metrics["r2"]

    return best_name, best_metrics


def collect_classifier_eval(model: Pipeline, X: pd.DataFrame, y: pd.Series, groups: pd.Series, base_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    cv = make_cv(groups)
    rows: list[dict[str, object]] = []
    truths: list[int] = []
    labels: list[int] = []
    probabilities: list[float] = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups if isinstance(cv, GroupKFold) else None), start=1):
        fitted = clone(model)
        fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_probs = fitted.predict_proba(X.iloc[test_idx])[:, 1]
        fold_labels = (fold_probs >= 0.5).astype(int)
        fold_truth = y.iloc[test_idx].astype(int).to_numpy()
        truths.extend(fold_truth.tolist())
        labels.extend(fold_labels.tolist())
        probabilities.extend(fold_probs.tolist())
        feature_frame = base_frame.iloc[test_idx]
        for pos, row_idx in enumerate(feature_frame.index):
            rows.append(
                {
                    "split": f"fold_{fold_idx}",
                    "case_id": feature_frame.loc[row_idx].get("filename", row_idx),
                    "y_true": int(fold_truth[pos]),
                    "y_pred": int(fold_labels[pos]),
                    "y_proba": float(fold_probs[pos]),
                    "mass_log10_kg": feature_frame.loc[row_idx].get("mass_log10_kg"),
                    "periapsis_Rm": feature_frame.loc[row_idx].get("periapsis_Rm"),
                    "v_inf_kms": feature_frame.loc[row_idx].get("v_inf_kms"),
                    "spin_period_hr": feature_frame.loc[row_idx].get("spin_period_hr"),
                    "spin_axis": feature_frame.loc[row_idx].get("spin_axis"),
                    "resolution_code": feature_frame.loc[row_idx].get("resolution_code"),
                    "resolution_value": feature_frame.loc[row_idx].get("resolution_value"),
                    "timestep": feature_frame.loc[row_idx].get("timestep"),
                    "fof_linking_length": feature_frame.loc[row_idx].get("fof_linking_length"),
                }
            )
    metrics = {
        "accuracy": float(accuracy_score(truths, labels)),
        "balanced_accuracy": float(balanced_accuracy_score(truths, labels)),
        "precision": float(precision_score(truths, labels, zero_division=0)),
        "recall": float(recall_score(truths, labels, zero_division=0)),
        "f1": float(f1_score(truths, labels, zero_division=0)),
        "roc_auc": float(roc_auc_score(truths, probabilities)),
        "pr_auc": float(average_precision_score(truths, probabilities)),
    }
    return pd.DataFrame(rows), metrics


def collect_regressor_eval(model: Pipeline, X: pd.DataFrame, y: pd.Series, groups: pd.Series, base_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    cv = make_cv(groups)
    rows: list[dict[str, object]] = []
    truths: list[float] = []
    predictions: list[float] = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups if isinstance(cv, GroupKFold) else None), start=1):
        fitted = clone(model)
        fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_preds = fitted.predict(X.iloc[test_idx])
        fold_truth = y.iloc[test_idx].astype(float).to_numpy()
        truths.extend(fold_truth.tolist())
        predictions.extend(np.asarray(fold_preds, dtype=float).tolist())
        feature_frame = base_frame.iloc[test_idx]
        for pos, row_idx in enumerate(feature_frame.index):
            rows.append(
                {
                    "split": f"fold_{fold_idx}",
                    "case_id": feature_frame.loc[row_idx].get("filename", row_idx),
                    "y_true": float(fold_truth[pos]),
                    "y_pred": float(fold_preds[pos]),
                    "mass_log10_kg": feature_frame.loc[row_idx].get("mass_log10_kg"),
                    "periapsis_Rm": feature_frame.loc[row_idx].get("periapsis_Rm"),
                    "v_inf_kms": feature_frame.loc[row_idx].get("v_inf_kms"),
                    "spin_period_hr": feature_frame.loc[row_idx].get("spin_period_hr"),
                    "spin_axis": feature_frame.loc[row_idx].get("spin_axis"),
                    "resolution_code": feature_frame.loc[row_idx].get("resolution_code"),
                    "resolution_value": feature_frame.loc[row_idx].get("resolution_value"),
                    "timestep": feature_frame.loc[row_idx].get("timestep"),
                    "fof_linking_length": feature_frame.loc[row_idx].get("fof_linking_length"),
                }
            )
    metrics = {
        "mae": float(mean_absolute_error(truths, predictions)),
        "rmse": float(rmse(truths, predictions)),
        "r2": float(r2_score(truths, predictions)),
        "median_absolute_error": float(median_absolute_error(truths, predictions)),
    }
    return pd.DataFrame(rows), metrics


def save_target_artifacts(output_dir: Path, target_name: str, model: Pipeline, metrics: dict[str, float], eval_df: pd.DataFrame) -> None:
    with (output_dir / f"{target_name}_model.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    (output_dir / f"{target_name}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    eval_df.to_csv(output_dir / f"{target_name}_eval_predictions.csv", index=False)


def build_training_domain(features: pd.DataFrame) -> dict[str, object]:
    numeric = {}
    for column in DOMAIN_NUMERIC_FEATURE_COLUMNS:
        values = pd.to_numeric(features[column], errors="coerce").dropna()
        if values.empty:
            continue
        unique_values = np.sort(values.unique())
        step_hint = 0.0
        if len(unique_values) > 1:
            step_hint = float(np.min(np.diff(unique_values)))
        numeric[column] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "step_hint": step_hint,
        }

    categorical = {}
    for column in DOMAIN_CATEGORICAL_FEATURE_COLUMNS:
        values = features[column].dropna().astype(str)
        allowed = sorted(values.unique().tolist())
        counts = {str(key): int(value) for key, value in values.value_counts().sort_index().to_dict().items()}
        categorical[column] = {"allowed": allowed, "counts": counts}

    return {"numeric": numeric, "categorical": categorical}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_fof_data(args.dataset)
    frame = add_proxy_targets(add_derived_features(frame))
    features = prepare_features(frame)
    groups = build_group_labels(frame)

    classifier_target = frame["is_fragmented_proxy"].astype(int)
    regressor_target = pd.to_numeric(frame["largest_fragment_mass_kg"], errors="coerce")
    valid_regression = regressor_target.notna()

    classifier_models = build_classifier_candidates()
    best_classifier_name, classifier_selection_metrics = evaluate_classifier(classifier_models, features, classifier_target, groups)
    classifier_model = classifier_models[best_classifier_name]
    classifier_model.fit(features, classifier_target)
    classifier_eval_df, classifier_eval_metrics = collect_classifier_eval(classifier_model, features, classifier_target, groups, frame)

    regressor_models = build_regressor_candidates()
    best_regressor_name, regressor_selection_metrics = evaluate_regressor(
        regressor_models,
        features.loc[valid_regression],
        regressor_target.loc[valid_regression],
        groups.loc[valid_regression],
    )
    regressor_model = regressor_models[best_regressor_name]
    regressor_model.fit(features.loc[valid_regression], regressor_target.loc[valid_regression])
    regressor_eval_df, regressor_eval_metrics = collect_regressor_eval(
        regressor_model,
        features.loc[valid_regression],
        regressor_target.loc[valid_regression],
        groups.loc[valid_regression],
        frame.loc[valid_regression],
    )

    regressor_fraction_eval_df = regressor_eval_df.copy()
    regressor_fraction_eval_df["parent_mass_kg"] = np.power(10.0, pd.to_numeric(regressor_fraction_eval_df["mass_log10_kg"], errors="coerce"))
    regressor_fraction_eval_df["y_true"] = regressor_fraction_eval_df["y_true"] / regressor_fraction_eval_df["parent_mass_kg"]
    regressor_fraction_eval_df["y_pred"] = regressor_fraction_eval_df["y_pred"] / regressor_fraction_eval_df["parent_mass_kg"]
    regressor_fraction_metrics = {
        "mae": float(mean_absolute_error(regressor_fraction_eval_df["y_true"], regressor_fraction_eval_df["y_pred"])),
        "rmse": float(rmse(regressor_fraction_eval_df["y_true"].tolist(), regressor_fraction_eval_df["y_pred"].tolist())),
        "r2": float(r2_score(regressor_fraction_eval_df["y_true"], regressor_fraction_eval_df["y_pred"])),
        "median_absolute_error": float(median_absolute_error(regressor_fraction_eval_df["y_true"], regressor_fraction_eval_df["y_pred"])),
    }

    with (args.output_dir / "fragmentation_classifier.pkl").open("wb") as handle:
        pickle.dump(classifier_model, handle)
    with (args.output_dir / "fragmentation_regressor.pkl").open("wb") as handle:
        pickle.dump(regressor_model, handle)
    save_target_artifacts(args.output_dir, "is_fragmented_proxy", classifier_model, classifier_eval_metrics, classifier_eval_df)
    save_target_artifacts(args.output_dir, "largest_fragment_mass_kg", regressor_model, regressor_eval_metrics, regressor_eval_df)
    save_target_artifacts(args.output_dir, "largest_fragment_mass_fraction", regressor_model, regressor_fraction_metrics, regressor_fraction_eval_df)

    metrics = {
        "data_summary": {
            "rows": int(len(frame)),
            "columns": int(frame.shape[1]),
            "group_count": int(groups.nunique(dropna=True)),
            "fragmented_share": float(classifier_target.mean()),
            "severity_counts": {str(key): int(value) for key, value in frame["severity_class"].value_counts(dropna=False).to_dict().items()},
        },
        "classifier": {
            "target": "is_fragmented_proxy",
            "selected_model": best_classifier_name,
            "selection_metrics": classifier_selection_metrics,
            "evaluation_metrics": classifier_eval_metrics,
        },
        "regressor": {
            "target": "largest_fragment_mass_kg",
            "selected_model": best_regressor_name,
            "selection_metrics": regressor_selection_metrics,
            "evaluation_metrics": regressor_eval_metrics,
        },
    }

    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "largest_fragment_mass_kg_metrics.json").write_text(
        json.dumps(regressor_eval_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "training_domain.json").write_text(
        json.dumps(build_training_domain(features), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Saved fragmentation triage artifacts to {args.output_dir}")
    print(f"Classifier: {best_classifier_name}")
    print(f"Regressor: {best_regressor_name}")


if __name__ == "__main__":
    main()
