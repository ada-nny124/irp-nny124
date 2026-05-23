#!/usr/bin/env python3
"""Train baseline regressors and diagnostics for FoF-derived fragment statistics."""

from __future__ import annotations

import argparse
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_IMPORTANCE_FEATURES = 20
RESIDUAL_GROUP_COLUMNS = [
    "periapsis_Rm",
    "mass_log10_kg",
    "v_inf_kms",
    "spin_axis",
    "fof_linking_length",
]
BASE_FEATURE_COLUMNS = [
    "mass_log10_kg",
    "particle_log10",
    "periapsis_Rm",
    "v_inf_kms",
    "spin_period_hr",
    "spin_axis",
    "has_explicit_spin",
    "timestep",
    "fof_linking_length",
]
FEATURE_SET_COLUMNS = {
    "with_fof_linking_length": BASE_FEATURE_COLUMNS,
    "without_fof_linking_length": [column for column in BASE_FEATURE_COLUMNS if column != "fof_linking_length"],
}
BASE_TARGET_COLUMNS = [
    "fragment_count_min_particles",
    "largest_fragment_particle_count",
    "largest_fragment_mass_kg",
]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    frame: pd.DataFrame
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("outputs/fof_outcomes.csv"),
        help="FoF outcome table with one row per simulation.",
    )
    parser.add_argument(
        "--ml-dir",
        type=Path,
        default=Path("ml"),
        help="Output directory for ML artifacts.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["mass_log10_kg"] = pd.to_numeric(frame["mass_value"], errors="coerce") / 100.0
    frame["particle_log10"] = pd.to_numeric(frame["resolution_value"], errors="coerce") / 10.0
    frame["periapsis_Rm"] = pd.to_numeric(frame["periapsis_value"], errors="coerce") / 10.0
    frame["v_inf_kms"] = pd.to_numeric(frame["velocity_value"], errors="coerce") / 10.0
    frame["spin_period_hr"] = pd.to_numeric(frame["spin_value"], errors="coerce") / 10.0
    frame["has_explicit_spin"] = (
        frame["has_explicit_spin"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
    )
    frame["spin_axis"] = frame["spin_axis"].fillna("none").replace("", "none")
    frame["special_case_code"] = frame.get("special_case_code", "").fillna("")
    return frame


def select_targets(df: pd.DataFrame) -> list[str]:
    targets: list[str] = []
    for column in BASE_TARGET_COLUMNS:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().sum() == 0:
            continue
        targets.append(column)
    return targets


def most_common_value(series: pd.Series):
    mode = series.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else None


def build_dataset_specs(df: pd.DataFrame) -> list[DatasetSpec]:
    full = DatasetSpec(
        name="full",
        frame=df.copy(),
        description="All simulations in outputs/fof_outcomes.csv.",
    )

    resolution_code = most_common_value(df["resolution_code"]) if "resolution_code" in df.columns else None
    fof_linking_length = (
        most_common_value(pd.to_numeric(df["fof_linking_length"], errors="coerce"))
        if "fof_linking_length" in df.columns
        else None
    )
    clean = df.copy()
    clean = clean[pd.to_numeric(clean["timestep"], errors="coerce") == 90000]
    if resolution_code is not None:
        clean = clean[clean["resolution_code"] == resolution_code]
    if fof_linking_length is not None:
        clean = clean[pd.to_numeric(clean["fof_linking_length"], errors="coerce") == fof_linking_length]
    clean = clean[clean["special_case_code"].fillna("") == ""]

    clean_desc = (
        "Subset with timestep == 90000, "
        f"resolution_code == {resolution_code}, "
        f"fof_linking_length == {fof_linking_length}, "
        "excluding special_case_code."
    )
    return [full, DatasetSpec(name="clean_subset", frame=clean.copy(), description=clean_desc)]


def build_preprocessor(X: pd.DataFrame, model_name: str) -> ColumnTransformer:
    numeric_features = [column for column in X.columns if column != "spin_axis"]
    categorical_features = [column for column in ["spin_axis"] if column in X.columns]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_name == "ridge":
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def build_models(X: pd.DataFrame) -> dict[str, Pipeline]:
    return {
        "dummy_mean": Pipeline(
            [("preprocessor", build_preprocessor(X, "dummy_mean")), ("model", DummyRegressor(strategy="mean"))]
        ),
        "ridge": Pipeline([("preprocessor", build_preprocessor(X, "ridge")), ("model", Ridge(alpha=1.0))]),
        "random_forest": Pipeline(
            [
                ("preprocessor", build_preprocessor(X, "random_forest")),
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
        ),
        "gradient_boosting": Pipeline(
            [
                ("preprocessor", build_preprocessor(X, "gradient_boosting")),
                ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
            ]
        ),
    }


def make_feature_frame(df: pd.DataFrame, feature_set_name: str) -> pd.DataFrame:
    columns = FEATURE_SET_COLUMNS[feature_set_name]
    return df[columns].copy()


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def safe_slug(text: str) -> str:
    return text.replace("/", "_").replace(" ", "_")


def model_plots_dir(plots_dir: Path, model_name: str, feature_set_name: str) -> Path:
    path = plots_dir / safe_slug(model_name) / safe_slug(feature_set_name)
    ensure_dir(path)
    return path


def save_plot(y_true: pd.Series, y_pred: np.ndarray, output_path: Path, title: str, x_label: str, y_label: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.7, color="#1f77b4", edgecolors="none")
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="#d62728", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_residual_plot(y_true: pd.Series, y_pred: np.ndarray, output_path: Path, title: str) -> None:
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_pred, residuals, alpha=0.7, color="#ff7f0e", edgecolors="none")
    ax.axhline(0.0, linestyle="--", color="#222222", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (actual - predicted)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_residual_feature_plot(frame: pd.DataFrame, feature: str, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    if feature == "spin_axis":
        grouped = []
        labels = []
        for label, subset in frame.groupby(feature, dropna=False):
            grouped.append(subset["residual"].to_numpy())
            labels.append(str(label))
        ax.boxplot(grouped, tick_labels=labels)
        ax.set_xlabel(feature)
    else:
        ax.scatter(frame[feature], frame["residual"], alpha=0.7, color="#2ca02c", edgecolors="none")
        ax.set_xlabel(feature)
    ax.axhline(0.0, linestyle="--", color="#222222", linewidth=1.5)
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def get_feature_names(pipeline: Pipeline) -> np.ndarray:
    return pipeline.named_steps["preprocessor"].get_feature_names_out()


def metric_summary(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def collect_importance_rows(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    dataset_name: str,
    feature_set_name: str,
    target_name: str,
    model_name: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    feature_names = get_feature_names(pipeline)
    model = pipeline.named_steps["model"]

    if hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_))
        order = np.argsort(values)[::-1][:MAX_IMPORTANCE_FEATURES]
        for rank, idx in enumerate(order, start=1):
            rows.append(
                {
                    "dataset": dataset_name,
                    "feature_set": feature_set_name,
                    "target": target_name,
                    "model": model_name,
                    "importance_type": "absolute_coefficient",
                    "rank": rank,
                    "feature": feature_names[idx],
                    "importance": float(values[idx]),
                }
            )

    if hasattr(model, "feature_importances_"):
        values = np.ravel(model.feature_importances_)
        order = np.argsort(values)[::-1][:MAX_IMPORTANCE_FEATURES]
        for rank, idx in enumerate(order, start=1):
            rows.append(
                {
                    "dataset": dataset_name,
                    "feature_set": feature_set_name,
                    "target": target_name,
                    "model": model_name,
                    "importance_type": "model_feature_importance",
                    "rank": rank,
                    "feature": feature_names[idx],
                    "importance": float(values[idx]),
                }
            )

    if model_name != "dummy_mean":
        result = permutation_importance(
            pipeline,
            X_test,
            y_test,
            n_repeats=10,
            random_state=RANDOM_STATE,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
        )
        order = np.argsort(result.importances_mean)[::-1][:MAX_IMPORTANCE_FEATURES]
        for rank, idx in enumerate(order, start=1):
            rows.append(
                {
                    "dataset": dataset_name,
                    "feature_set": feature_set_name,
                    "target": target_name,
                    "model": model_name,
                    "importance_type": "permutation_importance",
                    "rank": rank,
                    "feature": X_test.columns[idx],
                    "importance": float(result.importances_mean[idx]),
                }
            )
    return rows


def prediction_rows(
    dataset_name: str,
    feature_set_name: str,
    target_name: str,
    model_name: str,
    split_name: str,
    X_split: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in X_split.index:
        row = X_split.loc[idx].to_dict()
        row.update(
            {
                "dataset": dataset_name,
                "feature_set": feature_set_name,
                "target": target_name,
                "model": model_name,
                "split": split_name,
                "row_index": int(idx),
                "actual": float(y_true.loc[idx]),
                "predicted": float(y_pred[list(X_split.index).index(idx)]),
            }
        )
        row["residual"] = row["actual"] - row["predicted"]
        rows.append(row)
    return rows


def bias_summary_rows(
    dataset_name: str,
    feature_set_name: str,
    target_name: str,
    model_name: str,
    split_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, object]:
    residuals = y_true - y_pred
    low_threshold = float(y_true.quantile(0.25))
    high_threshold = float(y_true.quantile(0.75))
    low_mask = y_true <= low_threshold
    high_mask = y_true >= high_threshold
    return {
        "dataset": dataset_name,
        "feature_set": feature_set_name,
        "target": target_name,
        "model": model_name,
        "split": split_name,
        "overall_mean_residual": float(np.mean(residuals)),
        "overall_median_residual": float(np.median(residuals)),
        "low_actual_threshold": low_threshold,
        "high_actual_threshold": high_threshold,
        "mean_residual_low_actual": float(np.mean(residuals[low_mask])) if low_mask.any() else np.nan,
        "mean_residual_high_actual": float(np.mean(residuals[high_mask])) if high_mask.any() else np.nan,
        "overpredict_rate_low_actual": float(np.mean((residuals[low_mask] < 0))) if low_mask.any() else np.nan,
        "underpredict_rate_high_actual": float(np.mean((residuals[high_mask] > 0))) if high_mask.any() else np.nan,
    }


def train_for_target(
    dataset_name: str,
    frame: pd.DataFrame,
    feature_set_name: str,
    target_name: str,
    plots_dir: Path,
    diagnostics_plots_dir: Path,
    models_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    working = frame.copy()
    working[target_name] = pd.to_numeric(working[target_name], errors="coerce")
    working = working.dropna(subset=[target_name])

    X = make_feature_frame(working, feature_set_name)
    y = working[target_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    metric_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    prediction_record_rows: list[dict[str, object]] = []
    bias_rows: list[dict[str, object]] = []

    for model_name, pipeline in build_models(X).items():
        pipeline.fit(X_train, y_train)
        train_predictions = pipeline.predict(X_train)
        test_predictions = pipeline.predict(X_test)

        train_metrics = metric_summary(y_train, train_predictions)
        test_metrics = metric_summary(y_test, test_predictions)
        metric_rows.append(
            {
                "dataset": dataset_name,
                "feature_set": feature_set_name,
                "target": target_name,
                "model": model_name,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "mae": test_metrics["mae"],
                "rmse": test_metrics["rmse"],
                "r2": test_metrics["r2"],
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
                "mae_gap": test_metrics["mae"] - train_metrics["mae"],
                "rmse_gap": test_metrics["rmse"] - train_metrics["rmse"],
                "r2_gap": train_metrics["r2"] - test_metrics["r2"],
            }
        )

        stem = safe_slug(f"{dataset_name}__{feature_set_name}__{target_name}__{model_name}")
        current_plots_dir = model_plots_dir(plots_dir, model_name, feature_set_name)
        save_plot(
            y_test,
            test_predictions,
            current_plots_dir / f"{stem}__actual_vs_predicted.png",
            title=f"{dataset_name} | {feature_set_name} | {target_name} | {model_name}",
            x_label="Actual",
            y_label="Predicted",
        )
        save_residual_plot(
            y_test,
            test_predictions,
            current_plots_dir / f"{stem}__residuals.png",
            title=f"{dataset_name} | {feature_set_name} | {target_name} | {model_name} residuals",
        )

        model_diag_dir = diagnostics_plots_dir / safe_slug(model_name) / safe_slug(feature_set_name)
        ensure_dir(model_diag_dir)
        test_record_frame = X_test.copy()
        test_record_frame["residual"] = y_test.to_numpy() - test_predictions
        for feature in RESIDUAL_GROUP_COLUMNS:
            if feature in test_record_frame.columns:
                save_residual_feature_plot(
                    test_record_frame[[feature, "residual"]].copy(),
                    feature,
                    model_diag_dir / f"{stem}__residuals_by_{feature}.png",
                    title=f"{dataset_name} | {target_name} | {model_name} | residuals by {feature}",
                )

        with (models_dir / f"{stem}.pkl").open("wb") as handle:
            pickle.dump(pipeline, handle)

        importance_rows.extend(
            collect_importance_rows(
                pipeline,
                X_test,
                y_test,
                dataset_name,
                feature_set_name,
                target_name,
                model_name,
            )
        )

        prediction_record_rows.extend(
            prediction_rows(
                dataset_name,
                feature_set_name,
                target_name,
                model_name,
                "train",
                X_train,
                y_train,
                train_predictions,
            )
        )
        prediction_record_rows.extend(
            prediction_rows(
                dataset_name,
                feature_set_name,
                target_name,
                model_name,
                "test",
                X_test,
                y_test,
                test_predictions,
            )
        )

        bias_rows.append(
            bias_summary_rows(
                dataset_name,
                feature_set_name,
                target_name,
                model_name,
                "train",
                y_train,
                train_predictions,
            )
        )
        bias_rows.append(
            bias_summary_rows(
                dataset_name,
                feature_set_name,
                target_name,
                model_name,
                "test",
                y_test,
                test_predictions,
            )
        )

    return metric_rows, importance_rows, prediction_record_rows, bias_rows


def write_summary(
    ml_dir: Path,
    dataset_specs: list[DatasetSpec],
    targets: list[str],
    metrics: pd.DataFrame,
) -> None:
    best = (
        metrics.sort_values(["dataset", "feature_set", "target", "test_mae", "test_rmse", "model"])
        .groupby(["dataset", "feature_set", "target"], as_index=False)
        .first()
    )
    lines = [
        "Baseline ML summary for FoF-derived fragment statistics only.",
        "",
        "This workflow predicts FoF extraction outputs from simulation-level metadata.",
        "It does not model moon formation, bound debris, orbital capture, or other orbital outcomes.",
        "",
        f"Datasets evaluated: {', '.join(spec.name for spec in dataset_specs)}",
        f"Feature sets evaluated: {', '.join(FEATURE_SET_COLUMNS.keys())}",
        f"Targets evaluated: {', '.join(targets)}",
        "",
        "Dataset definitions:",
    ]
    for spec in dataset_specs:
        lines.append(f"- {spec.name}: {spec.description} ({len(spec.frame)} rows)")

    lines.extend(["", "Best model by target, dataset, and feature set (lowest test MAE):"])
    for _, row in best.iterrows():
        lines.append(
            f"- {row['dataset']} | {row['feature_set']} | {row['target']}: {row['model']} "
            f"(test_MAE={row['test_mae']:.3f}, test_RMSE={row['test_rmse']:.3f}, test_R2={row['test_r2']:.3f})"
        )

    (ml_dir / "ml_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarise_residual_groups(prediction_records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    test_records = prediction_records[prediction_records["split"] == "test"].copy()
    for feature in RESIDUAL_GROUP_COLUMNS:
        if feature not in test_records.columns:
            continue
        grouped = test_records.groupby(["dataset", "feature_set", "target", "model", feature], dropna=False)
        for keys, frame in grouped:
            dataset, feature_set, target, model, feature_value = keys
            rows.append(
                {
                    "dataset": dataset,
                    "feature_set": feature_set,
                    "target": target,
                    "model": model,
                    "group_feature": feature,
                    "group_value": feature_value,
                    "count": len(frame),
                    "mean_residual": float(frame["residual"].mean()),
                    "median_residual": float(frame["residual"].median()),
                    "mae_within_group": float(frame["residual"].abs().mean()),
                }
            )
    return pd.DataFrame(rows)


def build_target_difficulty_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    best = (
        metrics.sort_values(["dataset", "feature_set", "target", "test_mae", "test_rmse", "model"])
        .groupby(["dataset", "feature_set", "target"], as_index=False)
        .first()
    )
    best["difficulty_by_r2"] = best.groupby(["dataset", "feature_set"])["test_r2"].rank(ascending=False, method="dense")
    best["difficulty_by_mae"] = best.groupby(["dataset", "feature_set"])["test_mae"].rank(ascending=True, method="dense")
    return best[
        [
            "dataset",
            "feature_set",
            "target",
            "model",
            "test_mae",
            "test_rmse",
            "test_r2",
            "difficulty_by_r2",
            "difficulty_by_mae",
        ]
    ].sort_values(["dataset", "feature_set", "difficulty_by_r2", "difficulty_by_mae"])


def build_overfit_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics[
        [
            "dataset",
            "feature_set",
            "target",
            "model",
            "train_mae",
            "test_mae",
            "mae_gap",
            "train_rmse",
            "test_rmse",
            "rmse_gap",
            "train_r2",
            "test_r2",
            "r2_gap",
        ]
    ].sort_values(["dataset", "feature_set", "target", "model"])


def build_linking_length_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    best = (
        metrics.sort_values(["dataset", "feature_set", "target", "test_mae", "test_rmse", "model"])
        .groupby(["dataset", "feature_set", "target"], as_index=False)
        .first()
    )
    pivot = best.pivot_table(
        index=["dataset", "target"],
        columns="feature_set",
        values=["model", "test_mae", "test_rmse", "test_r2"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}__{feature_set}" for metric, feature_set in pivot.columns]
    frame = pivot.reset_index()
    if "test_mae__with_fof_linking_length" in frame.columns and "test_mae__without_fof_linking_length" in frame.columns:
        frame["mae_delta_without_minus_with"] = (
            frame["test_mae__without_fof_linking_length"] - frame["test_mae__with_fof_linking_length"]
        )
        frame["r2_delta_with_minus_without"] = (
            frame["test_r2__with_fof_linking_length"] - frame["test_r2__without_fof_linking_length"]
        )
    return frame.sort_values(["dataset", "target"])


def build_feature_stability_summary(importance_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    perm = importance_df[importance_df["importance_type"] == "permutation_importance"].copy()
    perm = perm[perm["rank"] <= 5]

    for (feature_set, target, model), group in perm.groupby(["feature_set", "target", "model"]):
        full_features = set(group[group["dataset"] == "full"]["feature"])
        clean_features = set(group[group["dataset"] == "clean_subset"]["feature"])
        union = full_features | clean_features
        rows.append(
            {
                "comparison": "full_vs_clean",
                "feature_set": feature_set,
                "target": target,
                "model": model,
                "top5_overlap_count": len(full_features & clean_features),
                "top5_jaccard": float(len(full_features & clean_features) / len(union)) if union else np.nan,
            }
        )

    for (dataset, target, model), group in perm.groupby(["dataset", "target", "model"]):
        with_features = set(group[group["feature_set"] == "with_fof_linking_length"]["feature"])
        without_features = set(group[group["feature_set"] == "without_fof_linking_length"]["feature"])
        union = with_features | without_features
        rows.append(
            {
                "comparison": "with_vs_without_fof_linking",
                "dataset": dataset,
                "target": target,
                "model": model,
                "top5_overlap_count": len(with_features & without_features),
                "top5_jaccard": float(len(with_features & without_features) / len(union)) if union else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_diagnostics_summary(
    diagnostics_dir: Path,
    metrics: pd.DataFrame,
    bias_df: pd.DataFrame,
    linking_df: pd.DataFrame,
    target_difficulty_df: pd.DataFrame,
) -> None:
    best = (
        metrics.sort_values(["dataset", "feature_set", "target", "test_mae", "test_rmse", "model"])
        .groupby(["dataset", "feature_set", "target"], as_index=False)
        .first()
    )
    hardest = target_difficulty_df.sort_values(["dataset", "feature_set", "difficulty_by_r2"], ascending=[True, True, False])
    bias_test = bias_df[bias_df["split"] == "test"].sort_values(["dataset", "feature_set", "target", "model"])

    lines = [
        "ML diagnostics summary for FoF-derived fragment statistics only.",
        "",
        "This analysis covers feature importance, residual behavior, overfitting checks, prediction bias,",
        "target difficulty, feature-stability comparisons, and the effect of including FoF linking length.",
        "",
        "Best test models by dataset, feature set, and target:",
    ]
    for _, row in best.iterrows():
        lines.append(
            f"- {row['dataset']} | {row['feature_set']} | {row['target']}: {row['model']} "
            f"(test_R2={row['test_r2']:.3f}, test_MAE={row['test_mae']:.3f}, r2_gap={row['r2_gap']:.3f})"
        )

    lines.extend(["", "Hardest targets by dataset and feature set (lowest best test R2):"])
    for (dataset, feature_set), group in hardest.groupby(["dataset", "feature_set"]):
        worst = group.iloc[0]
        lines.append(
            f"- {dataset} | {feature_set}: hardest target is {worst['target']} "
            f"(best test_R2={worst['test_r2']:.3f}, model={worst['model']})"
        )

    lines.extend(["", "Bias checks on test predictions:"])
    for _, row in bias_test.iterrows():
        if row["model"] not in {"random_forest", "gradient_boosting"}:
            continue
        lines.append(
            f"- {row['dataset']} | {row['feature_set']} | {row['target']} | {row['model']}: "
            f"mean_residual_high_actual={row['mean_residual_high_actual']:.3f}, "
            f"underpredict_rate_high_actual={row['underpredict_rate_high_actual']:.3f}, "
            f"mean_residual_low_actual={row['mean_residual_low_actual']:.3f}, "
            f"overpredict_rate_low_actual={row['overpredict_rate_low_actual']:.3f}"
        )

    lines.extend(["", "FoF linking length inclusion check (positive delta means including it helped):"])
    for _, row in linking_df.iterrows():
        lines.append(
            f"- {row['dataset']} | {row['target']}: "
            f"R2 delta with-vs-without={row.get('r2_delta_with_minus_without', np.nan):.3f}, "
            f"MAE delta without-minus-with={row.get('mae_delta_without_minus_with', np.nan):.3f}"
        )

    (diagnostics_dir / "diagnostics_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    df = add_engineered_features(load_dataset(args.dataset))
    targets = select_targets(df)
    if not targets:
        raise SystemExit("No valid target columns available in the FoF outcome table.")

    ml_dir = args.ml_dir
    tables_dir = ml_dir / "tables"
    plots_dir = ml_dir / "plots"
    models_dir = ml_dir / "models"
    diagnostics_dir = ml_dir / "model_diagnostics"
    diagnostics_tables_dir = diagnostics_dir / "tables"
    diagnostics_plots_dir = diagnostics_dir / "plots"
    for path in [tables_dir, plots_dir, models_dir, diagnostics_tables_dir, diagnostics_plots_dir]:
        ensure_dir(path)

    dataset_specs = build_dataset_specs(df)
    subset_rows = [
        {"dataset": spec.name, "rows": len(spec.frame), "description": spec.description}
        for spec in dataset_specs
    ]
    pd.DataFrame(subset_rows).to_csv(tables_dir / "dataset_summaries.csv", index=False)

    metric_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    prediction_record_rows: list[dict[str, object]] = []
    bias_rows: list[dict[str, object]] = []
    trained_targets: list[str] = []

    for feature_set_name in FEATURE_SET_COLUMNS:
        for target in targets:
            if target not in trained_targets:
                trained_targets.append(target)
            for spec in dataset_specs:
                rows, importance, prediction_rows_list, bias_rows_list = train_for_target(
                    spec.name,
                    spec.frame,
                    feature_set_name,
                    target,
                    plots_dir,
                    diagnostics_plots_dir,
                    models_dir,
                )
                metric_rows.extend(rows)
                importance_rows.extend(importance)
                prediction_record_rows.extend(prediction_rows_list)
                bias_rows.extend(bias_rows_list)

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["dataset", "feature_set", "target", "test_mae", "test_rmse", "model"]
    )
    metrics.to_csv(tables_dir / "model_metrics.csv", index=False)

    importance_df = pd.DataFrame(importance_rows).sort_values(
        ["dataset", "feature_set", "target", "model", "importance_type", "rank"]
    )
    importance_df.to_csv(tables_dir / "feature_importance.csv", index=False)

    prediction_df = pd.DataFrame(prediction_record_rows).sort_values(
        ["dataset", "feature_set", "target", "model", "split", "row_index"]
    )
    prediction_df.to_csv(diagnostics_tables_dir / "prediction_records.csv", index=False)

    bias_df = pd.DataFrame(bias_rows).sort_values(["dataset", "feature_set", "target", "model", "split"])
    bias_df.to_csv(diagnostics_tables_dir / "prediction_bias_summary.csv", index=False)

    residual_group_df = summarise_residual_groups(prediction_df)
    residual_group_df.to_csv(diagnostics_tables_dir / "residual_group_stats.csv", index=False)

    target_difficulty_df = build_target_difficulty_summary(metrics)
    target_difficulty_df.to_csv(diagnostics_tables_dir / "target_difficulty_summary.csv", index=False)

    overfit_df = build_overfit_summary(metrics)
    overfit_df.to_csv(diagnostics_tables_dir / "overfit_summary.csv", index=False)

    linking_df = build_linking_length_comparison(metrics)
    linking_df.to_csv(diagnostics_tables_dir / "fof_linking_length_comparison.csv", index=False)

    feature_stability_df = build_feature_stability_summary(importance_df)
    feature_stability_df.to_csv(diagnostics_tables_dir / "feature_stability_summary.csv", index=False)

    write_summary(ml_dir, dataset_specs, trained_targets, metrics)
    write_diagnostics_summary(diagnostics_dir, metrics, bias_df, linking_df, target_difficulty_df)

    print(f"Loaded {len(df)} simulation rows from {args.dataset}")
    print(f"Feature sets: {', '.join(FEATURE_SET_COLUMNS.keys())}")
    print(f"Trained targets: {', '.join(trained_targets)}")
    print(f"Wrote metrics to {tables_dir / 'model_metrics.csv'}")
    print(f"Wrote diagnostics to {diagnostics_dir}")
    print(f"Wrote models to {models_dir}")
    print(f"Wrote plots to {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
