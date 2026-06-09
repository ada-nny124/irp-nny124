#!/usr/bin/env python3
"""Train grouped baseline models for run-level bound outcome prediction."""

from __future__ import annotations

import argparse
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, mean_absolute_error, precision_score, r2_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
N_SPLITS = 5
CLASSIFICATION_TARGET = "has_any_bound_mass"
REGRESSION_TARGET = "bound_mass_fraction"
HIGH_RETENTION_THRESHOLD = 0.1
RESIDUAL_GROUP_COLUMNS = ["periapsis_Rm", "mass_log10_kg", "v_inf_kms", "spin_axis", "fof_linking_length"]
FILENAME_RE = re.compile(
    r"^(?P<prefix>Ma_xp)_(?P<mass>A\d{4}(?:c30)?)(?:_(?P<spin>s\d{3}[A-Za-z]*))?"
    r"_n(?P<resolution>\d+)_r(?P<periapsis>\d+)_v(?P<velocity>\d+)"
    r"_(?P<timestep>\d+)"
    r"_fof_(?P<linking_length>[0-9.]+)_"
    r"(?P<chunk>\d+)\.hdf5$"
)
BASE_FEATURE_COLUMNS = [
    "mass_log10_kg",
    "particle_log10",
    "periapsis_Rm",
    "v_inf_kms",
    "spin_period_hr",
    "spin_axis",
    "has_explicit_spin",
    "special_case_code",
    "timestep",
    "fof_linking_length",
]
FEATURE_SET_COLUMNS = {
    "with_fof_linking_length": BASE_FEATURE_COLUMNS,
    "without_fof_linking_length": [column for column in BASE_FEATURE_COLUMNS if column != "fof_linking_length"],
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    frame: pd.DataFrame
    description: str


@dataclass(frozen=True)
class TargetSpec:
    name: str
    task: str
    source_column: str
    description: str


TARGET_SPECS = [
    TargetSpec(
        name="has_any_bound_mass",
        task="classification",
        source_column="has_any_bound_mass",
        description="Does a successful FoF run retain any bound mass at all?",
    ),
    TargetSpec(
        name="bound_mass_fraction_ge_0_1",
        task="classification",
        source_column="bound_mass_fraction_ge_0_1",
        description="Does a successful FoF run retain at least 10% of its mass in bound fragments?",
    ),
    TargetSpec(
        name="bound_mass_fraction",
        task="regression",
        source_column="bound_mass_fraction",
        description="What fraction of the run mass remains bound?",
    ),
    TargetSpec(
        name="bound_fragment_count",
        task="regression",
        source_column="bound_fragment_count",
        description="How many bound fragments are retained?",
    ),
    TargetSpec(
        name="largest_bound_fragment_mass_kg",
        task="regression",
        source_column="largest_bound_fragment_mass_kg",
        description="What is the largest bound fragment mass?",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("outputs/bound_outcomes.csv"),
        help="Bound outcome table with one row per FoF run.",
    )
    parser.add_argument(
        "--ml-dir",
        type=Path,
        default=Path("ml/bound_outcomes"),
        help="Output directory for bound outcome ML artifacts.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def parse_simulation_filename(filename: str) -> dict[str, object]:
    match = FILENAME_RE.match(filename)
    if not match:
        raise ValueError(f"Unrecognized FoF filename pattern: {filename}")

    mass_code = match.group("mass")
    spin_code = match.group("spin") or ""
    special_case_code = "c30" if mass_code.endswith("c30") else ""
    mass_digits = mass_code[1:5]
    spin_axis = spin_code[4:] if len(spin_code) > 4 else ""
    spin_value = spin_code[1:4] if spin_code else ""

    resolution_value = int(match.group("resolution"))
    periapsis_value = int(match.group("periapsis"))
    velocity_value = int(match.group("velocity"))
    timestep = int(match.group("timestep"))
    chunk_index = int(match.group("chunk"))
    linking_length = float(match.group("linking_length"))

    return {
        "filename": filename,
        "mass_code": mass_code,
        "mass_value": int(mass_digits),
        "special_case_code": special_case_code,
        "spin_code": spin_code,
        "spin_value": int(spin_value) if spin_value else "",
        "spin_axis": spin_axis,
        "has_explicit_spin": bool(spin_code),
        "resolution_code": f"n{resolution_value}",
        "resolution_value": resolution_value,
        "periapsis_code": f"r{periapsis_value}",
        "periapsis_value": periapsis_value,
        "velocity_code": f"v{velocity_value:02d}",
        "velocity_value": velocity_value,
        "timestep": timestep,
        "fof_linking_length": linking_length,
        "chunk_index": chunk_index,
    }


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    parsed = frame["fof_file"].map(parse_simulation_filename).apply(pd.Series)
    for column in parsed.columns:
        if column not in frame.columns:
            frame[column] = parsed[column]

    frame["mass_log10_kg"] = pd.to_numeric(frame["mass_value"], errors="coerce") / 100.0
    resolution_values = pd.to_numeric(frame["resolution_value"], errors="coerce")
    frame["particle_log10"] = resolution_values.map(lambda x: np.nan if pd.isna(x) else np.log10(x))
    frame["periapsis_Rm"] = pd.to_numeric(frame["periapsis_value"], errors="coerce") / 10.0
    frame["v_inf_kms"] = pd.to_numeric(frame["velocity_value"], errors="coerce") / 10.0
    frame["spin_period_hr"] = pd.to_numeric(frame["spin_value"], errors="coerce") / 10.0
    frame["has_explicit_spin"] = (
        frame["has_explicit_spin"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
    )
    frame["spin_axis"] = frame["spin_axis"].fillna("none").replace("", "none")
    frame["special_case_code"] = frame["special_case_code"].fillna("").replace("", "none")
    frame["has_any_bound_mass"] = pd.to_numeric(frame["bound_mass_fraction"], errors="coerce") > 0
    frame["bound_mass_fraction_ge_0_1"] = pd.to_numeric(frame["bound_mass_fraction"], errors="coerce") >= 0.1
    return frame


def build_dataset_specs(df: pd.DataFrame) -> list[DatasetSpec]:
    full = DatasetSpec(
        name="all_successful_runs",
        frame=df.copy(),
        description="All successful rows from outputs/bound_outcomes.csv.",
    )
    positive = df[pd.to_numeric(df["bound_mass_fraction"], errors="coerce") > 0].copy()
    positive_spec = DatasetSpec(
        name="positive_bound_runs",
        frame=positive,
        description="Only runs with bound_mass_fraction > 0.",
    )
    return [full, positive_spec]


def write_dataset_summary(dataset_specs: list[DatasetSpec], output_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for spec in dataset_specs:
        frame = spec.frame
        groups = frame["physical_file"].nunique(dropna=True)
        mixed_group_count = 0
        if "has_any_bound_mass" in frame.columns and not frame.empty:
            mixed_group_count = int(frame.groupby("physical_file")["has_any_bound_mass"].nunique().gt(1).sum())
        rows.append(
            {
                "dataset": spec.name,
                "rows": len(frame),
                "unique_physical_files": groups,
                "mean_bound_mass_fraction": pd.to_numeric(frame["bound_mass_fraction"], errors="coerce").mean(),
                "positive_bound_share": frame["has_any_bound_mass"].mean() if "has_any_bound_mass" in frame.columns and len(frame) else pd.NA,
                "mixed_label_physical_files": mixed_group_count,
                "description": spec.description,
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def build_preprocessor(X: pd.DataFrame, model_name: str) -> ColumnTransformer:
    categorical_features = [column for column in ["spin_axis", "special_case_code"] if column in X.columns]
    numeric_features = [column for column in X.columns if column not in categorical_features]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_name in {"logistic_regression", "ridge"}:
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


def build_classifier_models(X: pd.DataFrame) -> dict[str, Pipeline]:
    return {
        "dummy_most_frequent": Pipeline(
            [("preprocessor", build_preprocessor(X, "dummy_most_frequent")), ("model", DummyClassifier(strategy="most_frequent"))]
        ),
        "logistic_regression": Pipeline(
            [
                ("preprocessor", build_preprocessor(X, "logistic_regression")),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "random_forest_classifier": Pipeline(
            [
                ("preprocessor", build_preprocessor(X, "random_forest_classifier")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting_classifier": Pipeline(
            [
                ("preprocessor", build_preprocessor(X, "gradient_boosting_classifier")),
                ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
            ]
        ),
    }


def build_regressor_models(X: pd.DataFrame) -> dict[str, Pipeline]:
    return {
        "dummy_mean": Pipeline([("preprocessor", build_preprocessor(X, "dummy_mean")), ("model", DummyRegressor(strategy="mean"))]),
        "ridge": Pipeline([("preprocessor", build_preprocessor(X, "ridge")), ("model", Ridge(alpha=1.0))]),
        "random_forest_regressor": Pipeline(
            [
                ("preprocessor", build_preprocessor(X, "random_forest_regressor")),
                ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)),
            ]
        ),
        "gradient_boosting_regressor": Pipeline(
            [
                ("preprocessor", build_preprocessor(X, "gradient_boosting_regressor")),
                ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
            ]
        ),
    }


def make_feature_frame(df: pd.DataFrame, feature_set_name: str) -> pd.DataFrame:
    return df[FEATURE_SET_COLUMNS[feature_set_name]].copy()


def safe_slug(text: str) -> str:
    return text.replace("/", "_").replace(" ", "_")


def model_plots_dir(plots_dir: Path, model_name: str, feature_set_name: str) -> Path:
    path = plots_dir / safe_slug(model_name) / safe_slug(feature_set_name)
    ensure_dir(path)
    return path


def sort_or_empty(rows: list[dict[str, object]], sort_columns: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=sort_columns)
    return pd.DataFrame(rows).sort_values(sort_columns)


def get_feature_names(pipeline: Pipeline) -> np.ndarray:
    return pipeline.named_steps["preprocessor"].get_feature_names_out()


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return math.sqrt(np.mean((y_true - y_pred) ** 2))


def grouped_splitter(groups: pd.Series) -> GroupKFold:
    return GroupKFold(n_splits=min(N_SPLITS, groups.nunique()))


def classification_metric_summary(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray | None) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    metrics["roc_auc"] = float(roc_auc_score(y_true, y_score)) if y_score is not None and y_true.nunique() > 1 else np.nan
    return metrics


def save_confusion_matrix_plot(y_true: pd.Series, y_pred: np.ndarray, output_path: Path, title: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[False, True])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["False", "True"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["False", "True"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_roc_curve_plot(y_true: pd.Series, y_score: np.ndarray, output_path: Path, title: str) -> None:
    if y_true.nunique() < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train_classifiers_for_dataset(
    dataset_name: str,
    frame: pd.DataFrame,
    feature_set_name: str,
    target_spec: TargetSpec,
    plots_dir: Path,
    models_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if frame.empty or target_spec.source_column not in frame.columns:
        return [], []
    X = make_feature_frame(frame, feature_set_name)
    y = frame[target_spec.source_column].astype(bool)
    if y.nunique(dropna=True) < 2:
        return [], []
    groups = frame["physical_file"].astype(str)
    splitter = grouped_splitter(groups)
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for model_name, base_pipeline in build_classifier_models(X).items():
        y_pred = pd.Series(index=y.index, dtype="bool")
        y_score = pd.Series(index=y.index, dtype="float64")
        for train_idx, test_idx in splitter.split(X, y, groups):
            pipeline = clone(base_pipeline)
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            pipeline.fit(X_train, y_train)
            y_pred.loc[X_test.index] = pipeline.predict(X_test).astype(bool)
            if hasattr(pipeline, "predict_proba"):
                y_score.loc[X_test.index] = pipeline.predict_proba(X_test)[:, 1]

        score_values = y_score.loc[y.index].to_numpy() if y_score.notna().any() else None
        test_metrics = classification_metric_summary(y, y_pred.loc[y.index].to_numpy(), score_values)

        final_pipeline = clone(base_pipeline)
        final_pipeline.fit(X, y)
        train_pred = final_pipeline.predict(X).astype(bool)
        train_score = final_pipeline.predict_proba(X)[:, 1] if hasattr(final_pipeline, "predict_proba") else None
        train_metrics = classification_metric_summary(y, train_pred, train_score)

        metrics = {
            **test_metrics,
            **{f"train_{key}": value for key, value in train_metrics.items()},
        }
        metrics.update(
            {
                "task": "classification",
                "dataset": dataset_name,
                "feature_set": feature_set_name,
                "target": target_spec.name,
                "model": model_name,
                "rows": len(frame),
                "unique_physical_files": int(groups.nunique()),
                "source_column": target_spec.source_column,
            }
        )
        metric_rows.append(metrics)

        stem = safe_slug(f"{dataset_name}__{feature_set_name}__{target_spec.name}__{model_name}")
        current_plots_dir = model_plots_dir(plots_dir, model_name, feature_set_name)
        save_confusion_matrix_plot(y, y_pred.loc[y.index].to_numpy(), current_plots_dir / f"{stem}__confusion_matrix.png", title=f"{dataset_name} | {feature_set_name} | {model_name}")
        if score_values is not None:
            save_roc_curve_plot(y, score_values, current_plots_dir / f"{stem}__roc_curve.png", title=f"{dataset_name} | {feature_set_name} | {model_name}")

        with (models_dir / f"{stem}.pkl").open("wb") as handle:
            pickle.dump(final_pipeline, handle)

        prediction_rows.extend(
            {
                "task": "classification",
                "dataset": dataset_name,
                "feature_set": feature_set_name,
                "target": target_spec.name,
                "model": model_name,
                "physical_file": row["physical_file"],
                "actual": bool(y.loc[index]),
                "predicted": bool(y_pred.loc[index]),
                "score": float(y_score.loc[index]) if pd.notna(y_score.loc[index]) else np.nan,
                "residual": float(int(y.loc[index]) - int(bool(y_pred.loc[index]))),
                **{column: row[column] for column in X.columns},
            }
            for index, row in frame.loc[y.index].iterrows()
        )

    return metric_rows, prediction_rows


def regression_metric_summary(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_regression_scatter_plot(y_true: pd.Series, y_pred: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_true, y_pred, alpha=0.7, color="#1f77b4", edgecolors="none")
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="#d62728", linewidth=1.2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_regression_residual_plot(y_true: pd.Series, y_pred: np.ndarray, output_path: Path, title: str) -> None:
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_pred, residuals, alpha=0.7, color="#ff7f0e", edgecolors="none")
    ax.axhline(0.0, linestyle="--", color="#222222", linewidth=1.2)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train_regressors_for_dataset(
    dataset_name: str,
    frame: pd.DataFrame,
    feature_set_name: str,
    target_spec: TargetSpec,
    plots_dir: Path,
    models_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    working = frame.copy()
    if target_spec.source_column not in working.columns:
        return [], []
    working[target_spec.source_column] = pd.to_numeric(working[target_spec.source_column], errors="coerce")
    working = working.dropna(subset=[target_spec.source_column])
    if working.empty or working[target_spec.source_column].nunique(dropna=True) < 2:
        return [], []

    X = make_feature_frame(working, feature_set_name)
    y = working[target_spec.source_column]
    groups = working["physical_file"].astype(str)
    splitter = grouped_splitter(groups)
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for model_name, base_pipeline in build_regressor_models(X).items():
        y_pred = pd.Series(index=y.index, dtype="float64")
        for train_idx, test_idx in splitter.split(X, y, groups):
            pipeline = clone(base_pipeline)
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            pipeline.fit(X_train, y_train)
            y_pred.loc[X_test.index] = pipeline.predict(X_test)

        metrics = regression_metric_summary(y, y_pred.loc[y.index].to_numpy())
        final_pipeline = clone(base_pipeline)
        final_pipeline.fit(X, y)
        train_pred = final_pipeline.predict(X)
        train_metrics = regression_metric_summary(y, train_pred)
        metrics = {
            **metrics,
            **{f"train_{key}": value for key, value in train_metrics.items()},
        }
        metrics.update(
            {
                "task": "regression",
                "dataset": dataset_name,
                "feature_set": feature_set_name,
                "target": target_spec.name,
                "model": model_name,
                "rows": len(working),
                "unique_physical_files": int(groups.nunique()),
                "source_column": target_spec.source_column,
            }
        )
        metric_rows.append(metrics)

        stem = safe_slug(f"{dataset_name}__{feature_set_name}__{target_spec.name}__{model_name}")
        current_plots_dir = model_plots_dir(plots_dir, model_name, feature_set_name)
        save_regression_scatter_plot(y, y_pred.loc[y.index].to_numpy(), current_plots_dir / f"{stem}__actual_vs_predicted.png", title=f"{dataset_name} | {feature_set_name} | {model_name}")
        save_regression_residual_plot(y, y_pred.loc[y.index].to_numpy(), current_plots_dir / f"{stem}__residuals.png", title=f"{dataset_name} | {feature_set_name} | {model_name}")

        with (models_dir / f"{stem}.pkl").open("wb") as handle:
            pickle.dump(final_pipeline, handle)

        prediction_rows.extend(
            {
                "task": "regression",
                "dataset": dataset_name,
                "feature_set": feature_set_name,
                "target": target_spec.name,
                "model": model_name,
                "physical_file": row["physical_file"],
                "actual": float(y.loc[index]),
                "predicted": float(y_pred.loc[index]),
                "score": np.nan,
                "residual": float(y.loc[index] - y_pred.loc[index]),
                **{column: row[column] for column in X.columns},
            }
            for index, row in working.loc[y.index].iterrows()
        )

    return metric_rows, prediction_rows


def best_rows(metrics: pd.DataFrame, sort_columns: list[str], group_columns: list[str]) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    return metrics.sort_values(sort_columns).groupby(group_columns, as_index=False).first()


def write_classification_summary(ml_dir: Path, metrics: pd.DataFrame) -> None:
    if metrics.empty:
        lines = ["Bound outcome classification summary.", "", "No classification models were trained."]
    else:
        best = best_rows(
            metrics,
            ["target", "dataset", "feature_set", "balanced_accuracy", "f1", "roc_auc", "model"],
            ["target", "dataset", "feature_set"],
        )
        lines = [
            "Bound outcome classification summary.",
            "",
            "This stage predicts whether a successful FoF run retains any bound mass at all, plus a 10% retention threshold.",
            "Evaluation uses grouped folds by physical_file so alternate FoF linking lengths from the same physical snapshot stay together.",
            "",
            "Best classifiers by target, dataset, and feature set:",
        ]
        for _, row in best.iterrows():
            lines.append(
                f"- {row['target']} | {row['dataset']} | {row['feature_set']}: {row['model']} "
                f"(balanced_accuracy={row['balanced_accuracy']:.3f}, f1={row['f1']:.3f}, roc_auc={row['roc_auc']:.3f})"
            )
    (ml_dir / "classification_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_regression_summary(ml_dir: Path, metrics: pd.DataFrame) -> None:
    if metrics.empty:
        lines = ["Bound outcome regression summary.", "", "No regression models were trained."]
    else:
        best = best_rows(
            metrics,
            ["target", "dataset", "feature_set", "mae", "rmse", "model"],
            ["target", "dataset", "feature_set"],
        )
        lines = [
            "Bound outcome regression summary.",
            "",
            "This stage predicts the retained bound mass fraction, bound fragment count, and the largest bound fragment mass.",
            "Evaluation again uses grouped folds by physical_file so different FoF linking lengths from the same physical case do not leak across folds.",
            "",
            "Best regressors by target, dataset, and feature set:",
        ]
        for _, row in best.iterrows():
            lines.append(
                f"- {row['target']} | {row['dataset']} | {row['feature_set']}: {row['model']} "
                f"(mae={row['mae']:.4f}, rmse={row['rmse']:.4f}, r2={row['r2']:.3f})"
            )
    (ml_dir / "regression_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_prediction_records(output_path: Path, prediction_rows: list[dict[str, object]]) -> None:
    pd.DataFrame(prediction_rows).to_csv(output_path, index=False)


def write_overfit_summary(output_path: Path, metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in metrics.iterrows():
        if row["task"] == "classification":
            rows.append(
                {
                    "task": row["task"],
                    "dataset": row["dataset"],
                    "feature_set": row["feature_set"],
                    "target": row["target"],
                    "model": row["model"],
                    "rows": row["rows"],
                    "unique_physical_files": row["unique_physical_files"],
                    "train_accuracy": row["train_accuracy"],
                    "test_accuracy": row["accuracy"],
                    "accuracy_gap": row["train_accuracy"] - row["accuracy"],
                    "train_balanced_accuracy": row["train_balanced_accuracy"],
                    "test_balanced_accuracy": row["balanced_accuracy"],
                    "balanced_accuracy_gap": row["train_balanced_accuracy"] - row["balanced_accuracy"],
                    "train_f1": row["train_f1"],
                    "test_f1": row["f1"],
                    "f1_gap": row["train_f1"] - row["f1"],
                    "train_roc_auc": row["train_roc_auc"],
                    "test_roc_auc": row["roc_auc"],
                    "roc_auc_gap": row["train_roc_auc"] - row["roc_auc"],
                }
            )
        else:
            rows.append(
                {
                    "task": row["task"],
                    "dataset": row["dataset"],
                    "feature_set": row["feature_set"],
                    "target": row["target"],
                    "model": row["model"],
                    "rows": row["rows"],
                    "unique_physical_files": row["unique_physical_files"],
                    "train_mae": row["train_mae"],
                    "test_mae": row["mae"],
                    "mae_gap": row["mae"] - row["train_mae"],
                    "train_rmse": row["train_rmse"],
                    "test_rmse": row["rmse"],
                    "rmse_gap": row["rmse"] - row["train_rmse"],
                    "train_r2": row["train_r2"],
                    "test_r2": row["r2"],
                    "r2_gap": row["train_r2"] - row["r2"],
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_path, index=False)
    return summary


def write_calibration_summary(output_path: Path, predictions: pd.DataFrame) -> pd.DataFrame:
    classification = predictions[predictions["task"] == "classification"].copy()
    if classification.empty:
        summary = pd.DataFrame(columns=["task", "dataset", "feature_set", "target", "model", "bin", "count", "mean_score", "actual_rate"])
        summary.to_csv(output_path, index=False)
        return summary

    rows: list[dict[str, object]] = []
    group_columns = ["task", "dataset", "feature_set", "target", "model"]
    for keys, group in classification.groupby(group_columns, dropna=False):
        scores = pd.to_numeric(group["score"], errors="coerce")
        actuals = pd.to_numeric(group["actual"], errors="coerce")
        valid = pd.DataFrame({"score": scores, "actual": actuals}).dropna()
        if valid.empty:
            continue
        bins = min(5, valid["score"].nunique())
        if bins < 2:
            rows.append(
                {
                    **{column: value for column, value in zip(group_columns, keys)},
                    "bin": "all",
                    "count": int(len(valid)),
                    "mean_score": float(valid["score"].mean()),
                    "actual_rate": float(valid["actual"].mean()),
                }
            )
            continue
        bin_labels = pd.qcut(valid["score"].rank(method="first"), q=bins, duplicates="drop")
        valid = valid.assign(score_bin=bin_labels)
        for label, subset in valid.groupby("score_bin", dropna=False):
            rows.append(
                {
                    **{column: value for column, value in zip(group_columns, keys)},
                    "bin": str(label),
                    "count": int(len(subset)),
                    "mean_score": float(subset["score"].mean()),
                    "actual_rate": float(subset["actual"].mean()),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_path, index=False)
    return summary


def write_prediction_bias_summary(output_path: Path, predictions: pd.DataFrame) -> pd.DataFrame:
    regression = predictions[predictions["task"] == "regression"].copy()
    if regression.empty:
        summary = pd.DataFrame(columns=["task", "dataset", "feature_set", "target", "model", "bucket", "count", "mean_actual", "mean_predicted", "mean_residual", "underprediction_rate"])
        summary.to_csv(output_path, index=False)
        return summary

    rows: list[dict[str, object]] = []
    group_columns = ["task", "dataset", "feature_set", "target", "model"]
    for keys, group in regression.groupby(group_columns, dropna=False):
        actuals = pd.to_numeric(group["actual"], errors="coerce")
        predicted = pd.to_numeric(group["predicted"], errors="coerce")
        residual = pd.to_numeric(group["residual"], errors="coerce")
        valid = pd.DataFrame({"actual": actuals, "predicted": predicted, "residual": residual}).dropna()
        if valid.empty:
            continue
        q25 = valid["actual"].quantile(0.25)
        q75 = valid["actual"].quantile(0.75)
        buckets = {
            "low_actual": valid[valid["actual"] <= q25],
            "high_actual": valid[valid["actual"] >= q75],
        }
        for bucket_name, subset in buckets.items():
            if subset.empty:
                continue
            rows.append(
                {
                    **{column: value for column, value in zip(group_columns, keys)},
                    "bucket": bucket_name,
                    "count": int(len(subset)),
                    "mean_actual": float(subset["actual"].mean()),
                    "mean_predicted": float(subset["predicted"].mean()),
                    "mean_residual": float(subset["residual"].mean()),
                    "underprediction_rate": float((subset["residual"] > 0).mean()),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_path, index=False)
    return summary


def write_residual_group_stats(output_path: Path, predictions: pd.DataFrame) -> pd.DataFrame:
    regression = predictions[predictions["task"] == "regression"].copy()
    if regression.empty:
        summary = pd.DataFrame(columns=["task", "dataset", "feature_set", "target", "model", "group_column", "group_value", "count", "mean_residual", "mean_absolute_error", "underprediction_rate"])
        summary.to_csv(output_path, index=False)
        return summary

    rows: list[dict[str, object]] = []
    group_columns = ["task", "dataset", "feature_set", "target", "model"]
    for group_column in RESIDUAL_GROUP_COLUMNS:
        if group_column not in regression.columns:
            continue
        for keys, group in regression.groupby(group_columns, dropna=False):
            working = group[[group_column, "residual", "actual", "predicted"]].copy()
            working = working.dropna(subset=["residual"])
            if working.empty:
                continue
            grouped = working.groupby(group_column, dropna=False)
            for group_value, subset in grouped:
                rows.append(
                    {
                        **{column: value for column, value in zip(group_columns, keys)},
                        "group_column": group_column,
                        "group_value": group_value,
                        "count": int(len(subset)),
                        "mean_residual": float(subset["residual"].mean()),
                        "mean_absolute_error": float(np.abs(subset["residual"]).mean()),
                        "underprediction_rate": float((subset["residual"] > 0).mean()),
                    }
                )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_path, index=False)
    return summary


def write_fof_linking_length_comparison(output_path: Path, metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        summary = pd.DataFrame()
        summary.to_csv(output_path, index=False)
        return summary

    rows: list[dict[str, object]] = []
    comparison_keys = ["task", "dataset", "target", "model"]
    for keys, group in metrics.groupby(comparison_keys, dropna=False):
        if set(group["feature_set"]) != {"with_fof_linking_length", "without_fof_linking_length"}:
            continue
        with_row = group[group["feature_set"] == "with_fof_linking_length"].iloc[0]
        without_row = group[group["feature_set"] == "without_fof_linking_length"].iloc[0]
        base = {column: value for column, value in zip(comparison_keys, keys)}
        base.update(
            {
                "with_feature_set": "with_fof_linking_length",
                "without_feature_set": "without_fof_linking_length",
                "rows": int(with_row["rows"]),
                "unique_physical_files": int(with_row["unique_physical_files"]),
            }
        )
        if with_row["task"] == "classification":
            base.update(
                {
                    "test_balanced_accuracy_with": with_row["balanced_accuracy"],
                    "test_balanced_accuracy_without": without_row["balanced_accuracy"],
                    "balanced_accuracy_delta_with_minus_without": with_row["balanced_accuracy"] - without_row["balanced_accuracy"],
                    "test_f1_with": with_row["f1"],
                    "test_f1_without": without_row["f1"],
                    "f1_delta_with_minus_without": with_row["f1"] - without_row["f1"],
                    "test_roc_auc_with": with_row["roc_auc"],
                    "test_roc_auc_without": without_row["roc_auc"],
                    "roc_auc_delta_with_minus_without": with_row["roc_auc"] - without_row["roc_auc"],
                }
            )
        else:
            base.update(
                {
                    "test_mae_with": with_row["mae"],
                    "test_mae_without": without_row["mae"],
                    "mae_delta_without_minus_with": without_row["mae"] - with_row["mae"],
                    "test_rmse_with": with_row["rmse"],
                    "test_rmse_without": without_row["rmse"],
                    "rmse_delta_without_minus_with": without_row["rmse"] - with_row["rmse"],
                    "test_r2_with": with_row["r2"],
                    "test_r2_without": without_row["r2"],
                    "r2_delta_with_minus_without": with_row["r2"] - without_row["r2"],
                }
            )
        rows.append(base)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_path, index=False)
    return summary


def write_target_difficulty_summary(output_path: Path, metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        summary = pd.DataFrame()
        summary.to_csv(output_path, index=False)
        return summary

    rows: list[dict[str, object]] = []
    for target, group in metrics.groupby("target", dropna=False):
        if group.iloc[0]["task"] == "classification":
            best = group.sort_values(["balanced_accuracy", "f1", "roc_auc", "model"], ascending=[False, False, False, True]).iloc[0]
            score_name = "balanced_accuracy"
            score_value = best["balanced_accuracy"]
        else:
            best = group.sort_values(["r2", "mae", "rmse", "model"], ascending=[False, True, True, True]).iloc[0]
            score_name = "r2"
            score_value = best["r2"]
        rows.append(
            {
                "target": target,
                "task": best["task"],
                "best_dataset": best["dataset"],
                "best_feature_set": best["feature_set"],
                "best_model": best["model"],
                "best_metric_name": score_name,
                "best_metric_value": score_value,
            }
        )
    summary = pd.DataFrame(rows).sort_values(["task", "best_metric_value"], ascending=[True, False])
    summary.to_csv(output_path, index=False)
    return summary


def main() -> int:
    args = parse_args()
    ml_dir = args.ml_dir
    tables_dir = ml_dir / "tables"
    plots_dir = ml_dir / "plots"
    models_dir = ml_dir / "models"
    for path in [tables_dir, plots_dir, models_dir]:
        ensure_dir(path)

    df = add_engineered_features(load_dataset(args.dataset))
    dataset_specs = build_dataset_specs(df)
    write_dataset_summary(dataset_specs, tables_dir / "dataset_summaries.csv")

    classification_rows: list[dict[str, object]] = []
    regression_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for feature_set_name in FEATURE_SET_COLUMNS:
        for spec in dataset_specs:
            for target_spec in TARGET_SPECS:
                if target_spec.task == "classification":
                    rows, predictions = train_classifiers_for_dataset(
                        spec.name,
                        spec.frame,
                        feature_set_name,
                        target_spec,
                        plots_dir,
                        models_dir,
                    )
                    classification_rows.extend(rows)
                    prediction_rows.extend(predictions)
                else:
                    rows, predictions = train_regressors_for_dataset(
                        spec.name,
                        spec.frame,
                        feature_set_name,
                        target_spec,
                        plots_dir,
                        models_dir,
                    )
                    regression_rows.extend(rows)
                    prediction_rows.extend(predictions)

    classification_metrics = sort_or_empty(
        classification_rows,
        ["target", "dataset", "feature_set", "balanced_accuracy", "f1", "model"],
    )
    classification_metrics.to_csv(tables_dir / "classification_metrics.csv", index=False)
    write_classification_summary(ml_dir, classification_metrics)
    regression_metrics = sort_or_empty(
        regression_rows,
        ["target", "dataset", "feature_set", "mae", "rmse", "model"],
    )
    regression_metrics.to_csv(tables_dir / "regression_metrics.csv", index=False)
    write_regression_summary(ml_dir, regression_metrics)
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(tables_dir / "prediction_records.csv", index=False)
    overfit_summary = write_overfit_summary(tables_dir / "overfit_summary.csv", pd.concat([classification_metrics, regression_metrics], ignore_index=True))
    calibration_summary = write_calibration_summary(tables_dir / "calibration_summary.csv", predictions)
    bias_summary = write_prediction_bias_summary(tables_dir / "prediction_bias_summary.csv", predictions)
    residual_summary = write_residual_group_stats(tables_dir / "residual_group_stats.csv", predictions)
    fof_comparison = write_fof_linking_length_comparison(tables_dir / "fof_linking_length_comparison.csv", pd.concat([classification_metrics, regression_metrics], ignore_index=True))
    target_difficulty = write_target_difficulty_summary(tables_dir / "target_difficulty_summary.csv", pd.concat([classification_metrics, regression_metrics], ignore_index=True))

    print(f"Loaded {len(df)} successful bound outcome rows from {args.dataset}")
    print(f"Wrote dataset summary to {tables_dir / 'dataset_summaries.csv'}")
    print(f"Wrote classification metrics to {tables_dir / 'classification_metrics.csv'}")
    print(f"Wrote regression metrics to {tables_dir / 'regression_metrics.csv'}")
    print(f"Wrote prediction records to {tables_dir / 'prediction_records.csv'}")
    print(f"Wrote overfit summary to {tables_dir / 'overfit_summary.csv'}")
    print(f"Wrote calibration summary to {tables_dir / 'calibration_summary.csv'}")
    print(f"Wrote prediction bias summary to {tables_dir / 'prediction_bias_summary.csv'}")
    print(f"Wrote residual group stats to {tables_dir / 'residual_group_stats.csv'}")
    print(f"Wrote FoF linking length comparison to {tables_dir / 'fof_linking_length_comparison.csv'}")
    print(f"Wrote target difficulty summary to {tables_dir / 'target_difficulty_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
