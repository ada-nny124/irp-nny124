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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
N_SPLITS = 5
CLASSIFICATION_TARGET = "has_any_bound_mass"
REGRESSION_TARGET = "bound_mass_fraction"
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
    plots_dir: Path,
    models_dir: Path,
) -> list[dict[str, object]]:
    if frame.empty or frame[CLASSIFICATION_TARGET].nunique(dropna=True) < 2:
        return []
    X = make_feature_frame(frame, feature_set_name)
    y = frame[CLASSIFICATION_TARGET].astype(bool)
    groups = frame["physical_file"].astype(str)
    splitter = grouped_splitter(groups)
    metric_rows: list[dict[str, object]] = []

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
        metrics = classification_metric_summary(y, y_pred.loc[y.index].to_numpy(), score_values)
        metrics.update(
            {
                "task": "classification",
                "dataset": dataset_name,
                "feature_set": feature_set_name,
                "target": CLASSIFICATION_TARGET,
                "model": model_name,
                "rows": len(frame),
                "unique_physical_files": int(groups.nunique()),
            }
        )
        metric_rows.append(metrics)

        stem = safe_slug(f"{dataset_name}__{feature_set_name}__{CLASSIFICATION_TARGET}__{model_name}")
        current_plots_dir = model_plots_dir(plots_dir, model_name, feature_set_name)
        save_confusion_matrix_plot(y, y_pred.loc[y.index].to_numpy(), current_plots_dir / f"{stem}__confusion_matrix.png", title=f"{dataset_name} | {feature_set_name} | {model_name}")
        if score_values is not None:
            save_roc_curve_plot(y, score_values, current_plots_dir / f"{stem}__roc_curve.png", title=f"{dataset_name} | {feature_set_name} | {model_name}")

        final_pipeline = clone(base_pipeline)
        final_pipeline.fit(X, y)
        with (models_dir / f"{stem}.pkl").open("wb") as handle:
            pickle.dump(final_pipeline, handle)

    return metric_rows


def main() -> int:
    args = parse_args()
    ml_dir = args.ml_dir
    tables_dir = ml_dir / "tables"
    ensure_dir(tables_dir)

    df = add_engineered_features(load_dataset(args.dataset))
    dataset_specs = build_dataset_specs(df)
    write_dataset_summary(dataset_specs, tables_dir / "dataset_summaries.csv")

    print(f"Loaded {len(df)} successful bound outcome rows from {args.dataset}")
    print(f"Wrote dataset summary to {tables_dir / 'dataset_summaries.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
