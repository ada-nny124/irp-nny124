#!/usr/bin/env python3
"""Train baseline regressors for FoF-derived fragment statistics."""

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
    if "special_case_code" in clean.columns:
        clean = clean[clean["special_case_code"].fillna("") == ""]

    clean_desc = (
        "Subset with timestep == 90000, "
        f"resolution_code == {resolution_code}, "
        f"fof_linking_length == {fof_linking_length}, "
        "excluding special_case_code."
    )
    return [full, DatasetSpec(name="clean_subset", frame=clean.copy(), description=clean_desc)]


def build_preprocessor(X: pd.DataFrame, model_name: str) -> ColumnTransformer:
    numeric_features = [column for column in BASE_FEATURE_COLUMNS if column in X.columns and column != "spin_axis"]
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


def make_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[BASE_FEATURE_COLUMNS].copy()


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def safe_slug(text: str) -> str:
    return text.replace("/", "_").replace(" ", "_")


def model_plots_dir(plots_dir: Path, model_name: str) -> Path:
    path = plots_dir / safe_slug(model_name)
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


def get_feature_names(pipeline: Pipeline) -> np.ndarray:
    return pipeline.named_steps["preprocessor"].get_feature_names_out()


def collect_importance_rows(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    dataset_name: str,
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
                    "target": target_name,
                    "model": model_name,
                    "importance_type": "permutation_importance",
                    "rank": rank,
                    "feature": X_test.columns[idx] if idx < len(X_test.columns) else str(idx),
                    "importance": float(result.importances_mean[idx]),
                }
            )
    return rows


def train_for_target(
    dataset_name: str,
    frame: pd.DataFrame,
    target_name: str,
    plots_dir: Path,
    models_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    working = frame.copy()
    working[target_name] = pd.to_numeric(working[target_name], errors="coerce")
    working = working.dropna(subset=[target_name])

    X = make_feature_frame(working)
    y = working[target_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    metric_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []

    for model_name, pipeline in build_models(X).items():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        metric_rows.append(
            {
                "dataset": dataset_name,
                "target": target_name,
                "model": model_name,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "mae": float(mean_absolute_error(y_test, predictions)),
                "rmse": float(rmse(y_test, predictions)),
                "r2": float(r2_score(y_test, predictions)),
            }
        )

        stem = safe_slug(f"{dataset_name}__{target_name}__{model_name}")
        current_plots_dir = model_plots_dir(plots_dir, model_name)
        save_plot(
            y_test,
            predictions,
            current_plots_dir / f"{stem}__actual_vs_predicted.png",
            title=f"{dataset_name} | {target_name} | {model_name}",
            x_label="Actual",
            y_label="Predicted",
        )
        save_residual_plot(
            y_test,
            predictions,
            current_plots_dir / f"{stem}__residuals.png",
            title=f"{dataset_name} | {target_name} | {model_name} residuals",
        )

        with (models_dir / f"{stem}.pkl").open("wb") as handle:
            pickle.dump(pipeline, handle)

        importance_rows.extend(
            collect_importance_rows(pipeline, X_test, y_test, dataset_name, target_name, model_name)
        )

    return metric_rows, importance_rows


def write_summary(
    ml_dir: Path,
    dataset_specs: list[DatasetSpec],
    targets: list[str],
    metrics: pd.DataFrame,
) -> None:
    best = metrics.sort_values(["dataset", "target", "mae"]).groupby(["dataset", "target"], as_index=False).first()
    lines = [
        "Baseline ML summary for FoF-derived fragment statistics only.",
        "",
        "This workflow predicts FoF extraction outputs from simulation-level metadata.",
        "It does not model moon formation, bound debris, orbital capture, or other orbital outcomes.",
        "",
        f"Datasets evaluated: {', '.join(spec.name for spec in dataset_specs)}",
        f"Targets evaluated: {', '.join(targets)}",
        "",
        "Dataset definitions:",
    ]
    for spec in dataset_specs:
        lines.append(f"- {spec.name}: {spec.description} ({len(spec.frame)} rows)")

    lines.extend(["", "Best model by target and dataset (lowest MAE):"])
    for _, row in best.iterrows():
        lines.append(
            f"- {row['dataset']} | {row['target']}: {row['model']} "
            f"(MAE={row['mae']:.3f}, RMSE={row['rmse']:.3f}, R2={row['r2']:.3f})"
        )

    (ml_dir / "ml_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    for path in [tables_dir, plots_dir, models_dir]:
        ensure_dir(path)

    dataset_specs = build_dataset_specs(df)
    subset_rows = [
        {"dataset": spec.name, "rows": len(spec.frame), "description": spec.description}
        for spec in dataset_specs
    ]
    pd.DataFrame(subset_rows).to_csv(tables_dir / "dataset_summaries.csv", index=False)

    metric_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    trained_targets: list[str] = []

    for target in targets:
        trained_targets.append(target)
        for spec in dataset_specs:
            rows, importance = train_for_target(spec.name, spec.frame, target, plots_dir, models_dir)
            metric_rows.extend(rows)
            importance_rows.extend(importance)

    metrics = pd.DataFrame(metric_rows).sort_values(["dataset", "target", "mae", "rmse", "model"])
    metrics.to_csv(tables_dir / "model_metrics.csv", index=False)

    if importance_rows:
        importance_df = pd.DataFrame(importance_rows).sort_values(
            ["dataset", "target", "model", "importance_type", "rank"]
        )
        importance_df.to_csv(tables_dir / "feature_importance.csv", index=False)

    write_summary(ml_dir, dataset_specs, trained_targets, metrics)

    print(f"Loaded {len(df)} simulation rows from {args.dataset}")
    print(f"Trained targets: {', '.join(trained_targets)}")
    print(f"Wrote metrics to {tables_dir / 'model_metrics.csv'}")
    print(f"Wrote models to {models_dir}")
    print(f"Wrote plots to {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
