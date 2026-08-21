#!/usr/bin/env python3
"""Train requested model-optimization candidates on the BMF surrogate task."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from scripts.eda.train_physics_structured_surrogate import (
    PHYSICS_FEATURE_COLUMNS,
    RANDOM_STATE,
    add_physics_features,
    build_group_folds,
    build_preprocessor,
    feature_columns_for_set,
    load_canonical_dataset,
)


OUTPUT_ROOT = Path("ml/model_optimization_candidates")
TABLES_DIR = OUTPUT_ROOT / "tables"
REPORTS_DIR = OUTPUT_ROOT / "reports"
PRIMARY_TARGET = "bound_mass_fraction"
FEATURE_SET_NAME = "with_fof_linking_length"
LEAKY_FEATURES = {"largest_fragment_mass_fraction"}
PYRSR_FEATURE_COLUMNS = [
    "mass_log10_kg",
    "particle_log10",
    "periapsis_Rm",
    "v_inf_kms",
    "spin_period_hr",
    "fof_linking_length",
    "encounter_eccentricity_proxy",
    "periapsis_inverse",
    "angular_momentum_proxy",
    "time_within_2_mars_radii_hr",
    "time_within_tidal_disruption_hr",
    "spin_frequency_hr_inv",
    "particle_mass_proxy",
    "mass_resolution_interaction",
]


@dataclass(frozen=True)
class ModelResult:
    model_key: str
    model_label: str
    task_scope: str
    metrics: dict[str, Any]
    architecture: list[str]
    notes: list[str]
    status: str
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("extraction-outputs/tables/bound_outcomes.csv"),
        help="Canonical bound outcomes table.",
    )
    return parser.parse_args()


def ensure_output_dirs() -> None:
    for path in [OUTPUT_ROOT, TABLES_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def safe_feature_columns() -> list[str]:
    columns = feature_columns_for_set(FEATURE_SET_NAME, include_physics=True)
    return [column for column in columns if column not in LEAKY_FEATURES]


def prepare_frame(dataset_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    frame = load_canonical_dataset(dataset_path)
    frame = add_physics_features(frame)
    fold_assignments = build_group_folds(frame, frame["physical_file"].astype(str), TABLES_DIR / "fold_assignments.csv")
    return frame, fold_assignments, safe_feature_columns()


def categorical_columns(columns: list[str]) -> list[str]:
    return [column for column in columns if column in {"spin_axis", "special_case_code"}]


def clip_bmf(preds: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(preds, dtype=float), 0.0, 1.0)


def metric_payload(y_true: pd.Series, y_pred: np.ndarray, fold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    y_pred = clip_bmf(y_pred)
    fold_frame = pd.DataFrame(fold_rows)
    return {
        "rows": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "fold_r2_mean": float(fold_frame["r2"].mean()),
        "fold_r2_std": float(fold_frame["r2"].std(ddof=0)),
        "fold_mae_mean": float(fold_frame["mae"].mean()),
        "fold_mae_std": float(fold_frame["mae"].std(ddof=0)),
        "fold_rmse_mean": float(fold_frame["rmse"].mean()),
        "fold_rmse_std": float(fold_frame["rmse"].std(ddof=0)),
        "fold_metrics": fold_frame.to_dict(orient="records"),
    }


def evaluate_preprocessed_model(
    frame: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    feature_columns: list[str],
    model_key: str,
    model_label: str,
    estimator_factory,
    architecture: list[str],
    notes: list[str],
) -> ModelResult:
    valid = frame[frame[PRIMARY_TARGET].notna()].copy()
    valid = valid.merge(fold_assignments[["row_index", "fold_index"]], left_index=True, right_on="row_index", how="left")
    X = valid[feature_columns].copy()
    y = pd.to_numeric(valid[PRIMARY_TARGET], errors="coerce")
    preprocessor = build_preprocessor(X, scaled=False)

    oof = np.full(len(valid), np.nan)
    fold_rows: list[dict[str, Any]] = []
    for fold_index in sorted(valid["fold_index"].dropna().unique()):
        train_mask = valid["fold_index"] != fold_index
        test_mask = valid["fold_index"] == fold_index
        fitted_preprocessor = clone(preprocessor)
        X_train = fitted_preprocessor.fit_transform(X.loc[train_mask])
        X_test = fitted_preprocessor.transform(X.loc[test_mask])
        model = estimator_factory()
        model.fit(X_train, y.loc[train_mask])
        preds = clip_bmf(model.predict(X_test))
        oof[test_mask.to_numpy()] = preds
        fold_rows.append(
            {
                "fold_index": int(fold_index),
                "r2": float(r2_score(y.loc[test_mask], preds)),
                "mae": float(mean_absolute_error(y.loc[test_mask], preds)),
                "rmse": float(np.sqrt(mean_squared_error(y.loc[test_mask], preds))),
            }
        )

    metrics = metric_payload(y, oof, fold_rows)
    return ModelResult(
        model_key=model_key,
        model_label=model_label,
        task_scope="full grouped-CV BMF regression",
        metrics=metrics,
        architecture=architecture,
        notes=notes,
        status="ok",
    )


def evaluate_catboost(
    frame: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    feature_columns: list[str],
) -> ModelResult:
    valid = frame[frame[PRIMARY_TARGET].notna()].copy()
    valid = valid.merge(fold_assignments[["row_index", "fold_index"]], left_index=True, right_on="row_index", how="left")
    X = valid[feature_columns].copy()
    y = pd.to_numeric(valid[PRIMARY_TARGET], errors="coerce")
    cat_columns = categorical_columns(feature_columns)
    cat_indices = [X.columns.get_loc(column) for column in cat_columns]

    oof = np.full(len(valid), np.nan)
    fold_rows: list[dict[str, Any]] = []
    for fold_index in sorted(valid["fold_index"].dropna().unique()):
        train_mask = valid["fold_index"] != fold_index
        test_mask = valid["fold_index"] == fold_index
        model = CatBoostRegressor(
            iterations=600,
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=RANDOM_STATE,
            verbose=False,
        )
        model.fit(X.loc[train_mask], y.loc[train_mask], cat_features=cat_indices)
        preds = clip_bmf(model.predict(X.loc[test_mask]))
        oof[test_mask.to_numpy()] = preds
        fold_rows.append(
            {
                "fold_index": int(fold_index),
                "r2": float(r2_score(y.loc[test_mask], preds)),
                "mae": float(mean_absolute_error(y.loc[test_mask], preds)),
                "rmse": float(np.sqrt(mean_squared_error(y.loc[test_mask], preds))),
            }
        )

    metrics = metric_payload(y, oof, fold_rows)
    return ModelResult(
        model_key="catboost",
        model_label="CatBoost regressor",
        task_scope="full grouped-CV BMF regression",
        metrics=metrics,
        architecture=[
            "Raw tabular CatBoostRegressor with native categorical handling for `spin_axis` and `special_case_code`.",
            "Feature set: `with_fof_linking_length` plus leakage-free physics features.",
            "Hyperparameters: `iterations=600`, `learning_rate=0.05`, `depth=6`, `loss_function=RMSE`.",
        ],
        notes=[
            "CatBoost is the primary modern boosted-tree benchmark for this run.",
            "Predictions are clipped to the physical BMF range `[0, 1]`.",
        ],
        status="ok",
    )


def evaluate_hurdle_model(
    frame: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    feature_columns: list[str],
) -> ModelResult:
    valid = frame[frame[PRIMARY_TARGET].notna()].copy()
    valid = valid.merge(fold_assignments[["row_index", "fold_index"]], left_index=True, right_on="row_index", how="left")
    X = valid[feature_columns].copy()
    y = pd.to_numeric(valid[PRIMARY_TARGET], errors="coerce")
    is_positive = (y > 0).astype(int)
    cat_columns = categorical_columns(feature_columns)
    cat_indices = [X.columns.get_loc(column) for column in cat_columns]

    oof = np.full(len(valid), np.nan)
    fold_rows: list[dict[str, Any]] = []
    for fold_index in sorted(valid["fold_index"].dropna().unique()):
        train_mask = valid["fold_index"] != fold_index
        test_mask = valid["fold_index"] == fold_index
        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]
        y_train = y.loc[train_mask]
        is_positive_train = is_positive.loc[train_mask]

        classifier = CatBoostClassifier(
            iterations=400,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=RANDOM_STATE,
            verbose=False,
        )
        classifier.fit(X_train, is_positive_train, cat_features=cat_indices)
        positive_prob = classifier.predict_proba(X_test)[:, 1]

        regressor = CatBoostRegressor(
            iterations=600,
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=RANDOM_STATE,
            verbose=False,
        )
        positive_train_mask = y_train > 0
        regressor.fit(X_train.loc[positive_train_mask], y_train.loc[positive_train_mask], cat_features=cat_indices)
        positive_pred = clip_bmf(regressor.predict(X_test))

        preds = clip_bmf(positive_prob * positive_pred)
        oof[test_mask.to_numpy()] = preds
        fold_rows.append(
            {
                "fold_index": int(fold_index),
                "r2": float(r2_score(y.loc[test_mask], preds)),
                "mae": float(mean_absolute_error(y.loc[test_mask], preds)),
                "rmse": float(np.sqrt(mean_squared_error(y.loc[test_mask], preds))),
            }
        )

    metrics = metric_payload(y, oof, fold_rows)
    metrics["zero_share"] = float((y == 0).mean())
    return ModelResult(
        model_key="two_stage_hurdle",
        model_label="Two-stage CatBoost hurdle model",
        task_scope="full grouped-CV BMF regression",
        metrics=metrics,
        architecture=[
            "Stage 1: CatBoostClassifier predicts `bound_mass_fraction > 0`.",
            "Stage 2: CatBoostRegressor predicts positive-only BMF magnitude on rows where `bound_mass_fraction > 0` in the training fold.",
            "Final regression prediction uses the hurdle expectation `P(BMF > 0 | x) * E[BMF | BMF > 0, x]`.",
        ],
        notes=[
            "This architecture explicitly matches the archive's exact-zero BMF mass.",
            "Using the expectation rather than a hard gate makes the point prediction smooth and regression-compatible.",
        ],
        status="ok",
    )


def evaluate_tabpfn(
    frame: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    feature_columns: list[str],
) -> ModelResult:
    architecture = [
        "Dense numeric benchmark using the existing leakage-free tabular preprocessor followed by TabPFNRegressor.",
        "TabPFN uses pretrained transformer inference rather than a standard boosted-tree fit.",
        "Configuration: `n_estimators=8`, `fit_mode='fit_preprocessors'`, `device='auto'`.",
    ]
    notes = [
        "The tabular foundation model is evaluated on the same grouped folds as the tree baselines.",
        "One-hot preprocessing is used here for compatibility with mixed categorical inputs.",
    ]
    try:
        from tabpfn import TabPFNRegressor

        valid = frame[frame[PRIMARY_TARGET].notna()].copy()
        valid = valid.merge(fold_assignments[["row_index", "fold_index"]], left_index=True, right_on="row_index", how="left")
        X = valid[feature_columns].copy()
        y = pd.to_numeric(valid[PRIMARY_TARGET], errors="coerce")
        preprocessor = build_preprocessor(X, scaled=False)

        oof = np.full(len(valid), np.nan)
        fold_rows: list[dict[str, Any]] = []
        for fold_index in sorted(valid["fold_index"].dropna().unique()):
            train_mask = valid["fold_index"] != fold_index
            test_mask = valid["fold_index"] == fold_index
            fitted_preprocessor = clone(preprocessor)
            X_train = fitted_preprocessor.fit_transform(X.loc[train_mask])
            X_test = fitted_preprocessor.transform(X.loc[test_mask])
            X_train_dense = X_train.toarray() if hasattr(X_train, "toarray") else np.asarray(X_train)
            X_test_dense = X_test.toarray() if hasattr(X_test, "toarray") else np.asarray(X_test)
            model = TabPFNRegressor(
                random_state=RANDOM_STATE,
                device="auto",
                fit_mode="fit_preprocessors",
                show_progress_bar=False,
                n_estimators=8,
            )
            model.fit(X_train_dense, y.loc[train_mask].to_numpy(dtype=float))
            preds = clip_bmf(model.predict(X_test_dense))
            oof[test_mask.to_numpy()] = preds
            fold_rows.append(
                {
                    "fold_index": int(fold_index),
                    "r2": float(r2_score(y.loc[test_mask], preds)),
                    "mae": float(mean_absolute_error(y.loc[test_mask], preds)),
                    "rmse": float(np.sqrt(mean_squared_error(y.loc[test_mask], preds))),
                }
            )

        metrics = metric_payload(y, oof, fold_rows)
        return ModelResult(
            model_key="tabpfn",
            model_label="TabPFN regressor",
            task_scope="full grouped-CV BMF regression",
            metrics=metrics,
            architecture=architecture,
            notes=notes,
            status="ok",
        )
    except Exception as exc:
        return ModelResult(
            model_key="tabpfn",
            model_label="TabPFN regressor",
            task_scope="full grouped-CV BMF regression",
            metrics={},
            architecture=architecture,
            notes=notes
            + [
                "Execution stopped before the first fold completed because TabPFN requires an external license token to fetch model weights on this machine.",
            ],
            status="failed",
            error=" | ".join(str(part) for part in [type(exc).__name__, exc] if part)[:2000],
        )


def evaluate_pysr(
    frame: pd.DataFrame,
    fold_assignments: pd.DataFrame,
) -> ModelResult:
    from pysr import PySRRegressor

    positive = frame[frame[PRIMARY_TARGET].gt(0)].copy()
    positive = positive.merge(fold_assignments[["row_index", "fold_index"]], left_index=True, right_on="row_index", how="left")
    feature_columns = [column for column in PYRSR_FEATURE_COLUMNS if column in positive.columns]
    valid_columns = [column for column in feature_columns if positive[column].notna().sum() > 0]
    positive = positive.dropna(subset=[PRIMARY_TARGET]).copy()
    positive = positive.loc[:, valid_columns + [PRIMARY_TARGET, "fold_index"]].copy()
    positive = positive.dropna().reset_index(drop=True)
    X = positive[valid_columns]
    y = pd.to_numeric(positive[PRIMARY_TARGET], errors="coerce")

    oof = np.full(len(positive), np.nan)
    fold_rows: list[dict[str, Any]] = []
    last_equation = ""
    equations_path = TABLES_DIR / "pysr_equations.csv"
    for fold_index in sorted(positive["fold_index"].dropna().unique()):
        train_mask = positive["fold_index"] != fold_index
        test_mask = positive["fold_index"] == fold_index
        model = PySRRegressor(
            niterations=40,
            populations=8,
            population_size=40,
            model_selection="best",
            maxsize=20,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["square", "sqrt", "log"],
            temp_equation_file=True,
            random_state=RANDOM_STATE,
            procs=0,
            progress=False,
        )
        model.fit(X.loc[train_mask].to_numpy(dtype=float), y.loc[train_mask].to_numpy(dtype=float))
        preds = clip_bmf(model.predict(X.loc[test_mask].to_numpy(dtype=float)))
        oof[test_mask.to_numpy()] = preds
        fold_rows.append(
            {
                "fold_index": int(fold_index),
                "r2": float(r2_score(y.loc[test_mask], preds)),
                "mae": float(mean_absolute_error(y.loc[test_mask], preds)),
                "rmse": float(np.sqrt(mean_squared_error(y.loc[test_mask], preds))),
            }
        )
        try:
            sym = model.sympy()
            last_equation = str(sym)
        except Exception:
            last_equation = ""

    metrics = metric_payload(y, oof, fold_rows)
    metrics["positive_only_rows"] = int(len(y))
    metrics["feature_columns"] = valid_columns
    if last_equation:
        metrics["best_equation"] = last_equation
    return ModelResult(
        model_key="pysr",
        model_label="PySR symbolic regression",
        task_scope="positive-only grouped-CV BMF regression",
        metrics=metrics,
        architecture=[
            "Positive-only symbolic regression companion on leakage-free continuous physics/setup features.",
            "Operator set: `+`, `-`, `*`, `/`, `square`, `sqrt`, `log`.",
            "Search budget: `niterations=40`, `populations=8`, `population_size=40`, `maxsize=20`.",
        ],
        notes=[
            "This report is intentionally positive-only because PySR is being used as a scientific equation search, not as the main zero-inflated predictor.",
            "The symbolic benchmark is therefore not directly comparable to the full-target hurdle and boosted-tree scores.",
        ],
        status="ok",
    )


def serialize_result(result: ModelResult) -> dict[str, Any]:
    payload = {
        "model_key": result.model_key,
        "model_label": result.model_label,
        "task_scope": result.task_scope,
        "status": result.status,
        "architecture": result.architecture,
        "notes": result.notes,
    }
    payload.update(result.metrics)
    if result.error:
        payload["error"] = result.error
    return payload


def render_metrics_lines(metrics: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key in ["rows", "positive_only_rows", "r2", "mae", "rmse", "fold_r2_mean", "fold_r2_std", "fold_mae_mean", "fold_mae_std", "fold_rmse_mean", "fold_rmse_std", "zero_share"]:
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, float):
            lines.append(f"- `{key}`: `{value:.6f}`")
        else:
            lines.append(f"- `{key}`: `{value}`")
    if "best_equation" in metrics:
        lines.append(f"- `best_equation`: `{metrics['best_equation']}`")
    if "feature_columns" in metrics:
        lines.append(f"- `feature_columns`: `{', '.join(metrics['feature_columns'])}`")
    return lines


def write_model_report(result: ModelResult) -> None:
    report_path = REPORTS_DIR / f"{result.model_key}.md"
    lines = [
        f"# {result.model_label}",
        "",
        f"- Status: `{result.status}`",
        f"- Task scope: `{result.task_scope}`",
        f"- Feature set: `{FEATURE_SET_NAME}` plus leakage-free physics features unless noted otherwise.",
        f"- Date run: `2026-07-29`",
        "",
        "## Architecture used",
        "",
    ]
    lines.extend([f"- {line}" for line in result.architecture])
    lines.extend(["", "## Results", ""])
    if result.status == "ok":
        lines.extend(render_metrics_lines(result.metrics))
        fold_metrics = result.metrics.get("fold_metrics", [])
        if fold_metrics:
            lines.extend(["", "## Fold metrics", ""])
            for row in fold_metrics:
                lines.append(
                    f"- fold `{row['fold_index']}`: `R²={row['r2']:.6f}`, `MAE={row['mae']:.6f}`, `RMSE={row['rmse']:.6f}`"
                )
    else:
        lines.append(f"- Error: `{result.error}`")
    lines.extend(["", "## Notes", ""])
    lines.extend([f"- {line}" for line in result.notes])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(results: list[ModelResult], feature_columns: list[str]) -> None:
    summary_rows = []
    for result in results:
        row = {
            "model_key": result.model_key,
            "model_label": result.model_label,
            "task_scope": result.task_scope,
            "status": result.status,
            "r2": result.metrics.get("r2"),
            "mae": result.metrics.get("mae"),
            "rmse": result.metrics.get("rmse"),
        }
        if result.error:
            row["error"] = result.error
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary["status_rank"] = summary["status"].map({"ok": 0, "failed": 1}).fillna(2)
    summary = summary.sort_values(["status_rank", "r2"], ascending=[True, False], na_position="last").drop(columns=["status_rank"])
    summary.to_csv(TABLES_DIR / "candidate_model_summary.csv", index=False)

    md_lines = [
        "# Model Optimization Candidate Summary",
        "",
        "- Date run: `2026-07-29`",
        f"- Dataset: `extraction-outputs/tables/bound_outcomes.csv`",
        f"- Full-target feature columns: `{', '.join(feature_columns)}`",
        "- Leakage guard: `largest_fragment_mass_fraction` was excluded because it is outcome-derived.",
        "",
        "## Ranked summary",
        "",
        "| Model | Scope | Status | R² | MAE | RMSE |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary.to_dict(orient="records"):
        r2 = "n/a" if row.get("r2") is None or (isinstance(row.get("r2"), float) and math.isnan(row["r2"])) else f"{row['r2']:.6f}"
        mae = "n/a" if row.get("mae") is None or (isinstance(row.get("mae"), float) and math.isnan(row["mae"])) else f"{row['mae']:.6f}"
        rmse = "n/a" if row.get("rmse") is None or (isinstance(row.get("rmse"), float) and math.isnan(row["rmse"])) else f"{row['rmse']:.6f}"
        md_lines.append(f"| {row['model_label']} | {row['task_scope']} | {row['status']} | {r2} | {mae} | {rmse} |")
    md_lines.extend(
        [
            "",
            "## Reports",
            "",
            "- `catboost.md`",
            "- `two_stage_hurdle.md`",
            "- `xgboost.md`",
            "- `lightgbm.md`",
            "- `tabpfn.md`",
            "- `pysr.md`",
        ]
    )
    (REPORTS_DIR / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def run_with_capture(label: str, func, *args) -> ModelResult:
    try:
        return func(*args)
    except Exception as exc:
        return ModelResult(
            model_key=label,
            model_label=label.replace("_", " ").title(),
            task_scope="run failed",
            metrics={},
            architecture=["Execution did not complete."],
            notes=["See error field for the captured exception."],
            status="failed",
            error=" | ".join(str(part) for part in [type(exc).__name__, exc] if part)[:2000],
        )


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    frame, fold_assignments, feature_columns = prepare_frame(args.dataset)

    results = [
        run_with_capture("catboost", evaluate_catboost, frame, fold_assignments, feature_columns),
        run_with_capture("two_stage_hurdle", evaluate_hurdle_model, frame, fold_assignments, feature_columns),
        run_with_capture(
            "xgboost",
            evaluate_preprocessed_model,
            frame,
            fold_assignments,
            feature_columns,
            "xgboost",
            "XGBoost regressor",
            lambda: XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
            ),
            [
                "Preprocessed dense/sparse tabular benchmark with XGBRegressor.",
                "Hyperparameters: `n_estimators=500`, `learning_rate=0.05`, `max_depth=4`, `subsample=0.9`, `colsample_bytree=0.9`.",
                "Uses the same leakage-free grouped-CV feature set as CatBoost.",
            ],
            ["XGBoost is included as a modern boosting alternative comparator."],
        ),
        run_with_capture(
            "lightgbm",
            evaluate_preprocessed_model,
            frame,
            fold_assignments,
            feature_columns,
            "lightgbm",
            "LightGBM regressor",
            lambda: LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=RANDOM_STATE,
                verbose=-1,
            ),
            [
                "Preprocessed tabular benchmark with LGBMRegressor.",
                "Hyperparameters: `n_estimators=500`, `learning_rate=0.05`, `num_leaves=31`, `subsample=0.9`, `colsample_bytree=0.9`.",
                "Uses the same leakage-free grouped-CV feature set as CatBoost.",
            ],
            ["LightGBM is included as an optional modern boosting comparator."],
        ),
        run_with_capture("tabpfn", evaluate_tabpfn, frame, fold_assignments, feature_columns),
        run_with_capture("pysr", evaluate_pysr, frame, fold_assignments),
    ]

    payload = [serialize_result(result) for result in results]
    (TABLES_DIR / "candidate_model_results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for result in results:
        write_model_report(result)
    write_summary(results, feature_columns)


if __name__ == "__main__":
    main()
