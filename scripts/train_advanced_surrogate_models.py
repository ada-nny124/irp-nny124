#!/usr/bin/env python3
"""Train advanced surrogate models for BMF screening."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from ngboost import NGBRegressor
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.othermod.betareg import BetaModel
from xgboost import XGBRegressor

from train_physics_structured_surrogate import (
    RANDOM_STATE,
    add_physics_features,
    build_group_folds,
    build_preprocessor,
    feature_columns_for_set,
    load_canonical_dataset,
)


OUTPUT_ROOT = Path("ml/model_optimization_candidates/advanced")
TABLES_DIR = OUTPUT_ROOT / "tables"
REPORTS_DIR = OUTPUT_ROOT / "reports"
PRIMARY_TARGET = "bound_mass_fraction"
SECONDARY_TARGETS = ["n_fragments", "largest_fragment_mass_kg", "largest_fragment_particle_count"]
FEATURE_SET_NAME = "with_fof_linking_length"
LEAKY_FEATURES = {"largest_fragment_mass_fraction"}


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
    parser.add_argument("--dataset", type=Path, default=Path("extraction_outputs/bound_outcomes.csv"))
    return parser.parse_args()


def ensure_output_dirs() -> None:
    for path in [OUTPUT_ROOT, TABLES_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def safe_feature_columns() -> list[str]:
    columns = feature_columns_for_set(FEATURE_SET_NAME, include_physics=True)
    return [column for column in columns if column not in LEAKY_FEATURES]


def categorical_columns(columns: list[str]) -> list[str]:
    return [column for column in columns if column in {"spin_axis", "special_case_code"}]


def clip_bmf(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def beta_shrink(y: np.ndarray) -> np.ndarray:
    # Smithson-Verkuilen style shrinkage keeps the positive BMF target inside (0, 1).
    y = np.asarray(y, dtype=float)
    n = len(y)
    return (y * (n - 1) + 0.5) / n


def fit_dense_preprocessor(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    preprocessor = build_preprocessor(X_train, scaled=True)
    fitted = clone(preprocessor)
    train = fitted.fit_transform(X_train)
    test = fitted.transform(X_test)
    train_dense = train.toarray() if hasattr(train, "toarray") else np.asarray(train)
    test_dense = test.toarray() if hasattr(test, "toarray") else np.asarray(test)
    return train_dense, test_dense


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


def prepare_frame(dataset_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    frame = add_physics_features(load_canonical_dataset(dataset_path))
    frame["spin_period_hr"] = pd.to_numeric(frame["spin_period_hr"], errors="coerce").fillna(0.0)
    frame["spin_frequency_hr_inv"] = pd.to_numeric(frame["spin_frequency_hr_inv"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    fold_assignments = build_group_folds(frame, frame["physical_file"].astype(str), TABLES_DIR / "fold_assignments.csv")
    return frame, fold_assignments, safe_feature_columns()


def evaluate_hurdle_beta(frame: pd.DataFrame, fold_assignments: pd.DataFrame, feature_columns: list[str]) -> ModelResult:
    valid = frame[frame[PRIMARY_TARGET].notna()].copy()
    valid = valid.merge(fold_assignments[["row_index", "fold_index"]], left_index=True, right_on="row_index", how="left")
    X = valid[feature_columns].copy()
    y = pd.to_numeric(valid[PRIMARY_TARGET], errors="coerce")
    cat_columns = categorical_columns(feature_columns)
    cat_indices = [X.columns.get_loc(column) for column in cat_columns]

    oof = np.full(len(valid), np.nan)
    fold_rows: list[dict[str, Any]] = []
    for fold in sorted(valid["fold_index"].dropna().unique()):
        train_mask = valid["fold_index"] != fold
        test_mask = valid["fold_index"] == fold
        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]
        y_train = y.loc[train_mask]

        gate = CatBoostClassifier(
            iterations=400,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            random_seed=RANDOM_STATE,
            verbose=False,
        )
        gate.fit(X_train, (y_train > 0).astype(int), cat_features=cat_indices)
        positive_prob = gate.predict_proba(X_test)[:, 1]

        positive_train = y_train > 0
        dense_train, dense_test = fit_dense_preprocessor(X_train.loc[positive_train], X_test)
        y_beta = beta_shrink(y_train.loc[positive_train].to_numpy(dtype=float))
        beta_model = BetaModel(y_beta, dense_train)
        beta_result = beta_model.fit(disp=False)
        positive_mean = clip_bmf(beta_result.predict(dense_test))

        preds = clip_bmf(positive_prob * positive_mean)
        oof[test_mask.to_numpy()] = preds
        fold_rows.append(
            {
                "fold_index": int(fold),
                "r2": float(r2_score(y.loc[test_mask], preds)),
                "mae": float(mean_absolute_error(y.loc[test_mask], preds)),
                "rmse": float(np.sqrt(mean_squared_error(y.loc[test_mask], preds))),
            }
        )

    metrics = metric_payload(y, oof, fold_rows)
    metrics["zero_share"] = float((y == 0).mean())
    return ModelResult(
        model_key="hurdle_beta",
        model_label="Hurdle beta surrogate",
        task_scope="full grouped-CV BMF regression",
        metrics=metrics,
        architecture=[
            "Stage 1: CatBoostClassifier predicts `bound_mass_fraction > 0`.",
            "Stage 2: statsmodels `BetaModel` predicts the positive-only BMF mean after shrinkage into `(0, 1)`.",
            "Final prediction uses the hurdle expectation `P(BMF > 0 | x) * E[BMF | BMF > 0, x]`.",
        ],
        notes=[
            "This is an approximate zero-inflated beta implementation rather than a single joint likelihood fit.",
            "It is designed to test whether a beta-shaped positive component helps beyond the tree-based hurdle winner.",
        ],
        status="ok",
    )


def evaluate_hurdle_ngboost(frame: pd.DataFrame, fold_assignments: pd.DataFrame, feature_columns: list[str]) -> ModelResult:
    valid = frame[frame[PRIMARY_TARGET].notna()].copy()
    valid = valid.merge(fold_assignments[["row_index", "fold_index"]], left_index=True, right_on="row_index", how="left")
    X = valid[feature_columns].copy()
    y = pd.to_numeric(valid[PRIMARY_TARGET], errors="coerce")
    cat_columns = categorical_columns(feature_columns)
    cat_indices = [X.columns.get_loc(column) for column in cat_columns]

    oof = np.full(len(valid), np.nan)
    fold_rows: list[dict[str, Any]] = []
    for fold in sorted(valid["fold_index"].dropna().unique()):
        train_mask = valid["fold_index"] != fold
        test_mask = valid["fold_index"] == fold
        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]
        y_train = y.loc[train_mask]

        gate = CatBoostClassifier(
            iterations=400,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            random_seed=RANDOM_STATE,
            verbose=False,
        )
        gate.fit(X_train, (y_train > 0).astype(int), cat_features=cat_indices)
        positive_prob = gate.predict_proba(X_test)[:, 1]

        positive_train = y_train > 0
        dense_train, dense_test = fit_dense_preprocessor(X_train.loc[positive_train], X_test)
        ngb = NGBRegressor(
            n_estimators=400,
            learning_rate=0.03,
            random_state=RANDOM_STATE,
            verbose=False,
        )
        ngb.fit(dense_train, y_train.loc[positive_train].to_numpy(dtype=float))
        positive_mean = clip_bmf(ngb.predict(dense_test))
        preds = clip_bmf(positive_prob * positive_mean)
        oof[test_mask.to_numpy()] = preds
        fold_rows.append(
            {
                "fold_index": int(fold),
                "r2": float(r2_score(y.loc[test_mask], preds)),
                "mae": float(mean_absolute_error(y.loc[test_mask], preds)),
                "rmse": float(np.sqrt(mean_squared_error(y.loc[test_mask], preds))),
            }
        )

    metrics = metric_payload(y, oof, fold_rows)
    return ModelResult(
        model_key="hurdle_ngboost",
        model_label="Hurdle NGBoost surrogate",
        task_scope="full grouped-CV BMF regression",
        metrics=metrics,
        architecture=[
            "Stage 1: CatBoostClassifier predicts zero versus positive BMF.",
            "Stage 2: NGBRegressor predicts positive-only BMF with probabilistic gradient boosting.",
            "Point prediction uses the hurdle expectation from the gate probability and NGBoost mean.",
        ],
        notes=[
            "This tests whether probabilistic boosting improves the positive BMF component enough to matter.",
            "The run is evaluated on point metrics, not full calibration metrics.",
        ],
        status="ok",
    )


def evaluate_multitask_gp(frame: pd.DataFrame, fold_assignments: pd.DataFrame, feature_columns: list[str]) -> ModelResult:
    needed = [PRIMARY_TARGET] + [target for target in SECONDARY_TARGETS if target in frame.columns]
    valid = frame.dropna(subset=needed).copy()
    valid = valid.merge(fold_assignments[["row_index", "fold_index"]], left_index=True, right_on="row_index", how="left")
    X = valid[feature_columns].copy()
    target_frame = valid[needed].copy()

    oof = np.full(len(valid), np.nan)
    fold_rows: list[dict[str, Any]] = []
    for fold in sorted(valid["fold_index"].dropna().unique()):
        train_mask = valid["fold_index"] != fold
        test_mask = valid["fold_index"] == fold
        dense_train, dense_test = fit_dense_preprocessor(X.loc[train_mask], X.loc[test_mask])
        y_train = target_frame.loc[train_mask].to_numpy(dtype=float)
        y_test = target_frame.loc[test_mask].to_numpy(dtype=float)

        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train)
        n_components = min(2, y_train_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        z_train = pca.fit_transform(y_train_scaled)

        gp_kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-4)
        latent_preds = []
        for component in range(n_components):
            gp = GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True, random_state=RANDOM_STATE, alpha=1e-6)
            gp.fit(dense_train, z_train[:, component])
            latent_preds.append(gp.predict(dense_test))
        z_pred = np.column_stack(latent_preds)
        y_pred_scaled = pca.inverse_transform(z_pred)
        y_pred = target_scaler.inverse_transform(y_pred_scaled)
        bmf_pred = clip_bmf(y_pred[:, 0])

        oof[test_mask.to_numpy()] = bmf_pred
        fold_rows.append(
            {
                "fold_index": int(fold),
                "r2": float(r2_score(y_test[:, 0], bmf_pred)),
                "mae": float(mean_absolute_error(y_test[:, 0], bmf_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test[:, 0], bmf_pred))),
            }
        )

    metrics = metric_payload(pd.to_numeric(valid[PRIMARY_TARGET], errors="coerce"), oof, fold_rows)
    metrics["joint_targets"] = needed
    return ModelResult(
        model_key="multitask_gp",
        model_label="Multi-output GP surrogate",
        task_scope="full grouped-CV BMF regression using joint targets",
        metrics=metrics,
        architecture=[
            "Shared multi-output surrogate built by standardizing `[BMF, n_fragments, largest_fragment_mass_kg, largest_fragment_particle_count]`.",
            "A PCA latent target basis is learned on the training fold, then separate GaussianProcessRegressor models fit the latent coordinates.",
            "The latent predictions are inverted back to the original target space and scored on BMF.",
        ],
        notes=[
            "This is a practical multi-output GP surrogate, not a full intrinsic-coregionalization implementation.",
            "It tests whether cross-target structure helps BMF screening through a shared latent Gaussian surrogate.",
        ],
        status="ok",
    )


def evaluate_mixture_of_experts(frame: pd.DataFrame, fold_assignments: pd.DataFrame, feature_columns: list[str]) -> ModelResult:
    valid = frame[frame[PRIMARY_TARGET].notna()].copy()
    valid = valid.merge(fold_assignments[["row_index", "fold_index"]], left_index=True, right_on="row_index", how="left")
    X = valid[feature_columns].copy()
    y = pd.to_numeric(valid[PRIMARY_TARGET], errors="coerce")
    cat_columns = categorical_columns(feature_columns)
    cat_indices = [X.columns.get_loc(column) for column in cat_columns]

    regime = pd.Series(np.where(y == 0, 0, np.where(y < 0.1, 1, 2)), index=y.index)
    oof = np.full(len(valid), np.nan)
    fold_rows: list[dict[str, Any]] = []
    for fold in sorted(valid["fold_index"].dropna().unique()):
        train_mask = valid["fold_index"] != fold
        test_mask = valid["fold_index"] == fold
        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]
        y_train = y.loc[train_mask]
        reg_train = regime.loc[train_mask]

        gate = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function="MultiClass",
            random_seed=RANDOM_STATE,
            verbose=False,
        )
        gate.fit(X_train, reg_train, cat_features=cat_indices)
        gate_prob = gate.predict_proba(X_test)

        expert_means = np.zeros((len(X_test), 3), dtype=float)
        expert_means[:, 0] = 0.0
        for regime_id in [1, 2]:
            regime_mask = reg_train == regime_id
            expert = CatBoostRegressor(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                loss_function="RMSE",
                random_seed=RANDOM_STATE,
                verbose=False,
            )
            expert.fit(X_train.loc[regime_mask], y_train.loc[regime_mask], cat_features=cat_indices)
            expert_means[:, regime_id] = clip_bmf(expert.predict(X_test))

        preds = clip_bmf(np.sum(gate_prob * expert_means, axis=1))
        oof[test_mask.to_numpy()] = preds
        fold_rows.append(
            {
                "fold_index": int(fold),
                "r2": float(r2_score(y.loc[test_mask], preds)),
                "mae": float(mean_absolute_error(y.loc[test_mask], preds)),
                "rmse": float(np.sqrt(mean_squared_error(y.loc[test_mask], preds))),
            }
        )

    metrics = metric_payload(y, oof, fold_rows)
    return ModelResult(
        model_key="mixture_of_experts",
        model_label="Regime-aware mixture of experts",
        task_scope="full grouped-CV BMF regression",
        metrics=metrics,
        architecture=[
            "A CatBoost multiclass gating model routes rows into three BMF regimes: zero, low-positive `(0, 0.1)`, and high-positive `>= 0.1`.",
            "Expert 0 is a fixed zero expert; experts 1 and 2 are CatBoost regressors trained on their regime-specific subsets.",
            "The final prediction is the gate-weighted expectation across expert outputs.",
        ],
        notes=[
            "This is a regime-aware MoE tailored to the screening threshold already used in the repo.",
            "It tests whether local specialists beat a single global regressor on the archive's regime structure.",
        ],
        status="ok",
    )


def run_with_capture(model_key: str, func, *args) -> ModelResult:
    try:
        return func(*args)
    except Exception as exc:
        return ModelResult(
            model_key=model_key,
            model_label=model_key.replace("_", " ").title(),
            task_scope="run failed",
            metrics={},
            architecture=["Execution did not complete."],
            notes=["See the error field for the captured exception."],
            status="failed",
            error=f"{type(exc).__name__} | {exc}",
        )


def write_report(result: ModelResult) -> None:
    path = REPORTS_DIR / f"{result.model_key}.md"
    lines = [
        f"# {result.model_label}",
        "",
        f"- Status: `{result.status}`",
        f"- Task scope: `{result.task_scope}`",
        f"- Date run: `2026-07-29`",
        "",
        "## Architecture used",
        "",
    ]
    lines.extend([f"- {line}" for line in result.architecture])
    lines.extend(["", "## Results", ""])
    if result.status == "ok":
        for key in ["rows", "r2", "mae", "rmse", "fold_r2_mean", "fold_r2_std", "fold_mae_mean", "fold_mae_std", "fold_rmse_mean", "fold_rmse_std", "zero_share"]:
            if key in result.metrics:
                value = result.metrics[key]
                lines.append(f"- `{key}`: `{value:.6f}`" if isinstance(value, float) else f"- `{key}`: `{value}`")
        if "joint_targets" in result.metrics:
            lines.append(f"- `joint_targets`: `{', '.join(result.metrics['joint_targets'])}`")
        lines.extend(["", "## Fold metrics", ""])
        for row in result.metrics.get("fold_metrics", []):
            lines.append(f"- fold `{row['fold_index']}`: `R²={row['r2']:.6f}`, `MAE={row['mae']:.6f}`, `RMSE={row['rmse']:.6f}`")
    else:
        lines.append(f"- Error: `{result.error}`")
    lines.extend(["", "## Notes", ""])
    lines.extend([f"- {line}" for line in result.notes])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(results: list[ModelResult]) -> None:
    rows = []
    for result in results:
        row = {
            "model_key": result.model_key,
            "model_label": result.model_label,
            "task_scope": result.task_scope,
            "status": result.status,
            "r2": result.metrics.get("r2"),
            "mae": result.metrics.get("mae"),
            "rmse": result.metrics.get("rmse"),
            "error": result.error,
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df["status_rank"] = df["status"].map({"ok": 0, "failed": 1}).fillna(2)
    df = df.sort_values(["status_rank", "r2"], ascending=[True, False], na_position="last").drop(columns=["status_rank"])
    df.to_csv(TABLES_DIR / "advanced_model_summary.csv", index=False)

    lines = [
        "# Advanced Model Summary",
        "",
        "- Date run: `2026-07-29`",
        "- Dataset: `extraction_outputs/bound_outcomes.csv`",
        "- Scope: advanced zero-inflated, probabilistic, GP, and expert-routing surrogates",
        "",
        "| Model | Scope | Status | R² | MAE | RMSE |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in df.to_dict(orient="records"):
        r2 = "n/a" if pd.isna(row["r2"]) else f"{row['r2']:.6f}"
        mae = "n/a" if pd.isna(row["mae"]) else f"{row['mae']:.6f}"
        rmse = "n/a" if pd.isna(row["rmse"]) else f"{row['rmse']:.6f}"
        lines.append(f"| {row['model_label']} | {row['task_scope']} | {row['status']} | {r2} | {mae} | {rmse} |")
    (REPORTS_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    frame, fold_assignments, feature_columns = prepare_frame(args.dataset)
    results = [
        run_with_capture("hurdle_beta", evaluate_hurdle_beta, frame, fold_assignments, feature_columns),
        run_with_capture("hurdle_ngboost", evaluate_hurdle_ngboost, frame, fold_assignments, feature_columns),
        run_with_capture("multitask_gp", evaluate_multitask_gp, frame, fold_assignments, feature_columns),
        run_with_capture("mixture_of_experts", evaluate_mixture_of_experts, frame, fold_assignments, feature_columns),
    ]
    (TABLES_DIR / "advanced_model_results.json").write_text(
        json.dumps(
            [
                {
                    "model_key": result.model_key,
                    "model_label": result.model_label,
                    "task_scope": result.task_scope,
                    "status": result.status,
                    "metrics": result.metrics,
                    "architecture": result.architecture,
                    "notes": result.notes,
                    "error": result.error,
                }
                for result in results
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for result in results:
        write_report(result)
    write_summary(results)


if __name__ == "__main__":
    main()
