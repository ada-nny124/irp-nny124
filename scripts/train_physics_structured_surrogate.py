#!/usr/bin/env python3
"""Train a physics-structured tabular surrogate for SPH-derived outcomes."""

from __future__ import annotations

import argparse
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
N_SPLITS = 5
MARS_MU_KM3_S2 = 4.282837e4
MARS_RADIUS_KM = 3389.5
OUTPUT_ROOT = Path("ml/physics_structured_surrogate")
TABLES_DIR = OUTPUT_ROOT / "tables"
PLOTS_DIR = OUTPUT_ROOT / "plots"
MODELS_DIR = OUTPUT_ROOT / "models"
FOLD_ASSIGNMENTS_PATH = TABLES_DIR / "fold_assignments.csv"
PROMOTED_MODEL_INFO_PATH = TABLES_DIR / "promoted_model_info.json"
PRIMARY_TARGET = "bound_mass_fraction"
SECONDARY_TARGETS = ["n_fragments", "largest_fragment_mass_kg", "largest_fragment_particle_count"]
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
PHYSICS_FEATURE_COLUMNS = [
    "encounter_eccentricity_proxy",
    "v_inf_squared",
    "periapsis_inverse",
    "angular_momentum_proxy",
    "spin_frequency_hr_inv",
    "has_spin",
    "particle_mass_proxy",
    "mass_resolution_interaction",
    "largest_fragment_mass_fraction",
]
FEATURE_SET_COLUMNS = {
    "with_fof_linking_length": BASE_FEATURE_COLUMNS,
    "without_fof_linking_length": [column for column in BASE_FEATURE_COLUMNS if column != "fof_linking_length"],
}
FILENAME_RE = re.compile(
    r"^(?P<prefix>Ma_xp)_(?P<mass>A\d{4}(?:c30)?)(?:_(?P<spin>s\d{3}[A-Za-z]*))?"
    r"_n(?P<resolution>\d+)_r(?P<periapsis>\d+)_v(?P<velocity>\d+)"
    r"_(?P<timestep>\d+)"
    r"_fof_(?P<linking_length>[0-9.]+)_"
    r"(?P<chunk>\d+)\.hdf5$"
)


@dataclass(frozen=True)
class StagePaths:
    root: Path
    tables: Path
    plots: Path
    models: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("extraction_outputs/bound_outcomes.csv"),
        help="Bound outcome table used as the canonical surrogate dataset.",
    )
    parser.add_argument(
        "--stage",
        choices=[
            "baseline",
            "tuning",
            "fof_compare",
            "feature_ablation",
            "target_transforms",
            "trust",
            "diagnostics",
            "all",
        ],
        default="all",
        help="Pipeline stage to run.",
    )
    return parser.parse_args()


def ensure_output_dirs() -> StagePaths:
    for path in [OUTPUT_ROOT, TABLES_DIR, PLOTS_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    return StagePaths(root=OUTPUT_ROOT, tables=TABLES_DIR, plots=PLOTS_DIR, models=MODELS_DIR)


def build_group_folds(frame: pd.DataFrame, groups: pd.Series, output_path: Path = FOLD_ASSIGNMENTS_PATH) -> pd.DataFrame:
    splitter = GroupKFold(n_splits=min(N_SPLITS, groups.nunique()))
    fold_assignments = np.full(len(frame), -1, dtype=int)
    for fold_index, (_, test_idx) in enumerate(splitter.split(frame, groups=groups)):
        fold_assignments[test_idx] = fold_index
    fold_frame = frame.loc[:, ["physical_file"]].copy()
    fold_frame["row_index"] = frame.index.to_numpy()
    fold_frame["fold_index"] = fold_assignments
    fold_frame.to_csv(output_path, index=False)
    return fold_frame


def write_promoted_model_info(payload: dict[str, object], output_path: Path = PROMOTED_MODEL_INFO_PATH) -> None:
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_simulation_filename(filename: str) -> dict[str, object]:
    match = FILENAME_RE.match(filename)
    if not match:
        raise ValueError(f"Unrecognized FoF filename pattern: {filename}")

    mass_code = match.group("mass")
    spin_code = match.group("spin") or ""
    spin_axis = spin_code[4:] if len(spin_code) > 4 else ""
    spin_value = spin_code[1:4] if spin_code else ""
    resolution_value = int(match.group("resolution"))
    periapsis_value = int(match.group("periapsis"))
    velocity_value = int(match.group("velocity"))
    linking_length = float(match.group("linking_length"))

    return {
        "mass_code": mass_code,
        "mass_value": int(mass_code[1:5]),
        "special_case_code": "c30" if mass_code.endswith("c30") else "",
        "spin_code": spin_code,
        "spin_value": int(spin_value) if spin_value else np.nan,
        "spin_axis": spin_axis or "none",
        "has_explicit_spin": bool(spin_code),
        "resolution_code": f"n{resolution_value}",
        "resolution_value": resolution_value,
        "periapsis_code": f"r{periapsis_value}",
        "periapsis_value": periapsis_value,
        "velocity_code": f"v{velocity_value:02d}",
        "velocity_value": velocity_value,
        "timestep": int(match.group("timestep")),
        "fof_linking_length": linking_length,
        "chunk_index": int(match.group("chunk")),
    }


def build_canonical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    parsed = frame["fof_file"].map(parse_simulation_filename).apply(pd.Series)
    canonical = frame.copy()
    for column in parsed.columns:
        if column not in canonical.columns:
            canonical[column] = parsed[column]

    canonical["mass_log10_kg"] = pd.to_numeric(canonical["mass_value"], errors="coerce") / 100.0
    canonical["target_mass_kg"] = 10 ** canonical["mass_log10_kg"]
    canonical["particle_log10"] = np.log10(pd.to_numeric(canonical["resolution_value"], errors="coerce"))
    canonical["periapsis_Rm"] = pd.to_numeric(canonical["periapsis_value"], errors="coerce") / 10.0
    canonical["v_inf_kms"] = pd.to_numeric(canonical["velocity_value"], errors="coerce") / 10.0
    canonical["spin_period_hr"] = pd.to_numeric(canonical["spin_value"], errors="coerce") / 10.0
    canonical["spin_axis"] = canonical["spin_axis"].fillna("none").replace("", "none")
    canonical["special_case_code"] = canonical["special_case_code"].fillna("").replace("", "none")
    canonical["has_explicit_spin"] = canonical["has_explicit_spin"].fillna(False).astype(bool)
    canonical["has_spin"] = canonical["has_explicit_spin"].astype(int)
    canonical["bound_mass_fraction_ge_0_1"] = pd.to_numeric(canonical["bound_mass_fraction"], errors="coerce") >= 0.1
    largest_bound = pd.to_numeric(canonical["largest_bound_fragment_mass_kg"], errors="coerce")
    largest_unbound = pd.to_numeric(canonical["largest_unbound_fragment_mass_kg"], errors="coerce")
    canonical["largest_fragment_mass_kg"] = np.maximum(largest_bound.fillna(-np.inf), largest_unbound.fillna(-np.inf))
    canonical["largest_fragment_mass_kg"] = canonical["largest_fragment_mass_kg"].replace(-np.inf, np.nan)
    canonical["largest_fragment_mass_fraction"] = canonical["largest_fragment_mass_kg"] / canonical["target_mass_kg"]
    return canonical


def load_canonical_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    return build_canonical_frame(frame)


def eccentricity_proxy(periapsis_rm_values: pd.Series, velocity_kms_values: pd.Series) -> pd.Series:
    periapsis_km = periapsis_rm_values * MARS_RADIUS_KM
    with np.errstate(divide="ignore", invalid="ignore"):
        proxy = 1.0 + (periapsis_km * np.square(velocity_kms_values)) / MARS_MU_KM3_S2
    return pd.Series(proxy, index=periapsis_rm_values.index).replace([np.inf, -np.inf], np.nan)


def add_physics_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["encounter_eccentricity_proxy"] = eccentricity_proxy(enriched["periapsis_Rm"], enriched["v_inf_kms"])
    enriched["v_inf_squared"] = np.square(enriched["v_inf_kms"])
    with np.errstate(divide="ignore", invalid="ignore"):
        enriched["periapsis_inverse"] = 1.0 / enriched["periapsis_Rm"]
        enriched["spin_frequency_hr_inv"] = 1.0 / enriched["spin_period_hr"]
    enriched["angular_momentum_proxy"] = enriched["periapsis_Rm"] * enriched["v_inf_kms"]
    enriched["particle_mass_proxy"] = enriched["target_mass_kg"] / pd.to_numeric(enriched["resolution_value"], errors="coerce")
    enriched["mass_resolution_interaction"] = enriched["mass_log10_kg"] - enriched["particle_log10"]
    return enriched


def feature_columns_for_set(feature_set_name: str, include_physics: bool) -> list[str]:
    columns = FEATURE_SET_COLUMNS[feature_set_name].copy()
    if include_physics:
        columns.extend([column for column in PHYSICS_FEATURE_COLUMNS if column not in columns])
    return columns


def build_preprocessor(X: pd.DataFrame, scaled: bool) -> ColumnTransformer:
    categorical_features = [column for column in ["spin_axis", "special_case_code"] if column in X.columns]
    numeric_features = [column for column in X.columns if column not in categorical_features]
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scaled:
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


def baseline_regression_models(X: pd.DataFrame) -> dict[str, Pipeline]:
    return {
        "ridge": Pipeline([("preprocessor", build_preprocessor(X, scaled=True)), ("model", Ridge(alpha=1.0))]),
        "random_forest": Pipeline(
            [
                ("preprocessor", build_preprocessor(X, scaled=False)),
                ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)),
            ]
        ),
        "gradient_boosting": Pipeline(
            [("preprocessor", build_preprocessor(X, scaled=False)), ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))]
        ),
    }


def evaluate_grouped_oof_models(
    frame: pd.DataFrame,
    target: str,
    feature_columns: list[str],
    fold_assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = frame[frame[target].notna()].copy()
    valid = valid.merge(fold_assignments[["row_index", "fold_index"]], left_index=True, right_on="row_index", how="left")
    X = valid[feature_columns].copy()
    y = pd.to_numeric(valid[target], errors="coerce")
    models = baseline_regression_models(X)
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for model_name, pipeline in models.items():
        oof = np.full(len(valid), np.nan)
        fold_metrics: list[dict[str, object]] = []
        for fold_index in sorted(valid["fold_index"].dropna().unique()):
            train_mask = valid["fold_index"] != fold_index
            test_mask = valid["fold_index"] == fold_index
            fitted = clone(pipeline)
            fitted.fit(X.loc[train_mask], y.loc[train_mask])
            preds = fitted.predict(X.loc[test_mask])
            oof[test_mask.to_numpy()] = preds
            fold_metrics.append(
                {
                    "fold_index": int(fold_index),
                    "r2": r2_score(y.loc[test_mask], preds),
                    "mae": mean_absolute_error(y.loc[test_mask], preds),
                    "rmse": float(np.sqrt(mean_squared_error(y.loc[test_mask], preds))),
                }
            )
        fold_frame = pd.DataFrame(fold_metrics)
        metric_rows.append(
            {
                "target": target,
                "model": model_name,
                "feature_set": "with_fof_linking_length",
                "rows": len(valid),
                "r2": r2_score(y, oof),
                "mae": mean_absolute_error(y, oof),
                "rmse": float(np.sqrt(mean_squared_error(y, oof))),
                "fold_r2_mean": fold_frame["r2"].mean(),
                "fold_r2_std": fold_frame["r2"].std(ddof=0),
                "fold_mae_mean": fold_frame["mae"].mean(),
                "fold_mae_std": fold_frame["mae"].std(ddof=0),
                "fold_rmse_mean": fold_frame["rmse"].mean(),
                "fold_rmse_std": fold_frame["rmse"].std(ddof=0),
            }
        )
        pred_frame = valid.copy()
        pred_frame["target"] = target
        pred_frame["model"] = model_name
        pred_frame["feature_set"] = "with_fof_linking_length"
        pred_frame["predicted"] = oof
        pred_frame["residual"] = pred_frame[target] - pred_frame["predicted"]
        prediction_rows.append(pred_frame)
    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True)


def run_baseline_stage(dataset_path: Path) -> dict[str, pd.DataFrame]:
    ensure_output_dirs()
    frame = load_canonical_dataset(dataset_path)
    fold_assignments = build_group_folds(frame, frame["physical_file"].astype(str))
    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    target_columns = [PRIMARY_TARGET] + [column for column in SECONDARY_TARGETS if column in frame.columns]
    for target in target_columns:
        metrics, predictions = evaluate_grouped_oof_models(frame, target, BASE_FEATURE_COLUMNS, fold_assignments)
        metric_frames.append(metrics)
        prediction_frames.append(predictions)
    baseline_metrics = pd.concat(metric_frames, ignore_index=True).sort_values(["target", "model"]).reset_index(drop=True)
    baseline_predictions = pd.concat(prediction_frames, ignore_index=True)
    baseline_metrics.to_csv(TABLES_DIR / "baseline_metrics.csv", index=False)
    baseline_predictions.to_csv(TABLES_DIR / "baseline_oof_predictions.csv", index=False)
    return {"frame": frame, "fold_assignments": fold_assignments, "baseline_metrics": baseline_metrics, "baseline_predictions": baseline_predictions}


def random_forest_search_space() -> list[dict[str, object]]:
    return [
        {
            "model": "random_forest",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
        }
        for n_estimators, max_depth, min_samples_leaf, max_features in itertools.product(
            [300, 500, 800],
            [None, 6, 10, 16],
            [1, 2, 4, 8],
            ["sqrt", 0.5, 0.8, 1.0],
        )
    ]


def gradient_boosting_search_space() -> list[dict[str, object]]:
    return [
        {
            "model": "gradient_boosting",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "min_samples_leaf": min_samples_leaf,
        }
        for n_estimators, learning_rate, max_depth, subsample, min_samples_leaf in itertools.product(
            [100, 200, 400],
            [0.03, 0.05, 0.1],
            [2, 3, 4],
            [0.7, 0.9, 1.0],
            [1, 2, 4],
        )
    ]


def evaluate_tuning_candidates(
    frame: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    feature_columns: list[str],
    candidates: list[dict[str, object]],
    feature_set_name: str,
) -> pd.DataFrame:
    valid = frame[frame[PRIMARY_TARGET].notna()].copy()
    valid = valid.merge(fold_assignments[["row_index", "fold_index"]], left_index=True, right_on="row_index", how="left")
    X = valid[feature_columns].copy()
    y = pd.to_numeric(valid[PRIMARY_TARGET], errors="coerce")
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        model_name = str(candidate["model"])
        if model_name == "random_forest":
            pipeline = Pipeline(
                [
                    ("preprocessor", build_preprocessor(X, scaled=False)),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=int(candidate["n_estimators"]),
                            max_depth=candidate["max_depth"],
                            min_samples_leaf=int(candidate["min_samples_leaf"]),
                            max_features=candidate["max_features"],
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
        else:
            pipeline = Pipeline(
                [
                    ("preprocessor", build_preprocessor(X, scaled=False)),
                    (
                        "model",
                        GradientBoostingRegressor(
                            n_estimators=int(candidate["n_estimators"]),
                            learning_rate=float(candidate["learning_rate"]),
                            max_depth=int(candidate["max_depth"]),
                            subsample=float(candidate["subsample"]),
                            min_samples_leaf=int(candidate["min_samples_leaf"]),
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            )
        oof = np.full(len(valid), np.nan)
        fold_scores: list[dict[str, float]] = []
        for fold_index in sorted(valid["fold_index"].dropna().unique()):
            train_mask = valid["fold_index"] != fold_index
            test_mask = valid["fold_index"] == fold_index
            fitted = clone(pipeline)
            fitted.fit(X.loc[train_mask], y.loc[train_mask])
            preds = fitted.predict(X.loc[test_mask])
            oof[test_mask.to_numpy()] = preds
            fold_scores.append(
                {
                    "r2": r2_score(y.loc[test_mask], preds),
                    "mae": mean_absolute_error(y.loc[test_mask], preds),
                    "rmse": float(np.sqrt(mean_squared_error(y.loc[test_mask], preds))),
                }
            )
        fold_frame = pd.DataFrame(fold_scores)
        rows.append(
            {
                "target": PRIMARY_TARGET,
                "feature_set": feature_set_name,
                "model": model_name,
                "params_json": json.dumps(candidate, sort_keys=True),
                "r2": r2_score(y, oof),
                "mae": mean_absolute_error(y, oof),
                "rmse": float(np.sqrt(mean_squared_error(y, oof))),
                "fold_r2_mean": fold_frame["r2"].mean(),
                "fold_r2_std": fold_frame["r2"].std(ddof=0),
                "fold_mae_mean": fold_frame["mae"].mean(),
                "fold_mae_std": fold_frame["mae"].std(ddof=0),
                "fold_rmse_mean": fold_frame["rmse"].mean(),
                "fold_rmse_std": fold_frame["rmse"].std(ddof=0),
            }
        )
    return pd.DataFrame(rows)


def build_regression_pipeline(X: pd.DataFrame, model_name: str, params: dict[str, object] | None = None) -> Pipeline:
    params = params or {}
    if model_name == "ridge":
        return Pipeline([("preprocessor", build_preprocessor(X, scaled=True)), ("model", Ridge(alpha=float(params.get("alpha", 1.0))))])
    if model_name == "random_forest":
        return Pipeline(
            [
                ("preprocessor", build_preprocessor(X, scaled=False)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=int(params.get("n_estimators", 300)),
                        max_depth=params.get("max_depth"),
                        min_samples_leaf=int(params.get("min_samples_leaf", 2)),
                        max_features=params.get("max_features", "sqrt"),
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(X, scaled=False)),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=int(params.get("n_estimators", 100)),
                    learning_rate=float(params.get("learning_rate", 0.1)),
                    max_depth=int(params.get("max_depth", 3)),
                    subsample=float(params.get("subsample", 1.0)),
                    min_samples_leaf=int(params.get("min_samples_leaf", 1)),
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_model_config_oof(
    frame: pd.DataFrame,
    target: str,
    feature_columns: list[str],
    fold_assignments: pd.DataFrame,
    model_name: str,
    params: dict[str, object] | None,
    transform_name: str = "raw",
) -> tuple[pd.DataFrame, pd.DataFrame, Pipeline]:
    valid = frame[frame[target].notna()].copy()
    valid = valid.merge(fold_assignments[["row_index", "fold_index"]], left_index=True, right_on="row_index", how="left")
    X = valid[feature_columns].copy()
    y = pd.to_numeric(valid[target], errors="coerce")
    pipeline = build_regression_pipeline(X, model_name, params)
    oof = np.full(len(valid), np.nan)
    fold_metrics: list[dict[str, object]] = []
    for fold_index in sorted(valid["fold_index"].dropna().unique()):
        train_mask = valid["fold_index"] != fold_index
        test_mask = valid["fold_index"] == fold_index
        fitted = clone(pipeline)
        fitted.fit(X.loc[train_mask], y.loc[train_mask])
        preds = fitted.predict(X.loc[test_mask])
        oof[test_mask.to_numpy()] = preds
        fold_metrics.append(
            {
                "fold_index": int(fold_index),
                "r2": r2_score(y.loc[test_mask], preds),
                "mae": mean_absolute_error(y.loc[test_mask], preds),
                "rmse": float(np.sqrt(mean_squared_error(y.loc[test_mask], preds))),
            }
        )
    fitted_full = clone(pipeline).fit(X, y)
    fold_frame = pd.DataFrame(fold_metrics)
    metrics = pd.DataFrame(
        [
            {
                "target": target,
                "model": model_name,
                "transform": transform_name,
                "rows": len(valid),
                "r2": r2_score(y, oof),
                "mae": mean_absolute_error(y, oof),
                "rmse": float(np.sqrt(mean_squared_error(y, oof))),
                "fold_r2_mean": fold_frame["r2"].mean(),
                "fold_r2_std": fold_frame["r2"].std(ddof=0),
                "fold_mae_mean": fold_frame["mae"].mean(),
                "fold_mae_std": fold_frame["mae"].std(ddof=0),
                "fold_rmse_mean": fold_frame["rmse"].mean(),
                "fold_rmse_std": fold_frame["rmse"].std(ddof=0),
            }
        ]
    )
    predictions = valid.copy()
    predictions["target"] = target
    predictions["model"] = model_name
    predictions["transform"] = transform_name
    predictions["predicted"] = oof
    predictions["residual"] = predictions[target] - predictions["predicted"]
    return metrics, predictions, fitted_full


def summarize_tuning_promotion(
    baseline_metrics: pd.DataFrame,
    tuning_results: pd.DataFrame,
    feature_set_name: str,
) -> pd.DataFrame:
    baseline_row = baseline_metrics[
        (baseline_metrics["target"] == PRIMARY_TARGET)
        & (baseline_metrics["model"] == "random_forest")
        & (baseline_metrics["feature_set"] == feature_set_name)
    ].iloc[0]
    best_row = tuning_results.sort_values(["r2", "mae"], ascending=[False, True]).iloc[0]
    r2_gain = float(best_row["r2"] - baseline_row["r2"])
    mae_gain = float(baseline_row["mae"] - best_row["mae"])
    stability_worse = float(best_row["fold_r2_std"]) > float(baseline_row["fold_r2_std"]) + 0.01
    promote = (r2_gain >= 0.02 or mae_gain > 0.001) and not stability_worse
    promoted_label = f"tuned {'RF' if best_row['model'] == 'random_forest' else 'GB'}" if promote else "baseline RF"
    reason = "simplicity preferred" if not promote else "tuning materially improved BMF without worse fold stability"
    return pd.DataFrame(
        [
            {
                "feature_set": feature_set_name,
                "baseline_model": "baseline RF",
                "candidate_model": best_row["model"],
                "candidate_params_json": best_row["params_json"],
                "baseline_r2": baseline_row["r2"],
                "candidate_r2": best_row["r2"],
                "r2_gain": r2_gain,
                "baseline_mae": baseline_row["mae"],
                "candidate_mae": best_row["mae"],
                "mae_reduction": mae_gain,
                "baseline_fold_r2_std": baseline_row["fold_r2_std"],
                "candidate_fold_r2_std": best_row["fold_r2_std"],
                "promoted_model": promoted_label,
                "promotion_reason": reason,
                "promote_tuned_model": promote,
            }
        ]
    )


def run_tuning_stage(dataset_path: Path) -> dict[str, pd.DataFrame]:
    ensure_output_dirs()
    frame = load_canonical_dataset(dataset_path)
    fold_assignments = pd.read_csv(FOLD_ASSIGNMENTS_PATH) if FOLD_ASSIGNMENTS_PATH.exists() else build_group_folds(frame, frame["physical_file"].astype(str))
    baseline_metrics_path = TABLES_DIR / "baseline_metrics.csv"
    baseline_metrics = pd.read_csv(baseline_metrics_path) if baseline_metrics_path.exists() else run_baseline_stage(dataset_path)["baseline_metrics"]
    feature_set_name = "with_fof_linking_length"
    search_results = pd.concat(
        [
            evaluate_tuning_candidates(frame, fold_assignments, FEATURE_SET_COLUMNS[feature_set_name], random_forest_search_space(), feature_set_name),
            evaluate_tuning_candidates(frame, fold_assignments, FEATURE_SET_COLUMNS[feature_set_name], gradient_boosting_search_space(), feature_set_name),
        ],
        ignore_index=True,
    )
    tuned_metrics = search_results.sort_values(["r2", "mae"], ascending=[False, True]).groupby(["target", "feature_set", "model"], as_index=False).head(1)
    promotion_summary = summarize_tuning_promotion(baseline_metrics, search_results, feature_set_name)
    search_results.to_csv(TABLES_DIR / "tuning_search_results.csv", index=False)
    tuned_metrics.to_csv(TABLES_DIR / "tuned_model_metrics.csv", index=False)
    promotion_summary.to_csv(TABLES_DIR / "promotion_summary.csv", index=False)
    return {"tuning_search_results": search_results, "tuned_model_metrics": tuned_metrics, "promotion_summary": promotion_summary}


def main() -> None:
    args = parse_args()
    if args.stage not in {"baseline", "all"}:
        raise NotImplementedError(f"Stage not implemented yet: {args.stage}")
    run_baseline_stage(args.dataset)


if __name__ == "__main__":
    main()
