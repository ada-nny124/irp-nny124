#!/usr/bin/env python3
"""Train a physics-structured tabular surrogate for SPH-derived outcomes."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
N_SPLITS = 5
MARS_MU_KM3_S2 = 4.282837e4
MARS_RADIUS_KM = 3389.5
MARS_DENSITY_KG_M3 = 3933.5
ASTEROID_BULK_DENSITY_KG_M3 = 2700.0
PROXIMITY_DISTANCE_RM = 2.0
FLUID_ROCHE_FACTOR = 2.44
OUTPUT_ROOT = Path("ml/physics_structured_surrogate")
TABLES_DIR = OUTPUT_ROOT / "tables"
PLOTS_DIR = OUTPUT_ROOT / "plots"
MODELS_DIR = OUTPUT_ROOT / "models"
FOLD_ASSIGNMENTS_PATH = TABLES_DIR / "fold_assignments.csv"
PROMOTED_MODEL_INFO_PATH = TABLES_DIR / "promoted_model_info.json"
PRIMARY_TARGET = "bound_mass_fraction"
SECONDARY_TARGETS = ["n_fragments", "largest_fragment_mass_kg", "largest_fragment_particle_count"]
REPRESENTATIVE_SLICE = {
    "mass_code": "A2000",
    "resolution_code": "n65",
    "velocity_code": "v00",
    "spin_code": "s030z",
    "timestep": 90000,
    "fof_linking_length": 0.004,
}
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
    "asteroid_radius_km",
    "time_within_2_mars_radii_hr",
    "time_within_tidal_disruption_hr",
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
            "tune",
            "fof_compare",
            "target_transforms",
            "trust",
            "diagnostics",
            "package",
            "all",
        ],
        default="all",
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--fast",
        "--compact-search",
        dest="compact_search",
        action="store_true",
        help="Use the compact hyperparameter grid for the first BMF tuning pass.",
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


def asteroid_radius_km(target_mass_kg_values: pd.Series, density_kg_m3: float = ASTEROID_BULK_DENSITY_KG_M3) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        radius_m = np.cbrt((3.0 * target_mass_kg_values) / (4.0 * np.pi * density_kg_m3))
    return pd.Series(radius_m / 1000.0, index=target_mass_kg_values.index).replace([np.inf, -np.inf], np.nan)


def tidal_disruption_radius_rm(
    asteroid_density_kg_m3: float = ASTEROID_BULK_DENSITY_KG_M3,
    mars_density_kg_m3: float = MARS_DENSITY_KG_M3,
) -> float:
    return FLUID_ROCHE_FACTOR * (mars_density_kg_m3 / asteroid_density_kg_m3) ** (1.0 / 3.0)


def time_inside_radius_hours(periapsis_rm: float, velocity_kms: float, threshold_rm: float) -> float:
    if not math.isfinite(periapsis_rm) or not math.isfinite(velocity_kms) or not math.isfinite(threshold_rm):
        return math.nan
    if periapsis_rm <= 0.0 or velocity_kms < 0.0 or threshold_rm <= periapsis_rm:
        return 0.0

    periapsis_km = periapsis_rm * MARS_RADIUS_KM
    threshold_km = threshold_rm * MARS_RADIUS_KM

    if math.isclose(velocity_kms, 0.0, abs_tol=1e-12):
        cos_theta = max(-1.0, min(1.0, (2.0 * periapsis_km / threshold_km) - 1.0))
        theta = math.acos(cos_theta)
        d_value = math.tan(theta / 2.0)
        time_seconds = math.sqrt((2.0 * periapsis_km**3) / MARS_MU_KM3_S2) * (d_value + (d_value**3) / 3.0)
        return (2.0 * time_seconds) / 3600.0

    eccentricity = 1.0 + (periapsis_km * (velocity_kms**2)) / MARS_MU_KM3_S2
    if eccentricity <= 1.0:
        return math.nan

    semi_latus_rectum_km = periapsis_km * (1.0 + eccentricity)
    cos_theta = (semi_latus_rectum_km / threshold_km - 1.0) / eccentricity
    if cos_theta >= 1.0:
        return 0.0
    theta = math.acos(max(-1.0, min(1.0, cos_theta)))
    tan_half_theta = math.tan(theta / 2.0)
    hyperbolic_arg = math.sqrt((eccentricity - 1.0) / (eccentricity + 1.0)) * tan_half_theta
    if abs(hyperbolic_arg) >= 1.0:
        return math.nan
    hyperbolic_anomaly = 2.0 * math.atanh(hyperbolic_arg)
    semi_major_axis_abs_km = MARS_MU_KM3_S2 / (velocity_kms**2)
    time_seconds = math.sqrt((semi_major_axis_abs_km**3) / MARS_MU_KM3_S2) * (
        eccentricity * math.sinh(hyperbolic_anomaly) - hyperbolic_anomaly
    )
    return (2.0 * time_seconds) / 3600.0


def add_physics_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["encounter_eccentricity_proxy"] = eccentricity_proxy(enriched["periapsis_Rm"], enriched["v_inf_kms"])
    enriched["v_inf_squared"] = np.square(enriched["v_inf_kms"])
    with np.errstate(divide="ignore", invalid="ignore"):
        enriched["periapsis_inverse"] = 1.0 / enriched["periapsis_Rm"]
        enriched["spin_frequency_hr_inv"] = 1.0 / enriched["spin_period_hr"]
    enriched["angular_momentum_proxy"] = enriched["periapsis_Rm"] * enriched["v_inf_kms"]
    enriched["asteroid_radius_km"] = asteroid_radius_km(enriched["target_mass_kg"])
    tidal_threshold_rm = tidal_disruption_radius_rm()
    enriched["time_within_2_mars_radii_hr"] = [
        time_inside_radius_hours(float(periapsis_rm), float(velocity_kms), PROXIMITY_DISTANCE_RM)
        for periapsis_rm, velocity_kms in zip(enriched["periapsis_Rm"], enriched["v_inf_kms"])
    ]
    enriched["time_within_tidal_disruption_hr"] = [
        time_inside_radius_hours(float(periapsis_rm), float(velocity_kms), tidal_threshold_rm)
        for periapsis_rm, velocity_kms in zip(enriched["periapsis_Rm"], enriched["v_inf_kms"])
    ]
    enriched["particle_mass_proxy"] = enriched["target_mass_kg"] / pd.to_numeric(enriched["resolution_value"], errors="coerce")
    enriched["mass_resolution_interaction"] = enriched["mass_log10_kg"] - enriched["particle_log10"]
    return enriched.replace([np.inf, -np.inf], np.nan)


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
    feature_set_name: str,
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
                "feature_set": feature_set_name,
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
        pred_frame["feature_set"] = feature_set_name
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
    for feature_set_name, feature_columns in FEATURE_SET_COLUMNS.items():
        for target in target_columns:
            metrics, predictions = evaluate_grouped_oof_models(frame, target, feature_columns, fold_assignments, feature_set_name)
            metric_frames.append(metrics)
            prediction_frames.append(predictions)
    baseline_metrics = pd.concat(metric_frames, ignore_index=True).sort_values(["target", "model"]).reset_index(drop=True)
    baseline_predictions = pd.concat(prediction_frames, ignore_index=True)
    baseline_metrics.to_csv(TABLES_DIR / "baseline_metrics.csv", index=False)
    baseline_predictions.to_csv(TABLES_DIR / "baseline_oof_predictions.csv", index=False)
    return {"frame": frame, "fold_assignments": fold_assignments, "baseline_metrics": baseline_metrics, "baseline_predictions": baseline_predictions}


def random_forest_search_space(compact: bool = False) -> list[dict[str, object]]:
    if compact:
        estimator_values = [300, 500]
        depth_values = [None, 10]
        leaf_values = [2, 4]
        feature_values = ["sqrt", 0.8]
    else:
        estimator_values = [300, 500, 800]
        depth_values = [None, 6, 10, 16]
        leaf_values = [1, 2, 4, 8]
        feature_values = ["sqrt", 0.5, 0.8, 1.0]
    return [
        {
            "model": "random_forest",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
        }
        for n_estimators, max_depth, min_samples_leaf, max_features in itertools.product(
            estimator_values,
            depth_values,
            leaf_values,
            feature_values,
        )
    ]


def gradient_boosting_search_space(compact: bool = False) -> list[dict[str, object]]:
    if compact:
        estimator_values = [100, 200]
        learning_rates = [0.05, 0.1]
        depth_values = [2, 3]
        subsample_values = [0.9, 1.0]
        leaf_values = [2]
    else:
        estimator_values = [100, 200, 400]
        learning_rates = [0.03, 0.05, 0.1]
        depth_values = [2, 3, 4]
        subsample_values = [0.7, 0.9, 1.0]
        leaf_values = [1, 2, 4]
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
            estimator_values,
            learning_rates,
            depth_values,
            subsample_values,
            leaf_values,
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
    transform_pair: tuple[callable, callable] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Pipeline]:
    valid = frame[frame[target].notna()].copy()
    valid = valid.merge(fold_assignments[["row_index", "fold_index"]], left_index=True, right_on="row_index", how="left")
    X = valid[feature_columns].copy()
    y = pd.to_numeric(valid[target], errors="coerce")
    forward_transform, inverse_transform = transform_pair if transform_pair is not None else (lambda s: s, lambda s: s)
    y_train = forward_transform(y)
    pipeline = build_regression_pipeline(X, model_name, params)
    oof = np.full(len(valid), np.nan)
    fold_metrics: list[dict[str, object]] = []
    for fold_index in sorted(valid["fold_index"].dropna().unique()):
        train_mask = valid["fold_index"] != fold_index
        test_mask = valid["fold_index"] == fold_index
        fitted = clone(pipeline)
        fitted.fit(X.loc[train_mask], y_train.loc[train_mask])
        preds = inverse_transform(pd.Series(fitted.predict(X.loc[test_mask]), index=y.loc[test_mask].index)).to_numpy()
        oof[test_mask.to_numpy()] = preds
        fold_metrics.append(
            {
                "fold_index": int(fold_index),
                "r2": r2_score(y.loc[test_mask], preds),
                "mae": mean_absolute_error(y.loc[test_mask], preds),
                "rmse": float(np.sqrt(mean_squared_error(y.loc[test_mask], preds))),
            }
        )
    fitted_full = clone(pipeline).fit(X, y_train)
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


def target_transform_registry() -> dict[str, tuple[callable, callable]]:
    return {
        "raw": (lambda s: s, lambda s: s),
        "log1p": (lambda s: np.log1p(s.clip(lower=0.0)), lambda s: pd.Series(np.expm1(s), index=s.index)),
    }


def run_target_transform_stage(dataset_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_output_dirs()
    frame = load_canonical_dataset(dataset_path)
    fold_assignments = pd.read_csv(FOLD_ASSIGNMENTS_PATH) if FOLD_ASSIGNMENTS_PATH.exists() else build_group_folds(frame, frame["physical_file"].astype(str))
    transform_specs = {
        "n_fragments": ["raw", "log1p"],
        "largest_fragment_mass_kg": ["raw", "log1p"],
        "largest_fragment_particle_count": ["raw", "log1p"] if "largest_fragment_particle_count" in frame.columns else [],
        "largest_fragment_mass_fraction": ["raw", "log1p"],
    }
    registry = target_transform_registry()
    metrics_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for target, transforms in transform_specs.items():
        if target not in frame.columns or not transforms:
            continue
        for transform_name in transforms:
            metrics, predictions, _ = evaluate_model_config_oof(
                frame,
                target,
                FEATURE_SET_COLUMNS["with_fof_linking_length"],
                fold_assignments,
                "random_forest",
                None,
                transform_name=transform_name,
                transform_pair=registry[transform_name],
            )
            metrics_frames.append(metrics)
            prediction_frames.append(predictions)
    metrics_frame = pd.concat(metrics_frames, ignore_index=True)
    predictions_frame = pd.concat(prediction_frames, ignore_index=True)
    metrics_frame.to_csv(TABLES_DIR / "target_transform_metrics.csv", index=False)
    predictions_frame.to_csv(TABLES_DIR / "target_transform_oof_predictions.csv", index=False)
    return metrics_frame, predictions_frame


def determine_promoted_model(dataset_path: Path) -> dict[str, object]:
    ensure_output_dirs()
    frame = add_physics_features(load_canonical_dataset(dataset_path))
    baseline_metrics = pd.read_csv(TABLES_DIR / "baseline_metrics.csv") if (TABLES_DIR / "baseline_metrics.csv").exists() else run_baseline_stage(dataset_path)["baseline_metrics"]
    ablation_metrics = pd.read_csv(TABLES_DIR / "physics_feature_ablation_metrics.csv") if (TABLES_DIR / "physics_feature_ablation_metrics.csv").exists() else run_feature_ablation_stage(dataset_path)[0]
    tuning_promotion = pd.read_csv(TABLES_DIR / "promotion_summary.csv") if (TABLES_DIR / "promotion_summary.csv").exists() else run_tuning_stage(dataset_path)["promotion_summary"]
    baseline_row = baseline_metrics[(baseline_metrics["target"] == PRIMARY_TARGET) & (baseline_metrics["model"] == "random_forest")].sort_values("r2", ascending=False).iloc[0]
    best_ablation = ablation_metrics[ablation_metrics["target"] == PRIMARY_TARGET].sort_values(["r2", "mae"], ascending=[False, True]).iloc[0]
    promoted = {
        "promotion_label": "baseline RF",
        "model_name": "random_forest",
        "feature_set": str(baseline_row["feature_set"]),
        "include_physics_features": False,
        "params": None,
        "r2": float(baseline_row["r2"]),
        "mae": float(baseline_row["mae"]),
        "reason": "simplicity preferred because gains were small",
    }
    if bool(tuning_promotion["promote_tuned_model"].any()):
        tuned_choice = tuning_promotion[tuning_promotion["promote_tuned_model"]].sort_values("r2_gain", ascending=False).iloc[0]
        promoted.update(
            {
                "promotion_label": "tuned RF" if tuned_choice["candidate_model"] == "random_forest" else "tuned GB",
                "model_name": str(tuned_choice["candidate_model"]),
                "feature_set": str(tuned_choice["feature_set"]),
                "include_physics_features": False,
                "params": json.loads(tuned_choice["candidate_params_json"]),
                "r2": float(tuned_choice["candidate_r2"]),
                "mae": float(tuned_choice["candidate_mae"]),
                "reason": str(tuned_choice["promotion_reason"]),
            }
        )
    baseline_slice = representative_slice_plausibility(frame, "random_forest", str(baseline_row["feature_set"]), False, None)
    candidate_slice = representative_slice_plausibility(
        frame,
        str(best_ablation["model"]),
        str(best_ablation["feature_set"]),
        bool(best_ablation["include_physics_features"]),
        None,
    )
    slice_plausible = (
        candidate_slice["slice_mae"] <= baseline_slice["slice_mae"] * 1.05
        and candidate_slice["slice_prediction_range"] >= baseline_slice["slice_prediction_range"] * 0.7
    )
    if float(best_ablation["r2"]) >= promoted["r2"] + 0.02 and slice_plausible:
        promoted.update(
            {
                "promotion_label": "physics-feature RF" if best_ablation["model"] == "random_forest" else "physics-feature GB",
                "model_name": str(best_ablation["model"]),
                "feature_set": str(best_ablation["feature_set"]),
                "include_physics_features": bool(best_ablation["include_physics_features"]),
                "params": None,
                "r2": float(best_ablation["r2"]),
                "mae": float(best_ablation["mae"]),
                "reason": "physics-feature ablation materially improved BMF",
            }
        )
    elif float(best_ablation["r2"]) >= promoted["r2"] + 0.02:
        promoted["reason"] = "simplicity preferred because the physics-feature candidate failed the representative-slice plausibility check"
    write_promoted_model_info(promoted)
    return promoted


def add_trust_flags(predictions: pd.DataFrame, training_frame: pd.DataFrame, spread_threshold: float) -> pd.DataFrame:
    flagged = predictions.copy()
    ranges = {
        "mass_log10_kg": (training_frame["mass_log10_kg"].min(), training_frame["mass_log10_kg"].max()),
        "periapsis_Rm": (training_frame["periapsis_Rm"].min(), training_frame["periapsis_Rm"].max()),
        "v_inf_kms": (training_frame["v_inf_kms"].min(), training_frame["v_inf_kms"].max()),
    }
    coverage = training_frame.groupby(["mass_log10_kg", "periapsis_Rm"]).size().rename("bin_count").reset_index()
    flagged = flagged.merge(coverage, on=["mass_log10_kg", "periapsis_Rm"], how="left")
    flagged["bin_count"] = flagged["bin_count"].fillna(0)
    flagged["in_training_range"] = True
    for column, (lo, hi) in ranges.items():
        flagged["in_training_range"] &= flagged[column].between(lo, hi, inclusive="both")
    flagged["near_training_edge"] = (
        (flagged["periapsis_Rm"] <= ranges["periapsis_Rm"][0] + 0.1)
        | (flagged["periapsis_Rm"] >= ranges["periapsis_Rm"][1] - 0.1)
        | (flagged["v_inf_kms"] <= ranges["v_inf_kms"][0] + 0.1)
        | (flagged["v_inf_kms"] >= ranges["v_inf_kms"][1] - 0.1)
    )
    flagged["sparse_bin_flag"] = flagged["bin_count"] <= training_frame.groupby(["mass_log10_kg", "periapsis_Rm"]).size().median()
    flagged["extrapolation_flag"] = ~flagged["in_training_range"]
    flagged["borderline_bmf"] = flagged["predicted"].between(0.0771, 0.1229, inclusive="both")
    flagged["high_confidence"] = (
        flagged["in_training_range"]
        & ~flagged["near_training_edge"]
        & ~flagged["sparse_bin_flag"]
        & (flagged["model_spread"] <= spread_threshold)
        & ~flagged["borderline_bmf"]
    )
    flagged["recommendation"] = np.where(
        flagged["high_confidence"],
        "high_confidence_screening",
        np.where(
            flagged["extrapolation_flag"] | flagged["borderline_bmf"] | (flagged["model_spread"] > spread_threshold),
            "low_confidence_sph_required",
            "medium_confidence_screening",
        ),
    )
    return flagged


def run_trust_stage(dataset_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_output_dirs()
    promoted = determine_promoted_model(dataset_path)
    frame = add_physics_features(load_canonical_dataset(dataset_path))
    fold_assignments = pd.read_csv(FOLD_ASSIGNMENTS_PATH) if FOLD_ASSIGNMENTS_PATH.exists() else build_group_folds(frame, frame["physical_file"].astype(str))
    feature_columns = feature_columns_for_set(str(promoted["feature_set"]), bool(promoted["include_physics_features"]))
    promoted_metrics, promoted_predictions, promoted_model = evaluate_model_config_oof(
        frame, PRIMARY_TARGET, feature_columns, fold_assignments, str(promoted["model_name"]), promoted["params"]
    )
    rf_metrics, rf_predictions, rf_model = evaluate_model_config_oof(frame, PRIMARY_TARGET, feature_columns, fold_assignments, "random_forest", None)
    gb_metrics, gb_predictions, gb_model = evaluate_model_config_oof(frame, PRIMARY_TARGET, feature_columns, fold_assignments, "gradient_boosting", None)
    X = frame.loc[frame[PRIMARY_TARGET].notna(), feature_columns]
    promoted_predictions["predicted"] = promoted_predictions["predicted"].clip(0.0, 1.0)
    promoted_predictions["rf_full_prediction"] = rf_model.predict(X)[: len(promoted_predictions)]
    promoted_predictions["gb_full_prediction"] = gb_model.predict(X)[: len(promoted_predictions)]
    promoted_predictions["model_spread"] = np.abs(promoted_predictions["rf_full_prediction"] - promoted_predictions["gb_full_prediction"])
    fold_spread = promoted_predictions.groupby("fold_index")["residual"].transform("std").fillna(0.0)
    promoted_predictions["fold_spread"] = fold_spread
    spread_threshold = float(promoted_predictions["model_spread"].quantile(0.75))
    flagged_predictions = add_trust_flags(promoted_predictions, frame.loc[frame[PRIMARY_TARGET].notna()].copy(), spread_threshold)
    trust_summary = pd.DataFrame(
        [
            {
                "promoted_model": promoted["promotion_label"],
                "feature_set": promoted["feature_set"],
                "include_physics_features": promoted["include_physics_features"],
                "spread_threshold": spread_threshold,
                "high_confidence_rows": int((flagged_predictions["recommendation"] == "high_confidence_screening").sum()),
                "medium_confidence_rows": int((flagged_predictions["recommendation"] == "medium_confidence_screening").sum()),
                "low_confidence_rows": int((flagged_predictions["recommendation"] == "low_confidence_sph_required").sum()),
            }
        ]
    )
    flagged_predictions.to_csv(TABLES_DIR / "predictions_with_trust_flags.csv", index=False)
    trust_summary.to_csv(TABLES_DIR / "trust_summary.csv", index=False)
    return trust_summary, flagged_predictions


def representative_slice_mask(frame: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in REPRESENTATIVE_SLICE.items():
        mask &= frame[column] == value
    return mask


def representative_slice_plausibility(
    frame: pd.DataFrame,
    model_name: str,
    feature_set: str,
    include_physics_features: bool,
    params: dict[str, object] | None,
) -> dict[str, float]:
    slice_frame = frame.loc[representative_slice_mask(frame) & frame[PRIMARY_TARGET].notna()].sort_values("periapsis_Rm").copy()
    feature_columns = feature_columns_for_set(feature_set, include_physics_features)
    fitted = build_regression_pipeline(frame[feature_columns], model_name, params).fit(
        frame[feature_columns], pd.to_numeric(frame[PRIMARY_TARGET], errors="coerce")
    )
    predictions = pd.Series(fitted.predict(slice_frame[feature_columns]), index=slice_frame.index).clip(0.0, 1.0)
    return {
        "slice_mae": float(np.abs(slice_frame[PRIMARY_TARGET] - predictions).mean()),
        "slice_prediction_range": float(predictions.max() - predictions.min()),
    }


def plot_slice_diagnostic(
    frame: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    promoted_predictions: pd.DataFrame,
    target: str,
    feature_columns: list[str],
    promoted_model: Pipeline,
    output_path: Path,
) -> None:
    slice_frame = frame.loc[representative_slice_mask(frame)].sort_values("periapsis_Rm").copy()
    if slice_frame.empty or target not in slice_frame.columns:
        return
    grid = pd.concat([slice_frame.iloc[[0]].copy()] * 250, ignore_index=True)
    observed = np.sort(slice_frame["periapsis_Rm"].unique())
    grid["periapsis_Rm"] = np.linspace(max(0.0, observed.min() - 0.1), observed.max() + 0.1, len(grid))
    grid["periapsis_value"] = np.round(grid["periapsis_Rm"] * 10.0).astype(int)
    full_curve = promoted_model.predict(grid[feature_columns])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axvspan(grid["periapsis_Rm"].min(), observed.min(), color="#f3d3d3", alpha=0.45)
    ax.axvspan(observed.min(), observed.max(), color="#dbe8ff", alpha=0.25)
    ax.axvspan(observed.max(), grid["periapsis_Rm"].max(), color="#f3d3d3", alpha=0.45)
    ax.scatter(slice_frame["periapsis_Rm"], slice_frame[target], color="black", s=35, label="SPH simulation")
    base_slice = baseline_predictions[(baseline_predictions["target"] == target) & (baseline_predictions["model"] == "random_forest")]
    base_slice = base_slice.loc[representative_slice_mask(base_slice)]
    promoted_slice = promoted_predictions[promoted_predictions["target"] == target].loc[representative_slice_mask(promoted_predictions)]
    if not base_slice.empty:
        ax.scatter(base_slice["periapsis_Rm"], base_slice["predicted"], color="#1f77b4", marker="^", s=45, label="baseline RF OOF")
    if not promoted_slice.empty:
        ax.scatter(promoted_slice["periapsis_Rm"], promoted_slice["predicted"], color="#d62728", marker="D", s=45, label="promoted OOF")
    ax.plot(grid["periapsis_Rm"], full_curve, color="#d62728", linewidth=2.2, label="promoted full-model curve")
    ax.set_xlabel("Periapsis ($R_{Mars}$)")
    ax.set_ylabel(target)
    ax.set_title(f"{target} slice: baseline vs promoted")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_slice_diagnostics_stage(dataset_path: Path) -> None:
    frame = add_physics_features(load_canonical_dataset(dataset_path))
    promoted = determine_promoted_model(dataset_path)
    baseline_predictions = pd.read_csv(TABLES_DIR / "baseline_oof_predictions.csv")
    trust_predictions = pd.read_csv(TABLES_DIR / "predictions_with_trust_flags.csv") if (TABLES_DIR / "predictions_with_trust_flags.csv").exists() else run_trust_stage(dataset_path)[1]
    fold_assignments = pd.read_csv(FOLD_ASSIGNMENTS_PATH)
    feature_columns = feature_columns_for_set(str(promoted["feature_set"]), bool(promoted["include_physics_features"]))
    _, _, fitted_model = evaluate_model_config_oof(frame, PRIMARY_TARGET, feature_columns, fold_assignments, str(promoted["model_name"]), promoted["params"])
    plot_slice_diagnostic(frame, baseline_predictions, trust_predictions, PRIMARY_TARGET, feature_columns, fitted_model, PLOTS_DIR / "bmf_slice_baseline_vs_promoted.png")
    for target, path_name in [
        ("n_fragments", "fragment_count_slice_baseline_vs_promoted.png"),
        ("largest_fragment_mass_kg", "largest_fragment_mass_slice_baseline_vs_promoted.png"),
        ("largest_fragment_particle_count", "largest_fragment_particle_count_slice_baseline_vs_promoted.png"),
    ]:
        if target in frame.columns:
            _, target_predictions, fitted_target = evaluate_model_config_oof(frame, target, feature_columns, fold_assignments, str(promoted["model_name"]), promoted["params"])
            plot_slice_diagnostic(frame, baseline_predictions, target_predictions, target, feature_columns, fitted_target, PLOTS_DIR / path_name)


def draw_heatmap(
    ax: plt.Axes,
    table: pd.DataFrame,
    title: str,
    cmap: str,
    cbar_label: str,
    *,
    distinguish_zero_and_missing: bool = False,
) -> None:
    values = table.to_numpy(dtype=float)
    cmap_obj = colormaps.get_cmap(cmap).copy()
    image_kwargs: dict[str, object] = {"aspect": "auto", "origin": "lower", "cmap": cmap_obj}
    if distinguish_zero_and_missing:
        cmap_obj.set_bad("#cfecc7")
        cmap_obj.set_under("#e8f1fb")
        image_kwargs["vmin"] = 0.5
        values = np.ma.masked_invalid(values)
    image = ax.imshow(values, **image_kwargs)
    ax.set_title(title)
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels([str(value) for value in table.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels([str(value) for value in table.index])
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)


def plot_coverage_and_error_heatmaps(frame: pd.DataFrame, trust_predictions: pd.DataFrame) -> pd.DataFrame:
    coverage_mass_peri = frame.pivot_table(index="mass_log10_kg", columns="periapsis_Rm", values="physical_file", aggfunc="count", fill_value=0)
    coverage_peri_vel = frame.pivot_table(index="periapsis_Rm", columns="v_inf_kms", values="physical_file", aggfunc="count", fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    draw_heatmap(
        axes[0],
        coverage_mass_peri,
        "Coverage: mass vs periapsis",
        "Blues",
        "Runs",
        distinguish_zero_and_missing=True,
    )
    draw_heatmap(
        axes[1],
        coverage_peri_vel,
        "Coverage: periapsis vs velocity",
        "Blues",
        "Runs",
        distinguish_zero_and_missing=True,
    )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "parameter_coverage_heatmaps.png", dpi=180)
    plt.close(fig)
    trust_predictions = trust_predictions.copy()
    trust_predictions["abs_error"] = trust_predictions["residual"].abs()
    error_mass_peri = trust_predictions.pivot_table(index="mass_log10_kg", columns="periapsis_Rm", values="abs_error", aggfunc="mean")
    error_peri_vel = trust_predictions.pivot_table(index="periapsis_Rm", columns="v_inf_kms", values="abs_error", aggfunc="mean")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    draw_heatmap(axes[0], error_mass_peri, "Mean |error|: mass vs periapsis", "OrRd", "|error|")
    draw_heatmap(axes[1], error_peri_vel, "Mean |error|: periapsis vs velocity", "OrRd", "|error|")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "coverage_vs_error_heatmaps.png", dpi=180)
    plt.close(fig)
    sparse_mask = trust_predictions["sparse_bin_flag"]
    summary = pd.DataFrame(
        [
            {
                "occupied_mass_peri_bins": int((coverage_mass_peri > 0).sum().sum()),
                "total_mass_peri_bins": int(coverage_mass_peri.size),
                "occupied_peri_vel_bins": int((coverage_peri_vel > 0).sum().sum()),
                "total_peri_vel_bins": int(coverage_peri_vel.size),
                "mean_error_dense_bins": float(trust_predictions.loc[~sparse_mask, "abs_error"].mean()),
                "mean_error_sparse_bins": float(trust_predictions.loc[sparse_mask, "abs_error"].mean()),
                "worst_error_bin_mass_peri": str(error_mass_peri.stack().idxmax()) if error_mass_peri.notna().any().any() else "",
                "worst_error_bin_peri_vel": str(error_peri_vel.stack().idxmax()) if error_peri_vel.notna().any().any() else "",
            }
        ]
    )
    summary.to_csv(TABLES_DIR / "coverage_error_summary.csv", index=False)
    return summary


def pairwise_heatmap_table(df: pd.DataFrame, row: str, col: str, value: str | None = None, agg: str = "count") -> pd.DataFrame:
    if value is None:
        table = df.pivot_table(index=row, columns=col, values="physical_file", aggfunc="count", fill_value=0)
    else:
        table = df.pivot_table(index=row, columns=col, values=value, aggfunc=agg)
    return table.sort_index().sort_index(axis=1)


def plot_extended_pairwise_diagnostics(frame: pd.DataFrame, trust_predictions: pd.DataFrame) -> pd.DataFrame:
    pair_specs = [
        ("mass_log10_kg", "v_inf_kms", "Mass vs velocity"),
        ("mass_log10_kg", "fof_linking_length", "Mass vs FoF linking length"),
        ("periapsis_Rm", "v_inf_kms", "Periapsis vs velocity"),
        ("mass_log10_kg", "spin_axis", "Mass vs spin axis"),
    ]
    trust_predictions = trust_predictions.copy()
    trust_predictions["abs_error"] = trust_predictions["residual"].abs()
    fig, axes = plt.subplots(len(pair_specs), 2, figsize=(12.5, 4.2 * len(pair_specs)))
    summary_rows: list[dict[str, object]] = []
    for row_idx, (row_col, col_col, label) in enumerate(pair_specs):
        coverage = pairwise_heatmap_table(frame, row_col, col_col)
        error = pairwise_heatmap_table(trust_predictions, row_col, col_col, value="abs_error", agg="mean")
        draw_heatmap(
            axes[row_idx, 0],
            coverage,
            f"{label}: coverage",
            "Blues",
            "Runs",
            distinguish_zero_and_missing=True,
        )
        draw_heatmap(
            axes[row_idx, 1],
            error,
            f"{label}: mean |error|",
            "OrRd",
            "|error|",
            distinguish_zero_and_missing=True,
        )
        aligned_error = error.reindex(index=coverage.index, columns=coverage.columns)
        summary_rows.append(
            {
                "row_parameter": row_col,
                "column_parameter": col_col,
                "occupied_bins": int((coverage > 0).sum().sum()),
                "total_bins": int(coverage.size),
                "missing_error_bins": int(aligned_error.isna().sum().sum()),
                "worst_error_bin": str(error.stack().idxmax()) if error.notna().any().any() else "",
                "worst_error_value": float(error.stack().max()) if error.notna().any().any() else np.nan,
            }
        )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "extended_pairwise_coverage_error_heatmaps.png", dpi=180)
    plt.close(fig)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLES_DIR / "extended_pairwise_coverage_error_summary.csv", index=False)
    return summary


def run_diagnostics_stage(dataset_path: Path) -> pd.DataFrame:
    trust_predictions = pd.read_csv(TABLES_DIR / "predictions_with_trust_flags.csv") if (TABLES_DIR / "predictions_with_trust_flags.csv").exists() else run_trust_stage(dataset_path)[1]
    frame = add_physics_features(load_canonical_dataset(dataset_path))
    run_slice_diagnostics_stage(dataset_path)
    frame = frame.loc[frame[PRIMARY_TARGET].notna()].copy()
    plot_extended_pairwise_diagnostics(frame, trust_predictions)
    return plot_coverage_and_error_heatmaps(frame, trust_predictions)


def write_model_card(dataset_path: Path) -> Path:
    promoted = determine_promoted_model(dataset_path)
    trust_summary = pd.read_csv(TABLES_DIR / "trust_summary.csv") if (TABLES_DIR / "trust_summary.csv").exists() else run_trust_stage(dataset_path)[0]
    coverage_summary = pd.read_csv(TABLES_DIR / "coverage_error_summary.csv") if (TABLES_DIR / "coverage_error_summary.csv").exists() else run_diagnostics_stage(dataset_path)
    path = OUTPUT_ROOT / "model_card.md"
    text = "\n".join(
        [
            "# Physics-Structured Surrogate Model Card",
            "",
            "The surrogate is not a replacement for SPH. It is a fast in-domain screening model trained on SPH-derived outcomes.",
            "",
            f"- Promoted model name: `{promoted['promotion_label']}`",
            f"- Primary target: `{PRIMARY_TARGET}`",
            f"- Feature set: `{promoted['feature_set']}`",
            f"- Physics-derived features included: `{promoted['include_physics_features']}`",
            f"- Promotion reason: {promoted['reason']}",
            f"- Grouped-CV BMF R^2: {promoted['r2']:.4f}",
            f"- Grouped-CV BMF MAE: {promoted['mae']:.4f}",
            f"- Trust spread threshold: {trust_summary['spread_threshold'].iloc[0]:.4f}",
            f"- High-confidence predictions: {int(trust_summary['high_confidence_rows'].iloc[0])}",
            f"- Medium-confidence predictions: {int(trust_summary['medium_confidence_rows'].iloc[0])}",
            f"- Low-confidence / SPH required: {int(trust_summary['low_confidence_rows'].iloc[0])}",
            f"- Coverage summary file: `{(TABLES_DIR / 'coverage_error_summary.csv').as_posix()}`",
            "",
            "## Caution zones",
            "- outside the training range",
            "- near the sampled edge of parameter space",
            "- sparse coverage bins",
            "- borderline BMF around 0.10",
            "- cases needing detailed debris, orbit, or eccentricity evolution",
            "",
            "## Future work",
            "- expand the SPH archive in sparse regions",
            "- validate promoted predictions against newly run SPH cases",
            "- test stronger physics-aware proxies before considering neural methods",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")
    return path


def write_notebook_stub() -> Path:
    path = Path("physics_structured_surrogate.ipynb")
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Physics-Structured Surrogate\n", "Fast in-domain SPH screening surrogate.\n"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## Sections\n", "0. Config\n1. Load data and reproduce baseline\n2. Feature sets\n3. Target definitions and transforms\n4. Grouped CV tuning\n5. Baseline vs tuned results\n6. With-FoF vs without-FoF comparison\n7. Physics feature ablation\n8. Promoted model selection\n9. Trust flags and decision rules\n10. Slice diagnostics\n11. Coverage and error diagnostics\n12. Model card summary\n13. Conclusions and next steps\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["from pathlib import Path\n", "import pandas as pd\n", "root = Path('ml/physics_structured_surrogate')\n", "baseline = pd.read_csv(root / 'tables' / 'baseline_metrics.csv')\n", "baseline.head()\n"]},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.13"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    return path


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


def run_tuning_stage(dataset_path: Path, compact_search: bool = False) -> dict[str, pd.DataFrame]:
    ensure_output_dirs()
    frame = load_canonical_dataset(dataset_path)
    fold_assignments = pd.read_csv(FOLD_ASSIGNMENTS_PATH) if FOLD_ASSIGNMENTS_PATH.exists() else build_group_folds(frame, frame["physical_file"].astype(str))
    baseline_metrics_path = TABLES_DIR / "baseline_metrics.csv"
    baseline_metrics = pd.read_csv(baseline_metrics_path) if baseline_metrics_path.exists() else run_baseline_stage(dataset_path)["baseline_metrics"]
    search_frames: list[pd.DataFrame] = []
    promotion_frames: list[pd.DataFrame] = []
    for feature_set_name, feature_columns in FEATURE_SET_COLUMNS.items():
        feature_search_results = pd.concat(
            [
                evaluate_tuning_candidates(frame, fold_assignments, feature_columns, random_forest_search_space(compact=compact_search), feature_set_name),
                evaluate_tuning_candidates(frame, fold_assignments, feature_columns, gradient_boosting_search_space(compact=compact_search), feature_set_name),
            ],
            ignore_index=True,
        )
        search_frames.append(feature_search_results)
        promotion_frames.append(summarize_tuning_promotion(baseline_metrics, feature_search_results, feature_set_name))
    search_results = pd.concat(search_frames, ignore_index=True)
    tuned_metrics = search_results.sort_values(["r2", "mae"], ascending=[False, True]).groupby(["target", "feature_set", "model"], as_index=False).head(1)
    promotion_summary = pd.concat(promotion_frames, ignore_index=True)
    search_results.to_csv(TABLES_DIR / "tuning_search_results.csv", index=False)
    tuned_metrics.to_csv(TABLES_DIR / "tuned_model_metrics.csv", index=False)
    promotion_summary.to_csv(TABLES_DIR / "promotion_summary.csv", index=False)
    return {"tuning_search_results": search_results, "tuned_model_metrics": tuned_metrics, "promotion_summary": promotion_summary}


def load_best_candidate_params(feature_set_name: str) -> tuple[str, dict[str, object]]:
    tuning_results = pd.read_csv(TABLES_DIR / "tuning_search_results.csv")
    subset = tuning_results[tuning_results["feature_set"] == feature_set_name].sort_values(["r2", "mae"], ascending=[False, True]).iloc[0]
    return str(subset["model"]), json.loads(subset["params_json"])


def run_fof_compare_stage(dataset_path: Path) -> pd.DataFrame:
    ensure_output_dirs()
    frame = load_canonical_dataset(dataset_path)
    fold_assignments = pd.read_csv(FOLD_ASSIGNMENTS_PATH) if FOLD_ASSIGNMENTS_PATH.exists() else build_group_folds(frame, frame["physical_file"].astype(str))
    if not (TABLES_DIR / "tuning_search_results.csv").exists():
        run_tuning_stage(dataset_path)
    rows: list[pd.DataFrame] = []
    promotion_rows: list[dict[str, object]] = []
    for feature_set_name in FEATURE_SET_COLUMNS:
        model_name, params = load_best_candidate_params(feature_set_name)
        metrics, _, _ = evaluate_model_config_oof(frame, PRIMARY_TARGET, FEATURE_SET_COLUMNS[feature_set_name], fold_assignments, model_name, params)
        metrics["feature_set"] = feature_set_name
        metrics["selection_basis"] = "best_tuned_candidate"
        rows.append(metrics)
    metrics_frame = pd.concat(rows, ignore_index=True)
    predictive_row = metrics_frame.sort_values(["r2", "mae"], ascending=[False, True]).iloc[0]
    no_fof_row = metrics_frame[metrics_frame["feature_set"] == "without_fof_linking_length"].iloc[0]
    with_fof_row = metrics_frame[metrics_frame["feature_set"] == "with_fof_linking_length"].iloc[0]
    prefer_without = float(with_fof_row["r2"] - no_fof_row["r2"]) < 0.02
    promotion_rows.append(
        {
            "best_predictive_feature_set": predictive_row["feature_set"],
            "best_predictive_model": predictive_row["model"],
            "best_more_physical_feature_set": "without_fof_linking_length" if prefer_without else with_fof_row["feature_set"],
            "best_more_physical_model": no_fof_row["model"] if prefer_without else with_fof_row["model"],
            "decision_reason": "prefer without FoF when performance is close" if prefer_without else "with FoF materially improves prediction",
        }
    )
    metrics_frame.to_csv(TABLES_DIR / "with_vs_without_fof_metrics.csv", index=False)
    pd.DataFrame(promotion_rows).to_csv(TABLES_DIR / "with_vs_without_fof_promotion.csv", index=False)
    return metrics_frame


def run_feature_ablation_stage(dataset_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_output_dirs()
    frame = add_physics_features(load_canonical_dataset(dataset_path))
    fold_assignments = pd.read_csv(FOLD_ASSIGNMENTS_PATH) if FOLD_ASSIGNMENTS_PATH.exists() else build_group_folds(frame, frame["physical_file"].astype(str))
    ablation_specs = [
        ("original_with_fof", "with_fof_linking_length", False),
        ("physics_with_fof", "with_fof_linking_length", True),
        ("original_without_fof", "without_fof_linking_length", False),
        ("physics_without_fof", "without_fof_linking_length", True),
    ]
    metric_frames: list[pd.DataFrame] = []
    importance_rows: list[dict[str, object]] = []
    for ablation_name, feature_set_name, include_physics in ablation_specs:
        feature_columns = feature_columns_for_set(feature_set_name, include_physics)
        for model_name in ["random_forest", "gradient_boosting"]:
            metrics, _, fitted = evaluate_model_config_oof(frame, PRIMARY_TARGET, feature_columns, fold_assignments, model_name, None)
            metrics["ablation_name"] = ablation_name
            metrics["feature_set"] = feature_set_name
            metrics["include_physics_features"] = include_physics
            metric_frames.append(metrics)
            if model_name == "random_forest":
                X = frame.loc[frame[PRIMARY_TARGET].notna(), feature_columns]
                y = pd.to_numeric(frame.loc[frame[PRIMARY_TARGET].notna(), PRIMARY_TARGET], errors="coerce")
                result = permutation_importance(fitted, X, y, scoring="r2", n_repeats=5, random_state=RANDOM_STATE)
                for feature_name, importance_mean in zip(feature_columns, result.importances_mean):
                    importance_rows.append(
                        {
                            "ablation_name": ablation_name,
                            "feature_set": feature_set_name,
                            "feature": feature_name,
                            "importance_mean": importance_mean,
                        }
                    )
    metrics_frame = pd.concat(metric_frames, ignore_index=True)
    importance_frame = pd.DataFrame(importance_rows).sort_values(["ablation_name", "importance_mean"], ascending=[True, False])
    metrics_frame.to_csv(TABLES_DIR / "physics_feature_ablation_metrics.csv", index=False)
    importance_frame.to_csv(TABLES_DIR / "physics_feature_importance.csv", index=False)
    return metrics_frame, importance_frame


def main() -> None:
    args = parse_args()
    if args.stage in {"baseline", "all"}:
        run_baseline_stage(args.dataset)
    if args.stage in {"tune", "all"}:
        run_tuning_stage(args.dataset, compact_search=args.compact_search)
    if args.stage in {"fof_compare", "all"}:
        run_fof_compare_stage(args.dataset)
    if args.stage in {"target_transforms", "all"}:
        run_target_transform_stage(args.dataset)
    if args.stage in {"trust", "all"}:
        run_trust_stage(args.dataset)
    if args.stage in {"diagnostics", "all"}:
        run_diagnostics_stage(args.dataset)
    if args.stage in {"package", "all"}:
        write_model_card(args.dataset)
        write_notebook_stub()


if __name__ == "__main__":
    main()
