#!/usr/bin/env python3
"""Train a physics-structured tabular surrogate for SPH-derived outcomes."""

from __future__ import annotations

import argparse
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


def main() -> None:
    args = parse_args()
    if args.stage not in {"baseline", "all"}:
        raise NotImplementedError(f"Stage not implemented yet: {args.stage}")
    run_baseline_stage(args.dataset)


if __name__ == "__main__":
    main()
