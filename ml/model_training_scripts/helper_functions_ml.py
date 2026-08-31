from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PRIMARY_TARGET = "bound_mass_fraction"
RANDOM_STATE = 42
N_SPLITS = 5
MARS_MU_KM3_S2 = 4.282837e4
MARS_RADIUS_KM = 3389.5
MARS_DENSITY_KG_M3 = 3933.5
ASTEROID_BULK_DENSITY_KG_M3 = 2700.0
PROXIMITY_DISTANCE_RM = 2.0
FLUID_ROCHE_FACTOR = 2.44

FILENAME_RE = re.compile(
    r"^(?P<prefix>Ma_xp)_(?P<mass>A\d{4}(?:c30)?)(?:_(?P<spin>s\d{3}[A-Za-z]*))?"
    r"_n(?P<resolution>\d+)_r(?P<periapsis>\d+)_v(?P<velocity>\d+)"
    r"_(?P<timestep>\d+)"
    r"_fof_(?P<linking_length>[0-9.]+)_"
    r"(?P<chunk>\d+)\.hdf5$"
)


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
    if "bound_mass_fraction" not in canonical.columns:
        raise KeyError("Canonical dataset is missing bound_mass_fraction.")
    main_fraction = pd.to_numeric(canonical["bound_mass_fraction"], errors="coerce")
    canonical["bound_mass_fraction_ge_0_1"] = main_fraction >= 0.1
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
    return enriched.replace([np.inf, -np.inf], np.nan)


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
        ],
        remainder="drop",
    )


def build_regression_pipeline(X: pd.DataFrame, model_name: str, params: dict[str, object] | None = None) -> Pipeline:
    params = params or {}
    if model_name == "ridge":
        estimator = Ridge(**params)
        preprocessor = build_preprocessor(X, scaled=True)
    elif model_name == "gradient_boosting":
        estimator = GradientBoostingRegressor(**params)
        preprocessor = build_preprocessor(X, scaled=False)
    elif model_name == "random_forest":
        estimator = RandomForestRegressor(**params)
        preprocessor = build_preprocessor(X, scaled=False)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def build_or_load_group_folds(
    frame: pd.DataFrame,
    output_path: Path,
    group_column: str = "physical_file",
) -> pd.DataFrame:
    if output_path.exists():
        folds = pd.read_csv(output_path)
        required = {"row_index", "fold_index", group_column}
        if required.issubset(folds.columns):
            return folds

    groups = frame[group_column].astype(str)
    splitter = GroupKFold(n_splits=min(N_SPLITS, groups.nunique()))
    fold_assignments = np.full(len(frame), -1, dtype=int)
    for fold_index, (_, test_idx) in enumerate(splitter.split(frame, groups=groups)):
        fold_assignments[test_idx] = fold_index

    folds = pd.DataFrame(
        {
            "row_index": frame.index.to_numpy(),
            group_column: groups.to_numpy(),
            "fold_index": fold_assignments,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output_path, index=False)
    return folds


def evaluate_grouped_oof_regression(
    frame: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    params: dict[str, object] | None,
    fold_assignments: pd.DataFrame,
    target_column: str = PRIMARY_TARGET,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    valid = frame.loc[frame[target_column].notna()].copy()
    valid = valid.merge(fold_assignments[["row_index", "fold_index"]], left_index=True, right_on="row_index", how="left")
    X = valid[feature_columns].copy()
    y = pd.to_numeric(valid[target_column], errors="coerce")
    oof = np.full(len(valid), np.nan)
    fold_rows: list[dict[str, float | int]] = []

    for fold_index in sorted(valid["fold_index"].dropna().unique()):
        train_mask = valid["fold_index"] != fold_index
        test_mask = valid["fold_index"] == fold_index
        fitted = build_regression_pipeline(X.loc[train_mask], model_name, params)
        fitted.fit(X.loc[train_mask], y.loc[train_mask])
        preds = np.clip(fitted.predict(X.loc[test_mask]), 0.0, 1.0)
        oof[test_mask.to_numpy()] = preds
        fold_rows.append(
            {
                "fold_index": int(fold_index),
                "r2": float(r2_score(y.loc[test_mask], preds)),
                "mae": float(mean_absolute_error(y.loc[test_mask], preds)),
                "rmse": float(np.sqrt(mean_squared_error(y.loc[test_mask], preds))),
            }
        )

    fold_frame = pd.DataFrame(fold_rows)
    metrics = {
        "rows": int(len(valid)),
        "r2": float(r2_score(y, oof)),
        "mae": float(mean_absolute_error(y, oof)),
        "mse": float(mean_squared_error(y, oof)),
        "rmse": float(np.sqrt(mean_squared_error(y, oof))),
        "fold_r2_mean": float(fold_frame["r2"].mean()),
        "fold_r2_std": float(fold_frame["r2"].std(ddof=0)),
        "fold_mae_mean": float(fold_frame["mae"].mean()),
        "fold_mae_std": float(fold_frame["mae"].std(ddof=0)),
        "fold_rmse_mean": float(fold_frame["rmse"].mean()),
        "fold_rmse_std": float(fold_frame["rmse"].std(ddof=0)),
    }
    predictions = valid[["physical_file", target_column, "fold_index"] + feature_columns].copy()
    predictions["predicted_bmf"] = oof
    predictions["residual"] = predictions[target_column] - predictions["predicted_bmf"]
    predictions = predictions.rename(columns={target_column: "actual_bmf"})
    return metrics, predictions.reset_index(drop=True)
