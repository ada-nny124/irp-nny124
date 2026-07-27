"""Local web app for the SPH screening demo UI."""

from __future__ import annotations

import csv
import io
import json
import math
import pickle
import sys
from datetime import UTC, datetime
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from triage import add_derived_features, check_training_domain, get_artifact_status, load_artifacts, predict_cases

MODEL_DIR = ROOT / "ml" / "triage"
BOUND_MODELS_DIR = ROOT / "ml" / "bound_outcomes" / "models"
BOUND_TABLES_DIR = ROOT / "ml" / "bound_outcomes" / "tables"
SURROGATE_TABLES_DIR = ROOT / "ml" / "physics_structured_surrogate" / "tables"
DATASET_PATH = ROOT / "extraction_outputs" / "bound_outcomes.csv"
HTML_PATH = ROOT / "src" / "triage" / "templates" / "sph_triage_dashboard.html"

MARS_MU_KM3_S2 = 4.282837e4
MARS_RADIUS_KM = 3389.5
MARS_DENSITY_KG_M3 = 3933.5
ASTEROID_BULK_DENSITY_KG_M3 = 2700.0
PROXIMITY_DISTANCE_RM = 2.0
FLUID_ROCHE_FACTOR = 2.44
BMF_THRESHOLD = 0.10
BORDERLINE_BMF_MIN = 0.0771
BORDERLINE_BMF_MAX = 0.1229
HIGH_SUPPORT_MIN = 80.0
MODERATE_SUPPORT_MIN = 60.0
SCIENTIFIC_SEMANTICS_NOTE = (
    "Debris refers to all material outside the largest remnant. Current deployed BMF is trained on total fragment mass, "
    "so BMF and unbound values are shown as model outputs on that denominator, not as a strict additive parent-mass budget."
)
EXPORT_COLUMNS = [
    "prediction_timestamp",
    "model_bundle_id",
    "case_name",
    "mass_log10_kg",
    "periapsis_Rm",
    "v_inf_kms",
    "has_explicit_spin",
    "spin_axis",
    "spin_period_hr",
    "resolution_value",
    "timestep",
    "fof_linking_length",
    "parent_mass_kg",
    "largest_remnant_fraction",
    "largest_remnant_mass_kg",
    "fragmentation_label",
    "fragmentation_classifier_score",
    "predicted_bmf",
    "predicted_bound_mass_kg",
    "predicted_unbound_mass_fraction",
    "predicted_unbound_mass_kg",
    "support_score",
    "support_category",
    "training_range_status",
    "edge_status",
    "nearby_run_count",
    "model_spread_fraction",
    "model_spread_percentage_points",
    "recommendation",
    "recommendation_reason",
    "scientific_semantics_note",
]
INPUT_FIELD_ORDER = [
    "case_name",
    "mass_log10_kg",
    "periapsis_Rm",
    "v_inf_kms",
    "has_explicit_spin",
    "spin_axis",
    "spin_period_hr",
    "resolution_value",
    "timestep",
    "fof_linking_length",
]


@lru_cache(maxsize=1)
def load_dashboard_html() -> bytes:
    return HTML_PATH.read_bytes()


@lru_cache(maxsize=1)
def load_triage_bundle() -> tuple[object, object, dict[str, object]]:
    artifacts = load_artifacts(MODEL_DIR)
    if artifacts is None:
        raise FileNotFoundError("Missing required fragmentation artifacts in ml/triage")
    return artifacts


@lru_cache(maxsize=1)
def load_bound_metrics_tables() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    classification_path = BOUND_TABLES_DIR / "classification_metrics.csv"
    regression_path = BOUND_TABLES_DIR / "regression_metrics.csv"
    classification_df = pd.read_csv(classification_path) if classification_path.exists() else None
    regression_df = pd.read_csv(regression_path) if regression_path.exists() else None
    return classification_df, regression_df


@lru_cache(maxsize=1)
def select_best_bound_model_paths() -> dict[str, Path]:
    _, regression_df = load_bound_metrics_tables()
    selected: dict[str, Path] = {}
    if regression_df is not None and not regression_df.empty:
        for target in [
            "bound_mass_fraction",
            "bound_fragment_count",
            "largest_bound_fragment_mass_kg",
            "average_bound_fragment_mass_kg",
        ]:
            subset = regression_df[regression_df["target"] == target].sort_values(
                ["r2", "mae", "rmse"],
                ascending=[False, True, True],
            )
            if subset.empty:
                continue
            row = subset.iloc[0]
            selected[target] = BOUND_MODELS_DIR / (
                f"{row['dataset']}__{row['feature_set']}__{row['target']}__{row['model']}.pkl"
            )
    return selected


@lru_cache(maxsize=1)
def load_bound_models() -> dict[str, object]:
    models: dict[str, object] = {}
    for target, path in select_best_bound_model_paths().items():
        if not path.exists():
            continue
        with path.open("rb") as handle:
            models[target] = pickle.load(handle)
    return models


@lru_cache(maxsize=1)
def select_best_bound_classifiers() -> dict[str, dict[str, object]]:
    classification_df, _ = load_bound_metrics_tables()
    selected: dict[str, dict[str, object]] = {}
    if classification_df is None or classification_df.empty:
        return selected

    for target in ["has_any_bound_mass", "bound_mass_fraction_ge_0_1"]:
        subset = classification_df[classification_df["target"] == target].sort_values(
            ["balanced_accuracy", "f1", "roc_auc"],
            ascending=[False, False, False],
        )
        if subset.empty:
            continue
        row = subset.iloc[0]
        path = BOUND_MODELS_DIR / f"{row['dataset']}__{row['feature_set']}__{row['target']}__{row['model']}.pkl"
        selected[target] = {
            "target": str(row["target"]),
            "dataset": str(row["dataset"]),
            "feature_set": str(row["feature_set"]),
            "model_name": str(row["model"]),
            "balanced_accuracy": float(row["balanced_accuracy"]),
            "roc_auc": float(row["roc_auc"]),
            "path": str(path),
            "available": path.exists(),
        }
    return selected


@lru_cache(maxsize=1)
def load_spread_models() -> dict[str, object]:
    model_paths = {
        "random_forest": BOUND_MODELS_DIR
        / "all_successful_runs__with_fof_linking_length__bound_mass_fraction__random_forest_regressor.pkl",
        "gradient_boosting": BOUND_MODELS_DIR
        / "all_successful_runs__with_fof_linking_length__bound_mass_fraction__gradient_boosting_regressor.pkl",
    }
    models: dict[str, object] = {}
    for name, path in model_paths.items():
        if not path.exists():
            continue
        with path.open("rb") as handle:
            models[name] = pickle.load(handle)
    return models


def parse_numeric_code(series: pd.Series, pattern: str, scale: float = 1.0) -> pd.Series:
    extracted = series.fillna("").astype(str).str.extract(pattern)[0]
    return pd.to_numeric(extracted, errors="coerce") / scale


@lru_cache(maxsize=1)
def load_support_frame() -> pd.DataFrame:
    frame = pd.read_csv(DATASET_PATH, low_memory=False)
    frame["mass_log10_kg"] = parse_numeric_code(frame["mass_code"], r"A(\d{4})", scale=100.0)
    frame["resolution_value"] = parse_numeric_code(frame["resolution_code"], r"n(\d+)")
    frame["periapsis_Rm"] = parse_numeric_code(frame["periapsis_code"], r"r(\d+)", scale=10.0)
    frame["v_inf_kms"] = parse_numeric_code(frame["velocity_code"], r"v(\d+)", scale=10.0)
    frame["spin_period_hr"] = parse_numeric_code(frame["spin_code"], r"s(\d{3})", scale=10.0)
    frame["spin_axis"] = frame["spin_code"].fillna("").astype(str).str.extract(r"s\d{3}(m?)([xyz])")[1].fillna("none")
    frame["special_case_code"] = np.where(frame["mass_code"].fillna("").astype(str).str.contains("c30"), "c30", "none")
    frame["has_explicit_spin"] = frame["spin_code"].fillna("").astype(str).ne("")
    frame["target_mass_kg"] = np.power(10.0, frame["mass_log10_kg"])
    return add_physics_features(frame)


def add_physics_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    periapsis_km = enriched["periapsis_Rm"] * MARS_RADIUS_KM
    with np.errstate(divide="ignore", invalid="ignore"):
        enriched["encounter_eccentricity_proxy"] = 1.0 + (
            periapsis_km * np.square(enriched["v_inf_kms"])
        ) / MARS_MU_KM3_S2
        enriched["periapsis_inverse"] = 1.0 / enriched["periapsis_Rm"]
        enriched["spin_frequency_hr_inv"] = 1.0 / enriched["spin_period_hr"]
    enriched["v_inf_squared"] = np.square(enriched["v_inf_kms"])
    enriched["angular_momentum_proxy"] = enriched["periapsis_Rm"] * enriched["v_inf_kms"]
    enriched["has_spin"] = enriched["has_explicit_spin"].astype(int)
    enriched["particle_log10"] = np.log10(pd.to_numeric(enriched["resolution_value"], errors="coerce"))
    enriched["particle_mass_proxy"] = enriched["target_mass_kg"] / pd.to_numeric(
        enriched["resolution_value"], errors="coerce"
    )
    enriched["mass_resolution_interaction"] = enriched["mass_log10_kg"] - enriched["particle_log10"]
    with np.errstate(divide="ignore", invalid="ignore"):
        enriched["asteroid_radius_km"] = np.cbrt(
            (3.0 * enriched["target_mass_kg"]) / (4.0 * np.pi * ASTEROID_BULK_DENSITY_KG_M3)
        ) / 1000.0
    tidal_threshold_rm = FLUID_ROCHE_FACTOR * (MARS_DENSITY_KG_M3 / ASTEROID_BULK_DENSITY_KG_M3) ** (1.0 / 3.0)
    enriched["time_within_2_mars_radii_hr"] = [
        time_inside_radius_hours(float(periapsis_rm), float(velocity_kms), PROXIMITY_DISTANCE_RM)
        for periapsis_rm, velocity_kms in zip(enriched["periapsis_Rm"], enriched["v_inf_kms"])
    ]
    enriched["time_within_tidal_disruption_hr"] = [
        time_inside_radius_hours(float(periapsis_rm), float(velocity_kms), tidal_threshold_rm)
        for periapsis_rm, velocity_kms in zip(enriched["periapsis_Rm"], enriched["v_inf_kms"])
    ]
    return enriched.replace([np.inf, -np.inf], np.nan)


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


def make_bound_feature_frame(input_df: pd.DataFrame) -> pd.DataFrame:
    frame = add_derived_features(input_df)
    frame["particle_log10"] = pd.to_numeric(frame["resolution_value"], errors="coerce").map(
        lambda value: np.nan if pd.isna(value) else np.log10(value)
    )
    frame["special_case_code"] = frame.get("special_case_code", "none")
    frame["special_case_code"] = (
        pd.Series(frame["special_case_code"], index=frame.index).fillna("none").replace("", "none")
    )
    frame["target_mass_kg"] = np.power(10.0, pd.to_numeric(frame["mass_log10_kg"], errors="coerce"))
    return add_physics_features(frame)


def range_payload(series: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return {"min": float(numeric.min()), "max": float(numeric.max())}


def format_range(name: str, payload: dict[str, float]) -> str:
    if name in {"timestep", "resolution_value"}:
        return f"{payload['min']:.0f} to {payload['max']:.0f}"
    return f"{payload['min']:.4g} to {payload['max']:.4g}"


def get_input_specs() -> list[dict[str, object]]:
    return [
        {"name": "case_name", "label": "Case name", "unit": "", "type": "string"},
        {"name": "mass_log10_kg", "label": "log10 parent mass", "unit": "kg", "type": "number"},
        {"name": "periapsis_Rm", "label": "Periapsis", "unit": "Mars radii", "type": "number"},
        {"name": "v_inf_kms", "label": "Velocity", "unit": "km/s", "type": "number"},
        {"name": "has_explicit_spin", "label": "Spin on", "unit": "boolean", "type": "boolean"},
        {"name": "spin_axis", "label": "Spin axis", "unit": "axis", "type": "string"},
        {"name": "spin_period_hr", "label": "Spin period", "unit": "hr", "type": "number"},
        {"name": "resolution_value", "label": "Resolution setting", "unit": "archive code", "type": "number"},
        {"name": "timestep", "label": "Analysis time", "unit": "s", "type": "number"},
        {"name": "fof_linking_length", "label": "FoF length", "unit": "dimensionless", "type": "number"},
    ]


def get_support_thresholds() -> dict[str, float]:
    trust_summary = pd.read_csv(SURROGATE_TABLES_DIR / "trust_summary.csv").iloc[0].to_dict()
    return {
        "high_support_min": HIGH_SUPPORT_MIN,
        "moderate_support_min": MODERATE_SUPPORT_MIN,
        "spread_threshold": float(trust_summary["spread_threshold"]),
        "borderline_bmf_min": BORDERLINE_BMF_MIN,
        "borderline_bmf_max": BORDERLINE_BMF_MAX,
        "bmf_threshold": BMF_THRESHOLD,
    }


def get_selected_bound_model_metadata() -> dict[str, object]:
    selected_path = select_best_bound_model_paths().get("bound_mass_fraction")
    if selected_path is None:
        return {}
    regression_df = load_bound_metrics_tables()[1]
    if regression_df is None or regression_df.empty:
        return {"bundle_id": selected_path.stem, "path": str(selected_path)}

    parts = selected_path.stem.split("__")
    if len(parts) < 4:
        return {"bundle_id": selected_path.stem, "path": str(selected_path)}
    dataset, feature_set, target, model_name = parts[0], parts[1], parts[2], "__".join(parts[3:])
    subset = regression_df[
        (regression_df["dataset"] == dataset)
        & (regression_df["feature_set"] == feature_set)
        & (regression_df["target"] == target)
        & (regression_df["model"] == model_name)
    ]
    if subset.empty:
        return {"bundle_id": selected_path.stem, "path": str(selected_path)}
    row = subset.iloc[0]
    return {
        "bundle_id": selected_path.stem,
        "path": str(selected_path),
        "dataset": str(row["dataset"]),
        "feature_set": str(row["feature_set"]),
        "target": str(row["target"]),
        "model_name": str(row["model"]),
        "grouped_cv_r2": float(row["r2"]),
        "grouped_cv_mae_fraction": float(row["mae"]),
        "grouped_cv_mae_percentage_points": float(row["mae"]) * 100.0,
        "rows": int(row["rows"]),
        "unique_physical_files": int(row["unique_physical_files"]),
        "validation_grouping": "Group by physical SPH setup",
    }


@lru_cache(maxsize=1)
def get_fragmentation_model_metadata() -> dict[str, object]:
    return {
        "probability_screen": {
            "artifact": "fragmentation_classifier.pkl",
            "target": "qualitative fragmentation screen",
            "available": (MODEL_DIR / "fragmentation_classifier.pkl").exists(),
        },
        "largest_remnant_regression": {
            "artifact": "fragmentation_regressor.pkl",
            "target": "largest-remnant fraction / mass",
            "available": (MODEL_DIR / "fragmentation_regressor.pkl").exists(),
        },
    }


def build_validation_metadata() -> dict[str, object]:
    promoted_info = json.loads((SURROGATE_TABLES_DIR / "promoted_model_info.json").read_text(encoding="utf-8"))
    thresholds = get_support_thresholds()
    selected_bmf = get_selected_bound_model_metadata()
    return {
        "fragmentation_models": get_fragmentation_model_metadata(),
        "bmf_model": selected_bmf,
        "target_map": [
            {
                "target": "Largest-remnant fraction",
                "model_type": "regression",
                "source": "fragmentation_regressor.pkl",
                "used_in_dashboard": True,
            },
            {
                "target": "Fragmentation label",
                "model_type": "classification",
                "source": "fragmentation_classifier.pkl",
                "used_in_dashboard": True,
            },
            {
                "target": "Predicted BMF",
                "model_type": "regression",
                "source": selected_bmf.get("model_name", "bound_mass_fraction regressor"),
                "used_in_dashboard": True,
            },
            {
                "target": "Retention screen (BMF >= 10%)",
                "model_type": "threshold rule",
                "source": "Derived directly from predicted BMF; no separate deployed classifier used",
                "used_in_dashboard": True,
            },
        ],
        "hidden_classifiers": select_best_bound_classifiers(),
        "thresholds": {
            "support_score": {
                "high": f">= {HIGH_SUPPORT_MIN:.0f}",
                "moderate": f">= {MODERATE_SUPPORT_MIN:.0f} and < {HIGH_SUPPORT_MIN:.0f}",
                "low": f"< {MODERATE_SUPPORT_MIN:.0f}",
            },
            "model_spread_percentage_points": round(thresholds["spread_threshold"] * 100.0, 2),
            "borderline_bmf_percentage": [
                round(BORDERLINE_BMF_MIN * 100.0, 2),
                round(BORDERLINE_BMF_MAX * 100.0, 2),
            ],
            "bmf_threshold_percentage": round(BMF_THRESHOLD * 100.0, 1),
        },
        "limitations_note": (
            "The experimental physics-feature surrogate improved grouped-CV BMF performance, but it is not the deployed "
            "inference path for this dashboard because its promoted ablation bundle includes a post-outcome feature."
        ),
        "consistency_note": (
            "The dashboard uses regression for the continuous BMF output, then applies a visible 10% threshold rule for the "
            "retention screen. Archive classifiers for `has_any_bound_mass` and `bound_mass_fraction_ge_0_1` exist, but they are "
            "not used for the current dashboard recommendation or visible outputs."
        ),
        "experimental_reference": {
            "promotion_label": promoted_info.get("promotion_label"),
            "model_name": promoted_info.get("model_name"),
            "grouped_cv_r2": promoted_info.get("r2"),
            "grouped_cv_mae_fraction": promoted_info.get("mae"),
        },
    }


@lru_cache(maxsize=1)
def load_demo_metadata() -> dict[str, object]:
    support = load_support_frame()
    coverage_summary = pd.read_csv(SURROGATE_TABLES_DIR / "coverage_error_summary.csv").iloc[0].to_dict()
    ranges = {
        "mass_log10_kg": range_payload(support["mass_log10_kg"]),
        "periapsis_Rm": range_payload(support["periapsis_Rm"]),
        "v_inf_kms": range_payload(support["v_inf_kms"]),
        "spin_period_hr": range_payload(support["spin_period_hr"]),
        "resolution_value": range_payload(support["resolution_value"]),
        "fof_linking_length": range_payload(pd.to_numeric(support["fof_linking_length"], errors="coerce")),
        "timestep": range_payload(pd.to_numeric(support["timestep"], errors="coerce")),
    }
    defaults = {
        "case_name": "demo_case_001",
        "mass_log10_kg": 20.0,
        "periapsis_Rm": 2.0,
        "v_inf_kms": 0.0,
        "has_explicit_spin": True,
        "spin_axis": "z",
        "spin_period_hr": 3.0,
        "resolution_value": 65.0,
        "timestep": 90000.0,
        "fof_linking_length": 0.004,
    }
    return {
        "defaults": defaults,
        "ranges": ranges,
        "choices": {
            "spin_axis": ["x", "y", "z"],
        },
        "range_labels": {key: format_range(key, payload) for key, payload in ranges.items()},
        "dataset_summary": {
            "bound_rows": int(len(support)),
            "occupied_mass_peri_bins": int(coverage_summary["occupied_mass_peri_bins"]),
            "occupied_peri_vel_bins": int(coverage_summary["occupied_peri_vel_bins"]),
        },
        "scientific_semantics_note": SCIENTIFIC_SEMANTICS_NOTE,
        "input_specs": get_input_specs(),
        "support_thresholds": get_support_thresholds(),
        "model_validation": build_validation_metadata(),
        "batch_template_columns": INPUT_FIELD_ORDER,
        "batch_template_sample": defaults,
    }


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def build_input_frame(payload: dict[str, object]) -> pd.DataFrame:
    mass_log10_kg = float(payload["mass_log10_kg"])
    periapsis_rm = float(payload["periapsis_Rm"])
    v_inf_kms = float(payload["v_inf_kms"])
    resolution_value = float(payload["resolution_value"])
    timestep = float(payload["timestep"])
    fof_linking_length = float(payload["fof_linking_length"])
    has_explicit_spin = bool(payload.get("has_explicit_spin", True))
    spin_axis = str(payload.get("spin_axis", "none")) if has_explicit_spin else "none"
    spin_period_hr = payload.get("spin_period_hr")
    spin_period_hr = float(spin_period_hr) if has_explicit_spin and spin_period_hr not in (None, "") else np.nan
    mass_code = f"A{int(round(mass_log10_kg * 100)):04d}"
    resolution_code = f"n{int(round(resolution_value))}"

    row = {
        "case_name": str(payload.get("case_name", "custom_case")),
        "mass_log10_kg": mass_log10_kg,
        "mass_code": mass_code,
        "mass_value": int(round(mass_log10_kg * 100)),
        "periapsis_Rm": periapsis_rm,
        "periapsis_value": int(round(periapsis_rm * 10)),
        "v_inf_kms": v_inf_kms,
        "velocity_value": int(round(v_inf_kms * 10)),
        "spin_period_hr": spin_period_hr,
        "spin_value": int(round(spin_period_hr * 10)) if has_explicit_spin and not np.isnan(spin_period_hr) else np.nan,
        "spin_axis": spin_axis,
        "resolution_code": resolution_code,
        "resolution_value": resolution_value,
        "timestep": timestep,
        "fof_linking_length": fof_linking_length,
        "has_explicit_spin": has_explicit_spin,
        "special_case_code": "none",
    }
    return pd.DataFrame([row])


def normalize_payload(payload: dict[str, object]) -> dict[str, object]:
    normalized = dict(payload)
    normalized["case_name"] = str(normalized.get("case_name", "custom_case") or "custom_case")
    normalized["mass_log10_kg"] = float(normalized["mass_log10_kg"])
    normalized["periapsis_Rm"] = float(normalized["periapsis_Rm"])
    normalized["v_inf_kms"] = float(normalized["v_inf_kms"])
    normalized["has_explicit_spin"] = parse_bool(normalized.get("has_explicit_spin", True))
    normalized["spin_axis"] = str(normalized.get("spin_axis", "z") or "z")
    normalized["resolution_value"] = float(normalized["resolution_value"])
    normalized["timestep"] = float(normalized["timestep"])
    normalized["fof_linking_length"] = float(normalized["fof_linking_length"])
    if normalized["has_explicit_spin"]:
        normalized["spin_period_hr"] = float(normalized["spin_period_hr"])
    else:
        normalized["spin_period_hr"] = None
        normalized["spin_axis"] = "none"
    return normalized


def validate_payload(payload: dict[str, object]) -> dict[str, object]:
    required = [
        "case_name",
        "mass_log10_kg",
        "periapsis_Rm",
        "v_inf_kms",
        "resolution_value",
        "timestep",
        "fof_linking_length",
    ]
    missing = [key for key in required if key not in payload or payload[key] in ("", None)]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")
    if parse_bool(payload.get("has_explicit_spin", True)) and payload.get("spin_period_hr") in ("", None):
        raise ValueError("spin_period_hr is required when has_explicit_spin is true")
    return normalize_payload(payload)


def apply_bound_predictions(result: pd.Series, input_df: pd.DataFrame) -> pd.Series:
    models = load_bound_models()
    if not models:
        return result

    features = make_bound_feature_frame(input_df)
    for target, model in models.items():
        expected_columns = list(getattr(model.named_steps["preprocessor"], "feature_names_in_", []))
        if not expected_columns:
            continue
        X = features[expected_columns].copy()
        result[target] = float(model.predict(X)[0])

    bound_mass_fraction = float(np.clip(float(result.get("bound_mass_fraction", 0.0)), 0.0, 1.0))
    result["bound_mass_fraction"] = bound_mass_fraction
    return result


def add_spread_diagnostic(result: pd.Series, input_df: pd.DataFrame) -> pd.Series:
    spread_models = load_spread_models()
    if len(spread_models) < 2:
        result["bmf_model_spread"] = 0.0
        return result

    features = make_bound_feature_frame(input_df)
    predictions = []
    for model in spread_models.values():
        expected_columns = list(getattr(model.named_steps["preprocessor"], "feature_names_in_", []))
        X = features[expected_columns].copy()
        predictions.append(float(model.predict(X)[0]))
    result["bmf_model_spread"] = float(abs(predictions[0] - predictions[1]))
    return result


def classify_outcome(result: pd.Series) -> tuple[str, str]:
    largest_fragment_fraction = float(result.get("predicted_largest_fragment_mass_fraction", 1.0))
    fragmentation_probability = float(result.get("fragmentation_probability", 0.0))
    if fragmentation_probability >= 0.75 or largest_fragment_fraction < 0.2:
        return "Strong fragmentation", "Largest remnant is predicted to retain only a small share of the parent mass."
    if fragmentation_probability >= 0.35 or largest_fragment_fraction < 0.75:
        return "Fragmented", "Partial disruption is more likely than a clean mostly-intact remnant."
    return "Mostly intact", "The largest remnant remains dominant under the current surrogate estimate."


def classify_bmf_threshold(bound_mass_fraction: float) -> tuple[str, str]:
    if bound_mass_fraction >= BMF_THRESHOLD:
        return "Substantial bound-mass retention", "Predicted BMF exceeds the 10% retention threshold."
    return "Limited bound-mass retention", "Predicted BMF stays below the 10% retention threshold."


def describe_support_level(trust_score_pct: float) -> str:
    if trust_score_pct >= HIGH_SUPPORT_MIN:
        return "High"
    if trust_score_pct >= MODERATE_SUPPORT_MIN:
        return "Moderate"
    return "Low"


def build_support_flags(result: pd.Series, input_df: pd.DataFrame) -> dict[str, object]:
    metadata = load_demo_metadata()
    support = load_support_frame()
    row = input_df.iloc[0]
    ranges = metadata["ranges"]

    in_training_range = (
        ranges["mass_log10_kg"]["min"] <= float(row["mass_log10_kg"]) <= ranges["mass_log10_kg"]["max"]
        and ranges["periapsis_Rm"]["min"] <= float(row["periapsis_Rm"]) <= ranges["periapsis_Rm"]["max"]
        and ranges["v_inf_kms"]["min"] <= float(row["v_inf_kms"]) <= ranges["v_inf_kms"]["max"]
    )
    near_training_edge = (
        float(row["periapsis_Rm"]) <= ranges["periapsis_Rm"]["min"] + 0.1
        or float(row["periapsis_Rm"]) >= ranges["periapsis_Rm"]["max"] - 0.1
        or float(row["v_inf_kms"]) <= ranges["v_inf_kms"]["min"] + 0.1
        or float(row["v_inf_kms"]) >= ranges["v_inf_kms"]["max"] - 0.1
    )

    bin_counts = support.groupby(["mass_log10_kg", "periapsis_Rm"]).size()
    key = (float(row["mass_log10_kg"]), float(row["periapsis_Rm"]))
    bin_count = int(bin_counts.get(key, 0))
    sparse_threshold = float(bin_counts.median()) if not bin_counts.empty else 0.0
    sparse_bin_flag = bin_count <= sparse_threshold
    borderline_bmf = BORDERLINE_BMF_MIN <= float(result.get("bound_mass_fraction", 0.0)) <= BORDERLINE_BMF_MAX
    model_spread = float(result.get("bmf_model_spread", 0.0))
    spread_threshold = float(metadata["support_thresholds"]["spread_threshold"])
    return {
        "in_training_range": in_training_range,
        "near_training_edge": near_training_edge,
        "sparse_bin_flag": sparse_bin_flag,
        "bin_count": bin_count,
        "borderline_bmf": borderline_bmf,
        "model_spread": model_spread,
        "spread_threshold": spread_threshold,
    }


def make_demo_recommendation(
    predicted_outcome: str,
    bound_mass_fraction: float,
    support_flags: dict[str, object],
) -> tuple[str, str, str]:
    if not support_flags["in_training_range"]:
        return (
            "Full SPH required",
            "bad",
            "One or more core inputs sit outside the sampled training range, so this case should be treated as an extrapolation.",
        )
    if (
        support_flags["near_training_edge"]
        or support_flags["sparse_bin_flag"]
        or support_flags["model_spread"] > support_flags["spread_threshold"]
        or support_flags["borderline_bmf"]
    ):
        return (
            "SPH recommended",
            "warn",
            "The case is near the sampled edge, sparsely supported, borderline in BMF, or unstable across model families, so the surrogate is better used for prioritisation than as a stopping rule.",
        )
    if predicted_outcome == "Strong fragmentation" or bound_mass_fraction >= BMF_THRESHOLD:
        return (
            "SPH recommended",
            "warn",
            "The visible screening outputs indicate a scientifically interesting fragmentation or retained-mass signal that warrants direct SPH follow-up.",
        )
    return (
        "ML screening sufficient",
        "ok",
        "This query is in range, well supported, and does not trigger the current visible screening cautions.",
    )


def compute_support_score(support_flags: dict[str, object]) -> float:
    score = 100.0
    if not support_flags["in_training_range"]:
        score -= 45.0
    if support_flags["near_training_edge"]:
        score -= 18.0
    if support_flags["sparse_bin_flag"]:
        score -= 15.0
    if support_flags["borderline_bmf"]:
        score -= 12.0
    threshold = float(support_flags["spread_threshold"])
    if threshold > 0:
        spread_ratio = min(float(support_flags["model_spread"]) / threshold, 2.0)
        score -= spread_ratio * 10.0
    return float(np.clip(score, 5.0, 99.0))


def build_support_reason(support_flags: dict[str, object], support_level: str) -> str:
    if not support_flags["in_training_range"]:
        return "Outside the sampled training range, so this should be treated as extrapolative screening only."
    if support_flags["near_training_edge"]:
        return "In range but near the sampled edge, so the surrogate is suitable for prioritisation rather than as a stopping rule."
    if support_flags["sparse_bin_flag"]:
        return "In range but locally sparsely supported, so nearby archive evidence is limited."
    if support_flags["model_spread"] > support_flags["spread_threshold"]:
        return "In range, but model families disagree more than usual on BMF."
    if support_flags["borderline_bmf"]:
        return "In range, but the case sits near the 10% BMF decision boundary."
    return f"{support_level} support because the query is in range, not near edge, and model spread is low."


def build_support_rows(support_flags: dict[str, object], support_score: float) -> list[dict[str, str]]:
    return [
        {
            "label": "Support score",
            "value": f"{support_score:.0f}/100",
        },
        {
            "label": "Training range",
            "value": "In range" if support_flags["in_training_range"] else "Outside range",
        },
        {
            "label": "Edge status",
            "value": "Near edge" if support_flags["near_training_edge"] else "Interior",
        },
        {
            "label": "Nearby runs",
            "value": f"{support_flags['bin_count']}",
        },
        {
            "label": "Model spread",
            "value": f"{support_flags['model_spread'] * 100.0:.1f} percentage points",
        },
    ]


def build_decision_summary(
    predicted_outcome: str,
    largest_remnant_fraction: float,
    bound_mass_fraction: float,
    support_level: str,
    explanation: str,
) -> str:
    return (
        f"{predicted_outcome} predicted with a largest remnant of {largest_remnant_fraction * 100.0:.1f}%. "
        f"Predicted BMF is {bound_mass_fraction * 100.0:.1f}%. Model support is {support_level.lower()}. {explanation}"
    )


def build_export_row(
    normalized_payload: dict[str, object],
    response_payload: dict[str, object],
) -> dict[str, object]:
    return {
        "prediction_timestamp": response_payload["prediction_timestamp"],
        "model_bundle_id": response_payload["model_bundle_id"],
        "case_name": normalized_payload["case_name"],
        "mass_log10_kg": normalized_payload["mass_log10_kg"],
        "periapsis_Rm": normalized_payload["periapsis_Rm"],
        "v_inf_kms": normalized_payload["v_inf_kms"],
        "has_explicit_spin": normalized_payload["has_explicit_spin"],
        "spin_axis": normalized_payload["spin_axis"],
        "spin_period_hr": normalized_payload["spin_period_hr"],
        "resolution_value": normalized_payload["resolution_value"],
        "timestep": normalized_payload["timestep"],
        "fof_linking_length": normalized_payload["fof_linking_length"],
        "parent_mass_kg": response_payload["parent_mass_kg"],
        "largest_remnant_fraction": response_payload["largest_remnant_fraction"],
        "largest_remnant_mass_kg": response_payload["largest_remnant_mass_kg"],
        "fragmentation_label": response_payload["predicted_outcome"],
        "fragmentation_classifier_score": response_payload["fragmentation_probability"],
        "predicted_bmf": response_payload["predicted_bmf"],
        "predicted_bound_mass_kg": response_payload["predicted_bound_mass_kg"],
        "predicted_unbound_mass_fraction": response_payload["predicted_unbound_mass_fraction"],
        "predicted_unbound_mass_kg": response_payload["predicted_unbound_mass_kg"],
        "support_score": response_payload["support_score"],
        "support_category": response_payload["support_level"],
        "training_range_status": response_payload["training_range_status"],
        "edge_status": response_payload["edge_status"],
        "nearby_run_count": response_payload["nearby_run_count"],
        "model_spread_fraction": response_payload["model_spread_fraction"],
        "model_spread_percentage_points": response_payload["model_spread_percentage_points"],
        "recommendation": response_payload["recommendation"],
        "recommendation_reason": response_payload["recommendation_reason"],
        "scientific_semantics_note": response_payload["scientific_semantics_note"],
    }


def build_response_payload(result: pd.Series, input_df: pd.DataFrame, normalized_payload: dict[str, object]) -> dict[str, object]:
    selected_model = get_selected_bound_model_metadata()
    support_flags = build_support_flags(result, input_df)
    predicted_outcome, outcome_detail = classify_outcome(result)
    bound_mass_fraction = float(np.clip(float(result.get("bound_mass_fraction", 0.0)), 0.0, 1.0))
    threshold_label, threshold_detail = classify_bmf_threshold(bound_mass_fraction)
    recommendation, recommendation_style, recommendation_reason = make_demo_recommendation(
        predicted_outcome,
        bound_mass_fraction,
        support_flags,
    )
    support_score = compute_support_score(support_flags)
    support_level = describe_support_level(support_score)
    support_reason = build_support_reason(support_flags, support_level)
    parent_mass_kg = float(result.get("parent_mass_kg", np.power(10.0, float(result["mass_log10_kg"]))))
    largest_remnant_fraction = float(np.clip(float(result.get("predicted_largest_fragment_mass_fraction", 0.0)), 0.0, 1.0))
    largest_remnant_mass_kg = max(0.0, parent_mass_kg * largest_remnant_fraction)
    predicted_bound_mass_kg = max(0.0, parent_mass_kg * bound_mass_fraction)
    predicted_unbound_mass_fraction = float(np.clip(1.0 - bound_mass_fraction, 0.0, 1.0))
    predicted_unbound_mass_kg = max(0.0, parent_mass_kg * predicted_unbound_mass_fraction)
    support_frame = make_bound_feature_frame(input_df)
    domain = check_training_domain(support_frame.iloc[0].to_dict(), load_triage_bundle()[2])
    mass_semantics_warning = (
        "The dashboard does not show a stacked parent-mass decomposition because the deployed BMF target is defined on total fragment mass, "
        "not strictly on debris outside the largest remnant."
    )
    prediction_timestamp = datetime.now(UTC).replace(microsecond=0).isoformat()

    response_payload = {
        "prediction_timestamp": prediction_timestamp,
        "case_name": normalized_payload["case_name"],
        "input_payload": normalized_payload,
        "model_bundle_id": selected_model.get("bundle_id", "unknown_bundle"),
        "parent_mass_kg": parent_mass_kg,
        "largest_remnant_fraction": largest_remnant_fraction,
        "largest_remnant_percent": round(largest_remnant_fraction * 100.0, 1),
        "largest_remnant_mass_kg": largest_remnant_mass_kg,
        "predicted_outcome": predicted_outcome,
        "predicted_outcome_detail": outcome_detail,
        "fragmentation_probability": float(result.get("fragmentation_probability", 0.0)),
        "fragmentation_probability_pct": round(float(result.get("fragmentation_probability", 0.0)) * 100.0, 1),
        "predicted_bmf": bound_mass_fraction,
        "predicted_bmf_percent": round(bound_mass_fraction * 100.0, 1),
        "predicted_bound_mass_kg": predicted_bound_mass_kg,
        "predicted_unbound_mass_fraction": predicted_unbound_mass_fraction,
        "predicted_unbound_mass_percent": round(predicted_unbound_mass_fraction * 100.0, 1),
        "predicted_unbound_mass_kg": predicted_unbound_mass_kg,
        "bmf_threshold_label": threshold_label,
        "bmf_threshold_detail": threshold_detail,
        "support_score": round(support_score, 1),
        "support_level": support_level,
        "support_reason": support_reason,
        "support_rows": build_support_rows(support_flags, support_score),
        "recommendation": recommendation,
        "recommendation_style": recommendation_style,
        "recommendation_reason": recommendation_reason,
        "decision_summary": build_decision_summary(
            predicted_outcome,
            largest_remnant_fraction,
            bound_mass_fraction,
            support_level,
            recommendation_reason,
        ),
        "scientific_semantics_note": SCIENTIFIC_SEMANTICS_NOTE,
        "mass_semantics_warning": mass_semantics_warning,
        "training_range_status": "In range" if support_flags["in_training_range"] else "Outside range",
        "edge_status": "Near edge" if support_flags["near_training_edge"] else "Interior",
        "nearby_run_count": int(support_flags["bin_count"]),
        "model_spread_fraction": float(support_flags["model_spread"]),
        "model_spread_percentage_points": round(float(support_flags["model_spread"]) * 100.0, 2),
        "domain_status": domain["status"],
        "domain_near_edge_features": domain["near_edge_features"],
        "domain_out_of_domain_features": domain["out_of_domain_features"],
        "validation": load_demo_metadata()["model_validation"],
        "diagnostics_note": "Largest-remnant fraction and predicted BMF are the primary visible screening targets in this dashboard.",
    }
    response_payload["export_row"] = build_export_row(normalized_payload, response_payload)
    return response_payload


def predict_single_payload(payload: dict[str, object]) -> dict[str, object]:
    normalized_payload = validate_payload(payload)
    input_df = build_input_frame(normalized_payload)
    classifier, regressor, training_domain = load_triage_bundle()
    result = predict_cases(input_df, classifier, regressor, training_domain).iloc[0].copy()
    result = apply_bound_predictions(result, input_df)
    result = add_spread_diagnostic(result, input_df)
    return build_response_payload(result, input_df, normalized_payload)


def parse_batch_csv(csv_text: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(csv_text))
    fieldnames = reader.fieldnames or []
    missing = [name for name in INPUT_FIELD_ORDER if name not in fieldnames]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
    return [dict(row) for row in reader]


def predict_batch_csv(csv_text: str) -> dict[str, object]:
    rows = parse_batch_csv(csv_text)
    successes = []
    errors = []
    for index, row in enumerate(rows, start=2):
        try:
            result = predict_single_payload(row)
            successes.append({"row_number": index, "result": result})
        except Exception as exc:
            errors.append(
                {
                    "row_number": index,
                    "case_name": str(row.get("case_name", "")).strip(),
                    "error": str(exc),
                }
            )
    return {
        "total_rows": len(rows),
        "success_count": len(successes),
        "error_count": len(errors),
        "results": successes,
        "errors": errors,
    }


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "SPHScreeningHTTP/3.0"

    def _send_bytes(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status)

    def _read_json_body(self) -> dict[str, object]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        return json.loads(raw_body.decode("utf-8"))

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self._send_bytes(load_dashboard_html(), "text/html; charset=utf-8")
            return
        if self.path == "/api/metadata":
            self._send_json(load_demo_metadata())
            return
        if self.path == "/health":
            self._send_json({"status": "ok"})
            return
        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            if self.path == "/api/predict":
                payload = self._read_json_body()
                self._send_json(predict_single_payload(payload))
                return
            if self.path == "/api/predict-batch":
                payload = self._read_json_body()
                csv_text = str(payload.get("csv_text", ""))
                self._send_json(predict_batch_csv(csv_text))
                return
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:  # pragma: no cover
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8000), DashboardHandler)
    print("Serving SPH screening demo at http://127.0.0.1:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
