"""Local web app for the Mars flyby outcome explorer UI."""

from __future__ import annotations

import argparse
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

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from triage.decision import check_training_domain
    from triage.features import add_derived_features
    from triage.predict import get_artifact_status, load_artifacts, predict_cases
else:
    from .decision import check_training_domain
    from .features import add_derived_features
    from .predict import get_artifact_status, load_artifacts, predict_cases

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "ml" / "triage"
TRAINING_ARTIFACTS_DIR = ROOT / "ml" / "trainingartifacts"
DEPLOYED_BMF_DIR = TRAINING_ARTIFACTS_DIR / "tuned_gradient_boosting"
BENCHMARK_BMF_DIR = TRAINING_ARTIFACTS_DIR / "raw_rf"
DEPLOYED_BMF_MODEL_PATH = DEPLOYED_BMF_DIR / "main_bmf_tuned_gradient_boosting.pkl"
DEPLOYED_BMF_METRICS_PATH = DEPLOYED_BMF_DIR / "main_bmf_tuned_gradient_boosting_metrics.json"
DEPLOYED_BMF_OOF_PATH = DEPLOYED_BMF_DIR / "main_bmf_tuned_gradient_boosting_oof_predictions.csv"
BENCHMARK_BMF_MODEL_PATH = BENCHMARK_BMF_DIR / "main_bmf_raw_rf.pkl"
BENCHMARK_BMF_METRICS_PATH = BENCHMARK_BMF_DIR / "main_bmf_raw_rf_metrics.json"
BENCHMARK_BMF_OOF_PATH = BENCHMARK_BMF_DIR / "main_bmf_raw_rf_oof_predictions.csv"
DATASET_PATH = ROOT / "extraction-outputs" / "tables" / "bound_outcomes.csv"
HTML_PATH = Path(__file__).resolve().parent / "templates" / "sph_triage_dashboard.html"

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
PUBLIC_TITLE = "Mars Tidal Disruption Screening Explorer"
PUBLIC_SUBTITLE = (
    "One hypothesis for the origin of Mars’s moons is that a close-flying asteroid was torn apart by Mars’s gravity, a "
    "process called tidal disruption. Some of this debris may have remained bound to Mars and later contributed to moon "
    "formation. This tool uses models trained on SPH simulations to explore which flyby conditions produce disruption and "
    "retain debris around Mars."
)
EXPORT_COLUMNS = [
    "prediction_timestamp",
    "model_bundle_id",
    "case_name",
    "mass_log10_kg",
    "periapsis_Rm",
    "encounter_eccentricity",
    "has_explicit_spin",
    "spin_axis",
    "spin_period_hr",
    "parent_mass_kg",
    "largest_remnant_fraction",
    "largest_remnant_mass_kg",
    "fragmentation_label",
    "fragmentation_classifier_score",
    "predicted_bmf",
    "bound_mass_fraction_ge_0p1",
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
    "encounter_eccentricity",
    "has_explicit_spin",
    "spin_axis",
    "spin_period_hr",
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
def load_fragmentation_metrics() -> dict[str, object]:
    metrics_path = MODEL_DIR / "metrics.json"
    largest_mass_metrics_path = MODEL_DIR / "largest_fragment_mass_kg_metrics.json"
    payload: dict[str, object] = {}
    if metrics_path.exists():
        payload["summary"] = json.loads(metrics_path.read_text(encoding="utf-8"))
    if largest_mass_metrics_path.exists():
        payload["largest_fragment_mass_kg"] = json.loads(largest_mass_metrics_path.read_text(encoding="utf-8"))
    return payload


@lru_cache(maxsize=1)
def load_bmf_model_bundle() -> dict[str, object]:
    if not DEPLOYED_BMF_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing required deployed BMF model at {DEPLOYED_BMF_MODEL_PATH}")
    with DEPLOYED_BMF_MODEL_PATH.open("rb") as handle:
        return pickle.load(handle)


@lru_cache(maxsize=1)
def load_benchmark_bmf_bundle() -> dict[str, object]:
    if not BENCHMARK_BMF_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing required benchmark BMF model at {BENCHMARK_BMF_MODEL_PATH}")
    with BENCHMARK_BMF_MODEL_PATH.open("rb") as handle:
        return pickle.load(handle)


def _load_grouped_metrics(path: Path, bundle: dict[str, object] | None = None) -> dict[str, object]:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif bundle is not None and "grouped_cv_metrics" in bundle:
        payload = dict(bundle["grouped_cv_metrics"])
    else:
        raise FileNotFoundError(f"Missing grouped metrics at {path}")
    return {
        "grouped_cv_r2": float(payload["r2"]),
        "grouped_cv_mae_fraction": float(payload["mae"]),
        "grouped_cv_mae_percentage_points": float(payload["mae"]) * 100.0,
        "grouped_cv_rmse": float(payload["rmse"]),
        "rows": int(payload["rows"]),
    }


@lru_cache(maxsize=1)
def load_bmf_metrics() -> dict[str, object]:
    bundle = load_bmf_model_bundle()
    metrics = _load_grouped_metrics(DEPLOYED_BMF_METRICS_PATH, bundle)
    return {
        "bundle_id": "tuned_gradient_boosting_bmf_v1",
        "feature_set": "raw_inputs_with_fof",
        "model_name": "Tuned Gradient Boosting",
        "grouped_cv_r2": metrics["grouped_cv_r2"],
        "grouped_cv_mae_fraction": metrics["grouped_cv_mae_fraction"],
        "grouped_cv_mae_percentage_points": metrics["grouped_cv_mae_percentage_points"],
        "grouped_cv_rmse": metrics["grouped_cv_rmse"],
        "rows": metrics["rows"],
        "unique_physical_files": int(pd.read_csv(DEPLOYED_BMF_OOF_PATH, usecols=["physical_file"])["physical_file"].nunique()),
        "feature_columns": list(bundle["feature_columns"]),
    }


@lru_cache(maxsize=1)
def load_benchmark_bmf_metrics() -> dict[str, object]:
    metrics = _load_grouped_metrics(BENCHMARK_BMF_METRICS_PATH, load_benchmark_bmf_bundle())
    return {
        "model_name": "Random Forest",
        "grouped_cv_r2": metrics["grouped_cv_r2"],
        "grouped_cv_mae_fraction": metrics["grouped_cv_mae_fraction"],
        "grouped_cv_mae_percentage_points": metrics["grouped_cv_mae_percentage_points"],
        "grouped_cv_rmse": metrics["grouped_cv_rmse"],
        "rows": metrics["rows"],
    }


@lru_cache(maxsize=1)
def load_bmf_prediction_records() -> pd.DataFrame:
    return pd.read_csv(DEPLOYED_BMF_OOF_PATH, low_memory=False)


@lru_cache(maxsize=1)
def load_benchmark_prediction_records() -> pd.DataFrame:
    return pd.read_csv(BENCHMARK_BMF_OOF_PATH, low_memory=False)


@lru_cache(maxsize=1)
def load_bmf_local_diagnostics() -> pd.DataFrame:
    deployed = load_bmf_prediction_records().copy()
    benchmark = load_benchmark_prediction_records().copy()
    deployed["predicted_bmf"] = pd.to_numeric(deployed["predicted_bmf"], errors="coerce")
    deployed["residual"] = pd.to_numeric(deployed["residual"], errors="coerce")
    benchmark["predicted_bmf"] = pd.to_numeric(benchmark["predicted_bmf"], errors="coerce")
    join_cols = ["physical_file", "fof_linking_length"]
    benchmark = benchmark[join_cols + ["predicted_bmf"]].rename(columns={"predicted_bmf": "benchmark_predicted_bmf"})
    merged = deployed.merge(benchmark, on=join_cols, how="left")
    merged["model_spread"] = (merged["predicted_bmf"] - merged["benchmark_predicted_bmf"]).abs()
    merged["absolute_error"] = merged["residual"].abs()
    group_cols = [
        "mass_log10_kg",
        "periapsis_Rm",
        "v_inf_kms",
        "spin_period_hr",
        "spin_axis",
        "resolution_value",
        "fof_linking_length",
    ]
    local = (
        merged.groupby(group_cols, dropna=False)
        .agg(
            nearby_run_count=("physical_file", "nunique"),
            local_grouped_mae=("absolute_error", "mean"),
            benchmark_disagreement_mean=("model_spread", "mean"),
        )
        .reset_index()
    )
    sparse_threshold = float(local["nearby_run_count"].median()) if not local.empty else 0.0
    local["sparse_region_flag"] = local["nearby_run_count"] <= sparse_threshold
    local["local_grouped_mae_percentage_points"] = local["local_grouped_mae"] * 100.0
    local["benchmark_disagreement_percentage_points"] = local["benchmark_disagreement_mean"] * 100.0
    local["sparse_threshold"] = sparse_threshold
    return local


@lru_cache(maxsize=1)
def load_bmf_training_domain() -> dict[str, object]:
    bundle = load_bmf_model_bundle()
    feature_columns = list(bundle["feature_columns"])
    support = load_support_frame()
    categorical_columns = [column for column in feature_columns if column in {"spin_axis", "special_case_code"}]
    numeric = {}
    for column in feature_columns:
        if column in categorical_columns:
            continue
        series = pd.to_numeric(support[column], errors="coerce").dropna()
        if series.empty:
            continue
        numeric[column] = {"min": float(series.min()), "max": float(series.max())}
    categorical = {}
    for column in categorical_columns:
        counts = support[column].fillna("missing").astype(str).value_counts().sort_index()
        categorical[column] = {"allowed": counts.index.tolist(), "counts": {key: int(value) for key, value in counts.items()}}
    return {"numeric": numeric, "categorical": categorical}


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


def eccentricity_from_periapsis_and_vinf(periapsis_rm: float, velocity_kms: float) -> float:
    periapsis_km = periapsis_rm * MARS_RADIUS_KM
    return 1.0 + (periapsis_km * (velocity_kms**2)) / MARS_MU_KM3_S2


def radius_from_mass_and_density_km(mass_kg: float, density_kg_m3: float) -> float:
    if mass_kg <= 0.0 or density_kg_m3 <= 0.0:
        return math.nan
    radius_m = ((3.0 * mass_kg) / (4.0 * math.pi * density_kg_m3)) ** (1.0 / 3.0)
    return radius_m / 1000.0


def vinf_from_periapsis_and_eccentricity(periapsis_rm: float, eccentricity: float) -> float:
    periapsis_km = periapsis_rm * MARS_RADIUS_KM
    if periapsis_km <= 0.0 or eccentricity < 1.0:
        raise ValueError("Encounter eccentricity must be at least 1.0 for this flyby parameterisation.")
    return float(math.sqrt(max(0.0, (eccentricity - 1.0) * MARS_MU_KM3_S2 / periapsis_km)))


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
        {"name": "periapsis_Rm", "label": "Closest approach", "unit": "Mars radii", "type": "number"},
        {"name": "encounter_eccentricity", "label": "Encounter eccentricity", "unit": "", "type": "number"},
        {"name": "has_explicit_spin", "label": "Spin on", "unit": "boolean", "type": "boolean"},
        {"name": "spin_axis", "label": "Spin axis", "unit": "axis", "type": "string"},
        {"name": "spin_period_hr", "label": "Spin period", "unit": "hr", "type": "number"},
        {"name": "asteroid_density_kg_m3", "label": "Asteroid density", "unit": "kg/m^3", "type": "number"},
        {"name": "asteroid_type", "label": "Asteroid type", "unit": "", "type": "string"},
    ]


def get_support_thresholds() -> dict[str, float]:
    metrics = load_bmf_metrics()
    local = load_bmf_local_diagnostics()
    disagreement_threshold = float(local["benchmark_disagreement_mean"].quantile(0.75)) if not local.empty else 0.0
    local_error_threshold = float(local["local_grouped_mae"].median()) if not local.empty else float(
        metrics.get("grouped_cv_mae_fraction", 0.0)
    )
    return {
        "high_support_min": HIGH_SUPPORT_MIN,
        "moderate_support_min": MODERATE_SUPPORT_MIN,
        "disagreement_threshold": disagreement_threshold,
        "local_error_threshold": local_error_threshold,
        "borderline_bmf_min": BORDERLINE_BMF_MIN,
        "borderline_bmf_max": BORDERLINE_BMF_MAX,
        "bmf_threshold": BMF_THRESHOLD,
    }


def get_selected_bound_model_metadata() -> dict[str, object]:
    metrics = load_bmf_metrics()
    return {
        "bundle_id": str(metrics["bundle_id"]),
        "path": str(DEPLOYED_BMF_MODEL_PATH),
        "feature_set": str(metrics["feature_set"]),
        "target": "continuous bound mass fraction",
        "model_name": str(metrics["model_name"]),
        "grouped_cv_r2": float(metrics["grouped_cv_r2"]),
        "grouped_cv_mae_fraction": float(metrics["grouped_cv_mae_fraction"]),
        "grouped_cv_mae_percentage_points": float(metrics["grouped_cv_mae_percentage_points"]),
        "grouped_cv_rmse": float(metrics["grouped_cv_rmse"]),
        "rows": int(metrics["rows"]),
        "unique_physical_files": int(metrics["unique_physical_files"]),
        "validation_grouping": "Grouped by physical SPH simulation",
        "benchmark_model_name": "Random Forest benchmark",
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
    thresholds = get_support_thresholds()
    selected_bmf = get_selected_bound_model_metadata()
    metrics = load_bmf_metrics()
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
                "source": "Tuned Gradient Boosting regressor",
                "used_in_dashboard": True,
            },
            {
                "target": "Retention screen (BMF >= 10%)",
                "model_type": "threshold rule",
                "source": "Derived directly from predicted BMF; no separate deployed classifier used",
                "used_in_dashboard": True,
            },
        ],
        "hidden_classifiers": [],
        "thresholds": {
            "support_score": {
                "high": f">= {HIGH_SUPPORT_MIN:.0f}",
                "moderate": f">= {MODERATE_SUPPORT_MIN:.0f} and < {HIGH_SUPPORT_MIN:.0f}",
                "low": f"< {MODERATE_SUPPORT_MIN:.0f}",
            },
            "model_disagreement_percentage_points": round(thresholds["disagreement_threshold"] * 100.0, 2),
            "local_grouped_mae_percentage_points": round(thresholds["local_error_threshold"] * 100.0, 2),
            "borderline_bmf_percentage": [
                round(BORDERLINE_BMF_MIN * 100.0, 2),
                round(BORDERLINE_BMF_MAX * 100.0, 2),
            ],
            "bmf_threshold_percentage": round(BMF_THRESHOLD * 100.0, 1),
        },
        "limitations_note": (
            "The tuned Gradient Boosting model is the retained report-aligned continuous BMF model. The displayed validation summary and disagreement checks "
            "refer only to the models currently wired into this demo."
        ),
        "consistency_note": (
            "The dashboard uses regression for the continuous BMF output, then applies a visible 10% threshold rule for the "
            "retention screen. The displayed typical grouped held-out absolute error is a validation average, not a case-specific "
            "confidence interval. Any benchmark comparison is labelled as model disagreement."
        ),
        "benchmark_reference": {
            "deployed_bmf_model": {
                "model_name": str(metrics["model_name"]),
                "grouped_cv_r2": float(metrics["grouped_cv_r2"]),
                "grouped_cv_mae_fraction": float(metrics["grouped_cv_mae_fraction"]),
            },
            "disagreement_benchmark": {
                "model_name": "Random Forest",
                "grouped_cv_r2": float(load_benchmark_bmf_metrics()["grouped_cv_r2"]),
                "grouped_cv_mae_fraction": float(load_benchmark_bmf_metrics()["grouped_cv_mae_fraction"]),
            },
        },
        "deployed_bmf_summary": {
            "model": str(metrics["model_name"]),
            "target": "continuous bound mass fraction",
            "validation": "grouped by physical SPH simulation",
            "grouped_cv_r2": float(metrics["grouped_cv_r2"]),
            "grouped_cv_mae_fraction": float(metrics["grouped_cv_mae_fraction"]),
            "grouped_cv_mae_percentage_points": float(metrics["grouped_cv_mae_percentage_points"]),
            "grouped_cv_rmse": float(metrics["grouped_cv_rmse"]),
            "bundle_id": str(metrics["bundle_id"]),
        },
    }


@lru_cache(maxsize=1)
def load_demo_metadata() -> dict[str, object]:
    support = load_support_frame()
    bmf_predictions = load_bmf_prediction_records()
    occupied_mass_peri_bins = int(bmf_predictions[["mass_log10_kg", "periapsis_Rm"]].drop_duplicates().shape[0])
    occupied_peri_vel_bins = int(bmf_predictions[["periapsis_Rm", "v_inf_kms"]].drop_duplicates().shape[0])
    support["encounter_eccentricity"] = support["encounter_eccentricity_proxy"]
    ranges = {
        "mass_log10_kg": range_payload(support["mass_log10_kg"]),
        "periapsis_Rm": range_payload(support["periapsis_Rm"]),
        "encounter_eccentricity": range_payload(support["encounter_eccentricity"]),
        "v_inf_kms": range_payload(support["v_inf_kms"]),
        "spin_period_hr": range_payload(support["spin_period_hr"]),
    }
    defaults = {
        "case_name": "demo_case_001",
        "input_mode": "mass",
        "mass_log10_kg": 20.0,
        "asteroid_radius_km_input": round(radius_from_mass_and_density_km(10.0**20.0, ASTEROID_BULK_DENSITY_KG_M3), 1),
        "asteroid_density_kg_m3": ASTEROID_BULK_DENSITY_KG_M3,
        "asteroid_type": "rocky",
        "periapsis_Rm": 2.0,
        "encounter_eccentricity": round(eccentricity_from_periapsis_and_vinf(2.0, 0.0), 3),
        "has_explicit_spin": True,
        "spin_axis": "z",
        "spin_period_hr": 3.0,
        "resolution_value": 65.0,
        "timestep": 90000.0,
        "fof_linking_length": 0.004,
    }
    return {
        "title": PUBLIC_TITLE,
        "subtitle": PUBLIC_SUBTITLE,
        "defaults": defaults,
        "ranges": ranges,
        "choices": {
            "spin_axis": ["x", "y", "z"],
            "asteroid_type": ["rocky"],
            "input_mode": ["mass", "size"],
        },
        "range_labels": {key: format_range(key, payload) for key, payload in ranges.items()},
        "dataset_summary": {
            "bound_rows": int(len(support)),
            "occupied_mass_peri_bins": occupied_mass_peri_bins,
            "occupied_peri_vel_bins": occupied_peri_vel_bins,
            "local_diagnostic_groups": int(load_bmf_local_diagnostics().shape[0]),
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
    encounter_eccentricity = float(payload["encounter_eccentricity"])
    v_inf_kms = vinf_from_periapsis_and_eccentricity(periapsis_rm, encounter_eccentricity)
    defaults = load_demo_metadata()["defaults"]
    resolution_value = float(payload.get("resolution_value", defaults["resolution_value"]))
    timestep = float(payload.get("timestep", defaults["timestep"]))
    fof_linking_length = float(payload.get("fof_linking_length", defaults["fof_linking_length"]))
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
        "encounter_eccentricity": encounter_eccentricity,
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
    defaults = load_demo_metadata()["defaults"]
    normalized["case_name"] = str(normalized.get("case_name", "custom_case") or "custom_case")
    normalized["input_mode"] = str(normalized.get("input_mode", defaults["input_mode"]) or defaults["input_mode"])
    normalized["mass_log10_kg"] = float(normalized["mass_log10_kg"])
    normalized["periapsis_Rm"] = float(normalized["periapsis_Rm"])
    normalized["encounter_eccentricity"] = float(normalized["encounter_eccentricity"])
    normalized["has_explicit_spin"] = parse_bool(normalized.get("has_explicit_spin", True))
    normalized["spin_axis"] = str(normalized.get("spin_axis", "z") or "z")
    normalized["asteroid_density_kg_m3"] = float(
        normalized.get("asteroid_density_kg_m3", defaults["asteroid_density_kg_m3"])
    )
    radius_input = normalized.get("asteroid_radius_km_input", defaults["asteroid_radius_km_input"])
    normalized["asteroid_radius_km_input"] = None if radius_input in (None, "") else float(radius_input)
    normalized["asteroid_type"] = str(normalized.get("asteroid_type", defaults["asteroid_type"]) or defaults["asteroid_type"])
    if normalized["asteroid_type"] != "rocky":
        raise ValueError("Only rocky asteroid type is currently supported by the trained archive.")
    normalized["resolution_value"] = float(normalized.get("resolution_value", defaults["resolution_value"]))
    normalized["timestep"] = float(normalized.get("timestep", defaults["timestep"]))
    normalized["fof_linking_length"] = float(normalized.get("fof_linking_length", defaults["fof_linking_length"]))
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
        "encounter_eccentricity",
    ]
    missing = [key for key in required if key not in payload or payload[key] in ("", None)]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")
    if parse_bool(payload.get("has_explicit_spin", True)) and payload.get("spin_period_hr") in ("", None):
        raise ValueError("spin_period_hr is required when has_explicit_spin is true")
    if float(payload["encounter_eccentricity"]) < 1.0:
        raise ValueError("Encounter eccentricity must be at least 1.0 for a Mars flyby.")
    return normalize_payload(payload)


def apply_bound_predictions(result: pd.Series, input_df: pd.DataFrame) -> pd.Series:
    bundle = load_bmf_model_bundle()
    model = bundle["pipeline"]
    features = make_bound_feature_frame(input_df)
    feature_columns = list(bundle["feature_columns"])
    bound_mass_fraction = float(np.clip(model.predict(features[feature_columns])[0], 0.0, 1.0))
    result["bound_mass_fraction"] = bound_mass_fraction
    result["has_any_bound_mass"] = bool(bound_mass_fraction > 0.0)
    result["bound_mass_fraction_ge_0p1"] = bool(bound_mass_fraction >= BMF_THRESHOLD)
    return result


def add_spread_diagnostic(result: pd.Series, input_df: pd.DataFrame) -> pd.Series:
    bundle = load_benchmark_bmf_bundle()
    benchmark = bundle["pipeline"]
    features = make_bound_feature_frame(input_df)
    feature_columns = list(bundle["feature_columns"])
    benchmark_prediction = float(benchmark.predict(features[feature_columns])[0])
    disagreement = float(abs(float(result.get("bound_mass_fraction", 0.0)) - benchmark_prediction))
    result["benchmark_random_forest_bmf"] = benchmark_prediction
    result["bmf_model_disagreement"] = disagreement
    result["bmf_model_spread"] = disagreement
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
    local = load_bmf_local_diagnostics()
    row = make_bound_feature_frame(input_df).iloc[0]
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

    local_match = local[
        np.isclose(local["mass_log10_kg"], float(row["mass_log10_kg"]))
        & np.isclose(local["periapsis_Rm"], float(row["periapsis_Rm"]))
        & np.isclose(local["v_inf_kms"], float(row["v_inf_kms"]))
        & np.isclose(local["fof_linking_length"], float(row["fof_linking_length"]))
        & np.isclose(local["resolution_value"], float(row["resolution_value"]))
        & np.isclose(
            local["spin_period_hr"].fillna(-1.0),
            float(row["spin_period_hr"]) if pd.notna(row["spin_period_hr"]) else -1.0,
        )
        & (local["spin_axis"].astype(str) == str(row["spin_axis"]))
    ]
    if local_match.empty:
        bin_count = 0
        sparse_bin_flag = True
        local_grouped_mae = float(load_bmf_metrics()["grouped_cv_mae_fraction"])
        model_spread = float(metadata["support_thresholds"]["disagreement_threshold"])
    else:
        record = local_match.iloc[0]
        bin_count = int(record["nearby_run_count"])
        sparse_bin_flag = bool(record["sparse_region_flag"])
        local_grouped_mae = float(record["local_grouped_mae"])
        model_spread = float(record["benchmark_disagreement_mean"])
    borderline_bmf = BORDERLINE_BMF_MIN <= float(result.get("bound_mass_fraction", 0.0)) <= BORDERLINE_BMF_MAX
    spread_threshold = float(metadata["support_thresholds"]["disagreement_threshold"])
    local_error_threshold = float(metadata["support_thresholds"]["local_error_threshold"])
    return {
        "in_training_range": in_training_range,
        "near_training_edge": near_training_edge,
        "sparse_bin_flag": sparse_bin_flag,
        "bin_count": bin_count,
        "borderline_bmf": borderline_bmf,
        "local_grouped_mae": local_grouped_mae,
        "local_error_threshold": local_error_threshold,
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
        or support_flags["local_grouped_mae"] > support_flags["local_error_threshold"]
        or support_flags["model_spread"] > support_flags["spread_threshold"]
        or support_flags["borderline_bmf"]
    ):
        return (
            "SPH recommended",
            "warn",
            "The case is near the sampled edge, sparsely supported, locally high-error, borderline in BMF, or unstable relative to the benchmark model, so the surrogate is better used for prioritisation than as a stopping rule.",
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
    local_threshold = float(support_flags["local_error_threshold"])
    if local_threshold > 0:
        local_error_ratio = min(float(support_flags["local_grouped_mae"]) / local_threshold, 2.0)
        score -= local_error_ratio * 8.0
    threshold = float(support_flags["spread_threshold"])
    if threshold > 0:
        spread_ratio = min(float(support_flags["model_spread"]) / threshold, 2.0)
        score -= spread_ratio * 10.0
    return float(np.clip(score, 5.0, 99.0))


def build_support_reason(support_flags: dict[str, object], support_level: str) -> str:
    if not support_flags["in_training_range"]:
        return "Outside the sampled training range, so this estimate should be treated cautiously as an extrapolation."
    if support_flags["near_training_edge"]:
        return "Inside the sampled range but near its edge, so this estimate is less secure than a well-supported interior case."
    if support_flags["sparse_bin_flag"]:
        return "Inside the sampled range but locally sparse, so there are fewer similar archive cases nearby."
    if support_flags["local_grouped_mae"] > support_flags["local_error_threshold"]:
        return "Similar archive cases show a larger grouped held-out absolute error here than in better-supported regions."
    if support_flags["model_spread"] > support_flags["spread_threshold"]:
        return "The scenario is in range, but the deployed Random Forest and the gradient-boosting benchmark disagree more than usual on bound material."
    if support_flags["borderline_bmf"]:
        return "The scenario lies near the 10% bound-material threshold, so small changes could alter the interpretation."
    return f"{support_level} support because the query is in range, not near edge, and model disagreement is low."


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
            "label": "Local grouped MAE",
            "value": f"{support_flags['local_grouped_mae'] * 100.0:.2f} percentage points",
        },
        {
            "label": "Model disagreement",
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
        "encounter_eccentricity": normalized_payload["encounter_eccentricity"],
        "has_explicit_spin": normalized_payload["has_explicit_spin"],
        "spin_axis": normalized_payload["spin_axis"],
        "spin_period_hr": normalized_payload["spin_period_hr"],
        "parent_mass_kg": response_payload["parent_mass_kg"],
        "largest_remnant_fraction": response_payload["largest_remnant_fraction"],
        "largest_remnant_mass_kg": response_payload["largest_remnant_mass_kg"],
        "fragmentation_label": response_payload["predicted_outcome"],
        "fragmentation_classifier_score": response_payload["fragmentation_probability"],
        "predicted_bmf": response_payload["predicted_bmf"],
        "bound_mass_fraction_ge_0p1": response_payload["bound_mass_fraction_ge_0p1"],
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


def build_public_summary(
    predicted_outcome: str,
    largest_remnant_fraction: float,
    bound_mass_fraction: float,
    periapsis_rm: float,
    encounter_eccentricity: float,
) -> str:
    return (
        f"This scenario is most consistent with {predicted_outcome.lower()}. "
        f"The asteroid passes Mars at {periapsis_rm:.1f} Mars radii on an encounter with eccentricity "
        f"{encounter_eccentricity:.2f}. The model estimates a largest surviving remnant of "
        f"{largest_remnant_fraction * 100.0:.1f}% of the parent body and bound retained material of "
        f"{bound_mass_fraction * 100.0:.1f}% on the deployed BMF definition."
    )


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
    asteroid_density_kg_m3 = float(normalized_payload.get("asteroid_density_kg_m3", ASTEROID_BULK_DENSITY_KG_M3))
    display_radius_km = normalized_payload.get("asteroid_radius_km_input")
    if display_radius_km in (None, ""):
        display_radius_km = radius_from_mass_and_density_km(parent_mass_kg, asteroid_density_kg_m3)
    largest_remnant_fraction = float(np.clip(float(result.get("predicted_largest_fragment_mass_fraction", 0.0)), 0.0, 1.0))
    largest_remnant_mass_kg = max(0.0, parent_mass_kg * largest_remnant_fraction)
    predicted_bound_mass_kg = max(0.0, parent_mass_kg * bound_mass_fraction)
    predicted_unbound_mass_fraction = float(np.clip(1.0 - bound_mass_fraction, 0.0, 1.0))
    predicted_unbound_mass_kg = max(0.0, parent_mass_kg * predicted_unbound_mass_fraction)
    support_frame = make_bound_feature_frame(input_df)
    domain = check_training_domain(support_frame.iloc[0].to_dict(), load_bmf_training_domain())
    fragmentation_metrics = load_fragmentation_metrics()
    bmf_mae_fraction = float(selected_model.get("grouped_cv_mae_fraction", 0.0))
    largest_mae_kg = float(
        fragmentation_metrics.get("largest_fragment_mass_kg", {}).get(
            "median_absolute_error",
            fragmentation_metrics.get("summary", {}).get("regressor", {}).get("mae", 0.0),
        )
    )
    largest_mae_fraction = float(np.clip(largest_mae_kg / parent_mass_kg, 0.0, 1.0)) if parent_mass_kg > 0.0 else 0.0
    encounter_eccentricity = float(normalized_payload["encounter_eccentricity"])
    asteroid_radius_km = float(input_df.iloc[0].get("asteroid_radius_km", support_frame.iloc[0].get("asteroid_radius_km", np.nan)))
    tidal_radius_rm = FLUID_ROCHE_FACTOR * (MARS_DENSITY_KG_M3 / ASTEROID_BULK_DENSITY_KG_M3) ** (1.0 / 3.0)
    time_within_tidal_hr = float(support_frame.iloc[0].get("time_within_tidal_disruption_hr", 0.0))
    time_within_proximity_hr = float(support_frame.iloc[0].get("time_within_2_mars_radii_hr", 0.0))
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
        "asteroid_density_kg_m3": asteroid_density_kg_m3,
        "asteroid_type": normalized_payload.get("asteroid_type", "rocky"),
        "display_radius_km": round(float(display_radius_km), 2) if display_radius_km not in (None, "") and np.isfinite(float(display_radius_km)) else None,
        "largest_remnant_fraction": largest_remnant_fraction,
        "largest_remnant_percent": round(largest_remnant_fraction * 100.0, 1),
        "largest_remnant_mass_kg": largest_remnant_mass_kg,
        "predicted_outcome": predicted_outcome,
        "predicted_outcome_detail": outcome_detail,
        "fragmentation_probability": float(result.get("fragmentation_probability", 0.0)),
        "fragmentation_probability_pct": round(float(result.get("fragmentation_probability", 0.0)) * 100.0, 1),
        "predicted_bmf": bound_mass_fraction,
        "predicted_bmf_percent": round(bound_mass_fraction * 100.0, 1),
        "bound_mass_fraction_ge_0p1": bool(bound_mass_fraction >= BMF_THRESHOLD),
        "predicted_bmf_uncertainty_fraction": bmf_mae_fraction,
        "predicted_bmf_uncertainty_percentage_points": round(bmf_mae_fraction * 100.0, 2),
        "predicted_bmf_typical_absolute_error_fraction": bmf_mae_fraction,
        "predicted_bmf_typical_absolute_error_percentage_points": round(bmf_mae_fraction * 100.0, 2),
        "predicted_bound_mass_kg": predicted_bound_mass_kg,
        "predicted_unbound_mass_fraction": predicted_unbound_mass_fraction,
        "predicted_unbound_mass_percent": round(predicted_unbound_mass_fraction * 100.0, 1),
        "predicted_unbound_mass_kg": predicted_unbound_mass_kg,
        "largest_remnant_uncertainty_fraction": largest_mae_fraction,
        "largest_remnant_uncertainty_percentage_points": round(largest_mae_fraction * 100.0, 2),
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
        "public_summary": build_public_summary(
            predicted_outcome,
            largest_remnant_fraction,
            bound_mass_fraction,
            float(normalized_payload["periapsis_Rm"]),
            encounter_eccentricity,
        ),
        "encounter_eccentricity": encounter_eccentricity,
        "encounter_velocity_kms_internal": round(float(input_df.iloc[0]["v_inf_kms"]), 3),
        "asteroid_radius_km": round(asteroid_radius_km, 2) if np.isfinite(asteroid_radius_km) else None,
        "time_within_tidal_disruption_hr": round(time_within_tidal_hr, 3),
        "time_within_2_mars_radii_hr": round(time_within_proximity_hr, 3),
        "tidal_disruption_radius_rm": round(tidal_radius_rm, 3),
        "training_range_status": "In range" if support_flags["in_training_range"] else "Outside range",
        "edge_status": "Near edge" if support_flags["near_training_edge"] else "Interior",
        "nearby_run_count": int(support_flags["bin_count"]),
        "model_spread_fraction": float(support_flags["model_spread"]),
        "model_spread_percentage_points": round(float(support_flags["model_spread"]) * 100.0, 2),
        "model_disagreement_fraction": float(support_flags["model_spread"]),
        "model_disagreement_percentage_points": round(float(support_flags["model_spread"]) * 100.0, 2),
        "domain_status": domain["status"],
        "domain_near_edge_features": domain["near_edge_features"],
        "domain_out_of_domain_features": domain["out_of_domain_features"],
        "validation": load_demo_metadata()["model_validation"],
        "diagnostics_note": "Largest-remnant fraction and predicted BMF are the primary visible public-facing outcomes in this dashboard. The BMF error figure is a typical grouped held-out absolute error from the evaluated Random Forest dashboard prototype, not a case-specific uncertainty interval.",
        "visualization": {
            "periapsis_rm": float(normalized_payload["periapsis_Rm"]),
            "encounter_eccentricity": encounter_eccentricity,
            "tidal_disruption_radius_rm": round(tidal_radius_rm, 3),
            "time_within_tidal_disruption_hr": round(time_within_tidal_hr, 3),
            "time_within_2_mars_radii_hr": round(time_within_proximity_hr, 3),
            "predicted_bmf": bound_mass_fraction,
            "largest_remnant_fraction": largest_remnant_fraction,
            "predicted_outcome": predicted_outcome,
        },
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
    server_version = "MarsFlybyExplorerHTTP/1.0"

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on. Default: 8000")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Serving Mars flyby outcome explorer at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
