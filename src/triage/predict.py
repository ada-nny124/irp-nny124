"""Prediction helpers for the SPH fragmentation triage tool."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .decision import check_training_domain, make_sph_recommendation
from .features import add_derived_features, prepare_features


def load_artifacts(model_dir: str | Path) -> tuple[object, object, dict[str, object]] | None:
    model_dir = Path(model_dir)
    classifier_path = model_dir / "fragmentation_classifier.pkl"
    regressor_path = model_dir / "fragmentation_regressor.pkl"
    domain_path = model_dir / "training_domain.json"

    if not classifier_path.exists() or not regressor_path.exists() or not domain_path.exists():
        return None

    with classifier_path.open("rb") as handle:
        classifier = pickle.load(handle)
    with regressor_path.open("rb") as handle:
        regressor = pickle.load(handle)
    training_domain = json.loads(domain_path.read_text())
    return classifier, regressor, training_domain


def get_artifact_status(model_dir: str | Path) -> list[dict[str, str]]:
    model_dir = Path(model_dir)
    artifacts = [
        {
            "label": "fragmentation_classifier.pkl",
            "path": str(model_dir / "fragmentation_classifier.pkl"),
            "status": "loaded" if (model_dir / "fragmentation_classifier.pkl").exists() else "missing",
            "target": "fragmentation probability / is_fragmented_proxy",
        },
        {
            "label": "fragmentation_regressor.pkl",
            "path": str(model_dir / "fragmentation_regressor.pkl"),
            "status": "loaded" if (model_dir / "fragmentation_regressor.pkl").exists() else "missing",
            "target": "predicted_largest_fragment_mass_kg",
        },
        {
            "label": "training_domain.json",
            "path": str(model_dir / "training_domain.json"),
            "status": "loaded" if (model_dir / "training_domain.json").exists() else "missing",
            "target": "numeric/categorical training-domain metadata",
        },
    ]
    return artifacts


def add_severity_from_predictions(result: pd.DataFrame) -> pd.DataFrame:
    predicted_mass = pd.to_numeric(result["predicted_largest_fragment_mass_kg"], errors="coerce")
    total_mass = np.power(10.0, pd.to_numeric(result["mass_log10_kg"], errors="coerce"))
    with np.errstate(divide="ignore", invalid="ignore"):
        largest_fragment_mass_fraction = predicted_mass / total_mass
    largest_fragment_mass_fraction = largest_fragment_mass_fraction.clip(lower=0.0, upper=1.0)

    severity = pd.Series("not_available_yet", index=result.index, dtype="object")
    severity.loc[largest_fragment_mass_fraction > 0.9] = "no_or_very_weak_fragmentation"
    severity.loc[largest_fragment_mass_fraction.between(0.5, 0.9, inclusive="both")] = "weak_fragmentation"
    severity.loc[largest_fragment_mass_fraction.between(0.1, 0.5, inclusive="left")] = "moderate_fragmentation"
    severity.loc[largest_fragment_mass_fraction < 0.1] = "strong_fragmentation"
    result["parent_mass_kg"] = total_mass
    result["predicted_largest_fragment_mass_fraction"] = largest_fragment_mass_fraction
    result["severity_class"] = severity
    return result


def unavailable_reason(reason: str) -> str:
    return f"not available yet: {reason}"


def predict_cases(input_df: pd.DataFrame, classifier, regressor, training_domain: dict[str, object]) -> pd.DataFrame:
    enriched = add_derived_features(input_df)
    features = prepare_features(enriched)

    result = enriched.copy()
    result["fragmentation_probability"] = classifier.predict_proba(features)[:, 1]
    result["model_score"] = result["fragmentation_probability"].round(3)
    result["predicted_largest_fragment_mass_kg"] = regressor.predict(features)
    result = add_severity_from_predictions(result)
    result["risk_label"] = pd.cut(
        result["fragmentation_probability"],
        bins=[-np.inf, 0.25, 0.5, 0.75, np.inf],
        labels=["low", "medium", "high", "very high"],
    ).astype(str)
    result["calibration_warning"] = np.where(
        (result["fragmentation_probability"] > 0.98) | (result["fragmentation_probability"] < 0.02),
        "Probability is very close to 0 or 1 and should be treated as an uncalibrated model score rather than a precise probability.",
        "",
    )

    recommendations = []
    explanations = []
    domain_statuses = []
    out_features = []
    near_features = []
    domain_payloads = []
    for idx in result.index:
        domain = check_training_domain(features.loc[idx].to_dict(), training_domain)
        prediction = {
            "fragmentation_probability": result.loc[idx, "fragmentation_probability"],
            "model_score": result.loc[idx, "model_score"],
            "severity_class": result.loc[idx, "severity_class"],
            "predicted_largest_fragment_mass_fraction": result.loc[idx, "predicted_largest_fragment_mass_fraction"],
        }
        recommendation = make_sph_recommendation(prediction, domain)
        domain_statuses.append(domain["status"])
        out_features.append(", ".join(domain["out_of_domain_features"]))
        near_features.append(", ".join(domain["near_edge_features"]))
        recommendations.append(recommendation["recommendation"])
        explanations.append(recommendation["explanation"])
        domain_payloads.append(domain)

    result["domain_status"] = domain_statuses
    result["out_of_domain_features"] = out_features
    result["near_edge_features"] = near_features
    result["sph_recommendation"] = recommendations
    result["explanation"] = explanations
    result["domain_detail"] = domain_payloads
    result["is_fragmented_proxy"] = result["fragmentation_probability"] >= 0.5
    result["fragment_count_min_particles"] = unavailable_reason("no fragment-count regressor trained for the dashboard yet")
    result["largest_fragment_particle_count"] = unavailable_reason("no particle-count regressor trained for the dashboard yet")
    result["largest_fragment_mass_fraction"] = result["predicted_largest_fragment_mass_fraction"]
    result["has_any_bound_mass"] = unavailable_reason("no bound-mass regression score is connected to this dashboard yet")
    result["bound_mass_fraction"] = unavailable_reason("no bound-mass regressor is connected to this dashboard yet")
    result["bound_mass_fraction_ge_0p1"] = unavailable_reason("no bound-mass regression score is connected to this dashboard yet")
    result["bound_fragment_count"] = unavailable_reason("no bound-fragment-count regressor is connected to this dashboard yet")
    result["largest_bound_fragment_mass_kg"] = unavailable_reason("no largest-bound-fragment regressor is connected to this dashboard yet")
    result["average_bound_fragment_mass_kg"] = unavailable_reason("no average-bound-fragment-mass regressor is connected to this dashboard yet")
    result["bound_fragment_eccentricity"] = unavailable_reason("orbital-eccentricity targets have not been modelled yet")
    result["minimum_bound_eccentricity"] = unavailable_reason("orbital-eccentricity targets have not been modelled yet")
    result["low_eccentricity_bound_fragment_flag"] = unavailable_reason("orbital-eccentricity targets have not been modelled yet")
    result["prediction_result"] = result.apply(
        lambda row: {
            "fragmentation_probability": float(row["fragmentation_probability"]),
            "is_fragmented_proxy": bool(row["is_fragmented_proxy"]),
            "risk_label": str(row["risk_label"]),
            "predicted_largest_fragment_mass_kg": float(row["predicted_largest_fragment_mass_kg"]),
            "predicted_largest_fragment_mass_fraction": float(row["predicted_largest_fragment_mass_fraction"]),
            "severity_class": str(row["severity_class"]),
        },
        axis=1,
    )
    return result.sort_values("fragmentation_probability", ascending=False)
