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


def add_severity_from_predictions(result: pd.DataFrame) -> pd.DataFrame:
    predicted_mass = pd.to_numeric(result["predicted_largest_fragment_mass_kg"], errors="coerce")
    total_mass = predicted_mass.copy()
    if "mass_log10_kg" in result.columns:
        total_mass = np.power(10.0, pd.to_numeric(result["mass_log10_kg"], errors="coerce"))
    with np.errstate(divide="ignore", invalid="ignore"):
        dispersed_fraction = 1.0 - (predicted_mass / total_mass)
    dispersed_fraction = dispersed_fraction.clip(lower=0.0, upper=1.0)

    severity = pd.Series("no_fragmentation", index=result.index, dtype="object")
    fragmented = result["fragmentation_probability"] > 0.5
    severity.loc[fragmented & (dispersed_fraction < 0.1)] = "weak_fragmentation"
    severity.loc[fragmented & dispersed_fraction.between(0.1, 0.4, inclusive="left")] = "moderate_fragmentation"
    severity.loc[fragmented & (dispersed_fraction >= 0.4)] = "strong_fragmentation"
    result["severity_class"] = severity
    return result


def predict_cases(input_df: pd.DataFrame, classifier, regressor, training_domain: dict[str, object]) -> pd.DataFrame:
    enriched = add_derived_features(input_df)
    features = prepare_features(enriched)

    result = enriched.copy()
    result["fragmentation_probability"] = classifier.predict_proba(features)[:, 1]
    result["predicted_largest_fragment_mass_kg"] = regressor.predict(features)
    result = add_severity_from_predictions(result)

    recommendations = []
    explanations = []
    domain_statuses = []
    out_features = []
    near_features = []
    for idx in result.index:
        domain = check_training_domain(features.loc[idx].to_dict(), training_domain)
        prediction = {
            "fragmentation_probability": result.loc[idx, "fragmentation_probability"],
            "severity_class": result.loc[idx, "severity_class"],
            "low_periapsis_flag": result.loc[idx, "low_periapsis_flag"],
            "high_velocity_flag": result.loc[idx, "high_velocity_flag"],
        }
        recommendation = make_sph_recommendation(prediction, domain)
        domain_statuses.append(domain["status"])
        out_features.append(", ".join(domain["out_of_domain_features"]))
        near_features.append(", ".join(domain["near_edge_features"]))
        recommendations.append(recommendation["recommendation"])
        explanations.append(recommendation["explanation"])

    result["domain_status"] = domain_statuses
    result["out_of_domain_features"] = out_features
    result["near_edge_features"] = near_features
    result["sph_recommendation"] = recommendations
    result["explanation"] = explanations
    return result.sort_values("fragmentation_probability", ascending=False)
