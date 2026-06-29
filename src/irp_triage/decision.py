"""Decision rules for the SPH fragmentation triage tool."""

from __future__ import annotations

from typing import Any


def check_training_domain(input_row: dict[str, Any], training_domain: dict[str, Any]) -> dict[str, Any]:
    numeric_spec = training_domain.get("numeric", {})
    categorical_spec = training_domain.get("categorical", {})

    out_of_domain: list[str] = []
    near_edge: list[str] = []

    for feature_name, spec in numeric_spec.items():
        value = input_row.get(feature_name)
        if value is None:
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue

        min_value = spec.get("min")
        max_value = spec.get("max")
        if min_value is None or max_value is None:
            continue

        if numeric_value < min_value or numeric_value > max_value:
            out_of_domain.append(feature_name)
            continue

        span = max(max_value - min_value, 1e-9)
        margin = max(span * 0.1, spec.get("step_hint", 0.0))
        if numeric_value <= min_value + margin or numeric_value >= max_value - margin:
            near_edge.append(feature_name)

    for feature_name, spec in categorical_spec.items():
        value = input_row.get(feature_name)
        if value is None:
            continue
        allowed = set(spec.get("allowed", []))
        if allowed and value not in allowed:
            out_of_domain.append(feature_name)

    if out_of_domain:
        status = "out_of_domain"
    elif near_edge:
        status = "near_edge"
    else:
        status = "in_domain"

    return {
        "status": status,
        "out_of_domain_features": sorted(set(out_of_domain)),
        "near_edge_features": sorted(set(near_edge)),
    }


def make_sph_recommendation(prediction: dict[str, Any], domain_status: dict[str, Any]) -> dict[str, str]:
    status = domain_status["status"]
    fragmentation_probability = float(prediction.get("fragmentation_probability", 0.0))
    severity = prediction.get("severity_class", "unknown")
    low_periapsis_flag = bool(prediction.get("low_periapsis_flag", False))
    high_velocity_flag = bool(prediction.get("high_velocity_flag", False))

    if status == "out_of_domain":
        return {
            "recommendation": "must run SPH",
            "explanation": "One or more inputs sit outside the observed training range, so the surrogate should not be trusted without a direct SPH run.",
        }

    if low_periapsis_flag or high_velocity_flag:
        return {
            "recommendation": "must run SPH or low-res exploratory SPH first",
            "explanation": "The case sits in an extreme orbital regime where proxy predictions are less reliable and physically important outcomes can change quickly.",
        }

    if 0.4 <= fragmentation_probability <= 0.6:
        return {
            "recommendation": "run SPH because boundary case",
            "explanation": "The classifier is uncertain, so this is exactly the kind of ambiguous case that benefits from a direct simulation.",
        }

    if fragmentation_probability > 0.75:
        return {
            "recommendation": "run SPH if detailed fragment distribution or bound material matters",
            "explanation": f"The tool expects {severity.replace('_', ' ')} fragmentation in-domain, but only SPH can resolve detailed debris structure and retained bound mass.",
        }

    if fragmentation_probability < 0.25:
        return {
            "recommendation": "low priority for SPH",
            "explanation": "The proxy model predicts a low chance of fragmentation within the observed domain, so this case can usually be deprioritised.",
        }

    if status == "near_edge":
        return {
            "recommendation": "low-res SPH",
            "explanation": "The input is close to the edge of the training domain, so a cheaper exploratory SPH run is safer than relying on interpolation alone.",
        }

    return {
        "recommendation": "full SPH",
        "explanation": "The case is in-domain but not obviously low-risk, so a standard SPH follow-up remains justified.",
    }
