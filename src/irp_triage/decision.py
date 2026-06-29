"""Decision rules for the SPH fragmentation triage tool."""

from __future__ import annotations

from typing import Any


def check_training_domain(input_row: dict[str, Any], training_domain: dict[str, Any]) -> dict[str, Any]:
    numeric_spec = training_domain.get("numeric", {})
    categorical_spec = training_domain.get("categorical", {})

    out_of_domain: list[str] = []
    near_edge: list[str] = []
    numeric_details: list[dict[str, Any]] = []
    categorical_details: list[dict[str, Any]] = []

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

        span = max(max_value - min_value, 1e-9)
        relative_position = (numeric_value - min_value) / span
        detail = {
            "feature": feature_name,
            "value": numeric_value,
            "min": min_value,
            "max": max_value,
            "relative_position": relative_position,
        }

        if relative_position < 0.0 or relative_position > 1.0:
            out_of_domain.append(feature_name)
            detail["status"] = "out_of_domain"
        elif relative_position < 0.05 or relative_position > 0.95:
            near_edge.append(feature_name)
            detail["status"] = "near_edge"
        else:
            detail["status"] = "in_domain"
        numeric_details.append(detail)

    for feature_name, spec in categorical_spec.items():
        value = input_row.get(feature_name)
        if value is None:
            continue
        value_key = str(value)
        allowed = {str(item) for item in spec.get("allowed", [])}
        count_map = spec.get("counts", {})
        seen_in_training = not allowed or value_key in allowed
        count_in_training = int(count_map.get(value_key, 0)) if isinstance(count_map, dict) else 0
        detail = {
            "feature": feature_name,
            "value": value_key,
            "seen_in_training": seen_in_training,
            "count_in_training": count_in_training,
            "warning": "rare" if count_in_training > 0 and count_in_training <= 3 else "",
        }
        if allowed and value_key not in allowed:
            out_of_domain.append(feature_name)
            detail["status"] = "out_of_domain"
        else:
            detail["status"] = "in_domain"
        categorical_details.append(detail)

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
        "numeric_details": numeric_details,
        "categorical_details": categorical_details,
    }


def make_sph_recommendation(prediction: dict[str, Any], domain_status: dict[str, Any]) -> dict[str, str]:
    status = domain_status["status"]
    model_score = float(prediction.get("model_score", prediction.get("fragmentation_probability", 0.0)))
    largest_fragment_mass_fraction = prediction.get("largest_fragment_mass_fraction")
    if largest_fragment_mass_fraction is None:
        largest_fragment_mass_fraction = prediction.get("predicted_largest_fragment_mass_fraction")
    largest_fragment_mass_fraction = float(largest_fragment_mass_fraction) if largest_fragment_mass_fraction is not None else 1.0
    physical_regime_is_new = bool(prediction.get("physical_regime_is_new", False))
    needs_detailed_fragment_distribution = bool(prediction.get("needs_detailed_fragment_distribution", False))
    needs_bound_orbit_detail = bool(prediction.get("needs_bound_orbit_detail", False))
    bound_mass_fraction_ge_0p1_probability = prediction.get("bound_mass_fraction_ge_0p1_probability")

    ml_useful_when = (
        "ML is useful when the case is inside the sampled parameter space, the target is coarse "
        "(for example fragmentation yes/no or BMF >= 10%), and the goal is ranking or prioritising simulations."
    )
    sph_required_when = (
        "SPH is required when the case is out-of-domain or near a boundary, detailed fragment distribution is needed, "
        "bound orbit or eccentricity matters, the physical regime is new, or the ML model is uncertain."
    )

    if status == "out_of_domain":
        return {
            "recommendation": "must run SPH",
            "explanation": f"One or more required physical features are outside the sampled parameter space. {sph_required_when}",
        }

    if physical_regime_is_new or needs_bound_orbit_detail or needs_detailed_fragment_distribution:
        return {
            "recommendation": "must run SPH",
            "explanation": f"This use case needs physics that the coarse surrogate does not resolve directly. {sph_required_when}",
        }

    if status == "near_edge":
        return {
            "recommendation": "low-res exploratory SPH first",
            "explanation": f"The case sits near the edge of the sampled domain, so this is better treated as a boundary case than a pure interpolation problem. {sph_required_when}",
        }

    if 0.4 <= model_score <= 0.6:
        return {
            "recommendation": "run SPH because boundary case",
            "explanation": f"The classifier score is near the decision boundary, so SPH is the safer way to resolve the case. {sph_required_when}",
        }

    recommendation = ""
    explanation = ""
    if model_score > 0.75 and largest_fragment_mass_fraction < 0.5:
        recommendation = "full SPH if detailed fragment distribution or bound material matters"
        explanation = (
            "The model predicts a high fragmentation likelihood and a substantially disrupted largest remnant, "
            "which is useful for prioritisation but still not enough to replace detailed SPH debris physics."
        )
    elif model_score < 0.25:
        recommendation = "skip / low priority"
        explanation = f"The case is in-domain and looks low-risk under a coarse fragmentation target, so ML is suitable for ranking it lower. {ml_useful_when}"
    else:
        recommendation = "rank with ML, confirm with SPH if needed"
        explanation = f"This case is suitable for surrogate ranking, but any higher-fidelity orbital or fragment-physics question still belongs to SPH. {ml_useful_when}"

    if bound_mass_fraction_ge_0p1_probability is not None and float(bound_mass_fraction_ge_0p1_probability) > 0.75:
        return {
            "recommendation": "full SPH",
            "explanation": "The case is in-domain, but a strong retained-bound-mass signal would justify upgrading to a direct SPH run.",
        }

    return {"recommendation": recommendation, "explanation": explanation}
