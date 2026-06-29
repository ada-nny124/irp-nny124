from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irp_triage.decision import check_training_domain, make_sph_recommendation
from irp_triage.features import add_derived_features, prepare_features, validate_required_columns
from irp_triage.predict import add_severity_from_predictions


def test_prepare_features_adds_expected_columns():
    df = pd.DataFrame(
        [
            {
                "mass_value": 1800,
                "mass_code": "A1800",
                "periapsis_value": 12,
                "velocity_value": 8,
                "spin_value": 30,
                "spin_axis": "z",
                "has_explicit_spin": True,
                "resolution_code": "n60",
                "resolution_value": 60,
                "timestep": 90000,
                "fof_linking_length": 0.002,
            }
        ]
    )
    features = prepare_features(df)
    assert "mass_log10_kg" in features.columns
    assert "eccentricity_proxy" in features.columns
    assert "low_periapsis_flag" in features.columns
    assert features.loc[0, "mass_log10_kg"] == pytest.approx(18.0)
    assert features.loc[0, "periapsis_Rm"] == pytest.approx(1.2)
    assert features.loc[0, "v_inf_kms"] == pytest.approx(0.8)


def test_validate_required_columns_raises_for_missing_input():
    df = pd.DataFrame([{"mass_value": 1800}])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_required_columns(df, ["mass_value", "periapsis_value"])


def test_domain_check_flags_near_edge_and_out_of_domain():
    domain = {
        "numeric": {"periapsis_Rm": {"min": 1.1, "max": 3.0, "step_hint": 0.1}},
        "categorical": {"spin_axis": {"allowed": ["none", "x", "z"]}},
    }
    near = check_training_domain({"periapsis_Rm": 1.15, "spin_axis": "x"}, domain)
    assert near["status"] == "near_edge"

    out = check_training_domain({"periapsis_Rm": 3.5, "spin_axis": "y"}, domain)
    assert out["status"] == "out_of_domain"
    assert "periapsis_Rm" in out["out_of_domain_features"]
    assert "spin_axis" in out["out_of_domain_features"]


def test_recommendation_prefers_must_run_for_out_of_domain():
    recommendation = make_sph_recommendation(
        {"fragmentation_probability": 0.9, "severity_class": "strong_fragmentation"},
        {"status": "out_of_domain", "out_of_domain_features": ["periapsis_Rm"], "near_edge_features": []},
    )
    assert recommendation["recommendation"] == "must run SPH"


def test_add_derived_features_defaults_spin_axis():
    df = pd.DataFrame(
        [{"mass_value": 1800, "periapsis_value": 12, "velocity_value": 0, "resolution_code": "n60", "resolution_value": 60, "timestep": 90000, "fof_linking_length": 0.002}]
    )
    features = add_derived_features(df)
    assert features.loc[0, "spin_axis"] == "none"
    assert bool(features.loc[0, "has_explicit_spin"]) is False


def test_add_severity_from_predictions_creates_proxy_label():
    df = pd.DataFrame(
        [
            {
                "mass_log10_kg": 18.0,
                "fragmentation_probability": 0.9,
                "predicted_largest_fragment_mass_kg": 1.0e17,
            }
        ]
    )
    result = add_severity_from_predictions(df)
    assert result.loc[0, "severity_class"] == "strong_fragmentation"
