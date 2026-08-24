from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from triage.decision import check_training_domain, make_sph_recommendation
from triage.features import add_derived_features, prepare_features, validate_required_columns
from triage.predict import add_severity_from_predictions, get_artifact_status, load_artifacts, predict_cases


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
        "categorical": {"spin_axis": {"allowed": ["none", "x", "z"], "counts": {"x": 5, "z": 2}}},
    }
    near = check_training_domain({"periapsis_Rm": 1.15, "spin_axis": "x"}, domain)
    assert near["status"] == "near_edge"
    assert near["numeric_details"][0]["status"] == "near_edge"

    out = check_training_domain({"periapsis_Rm": 3.5, "spin_axis": "y"}, domain)
    assert out["status"] == "out_of_domain"
    assert "periapsis_Rm" in out["out_of_domain_features"]
    assert "spin_axis" in out["out_of_domain_features"]


def test_recommendation_prefers_must_run_for_out_of_domain():
    recommendation = make_sph_recommendation(
        {"fragmentation_probability": 0.9, "predicted_largest_fragment_mass_fraction": 0.05},
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
    assert result.loc[0, "severity_class"] == "moderate_fragmentation"
    assert result.loc[0, "predicted_largest_fragment_mass_fraction"] == pytest.approx(0.1)


class DummyFragmentationClassifier:
    def predict_proba(self, X):
        values = pd.to_numeric(X["mass_log10_kg"], errors="coerce").fillna(18.0).to_numpy()
        positive = (values - 18.0) / 2.0
        positive = positive.clip(0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])


class DummyFragmentationRegressor:
    def predict(self, X):
        mass_log10 = pd.to_numeric(X["mass_log10_kg"], errors="coerce").fillna(18.0).to_numpy()
        return np.power(10.0, mass_log10 - 0.2)


def test_fragmentation_predictions_still_work_with_separate_models():
    df = pd.DataFrame(
        [
            {
                "mass_value": 2000,
                "mass_code": "A2000",
                "periapsis_value": 15,
                "velocity_value": 0,
                "spin_value": 30,
                "spin_axis": "z",
                "has_explicit_spin": True,
                "resolution_code": "n65",
                "resolution_value": 65,
                "timestep": 90000,
                "fof_linking_length": 0.004,
            }
        ]
    )
    training_domain = {
        "numeric": {
            "mass_log10_kg": {"min": 18.0, "max": 21.0, "step_hint": 0.5},
            "periapsis_Rm": {"min": 1.1, "max": 3.0, "step_hint": 0.1},
        },
        "categorical": {"spin_axis": {"allowed": ["none", "x", "y", "z"], "counts": {"z": 1}}},
    }
    result = predict_cases(df, DummyFragmentationClassifier(), DummyFragmentationRegressor(), training_domain)
    assert "fragmentation_probability" in result.columns
    assert "predicted_largest_fragment_mass_fraction" in result.columns
    assert "severity_class" in result.columns
    assert result.loc[result.index[0], "fragmentation_probability"] == pytest.approx(1.0)


def test_artifact_status_reports_fragmentation_artifacts():
    labels = {item["label"] for item in get_artifact_status(PROJECT_ROOT / "ml" / "triage")}
    assert "fragmentation_classifier.pkl" in labels
    assert "fragmentation_regressor.pkl" in labels
    assert "training_domain.json" in labels


def test_load_artifacts_raises_clear_error_for_incompatible_sklearn_pickle(tmp_path, monkeypatch):
    model_dir = tmp_path / "triage"
    model_dir.mkdir()
    for name in ("fragmentation_classifier.pkl", "fragmentation_regressor.pkl"):
        (model_dir / name).write_bytes(b"placeholder")
    (model_dir / "training_domain.json").write_text("{}", encoding="utf-8")

    def fake_load(handle):
        path = Path(handle.name)
        if path.name == "fragmentation_regressor.pkl":
            raise ModuleNotFoundError("No module named '_loss'", name="_loss")
        return object()

    monkeypatch.setattr("triage.predict.pickle.load", fake_load)

    with pytest.raises(RuntimeError, match="Incompatible scikit-learn model artifact"):
        load_artifacts(model_dir)
