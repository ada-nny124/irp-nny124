from __future__ import annotations

import json
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
from triage.dashboard import build_response_payload, compute_support_score, compute_support_score_breakdown
from triage.features import add_derived_features, prepare_features, validate_required_columns
from triage.predict import add_severity_from_predictions, get_artifact_status, load_artifacts, predict_cases
from triage import cli as triage_cli


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


def test_support_score_breakdown_sums_to_raw_final_and_matches_score():
    support_flags = {
        "in_training_range": True,
        "near_training_edge": False,
        "sparse_bin_flag": False,
        "bin_count": 8,
        "borderline_bmf": False,
        "local_grouped_mae": 0.005,
        "local_error_threshold": 0.01,
        "model_spread": 0.009,
        "spread_threshold": 0.01,
    }

    breakdown = compute_support_score_breakdown(support_flags)

    penalty_sum = sum(float(component["penalty"]) for component in breakdown["components"])
    assert breakdown["raw_final_score"] == pytest.approx(breakdown["starting_score"] + penalty_sum)
    assert breakdown["final_score"] == pytest.approx(compute_support_score(support_flags))


def test_support_score_breakdown_clips_to_min_and_max():
    low_support_flags = {
        "in_training_range": False,
        "near_training_edge": True,
        "sparse_bin_flag": True,
        "bin_count": 0,
        "borderline_bmf": True,
        "local_grouped_mae": 0.03,
        "local_error_threshold": 0.01,
        "model_spread": 0.03,
        "spread_threshold": 0.01,
    }
    high_support_flags = {
        "in_training_range": True,
        "near_training_edge": False,
        "sparse_bin_flag": False,
        "bin_count": 12,
        "borderline_bmf": False,
        "local_grouped_mae": 0.0,
        "local_error_threshold": 0.01,
        "model_spread": 0.0,
        "spread_threshold": 0.01,
    }

    low_breakdown = compute_support_score_breakdown(low_support_flags)
    high_breakdown = compute_support_score_breakdown(high_support_flags)

    assert low_breakdown["raw_final_score"] < 5.0
    assert low_breakdown["final_score"] == pytest.approx(5.0)
    assert high_breakdown["raw_final_score"] == pytest.approx(100.0)
    assert high_breakdown["final_score"] == pytest.approx(99.0)


def test_build_response_payload_exposes_support_breakdown(monkeypatch):
    support_flags = {
        "in_training_range": True,
        "near_training_edge": False,
        "sparse_bin_flag": False,
        "bin_count": 6,
        "borderline_bmf": False,
        "local_grouped_mae": 0.004,
        "local_error_threshold": 0.01,
        "model_spread": 0.007,
        "spread_threshold": 0.01,
    }

    monkeypatch.setattr("triage.dashboard.get_selected_bound_model_metadata", lambda: {"bundle_id": "demo_bundle", "grouped_cv_mae_fraction": 0.03})
    monkeypatch.setattr("triage.dashboard.build_support_flags", lambda result, input_df: support_flags)
    monkeypatch.setattr("triage.dashboard.load_bmf_training_domain", lambda: {})
    monkeypatch.setattr(
        "triage.dashboard.check_training_domain",
        lambda row, domain: {"status": "in_domain", "near_edge_features": [], "out_of_domain_features": []},
    )
    monkeypatch.setattr(
        "triage.dashboard.load_fragmentation_metrics",
        lambda: {"largest_fragment_mass_kg": {"median_absolute_error": 1.0e18}},
    )
    monkeypatch.setattr("triage.dashboard.load_demo_metadata", lambda: {"model_validation": {}, "support_thresholds": {}})

    input_df = pd.DataFrame(
        [
            {
                "mass_log10_kg": 20.0,
                "periapsis_Rm": 2.0,
                "encounter_eccentricity": 1.1,
                "asteroid_radius_km": 95.0,
                "v_inf_kms": 0.5,
            }
        ]
    )
    result = pd.Series(
        {
            "mass_log10_kg": 20.0,
            "periapsis_Rm": 2.0,
            "parent_mass_kg": 1.0e20,
            "predicted_largest_fragment_mass_fraction": 0.92,
            "bound_mass_fraction": 0.25,
            "fragmentation_probability": 0.2,
        }
    )
    payload = {
        "case_name": "demo_case",
        "mass_log10_kg": 20.0,
        "periapsis_Rm": 2.0,
        "encounter_eccentricity": 1.1,
        "has_explicit_spin": False,
        "spin_axis": "none",
        "spin_period_hr": None,
        "asteroid_density_kg_m3": 2700.0,
        "asteroid_type": "rocky",
        "resolution_value": 65.0,
        "timestep": 90000.0,
        "fof_linking_length": 0.004,
    }

    response = build_response_payload(result, input_df, payload)

    assert round(response["support_breakdown"]["final_score"], 1) == pytest.approx(response["support_score"])
    assert response["support_breakdown"]["components"][0]["label"] == "Within training range"
    assert "heuristic screening indicator" in response["support_score_warning"]
    assert response["recommendation"] == "SPH recommended"
    assert response["predicted_bmf_percent"] == pytest.approx(25.0)


def test_cli_output_includes_support_breakdown_and_warning(tmp_path, monkeypatch, capsys):
    payload = {
        "predicted_bmf": 0.25,
        "predicted_bound_mass_kg": 2.5e19,
        "predicted_outcome": "Mostly intact",
        "support_level": "High",
        "support_score": 87.0,
        "recommendation": "ML screening sufficient",
        "recommendation_reason": "This query is in range, well supported, and does not trigger the current visible screening cautions.",
        "support_score_warning": (
            "This support score of 87.0/100 is a heuristic screening indicator. "
            "It is not a probability that the prediction is correct, a confidence interval, or a calibrated uncertainty estimate. "
            "The underlying diagnostics such as training-domain status, local SPH coverage, held-out error, and model disagreement should be interpreted individually."
        ),
        "support_breakdown": {
            "starting_score": 100.0,
            "final_score": 87.0,
            "components": [
                {"label": "Within training range", "penalty": 0.0, "diagnostic": "Within sampled range"},
                {"label": "Not near training edge", "penalty": 0.0, "diagnostic": "Interior case"},
                {"label": "Local SPH support", "penalty": 0.0, "diagnostic": "8 nearby independent SPH runs"},
                {"label": "Not near 10% BMF threshold", "penalty": 0.0, "diagnostic": None},
                {"label": "Local held-out error", "penalty": -4.0, "diagnostic": "1.8 pp"},
                {"label": "Model disagreement", "penalty": -9.0, "diagnostic": "2.3 pp"},
            ],
            "diagnostics": {
                "nearby_independent_sph_runs": 8,
                "local_grouped_held_out_mae_percentage_points": 1.8,
                "gb_rf_disagreement_percentage_points": 2.3,
            },
        },
        "export_row": {
            "case_name": "demo_case",
            "predicted_bmf": 0.25,
            "predicted_bound_mass_kg": 2.5e19,
            "fragmentation_label": "Mostly intact",
            "support_category": "High",
            "support_score": 87.0,
            "recommendation": "ML screening sufficient",
            "recommendation_reason": "This query is in range, well supported, and does not trigger the current visible screening cautions.",
        },
    }

    monkeypatch.setattr(triage_cli, "predict_single_payload", lambda case: payload)
    output_path = tmp_path / "predictions.csv"
    input_path = tmp_path / "case.json"
    input_path.write_text(
        json.dumps(
            {
                "case_name": "demo_case",
                "mass_log10_kg": 20.0,
                "periapsis_Rm": 2.0,
                "encounter_eccentricity": 1.1,
                "has_explicit_spin": False,
                "spin_axis": "none",
                "spin_period_hr": None,
                "asteroid_density_kg_m3": 2700.0,
                "asteroid_type": "rocky",
                "resolution_value": 65.0,
                "timestep": 90000.0,
                "fof_linking_length": 0.004,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["triage-cli", "--input", str(input_path), "--output", str(output_path)])
    triage_cli.main()
    stdout = capsys.readouterr().out

    assert "Model support: High (87/100)" in stdout
    assert "Support breakdown:" in stdout
    assert "Starting score" in stdout
    assert "Local held-out error (1.8 pp)" in stdout
    assert "GB-RF disagreement: 2.3 percentage points" in stdout
    assert "heuristic screening indicator" in stdout
