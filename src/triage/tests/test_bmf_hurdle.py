from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scripts import app as dashboard_app
from scripts.eda.train_deployed_bmf_hurdle import safe_feature_columns
from triage.bmf import LEAKY_FEATURES, load_bmf_bundle, predict_bmf_from_bundle


class DummyZeroPositiveClassifier:
    def predict_proba(self, X):
        signal = pd.to_numeric(X["mass_log10_kg"], errors="coerce").fillna(18.0).to_numpy()
        positive = np.clip((signal - 18.0) / 2.0, 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])


class DummyPositiveOnlyRegressor:
    def predict(self, X):
        periapsis = pd.to_numeric(X["periapsis_Rm"], errors="coerce").fillna(2.0).to_numpy()
        estimate = 0.35 - 0.08 * (periapsis - 1.5)
        return np.clip(estimate, 0.0, 1.0)


class DummyBenchmarkRegressor:
    def predict(self, X):
        mass = pd.to_numeric(X["mass_log10_kg"], errors="coerce").fillna(18.0).to_numpy()
        return np.clip(0.05 + 0.02 * (mass - 18.0), 0.0, 1.0)


def make_dummy_bundle():
    return {
        "bundle_id": "test_bundle",
        "model_name": "two-stage CatBoost hurdle",
        "feature_set": "with_fof_linking_length",
        "feature_columns": ["mass_log10_kg", "periapsis_Rm", "spin_axis"],
        "categorical_columns": ["spin_axis"],
        "zero_vs_positive_classifier": DummyZeroPositiveClassifier(),
        "positive_only_regressor": DummyPositiveOnlyRegressor(),
        "benchmark_random_forest": DummyBenchmarkRegressor(),
        "training_domain": {
            "numeric": {
                "mass_log10_kg": {"min": 18.0, "max": 21.0},
                "periapsis_Rm": {"min": 1.1, "max": 3.0},
            },
            "categorical": {"spin_axis": {"allowed": ["none", "x", "y", "z"], "counts": {"z": 3}}},
        },
        "metrics": {
            "model_name": "two-stage CatBoost hurdle",
            "bundle_id": "test_bundle",
            "grouped_cv_r2": 0.94,
            "grouped_cv_mae_fraction": 0.0123,
            "grouped_cv_mae_percentage_points": 1.23,
            "grouped_cv_rmse": 0.021,
        },
    }


def make_feature_frame():
    return pd.DataFrame(
        [
            {"mass_log10_kg": 20.0, "periapsis_Rm": 1.5, "spin_axis": "z"},
            {"mass_log10_kg": 18.5, "periapsis_Rm": 2.4, "spin_axis": "none"},
        ]
    )


def test_safe_feature_columns_exclude_leaky_features():
    columns = safe_feature_columns()
    assert not (LEAKY_FEATURES & set(columns))
    assert "largest_fragment_mass_fraction" not in columns


def test_load_bmf_bundle_requires_existing_artifact(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing required BMF CatBoost hurdle bundle"):
        load_bmf_bundle(tmp_path)


def test_load_bmf_bundle_rejects_leaky_feature(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    bundle = make_dummy_bundle()
    bundle["feature_columns"] = bundle["feature_columns"] + ["largest_fragment_mass_fraction"]
    with (model_dir / "bmf_hurdle_bundle.pkl").open("wb") as handle:
        pickle.dump(bundle, handle)
    with pytest.raises(ValueError, match="Leaky deployed BMF bundle"):
        load_bmf_bundle(model_dir)


def test_hurdle_prediction_is_product_and_clipped():
    bundle = make_dummy_bundle()
    frame = make_feature_frame()
    prediction = predict_bmf_from_bundle(bundle, frame)
    expected = np.clip(prediction.positive_probability * prediction.positive_estimate, 0.0, 1.0)
    assert np.all(np.isfinite(prediction.final_prediction))
    assert np.all(prediction.final_prediction >= 0.0)
    assert np.all(prediction.final_prediction <= 1.0)
    assert np.allclose(prediction.final_prediction, expected)


def test_hurdle_predictions_are_deterministic():
    bundle = make_dummy_bundle()
    frame = make_feature_frame()
    first = predict_bmf_from_bundle(bundle, frame)
    second = predict_bmf_from_bundle(bundle, frame)
    assert np.allclose(first.positive_probability, second.positive_probability)
    assert np.allclose(first.positive_estimate, second.positive_estimate)
    assert np.allclose(first.final_prediction, second.final_prediction)


def test_retention_screen_is_derived_from_predicted_bmf(monkeypatch):
    class DummyRfModel:
        feature_names_in_ = ["mass_log10_kg", "periapsis_Rm", "spin_axis"]

        def predict(self, X):
            return np.array([0.21])

    monkeypatch.setattr(dashboard_app, "load_rf_bmf_model", lambda: DummyRfModel())
    payload = {
        "case_name": "demo_test",
        "input_mode": "mass",
        "mass_log10_kg": 20.0,
        "periapsis_Rm": 1.5,
        "encounter_eccentricity": 1.0,
        "has_explicit_spin": True,
        "spin_axis": "z",
        "spin_period_hr": 3.0,
        "asteroid_density_kg_m3": 2700.0,
        "asteroid_type": "rocky",
        "resolution_value": 65.0,
        "timestep": 90000.0,
        "fof_linking_length": 0.004,
    }
    input_df = dashboard_app.build_input_frame(payload)
    result = pd.Series(
        {
            "mass_log10_kg": payload["mass_log10_kg"],
            "predicted_largest_fragment_mass_fraction": 0.25,
            "fragmentation_probability": 0.8,
            "parent_mass_kg": 10.0 ** payload["mass_log10_kg"],
        }
    )
    updated = dashboard_app.apply_bound_predictions(result, input_df)
    assert updated["bound_mass_fraction_ge_0p1"] == (updated["bound_mass_fraction"] >= dashboard_app.BMF_THRESHOLD)


def test_dashboard_metadata_reports_catboost_hurdle(monkeypatch):
    monkeypatch.setattr(
        dashboard_app,
        "load_rf_bmf_metrics",
        lambda: {
            "bundle_id": "rf_dashboard_bmf_v1",
            "feature_set": "with_fof_linking_length",
            "model_name": "Random Forest",
            "grouped_cv_r2": 0.8971,
            "grouped_cv_mae_fraction": 0.01839,
            "grouped_cv_mae_percentage_points": 1.839,
            "grouped_cv_rmse": 0.02984,
            "rows": 407,
            "unique_physical_files": 279,
        },
    )
    monkeypatch.setattr(
        dashboard_app,
        "load_rf_local_diagnostics",
        lambda: pd.DataFrame({"local_grouped_mae": [0.01, 0.015], "benchmark_disagreement_mean": [0.005, 0.02]}),
    )
    metadata = dashboard_app.build_validation_metadata()
    assert metadata["bmf_model"]["model_name"] == "Random Forest"
    assert metadata["deployed_bmf_summary"]["grouped_cv_mae_percentage_points"] == pytest.approx(1.839)
    assert metadata["benchmark_reference"]["future_deployment_candidate"]["model_name"] == "Two-stage CatBoost hurdle"
    assert metadata["consistency_note"].lower().find("case-specific confidence interval") != -1


def test_batch_prediction_wrapper_still_returns_rows(monkeypatch):
    monkeypatch.setattr(
        dashboard_app,
        "predict_single_payload",
        lambda row: {"case_name": row["case_name"], "predicted_bmf": 0.12},
    )
    csv_text = (
        "case_name,mass_log10_kg,periapsis_Rm,encounter_eccentricity,has_explicit_spin,spin_axis,spin_period_hr\n"
        "demo_case,20.0,2.0,1.0,true,z,3.0\n"
    )
    result = dashboard_app.predict_batch_csv(csv_text)
    assert result["success_count"] == 1
    assert result["error_count"] == 0
    assert result["results"][0]["result"]["predicted_bmf"] == pytest.approx(0.12)
