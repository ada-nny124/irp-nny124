"""Streamlit app for SPH fragmentation triage."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irp_triage.decision import check_training_domain, make_sph_recommendation
from irp_triage.features import add_derived_features, prepare_features


MODEL_DIR = ROOT / "ml" / "triage"
CLASSIFIER_PATH = MODEL_DIR / "fragmentation_classifier.pkl"
REGRESSOR_PATH = MODEL_DIR / "fragmentation_regressor.pkl"
DOMAIN_PATH = MODEL_DIR / "training_domain.json"


@st.cache_resource
def load_artifacts():
    if not CLASSIFIER_PATH.exists() or not REGRESSOR_PATH.exists() or not DOMAIN_PATH.exists():
        return None
    with CLASSIFIER_PATH.open("rb") as handle:
        classifier = pickle.load(handle)
    with REGRESSOR_PATH.open("rb") as handle:
        regressor = pickle.load(handle)
    training_domain = json.loads(DOMAIN_PATH.read_text())
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


def manual_input_frame() -> pd.DataFrame:
    col1, col2 = st.columns(2)
    with col1:
        mass_log10_kg = st.number_input("Mass log10 kg", min_value=15.0, max_value=25.0, value=18.0, step=0.1)
        periapsis_rm = st.number_input("Periapsis in Mars radii", min_value=0.5, max_value=5.0, value=1.6, step=0.1)
        v_inf_kms = st.number_input("Velocity at infinity km/s", min_value=0.0, max_value=5.0, value=0.8, step=0.1)
        spin_period_hr = st.number_input("Spin period hr", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
    with col2:
        spin_axis = st.selectbox("Spin axis", ["none", "mz", "x", "y", "z"], index=0)
        resolution_code = st.selectbox("Resolution code", ["n50", "n55", "n60", "n65", "n70"], index=2)
        timestep = st.number_input("Timestep", min_value=1000.0, max_value=200000.0, value=90000.0, step=1000.0)
        fof_linking_length = st.number_input("FoF linking length", min_value=0.0001, max_value=0.05, value=0.0020, step=0.0001, format="%.4f")

    resolution_value = float(resolution_code.replace("n", ""))
    return pd.DataFrame(
        [
            {
                "mass_log10_kg": mass_log10_kg,
                "mass_code": f"A{int(round(mass_log10_kg * 100)):04d}",
                "periapsis_Rm": periapsis_rm,
                "v_inf_kms": v_inf_kms,
                "spin_period_hr": spin_period_hr if spin_axis != "none" else np.nan,
                "spin_axis": spin_axis,
                "resolution_code": resolution_code,
                "resolution_value": resolution_value,
                "timestep": timestep,
                "fof_linking_length": fof_linking_length,
                "has_explicit_spin": spin_axis != "none",
            }
        ]
    )


def show_prediction_summary(result: pd.Series) -> None:
    metrics = st.columns(5)
    metrics[0].metric("Fragmentation Probability", f"{result['fragmentation_probability']:.2%}")
    metrics[1].metric("Pred. Largest Fragment Mass", f"{result['predicted_largest_fragment_mass_kg']:.3e} kg")
    metrics[2].metric("Severity Class", result["severity_class"].replace("_", " "))
    metrics[3].metric("Domain Status", result["domain_status"])
    metrics[4].metric("SPH Recommendation", result["sph_recommendation"])
    st.write(result["explanation"])

    if result["out_of_domain_features"]:
        st.warning(f"Outside training range: {result['out_of_domain_features']}")
    elif result["near_edge_features"]:
        st.info(f"Near edge of training range: {result['near_edge_features']}")


def main() -> None:
    st.set_page_config(page_title="SPH Fragmentation Triage Tool", layout="wide")
    st.title("SPH Fragmentation Triage Tool")
    st.caption(
        "This tool predicts FoF-derived fragmentation proxy outcomes. It does not replace SPH and does not directly validate long-term capture, disk mass, or moon formation."
    )
    st.write("Use this tool to prioritise which new simulations deserve expensive SPH follow-up.")

    artifacts = load_artifacts()
    if artifacts is None:
        st.error("Model artifacts are missing. Run `python scripts/train_triage_models.py` and reload the app.")
        return

    classifier, regressor, training_domain = artifacts
    mode = st.radio("Mode", ["Single-case manual input", "Batch CSV upload"], horizontal=True)

    if mode == "Single-case manual input":
        input_df = manual_input_frame()
        if st.button("Run triage", type="primary"):
            result = predict_cases(input_df, classifier, regressor, training_domain).iloc[0]
            show_prediction_summary(result)
            st.dataframe(pd.DataFrame([result]))
    else:
        st.write(
            "CSV columns: `mass_log10_kg`, `periapsis_Rm`, `v_inf_kms`, `spin_period_hr`, `spin_axis`, `resolution_code`, `resolution_value`, `timestep`, `fof_linking_length`, `mass_code`, `has_explicit_spin`."
        )
        uploaded = st.file_uploader("Upload input CSV", type=["csv"])
        if uploaded is not None:
            batch_df = pd.read_csv(uploaded)
            result = predict_cases(batch_df, classifier, regressor, training_domain)
            st.dataframe(result, use_container_width=True)
            st.download_button("Download results CSV", result.to_csv(index=False).encode("utf-8"), file_name="triage_predictions.csv", mime="text/csv")


if __name__ == "__main__":
    main()
