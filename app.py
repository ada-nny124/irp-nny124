"""Streamlit app for SPH fragmentation triage."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irp_triage.predict import load_artifacts, predict_cases


MODEL_DIR = ROOT / "ml" / "triage"
TEMPLATE_PATH = ROOT / "templates" / "triage_case_template.json"


@st.cache_resource
def cached_artifacts():
    return load_artifacts(MODEL_DIR)


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

    artifacts = cached_artifacts()
    if artifacts is None:
        st.error("Model artifacts are missing. Run `python scripts/train_triage_models.py` and reload the app.")
        return

    classifier, regressor, training_domain = artifacts
    st.write(f"Template file: `{TEMPLATE_PATH}`")
    st.write("Edit the template locally, then paste or upload the JSON here.")

    default_text = TEMPLATE_PATH.read_text() if TEMPLATE_PATH.exists() else "[]"
    template_text = st.text_area("Case JSON", value=default_text, height=320)
    if st.button("Run triage", type="primary"):
        input_df = pd.DataFrame(json.loads(template_text))
        result = predict_cases(input_df, classifier, regressor, training_domain)
        if len(result) == 1:
            show_prediction_summary(result.iloc[0])
        st.dataframe(result, use_container_width=True)


if __name__ == "__main__":
    main()
