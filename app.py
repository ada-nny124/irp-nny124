"""Streamlit dashboard for SPH fragmentation triage."""

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


@st.cache_data
def load_template_cases() -> list[dict[str, object]]:
    if not TEMPLATE_PATH.exists():
        return []
    data = json.loads(TEMPLATE_PATH.read_text())
    return data if isinstance(data, list) else [data]


def ensure_session_case() -> None:
    template_cases = load_template_cases()
    default_case = template_cases[0] if template_cases else {}
    for key, value in default_case.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("case_selector", default_case.get("case_name", "Custom case"))


def load_case_into_session(case_name: str) -> None:
    for case in load_template_cases():
        if case.get("case_name") != case_name:
            continue
        for key, value in case.items():
            st.session_state[key] = value
        st.session_state["case_selector"] = case_name
        return


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Scenario")
        template_cases = load_template_cases()
        case_names = [case.get("case_name", f"Case {idx + 1}") for idx, case in enumerate(template_cases)]
        if case_names:
            selected_case = st.selectbox(
                "Start from template",
                options=case_names,
                key="case_selector",
                on_change=lambda: load_case_into_session(st.session_state["case_selector"]),
            )
            if st.button("Reload Template Values"):
                load_case_into_session(selected_case)

        st.caption("Template source")
        st.code(str(TEMPLATE_PATH), language="text")
        st.caption(
            "This tool predicts FoF-derived fragmentation proxy outcomes. It does not replace SPH and does not directly validate long-term capture, disk mass, or moon formation."
        )


def render_form() -> pd.DataFrame:
    st.subheader("Simulation Inputs")
    with st.form("triage_inputs"):
        col1, col2, col3 = st.columns(3)

        with col1:
            case_name = st.text_input("Case name", value=st.session_state.get("case_name", "custom_case"))
            mass_log10_kg = st.number_input(
                "Mass log10 kg",
                min_value=15.0,
                max_value=25.0,
                value=float(st.session_state.get("mass_log10_kg", 18.0)),
                step=0.1,
            )
            periapsis_rm = st.number_input(
                "Periapsis in Mars radii",
                min_value=0.5,
                max_value=5.0,
                value=float(st.session_state.get("periapsis_Rm", 1.2)),
                step=0.1,
            )
            v_inf_kms = st.number_input(
                "Velocity at infinity km/s",
                min_value=0.0,
                max_value=5.0,
                value=float(st.session_state.get("v_inf_kms", 0.8)),
                step=0.1,
            )

        with col2:
            spin_axis_options = ["none", "mz", "x", "y", "z"]
            spin_axis_value = st.session_state.get("spin_axis", "none")
            spin_axis = st.selectbox(
                "Spin axis",
                options=spin_axis_options,
                index=spin_axis_options.index(spin_axis_value) if spin_axis_value in spin_axis_options else 0,
            )
            spin_period_hr = st.number_input(
                "Spin period hr",
                min_value=0.0,
                max_value=20.0,
                value=float(st.session_state.get("spin_period_hr", 0.0) or 0.0),
                step=0.1,
                disabled=spin_axis == "none",
            )
            resolution_options = ["n50", "n55", "n60", "n65", "n70"]
            resolution_value = st.session_state.get("resolution_code", "n60")
            resolution_code = st.selectbox(
                "Resolution code",
                options=resolution_options,
                index=resolution_options.index(resolution_value) if resolution_value in resolution_options else 2,
            )
            timestep = st.number_input(
                "Timestep",
                min_value=1000.0,
                max_value=200000.0,
                value=float(st.session_state.get("timestep", 90000.0)),
                step=1000.0,
            )

        with col3:
            fof_linking_length = st.number_input(
                "FoF linking length",
                min_value=0.0001,
                max_value=0.05,
                value=float(st.session_state.get("fof_linking_length", 0.002)),
                step=0.0001,
                format="%.4f",
            )
            has_explicit_spin = st.toggle(
                "Explicit spin enabled",
                value=bool(st.session_state.get("has_explicit_spin", spin_axis != "none")),
            )
            mass_code = st.text_input(
                "Mass code",
                value=st.session_state.get("mass_code", f"A{int(round(mass_log10_kg * 100)):04d}"),
            )
            resolution_numeric = float(resolution_code.replace("n", ""))
            st.metric("Resolution value", f"{int(resolution_numeric)}")

        submitted = st.form_submit_button("Run triage", type="primary")

    if not submitted and "latest_input_df" in st.session_state:
        return st.session_state["latest_input_df"]

    input_df = pd.DataFrame(
        [
            {
                "case_name": case_name,
                "mass_log10_kg": mass_log10_kg,
                "mass_code": mass_code,
                "periapsis_Rm": periapsis_rm,
                "v_inf_kms": v_inf_kms,
                "spin_period_hr": spin_period_hr if spin_axis != "none" and has_explicit_spin else None,
                "spin_axis": spin_axis,
                "resolution_code": resolution_code,
                "resolution_value": resolution_numeric,
                "timestep": timestep,
                "fof_linking_length": fof_linking_length,
                "has_explicit_spin": has_explicit_spin and spin_axis != "none",
            }
        ]
    )

    st.session_state["latest_input_df"] = input_df
    return input_df


def recommendation_tone(recommendation: str) -> str:
    if "must run" in recommendation:
        return "error"
    if "boundary case" in recommendation or "full SPH" in recommendation:
        return "warning"
    return "success"


def render_recommendation_panel(result: pd.Series) -> None:
    st.subheader("Decision Recommendation")
    tone = recommendation_tone(result["sph_recommendation"])
    message = f"**{result['sph_recommendation'].upper()}**\n\n{result['explanation']}"
    if tone == "error":
        st.error(message)
    elif tone == "warning":
        st.warning(message)
    else:
        st.success(message)


def render_metric_row(result: pd.Series) -> None:
    st.subheader("Prediction Summary")
    metric_cols = st.columns(5)
    metric_cols[0].metric("Fragmentation Probability", f"{result['fragmentation_probability']:.1%}")
    metric_cols[1].metric("Pred. Largest Fragment Mass", f"{result['predicted_largest_fragment_mass_kg']:.3e} kg")
    metric_cols[2].metric("Severity", result["severity_class"].replace("_", " "))
    metric_cols[3].metric("Domain Status", result["domain_status"].replace("_", " "))
    metric_cols[4].metric("Spin Active", "Yes" if bool(result["has_explicit_spin"]) else "No")


def render_probability_chart(result: pd.Series) -> None:
    st.subheader("Risk View")
    risk_df = pd.DataFrame(
        {
            "Outcome": ["Fragmentation", "Non-fragmentation"],
            "Probability": [
                float(result["fragmentation_probability"]),
                max(0.0, 1.0 - float(result["fragmentation_probability"])),
            ],
        }
    )
    st.bar_chart(risk_df.set_index("Outcome"), height=260)


def render_parameter_chart(result: pd.Series, training_domain: dict[str, object]) -> None:
    st.subheader("Parameter Position vs Training Domain")
    rows: list[dict[str, object]] = []
    tracked = ["mass_log10_kg", "periapsis_Rm", "v_inf_kms", "spin_period_hr", "timestep", "fof_linking_length"]
    numeric_domain = training_domain.get("numeric", {})
    for feature_name in tracked:
        spec = numeric_domain.get(feature_name)
        if not spec:
            continue
        min_value = spec["min"]
        max_value = spec["max"]
        current_value = result.get(feature_name)
        if pd.isna(current_value):
            continue
        span = max(max_value - min_value, 1e-9)
        rows.append(
            {
                "feature": feature_name,
                "current": float(current_value),
                "min": float(min_value),
                "max": float(max_value),
                "relative_position": (float(current_value) - float(min_value)) / span,
            }
        )

    if not rows:
        st.info("No numeric training-domain information available.")
        return

    chart_df = pd.DataFrame(rows).set_index("feature")
    st.bar_chart(chart_df[["relative_position"]], height=280)
    st.dataframe(chart_df, use_container_width=True)


def render_flag_panel(result: pd.Series) -> None:
    st.subheader("Domain Flags")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Out-of-domain features")
        if result["out_of_domain_features"]:
            st.code(result["out_of_domain_features"], language="text")
        else:
            st.caption("None")
    with col2:
        st.write("Near-edge features")
        if result["near_edge_features"]:
            st.code(result["near_edge_features"], language="text")
        else:
            st.caption("None")


def render_template_comparison(classifier, regressor, training_domain: dict[str, object]) -> None:
    template_cases = load_template_cases()
    if not template_cases:
        return
    st.subheader("Template Scenario Comparison")
    template_df = pd.DataFrame(template_cases)
    result = predict_cases(template_df, classifier, regressor, training_domain)
    comparison = result[
        [
            "case_name",
            "fragmentation_probability",
            "predicted_largest_fragment_mass_kg",
            "severity_class",
            "domain_status",
            "sph_recommendation",
        ]
    ].copy()
    st.dataframe(comparison, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="SPH Fragmentation Triage Dashboard", layout="wide")
    ensure_session_case()
    render_sidebar()

    st.title("SPH Fragmentation Triage Dashboard")
    st.write("Decision-support dashboard for screening proposed SPH runs using FoF-derived proxy models.")
    st.caption("Use this tool to prioritise which new simulations deserve expensive SPH follow-up.")

    artifacts = cached_artifacts()
    if artifacts is None:
        st.error("Model artifacts are missing. Run `python scripts/train_triage_models.py` first.")
        return

    classifier, regressor, training_domain = artifacts
    input_df = render_form()
    if input_df.empty:
        return

    result = predict_cases(input_df, classifier, regressor, training_domain).iloc[0]
    render_metric_row(result)

    col1, col2 = st.columns([1.2, 1.0])
    with col1:
        render_recommendation_panel(result)
        render_probability_chart(result)
    with col2:
        render_flag_panel(result)

    render_parameter_chart(result, training_domain)
    render_template_comparison(classifier, regressor, training_domain)


if __name__ == "__main__":
    main()
