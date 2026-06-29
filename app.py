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

from irp_triage import load_artifacts, predict_cases

try:
    from irp_triage import get_artifact_status
except ImportError:
    def get_artifact_status(model_dir):
        model_dir = Path(model_dir)
        return [
            {
                "label": "fragmentation_classifier.pkl",
                "path": str(model_dir / "fragmentation_classifier.pkl"),
                "status": "loaded" if (model_dir / "fragmentation_classifier.pkl").exists() else "missing",
                "target": "fragmentation probability / is_fragmented_proxy",
            },
            {
                "label": "fragmentation_regressor.pkl",
                "path": str(model_dir / "fragmentation_regressor.pkl"),
                "status": "loaded" if (model_dir / "fragmentation_regressor.pkl").exists() else "missing",
                "target": "predicted_largest_fragment_mass_kg",
            },
            {
                "label": "bound_mass_classifier.pkl",
                "path": str(model_dir / "bound_mass_classifier.pkl"),
                "status": "loaded" if (model_dir / "bound_mass_classifier.pkl").exists() else "missing",
                "target": "has_any_bound_mass",
            },
            {
                "label": "bmf_ge_0p1_classifier.pkl",
                "path": str(model_dir / "bmf_ge_0p1_classifier.pkl"),
                "status": "loaded" if (model_dir / "bmf_ge_0p1_classifier.pkl").exists() else "missing",
                "target": "bound_mass_fraction_ge_0p1",
            },
            {
                "label": "training_domain.json",
                "path": str(model_dir / "training_domain.json"),
                "status": "loaded" if (model_dir / "training_domain.json").exists() else "missing",
                "target": "numeric/categorical training-domain metadata",
            },
        ]


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
        st.markdown("**Why ML?**")
        st.caption(
            "ML is useful when the case is inside the sampled parameter space, the target is coarse, and the goal is ranking or prioritising simulations."
        )
        st.markdown("**Why SPH?**")
        st.caption(
            "SPH is still required for out-of-domain or boundary cases, detailed fragment physics, bound orbit questions, new physical regimes, and uncertain model outputs."
        )
        st.markdown("**Current model scope**")
        st.caption(
            "Currently active predictions: fragmentation probability and largest-fragment mass. Bound-retention and orbital-eccentricity targets are placeholders until their models are trained or connected."
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
            has_explicit_spin = st.toggle(
                "Explicit spin enabled",
                value=bool(st.session_state.get("has_explicit_spin", True)),
            )
            spin_axis_options = ["none", "mz", "x", "y", "z"]
            default_spin_axis = st.session_state.get("spin_axis", "z") if has_explicit_spin else "none"
            spin_axis_value = default_spin_axis if default_spin_axis in spin_axis_options else "none"
            spin_axis = st.selectbox(
                "Spin axis",
                options=spin_axis_options,
                index=spin_axis_options.index(spin_axis_value) if spin_axis_value in spin_axis_options else 0,
                disabled=not has_explicit_spin,
            )
            spin_period_hr = st.number_input(
                "Spin period (hr)",
                min_value=0.1,
                max_value=20.0,
                value=float(st.session_state.get("spin_period_hr", 3.0) or 3.0),
                step=0.1,
                disabled=not has_explicit_spin,
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
            mass_code = st.text_input(
                "Mass code",
                value=st.session_state.get("mass_code", f"A{int(round(mass_log10_kg * 100)):04d}"),
            )
            resolution_numeric = float(resolution_code.replace("n", ""))
            st.metric("Resolution value", f"{int(resolution_numeric)}")

        submitted = st.form_submit_button("Run triage", type="primary")

    if not submitted and "latest_input_df" in st.session_state:
        return st.session_state["latest_input_df"]

    effective_spin_axis = spin_axis if has_explicit_spin else "none"
    effective_spin_period = spin_period_hr if has_explicit_spin and spin_axis != "none" else None
    input_df = pd.DataFrame(
        [
            {
                "case_name": case_name,
                "mass_log10_kg": mass_log10_kg,
                "mass_code": mass_code,
                "periapsis_Rm": periapsis_rm,
                "v_inf_kms": v_inf_kms,
                "spin_period_hr": effective_spin_period,
                "spin_axis": effective_spin_axis,
                "resolution_code": resolution_code,
                "resolution_value": resolution_numeric,
                "timestep": timestep,
                "fof_linking_length": fof_linking_length,
                "has_explicit_spin": has_explicit_spin,
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
    fragmentation_probability = float(result.get("fragmentation_probability", 0.0))
    model_score = float(result.get("model_score", fragmentation_probability))
    largest_fragment_mass_fraction = float(
        result.get("predicted_largest_fragment_mass_fraction", result.get("largest_fragment_mass_fraction", 0.0))
    )
    parent_mass_kg = float(result.get("parent_mass_kg", 10 ** float(result.get("mass_log10_kg", 0.0))))
    predicted_largest_fragment_mass_kg = float(result.get("predicted_largest_fragment_mass_kg", 0.0))
    risk_label = str(result.get("risk_label", "not available yet")).replace("_", " ")
    domain_status = str(result.get("domain_status", "not available yet")).replace("_", " ")
    calibration_warning = str(result.get("calibration_warning", "") or "")

    metric_cols = st.columns(6)
    metric_cols[0].metric("Fragmentation Probability", f"{min(max(fragmentation_probability, 0.001), 0.999):.1%}")
    metric_cols[1].metric("Model Score", f"{model_score:.3f}")
    metric_cols[2].metric("Risk Label", risk_label)
    metric_cols[3].metric("Largest Fragment Mass Fraction", f"{largest_fragment_mass_fraction:.3f}")
    metric_cols[4].metric("Domain Status", domain_status)
    metric_cols[5].metric("Spin Active", "Yes" if bool(result["has_explicit_spin"]) else "No")
    if calibration_warning:
        st.warning("Prediction is very confident. Treat this as a model score unless probability calibration has been checked.")
    st.caption(f"Predicted largest fragment mass: {predicted_largest_fragment_mass_kg:.3e} kg")
    st.caption(f"Parent mass estimate: {parent_mass_kg:.3e} kg")


def render_probability_chart(result: pd.Series) -> None:
    st.subheader("Fragmentation Probability")
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


def render_parameter_chart(result: pd.Series) -> None:
    st.subheader("Input Position Within Training Domain")
    domain_detail = result.get("domain_detail", {})
    numeric_rows = domain_detail.get("numeric_details", []) if isinstance(domain_detail, dict) else []
    if not numeric_rows:
        st.info("No numeric training-domain information available.")
        return

    chart_df = pd.DataFrame(numeric_rows).set_index("feature")
    st.bar_chart(chart_df[["relative_position"]], height=280)
    st.dataframe(chart_df, use_container_width=True)


def render_categorical_domain_panel(result: pd.Series) -> None:
    st.subheader("Categorical Training Coverage")
    domain_detail = result.get("domain_detail", {})
    categorical_rows = domain_detail.get("categorical_details", []) if isinstance(domain_detail, dict) else []
    if not categorical_rows:
        st.caption("No categorical training coverage information available.")
        return
    st.dataframe(pd.DataFrame(categorical_rows), use_container_width=True)


def render_target_breakdown(result: pd.Series) -> None:
    st.subheader("Target Breakdown")

    def normalise_target_value(value: object) -> object:
        if pd.isna(value):
            return "not available yet — no model loaded for this target"
        if isinstance(value, float):
            return round(value, 3)
        return value

    fragmentation_targets = [
        ("fragmentation_probability", normalise_target_value(result.get("fragmentation_probability"))),
        ("is_fragmented_proxy", normalise_target_value(result.get("is_fragmented_proxy"))),
        ("risk_label", normalise_target_value(result.get("risk_label"))),
        ("predicted_fragment_count", "not available yet — no model loaded for this target"),
        ("predicted_largest_fragment_mass_kg", normalise_target_value(result.get("predicted_largest_fragment_mass_kg"))),
        ("predicted_largest_fragment_mass_fraction", normalise_target_value(result.get("predicted_largest_fragment_mass_fraction"))),
        ("severity_class", normalise_target_value(result.get("severity_class"))),
    ]
    bound_targets = [
        ("has_any_bound_mass", normalise_target_value(result.get("has_any_bound_mass", "not available yet — no model loaded for this target"))),
        ("bound_mass_fraction", normalise_target_value(result.get("bound_mass_fraction", "not available yet — no model loaded for this target"))),
        ("bound_mass_fraction_ge_0p1", normalise_target_value(result.get("bound_mass_fraction_ge_0p1", "not available yet — no model loaded for this target"))),
        ("bound_fragment_count", normalise_target_value(result.get("bound_fragment_count", "not available yet — no model loaded for this target"))),
        ("largest_bound_fragment_mass_kg", normalise_target_value(result.get("largest_bound_fragment_mass_kg", "not available yet — no model loaded for this target"))),
    ]
    orbital_targets = [
        ("bound_fragment_eccentricity", normalise_target_value(result.get("bound_fragment_eccentricity", "not available yet — bound-fragment extraction not implemented"))),
        ("minimum_bound_eccentricity", normalise_target_value(result.get("minimum_bound_eccentricity", "not available yet — bound-fragment extraction not implemented"))),
        ("low_eccentricity_bound_fragment_flag", normalise_target_value(result.get("low_eccentricity_bound_fragment_flag", "not available yet — bound-fragment extraction not implemented"))),
    ]

    def render_target_section(title: str, rows: list[tuple[str, object]]) -> None:
        st.markdown(f"**{title}**")
        for key, value in rows:
            left, right = st.columns([1.3, 2.7])
            left.caption(key)
            right.write(value)

    render_target_section("Fragmentation proxy targets", fragmentation_targets)
    render_target_section("Bound-retention targets", bound_targets)
    render_target_section("Orbital relevance targets", orbital_targets)


def render_model_status_panel() -> None:
    st.subheader("Model Status")
    status_df = pd.DataFrame(get_artifact_status(MODEL_DIR))
    st.dataframe(status_df, use_container_width=True)


def render_raw_prediction_outputs(result: pd.Series) -> None:
    st.subheader("Prediction Output Debugging")
    with st.expander("Raw prediction outputs"):
        st.json(result.get("prediction_result", {}))


def render_limitations_panel() -> None:
    st.subheader("Limitations")
    st.info(
        "This dashboard predicts FoF-derived fragmentation and bound-retention proxy outcomes from existing SPH metadata. "
        "It does not replace SPH and does not validate long-term capture, disk formation, or moon formation. "
        "SPH is still required for out-of-domain cases, boundary cases, and any case where detailed fragment physics or orbital evolution is needed."
    )


def render_flag_panel(result: pd.Series) -> None:
    st.subheader("Domain Flags")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Out-of-domain features")
        if result["out_of_domain_features"]:
            st.code(result["out_of_domain_features"], language="text")
        else:
            st.caption("None detected")
    with col2:
        st.write("Near-edge features")
        if result["near_edge_features"]:
            st.code(result["near_edge_features"], language="text")
        else:
            st.caption("None detected")


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
    render_model_status_panel()
    render_metric_row(result)

    col1, col2 = st.columns([1.2, 1.0])
    with col1:
        render_recommendation_panel(result)
        render_probability_chart(result)
    with col2:
        render_flag_panel(result)

    render_parameter_chart(result)
    render_categorical_domain_panel(result)
    render_target_breakdown(result)
    render_raw_prediction_outputs(result)
    render_template_comparison(classifier, regressor, training_domain)
    render_limitations_panel()


if __name__ == "__main__":
    main()
