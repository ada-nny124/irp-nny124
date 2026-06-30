"""Streamlit dashboard for SPH fragmentation triage."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irp_triage import add_derived_features, load_artifacts, predict_cases

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
BOUND_MODELS_DIR = ROOT / "ml" / "bound_outcomes" / "models"
BOUND_TABLES_DIR = ROOT / "ml" / "bound_outcomes" / "tables"

CLASSIFICATION_TARGETS = ["is_fragmented_proxy", "has_any_bound_mass", "bound_mass_fraction_ge_0p1"]
REGRESSION_TARGETS = [
    "fragment_count_min_particles",
    "largest_fragment_mass_kg",
    "largest_fragment_mass_fraction",
    "bound_mass_fraction",
    "largest_bound_fragment_mass_kg",
    "minimum_bound_eccentricity",
]
FEATURE_COLUMNS = [
    "mass_log10_kg",
    "periapsis_Rm",
    "v_inf_kms",
    "spin_period_hr",
    "spin_axis",
    "resolution_code",
    "timestep",
    "fof_linking_length",
]


@st.cache_resource
def cached_artifacts():
    return load_artifacts(MODEL_DIR)


@st.cache_data
def load_template_cases() -> list[dict[str, object]]:
    if not TEMPLATE_PATH.exists():
        return []
    data = json.loads(TEMPLATE_PATH.read_text())
    return data if isinstance(data, list) else [data]


@st.cache_data
def load_metrics_file() -> dict[str, object]:
    metrics_path = MODEL_DIR / "metrics.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text())


@st.cache_data
def load_target_metrics(target_name: str) -> dict[str, object] | None:
    path = MODEL_DIR / f"{target_name}_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data
def load_eval_predictions(target_name: str) -> pd.DataFrame | None:
    path = MODEL_DIR / f"{target_name}_eval_predictions.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_bound_metrics_tables() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    classification_path = BOUND_TABLES_DIR / "classification_metrics.csv"
    regression_path = BOUND_TABLES_DIR / "regression_metrics.csv"
    classification_df = pd.read_csv(classification_path) if classification_path.exists() else None
    regression_df = pd.read_csv(regression_path) if regression_path.exists() else None
    return classification_df, regression_df


@st.cache_data
def select_best_bound_model_paths() -> dict[str, str]:
    classification_df, regression_df = load_bound_metrics_tables()
    selected: dict[str, str] = {}
    if classification_df is not None and not classification_df.empty:
        for target in ["has_any_bound_mass", "bound_mass_fraction_ge_0_1"]:
            subset = classification_df[classification_df["target"] == target].sort_values(
                ["balanced_accuracy", "f1", "roc_auc"],
                ascending=[False, False, False],
            )
            if subset.empty:
                continue
            row = subset.iloc[0]
            selected[target] = str(BOUND_MODELS_DIR / f"{row['dataset']}__{row['feature_set']}__{row['target']}__{row['model']}.pkl")
    if regression_df is not None and not regression_df.empty:
        for target in ["bound_mass_fraction", "bound_fragment_count", "largest_bound_fragment_mass_kg"]:
            subset = regression_df[regression_df["target"] == target].sort_values(
                ["r2", "mae", "rmse"],
                ascending=[False, True, True],
            )
            if subset.empty:
                continue
            row = subset.iloc[0]
            selected[target] = str(BOUND_MODELS_DIR / f"{row['dataset']}__{row['feature_set']}__{row['target']}__{row['model']}.pkl")
    return selected


@st.cache_resource
def load_bound_models() -> dict[str, object]:
    import pickle

    models: dict[str, object] = {}
    for target, path in select_best_bound_model_paths().items():
        model_path = Path(path)
        if not model_path.exists():
            continue
        with model_path.open("rb") as handle:
            models[target] = pickle.load(handle)
    return models


def make_bound_feature_frame(input_df: pd.DataFrame) -> pd.DataFrame:
    frame = add_derived_features(input_df)
    frame["particle_log10"] = pd.to_numeric(frame["resolution_value"], errors="coerce").map(
        lambda x: np.nan if pd.isna(x) else np.log10(x)
    )
    frame["special_case_code"] = frame.get("special_case_code", "none")
    frame["special_case_code"] = pd.Series(frame["special_case_code"], index=frame.index).fillna("none").replace("", "none")
    return frame


def apply_bound_predictions(result: pd.Series, input_df: pd.DataFrame) -> pd.Series:
    models = load_bound_models()
    if not models:
        return result

    features = make_bound_feature_frame(input_df)
    for target, model in models.items():
        expected_columns = list(getattr(model.named_steps["preprocessor"], "feature_names_in_", []))
        if not expected_columns:
            continue
        X = features[expected_columns].copy()
        if target in {"has_any_bound_mass", "bound_mass_fraction_ge_0_1"}:
            proba = float(model.predict_proba(X)[:, 1][0])
            if target == "has_any_bound_mass":
                result["has_any_bound_mass"] = bool(proba >= 0.5)
                result["has_any_bound_mass_probability"] = proba
            else:
                result["bound_mass_fraction_ge_0p1"] = proba
                result["bound_mass_fraction_ge_0_1_probability"] = proba
        else:
            pred = float(model.predict(X)[0])
            if target == "bound_mass_fraction":
                result["bound_mass_fraction"] = pred
            elif target == "bound_fragment_count":
                result["bound_fragment_count"] = pred
            elif target == "largest_bound_fragment_mass_kg":
                result["largest_bound_fragment_mass_kg"] = pred
    return result


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
            "SPH is still required for out-of-domain cases, boundary cases, detailed fragment physics, bound orbit questions, and new physical regimes."
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
            mass_log10_kg = st.number_input("Mass log10 kg", min_value=15.0, max_value=25.0, value=float(st.session_state.get("mass_log10_kg", 19.5)), step=0.1)
            periapsis_rm = st.number_input("Periapsis (Mars radii)", min_value=0.5, max_value=5.0, value=float(st.session_state.get("periapsis_Rm", 2.0)), step=0.1)
            v_inf_kms = st.number_input("Velocity at infinity (km/s)", min_value=0.0, max_value=5.0, value=float(st.session_state.get("v_inf_kms", 0.8)), step=0.1)
        with col2:
            has_explicit_spin = st.toggle("Explicit spin enabled", value=bool(st.session_state.get("has_explicit_spin", True)))
            spin_axis_options = ["none", "mz", "x", "y", "z"]
            default_spin_axis = st.session_state.get("spin_axis", "z") if has_explicit_spin else "none"
            spin_axis = st.selectbox("Spin axis", options=spin_axis_options, index=spin_axis_options.index(default_spin_axis) if default_spin_axis in spin_axis_options else 0, disabled=not has_explicit_spin)
            spin_period_hr = st.number_input("Spin period (hr)", min_value=0.1, max_value=20.0, value=float(st.session_state.get("spin_period_hr", 8.6) or 8.6), step=0.1, disabled=not has_explicit_spin)
            resolution_options = ["n50", "n55", "n60", "n65", "n70"]
            resolution_value = st.session_state.get("resolution_code", "n65")
            resolution_code = st.selectbox("Resolution code", options=resolution_options, index=resolution_options.index(resolution_value) if resolution_value in resolution_options else 3)
            timestep = st.number_input("Timestep", min_value=1000.0, max_value=200000.0, value=float(st.session_state.get("timestep", 90000.0)), step=1000.0)
        with col3:
            fof_linking_length = st.number_input("FoF linking length", min_value=0.0001, max_value=0.05, value=float(st.session_state.get("fof_linking_length", 0.004)), step=0.0001, format="%.4f")
            mass_code = st.text_input("Mass code", value=st.session_state.get("mass_code", f"A{int(round(mass_log10_kg * 100)):04d}"))
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
                "spin_period_hr": spin_period_hr if has_explicit_spin and spin_axis != "none" else None,
                "spin_axis": spin_axis if has_explicit_spin else "none",
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


def derive_result_fields(result: pd.Series) -> dict[str, object]:
    mass_log10_kg = float(result.get("mass_log10_kg", 0.0))
    parent_mass_kg = result.get("parent_mass_kg")
    if pd.isna(parent_mass_kg) or parent_mass_kg is None:
        parent_mass_kg = 10 ** mass_log10_kg
    parent_mass_kg = float(parent_mass_kg)

    predicted_largest_fragment_mass_kg = result.get("predicted_largest_fragment_mass_kg")
    if pd.isna(predicted_largest_fragment_mass_kg) or predicted_largest_fragment_mass_kg is None:
        predicted_largest_fragment_mass_kg = 0.0
    predicted_largest_fragment_mass_kg = float(predicted_largest_fragment_mass_kg)

    largest_fragment_mass_fraction = result.get("predicted_largest_fragment_mass_fraction")
    if pd.isna(largest_fragment_mass_fraction) or largest_fragment_mass_fraction is None:
        if parent_mass_kg > 0.0:
            largest_fragment_mass_fraction = predicted_largest_fragment_mass_kg / parent_mass_kg
        else:
            largest_fragment_mass_fraction = 0.0
    largest_fragment_mass_fraction = float(largest_fragment_mass_fraction)

    fragmentation_probability = float(result.get("fragmentation_probability", 0.0))
    risk_label = result.get("risk_label")
    if pd.isna(risk_label) or risk_label is None or str(risk_label).strip() == "":
        if fragmentation_probability < 0.25:
            risk_label = "low"
        elif fragmentation_probability < 0.5:
            risk_label = "medium"
        elif fragmentation_probability < 0.75:
            risk_label = "high"
        else:
            risk_label = "very high"

    severity_class = result.get("severity_class")
    if pd.isna(severity_class) or severity_class is None or str(severity_class).strip() == "":
        if largest_fragment_mass_fraction > 0.9:
            severity_class = "no_or_very_weak_fragmentation"
        elif largest_fragment_mass_fraction >= 0.5:
            severity_class = "weak_fragmentation"
        elif largest_fragment_mass_fraction >= 0.1:
            severity_class = "moderate_fragmentation"
        else:
            severity_class = "strong_fragmentation"

    return {
        "parent_mass_kg": parent_mass_kg,
        "predicted_largest_fragment_mass_kg": predicted_largest_fragment_mass_kg,
        "predicted_largest_fragment_mass_fraction": largest_fragment_mass_fraction,
        "risk_label": str(risk_label),
        "severity_class": str(severity_class),
    }


def render_model_status_panel() -> None:
    st.subheader("Model Status")
    rows = []
    for artifact in get_artifact_status(MODEL_DIR):
        label = artifact["label"].replace(".pkl", "").replace(".json", "")
        if label == "fragmentation_regressor":
            metrics_target = "largest_fragment_mass_kg"
        elif label == "fragmentation_classifier":
            metrics_target = "is_fragmented_proxy"
        elif label == "training_domain":
            metrics_target = "training_domain"
        else:
            metrics_target = label
        rows.append(
            {
                "status": artifact["status"],
                "model_path": artifact["path"],
                "metrics_path": str(MODEL_DIR / f"{metrics_target}_metrics.json"),
                "eval_prediction_path": str(MODEL_DIR / f"{metrics_target}_eval_predictions.csv"),
                "target_name": artifact["target"],
            }
        )
    for target, path in select_best_bound_model_paths().items():
        rows.append(
            {
                "status": "loaded" if target in load_bound_models() else "missing",
                "model_path": path,
                "metrics_path": str(BOUND_TABLES_DIR / ("classification_metrics.csv" if target in {"has_any_bound_mass", "bound_mass_fraction_ge_0_1"} else "regression_metrics.csv")),
                "eval_prediction_path": str(BOUND_TABLES_DIR / "prediction_records.csv"),
                "target_name": target,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    with st.expander("Loaded files"):
        st.json(rows)


def render_metric_row(result: pd.Series) -> None:
    st.subheader("Prediction Summary")
    derived = derive_result_fields(result)
    fragmentation_probability = float(result.get("fragmentation_probability", 0.0))
    model_score = float(result.get("model_score", fragmentation_probability))
    largest_fragment_mass_fraction = float(derived["predicted_largest_fragment_mass_fraction"])
    metric_cols = st.columns(6)
    metric_cols[0].metric("Fragmentation Probability", f"{min(max(fragmentation_probability, 0.001), 0.999):.1%}")
    metric_cols[1].metric("Model Score", f"{model_score:.3f}")
    metric_cols[2].metric("Risk Label", str(derived["risk_label"]))
    metric_cols[3].metric("Largest Fragment Mass Fraction", f"{largest_fragment_mass_fraction:.3f}")
    metric_cols[4].metric("Domain Status", str(result.get("domain_status", "not available yet")).replace("_", " "))
    metric_cols[5].metric("Spin Active", "Yes" if bool(result.get("has_explicit_spin", False)) else "No")
    if float(result.get("fragmentation_probability", 0.0)) > 0.98 or float(result.get("fragmentation_probability", 0.0)) < 0.02:
        st.warning("Prediction is very confident. Treat this as a model score unless probability calibration has been checked.")
    st.caption(f"Predicted largest fragment mass: {float(derived['predicted_largest_fragment_mass_kg']):.3e} kg")
    st.caption(f"Parent mass estimate: {float(derived['parent_mass_kg']):.3e} kg")


def render_recommendation_card(result: pd.Series) -> None:
    st.subheader("SPH Decision Card")
    derived = derive_result_fields(result)
    triggers = []
    p = float(result.get("fragmentation_probability", 0.0))
    if float(derived["predicted_largest_fragment_mass_fraction"]) < 0.5 and p > 0.75:
        triggers.append("high fragmentation probability")
    if str(result.get("domain_status", "")) == "near_edge":
        triggers.append("near-edge domain status")
    if str(result.get("domain_status", "")) == "out_of_domain":
        triggers.append("out-of-domain input")
    if 0.4 <= p <= 0.6:
        triggers.append("uncertain prediction")
    if "not available yet" in str(result.get("bound_mass_fraction", "")):
        triggers.append("missing model")
    message = f"**{result['sph_recommendation'].upper()}**\n\n{result['explanation']}\n\nTriggered by: {', '.join(triggers) if triggers else 'ranking / prioritisation logic'}"
    tone = recommendation_tone(result["sph_recommendation"])
    if tone == "error":
        st.error(message)
    elif tone == "warning":
        st.warning(message)
    else:
        st.success(message)


def render_fragmentation_probability_bar(result: pd.Series) -> None:
    st.subheader("Fragmentation Probability")
    frag = float(result["fragmentation_probability"])
    chart_df = pd.DataFrame(
        {
            "Outcome": ["Fragmentation", "Non-fragmentation"],
            "Probability": [frag, 1.0 - frag],
        }
    )
    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Outcome:N", sort=None),
            y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Outcome:N", scale=alt.Scale(range=["#d95f02", "#1b9e77"]), legend=None),
            tooltip=["Outcome", alt.Tooltip("Probability:Q", format=".3f")],
        )
        .properties(height=260)
    )
    labels = chart.mark_text(dy=-10, color="white").encode(text=alt.Text("Probability:Q", format=".2f"))
    st.altair_chart(chart + labels, use_container_width=True)


def render_severity_card(result: pd.Series) -> None:
    st.subheader("Fragmentation Severity Card")
    derived = derive_result_fields(result)
    rows = [
        ("Predicted largest fragment mass", f"{float(derived['predicted_largest_fragment_mass_kg']):.3e} kg"),
        ("Parent mass estimate", f"{float(derived['parent_mass_kg']):.3e} kg"),
        ("Largest fragment mass fraction", f"{float(derived['predicted_largest_fragment_mass_fraction']):.3f}"),
        ("Severity class", str(derived["severity_class"])),
    ]
    for key, value in rows:
        left, right = st.columns([1.3, 2.7])
        left.caption(key)
        right.write(value)


def render_bound_retention_card(result: pd.Series) -> None:
    st.subheader("Bound-Retention Card")
    retained_mass_category = "not available yet — no bound-retention model loaded"
    try:
        bound_mass_fraction = float(result.get("bound_mass_fraction"))
        if bound_mass_fraction <= 0.01:
            retained_mass_category = "none/negligible"
        elif bound_mass_fraction < 0.1:
            retained_mass_category = "low"
        elif bound_mass_fraction < 0.3:
            retained_mass_category = "moderate"
        else:
            retained_mass_category = "high"
    except (TypeError, ValueError):
        pass
    rows = [
        ("Predicted bound mass fraction", str(result.get("bound_mass_fraction", "not available yet — no model loaded for this target"))),
        ("Probability BMF >= 10%", str(result.get("bound_mass_fraction_ge_0p1", "not available yet — no model loaded for this target"))),
        ("Retained mass category", retained_mass_category),
    ]
    for key, value in rows:
        left, right = st.columns([1.3, 2.7])
        left.caption(key)
        right.write(value)


def render_bound_percentage_graph(result: pd.Series) -> None:
    st.subheader("Bound Percentages")
    bound_mass_fraction = result.get("bound_mass_fraction")
    bmf_ge_0p1 = result.get("bound_mass_fraction_ge_0p1")
    if isinstance(bound_mass_fraction, str) and "not available yet" in bound_mass_fraction:
        st.info("Not available yet — no trained bound-retention model is connected, so bound percentage graphs cannot be drawn.")
        return
    try:
        bound_mass_fraction_value = float(bound_mass_fraction)
    except (TypeError, ValueError):
        st.info("Not available yet — no usable bound-mass percentage prediction found.")
        return

    try:
        bmf_ge_0p1_value = float(bmf_ge_0p1)
    except (TypeError, ValueError):
        bmf_ge_0p1_value = np.nan

    chart_rows = [
        {"metric": "Bound mass fraction", "value": bound_mass_fraction_value},
        {"metric": "Mass not retained", "value": max(0.0, 1.0 - bound_mass_fraction_value)},
    ]
    if not np.isnan(bmf_ge_0p1_value):
        chart_rows.extend(
            [
                {"metric": "P(BMF >= 10%)", "value": bmf_ge_0p1_value},
                {"metric": "P(BMF < 10%)", "value": max(0.0, 1.0 - bmf_ge_0p1_value)},
            ]
        )
    chart_df = pd.DataFrame(chart_rows)
    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("metric:N", sort=None, title=None),
            y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1]), title="Fraction / probability"),
            color=alt.Color("metric:N", legend=None),
            tooltip=["metric", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)


def render_domain_panels(result: pd.Series) -> None:
    st.subheader("Input Position Within Training Domain")
    domain_detail = result.get("domain_detail", {})
    numeric_rows = domain_detail.get("numeric_details", []) if isinstance(domain_detail, dict) else []
    if numeric_rows:
        chart_df = pd.DataFrame(numeric_rows).set_index("feature")
        domain_df = chart_df.reset_index()
        bars = (
            alt.Chart(domain_df)
            .mark_bar()
            .encode(
                x=alt.X("relative_position:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("feature:N", sort="-x"),
                color=alt.Color("status:N", scale=alt.Scale(domain=["in_domain", "near_edge", "out_of_domain"], range=["#1b9e77", "#f28e2b", "#d62728"])),
                tooltip=["feature", "value", "min", "max", alt.Tooltip("relative_position:Q", format=".3f"), "status"],
            )
            .properties(height=260)
        )
        st.altair_chart(bars, use_container_width=True)
        st.dataframe(chart_df, use_container_width=True)
    else:
        st.info("No numeric training-domain information available.")
    st.subheader("Categorical Training Coverage")
    categorical_rows = domain_detail.get("categorical_details", []) if isinstance(domain_detail, dict) else []
    if categorical_rows:
        st.dataframe(pd.DataFrame(categorical_rows), use_container_width=True)
    else:
        st.info("No categorical training-domain information available.")


def render_target_diagnostics(result: pd.Series) -> None:
    derived = derive_result_fields(result)
    st.markdown("**Target hierarchy**")
    st.caption("Level 1: Fragmentation proxy — multiple FoF fragments. Level 2: Largest-remnant proxy — mass left in the largest fragment. Level 3: Bound-retention proxy — mass retained around Mars. Level 4: Orbital relevance — whether bound fragments are potentially retained on low-eccentricity orbits.")
    sections = {
        "A. Fragmentation targets": [
            ("fragmentation_probability", round(float(result.get("fragmentation_probability", 0.0)), 3)),
            ("is_fragmented_proxy", bool(result.get("is_fragmented_proxy", False))),
            ("risk_label", str(derived["risk_label"])),
            ("predicted_largest_fragment_mass_kg", f"{float(derived['predicted_largest_fragment_mass_kg']):.3e}"),
            ("predicted_largest_fragment_mass_fraction", round(float(derived["predicted_largest_fragment_mass_fraction"]), 3)),
            ("severity_class", str(derived["severity_class"])),
        ],
        "B. Bound-retention targets": [
            ("has_any_bound_mass", str(result.get("has_any_bound_mass", "not available yet — no model loaded for this target"))),
            ("bound_mass_fraction", str(result.get("bound_mass_fraction", "not available yet — no model loaded for this target"))),
            ("bound_mass_fraction_ge_0p1", str(result.get("bound_mass_fraction_ge_0p1", "not available yet — no model loaded for this target"))),
            ("bound_fragment_count", str(result.get("bound_fragment_count", "not available yet — no model loaded for this target"))),
            ("largest_bound_fragment_mass_kg", str(result.get("largest_bound_fragment_mass_kg", "not available yet — no model loaded for this target"))),
        ],
        "C. Orbital-relevance targets": [
            ("bound_fragment_eccentricity", str(result.get("bound_fragment_eccentricity", "not available yet — bound-fragment extraction not implemented"))),
            ("minimum_bound_eccentricity", str(result.get("minimum_bound_eccentricity", "not available yet — bound-fragment extraction not implemented"))),
            ("low_eccentricity_bound_fragment_flag", str(result.get("low_eccentricity_bound_fragment_flag", "not available yet — bound-fragment extraction not implemented"))),
        ],
    }
    for title, rows in sections.items():
        st.markdown(f"**{title}**")
        for key, value in rows:
            left, right = st.columns([1.3, 2.7])
            left.caption(key)
            right.write(value)
    with st.expander("Raw prediction outputs"):
        st.json(result.get("prediction_result", {}))


def render_template_comparison(classifier, regressor, training_domain: dict[str, object]) -> None:
    template_cases = load_template_cases()
    if not template_cases:
        return
    st.subheader("Template Scenario Comparison")
    result = predict_cases(pd.DataFrame(template_cases), classifier, regressor, training_domain)
    st.dataframe(
        result[["case_name", "fragmentation_probability", "predicted_largest_fragment_mass_kg", "severity_class", "domain_status", "sph_recommendation"]],
        use_container_width=True,
    )


def render_plot_message(target_name: str) -> None:
    st.info(f"Not available yet — no trained model/evaluation file found for `{target_name}`.")


def render_feature_importance(target_name: str, eval_df: pd.DataFrame | None) -> None:
    model_path = MODEL_DIR / f"{target_name}_model.pkl"
    if not model_path.exists():
        st.info("Which inputs did this model rely on? Not available yet — no trained model/evaluation file found for this target.")
        return
    import pickle

    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    estimator = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessor"]
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = np.array(FEATURE_COLUMNS)
    if hasattr(estimator, "feature_importances_"):
        importance_df = pd.DataFrame({"feature": feature_names, "importance": estimator.feature_importances_}).sort_values("importance", ascending=False).head(12)
        st.markdown("**Which inputs did this model rely on?**")
        chart = (
            alt.Chart(importance_df)
            .mark_bar()
            .encode(
                x=alt.X("importance:Q"),
                y=alt.Y("feature:N", sort="-x"),
                tooltip=["feature", alt.Tooltip("importance:Q", format=".4f")],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)
        return
    if eval_df is not None and all(column in eval_df.columns for column in FEATURE_COLUMNS):
        X = eval_df[FEATURE_COLUMNS].copy()
        y = eval_df["y_true"]
        result = permutation_importance(model, X, y, n_repeats=5, random_state=42)
        importance_df = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": result.importances_mean}).sort_values("importance", ascending=False)
        st.markdown("**Which inputs did this model rely on?**")
        chart = (
            alt.Chart(importance_df)
            .mark_bar()
            .encode(
                x=alt.X("importance:Q"),
                y=alt.Y("feature:N", sort="-x"),
                tooltip=["feature", alt.Tooltip("importance:Q", format=".4f")],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)
        return
    st.info("Which inputs did this model rely on? Not available yet — feature importance could not be computed.")


def coerce_binary_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    normalized = series.astype(str).str.strip().str.lower()
    mapped = normalized.map({"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0})
    return mapped.fillna(0).astype(int)


def render_classification_plot(plot_type: str, eval_df: pd.DataFrame, metrics: dict[str, object]) -> None:
    y_true = coerce_binary_series(eval_df["y_true"])
    y_pred = coerce_binary_series(eval_df["y_pred"])
    y_proba = eval_df["y_proba"].astype(float) if "y_proba" in eval_df.columns else None
    plot_col, table_col = st.columns([1.8, 1.0])
    with plot_col:
        if plot_type == "Confusion matrix":
            cm = confusion_matrix(y_true, y_pred)
            cm_df = pd.DataFrame(
                [
                    {"True": "0", "Predicted": "0", "Count": int(cm[0, 0])},
                    {"True": "0", "Predicted": "1", "Count": int(cm[0, 1])},
                    {"True": "1", "Predicted": "0", "Count": int(cm[1, 0])},
                    {"True": "1", "Predicted": "1", "Count": int(cm[1, 1])},
                ]
            )
            heatmap = (
                alt.Chart(cm_df)
                .mark_rect()
                .encode(
                    x=alt.X("Predicted:N"),
                    y=alt.Y("True:N"),
                    color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")),
                    tooltip=["True", "Predicted", "Count"],
                )
                .properties(height=320, title="Confusion matrix")
            )
            labels = alt.Chart(cm_df).mark_text(fontSize=16).encode(
                x="Predicted:N",
                y="True:N",
                text="Count:Q",
                color=alt.condition(alt.datum.Count > cm_df["Count"].max() * 0.5, alt.value("white"), alt.value("black")),
            )
            st.altair_chart((heatmap + labels).interactive(), use_container_width=True)
        elif plot_type == "ROC curve" and y_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
            base = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color="gray").encode(x="x:Q", y="y:Q")
            line = alt.Chart(roc_df).mark_line(color="#1f77b4").encode(
                x=alt.X("fpr:Q", title="False positive rate"),
                y=alt.Y("tpr:Q", title="True positive rate"),
                tooltip=[alt.Tooltip("fpr:Q", format=".3f"), alt.Tooltip("tpr:Q", format=".3f")],
            )
            st.altair_chart((base + line).properties(height=320, title=f"ROC curve (AUC {metrics.get('roc_auc', float('nan')):.3f})").interactive(), use_container_width=True)
        elif plot_type == "Precision-recall curve" and y_proba is not None:
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            pr_df = pd.DataFrame({"recall": recall, "precision": precision})
            line = alt.Chart(pr_df).mark_line(color="#2ca02c").encode(
                x=alt.X("recall:Q", title="Recall"),
                y=alt.Y("precision:Q", title="Precision"),
                tooltip=[alt.Tooltip("recall:Q", format=".3f"), alt.Tooltip("precision:Q", format=".3f")],
            )
            st.altair_chart(line.properties(height=320, title=f"Precision-recall curve (PR AUC {metrics.get('pr_auc', float('nan')):.3f})").interactive(), use_container_width=True)
        elif plot_type == "Calibration curve" and y_proba is not None:
            n_bins = st.slider("Calibration bins", min_value=5, max_value=20, value=8, key="calibration_bins")
            frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
            cal_df = pd.DataFrame({"mean_pred": mean_pred, "frac_pos": frac_pos})
            base = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color="gray").encode(x="x:Q", y="y:Q")
            points = alt.Chart(cal_df).mark_line(point=True, color="#e15759").encode(
                x=alt.X("mean_pred:Q", title="Mean predicted probability"),
                y=alt.Y("frac_pos:Q", title="Observed fraction positive"),
                tooltip=[alt.Tooltip("mean_pred:Q", format=".3f"), alt.Tooltip("frac_pos:Q", format=".3f")],
            )
            st.altair_chart((base + points).properties(height=320, title="Calibration curve").interactive(), use_container_width=True)
        else:
            st.info("This plot needs probability outputs and is not available for the selected target.")
    with table_col:
        st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: "value"}), use_container_width=True)


def render_regression_plot(plot_type: str, eval_df: pd.DataFrame, metrics: dict[str, object]) -> None:
    plot_col, table_col = st.columns([1.8, 1.0])
    residuals = eval_df["y_true"] - eval_df["y_pred"]
    with plot_col:
        opacity = st.slider("Point opacity", min_value=0.1, max_value=1.0, value=0.5, step=0.1, key=f"opacity_{plot_type}")
        if plot_type == "Actual vs predicted scatter":
            min_val = float(min(eval_df["y_true"].min(), eval_df["y_pred"].min()))
            max_val = float(max(eval_df["y_true"].max(), eval_df["y_pred"].max()))
            scatter_df = pd.DataFrame({"y_true": eval_df["y_true"], "y_pred": eval_df["y_pred"]})
            points = alt.Chart(scatter_df).mark_circle(size=45, opacity=opacity, color="#1f77b4").encode(
                x=alt.X("y_true:Q", title="Actual"),
                y=alt.Y("y_pred:Q", title="Predicted"),
                tooltip=[alt.Tooltip("y_true:Q", format=".3e"), alt.Tooltip("y_pred:Q", format=".3e")],
            )
            ref = alt.Chart(pd.DataFrame({"x": [min_val, max_val], "y": [min_val, max_val]})).mark_line(strokeDash=[4, 4], color="gray").encode(x="x:Q", y="y:Q")
            st.altair_chart((points + ref).properties(height=320, title="Actual vs predicted").interactive(), use_container_width=True)
        elif plot_type == "Residual plot":
            residual_df = pd.DataFrame({"y_pred": eval_df["y_pred"], "residual": residuals})
            points = alt.Chart(residual_df).mark_circle(size=45, opacity=opacity, color="#e15759").encode(
                x=alt.X("y_pred:Q", title="Predicted"),
                y=alt.Y("residual:Q", title="Residual"),
                tooltip=[alt.Tooltip("y_pred:Q", format=".3e"), alt.Tooltip("residual:Q", format=".3e")],
            )
            ref = alt.Chart(pd.DataFrame({"x": [float(eval_df["y_pred"].min()), float(eval_df["y_pred"].max())], "y": [0.0, 0.0]})).mark_line(strokeDash=[4, 4], color="gray").encode(x="x:Q", y="y:Q")
            st.altair_chart((points + ref).properties(height=320, title="Residual plot").interactive(), use_container_width=True)
        elif plot_type == "Residual histogram":
            bins = st.slider("Histogram bins", min_value=10, max_value=50, value=20, key="residual_bins")
            residual_df = pd.DataFrame({"residual": residuals})
            chart = alt.Chart(residual_df).mark_bar(color="#4c78a8").encode(
                x=alt.X("residual:Q", bin=alt.Bin(maxbins=bins), title="Residual"),
                y=alt.Y("count():Q", title="Count"),
                tooltip=[alt.Tooltip("count():Q", title="Count")],
            ).properties(height=320, title="Residual histogram")
            st.altair_chart(chart, use_container_width=True)
    with table_col:
        st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: "value"}), use_container_width=True)


def render_model_performance_tab(training_domain: dict[str, object]) -> None:
    st.subheader("Model Performance & Target Diagnostics")
    all_targets = CLASSIFICATION_TARGETS + REGRESSION_TARGETS
    target_name = st.selectbox("Select target", all_targets, index=0)
    is_classification = target_name in CLASSIFICATION_TARGETS
    plot_options = (
        ["Confusion matrix", "ROC curve", "Precision-recall curve", "Calibration curve"] if is_classification else ["Actual vs predicted scatter", "Residual plot", "Residual histogram"]
    )
    plot_type = st.selectbox("Select plot type", plot_options, index=0)
    eval_df = load_eval_predictions(target_name)
    metrics = load_target_metrics(target_name)
    if eval_df is None or metrics is None:
        bound_prediction_path = BOUND_TABLES_DIR / "prediction_records.csv"
        classification_df, regression_df = load_bound_metrics_tables()
        if bound_prediction_path.exists():
            prediction_df = pd.read_csv(bound_prediction_path)
            if target_name in {"has_any_bound_mass", "bound_mass_fraction_ge_0p1"} and classification_df is not None:
                target_key = "bound_mass_fraction_ge_0_1" if target_name == "bound_mass_fraction_ge_0p1" else target_name
                eval_df = prediction_df[prediction_df["target"] == target_key].copy().rename(
                    columns={"actual": "y_true", "predicted": "y_pred", "score": "y_proba"}
                )
                best = classification_df[classification_df["target"] == target_key].sort_values(
                    ["balanced_accuracy", "f1", "roc_auc"], ascending=[False, False, False]
                )
                metrics = best.iloc[0].to_dict() if not best.empty else None
            elif target_name in {"bound_mass_fraction", "bound_fragment_count", "largest_bound_fragment_mass_kg"} and regression_df is not None:
                eval_df = prediction_df[prediction_df["target"] == target_name].copy().rename(
                    columns={"actual": "y_true", "predicted": "y_pred"}
                )
                best = regression_df[regression_df["target"] == target_name].sort_values(
                    ["r2", "mae", "rmse"], ascending=[False, True, True]
                )
                metrics = best.iloc[0].to_dict() if not best.empty else None
    if eval_df is None or metrics is None:
        render_plot_message(target_name)
        return
    if is_classification:
        render_classification_plot(plot_type, eval_df, metrics)
    else:
        render_regression_plot(plot_type, eval_df, metrics)
    render_feature_importance(target_name, eval_df)


def render_parameter_sweep_tab(input_df: pd.DataFrame, classifier, regressor, training_domain: dict[str, object]) -> None:
    st.subheader("Parameter Sweep")
    sweep_variable = st.selectbox("Sweep variable", ["periapsis_Rm", "v_inf_kms", "spin_period_hr", "fof_linking_length"])
    numeric_domain = training_domain.get("numeric", {})
    spec = numeric_domain.get(sweep_variable)
    if not spec:
        st.info("Training-domain range unavailable for this parameter.")
        return
    sweep_values = np.linspace(spec["min"], spec["max"], 30)
    sweep_df = pd.concat([input_df] * len(sweep_values), ignore_index=True)
    sweep_df[sweep_variable] = sweep_values
    if sweep_variable == "spin_period_hr":
        sweep_df["has_explicit_spin"] = True
        sweep_df["spin_axis"] = sweep_df["spin_axis"].replace("none", "z")
    result = predict_cases(sweep_df, classifier, regressor, training_domain).sort_values(sweep_variable)
    sweep_chart_df = result[[sweep_variable, "fragmentation_probability"]].copy()
    boundary = alt.Chart(pd.DataFrame({sweep_variable: [float(result[sweep_variable].min()), float(result[sweep_variable].max())], "boundary": [0.5, 0.5]})).mark_line(strokeDash=[4, 4], color="gray").encode(
        x=alt.X(f"{sweep_variable}:Q"),
        y=alt.Y("boundary:Q", title="Fragmentation probability"),
    )
    line = alt.Chart(sweep_chart_df).mark_line(color="#d95f02", point=True).encode(
        x=alt.X(f"{sweep_variable}:Q"),
        y=alt.Y("fragmentation_probability:Q", scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip(f"{sweep_variable}:Q", format=".3f"), alt.Tooltip("fragmentation_probability:Q", format=".3f")],
    )
    st.altair_chart((boundary + line).properties(height=320, title="Parameter sweep").interactive(), use_container_width=True)
    result["slope"] = np.gradient(result["fragmentation_probability"], result[sweep_variable])
    crossing = result.iloc[(result["fragmentation_probability"] - 0.5).abs().argsort()[:3]][[sweep_variable, "fragmentation_probability"]]
    steepest = result.iloc[result["slope"].abs().argsort()[::-1][:3]][[sweep_variable, "fragmentation_probability", "slope"]]
    st.markdown("**Boundary candidates near fragmentation probability = 0.5**")
    st.dataframe(crossing, use_container_width=True)
    st.markdown("**Suggested SPH runs near the steepest transition**")
    st.dataframe(steepest, use_container_width=True)
    st.caption("Recommend 2–3 SPH runs around the boundary and steepest transition region.")


def render_limitations_tab() -> None:
    st.subheader("Limitations")
    st.info(
        "This dashboard predicts FoF-derived fragmentation and bound-retention proxy outcomes from existing SPH metadata. "
        "It does not replace SPH and does not validate long-term capture, disk formation, or moon formation. "
        "SPH is still required for out-of-domain cases, boundary cases, detailed fragment physics, bound orbit questions, and new physical regimes."
    )


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
    result = apply_bound_predictions(result, input_df)

    tabs = st.tabs(["Triage Prediction", "Model Performance", "Target Diagnostics", "Parameter Sweep", "Limitations"])
    with tabs[0]:
        render_model_status_panel()
        render_metric_row(result)
        col1, col2 = st.columns([1.1, 1.0])
        with col1:
            render_fragmentation_probability_bar(result)
            render_recommendation_card(result)
        with col2:
            render_severity_card(result)
            render_bound_retention_card(result)
        render_bound_percentage_graph(result)
        render_domain_panels(result)
        render_template_comparison(classifier, regressor, training_domain)
    with tabs[1]:
        render_model_performance_tab(training_domain)
    with tabs[2]:
        render_target_diagnostics(result)
    with tabs[3]:
        render_parameter_sweep_tab(input_df, classifier, regressor, training_domain)
    with tabs[4]:
        render_limitations_tab()


if __name__ == "__main__":
    main()
