"""Local web app for the SPH screening demo UI."""

from __future__ import annotations

import json
import pickle
import sys
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from triage import add_derived_features, check_training_domain, load_artifacts, predict_cases

MODEL_DIR = ROOT / "ml" / "triage"
BOUND_MODELS_DIR = ROOT / "ml" / "bound_outcomes" / "models"
BOUND_TABLES_DIR = ROOT / "ml" / "bound_outcomes" / "tables"
SURROGATE_TABLES_DIR = ROOT / "ml" / "physics_structured_surrogate" / "tables"
DATASET_PATH = ROOT / "extraction_outputs" / "bound_outcomes.csv"
HTML_PATH = ROOT / "src" / "triage" / "templates" / "sph_triage_dashboard.html"

MARS_MU_KM3_S2 = 4.282837e4
MARS_RADIUS_KM = 3389.5


@lru_cache(maxsize=1)
def load_dashboard_html() -> bytes:
    return HTML_PATH.read_bytes()


@lru_cache(maxsize=1)
def load_triage_bundle() -> tuple[object, object, dict[str, object]]:
    artifacts = load_artifacts(MODEL_DIR)
    if artifacts is None:
        raise FileNotFoundError("Missing required fragmentation artifacts in ml/triage")
    return artifacts


@lru_cache(maxsize=1)
def load_bound_metrics_tables() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    classification_path = BOUND_TABLES_DIR / "classification_metrics.csv"
    regression_path = BOUND_TABLES_DIR / "regression_metrics.csv"
    classification_df = pd.read_csv(classification_path) if classification_path.exists() else None
    regression_df = pd.read_csv(regression_path) if regression_path.exists() else None
    return classification_df, regression_df


@lru_cache(maxsize=1)
def select_best_bound_model_paths() -> dict[str, Path]:
    classification_df, regression_df = load_bound_metrics_tables()
    selected: dict[str, Path] = {}
    if classification_df is not None and not classification_df.empty:
        for target in ["has_any_bound_mass", "bound_mass_fraction_ge_0_1"]:
            subset = classification_df[classification_df["target"] == target].sort_values(
                ["balanced_accuracy", "f1", "roc_auc"],
                ascending=[False, False, False],
            )
            if subset.empty:
                continue
            row = subset.iloc[0]
            selected[target] = BOUND_MODELS_DIR / (
                f"{row['dataset']}__{row['feature_set']}__{row['target']}__{row['model']}.pkl"
            )
    if regression_df is not None and not regression_df.empty:
        for target in [
            "bound_mass_fraction",
            "bound_fragment_count",
            "largest_bound_fragment_mass_kg",
            "average_bound_fragment_mass_kg",
        ]:
            subset = regression_df[regression_df["target"] == target].sort_values(
                ["r2", "mae", "rmse"],
                ascending=[False, True, True],
            )
            if subset.empty:
                continue
            row = subset.iloc[0]
            selected[target] = BOUND_MODELS_DIR / (
                f"{row['dataset']}__{row['feature_set']}__{row['target']}__{row['model']}.pkl"
            )
    return selected


@lru_cache(maxsize=1)
def load_bound_models() -> dict[str, object]:
    models: dict[str, object] = {}
    for target, path in select_best_bound_model_paths().items():
        if not path.exists():
            continue
        with path.open("rb") as handle:
            models[target] = pickle.load(handle)
    return models


@lru_cache(maxsize=1)
def load_spread_models() -> dict[str, object]:
    model_paths = {
        "random_forest": BOUND_MODELS_DIR
        / "all_successful_runs__with_fof_linking_length__bound_mass_fraction__random_forest_regressor.pkl",
        "gradient_boosting": BOUND_MODELS_DIR
        / "all_successful_runs__with_fof_linking_length__bound_mass_fraction__gradient_boosting_regressor.pkl",
    }
    models: dict[str, object] = {}
    for name, path in model_paths.items():
        if not path.exists():
            continue
        with path.open("rb") as handle:
            models[name] = pickle.load(handle)
    return models


def parse_numeric_code(series: pd.Series, pattern: str, scale: float = 1.0) -> pd.Series:
    extracted = series.fillna("").astype(str).str.extract(pattern)[0]
    return pd.to_numeric(extracted, errors="coerce") / scale


@lru_cache(maxsize=1)
def load_support_frame() -> pd.DataFrame:
    frame = pd.read_csv(DATASET_PATH, low_memory=False)
    frame["mass_log10_kg"] = parse_numeric_code(frame["mass_code"], r"A(\d{4})", scale=100.0)
    frame["resolution_value"] = parse_numeric_code(frame["resolution_code"], r"n(\d+)")
    frame["periapsis_Rm"] = parse_numeric_code(frame["periapsis_code"], r"r(\d+)", scale=10.0)
    frame["v_inf_kms"] = parse_numeric_code(frame["velocity_code"], r"v(\d+)", scale=10.0)
    frame["spin_period_hr"] = parse_numeric_code(frame["spin_code"], r"s(\d{3})", scale=10.0)
    frame["spin_axis"] = frame["spin_code"].fillna("").astype(str).str.extract(r"s\d{3}(m?)([xyz])")[1].fillna("none")
    frame["special_case_code"] = np.where(frame["mass_code"].fillna("").astype(str).str.contains("c30"), "c30", "none")
    frame["has_explicit_spin"] = frame["spin_code"].fillna("").astype(str).ne("")
    frame["target_mass_kg"] = np.power(10.0, frame["mass_log10_kg"])
    return add_physics_features(frame)


def add_physics_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    periapsis_km = enriched["periapsis_Rm"] * MARS_RADIUS_KM
    with np.errstate(divide="ignore", invalid="ignore"):
        enriched["encounter_eccentricity_proxy"] = 1.0 + (
            periapsis_km * np.square(enriched["v_inf_kms"])
        ) / MARS_MU_KM3_S2
        enriched["periapsis_inverse"] = 1.0 / enriched["periapsis_Rm"]
        enriched["spin_frequency_hr_inv"] = 1.0 / enriched["spin_period_hr"]
    enriched["v_inf_squared"] = np.square(enriched["v_inf_kms"])
    enriched["angular_momentum_proxy"] = enriched["periapsis_Rm"] * enriched["v_inf_kms"]
    enriched["has_spin"] = enriched["has_explicit_spin"].astype(int)
    enriched["particle_log10"] = np.log10(pd.to_numeric(enriched["resolution_value"], errors="coerce"))
    enriched["particle_mass_proxy"] = enriched["target_mass_kg"] / pd.to_numeric(
        enriched["resolution_value"], errors="coerce"
    )
    enriched["mass_resolution_interaction"] = enriched["mass_log10_kg"] - enriched["particle_log10"]
    return enriched.replace([np.inf, -np.inf], np.nan)


def make_bound_feature_frame(input_df: pd.DataFrame) -> pd.DataFrame:
    frame = add_derived_features(input_df)
    frame["particle_log10"] = pd.to_numeric(frame["resolution_value"], errors="coerce").map(
        lambda value: np.nan if pd.isna(value) else np.log10(value)
    )
    frame["special_case_code"] = frame.get("special_case_code", "none")
    frame["special_case_code"] = (
        pd.Series(frame["special_case_code"], index=frame.index).fillna("none").replace("", "none")
    )
    frame["target_mass_kg"] = np.power(10.0, pd.to_numeric(frame["mass_log10_kg"], errors="coerce"))
    return add_physics_features(frame)


@lru_cache(maxsize=1)
def load_demo_metadata() -> dict[str, object]:
    support = load_support_frame()
    trust_summary = pd.read_csv(SURROGATE_TABLES_DIR / "trust_summary.csv").iloc[0].to_dict()
    promoted_info = json.loads((SURROGATE_TABLES_DIR / "promoted_model_info.json").read_text(encoding="utf-8"))
    coverage_summary = pd.read_csv(SURROGATE_TABLES_DIR / "coverage_error_summary.csv").iloc[0].to_dict()

    ranges = {
        "mass_log10_kg": range_payload(support["mass_log10_kg"]),
        "periapsis_Rm": range_payload(support["periapsis_Rm"]),
        "v_inf_kms": range_payload(support["v_inf_kms"]),
        "spin_period_hr": range_payload(support["spin_period_hr"]),
        "resolution_value": range_payload(support["resolution_value"]),
        "fof_linking_length": range_payload(pd.to_numeric(support["fof_linking_length"], errors="coerce")),
        "timestep": range_payload(pd.to_numeric(support["timestep"], errors="coerce")),
    }
    return {
        "defaults": {
            "case_name": "demo_case_001",
            "mass_log10_kg": 20.0,
            "periapsis_Rm": 2.0,
            "v_inf_kms": 0.0,
            "has_explicit_spin": True,
            "spin_axis": "z",
            "spin_period_hr": 3.0,
            "resolution_value": 65.0,
            "timestep": 90000.0,
            "fof_linking_length": 0.004,
            "special_case_code": "none",
        },
        "ranges": ranges,
        "choices": {
            "spin_axis": ["x", "y", "z"],
            "special_case_code": ["none", "c30"],
            "common_mass_log10_kg": sorted(support["mass_log10_kg"].dropna().unique().tolist()),
            "common_resolution_value": sorted(support["resolution_value"].dropna().unique().tolist()),
        },
        "dataset_summary": {
            "bound_rows": int(len(support)),
            "mass_peri_bins": int(
                support.groupby(["mass_log10_kg", "periapsis_Rm"]).size().reset_index().shape[0]
            ),
            "occupied_mass_peri_bins": int(coverage_summary["occupied_mass_peri_bins"]),
            "occupied_peri_vel_bins": int(coverage_summary["occupied_peri_vel_bins"]),
        },
        "promoted_model": promoted_info,
        "trust_summary": trust_summary,
        "coverage_summary": coverage_summary,
        "deployability_note": (
            "The promoted physics-feature surrogate improved grouped-CV BMF performance, but its current ablation "
            "bundle includes a post-outcome feature. This demo therefore uses deployable direct-input models for "
            "screening, while still surfacing the physics-derived feature upgrade and trust rules."
        ),
    }


def range_payload(series: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return {"min": float(numeric.min()), "max": float(numeric.max())}


def build_input_frame(payload: dict[str, object]) -> pd.DataFrame:
    mass_log10_kg = float(payload["mass_log10_kg"])
    periapsis_rm = float(payload["periapsis_Rm"])
    v_inf_kms = float(payload["v_inf_kms"])
    resolution_value = float(payload["resolution_value"])
    timestep = float(payload["timestep"])
    fof_linking_length = float(payload["fof_linking_length"])
    has_explicit_spin = bool(payload.get("has_explicit_spin", True))
    spin_axis = str(payload.get("spin_axis", "none")) if has_explicit_spin else "none"
    spin_period_hr = payload.get("spin_period_hr")
    spin_period_hr = float(spin_period_hr) if has_explicit_spin and spin_period_hr not in (None, "") else np.nan
    special_case_code = str(payload.get("special_case_code", "none"))
    mass_code = f"A{int(round(mass_log10_kg * 100)):04d}"
    if special_case_code == "c30":
        mass_code = f"{mass_code}c30"
    resolution_code = f"n{int(round(resolution_value))}"

    row = {
        "case_name": str(payload.get("case_name", "custom_case")),
        "mass_log10_kg": mass_log10_kg,
        "mass_code": mass_code,
        "mass_value": int(round(mass_log10_kg * 100)),
        "periapsis_Rm": periapsis_rm,
        "periapsis_value": int(round(periapsis_rm * 10)),
        "v_inf_kms": v_inf_kms,
        "velocity_value": int(round(v_inf_kms * 10)),
        "spin_period_hr": spin_period_hr,
        "spin_value": int(round(spin_period_hr * 10)) if has_explicit_spin and not np.isnan(spin_period_hr) else np.nan,
        "spin_axis": spin_axis,
        "resolution_code": resolution_code,
        "resolution_value": resolution_value,
        "timestep": timestep,
        "fof_linking_length": fof_linking_length,
        "has_explicit_spin": has_explicit_spin,
        "special_case_code": special_case_code,
    }
    return pd.DataFrame([row])


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
            probability = float(model.predict_proba(X)[:, 1][0])
            result[f"{target}_classifier_probability"] = probability
        else:
            result[target] = float(model.predict(X)[0])

    bound_mass_fraction = float(np.clip(float(result.get("bound_mass_fraction", 0.0)), 0.0, 1.0))
    result["bound_mass_fraction"] = bound_mass_fraction
    result["has_any_bound_mass"] = bool(bound_mass_fraction > 0.0)
    result["bound_mass_fraction_ge_0p1"] = bool(bound_mass_fraction >= 0.1)
    return result


def add_spread_diagnostic(result: pd.Series, input_df: pd.DataFrame) -> pd.Series:
    spread_models = load_spread_models()
    if len(spread_models) < 2:
        result["bmf_model_spread"] = 0.0
        return result

    features = make_bound_feature_frame(input_df)
    predictions = []
    for model in spread_models.values():
        expected_columns = list(getattr(model.named_steps["preprocessor"], "feature_names_in_", []))
        X = features[expected_columns].copy()
        predictions.append(float(model.predict(X)[0]))
    result["bmf_model_spread"] = float(abs(predictions[0] - predictions[1]))
    return result


def classify_outcome(result: pd.Series) -> tuple[str, str]:
    largest_fragment_fraction = float(result.get("predicted_largest_fragment_mass_fraction", 1.0))
    fragmentation_probability = float(result.get("fragmentation_probability", 0.0))
    if fragmentation_probability >= 0.75 or largest_fragment_fraction < 0.2:
        return "Strong fragmentation", "Largest remnant is predicted to retain only a small share of the parent mass."
    if fragmentation_probability >= 0.35 or largest_fragment_fraction < 0.75:
        return "Fragmented", "Partial disruption is more likely than a clean mostly-intact remnant."
    return "Mostly intact", "The largest remnant remains dominant under the current surrogate estimate."


def classify_bound_retention(result: pd.Series) -> tuple[str, str]:
    probability = float(result.get("has_any_bound_mass_classifier_probability", 0.0))
    if probability >= 0.75:
        return "Likely bound retention", "Classifier signal strongly favours retained bound material."
    if probability >= 0.4:
        return "Possible limited retention", "Signal is mixed, so retained bound material is plausible but not secure."
    return "Low retention likelihood", "Classifier signal is weak for any substantial retained bound component."


def build_support_flags(result: pd.Series, input_df: pd.DataFrame) -> dict[str, object]:
    metadata = load_demo_metadata()
    support = load_support_frame()
    row = input_df.iloc[0]
    ranges = metadata["ranges"]

    in_training_range = (
        ranges["mass_log10_kg"]["min"] <= float(row["mass_log10_kg"]) <= ranges["mass_log10_kg"]["max"]
        and ranges["periapsis_Rm"]["min"] <= float(row["periapsis_Rm"]) <= ranges["periapsis_Rm"]["max"]
        and ranges["v_inf_kms"]["min"] <= float(row["v_inf_kms"]) <= ranges["v_inf_kms"]["max"]
    )
    near_training_edge = (
        float(row["periapsis_Rm"]) <= ranges["periapsis_Rm"]["min"] + 0.1
        or float(row["periapsis_Rm"]) >= ranges["periapsis_Rm"]["max"] - 0.1
        or float(row["v_inf_kms"]) <= ranges["v_inf_kms"]["min"] + 0.1
        or float(row["v_inf_kms"]) >= ranges["v_inf_kms"]["max"] - 0.1
    )

    bin_counts = support.groupby(["mass_log10_kg", "periapsis_Rm"]).size()
    key = (float(row["mass_log10_kg"]), float(row["periapsis_Rm"]))
    bin_count = int(bin_counts.get(key, 0))
    sparse_threshold = float(bin_counts.median()) if not bin_counts.empty else 0.0
    sparse_bin_flag = bin_count <= sparse_threshold
    borderline_bmf = 0.0771 <= float(result.get("bound_mass_fraction", 0.0)) <= 0.1229
    model_spread = float(result.get("bmf_model_spread", 0.0))
    spread_threshold = float(metadata["trust_summary"]["spread_threshold"])
    high_confidence = (
        in_training_range
        and not near_training_edge
        and not sparse_bin_flag
        and not borderline_bmf
        and model_spread <= spread_threshold
    )

    return {
        "in_training_range": in_training_range,
        "near_training_edge": near_training_edge,
        "sparse_bin_flag": sparse_bin_flag,
        "bin_count": bin_count,
        "sparse_threshold": sparse_threshold,
        "borderline_bmf": borderline_bmf,
        "model_spread": model_spread,
        "spread_threshold": spread_threshold,
        "high_confidence": high_confidence,
    }


def make_demo_recommendation(result: pd.Series, support_flags: dict[str, object]) -> tuple[str, str, str]:
    if not support_flags["in_training_range"]:
        return (
            "Full SPH required",
            "bad",
            "One or more core inputs sit outside the sampled training range, so this case should be treated as an extrapolation.",
        )
    if support_flags["near_training_edge"] or support_flags["sparse_bin_flag"]:
        return (
            "SPH recommended",
            "warn",
            "The case is near the sampled edge or in a sparsely supported bin, so the surrogate is better used as a prioritisation aid than as a stopping rule.",
        )
    if support_flags["borderline_bmf"] or support_flags["model_spread"] > support_flags["spread_threshold"]:
        return (
            "SPH recommended",
            "warn",
            "The retained-mass estimate is borderline or unstable across model families, which is exactly where a direct SPH run adds value.",
        )
    if float(result.get("bound_mass_fraction_ge_0_1_classifier_probability", 0.0)) >= 0.75:
        return (
            "SPH recommended",
            "warn",
            "The screening model indicates a strong bound-mass signal. That is scientifically interesting enough to justify a direct SPH follow-up.",
        )
    return (
        "ML screening sufficient",
        "ok",
        "This query is in-domain, not near a sparse edge bin, and does not trigger the current retained-mass caution flags.",
    )


def format_range(name: str, payload: dict[str, float]) -> str:
    if name == "timestep":
        return f"{payload['min']:.0f} to {payload['max']:.0f}"
    if name in {"resolution_value"}:
        return f"{payload['min']:.0f} to {payload['max']:.0f}"
    return f"{payload['min']:.4g} to {payload['max']:.4g}"


def build_physics_feature_cards(frame: pd.DataFrame) -> list[dict[str, str]]:
    row = frame.iloc[0]
    specs = [
        ("Encounter eccentricity proxy", row["encounter_eccentricity_proxy"], "{:.3f}"),
        ("v_inf squared", row["v_inf_squared"], "{:.3f}"),
        ("1 / periapsis", row["periapsis_inverse"], "{:.3f}"),
        ("Angular momentum proxy", row["angular_momentum_proxy"], "{:.3f}"),
        ("Particle mass proxy", row["particle_mass_proxy"], "{:.3e}"),
        ("Mass-resolution interaction", row["mass_resolution_interaction"], "{:.3f}"),
    ]
    cards = []
    for label, value, fmt in specs:
        if pd.isna(value):
            text = "n/a"
        else:
            text = fmt.format(float(value))
        cards.append({"label": label, "value": text})
    return cards


def build_response_payload(result: pd.Series, input_df: pd.DataFrame) -> dict[str, object]:
    metadata = load_demo_metadata()
    support_flags = build_support_flags(result, input_df)
    predicted_outcome, outcome_detail = classify_outcome(result)
    retention_label, retention_detail = classify_bound_retention(result)
    recommendation, recommendation_style, explanation = make_demo_recommendation(result, support_flags)
    support_frame = make_bound_feature_frame(input_df)
    domain = check_training_domain(support_frame.iloc[0].to_dict(), load_triage_bundle()[2])

    return {
        "predicted_outcome": predicted_outcome,
        "predicted_outcome_detail": outcome_detail,
        "fragmentation_probability_pct": round(float(result["fragmentation_probability"]) * 100.0, 1),
        "largest_fragment_percent": round(float(result["predicted_largest_fragment_mass_fraction"]) * 100.0, 1),
        "bound_retention_label": retention_label,
        "bound_retention_detail": retention_detail,
        "bound_retention_probability_pct": round(
            float(result.get("has_any_bound_mass_classifier_probability", 0.0)) * 100.0,
            1,
        ),
        "bound_mass_fraction": round(float(result.get("bound_mass_fraction", 0.0)), 3),
        "bound_mass_percent": round(float(result.get("bound_mass_fraction", 0.0)) * 100.0, 1),
        "bound_mass_threshold_probability_pct": round(
            float(result.get("bound_mass_fraction_ge_0_1_classifier_probability", 0.0)) * 100.0,
            1,
        ),
        "recommendation": recommendation,
        "recommendation_style": recommendation_style,
        "explanation": explanation,
        "domain_status": domain["status"],
        "domain_near_edge_features": domain["near_edge_features"],
        "domain_out_of_domain_features": domain["out_of_domain_features"],
        "support_flags": [
            {
                "label": "Training range",
                "value": "In range" if support_flags["in_training_range"] else "Outside range",
            },
            {
                "label": "Edge status",
                "value": "Near edge" if support_flags["near_training_edge"] else "Interior",
            },
            {
                "label": "Coverage bin",
                "value": f"{support_flags['bin_count']} runs",
            },
            {
                "label": "Model spread",
                "value": f"{support_flags['model_spread']:.4f}",
            },
        ],
        "physics_features": build_physics_feature_cards(support_frame),
        "screening_note": metadata["deployability_note"],
    }


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "SPHScreeningHTTP/2.0"

    def _send_bytes(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status)

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self._send_bytes(load_dashboard_html(), "text/html; charset=utf-8")
            return
        if self.path == "/api/metadata":
            self._send_json(load_demo_metadata())
            return
        if self.path == "/health":
            self._send_json({"status": "ok"})
            return
        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path != "/api/predict":
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
            input_df = build_input_frame(payload)
            classifier, regressor, training_domain = load_triage_bundle()
            result = predict_cases(input_df, classifier, regressor, training_domain).iloc[0].copy()
            result = apply_bound_predictions(result, input_df)
            result = add_spread_diagnostic(result, input_df)
            self._send_json(build_response_payload(result, input_df))
        except Exception as exc:  # pragma: no cover
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8000), DashboardHandler)
    print("Serving SPH screening demo at http://127.0.0.1:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
