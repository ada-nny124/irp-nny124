"""Minimal local web app for the SPH fragmentation triage dashboard."""

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

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irp_triage import add_derived_features, load_artifacts, predict_cases

MODEL_DIR = ROOT / "ml" / "triage"
BOUND_MODELS_DIR = ROOT / "ml" / "bound_outcomes" / "models"
BOUND_TABLES_DIR = ROOT / "ml" / "bound_outcomes" / "tables"
HTML_PATH = ROOT / "templates" / "sph_triage_dashboard.html"


@lru_cache(maxsize=1)
def load_dashboard_html() -> bytes:
    return HTML_PATH.read_bytes()


@lru_cache(maxsize=1)
def load_triage_bundle() -> tuple[object, object, dict[str, object]]:
    artifacts = load_artifacts(MODEL_DIR)
    if artifacts is None:
        raise FileNotFoundError("Missing required triage artifacts in ml/triage")
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
            selected[target] = BOUND_MODELS_DIR / f"{row['dataset']}__{row['feature_set']}__{row['target']}__{row['model']}.pkl"
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
            selected[target] = BOUND_MODELS_DIR / f"{row['dataset']}__{row['feature_set']}__{row['target']}__{row['model']}.pkl"
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


def make_bound_feature_frame(input_df: pd.DataFrame) -> pd.DataFrame:
    frame = add_derived_features(input_df)
    frame["particle_log10"] = pd.to_numeric(frame["resolution_value"], errors="coerce").map(
        lambda value: np.nan if pd.isna(value) else np.log10(value)
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
            probability = float(model.predict_proba(X)[:, 1][0])
            result[f"{target}_classifier_probability"] = probability
        else:
            result[target] = float(model.predict(X)[0])

    try:
        bound_mass_fraction = min(max(float(result.get("bound_mass_fraction", 0.0)), 0.0), 1.0)
        result["bound_mass_fraction"] = bound_mass_fraction
        result["has_any_bound_mass"] = bool(bound_mass_fraction > 0.0)
        result["bound_mass_fraction_ge_0p1"] = bool(bound_mass_fraction >= 0.1)
    except (TypeError, ValueError):
        pass
    return result


def build_input_frame(payload: dict[str, object]) -> pd.DataFrame:
    mass_log10_kg = float(payload["mass_log10_kg"])
    has_explicit_spin = bool(payload.get("has_explicit_spin", True))
    spin_axis = str(payload.get("spin_axis", "none")) if has_explicit_spin else "none"
    spin_period_hr = payload.get("spin_period_hr")
    spin_period_hr = float(spin_period_hr) if has_explicit_spin and spin_period_hr not in (None, "") else None
    resolution_code = str(payload["resolution_code"])
    resolution_value = float(str(resolution_code).replace("n", ""))
    mass_code = str(payload.get("mass_code") or f"A{int(round(mass_log10_kg * 100)):04d}")
    row = {
        "case_name": str(payload.get("case_name", "custom_case")),
        "mass_log10_kg": mass_log10_kg,
        "mass_code": mass_code,
        "periapsis_Rm": float(payload["periapsis_Rm"]),
        "v_inf_kms": float(payload["v_inf_kms"]),
        "spin_period_hr": spin_period_hr,
        "spin_axis": spin_axis,
        "resolution_code": resolution_code,
        "resolution_value": resolution_value,
        "timestep": float(payload["timestep"]),
        "fof_linking_length": float(payload["fof_linking_length"]),
        "has_explicit_spin": has_explicit_spin,
    }
    return pd.DataFrame([row])


def risk_to_card_style(risk_label: str) -> tuple[str, str]:
    mapping = {
        "low": ("Low", "ok"),
        "medium": ("Medium", "warn"),
        "high": ("High", "warn"),
        "very high": ("Very high", "bad"),
    }
    return mapping.get(risk_label, (risk_label.title(), "warn"))


def severity_to_text(severity_class: str) -> str:
    return severity_class.replace("_", " ")


def domain_to_label(domain_status: str) -> tuple[str, str]:
    mapping = {
        "in_domain": ("In domain", "ok"),
        "near_edge": ("Near edge", "warn"),
        "out_of_domain": ("Out of domain", "bad"),
    }
    return mapping.get(domain_status, (domain_status.replace("_", " "), "warn"))


def build_triggers(result: pd.Series) -> list[str]:
    triggers: list[str] = []
    if float(result.get("fragmentation_probability", 0.0)) >= 0.75:
        triggers.append("High fragmentation likelihood")
    try:
        if float(result.get("bound_mass_fraction", 0.0)) >= 0.1:
            triggers.append("BMF >= 10% possible")
    except (TypeError, ValueError):
        pass
    if str(result.get("domain_status", "")) == "near_edge":
        triggers.append("Near-edge domain status")
    if str(result.get("domain_status", "")) == "out_of_domain":
        triggers.append("Out-of-domain extrapolation")
    if not triggers:
        triggers.append("No risk flags triggered")
    return triggers


def build_diagnostics(result: pd.Series) -> list[dict[str, str]]:
    periapsis = float(result.get("periapsis_Rm", 0.0))
    velocity = float(result.get("v_inf_kms", 0.0))
    mass = float(result.get("mass_log10_kg", 0.0))
    spin_period = result.get("spin_period_hr")
    eccentricity_proxy = float(result.get("eccentricity_proxy", np.nan))
    domain_label, _ = domain_to_label(str(result.get("domain_status", "")))
    return [
        {
            "check": "Periapsis",
            "value": f"{periapsis:.2f} Rm",
            "interpretation": "Close encounter, stronger tidal forcing" if periapsis < 1.5 else "Wider approach, weaker tidal forcing",
        },
        {
            "check": "Velocity",
            "value": f"{velocity:.2f} km/s",
            "interpretation": "Slow encounter, longer interaction time" if velocity < 1.0 else "Faster flyby, shorter interaction time",
        },
        {
            "check": "Mass",
            "value": f"10^{mass:.1f} kg",
            "interpretation": "Within sampled range" if 18.0 <= mass <= 21.0 else "Near sampled range boundary",
        },
        {
            "check": "Spin period",
            "value": f"{float(spin_period):.1f} hr" if spin_period is not None and not pd.isna(spin_period) else "disabled",
            "interpretation": "May influence disruption dynamics" if spin_period is not None and not pd.isna(spin_period) else "Spin not active in this case",
        },
        {
            "check": "Eccentricity proxy",
            "value": f"{eccentricity_proxy:.2f}" if np.isfinite(eccentricity_proxy) else "n/a",
            "interpretation": "Approximate orbital energy regime",
        },
        {
            "check": "Domain status",
            "value": domain_label,
            "interpretation": "Prediction is well supported" if domain_label == "In domain" else "Prediction should be validated with SPH",
        },
    ]


def build_response_payload(result: pd.Series) -> dict[str, object]:
    fragmentation_probability = float(result["fragmentation_probability"])
    predicted_largest_fragment_mass_fraction = float(result["predicted_largest_fragment_mass_fraction"])
    bound_mass_fraction = result.get("bound_mass_fraction")
    if bound_mass_fraction is None or pd.isna(bound_mass_fraction):
        bound_mass_fraction = 0.0
    bound_mass_fraction = float(bound_mass_fraction)

    risk_label = str(result.get("risk_label", "medium"))
    risk_text, risk_style = risk_to_card_style(risk_label)
    domain_text, domain_style = domain_to_label(str(result.get("domain_status", "near_edge")))
    severity_class = str(result.get("severity_class", "not_available_yet"))

    return {
        "fragmentation_probability": fragmentation_probability,
        "fragmentation_probability_pct": round(fragmentation_probability * 100.0, 1),
        "risk_label": risk_label,
        "risk_text": risk_text,
        "risk_style": risk_style,
        "largest_fragment_mass_fraction": round(predicted_largest_fragment_mass_fraction, 3),
        "largest_fragment_percent": round(predicted_largest_fragment_mass_fraction * 100.0, 1),
        "severity_class": severity_class,
        "severity_text": severity_to_text(severity_class),
        "bound_mass_fraction": round(bound_mass_fraction, 3),
        "bound_mass_percent": round(bound_mass_fraction * 100.0, 1),
        "bound_probability_text": f"Derived from bound mass fraction = {bound_mass_fraction:.3f}",
        "domain_status": str(result.get("domain_status", "near_edge")),
        "domain_text": domain_text,
        "domain_style": domain_style,
        "decision": str(result.get("sph_recommendation", "rank with ML, confirm with SPH if needed")),
        "explanation": str(result.get("explanation", "")),
        "triggers": build_triggers(result),
        "diagnostics": build_diagnostics(result),
        "narrative": str(result.get("explanation", "")),
    }


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "SPHTriageHTTP/1.0"

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
            self._send_json(build_response_payload(result))
        except Exception as exc:  # pragma: no cover - defensive server path
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8000), DashboardHandler)
    print("Serving SPH triage dashboard at http://127.0.0.1:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
