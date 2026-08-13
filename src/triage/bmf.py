"""Deployment helpers for the continuous BMF CatBoost hurdle model."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LEAKY_FEATURES = {"largest_fragment_mass_fraction"}


@dataclass(frozen=True)
class BmfPrediction:
    positive_probability: np.ndarray
    positive_estimate: np.ndarray
    final_prediction: np.ndarray


def _bundle_path(model_dir: str | Path) -> Path:
    return Path(model_dir) / "bmf_hurdle_bundle.pkl"


def load_bmf_bundle(model_dir: str | Path) -> dict[str, Any]:
    bundle_path = _bundle_path(model_dir)
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Missing required BMF CatBoost hurdle bundle at {bundle_path}. "
            "Run the deployed BMF training/build step before starting the dashboard."
        )
    with bundle_path.open("rb") as handle:
        bundle = pickle.load(handle)
    feature_columns = list(bundle.get("feature_columns", []))
    leaky = sorted(LEAKY_FEATURES.intersection(feature_columns))
    if leaky:
        raise ValueError(f"Leaky deployed BMF bundle: forbidden features present: {', '.join(leaky)}")
    return bundle


def get_bmf_bundle_status(model_dir: str | Path) -> dict[str, str]:
    bundle_path = _bundle_path(model_dir)
    return {
        "label": "bmf_hurdle_bundle.pkl",
        "path": str(bundle_path),
        "status": "loaded" if bundle_path.exists() else "missing",
        "target": "continuous bound mass fraction via two-stage CatBoost hurdle",
    }


def get_bmf_feature_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"BMF feature frame is missing required columns: {', '.join(sorted(missing))}")
    return frame.loc[:, feature_columns].copy()


def predict_bmf_from_bundle(bundle: dict[str, Any], feature_frame: pd.DataFrame) -> BmfPrediction:
    feature_columns = list(bundle["feature_columns"])
    categorical_columns = list(bundle.get("categorical_columns", []))
    classifier = bundle["zero_vs_positive_classifier"]
    regressor = bundle["positive_only_regressor"]
    X = get_bmf_feature_frame(feature_frame, feature_columns)
    cat_indices = [X.columns.get_loc(column) for column in categorical_columns if column in X.columns]
    positive_probability = np.asarray(classifier.predict_proba(X)[:, 1], dtype=float)
    positive_estimate = np.asarray(regressor.predict(X), dtype=float)
    final_prediction = np.clip(positive_probability * positive_estimate, 0.0, 1.0)
    return BmfPrediction(
        positive_probability=positive_probability,
        positive_estimate=positive_estimate,
        final_prediction=final_prediction,
    )


def build_training_domain_metadata(frame: pd.DataFrame, feature_columns: list[str], categorical_columns: list[str]) -> dict[str, Any]:
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]
    numeric: dict[str, Any] = {}
    for column in numeric_columns:
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            continue
        numeric[column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
        }

    categorical: dict[str, Any] = {}
    for column in categorical_columns:
        values = frame[column].fillna("missing").astype(str)
        counts = values.value_counts().sort_index()
        categorical[column] = {
            "allowed": counts.index.tolist(),
            "counts": {key: int(value) for key, value in counts.items()},
        }
    return {"numeric": numeric, "categorical": categorical}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
