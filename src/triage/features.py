"""Feature preparation for the SPH fragmentation triage tool.

This module prepares FoF-derived proxy features for both training and inference.
The targets created downstream are surrogate labels based on FoF fragment tables
and do not directly validate bound disk material or moon formation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RAW_REQUIRED_COLUMNS = [
    "mass_value",
    "periapsis_value",
    "velocity_value",
    "resolution_code",
    "resolution_value",
    "timestep",
    "fof_linking_length",
]

MODEL_FEATURE_COLUMNS = [
    "mass_log10_kg",
    "mass_code",
    "periapsis_Rm",
    "v_inf_kms",
    "spin_period_hr",
    "spin_axis",
    "resolution_code",
    "resolution_value",
    "timestep",
    "fof_linking_length",
    "eccentricity_proxy",
    "low_periapsis_flag",
    "high_velocity_flag",
    "has_explicit_spin",
]

CATEGORICAL_FEATURE_COLUMNS = ["mass_code", "spin_axis", "resolution_code"]
NUMERIC_FEATURE_COLUMNS = [column for column in MODEL_FEATURE_COLUMNS if column not in CATEGORICAL_FEATURE_COLUMNS]
DOMAIN_NUMERIC_FEATURE_COLUMNS = [
    "mass_log10_kg",
    "periapsis_Rm",
    "v_inf_kms",
    "spin_period_hr",
    "timestep",
    "fof_linking_length",
]
DOMAIN_CATEGORICAL_FEATURE_COLUMNS = [
    "spin_axis",
    "resolution_code",
    "has_explicit_spin",
    "high_velocity_flag",
    "low_periapsis_flag",
]

MARS_MU_KM3_S2 = 4.282837e4
MARS_RADIUS_KM = 3389.5


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_text}")


def load_fof_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    validate_required_columns(df, RAW_REQUIRED_COLUMNS)
    return df


def _coerce_bool(series: pd.Series) -> pd.Series:
    normalised = series.where(series.notna(), False)
    return normalised.astype(str).str.lower().isin({"true", "1", "yes"})


def _scaled_eccentricity_proxy(periapsis_rm: pd.Series, v_inf_kms: pd.Series) -> pd.Series:
    periapsis_km = periapsis_rm * MARS_RADIUS_KM
    with np.errstate(divide="ignore", invalid="ignore"):
        proxy = 1.0 + (periapsis_km * np.square(v_inf_kms)) / MARS_MU_KM3_S2
    return proxy.replace([np.inf, -np.inf], np.nan)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()

    if "mass_log10_kg" not in frame.columns:
        if "mass_value" in frame.columns:
            frame["mass_log10_kg"] = pd.to_numeric(frame["mass_value"], errors="coerce") / 100.0
        else:
            frame["mass_log10_kg"] = np.nan

    if "periapsis_Rm" not in frame.columns:
        if "periapsis_value" in frame.columns:
            frame["periapsis_Rm"] = pd.to_numeric(frame["periapsis_value"], errors="coerce") / 10.0
        else:
            frame["periapsis_Rm"] = np.nan

    if "v_inf_kms" not in frame.columns:
        if "velocity_value" in frame.columns:
            frame["v_inf_kms"] = pd.to_numeric(frame["velocity_value"], errors="coerce") / 10.0
        else:
            frame["v_inf_kms"] = np.nan

    if "spin_period_hr" not in frame.columns:
        if "spin_value" in frame.columns:
            frame["spin_period_hr"] = pd.to_numeric(frame["spin_value"], errors="coerce") / 10.0
        else:
            frame["spin_period_hr"] = np.nan

    if "resolution_value" in frame.columns:
        frame["resolution_value"] = pd.to_numeric(frame["resolution_value"], errors="coerce")
    else:
        frame["resolution_value"] = np.nan

    if "timestep" in frame.columns:
        frame["timestep"] = pd.to_numeric(frame["timestep"], errors="coerce")
    else:
        frame["timestep"] = np.nan

    if "fof_linking_length" in frame.columns:
        frame["fof_linking_length"] = pd.to_numeric(frame["fof_linking_length"], errors="coerce")
    else:
        frame["fof_linking_length"] = np.nan

    frame["mass_code"] = frame.get("mass_code", pd.Series(index=frame.index, dtype="object")).fillna("unknown")
    frame["spin_axis"] = frame.get("spin_axis", pd.Series(index=frame.index, dtype="object")).fillna("none").replace("", "none")
    frame["resolution_code"] = frame.get("resolution_code", pd.Series(index=frame.index, dtype="object")).fillna("unknown")

    has_explicit = frame.get("has_explicit_spin", pd.Series(index=frame.index, dtype="object"))
    frame["has_explicit_spin"] = _coerce_bool(has_explicit)

    frame["eccentricity_proxy"] = _scaled_eccentricity_proxy(frame["periapsis_Rm"], frame["v_inf_kms"])
    frame["low_periapsis_flag"] = frame["periapsis_Rm"].le(1.4).astype(int)
    frame["high_velocity_flag"] = frame["v_inf_kms"].ge(1.5).astype(int)
    return frame


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = add_derived_features(df)
    missing = [column for column in MODEL_FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Feature preparation failed to create columns: {missing_text}")
    return frame[MODEL_FEATURE_COLUMNS].copy()
