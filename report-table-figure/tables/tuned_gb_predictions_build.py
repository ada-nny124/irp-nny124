from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BOUND_SOURCE = ROOT / "extraction-outputs" / "tables" / "bound_outcomes.csv"
OOF_SOURCE = ROOT / "ml" / "trainingartifacts" / "tuned_gradient_boosting" / "main_bmf_tuned_gradient_boosting_oof_predictions.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "tuned_gb_oof_predictions.csv"


def parse_numeric_code(series: pd.Series, pattern: str, scale: float = 1.0) -> pd.Series:
    extracted = series.fillna("").astype(str).str.extract(pattern)[0]
    return pd.to_numeric(extracted, errors="coerce") / scale


def parse_spin_period(series: pd.Series) -> pd.Series:
    extracted = series.fillna("").astype(str).str.extract(r"s(\d{3})")[0]
    values = pd.to_numeric(extracted, errors="coerce") / 10.0
    return values.where(series.fillna("") != "", pd.NA)


def parse_spin_axis(series: pd.Series) -> pd.Series:
    axis = series.fillna("").astype(str).str.extract(r"s\d{3}([A-Za-z]*)")[0].fillna("")
    return axis.replace("", "none")


def load_bound_frame() -> pd.DataFrame:
    frame = pd.read_csv(BOUND_SOURCE, low_memory=False)
    frame["mass_log10_kg"] = parse_numeric_code(frame["mass_code"], r"A(\d+)", 100.0)
    frame["periapsis_Rm"] = parse_numeric_code(frame["periapsis_code"], r"r(\d+)", 10.0)
    frame["v_inf_kms"] = parse_numeric_code(frame["velocity_code"], r"v(\d+)", 10.0)
    frame["spin_period_hr"] = parse_spin_period(frame["spin_code"])
    frame["spin_axis"] = parse_spin_axis(frame["spin_code"])
    frame["resolution_value"] = parse_numeric_code(frame["resolution_code"], r"n(\d+)")
    frame["actual_bmf"] = pd.to_numeric(frame["bound_mass_fraction"], errors="coerce")
    return frame


def build_prediction_table() -> pd.DataFrame:
    bound = load_bound_frame()
    oof = pd.read_csv(OOF_SOURCE, low_memory=False)
    oof["actual_bmf"] = pd.to_numeric(oof["actual_bmf"], errors="coerce")
    oof["predicted_bmf"] = pd.to_numeric(oof["predicted_bmf"], errors="coerce")
    oof["residual"] = pd.to_numeric(oof["residual"], errors="coerce")

    merge_cols = [
        "physical_file",
        "mass_log10_kg",
        "periapsis_Rm",
        "v_inf_kms",
        "spin_period_hr",
        "spin_axis",
        "resolution_value",
        "fof_linking_length",
    ]
    extra_cols = [
        "fof_file",
        "mass_code",
        "resolution_code",
        "periapsis_code",
        "velocity_code",
        "spin_code",
        "timestep",
        "bound_mass_fraction",
        "target_mass_kg",
        "captured_mass_fraction",
    ]
    bound_lookup = bound[merge_cols + extra_cols].drop_duplicates(subset=merge_cols, keep="first")
    merged = oof.merge(bound_lookup, on=merge_cols, how="left", validate="many_to_one")
    merged["abs_error"] = merged["residual"].abs()
    merged["target"] = "bound_mass_fraction"
    merged["model"] = "tuned_gradient_boosting"
    merged["transform"] = "identity"
    merged["predicted"] = merged["predicted_bmf"]
    merged["bound_mass_fraction"] = merged["actual_bmf"]
    return merged[
        [
            "fof_file",
            "physical_file",
            "mass_code",
            "resolution_code",
            "periapsis_code",
            "velocity_code",
            "spin_code",
            "timestep",
            "fof_linking_length",
            "mass_log10_kg",
            "periapsis_Rm",
            "v_inf_kms",
            "spin_period_hr",
            "spin_axis",
            "resolution_value",
            "target_mass_kg",
            "captured_mass_fraction",
            "bound_mass_fraction",
            "actual_bmf",
            "predicted_bmf",
            "predicted",
            "residual",
            "abs_error",
            "fold_index",
            "target",
            "model",
            "transform",
        ]
    ].sort_values(
        [
            "mass_log10_kg",
            "periapsis_Rm",
            "v_inf_kms",
            "spin_axis",
            "spin_period_hr",
            "resolution_value",
            "fof_linking_length",
        ]
    ).reset_index(drop=True)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table = build_prediction_table()
    table.to_csv(OUTPUT_PATH, index=False)
    print(f"Rows: {len(table)}")
    print(f"CSV: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
