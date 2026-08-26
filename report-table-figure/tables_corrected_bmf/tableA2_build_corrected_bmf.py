#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_TABLES = ROOT / "report-table-figure" / "tables_corrected_bmf"
CORRECTED_ARTIFACTS = ROOT / "ml" / "trainingartifacts_corrected_bmf"
OUTPUT_PATH = OUTPUT_TABLES / "tableA2_used_in_report_corrected_bmf.csv"

MODEL_SPECS = [
    ("Gradient Boosting", CORRECTED_ARTIFACTS / "gradient_boosting" / "main_bmf_gradient_boosting_metrics.json"),
    ("Tuned Gradient Boosting", CORRECTED_ARTIFACTS / "tuned_gradient_boosting" / "main_bmf_tuned_gradient_boosting_metrics.json"),
    ("Random Forest", CORRECTED_ARTIFACTS / "raw_rf" / "main_bmf_raw_rf_metrics.json"),
    ("Tuned RF", CORRECTED_ARTIFACTS / "tuned_rf" / "main_bmf_tuned_rf_metrics.json"),
    ("RF + derived features", CORRECTED_ARTIFACTS / "physics_rf" / "main_bmf_physics_rf_metrics.json"),
    ("GB + derived features", CORRECTED_ARTIFACTS / "tuned_physics_gradient_boosting" / "main_bmf_tuned_physics_gradient_boosting_metrics.json"),
]


def load_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if {"r2", "mae", "rmse"} <= payload.keys():
        return {"r2": float(payload["r2"]), "mae": float(payload["mae"]), "rmse": float(payload["rmse"])}
    return {
        "r2": float(payload["grouped_cv_r2"]),
        "mae": float(payload["grouped_cv_mae_fraction"]),
        "rmse": float(payload["grouped_cv_rmse"]),
    }


def main() -> None:
    rows = []
    missing = []
    for label, path in MODEL_SPECS:
        if not path.exists():
            missing.append(path)
            continue
        metrics = load_metrics(path)
        rows.append(
            {
                "Model": label,
                "R2": round(float(metrics["r2"]), 4),
                "MAE": round(float(metrics["mae"]), 4),
                "RMSE": round(float(metrics["rmse"]), 4),
            }
        )
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing corrected-BMF metric files:\n{missing_text}")
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
