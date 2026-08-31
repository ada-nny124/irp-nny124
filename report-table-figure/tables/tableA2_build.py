from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT_ENV = os.environ.get("PIPELINE_OUTPUT_ROOT")
OUTPUT_BASE = Path(OUTPUT_ROOT_ENV).resolve() if OUTPUT_ROOT_ENV else ROOT
OUTPUT_PATH = OUTPUT_BASE / "report-table-figure" / "tables" / "tableA2_used_in_report.csv"

MODEL_SPECS = [
    ("Raw GB", OUTPUT_BASE / "ml" / "trainingartifacts" / "raw_gb" / "raw_gb_metrics.json"),
    ("Tuned GB", OUTPUT_BASE / "ml" / "trainingartifacts" / "tuned_gb" / "tuned_gb_metrics.json"),
    ("Raw RF", OUTPUT_BASE / "ml" / "trainingartifacts" / "raw_rf" / "raw_rf_metrics.json"),
    ("Tuned RF", OUTPUT_BASE / "ml" / "trainingartifacts" / "tuned_rf" / "tuned_rf_metrics.json"),
    ("Derived RF", OUTPUT_BASE / "ml" / "trainingartifacts" / "derived_rf" / "derived_rf_metrics.json"),
    ("Derived GB", OUTPUT_BASE / "ml" / "trainingartifacts" / "derived_gb" / "derived_gb_metrics.json")
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def load_bundle(path: Path) -> dict[str, object] | None:
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def load_fallback_metrics(source_path: Path, lookup_key: str | None = None) -> dict[str, float]:
    if source_path.suffix == ".csv":
        frame = pd.read_csv(source_path)
        key = lookup_key or source_path.stem
        if "model_key" in frame.columns:
            row = frame.loc[frame["model_key"] == key]
        elif "model_label" in frame.columns:
            row = frame.loc[frame["model_label"] == key]
        else:
            raise ValueError(f"Unsupported summary CSV format: {source_path}")
        if row.empty:
            raise ValueError(f"No summary row for {key!r} in {source_path}")
        row = row.iloc[0]
        return {"r2": float(row["r2"]), "mae": float(row["mae"]), "rmse": float(row["rmse"])}

    if source_path.suffix == ".json":
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        if {"r2", "mae", "rmse"} <= payload.keys():
            return {"r2": float(payload["r2"]), "mae": float(payload["mae"]), "rmse": float(payload["rmse"])}
        return {
            "r2": float(payload["grouped_cv_r2"]),
            "mae": float(payload["grouped_cv_mae_fraction"]),
            "rmse": float(payload["grouped_cv_rmse"]),
        }

    if source_path.suffix == ".pkl":
        raise ValueError(f"Legacy .pkl model bundle is incompatible with the current sklearn version: {source_path}")

    raise ValueError(f"Unsupported metric source: {source_path}")


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in MODEL_SPECS:
        label = item[0]
        path = item[1]
        lookup_key = item[2] if len(item) > 2 else None

        if path.exists() and path.suffix == ".pkl":
            bundle = load_bundle(path)
            if bundle is None or "grouped_cv_metrics" not in bundle:
                raise ValueError(f"No compatible metrics available for {label!r} at {path}")
            metrics = bundle["grouped_cv_metrics"]
        else:
            metrics = load_fallback_metrics(path, lookup_key)

        rows.append(
            {
                "Model": label,
                "R2": round(float(metrics["r2"]), 4),
                "MAE": round(float(metrics["mae"]), 4),
                "RMSE": round(float(metrics["rmse"]), 4),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    frame = pd.DataFrame(build_rows())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
