from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "report-table-figure" / "tables" / "tableA2_used_in_report.csv"

MODEL_SPECS = [
    ("Gradient Boosting", ROOT / "ml" / "trainingartifacts" / "gradient_boosting" / "main_bmf_gradient_boosting_metrics.json"),
    ("Tuned Gradient Boosting", ROOT / "ml" / "trainingartifacts" / "tuned_gradient_boosting" / "main_bmf_tuned_gradient_boosting_metrics.json"),
    ("Random Forest", ROOT / "ml" / "trainingartifacts" / "raw_rf" / "main_bmf_raw_rf_metrics.json"),
    ("Tuned RF", ROOT / "ml" / "trainingartifacts" / "tuned_rf" / "main_bmf_tuned_rf_metrics.json"),
    ("RF + derived features", ROOT / "ml" / "trainingartifacts" / "physics_rf" / "main_bmf_physics_rf_metrics.json"),
    ("GB + derived features", ROOT / "ml" / "trainingartifacts" / "tuned_physics_gradient_boosting" / "main_bmf_tuned_physics_gradient_boosting_metrics.json"),
    ("XGBoost regressor", ROOT / "archived" / "ml" / "model_optimization_candidates" / "tables" / "candidate_model_summary.csv", "xgboost"),
    ("Hurdle NGBoost surrogate", ROOT / "archived" / "ml" / "model_optimization_candidates" / "advanced" / "tables" / "advanced_model_summary.csv", "hurdle_ngboost"),
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
