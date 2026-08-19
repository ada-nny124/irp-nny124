from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "report-table-figure" / "tables" / "tableA2_used_in_report.csv"

MODEL_SPECS = [
    ("Gradient Boosting", ROOT / "ml" / "trainingartifacts" / "gradient_boosting" / "main_bmf_gradient_boosting.pkl"),
    ("Tuned Gradient Boosting", ROOT / "ml" / "trainingartifacts" / "tuned_gradient_boosting" / "main_bmf_tuned_gradient_boosting.pkl"),
    ("Random Forest", ROOT / "ml" / "trainingartifacts" / "raw_rf" / "main_bmf_raw_rf.pkl"),
    ("Tuned RF", ROOT / "ml" / "trainingartifacts" / "tuned_rf" / "main_bmf_tuned_rf.pkl"),
    ("RF + derived features", ROOT / "ml" / "trainingartifacts" / "physics_rf" / "main_bmf_physics_rf.pkl"),
    ("GB + derived features", ROOT / "ml" / "trainingartifacts" / "tuned_physics_gradient_boosting" / "main_bmf_tuned_physics_gradient_boosting.pkl"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def load_bundle(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    baseline_bundle = load_bundle(ROOT / "ml" / "trainingartifacts" / "raw_rf" / "main_bmf_raw_rf.pkl")
    baseline = baseline_bundle["grouped_cv_metrics"]
    baseline_r2 = float(baseline["r2"])
    baseline_mse = float(baseline["mse"])
    baseline_rmse = float(baseline["rmse"])

    for label, path in MODEL_SPECS:
        bundle = load_bundle(path)
        metrics = bundle["grouped_cv_metrics"]
        rows.append(
            {
                "Model": label,
                "R2": round(float(metrics["r2"]), 4),
                "MSE": round(float(metrics["mse"]), 4),
                "RMSE": round(float(metrics["rmse"]), 4),
                "Delta_R2_vs_RF": round(float(metrics["r2"]) - baseline_r2, 4),
                "Delta_MSE_vs_RF": round(float(metrics["mse"]) - baseline_mse, 4),
                "Delta_RMSE_vs_RF": round(float(metrics["rmse"]) - baseline_rmse, 4),
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
