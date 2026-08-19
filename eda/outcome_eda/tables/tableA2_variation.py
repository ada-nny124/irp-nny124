from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "report-table-figure" / "tables" / "tableA1_used_in_report.csv"

MODEL_SPECS = [
    ("Random Forest (raw features)", ROOT / "ml" / "trainingartifacts" / "raw_rf" / "main_bmf_raw_rf.pkl"),
    ("Random Forest (raw + physics features)", ROOT / "ml" / "trainingartifacts" / "physics_rf" / "main_bmf_physics_rf.pkl"),
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
    for label, path in MODEL_SPECS:
        bundle = load_bundle(path)
        metrics = bundle["grouped_cv_metrics"]
        feature_columns = bundle["feature_columns"]
        rows.append(
            {
                "Model": label,
                "R2": round(float(metrics["r2"]), 4),
                "MAE": round(float(metrics["mae"]), 4),
                "RMSE": round(float(metrics["rmse"]), 4),
                "Features": ", ".join(str(column) for column in feature_columns),
                "Dataset": str(bundle.get("dataset_path", "")),
                "Evaluation": "Grouped OOF CV",
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
