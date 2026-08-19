from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "report-table-figure" / "tables" / "table2_used_in_report.csv"
OOF_PATH = ROOT / "ml" / "trainingartifacts" / "tuned_rf" / "main_bmf_tuned_rf_oof_predictions.csv"

TARGET_MASS_LOG10_KG = 19.5
TARGET_PERIAPSIS = [1.2, 1.6]
TARGET_V_INF_KMS = 0.0
TARGET_RESOLUTION = 65


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--oof", type=Path, default=OOF_PATH)
    return parser.parse_args()


def load_sparse_failure_rows(oof_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(oof_path)
    subset = frame.loc[
        (frame["mass_log10_kg"] == TARGET_MASS_LOG10_KG)
        & (frame["periapsis_Rm"].isin(TARGET_PERIAPSIS))
        & (frame["v_inf_kms"] == TARGET_V_INF_KMS)
        & (frame["resolution_value"] == TARGET_RESOLUTION)
        & (frame["spin_axis"] == "none")
    ].copy()

    if subset.empty:
        raise ValueError("No matching OOF rows found for the sparse-support Table 2 case.")

    grouped = (
        subset.groupby(["mass_log10_kg", "periapsis_Rm"], as_index=False)
        .agg(
            actual_bmf=("actual_bmf", "mean"),
            oof_prediction=("predicted_bmf", "mean"),
            abs_error=("residual", lambda s: s.abs().mean()),
            post_processing_rows=("physical_file", "size"),
            unique_physical_sims=("physical_file", "nunique"),
        )
        .sort_values("periapsis_Rm")
    )
    grouped["Failure"] = grouped.apply(
        lambda row: "Underpredicts" if row["oof_prediction"] < row["actual_bmf"] else "Overpredicts",
        axis=1,
    )
    return grouped


def build_report_table(oof_path: Path) -> pd.DataFrame:
    grouped = load_sparse_failure_rows(oof_path)
    report = grouped.rename(
        columns={
            "periapsis_Rm": "Periapsis",
            "actual_bmf": "Actual BMF",
            "oof_prediction": "OOF prediction",
            "abs_error": "Abs. error",
        }
    )[
        [
            "Periapsis",
            "Actual BMF",
            "OOF prediction",
            "Abs. error",
            "Failure",
        ]
    ].copy()

    report["Periapsis"] = report["Periapsis"].map(lambda value: f"{value:.1f}")
    report["Actual BMF"] = report["Actual BMF"].map(lambda value: f"{value:.6f}".rstrip("0").rstrip("."))
    report["OOF prediction"] = report["OOF prediction"].map(lambda value: f"{value:.4f}")
    report["Abs. error"] = report["Abs. error"].map(lambda value: f"{value:.4f}")
    return report


def main() -> None:
    args = parse_args()
    frame = build_report_table(args.oof)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
