from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT_ENV = os.environ.get("PIPELINE_OUTPUT_ROOT")
OUTPUT_BASE = Path(OUTPUT_ROOT_ENV).resolve() if OUTPUT_ROOT_ENV else ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.model_training_scripts.helper_functions_ml import (
    add_physics_features,
    evaluate_grouped_oof_regression,
    load_canonical_dataset,
)


DATASET_PATH = ROOT / "extraction-outputs" / "tables" / "bound_outcomes.csv"
MODEL_BUNDLE_PATH = OUTPUT_BASE / "ml" / "trainingartifacts" / "tuned_gb" / "tuned_gb.pkl"
OUTPUT_PATH = OUTPUT_BASE / "report-table-figure" / "tables" / "section34_used_in_report.csv"

FAMILY_SPECS = [
    {
        "family": "Periapsis",
        "raw_parameter": "periapsis",
        "note": "Removes the direct closest-approach variable and any derived features that depend on periapsis.",
        "removed_features": [
            "periapsis_Rm",
            "periapsis_inverse",
            "angular_momentum_proxy",
            "encounter_eccentricity_proxy",
            "time_within_2_mars_radii_hr",
            "time_within_tidal_disruption_hr",
        ],
    },
    {
        "family": "Velocity",
        "raw_parameter": "v_inf",
        "note": "Removes encounter speed and any derived features that explicitly depend on velocity.",
        "removed_features": [
            "v_inf_kms",
            "v_inf_squared",
            "angular_momentum_proxy",
            "encounter_eccentricity_proxy",
            "time_within_2_mars_radii_hr",
            "time_within_tidal_disruption_hr",
        ],
    },
    {
        "family": "Spin",
        "raw_parameter": "spin",
        "note": "Removes spin magnitude, spin axis, and any spin-derived helper features.",
        "removed_features": [
            "spin_period_hr",
            "spin_axis",
            "spin_frequency_hr_inv",
            "has_explicit_spin",
            "has_spin",
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--bundle", type=Path, default=MODEL_BUNDLE_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def load_bundle(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def summarize_rule(frame: pd.DataFrame) -> str:
    order = frame.sort_values("delta_r2")
    ranking = " > ".join(order["raw_parameter"].tolist())
    ranking = ranking.replace("v_inf", "velocity")
    return ranking


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.bundle)
    frame = add_physics_features(load_canonical_dataset(args.dataset))
    feature_columns = list(bundle["feature_columns"])
    model_name = str(bundle["model_name"])
    if model_name.endswith("_rf") or "random_forest" in model_name:
        eval_model_name = "random_forest"
        params = bundle.get("rf_params", {})
    elif model_name.endswith("_gb") or "gradient_boosting" in model_name:
        eval_model_name = "gradient_boosting"
        params = bundle.get("gb_params", {})
    else:
        raise ValueError(f"Unsupported model bundle: {bundle['model_name']}")

    folds = pd.read_csv(Path(bundle["fold_assignments_path"]))
    baseline_metrics, _ = evaluate_grouped_oof_regression(frame, feature_columns, eval_model_name, params, folds)

    rows: list[dict[str, object]] = []
    for spec in FAMILY_SPECS:
        removed = [column for column in spec["removed_features"] if column in feature_columns]
        retained = [column for column in feature_columns if column not in removed]
        metrics, _ = evaluate_grouped_oof_regression(frame, retained, eval_model_name, params, folds)
        rows.append(
            {
                "family": spec["family"],
                "raw_parameter": spec["raw_parameter"],
                "note": spec["note"],
                "baseline_r2": float(baseline_metrics["r2"]),
                "baseline_mae": float(baseline_metrics["mae"]),
                "ablation_r2": float(metrics["r2"]),
                "ablation_mae": float(metrics["mae"]),
                "delta_r2": float(metrics["r2"]) - float(baseline_metrics["r2"]),
                "delta_mae": float(metrics["mae"]) - float(baseline_metrics["mae"]),
                "removed_features": str(removed),
                "retained_feature_count": len(retained),
            }
        )

    result = pd.DataFrame(rows).sort_values("delta_r2")
    overall_rule = summarize_rule(result)
    summary = {
        "model_bundle": str(args.bundle),
        "model_name": bundle["model_name"],
        "baseline_r2": float(baseline_metrics["r2"]),
        "baseline_mae": float(baseline_metrics["mae"]),
        "overall_rule": overall_rule,
        "display_lines": [
            f"Removing periapsis: ΔR² = {result.loc[result['raw_parameter'] == 'periapsis', 'delta_r2'].iloc[0]:+.3f}",
            f"Removing velocity: ΔR² = {result.loc[result['raw_parameter'] == 'v_inf', 'delta_r2'].iloc[0]:+.3f}",
            f"Removing spin: ΔR² = {result.loc[result['raw_parameter'] == 'spin', 'delta_r2'].iloc[0]:+.3f}",
            f"Overall Rule: {overall_rule}",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(args.output)
    for line in summary["display_lines"]:
        print(line)


if __name__ == "__main__":
    main()
