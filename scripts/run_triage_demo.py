#!/usr/bin/env python3
"""Run the SPH fragmentation triage demo locally from an editable template file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irp_triage.predict import load_artifacts, predict_cases


DEFAULT_TEMPLATE = ROOT / "templates" / "triage_case_template.json"
DEFAULT_MODEL_DIR = ROOT / "ml" / "triage"
DEFAULT_OUTPUT = ROOT / "outputs" / "triage_demo_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_TEMPLATE, help="Path to a JSON or CSV file containing one or more triage cases.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Directory containing trained triage model artifacts.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where to save the prediction table as CSV.")
    parser.add_argument("--print-template", action="store_true", help="Print the expected template structure and exit.")
    return parser.parse_args()


def load_cases(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            data = [data]
        return pd.DataFrame(data)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path.suffix}. Use .json or .csv")


def print_template() -> None:
    template = json.loads(DEFAULT_TEMPLATE.read_text())
    print(json.dumps(template, indent=2))


def print_case_summary(result: pd.DataFrame) -> None:
    for idx, row in result.reset_index(drop=True).iterrows():
        print(f"Case {idx + 1}")
        print(f"  Fragmentation probability: {row['fragmentation_probability']:.2%}")
        print(f"  Predicted largest fragment mass: {row['predicted_largest_fragment_mass_kg']:.3e} kg")
        print(f"  Severity class: {row['severity_class']}")
        print(f"  Domain status: {row['domain_status']}")
        print(f"  Recommendation: {row['sph_recommendation']}")
        print(f"  Explanation: {row['explanation']}")
        if row["out_of_domain_features"]:
            print(f"  Out-of-domain features: {row['out_of_domain_features']}")
        elif row["near_edge_features"]:
            print(f"  Near-edge features: {row['near_edge_features']}")
        print()

    print("Disclaimer:")
    print("  This tool predicts FoF-derived fragmentation proxy outcomes.")
    print("  It does not replace SPH and does not directly validate long-term capture, disk mass, or moon formation.")


def main() -> None:
    args = parse_args()
    if args.print_template:
        print_template()
        return

    artifacts = load_artifacts(args.model_dir)
    if artifacts is None:
        raise SystemExit("Model artifacts are missing. Run `python scripts/train_triage_models.py` first.")

    classifier, regressor, training_domain = artifacts
    cases = load_cases(args.input)
    result = predict_cases(cases, classifier, regressor, training_domain)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    print_case_summary(result)
    print(f"Saved full prediction table to {args.output}")


if __name__ == "__main__":
    main()
