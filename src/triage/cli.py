#!/usr/bin/env python3
"""Run the SPH BMF triage demo locally from an editable template file."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from triage.dashboard import eccentricity_from_periapsis_and_vinf, load_demo_metadata, predict_single_payload
else:
    from .dashboard import eccentricity_from_periapsis_and_vinf, load_demo_metadata, predict_single_payload

ROOT = Path(__file__).resolve().parents[2]


DEFAULT_TEMPLATE = Path(__file__).resolve().parent / "templates" / "triage_case_template.json"
DEFAULT_MODEL_DIR = ROOT / "ml" / "triage"
DEFAULT_OUTPUT = ROOT / "outputs" / "triage_demo_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_TEMPLATE, help="Path to a JSON or CSV file containing one or more triage cases.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Deprecated; BMF artifacts are resolved by the dashboard prediction module.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where to save the prediction table as CSV.")
    parser.add_argument("--print-template", action="store_true", help="Print the expected template structure and exit.")
    parser.add_argument("--case-name", help="Run one case directly from CLI flags instead of --input.")
    parser.add_argument("--mass-log10", type=float, help="log10 parent mass in kg.")
    parser.add_argument("--periapsis", type=float, help="Closest approach in Mars radii.")
    parser.add_argument("--eccentricity", type=float, help="Encounter eccentricity. If omitted, --v-inf is converted to eccentricity.")
    parser.add_argument("--v-inf", type=float, help="Encounter velocity at infinity in km/s. Used only when --eccentricity is omitted.")
    parser.add_argument("--spin-axis", default="z", help="Spin axis for explicit spin cases. Default: z.")
    parser.add_argument("--spin-period", type=float, help="Spin period in hours.")
    parser.add_argument("--no-spin", action="store_true", help="Use a no-spin case.")
    parser.add_argument("--density", type=float, default=2700.0, help="Asteroid density in kg/m^3. Default: 2700.")
    parser.add_argument("--resolution", type=float, default=65.0, help="Resolution value. Default: 65.")
    parser.add_argument("--timestep", type=float, default=90000.0, help="Simulation timestep. Default: 90000.")
    parser.add_argument("--fof-linking-length", type=float, default=0.004, help="FoF linking length. Default: 0.004.")
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
    defaults = load_demo_metadata()["defaults"]
    template = {
        "case_name": "demo_case",
        "mass_log10_kg": defaults["mass_log10_kg"],
        "periapsis_Rm": defaults["periapsis_Rm"],
        "encounter_eccentricity": defaults["encounter_eccentricity"],
        "has_explicit_spin": defaults["has_explicit_spin"],
        "spin_axis": defaults["spin_axis"],
        "spin_period_hr": defaults["spin_period_hr"],
        "asteroid_density_kg_m3": defaults["asteroid_density_kg_m3"],
        "asteroid_type": defaults["asteroid_type"],
        "resolution_value": defaults["resolution_value"],
        "timestep": defaults["timestep"],
        "fof_linking_length": defaults["fof_linking_length"],
    }
    print(json.dumps(template, indent=2))


def normalize_case_for_demo(row: dict[str, object]) -> dict[str, object]:
    payload = {key: value for key, value in row.items() if value not in ("", None)}
    if "encounter_eccentricity" not in payload and "v_inf_kms" in payload:
        payload["encounter_eccentricity"] = eccentricity_from_periapsis_and_vinf(
            float(payload["periapsis_Rm"]),
            float(payload["v_inf_kms"]),
        )
    return payload


def has_direct_case_args(args: argparse.Namespace) -> bool:
    return any(
        value is not None
        for value in (
            args.case_name,
            args.mass_log10,
            args.periapsis,
            args.eccentricity,
            args.v_inf,
            args.spin_period,
        )
    ) or args.no_spin


def load_direct_case(args: argparse.Namespace) -> pd.DataFrame:
    missing = [
        name
        for name, value in {
            "--mass-log10": args.mass_log10,
            "--periapsis": args.periapsis,
        }.items()
        if value is None
    ]
    if args.eccentricity is None and args.v_inf is None:
        missing.append("--eccentricity or --v-inf")
    if not args.no_spin and args.spin_period is None:
        missing.append("--spin-period or --no-spin")
    if missing:
        raise SystemExit(f"Missing direct-case argument(s): {', '.join(missing)}")

    eccentricity = args.eccentricity
    if eccentricity is None:
        eccentricity = eccentricity_from_periapsis_and_vinf(float(args.periapsis), float(args.v_inf))

    payload = {
        "case_name": args.case_name or "cli_demo_case",
        "mass_log10_kg": args.mass_log10,
        "periapsis_Rm": args.periapsis,
        "encounter_eccentricity": eccentricity,
        "has_explicit_spin": not args.no_spin,
        "spin_axis": "none" if args.no_spin else args.spin_axis,
        "spin_period_hr": None if args.no_spin else args.spin_period,
        "asteroid_density_kg_m3": args.density,
        "asteroid_type": "rocky",
        "resolution_value": args.resolution,
        "timestep": args.timestep,
        "fof_linking_length": args.fof_linking_length,
    }
    return pd.DataFrame([payload])


def prediction_rows(cases: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, case in cases.iterrows():
        result = predict_single_payload(normalize_case_for_demo(case.to_dict()))
        rows.append(result["export_row"])
    return pd.DataFrame(rows)


def print_case_summary(result: pd.DataFrame) -> None:
    for _, row in result.reset_index(drop=True).iterrows():
        print(f"  Predicted BMF: {float(row['predicted_bmf']) * 100.0:.1f}%")
        print(f"  Predicted bound mass: {float(row['predicted_bound_mass_kg']):.3e} kg")
        print(f"  Fragmentation label: {row['fragmentation_label']}")
        print(f"  Support: {row['support_category']} ({float(row['support_score']):.0f}/100)")
        print(f"  Recommendation: {row['recommendation']}")
        print(f"  Reason: {row['recommendation_reason']}")
        print()


def main() -> None:
    args = parse_args()
    if args.print_template:
        print_template()
        return

    cases = load_direct_case(args) if has_direct_case_args(args) else load_cases(args.input)
    result = prediction_rows(cases)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    print_case_summary(result)
    print(f"Saved full prediction table to {args.output}")


if __name__ == "__main__":
    main()
