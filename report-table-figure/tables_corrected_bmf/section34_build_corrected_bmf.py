#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT_ENV = os.environ.get("CORRECTED_BMF_OUTPUT_ROOT")
OUTPUT_BASE = Path(OUTPUT_ROOT_ENV).resolve() if OUTPUT_ROOT_ENV else ROOT
OUTPUT_TABLES = OUTPUT_BASE / "report-table-figure" / "tables_corrected_bmf"
TABLES_DIR = ROOT / "report-table-figure" / "tables"
CORRECTED_BOUND = ROOT / "extraction-outputs_corrected_bmf" / "tables" / "bound_outcomes.csv"
CORRECTED_ARTIFACTS = OUTPUT_BASE / "ml" / "trainingartifacts_corrected_bmf"


def run_cli(script_path: Path, *args: str) -> None:
    subprocess.run([sys.executable, str(script_path), *args], check=True, cwd=ROOT)


def main() -> None:
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    run_cli(
        TABLES_DIR / "section34_build.py",
        "--dataset",
        str(CORRECTED_BOUND),
        "--bundle",
        str(CORRECTED_ARTIFACTS / "tuned_gradient_boosting" / "main_bmf_tuned_gradient_boosting.pkl"),
        "--output",
        str(OUTPUT_TABLES / "section34_used_in_report_corrected_bmf.csv"),
    )


if __name__ == "__main__":
    main()
