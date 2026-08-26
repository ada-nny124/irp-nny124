#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT_ENV = os.environ.get("CMF_OUTPUT_ROOT")
OUTPUT_BASE = Path(OUTPUT_ROOT_ENV).resolve() if OUTPUT_ROOT_ENV else ROOT
OUTPUT_TABLES = OUTPUT_BASE / "report-table-figure" / "tables_cmf"
TABLES_DIR = ROOT / "report-table-figure" / "tables"
CORRECTED_ARTIFACTS = OUTPUT_BASE / "ml" / "trainingartifacts_cmf"


def run_cli(script_path: Path, *args: str) -> None:
    subprocess.run([sys.executable, str(script_path), *args], check=True, cwd=ROOT)


def main() -> None:
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_TABLES / "table2_used_in_report_cmf.csv"
    run_cli(
        TABLES_DIR / "table2_build.py",
        "--oof",
        str(CORRECTED_ARTIFACTS / "tuned_rf" / "main_cmf_tuned_rf_oof_predictions.csv"),
        "--output",
        str(output_path),
    )
    frame = pd.read_csv(output_path)
    frame = frame.rename(columns={"Actual BMF": "Actual mass fraction"})
    frame.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
