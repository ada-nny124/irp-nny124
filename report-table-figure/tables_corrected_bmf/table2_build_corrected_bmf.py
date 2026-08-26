#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_TABLES = ROOT / "report-table-figure" / "tables_corrected_bmf"
TABLES_DIR = ROOT / "report-table-figure" / "tables"
CORRECTED_ARTIFACTS = ROOT / "ml" / "trainingartifacts_corrected_bmf"


def run_cli(script_path: Path, *args: str) -> None:
    subprocess.run([sys.executable, str(script_path), *args], check=True, cwd=ROOT)


def main() -> None:
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    run_cli(
        TABLES_DIR / "table2_build.py",
        "--oof",
        str(CORRECTED_ARTIFACTS / "tuned_rf" / "main_bmf_tuned_rf_oof_predictions.csv"),
        "--output",
        str(OUTPUT_TABLES / "table2_used_in_report_corrected_bmf.csv"),
    )


if __name__ == "__main__":
    main()
