#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmf_runtime import aliased_bound_dataset

OUTPUT_ROOT_ENV = os.environ.get("CMF_OUTPUT_ROOT")
OUTPUT_BASE = Path(OUTPUT_ROOT_ENV).resolve() if OUTPUT_ROOT_ENV else ROOT
OUTPUT_TABLES = OUTPUT_BASE / "report-table-figure" / "tables_cmf"
TABLES_DIR = ROOT / "report-table-figure" / "tables"
CORRECTED_BOUND = ROOT / "extraction-outputs_corrected_bmf" / "tables" / "bound_outcomes.csv"
CORRECTED_ARTIFACTS = OUTPUT_BASE / "ml" / "trainingartifacts_cmf"


def run_cli(script_path: Path, *args: str) -> None:
    subprocess.run([sys.executable, str(script_path), *args], check=True, cwd=ROOT)


def main() -> None:
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    with aliased_bound_dataset(CORRECTED_BOUND, "captured_mass_fraction") as aliased_bound:
        run_cli(
            TABLES_DIR / "section34_build.py",
            "--dataset",
            str(aliased_bound),
            "--bundle",
            str(CORRECTED_ARTIFACTS / "tuned_gradient_boosting" / "main_cmf_tuned_gradient_boosting.pkl"),
            "--output",
            str(OUTPUT_TABLES / "section34_used_in_report_cmf.csv"),
        )


if __name__ == "__main__":
    main()
