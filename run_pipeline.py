#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

STEPS = [
    ROOT / "ml" / "model_training_scripts_corrected_bmf" / "run_corrected_bmf_pipeline.py",
    ROOT / "report-table-figure" / "figures_corrected_bmf" / "figure1_build_corrected_bmf.py",
    ROOT / "report-table-figure" / "figures_corrected_bmf" / "figure2_build_corrected_bmf.py",
    ROOT / "report-table-figure" / "figures_corrected_bmf" / "figure3_build_corrected_bmf.py",
    ROOT / "report-table-figure" / "figures_corrected_bmf" / "figure4_build_corrected_bmf.py",
    ROOT / "report-table-figure" / "figures_corrected_bmf" / "figure5_build_corrected_bmf.py",
    ROOT / "report-table-figure" / "figures_corrected_bmf" / "figureA1_build_corrected_bmf.py",
    ROOT / "report-table-figure" / "figures_corrected_bmf" / "tableA2_details_corrected_bmf.py",
    ROOT / "report-table-figure" / "tables_corrected_bmf" / "section34_build_corrected_bmf.py",
    ROOT / "report-table-figure" / "tables_corrected_bmf" / "table2_build_corrected_bmf.py",
    ROOT / "report-table-figure" / "tables_corrected_bmf" / "tableA2_build_corrected_bmf.py",
]


def main() -> None:
    for script_path in STEPS:
        cmd = [sys.executable, str(script_path)]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
