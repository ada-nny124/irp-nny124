#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_TEST_ROOT = ROOT / "_testing_reproducibility_cmf"

STEPS = [
    ROOT / "ml" / "model_training_scripts_cmf" / "run_cmf_pipeline.py",
    ROOT / "report-table-figure" / "figures_cmf" / "figure1_build_cmf.py",
    ROOT / "report-table-figure" / "figures_cmf" / "figure2_build_cmf.py",
    ROOT / "report-table-figure" / "figures_cmf" / "figure3_build_cmf.py",
    ROOT / "report-table-figure" / "figures_cmf" / "figure4_build_cmf.py",
    ROOT / "report-table-figure" / "figures_cmf" / "figure5_build_cmf.py",
    ROOT / "report-table-figure" / "figures_cmf" / "figureA1_build_cmf.py",
    ROOT / "report-table-figure" / "figures_cmf" / "tableA2_details_cmf.py",
    ROOT / "report-table-figure" / "tables_cmf" / "section34_build_cmf.py",
    ROOT / "report-table-figure" / "tables_cmf" / "table2_build_cmf.py",
    ROOT / "report-table-figure" / "tables_cmf" / "tableA2_build_cmf.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CMF training and report pipeline.")
    parser.add_argument("--testing-reproducibility", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def resolve_output_root(args: argparse.Namespace) -> Path | None:
    if args.output_root is not None:
        return args.output_root.resolve()
    if args.testing_reproducibility:
        return DEFAULT_TEST_ROOT.resolve()
    return None


def prepare_output_root(output_root: Path | None, keep_existing: bool) -> None:
    if output_root is None:
        return
    if output_root.exists() and not keep_existing:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    output_root = resolve_output_root(args)
    prepare_output_root(output_root, args.keep_existing)

    env = os.environ.copy()
    if output_root is not None:
        env["CMF_OUTPUT_ROOT"] = str(output_root)

    for script_path in STEPS:
        cmd = [sys.executable, str(script_path)]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=ROOT, env=env)


if __name__ == "__main__":
    main()
