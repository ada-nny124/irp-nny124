#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_TEST_ROOT = ROOT / "_testing_reproducibility"

STEPS = [
    ROOT / "ml" / "model_training_scripts" / "train_all_models.py",
    ROOT / "report-table-figure" / "figures" / "figure1_build.py",
    ROOT / "report-table-figure" / "figures" / "figure2_build.py",
    ROOT / "report-table-figure" / "figures" / "figure3_build.py",
    ROOT / "report-table-figure" / "figures" / "figure4_build.py",
    ROOT / "report-table-figure" / "tables" / "tuned_gb_oof_predictions_build.py",
    ROOT / "report-table-figure" / "figures" / "figure5_build.py",
    ROOT / "report-table-figure" / "figures" / "figureA1_build.py",
    ROOT / "report-table-figure" / "figures" / "tableA2_details_build.py",
    ROOT / "report-table-figure" / "tables" / "section34_build.py",
    ROOT / "report-table-figure" / "tables" / "table2_build.py",
    ROOT / "report-table-figure" / "tables" / "tableA2_build.py",
]

REQUIRED_OUTPUTS = [
    "ml/trainingartifacts/raw_gb/raw_gb.pkl",
    "ml/trainingartifacts/tuned_gb/tuned_gb.pkl",
    "ml/trainingartifacts/raw_rf/raw_rf.pkl",
    "ml/trainingartifacts/tuned_rf/tuned_rf.pkl",
    "ml/trainingartifacts/derived_rf/derived_rf.pkl",
    "ml/trainingartifacts/derived_gb/derived_gb.pkl",
    "report-table-figure/figures/figure1_used_in_report.png",
    "report-table-figure/figures/figure2_used_in_report.png",
    "report-table-figure/figures/figure3_used_in_report.png",
    "report-table-figure/figures/figure4_used_in_report.png",
    "report-table-figure/figures/figure5_used_in_report.png",
    "report-table-figure/figures/figureA1_used_in_report.png",
    "report-table-figure/figures/tableA2_details.png",
    "report-table-figure/tables/tuned_gb_oof_predictions.csv",
    "report-table-figure/tables/section34_used_in_report.csv",
    "report-table-figure/tables/table2_used_in_report.csv",
    "report-table-figure/tables/tableA2_used_in_report.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BMF training and report pipeline.")
    parser.add_argument(
        "--testing-reproducibility",
        "--_testing_reproducibility",
        dest="testing_reproducibility",
        action="store_true",
        help="Run into an isolated _testing_reproducibility output tree instead of the live output folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional alternate output root. Expected layout will be created inside this folder.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not delete a pre-existing test output root before rerunning.",
    )
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


def verify_outputs(output_root: Path | None) -> list[Path]:
    base = output_root if output_root is not None else ROOT
    missing = [base / rel for rel in REQUIRED_OUTPUTS if not (base / rel).exists()]
    return missing


def write_manifest(output_root: Path) -> Path:
    manifest_path = output_root / "reproducibility_manifest.txt"
    lines = [
        "BMF reproducibility test output",
        "",
        f"Output root: {output_root}",
        "",
        "Required outputs verified:",
    ]
    lines.extend(f"- {rel}" for rel in REQUIRED_OUTPUTS)
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    args = parse_args()
    output_root = resolve_output_root(args)
    prepare_output_root(output_root, args.keep_existing)

    env = os.environ.copy()
    if output_root is not None:
        env["PIPELINE_OUTPUT_ROOT"] = str(output_root)

    for script_path in STEPS:
        cmd = [sys.executable, str(script_path)]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=ROOT, env=env)

    missing = verify_outputs(output_root)
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Reproducibility run completed but some expected outputs are missing:\n{missing_text}")

    if output_root is not None:
        manifest_path = write_manifest(output_root)
        print(f"Verified reproducibility outputs under {output_root}")
        print(manifest_path)


if __name__ == "__main__":
    main()
