#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent

TRAINING_SCRIPTS = [
    "train_gradient_boosting_corrected_bmf.py",
    "train_tuned_gradient_boosting_corrected_bmf.py",
    "train_raw_rf_corrected_bmf.py",
    "train_tuned_rf_corrected_bmf.py",
    "train_main_physics_rf_corrected_bmf.py",
    "train_tuned_physics_gradient_boosting_corrected_bmf.py",
    "train_tuned_physics_rf_corrected_bmf.py",
]


def main() -> None:
    for script_name in TRAINING_SCRIPTS:
        script_path = SCRIPT_DIR / script_name
        cmd = [sys.executable, str(script_path)]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
