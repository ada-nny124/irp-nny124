from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "scripts" / "eda" / "plot_figure1_mass20_temp.py"
OUTPUT_PATH = ROOT / "eda" / "bound_eda" / "plots" / "figure1_mass19_only_temp.png"


def main() -> None:
    subprocess.run(
        [
            sys.executable,
            str(BASE_SCRIPT),
            "--mass-code",
            "A1900",
            "--output",
            str(OUTPUT_PATH),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
