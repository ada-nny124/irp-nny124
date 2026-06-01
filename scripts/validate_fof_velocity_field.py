#!/usr/bin/env python3
"""Validate whether a FoF HDF5 file contains usable velocity data."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fof-file", required=True, help="Path to a FoF HDF5 file.")
    parser.add_argument("--table-out", required=True, help="CSV output path.")
    parser.add_argument("--plot-out", required=True, help="PNG output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    table_out = Path(args.table_out)
    plot_out = Path(args.plot_out)
    table_out.parent.mkdir(parents=True, exist_ok=True)
    plot_out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.fof_file, "r") as handle:
        velocities = handle["PartType0/Velocities"][()]

    speed = np.linalg.norm(velocities, axis=1)
    zero_mask = np.isclose(speed, 0.0)
    summary = pd.DataFrame(
        [
            {
                "fof_file": str(Path(args.fof_file).resolve()),
                "particle_rows": int(len(speed)),
                "zero_speed_particle_count": int(zero_mask.sum()),
                "nonzero_speed_particle_count": int((~zero_mask).sum()),
                "min_speed": float(speed.min()),
                "median_speed": float(np.median(speed)),
                "max_speed": float(speed.max()),
                "all_particle_speeds_zero": bool(np.all(zero_mask)),
                "fragment_com_velocity_implication": "If all particle velocities are zero, every fragment COM velocity is also zero.",
            }
        ]
    )
    summary.to_csv(table_out, index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Zero speed", "Non-zero speed"], [int(zero_mask.sum()), int((~zero_mask).sum())], color=["#4C78A8", "#E45756"])
    ax.set_title("FoF particle velocity validation")
    ax.set_ylabel("Particle count")
    ax.ticklabel_format(axis="y", style="plain")
    fig.tight_layout()
    fig.savefig(plot_out, dpi=150)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
