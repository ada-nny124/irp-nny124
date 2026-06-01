#!/usr/bin/env python3
"""Validate whether FoF HDF5 files contain usable particle velocity data."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--fof-file", help="Path to one FoF HDF5 file.")
    input_group.add_argument(
        "--file-table",
        help="CSV containing a column of FoF HDF5 paths, for example outputs/manifest.csv.",
    )
    parser.add_argument(
        "--path-column",
        default="file_path",
        help="Column name in --file-table containing HDF5 paths.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Only validate the first N files after reading the table.",
    )
    parser.add_argument(
        "--sample-particles-per-file",
        type=int,
        default=0,
        help="If > 0, write up to this many sampled particle rows per file to --samples-out.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Seed used when sampling particle rows.",
    )
    parser.add_argument("--table-out", required=True, help="CSV summary output path.")
    parser.add_argument(
        "--samples-out",
        default=None,
        help="Optional CSV containing sampled per-particle velocities and computed speeds.",
    )
    parser.add_argument("--plot-out", required=True, help="PNG output path.")
    return parser.parse_args()


def resolve_input_files(args: argparse.Namespace) -> list[Path]:
    if args.fof_file:
        return [Path(args.fof_file)]

    table = pd.read_csv(args.file_table)
    if args.path_column not in table.columns:
        raise KeyError(f"Column '{args.path_column}' was not found in {args.file_table}.")

    paths = [Path(str(value)) for value in table[args.path_column].dropna().tolist()]
    if args.max_files is not None:
        paths = paths[: args.max_files]
    return paths


def sample_particle_indices(count: int, sample_size: int, rng: np.random.Generator) -> np.ndarray:
    if sample_size <= 0 or sample_size >= count:
        return np.arange(count, dtype=int)
    return np.sort(rng.choice(count, size=sample_size, replace=False))


def validate_one_file(
    fof_file: Path,
    sample_particles_per_file: int,
    rng: np.random.Generator,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    with h5py.File(fof_file, "r") as handle:
        velocities = handle["PartType0/Velocities"][()]

    speed = np.linalg.norm(velocities, axis=1)
    zero_mask = np.isclose(speed, 0.0)

    summary_row = {
        "fof_file": str(fof_file),
        "file_exists_at_runtime": fof_file.exists(),
        "particle_rows": int(len(speed)),
        "zero_speed_particle_count": int(zero_mask.sum()),
        "nonzero_speed_particle_count": int((~zero_mask).sum()),
        "zero_speed_fraction": float(zero_mask.mean()),
        "min_speed": float(speed.min()),
        "median_speed": float(np.median(speed)),
        "max_speed": float(speed.max()),
        "all_particle_speeds_zero": bool(np.all(zero_mask)),
        "speed_formula": "speed = sqrt(vx^2 + vy^2 + vz^2)",
        "fragment_com_velocity_implication": "If all particle velocities are zero, every fragment COM velocity is also zero.",
    }

    sample_rows: list[dict[str, object]] = []
    sample_indices = sample_particle_indices(len(speed), sample_particles_per_file, rng)
    for particle_index in sample_indices.tolist():
        vx, vy, vz = velocities[particle_index]
        sample_rows.append(
            {
                "fof_file": str(fof_file),
                "particle_index": int(particle_index),
                "vx": float(vx),
                "vy": float(vy),
                "vz": float(vz),
                "speed": float(speed[particle_index]),
                "is_zero_speed": bool(zero_mask[particle_index]),
            }
        )

    return summary_row, sample_rows


def write_plot(summary: pd.DataFrame, plot_out: Path) -> None:
    zero_count = int(summary["zero_speed_particle_count"].sum())
    nonzero_count = int(summary["nonzero_speed_particle_count"].sum())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Zero speed", "Non-zero speed"], [zero_count, nonzero_count], color=["#4C78A8", "#E45756"])
    ax.set_title("FoF particle velocity validation")
    ax.set_ylabel("Particle count")
    ax.ticklabel_format(axis="y", style="plain")
    fig.tight_layout()
    fig.savefig(plot_out, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    table_out = Path(args.table_out)
    plot_out = Path(args.plot_out)
    samples_out = Path(args.samples_out) if args.samples_out else None
    table_out.parent.mkdir(parents=True, exist_ok=True)
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    if samples_out is not None:
        samples_out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.random_seed)
    fof_files = resolve_input_files(args)

    summary_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    for fof_file in fof_files:
        summary_row, one_file_samples = validate_one_file(
            fof_file=fof_file,
            sample_particles_per_file=args.sample_particles_per_file,
            rng=rng,
        )
        summary_rows.append(summary_row)
        sample_rows.extend(one_file_samples)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(table_out, index=False)

    if samples_out is not None:
        pd.DataFrame(sample_rows).to_csv(samples_out, index=False)

    write_plot(summary, plot_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
