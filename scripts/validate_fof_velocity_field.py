#!/usr/bin/env python3
"""Validate whether FoF HDF5 files contain usable particle velocity data."""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRIORITY_VELOCITY_CODES = [
    "v00",
    "v02",
    "v04",
    "v06",
    "v08",
    "v10",
    "v12",
    "v14",
    "v15",
    "v16",
    "v20",
    "v25",
    "v30",
]
DEFAULT_CHUNK_ROWS = 100_000


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


def balanced_manifest_paths(table: pd.DataFrame, path_column: str, max_files: Optional[int]) -> List[Path]:
    paths = table[path_column].dropna().astype(str)
    if max_files is None or max_files >= len(paths):
        return [Path(value) for value in paths.tolist()]

    if "velocity_code" not in table.columns:
        return [Path(value) for value in paths.iloc[:max_files].tolist()]

    grouped: Dict[str, List[str]] = {}
    for _, row in table.loc[paths.index].iterrows():
        velocity_code = str(row.get("velocity_code", ""))
        grouped.setdefault(velocity_code, []).append(str(row[path_column]))

    ordered_codes = [code for code in PRIORITY_VELOCITY_CODES if code in grouped]
    ordered_codes.extend(sorted(code for code in grouped if code not in PRIORITY_VELOCITY_CODES))

    selected: List[str] = []
    positions = {code: 0 for code in ordered_codes}

    for code in ordered_codes:
        if len(selected) >= max_files:
            break
        entries = grouped.get(code, [])
        if entries:
            selected.append(entries[0])
            positions[code] = 1

    while len(selected) < max_files:
        added_any = False
        for code in ordered_codes:
            pos = positions[code]
            entries = grouped[code]
            if pos < len(entries):
                selected.append(entries[pos])
                positions[code] = pos + 1
                added_any = True
                if len(selected) >= max_files:
                    break
        if not added_any:
            break

    return [Path(value) for value in selected]


def resolve_input_files(args: argparse.Namespace) -> List[Path]:
    if args.fof_file:
        return [Path(args.fof_file)]

    table = pd.read_csv(args.file_table)
    if args.path_column not in table.columns:
        raise KeyError(f"Column '{args.path_column}' was not found in {args.file_table}.")

    return balanced_manifest_paths(table, args.path_column, args.max_files)


def sample_particle_indices(count: int, sample_size: int, rng: np.random.Generator) -> np.ndarray:
    if sample_size <= 0 or sample_size >= count:
        return np.arange(count, dtype=int)
    return np.sort(rng.choice(count, size=sample_size, replace=False))


def empty_summary_row(fof_file: Path) -> Dict[str, object]:
    return {
        "fof_file": str(fof_file),
        "exists": fof_file.exists(),
        "readable": False,
        "n_particles": 0,
        "velocity_dataset_shape": "",
        "velocity_dataset_dtype": "",
        "min_vx": math.nan,
        "max_vx": math.nan,
        "min_vy": math.nan,
        "max_vy": math.nan,
        "min_vz": math.nan,
        "max_vz": math.nan,
        "max_abs_velocity": math.nan,
        "min_speed": math.nan,
        "max_speed": math.nan,
        "mean_speed": math.nan,
        "nonzero_velocity_count": 0,
        "all_particle_speeds_zero": False,
    }


def validate_one_file(
    fof_file: Path,
    sample_particles_per_file: int,
    rng: np.random.Generator,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    if not fof_file.exists():
        return empty_summary_row(fof_file), []

    try:
        with h5py.File(fof_file, "r") as handle:
            velocities_ds = handle["PartType0/Velocities"]
            coordinates_ds = handle["PartType0/Coordinates"]

            n_particles = int(velocities_ds.shape[0])
            summary_row = {
                "fof_file": str(fof_file),
                "exists": True,
                "readable": True,
                "n_particles": n_particles,
                "velocity_dataset_shape": str(tuple(int(dim) for dim in velocities_ds.shape)),
                "velocity_dataset_dtype": str(velocities_ds.dtype),
                "min_vx": math.inf,
                "max_vx": -math.inf,
                "min_vy": math.inf,
                "max_vy": -math.inf,
                "min_vz": math.inf,
                "max_vz": -math.inf,
                "max_abs_velocity": 0.0,
                "min_speed": math.inf,
                "max_speed": 0.0,
                "mean_speed": 0.0,
                "nonzero_velocity_count": 0,
                "all_particle_speeds_zero": True,
            }

            sample_indices = sample_particle_indices(n_particles, sample_particles_per_file, rng)
            sample_rows: List[Dict[str, object]] = []
            next_sample_idx = 0
            total_speed = 0.0

            for start in range(0, n_particles, DEFAULT_CHUNK_ROWS):
                stop = min(start + DEFAULT_CHUNK_ROWS, n_particles)
                velocities = velocities_ds[start:stop]
                speeds = np.linalg.norm(velocities, axis=1)
                component_mins = velocities.min(axis=0)
                component_maxs = velocities.max(axis=0)
                nonzero_mask = np.any(velocities != 0, axis=1)

                summary_row["min_vx"] = min(float(summary_row["min_vx"]), float(component_mins[0]))
                summary_row["max_vx"] = max(float(summary_row["max_vx"]), float(component_maxs[0]))
                summary_row["min_vy"] = min(float(summary_row["min_vy"]), float(component_mins[1]))
                summary_row["max_vy"] = max(float(summary_row["max_vy"]), float(component_maxs[1]))
                summary_row["min_vz"] = min(float(summary_row["min_vz"]), float(component_mins[2]))
                summary_row["max_vz"] = max(float(summary_row["max_vz"]), float(component_maxs[2]))
                summary_row["max_abs_velocity"] = max(
                    float(summary_row["max_abs_velocity"]),
                    float(np.max(np.abs(velocities))),
                )
                summary_row["min_speed"] = min(float(summary_row["min_speed"]), float(speeds.min()))
                summary_row["max_speed"] = max(float(summary_row["max_speed"]), float(speeds.max()))
                summary_row["nonzero_velocity_count"] += int(nonzero_mask.sum())
                total_speed += float(speeds.sum())

                if np.any(nonzero_mask):
                    summary_row["all_particle_speeds_zero"] = False

                while next_sample_idx < len(sample_indices) and int(sample_indices[next_sample_idx]) < stop:
                    particle_index = int(sample_indices[next_sample_idx])
                    local_index = particle_index - start
                    x, y, z = coordinates_ds[particle_index]
                    vx, vy, vz = velocities[local_index]
                    speed = float(speeds[local_index])
                    sample_rows.append(
                        {
                            "fof_file": str(fof_file),
                            "particle_index": particle_index,
                            "x": float(x),
                            "y": float(y),
                            "z": float(z),
                            "vx": float(vx),
                            "vy": float(vy),
                            "vz": float(vz),
                            "speed": speed,
                            "is_zero_speed": bool(speed == 0.0),
                        }
                    )
                    next_sample_idx += 1

            if n_particles == 0:
                summary_row["min_vx"] = math.nan
                summary_row["max_vx"] = math.nan
                summary_row["min_vy"] = math.nan
                summary_row["max_vy"] = math.nan
                summary_row["min_vz"] = math.nan
                summary_row["max_vz"] = math.nan
                summary_row["max_abs_velocity"] = math.nan
                summary_row["min_speed"] = math.nan
                summary_row["max_speed"] = math.nan
                summary_row["mean_speed"] = math.nan
            else:
                summary_row["mean_speed"] = total_speed / n_particles

            return summary_row, sample_rows
    except (OSError, KeyError, ValueError):
        return empty_summary_row(fof_file), []


def write_plot(summary: pd.DataFrame, plot_out: Path) -> None:
    zero_count = int(summary["all_particle_speeds_zero"].fillna(False).sum())
    nonzero_count = int((~summary["all_particle_speeds_zero"].fillna(False)).sum())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["All speeds zero", "Any non-zero speed"], [zero_count, nonzero_count], color=["#4C78A8", "#E45756"])
    ax.set_title("FoF velocity validation by file")
    ax.set_ylabel("File count")
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

    summary_rows: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []
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
