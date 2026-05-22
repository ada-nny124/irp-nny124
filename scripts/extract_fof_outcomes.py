#!/usr/bin/env python3
"""
Extract filename parameters, HDF5 schema, and simple FoF fragment outcomes
from Martian tidal-disruption HDF5 snapshots.

This script is designed to run on HPC, close to the data.

It does NOT claim to compute bound mass/orbital capture unless those quantities
are explicitly available or later implemented with validated physics.
Current physical outputs are conservative FoF group statistics:
- number of FoF groups/fragments
- particle count per group
- largest group by particle count
- largest group by mass, if particle masses exist
- total grouped mass, if masses exist
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


FILENAME_RE = re.compile(
    r"^(?P<model_prefix>Ma_xp)"
    r"_A(?P<A_code>\d{4})"
    r"(?P<special_case>c\d+)?"
    r"(?P<spin>_s\d+(?:mz|x|y|z))?"
    r"_n(?P<n_code>\d+)"
    r"_r(?P<r_code>\d+)"
    r"_v(?P<v_code>\d+)"
    r"_(?P<timestep>\d+)"
    r"_fof_(?P<fof_linking>[0-9.]+)"
    r"_(?P<file_index>\d+)"
    r"\.hdf5$"
)


GROUP_ID_PRIORITY = [
    "PartType0/FOFGroupIDs",
    "PartType0/FOFGroupID",
    "PartType0/GroupIDs",
    "PartType0/GroupID",
    "PartType0/GroupNumber",
    "PartType0/GroupNr_all",
    "PartType0/FOFGroupNr",
    "PartType0/FOFGroupNumber",
    "PartType1/FOFGroupIDs",
    "PartType1/FOFGroupID",
    "PartType1/GroupIDs",
    "PartType1/GroupID",
    "PartType1/GroupNumber",
    "PartType1/GroupNr_all",
]

MASS_PRIORITY = [
    "PartType0/Masses",
    "PartType0/Mass",
    "PartType1/Masses",
    "PartType1/Mass",
]


@dataclass
class DatasetInfo:
    path: str
    shape: tuple[int, ...]
    dtype: str


def parse_filename(filename: str) -> dict[str, Any]:
    match = FILENAME_RE.match(filename)
    if not match:
        return {
            "filename": filename,
            "parsed_ok": False,
            "parse_error": "filename did not match expected pattern",
        }

    d = match.groupdict()
    a_code = int(d["A_code"])
    n_code = int(d["n_code"])
    r_code = int(d["r_code"])
    v_code = int(d["v_code"])
    timestep = int(d["timestep"])
    fof_linking = float(d["fof_linking"])

    spin_raw = None
    spin_period_hr = np.nan
    spin_direction = "none"
    has_spin = False

    if d.get("spin"):
        has_spin = True
        spin_raw = d["spin"].lstrip("_")
        spin_body = spin_raw[1:]

        match_spin = re.match(r"(?P<period>\d+)(?P<direction>mz|x|y|z)$", spin_body)
        if match_spin:
            spin_period_hr = int(match_spin.group("period")) / 10.0
            direction = match_spin.group("direction")
            spin_direction = "-z" if direction == "mz" else direction

    special_case = d.get("special_case") or ""

    return {
        "filename": filename,
        "parsed_ok": True,
        "parse_error": "",
        "model_prefix": d["model_prefix"],
        "A_code": a_code,
        "mass_log10_kg": a_code / 100.0,
        "special_case": special_case,
        "has_spin": has_spin,
        "spin_raw": spin_raw or "",
        "spin_period_hr": spin_period_hr,
        "spin_direction": spin_direction,
        "n_code": n_code,
        "particle_log10": n_code / 10.0,
        "r_code": r_code,
        "periapsis_Rm": r_code / 10.0,
        "v_code": v_code,
        "v_inf_kms": v_code / 10.0,
        "timestep": timestep,
        "fof_linking": fof_linking,
        "file_index": d["file_index"],
    }


def list_hdf5_files(data_dir: Path, limit: int | None = None) -> list[Path]:
    files = sorted(data_dir.glob("*.hdf5"))
    if limit is not None:
        files = files[:limit]
    return files


def collect_datasets(h5: h5py.File) -> list[DatasetInfo]:
    datasets: list[DatasetInfo] = []

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            shape = tuple(int(x) for x in obj.shape)
            datasets.append(DatasetInfo(path=name, shape=shape, dtype=str(obj.dtype)))

    h5.visititems(visitor)
    return datasets


def schema_rows(filename: str, h5: h5py.File) -> list[dict[str, Any]]:
    rows = []
    for ds in collect_datasets(h5):
        rows.append(
            {
                "filename": filename,
                "dataset_path": ds.path,
                "shape": json.dumps(ds.shape),
                "dtype": ds.dtype,
            }
        )
    return rows


def dataset_exists(h5: h5py.File, path: str) -> bool:
    return path in h5 and isinstance(h5[path], h5py.Dataset)


def find_group_id_dataset(h5: h5py.File) -> str | None:
    for path in GROUP_ID_PRIORITY:
        if dataset_exists(h5, path):
            return path

    candidates = []
    for ds in collect_datasets(h5):
        lower = ds.path.lower()
        if len(ds.shape) == 1 and ("fof" in lower or "group" in lower):
            try:
                dtype = h5[ds.path].dtype
                if np.issubdtype(dtype, np.integer):
                    candidates.append(ds.path)
            except Exception:
                continue

    candidates = sorted(
        candidates,
        key=lambda path: (
            0 if path.startswith("PartType0/") else 1 if path.startswith("PartType1/") else 2,
            len(path),
        ),
    )
    return candidates[0] if candidates else None


def find_mass_dataset(h5: h5py.File, expected_len: int) -> str | None:
    for path in MASS_PRIORITY:
        if dataset_exists(h5, path) and h5[path].shape == (expected_len,):
            return path

    candidates = []
    for ds in collect_datasets(h5):
        lower = ds.path.lower()
        if len(ds.shape) == 1 and ds.shape[0] == expected_len and "mass" in lower:
            try:
                dtype = h5[ds.path].dtype
                if np.issubdtype(dtype, np.number):
                    candidates.append(ds.path)
            except Exception:
                continue

    candidates = sorted(
        candidates,
        key=lambda path: (
            0 if path.startswith("PartType0/") else 1 if path.startswith("PartType1/") else 2,
            len(path),
        ),
    )
    return candidates[0] if candidates else None


def summarise_fof_groups(
    h5_path: Path,
    min_particles: int,
    exclude_group_ids: set[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    filename = h5_path.name

    with h5py.File(h5_path, "r") as h5:
        group_path = find_group_id_dataset(h5)
        if group_path is None:
            return (
                {
                    "filename": filename,
                    "extraction_ok": False,
                    "extraction_error": "No FoF/group ID dataset found",
                    "group_id_dataset": "",
                    "mass_dataset": "",
                    "mass_source": "",
                },
                [],
            )

        group_ids = np.asarray(h5[group_path][...]).reshape(-1)
        n_particles = int(group_ids.size)

        mass_path = find_mass_dataset(h5, expected_len=n_particles)

        masses = None
        mass_source = "particle_count_proxy"
        if mass_path is not None:
            masses = np.asarray(h5[mass_path][...], dtype=np.float64).reshape(-1)
            mass_source = "particle_masses"

        valid_mask = np.ones(n_particles, dtype=bool)
        if exclude_group_ids:
            valid_mask &= ~np.isin(group_ids, list(exclude_group_ids))

        valid_group_ids = group_ids[valid_mask]

        if valid_group_ids.size == 0:
            return (
                {
                    "filename": filename,
                    "extraction_ok": True,
                    "extraction_error": "",
                    "group_id_dataset": group_path,
                    "mass_dataset": mass_path or "",
                    "mass_source": mass_source,
                    "n_particles": n_particles,
                    "n_grouped_particles": 0,
                    "n_fof_groups": 0,
                    "fragment_count_min_particles": 0,
                    "largest_fragment_particle_count": 0,
                    "largest_fragment_mass": np.nan,
                    "total_fragment_mass": np.nan,
                    "total_particle_mass": float(np.nansum(masses)) if masses is not None else np.nan,
                },
                [],
            )

        unique_ids, inverse, counts = np.unique(valid_group_ids, return_inverse=True, return_counts=True)

        if masses is not None:
            valid_masses = masses[valid_mask]
            group_masses = np.bincount(inverse, weights=valid_masses)
            total_particle_mass = float(np.nansum(masses))
        else:
            group_masses = counts.astype(float)
            total_particle_mass = np.nan

        fragment_mask = counts >= min_particles
        fragment_count = int(np.sum(fragment_mask))

        if fragment_count > 0:
            largest_by_particles_idx = int(np.argmax(np.where(fragment_mask, counts, -1)))
            largest_particle_count = int(counts[largest_by_particles_idx])
            largest_mass = float(group_masses[largest_by_particles_idx])
            total_fragment_mass = float(np.sum(group_masses[fragment_mask]))
        else:
            largest_particle_count = 0
            largest_mass = np.nan
            total_fragment_mass = np.nan

        fragment_rows = []
        for group_id, particle_count, group_mass in zip(unique_ids, counts, group_masses):
            if particle_count < min_particles:
                continue

            fragment_rows.append(
                {
                    "filename": filename,
                    "group_id": int(group_id),
                    "particle_count": int(particle_count),
                    "group_mass_or_proxy": float(group_mass),
                    "mass_source": mass_source,
                    "passes_min_particles": True,
                }
            )

        outcome = {
            "filename": filename,
            "extraction_ok": True,
            "extraction_error": "",
            "group_id_dataset": group_path,
            "mass_dataset": mass_path or "",
            "mass_source": mass_source,
            "n_particles": n_particles,
            "n_grouped_particles": int(valid_group_ids.size),
            "n_fof_groups": int(unique_ids.size),
            "fragment_count_min_particles": fragment_count,
            "largest_fragment_particle_count": largest_particle_count,
            "largest_fragment_mass": largest_mass,
            "total_fragment_mass": total_fragment_mass,
            "total_particle_mass": total_particle_mass,
            "fragment_mass_fraction": (
                total_fragment_mass / total_particle_mass
                if np.isfinite(total_fragment_mass)
                and np.isfinite(total_particle_mass)
                and total_particle_mass > 0
                else np.nan
            ),
        }
        return outcome, fragment_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def resolve_data_dir(path: Path) -> Path:
    raw = os.path.expandvars(str(path))
    return Path(raw).expanduser()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("$EPHEMERAL/martian_moons_data"),
        help="Directory containing .hdf5 files. Pass expanded HPC path or quote env var in shell.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files for testing.",
    )
    parser.add_argument(
        "--schema-samples",
        type=int,
        default=5,
        help="Number of files to include in schema summary.",
    )
    parser.add_argument(
        "--min-particles",
        type=int,
        default=20,
        help="Minimum particles required for a FoF group to count as a fragment.",
    )
    parser.add_argument(
        "--exclude-group-id",
        action="append",
        type=int,
        default=[-1],
        help="Group IDs to exclude. Default excludes -1 only. Repeat flag for more.",
    )

    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    if not data_dir.exists():
        print(
            f"ERROR: data directory does not exist: {data_dir}\n"
            f"Run with: --data-dir \"$EPHEMERAL/martian_moons_data\"",
            file=sys.stderr,
        )
        return 2

    outputs_dir: Path = args.outputs_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)

    files = list_hdf5_files(data_dir, args.limit)
    if not files:
        print(f"ERROR: no .hdf5 files found in {data_dir}", file=sys.stderr)
        return 2

    manifest_rows: list[dict[str, Any]] = []
    schema_summary_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    fragment_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    exclude_group_ids = set(args.exclude_group_id or [])

    print(f"Found {len(files)} HDF5 files")
    print(f"Writing outputs to {outputs_dir}")

    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {path.name}", flush=True)

        manifest = parse_filename(path.name)
        manifest_rows.append(manifest)

        try:
            if index <= args.schema_samples:
                with h5py.File(path, "r") as h5:
                    schema_summary_rows.extend(schema_rows(path.name, h5))

            outcome, frags = summarise_fof_groups(
                h5_path=path,
                min_particles=args.min_particles,
                exclude_group_ids=exclude_group_ids,
            )

            outcome_rows.append({**manifest, **outcome})
            fragment_rows.extend(frags)

            if not outcome.get("extraction_ok", False):
                error_rows.append(
                    {
                        "filename": path.name,
                        "error": outcome.get("extraction_error", "unknown extraction error"),
                    }
                )

        except Exception as exc:
            error_rows.append({"filename": path.name, "error": repr(exc)})
            outcome_rows.append(
                {
                    **manifest,
                    "filename": path.name,
                    "extraction_ok": False,
                    "extraction_error": repr(exc),
                }
            )

    write_csv(outputs_dir / "manifest.csv", manifest_rows)
    write_csv(outputs_dir / "fof_outcomes.csv", outcome_rows)
    write_csv(outputs_dir / "fragment_catalog.csv", fragment_rows)
    write_csv(outputs_dir / "hdf5_schema_summary.csv", schema_summary_rows)

    if error_rows:
        error_path = outputs_dir / "extraction_errors.csv"
        write_csv(error_path, error_rows)
        print(f"Completed with {len(error_rows)} errors. See {error_path}")
    else:
        print("Completed without extraction errors.")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
