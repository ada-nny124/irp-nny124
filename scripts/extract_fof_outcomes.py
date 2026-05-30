#!/usr/bin/env python3
"""Extract conservative FoF-level outcomes from Martian-moons HDF5 snapshots."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

try:
    import h5py
except ImportError as exc:  # pragma: no cover - runtime dependency on HPC module stack
    raise SystemExit(
        "h5py is required. On Imperial HPC load a compatible module stack first, for example:\n"
        "  module load tools/prod h5py/3.12.1-foss-2024a"
    ) from exc


GRAVITATIONAL_CONSTANT = 6.67430e-11
DEFAULT_MARS_MASS_KG = 6.4171e23
DEFAULT_MARS_RADIUS_M = 3.3895e6

FILENAME_RE = re.compile(
    r"^(?P<prefix>Ma_xp)_(?P<mass>A\d{4}(?:c30)?)(?:_(?P<spin>s\d{3}[A-Za-z]*))?"
    r"_n(?P<resolution>\d+)_r(?P<periapsis>\d+)_v(?P<velocity>\d+)"
    r"_(?P<timestep>\d+)_fof_(?P<linking_length>[0-9.]+)_(?P<chunk>\d+)\.hdf5$"
)

MANIFEST_FIELDS = [
    "simulation_id",
    "filename",
    "file_path",
    "file_size_bytes",
    "mass_code",
    "mass_value",
    "special_case_code",
    "spin_code",
    "spin_value",
    "spin_axis",
    "has_explicit_spin",
    "resolution_code",
    "resolution_value",
    "periapsis_code",
    "periapsis_value",
    "velocity_code",
    "velocity_value",
    "timestep",
    "fof_linking_length",
    "chunk_index",
]

OUTCOME_FIELDS = MANIFEST_FIELDS + [
    "parttype_group",
    "particle_count_total",
    "n_fof_groups",
    "fragment_count_min_particles",
    "largest_fragment_particle_count",
    "mass_metrics_available",
    "mass_unit",
    "total_particle_mass_kg",
    "largest_fragment_mass_kg",
    "total_fragment_mass_kg",
    "fragment_mass_fraction",
    "particle_count_metrics_are_proxies",
    "excluded_group_ids",
    "min_particles",
]

FRAGMENT_FIELDS = MANIFEST_FIELDS + [
    "parttype_group",
    "group_id",
    "particle_count",
    "particle_fraction_of_snapshot",
    "passes_min_particles",
    "mass_metrics_available",
    "fragment_mass_kg",
    "fragment_mass_fraction_of_snapshot",
]

ERROR_FIELDS = ["file_path", "filename", "error_type", "error_message"]

SCHEMA_FIELDS = [
    "sample_filename",
    "sample_file_path",
    "node_path",
    "node_type",
    "shape",
    "dtype",
    "attribute_name",
    "attribute_value_preview",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Directory containing FoF HDF5 files.")
    parser.add_argument("--outputs-dir", default="outputs", help="Directory for generated CSV outputs.")
    parser.add_argument(
        "--physical-data-dir",
        default=None,
        help="Optional directory containing matching non-FoF physical snapshots for bound/unbound analysis.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N files after sorting.")
    parser.add_argument(
        "--schema-samples",
        type=int,
        default=0,
        help="Inspect and summarize the schema of the first N selected HDF5 files.",
    )
    parser.add_argument(
        "--min-particles",
        type=int,
        default=20,
        help="Minimum particle count for fragment-level summary metrics.",
    )
    parser.add_argument(
        "--exclude-group-id",
        type=int,
        action="append",
        default=[],
        help="FoF group ID to exclude. Pass multiple times to exclude multiple IDs.",
    )
    return parser.parse_args()


def parse_simulation_filename(filename: str) -> dict[str, object]:
    match = FILENAME_RE.match(filename)
    if not match:
        raise ValueError(f"Unrecognized FoF filename pattern: {filename}")

    mass_code = match.group("mass")
    spin_code = match.group("spin") or ""
    special_case_code = "c30" if mass_code.endswith("c30") else ""
    mass_digits = mass_code[1:5]
    spin_axis = spin_code[4:] if len(spin_code) > 4 else ""
    spin_value = spin_code[1:4] if spin_code else ""

    resolution_value = int(match.group("resolution"))
    periapsis_value = int(match.group("periapsis"))
    velocity_value = int(match.group("velocity"))
    timestep = int(match.group("timestep"))
    chunk_index = int(match.group("chunk"))
    linking_length = float(match.group("linking_length"))

    return {
        "simulation_id": filename.replace(".hdf5", ""),
        "filename": filename,
        "mass_code": mass_code,
        "mass_value": int(mass_digits),
        "special_case_code": special_case_code,
        "spin_code": spin_code,
        "spin_value": int(spin_value) if spin_value else "",
        "spin_axis": spin_axis,
        "has_explicit_spin": bool(spin_code),
        "resolution_code": f"n{resolution_value}",
        "resolution_value": resolution_value,
        "periapsis_code": f"r{periapsis_value}",
        "periapsis_value": periapsis_value,
        "velocity_code": f"v{velocity_value:02d}",
        "velocity_value": velocity_value,
        "timestep": timestep,
        "fof_linking_length": linking_length,
        "chunk_index": chunk_index,
    }


def select_hdf5_files(data_dir: Path, limit: int | None = None) -> list[Path]:
    files = sorted(path for path in data_dir.iterdir() if path.is_file() and path.suffix == ".hdf5")
    if limit is not None:
        files = files[:limit]
    return files


def build_manifest_rows(files: Iterable[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in files:
        row = parse_simulation_filename(path.name)
        row["file_path"] = str(path.resolve())
        row["file_size_bytes"] = path.stat().st_size
        rows.append(row)
    return rows


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def stringify_attr_value(value: object, max_len: int = 120) -> str:
    text = str(value)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def summarize_hdf5_schema(files: Iterable[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in files:
        with h5py.File(path, "r") as handle:
            def visitor(node_path: str, obj: h5py.Dataset | h5py.Group) -> None:
                node_type = "dataset" if isinstance(obj, h5py.Dataset) else "group"
                shape = tuple(obj.shape) if isinstance(obj, h5py.Dataset) else ""
                dtype = str(obj.dtype) if isinstance(obj, h5py.Dataset) else ""
                if obj.attrs:
                    for attr_name, attr_value in obj.attrs.items():
                        rows.append(
                            {
                                "sample_filename": path.name,
                                "sample_file_path": str(path.resolve()),
                                "node_path": node_path or "/",
                                "node_type": node_type,
                                "shape": shape,
                                "dtype": dtype,
                                "attribute_name": attr_name,
                                "attribute_value_preview": stringify_attr_value(attr_value),
                            }
                        )
                else:
                    rows.append(
                        {
                            "sample_filename": path.name,
                            "sample_file_path": str(path.resolve()),
                            "node_path": node_path or "/",
                            "node_type": node_type,
                            "shape": shape,
                            "dtype": dtype,
                            "attribute_name": "",
                            "attribute_value_preview": "",
                        }
                    )

            handle.visititems(visitor)
    return rows


def get_parttype_group(handle: h5py.File) -> h5py.Group:
    if "PartType0" in handle:
        return handle["PartType0"]
    if "GasParticles" in handle:
        return handle["GasParticles"]
    raise KeyError("Neither PartType0 nor GasParticles was found in the HDF5 file.")


def get_mass_conversion_to_kg(handle: h5py.File) -> float | None:
    attr_candidates = [
        ("Units", "Unit mass in cgs (U_M)"),
        ("InternalCodeUnits", "Unit mass in cgs (U_M)"),
    ]
    for group_name, attr_name in attr_candidates:
        if group_name in handle and attr_name in handle[group_name].attrs:
            raw = handle[group_name].attrs[attr_name]
            value = raw[0] if hasattr(raw, "__len__") and not isinstance(raw, (bytes, str)) else raw
            return float(value) / 1000.0
    return None


def get_particle_masses_kg(handle: h5py.File, part_group: h5py.Group) -> list[float]:
    mass_conversion = get_mass_conversion_to_kg(handle)
    if mass_conversion is None:
        raise ValueError("Could not recover the mass-unit conversion for particle masses.")

    if "Masses" in part_group:
        return (part_group["Masses"][()] * mass_conversion).tolist()

    if "Header" in handle and "InitialMassTable" in handle["Header"].attrs:
        raw = handle["Header"].attrs["InitialMassTable"]
        initial_mass = raw[0] if hasattr(raw, "__len__") and not isinstance(raw, (bytes, str)) else raw
        if float(initial_mass) > 0.0:
            return [float(initial_mass) * mass_conversion] * len(part_group["ParticleIDs"])

    raise KeyError("Could not recover particle masses from either Masses or Header/InitialMassTable.")


def get_length_conversion_to_m(handle: h5py.File) -> float | None:
    attr_candidates = [
        ("Units", "Unit length in cgs (U_L)"),
        ("InternalCodeUnits", "Unit length in cgs (U_L)"),
    ]
    for group_name, attr_name in attr_candidates:
        if group_name in handle and attr_name in handle[group_name].attrs:
            raw = handle[group_name].attrs[attr_name]
            value = raw[0] if hasattr(raw, "__len__") and not isinstance(raw, (bytes, str)) else raw
            return float(value) * 1e-2
    return None


def get_time_conversion_to_s(handle: h5py.File) -> float | None:
    attr_candidates = [
        ("Units", "Unit time in cgs (U_t)"),
        ("InternalCodeUnits", "Unit time in cgs (U_t)"),
    ]
    for group_name, attr_name in attr_candidates:
        if group_name in handle and attr_name in handle[group_name].attrs:
            raw = handle[group_name].attrs[attr_name]
            value = raw[0] if hasattr(raw, "__len__") and not isinstance(raw, (bytes, str)) else raw
            return float(value)
    return None


def matching_physical_snapshot_path(fof_path: Path, physical_data_dir: Path | None) -> Path | None:
    if physical_data_dir is None:
        return None
    base_name = fof_path.name.split("_fof_", 1)[0] + ".hdf5"
    candidate = physical_data_dir / base_name
    return candidate if candidate.exists() else None


def get_point_mass_kg(handle: h5py.File, default_mars_mass_kg: float) -> float:
    mass_conversion = get_mass_conversion_to_kg(handle)
    if "UnusedParameters" in handle and "PointMassPotential:mass" in handle["UnusedParameters"].attrs:
        raw = handle["UnusedParameters"].attrs["PointMassPotential:mass"]
        value = raw[0] if hasattr(raw, "__len__") and not isinstance(raw, (bytes, str)) else raw
        if mass_conversion is not None:
            return float(value) * mass_conversion
    return default_mars_mass_kg


def extract_particle_physics(
    path: Path,
    default_mars_mass_kg: float,
) -> tuple[dict[int, dict[str, float]], float, str]:
    with h5py.File(path, "r") as handle:
        part_group = get_parttype_group(handle)
        if "ParticleIDs" not in part_group or "Coordinates" not in part_group or "Velocities" not in part_group:
            raise KeyError("Physical snapshot is missing one of ParticleIDs, Coordinates, or Velocities.")

        length_conversion = get_length_conversion_to_m(handle)
        time_conversion = get_time_conversion_to_s(handle)
        if length_conversion is None or time_conversion is None:
            raise ValueError("Could not recover length and time unit conversions from the physical snapshot.")

        velocity_conversion = length_conversion / time_conversion
        particle_ids = part_group["ParticleIDs"][()].tolist()
        coordinates = part_group["Coordinates"][()]
        velocities = part_group["Velocities"][()]
        masses_kg = get_particle_masses_kg(handle, part_group)

        box_size_raw = handle["Header"].attrs["BoxSize"] if "Header" in handle and "BoxSize" in handle["Header"].attrs else 0.0
        if hasattr(box_size_raw, "__len__") and not isinstance(box_size_raw, (bytes, str)):
            box_size = float(box_size_raw[0])
        else:
            box_size = float(box_size_raw)
        box_center_m = 0.5 * box_size * length_conversion
        point_mass_kg = get_point_mass_kg(handle, default_mars_mass_kg)

        particle_rows: dict[int, dict[str, float]] = {}
        for particle_id, coord, vel, mass_kg in zip(particle_ids, coordinates, velocities, masses_kg):
            x_m = float(coord[0]) * length_conversion - box_center_m
            y_m = float(coord[1]) * length_conversion - box_center_m
            z_m = float(coord[2]) * length_conversion - box_center_m
            vx_ms = float(vel[0]) * velocity_conversion
            vy_ms = float(vel[1]) * velocity_conversion
            vz_ms = float(vel[2]) * velocity_conversion
            radius_m = math.sqrt(x_m * x_m + y_m * y_m + z_m * z_m)
            speed_sq = vx_ms * vx_ms + vy_ms * vy_ms + vz_ms * vz_ms
            specific_energy = math.nan
            is_bound = False
            if radius_m > 0.0:
                specific_energy = 0.5 * speed_sq - (GRAVITATIONAL_CONSTANT * point_mass_kg / radius_m)
                is_bound = specific_energy < 0.0

            particle_rows[int(particle_id)] = {
                "mass_kg": float(mass_kg),
                "x_m": x_m,
                "y_m": y_m,
                "z_m": z_m,
                "vx_ms": vx_ms,
                "vy_ms": vy_ms,
                "vz_ms": vz_ms,
                "specific_orbital_energy_j_per_kg": specific_energy,
                "is_bound": is_bound,
            }

    return particle_rows, point_mass_kg, str(path.resolve())


def get_auto_excluded_group_ids(handle: h5py.File) -> set[int]:
    auto_ids: set[int] = set()
    if "Parameters" in handle and "FOF:group_id_default" in handle["Parameters"].attrs:
        raw = handle["Parameters"].attrs["FOF:group_id_default"]
        value = raw[0] if hasattr(raw, "__len__") and not isinstance(raw, (bytes, str)) else raw
        auto_ids.add(int(value))
    return auto_ids


def extract_group_metrics(
    path: Path,
    min_particles: int,
    exclude_group_ids: set[int],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    manifest = parse_simulation_filename(path.name)
    manifest["file_path"] = str(path.resolve())
    manifest["file_size_bytes"] = path.stat().st_size

    with h5py.File(path, "r") as handle:
        part_group = get_parttype_group(handle)
        if "FOFGroupIDs" not in part_group:
            raise KeyError("FOFGroupIDs dataset is missing.")

        all_excluded_group_ids = set(exclude_group_ids) | get_auto_excluded_group_ids(handle)
        group_ids = part_group["FOFGroupIDs"][()]
        particle_count_total = int(len(group_ids))
        if particle_count_total == 0:
            raise ValueError("FOFGroupIDs dataset is empty.")

        mass_conversion = get_mass_conversion_to_kg(handle)
        masses = None
        if "Masses" in part_group:
            masses = part_group["Masses"][()]
            if len(masses) != particle_count_total:
                raise ValueError("Masses length does not match FOFGroupIDs length.")

        valid_group_ids = [int(group_id) for group_id in group_ids.tolist() if int(group_id) not in all_excluded_group_ids]
        group_counter = Counter(valid_group_ids)
        all_groups_sorted = sorted(group_counter.items(), key=lambda item: item[0])
        selected_groups = [(group_id, count) for group_id, count in all_groups_sorted if count >= min_particles]

        group_mass_sums: dict[int, float] = {}
        total_particle_mass_kg = ""
        largest_fragment_mass_kg = ""
        total_fragment_mass_kg = ""
        fragment_mass_fraction = ""
        mass_metrics_available = False

        if masses is not None and mass_conversion is not None:
            mass_metrics_available = True
            group_mass_sums = {}
            for group_id, mass in zip(group_ids.tolist(), masses.tolist()):
                gid = int(group_id)
                if gid in all_excluded_group_ids:
                    continue
                group_mass_sums[gid] = group_mass_sums.get(gid, 0.0) + float(mass) * mass_conversion

            total_particle_mass_kg_value = sum(group_mass_sums.values())
            total_fragment_mass_kg_value = sum(group_mass_sums[group_id] for group_id, _ in selected_groups)
            largest_fragment_mass_kg_value = (
                max(group_mass_sums[group_id] for group_id, _ in selected_groups) if selected_groups else 0.0
            )
            fragment_mass_fraction_value = (
                total_fragment_mass_kg_value / total_particle_mass_kg_value if total_particle_mass_kg_value else math.nan
            )

            total_particle_mass_kg = total_particle_mass_kg_value
            total_fragment_mass_kg = total_fragment_mass_kg_value
            largest_fragment_mass_kg = largest_fragment_mass_kg_value
            fragment_mass_fraction = fragment_mass_fraction_value

        outcome_row = {
            **manifest,
            "parttype_group": part_group.name.strip("/"),
            "particle_count_total": particle_count_total,
            "n_fof_groups": len(all_groups_sorted),
            "fragment_count_min_particles": len(selected_groups),
            "largest_fragment_particle_count": max((count for _, count in selected_groups), default=0),
            "mass_metrics_available": mass_metrics_available,
            "mass_unit": "kg" if mass_metrics_available else "",
            "total_particle_mass_kg": total_particle_mass_kg,
            "largest_fragment_mass_kg": largest_fragment_mass_kg,
            "total_fragment_mass_kg": total_fragment_mass_kg,
            "fragment_mass_fraction": fragment_mass_fraction,
            "particle_count_metrics_are_proxies": True,
            "excluded_group_ids": ",".join(str(group_id) for group_id in sorted(all_excluded_group_ids)),
            "min_particles": min_particles,
        }

        fragment_rows: list[dict[str, object]] = []
        for group_id, count in all_groups_sorted:
            fragment_mass = group_mass_sums.get(group_id, "") if mass_metrics_available else ""
            fragment_fraction = (
                fragment_mass / total_particle_mass_kg if mass_metrics_available and total_particle_mass_kg else ""
            )
            fragment_rows.append(
                {
                    **manifest,
                    "parttype_group": part_group.name.strip("/"),
                    "group_id": group_id,
                    "particle_count": count,
                    "particle_fraction_of_snapshot": count / particle_count_total,
                    "passes_min_particles": count >= min_particles,
                    "mass_metrics_available": mass_metrics_available,
                    "fragment_mass_kg": fragment_mass,
                    "fragment_mass_fraction_of_snapshot": fragment_fraction,
                }
            )

    return outcome_row, fragment_rows


def main() -> int:
    args = parse_args()
    data_dir = Path(os.path.expandvars(args.data_dir)).expanduser().resolve()
    outputs_dir = Path(args.outputs_dir).expanduser().resolve()
    ensure_dir(outputs_dir)

    files = select_hdf5_files(data_dir, args.limit)
    if not files:
        raise SystemExit(f"No .hdf5 files found in {data_dir}")

    manifest_rows = build_manifest_rows(files)
    write_csv(outputs_dir / "manifest.csv", manifest_rows, MANIFEST_FIELDS)

    if args.schema_samples > 0:
        schema_rows = summarize_hdf5_schema(files[: args.schema_samples])
        write_csv(outputs_dir / "hdf5_schema_summary.csv", schema_rows, SCHEMA_FIELDS)

    outcome_rows: list[dict[str, object]] = []
    fragment_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    exclude_group_ids = set(args.exclude_group_id)

    for path in files:
        try:
            outcome_row, current_fragment_rows = extract_group_metrics(
                path=path,
                min_particles=args.min_particles,
                exclude_group_ids=exclude_group_ids,
            )
            outcome_rows.append(outcome_row)
            fragment_rows.extend(current_fragment_rows)
        except Exception as exc:  # pragma: no cover - exercised on real data
            error_rows.append(
                {
                    "file_path": str(path.resolve()),
                    "filename": path.name,
                    "error_type": exc.__class__.__name__,
                    "error_message": str(exc),
                }
            )

    write_csv(outputs_dir / "fof_outcomes.csv", outcome_rows, OUTCOME_FIELDS)
    write_csv(outputs_dir / "fragment_catalog.csv", fragment_rows, FRAGMENT_FIELDS)
    if error_rows:
        write_csv(outputs_dir / "extraction_errors.csv", error_rows, ERROR_FIELDS)
    else:
        errors_path = outputs_dir / "extraction_errors.csv"
        if errors_path.exists():
            errors_path.unlink()

    print(f"Processed {len(files)} files from {data_dir}")
    print(f"Wrote manifest: {outputs_dir / 'manifest.csv'}")
    print(f"Wrote outcomes: {outputs_dir / 'fof_outcomes.csv'}")
    print(f"Wrote fragment catalog: {outputs_dir / 'fragment_catalog.csv'}")
    if args.schema_samples > 0:
        print(f"Wrote schema summary: {outputs_dir / 'hdf5_schema_summary.csv'}")
    if error_rows:
        print(f"Encountered {len(error_rows)} extraction errors. See {outputs_dir / 'extraction_errors.csv'}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
