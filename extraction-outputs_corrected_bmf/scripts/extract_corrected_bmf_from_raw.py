#!/usr/bin/env python3
"""Build corrected parent-normalized bound/captured outputs from raw SPH snapshots."""

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


GRAVITATIONAL_CONSTANT = 6.67430e-11
DEFAULT_MARS_MASS_KG = 6.4171e23
DEFAULT_MARS_HILL_RADIUS_M = 1.08e9
PREFERRED_FOF_LINKING_LENGTH = 0.004

PAPER_TABLE2_ROWS = [
    {"scenario": "A2000_n65_r11_v00_no_spin", "mass_code": "A2000", "resolution_code": "n65", "periapsis_code": "r11", "velocity_code": "v00", "spin_code": "", "paper_f_bnd": 0.526, "paper_f_capt": 0.479},
    {"scenario": "A2000_n65_r12_v00_no_spin", "mass_code": "A2000", "resolution_code": "n65", "periapsis_code": "r12", "velocity_code": "v00", "spin_code": "", "paper_f_bnd": 0.516, "paper_f_capt": 0.444},
    {"scenario": "A2000_n65_r13_v00_no_spin", "mass_code": "A2000", "resolution_code": "n65", "periapsis_code": "r13", "velocity_code": "v00", "spin_code": "", "paper_f_bnd": 0.516, "paper_f_capt": 0.424},
    {"scenario": "A2000_n65_r14_v00_no_spin", "mass_code": "A2000", "resolution_code": "n65", "periapsis_code": "r14", "velocity_code": "v00", "spin_code": "", "paper_f_bnd": 0.513, "paper_f_capt": 0.442},
    {"scenario": "A2000_n65_r15_v00_no_spin", "mass_code": "A2000", "resolution_code": "n65", "periapsis_code": "r15", "velocity_code": "v00", "spin_code": "", "paper_f_bnd": 0.434, "paper_f_capt": 0.428},
    {"scenario": "A2000_n65_r16_v00_no_spin", "mass_code": "A2000", "resolution_code": "n65", "periapsis_code": "r16", "velocity_code": "v00", "spin_code": "", "paper_f_bnd": 0.466, "paper_f_capt": 0.368},
    {"scenario": "A2000_n65_r17_v00_no_spin", "mass_code": "A2000", "resolution_code": "n65", "periapsis_code": "r17", "velocity_code": "v00", "spin_code": "", "paper_f_bnd": 0.388, "paper_f_capt": 0.336},
    {"scenario": "A2000_n65_r18_v00_no_spin", "mass_code": "A2000", "resolution_code": "n65", "periapsis_code": "r18", "velocity_code": "v00", "spin_code": "", "paper_f_bnd": 0.238, "paper_f_capt": 0.235},
    {"scenario": "A2000_n65_r19_v00_no_spin", "mass_code": "A2000", "resolution_code": "n65", "periapsis_code": "r19", "velocity_code": "v00", "spin_code": "", "paper_f_bnd": 0.181, "paper_f_capt": 0.181},
    {"scenario": "A2000_n65_r20_v00_no_spin", "mass_code": "A2000", "resolution_code": "n65", "periapsis_code": "r20", "velocity_code": "v00", "spin_code": "", "paper_f_bnd": 0.000, "paper_f_capt": 0.000},
    {"scenario": "A2000_n65_r22_v00_no_spin", "mass_code": "A2000", "resolution_code": "n65", "periapsis_code": "r22", "velocity_code": "v00", "spin_code": "", "paper_f_bnd": 0.000, "paper_f_capt": 0.000},
    {"scenario": "A2000_n65_r24_v00_no_spin", "mass_code": "A2000", "resolution_code": "n65", "periapsis_code": "r24", "velocity_code": "v00", "spin_code": "", "paper_f_bnd": 0.000, "paper_f_capt": 0.000},
    {"scenario": "A2000_n65_r12_v02_no_spin", "mass_code": "A2000", "resolution_code": "n65", "periapsis_code": "r12", "velocity_code": "v02", "spin_code": "", "paper_f_bnd": 0.493, "paper_f_capt": 0.441},
    {"scenario": "A2000_n65_r14_v02_no_spin", "mass_code": "A2000", "resolution_code": "n65", "periapsis_code": "r14", "velocity_code": "v02", "spin_code": "", "paper_f_bnd": 0.452, "paper_f_capt": 0.405},
    {"scenario": "A2000_n65_r16_v02_no_spin", "mass_code": "A2000", "resolution_code": "n65", "periapsis_code": "r16", "velocity_code": "v02", "spin_code": "", "paper_f_bnd": 0.392, "paper_f_capt": 0.352},
    {"scenario": "A2000_n65_r18_v02_no_spin", "mass_code": "A2000", "resolution_code": "n65", "periapsis_code": "r18", "velocity_code": "v02", "spin_code": "", "paper_f_bnd": 0.237, "paper_f_capt": 0.188},
    {"scenario": "A2000_n65_r20_v02_no_spin", "mass_code": "A2000", "resolution_code": "n65", "periapsis_code": "r20", "velocity_code": "v02", "spin_code": "", "paper_f_bnd": 0.121, "paper_f_capt": 0.000},
    {"scenario": "A2000_n65_r12_v04_no_spin", "mass_code": "A2000", "resolution_code": "n65", "periapsis_code": "r12", "velocity_code": "v04", "spin_code": "", "paper_f_bnd": 0.426, "paper_f_capt": 0.344},
    {"scenario": "A2000_n65_r14_v04_no_spin", "mass_code": "A2000", "resolution_code": "n65", "periapsis_code": "r14", "velocity_code": "v04", "spin_code": "", "paper_f_bnd": 0.380, "paper_f_capt": 0.330},
    {"scenario": "A2000_n65_r16_v04_no_spin", "mass_code": "A2000", "resolution_code": "n65", "periapsis_code": "r16", "velocity_code": "v04", "spin_code": "", "paper_f_bnd": 0.334, "paper_f_capt": 0.298},
    {"scenario": "A2000_n65_r18_v04_no_spin", "mass_code": "A2000", "resolution_code": "n65", "periapsis_code": "r18", "velocity_code": "v04", "spin_code": "", "paper_f_bnd": 0.183, "paper_f_capt": 0.168},
    {"scenario": "A2000_n65_r20_v04_no_spin", "mass_code": "A2000", "resolution_code": "n65", "periapsis_code": "r20", "velocity_code": "v04", "spin_code": "", "paper_f_bnd": 0.000, "paper_f_capt": 0.000},
    {"scenario": "A2000_n65_r12_v06_no_spin", "mass_code": "A2000", "resolution_code": "n65", "periapsis_code": "r12", "velocity_code": "v06", "spin_code": "", "paper_f_bnd": 0.395, "paper_f_capt": 0.278},
    {"scenario": "A2000_n65_r14_v06_no_spin", "mass_code": "A2000", "resolution_code": "n65", "periapsis_code": "r14", "velocity_code": "v06", "spin_code": "", "paper_f_bnd": 0.321, "paper_f_capt": 0.243},
    {"scenario": "A2000_n65_r16_v06_no_spin", "mass_code": "A2000", "resolution_code": "n65", "periapsis_code": "r16", "velocity_code": "v06", "spin_code": "", "paper_f_bnd": 0.253, "paper_f_capt": 0.213},
    {"scenario": "A2000_n65_r18_v06_no_spin", "mass_code": "A2000", "resolution_code": "n65", "periapsis_code": "r18", "velocity_code": "v06", "spin_code": "", "paper_f_bnd": 0.112, "paper_f_capt": 0.087},
    {"scenario": "A2000_n65_r20_v06_no_spin", "mass_code": "A2000", "resolution_code": "n65", "periapsis_code": "r20", "velocity_code": "v06", "spin_code": "", "paper_f_bnd": 0.000, "paper_f_capt": 0.000},
]

FILENAME_RE = re.compile(
    r"^(?P<prefix>Ma_xp)_(?P<mass>A\d{4}(?:c30)?)"
    r"(?:_(?P<spin>s[^_]+))?"
    r"_(?P<resolution>n\d+)_(?P<periapsis>r\d+)_(?P<velocity>v\d+)_(?P<timestep>\d+)\.hdf5$"
)
FOF_BASE_RE = re.compile(r"^(?P<base>.+?)_fof_[^_]+_[^_]+\.hdf5$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("martian_moon_data"),
        help="Directory containing raw physical SPH HDF5 snapshots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("extraction-outputs_corrected_bmf/tables"),
        help="Directory for corrected table outputs.",
    )
    parser.add_argument(
        "--bound-outcomes",
        type=Path,
        default=Path("extraction-outputs/tables/bound_outcomes.csv"),
        help="Existing pipeline bound_outcomes.csv used only to preserve FoF-compatible columns.",
    )
    parser.add_argument(
        "--fof-outcomes",
        type=Path,
        default=Path("extraction-outputs/tables/fof_outcomes.csv"),
        help="Existing pipeline fof_outcomes.csv used only to preserve FoF-compatible columns.",
    )
    parser.add_argument("--mars-mass-kg", type=float, default=DEFAULT_MARS_MASS_KG)
    parser.add_argument("--mars-hill-radius-m", type=float, default=DEFAULT_MARS_HILL_RADIUS_M)
    return parser.parse_args()


def attr_scalar(attrs: Any, name: str) -> float:
    raw = attrs[name]
    if hasattr(raw, "__len__") and not isinstance(raw, (bytes, str)):
        raw = raw[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return float(raw)


def get_part_group(handle: h5py.File) -> h5py.Group:
    if "PartType0" in handle:
        return handle["PartType0"]
    if "GasParticles" in handle:
        return handle["GasParticles"]
    raise KeyError("Neither PartType0 nor GasParticles exists in snapshot.")


def particle_masses_kg(handle: h5py.File, part_group: h5py.Group) -> np.ndarray:
    if "Masses" in part_group:
        factor_cgs = attr_scalar(
            part_group["Masses"].attrs,
            "Conversion factor to physical CGS (including cosmological corrections)",
        )
        return part_group["Masses"][...].astype(np.float64, copy=False) * factor_cgs * 1e-3

    initial_mass_table = handle["Header"].attrs["InitialMassTable"]
    initial_mass_code = float(initial_mass_table[0])
    unit_mass_kg = attr_scalar(handle["Units"].attrs, "Unit mass in cgs (U_M)") * 1e-3
    return np.full(len(part_group["ParticleIDs"]), initial_mass_code * unit_mass_kg, dtype=np.float64)


def positions_m(handle: h5py.File, part_group: h5py.Group) -> np.ndarray:
    factor_cgs = attr_scalar(
        part_group["Coordinates"].attrs,
        "Conversion factor to physical CGS (including cosmological corrections)",
    )
    coords_m = part_group["Coordinates"][...].astype(np.float64, copy=False) * factor_cgs * 1e-2
    box_size_raw = handle["Header"].attrs["BoxSize"]
    box_size_code = float(box_size_raw[0] if hasattr(box_size_raw, "__len__") else box_size_raw)
    unit_length_m = attr_scalar(handle["Units"].attrs, "Unit length in cgs (U_L)") * 1e-2
    return coords_m - 0.5 * box_size_code * unit_length_m


def velocities_m_s(part_group: h5py.Group) -> np.ndarray:
    factor_cgs = attr_scalar(
        part_group["Velocities"].attrs,
        "Conversion factor to physical CGS (including cosmological corrections)",
    )
    return part_group["Velocities"][...].astype(np.float64, copy=False) * factor_cgs * 1e-2


def point_mass_kg(handle: h5py.File, fallback_mars_mass_kg: float) -> float:
    unit_mass_kg = attr_scalar(handle["Units"].attrs, "Unit mass in cgs (U_M)") * 1e-3
    for group_name in ("Parameters", "UnusedParameters"):
        if group_name in handle and "PointMassPotential:mass" in handle[group_name].attrs:
            raw = handle[group_name].attrs["PointMassPotential:mass"]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if hasattr(raw, "__len__") and not isinstance(raw, (bytes, str)):
                raw = raw[0]
            return float(raw) * unit_mass_kg
    return fallback_mars_mass_kg


def parse_filename_metadata(filename):
    match = FILENAME_RE.match(filename)
    if not match:
        raise ValueError(f"Unrecognized raw snapshot filename: {filename}")
    mass_code = match.group("mass")
    spin_code = match.group("spin") or ""
    return {
        "physical_file": filename,
        "mass_code": mass_code,
        "resolution_code": match.group("resolution"),
        "periapsis_code": match.group("periapsis"),
        "velocity_code": match.group("velocity"),
        "spin_code": spin_code,
        "timestep": int(match.group("timestep")),
    }


def format_float(value, digits=12):
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return ""
    return f"{numeric:.{digits}g}"


def read_csv_rows(path: Path):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def compute_particle_level_metrics(snapshot_path, fallback_mars_mass_kg, mars_hill_radius_m):
    with h5py.File(snapshot_path, "r") as handle:
        part_group = get_part_group(handle)
        masses_kg = particle_masses_kg(handle, part_group)
        pos_m = positions_m(handle, part_group)
        vel_m_s = velocities_m_s(part_group)
        mars_mass_kg = point_mass_kg(handle, fallback_mars_mass_kg)
        time_raw = handle["Header"].attrs["Time"]
        time_code = float(time_raw[0] if hasattr(time_raw, "__len__") else time_raw)

    mu = GRAVITATIONAL_CONSTANT * mars_mass_kg
    radius_m = np.linalg.norm(pos_m, axis=1)
    speed_sq = np.sum(vel_m_s * vel_m_s, axis=1)
    valid_radius = np.isfinite(radius_m) & (radius_m > 0.0)
    specific_energy = np.full(len(radius_m), np.nan, dtype=np.float64)
    specific_energy[valid_radius] = 0.5 * speed_sq[valid_radius] - mu / radius_m[valid_radius]
    bound_mask = np.isfinite(specific_energy) & (specific_energy < 0.0)

    angular_momentum = np.cross(pos_m, vel_m_s)
    h_sq = np.sum(angular_momentum * angular_momentum, axis=1)
    eccentricity_sq = np.full(len(radius_m), np.nan, dtype=np.float64)
    finite_orbit = np.isfinite(specific_energy) & np.isfinite(h_sq)
    eccentricity_sq[finite_orbit] = 1.0 + (2.0 * specific_energy[finite_orbit] * h_sq[finite_orbit]) / (mu * mu)
    eccentricity = np.sqrt(np.maximum(eccentricity_sq, 0.0))

    semi_major_axis_m = np.full(len(radius_m), np.nan, dtype=np.float64)
    valid_bound = bound_mask & (np.abs(specific_energy) > 0.0)
    semi_major_axis_m[valid_bound] = -mu / (2.0 * specific_energy[valid_bound])

    apoapsis_m = np.full(len(radius_m), np.nan, dtype=np.float64)
    apoapsis_m[valid_bound] = semi_major_axis_m[valid_bound] * (1.0 + eccentricity[valid_bound])
    captured_mask = bound_mask & np.isfinite(apoapsis_m) & (apoapsis_m < mars_hill_radius_m)

    target_mass_kg = float(np.sum(masses_kg))
    bound_mass_kg = float(np.sum(masses_kg[bound_mask]))
    captured_mass_kg = float(np.sum(masses_kg[captured_mask]))
    unbound_mass_kg = target_mass_kg - bound_mass_kg

    return {
        "raw_hdf5_path": str(snapshot_path.resolve()),
        "target_mass_kg": target_mass_kg,
        "bound_mass_kg": bound_mass_kg,
        "captured_mass_kg": captured_mass_kg,
        "unbound_mass_kg": unbound_mass_kg,
        "f_bnd_parent": bound_mass_kg / target_mass_kg if target_mass_kg else math.nan,
        "f_capt_parent": captured_mass_kg / target_mass_kg if target_mass_kg else math.nan,
        "unbound_mass_fraction_parent": unbound_mass_kg / target_mass_kg if target_mass_kg else math.nan,
        "particle_count": int(len(masses_kg)),
        "bound_particle_count": int(np.count_nonzero(bound_mask)),
        "captured_particle_count": int(np.count_nonzero(captured_mask)),
        "mars_mass_kg": mars_mass_kg,
        "mars_hill_radius_m": mars_hill_radius_m,
        "time_code_units": time_code,
        "method": "particle_level_raw_snapshot_parent_normalized",
        "unresolved_handling": (
            "All raw asteroid particles in the physical snapshot are included in the denominator; "
            "all raw bound/captured asteroid particles are included in the numerator. "
            "No FoF background or unresolved material is discarded for the particle-level metrics."
        ),
        "notes": (
            "Particle-level Mars-centric energy and apoapsis classification from raw SPH snapshot. "
            "FoF-group diagnostics are preserved from existing CSV outputs only because FoF HDF5 catalogues are not available locally."
        ),
    }


def choose_preferred_linking_length(physical_file, bound_rows):
    candidates = [row.get("fof_linking_length", "") for row in bound_rows if row.get("physical_file") == physical_file]
    numeric_candidates = []
    for candidate in candidates:
        try:
            numeric_candidates.append((abs(float(candidate) - PREFERRED_FOF_LINKING_LENGTH), float(candidate), candidate))
        except (TypeError, ValueError):
            continue
    if not numeric_candidates:
        return ""
    numeric_candidates.sort()
    return numeric_candidates[0][2]


def build_paper_rows(raw_files, bound_rows, fallback_mars_mass_kg, mars_hill_radius_m):
    rows = []
    metrics_by_physical = {}
    for snapshot_path in raw_files:
        meta = parse_filename_metadata(snapshot_path.name)
        try:
            metrics = compute_particle_level_metrics(snapshot_path, fallback_mars_mass_kg, mars_hill_radius_m)
            row = {
                **meta,
                "raw_hdf5_path": metrics["raw_hdf5_path"],
                "fof_linking_length": choose_preferred_linking_length(snapshot_path.name, bound_rows),
                "target_mass_kg": metrics["target_mass_kg"],
                "bound_mass_kg": metrics["bound_mass_kg"],
                "captured_mass_kg": metrics["captured_mass_kg"],
                "f_bnd_parent": metrics["f_bnd_parent"],
                "f_capt_parent": metrics["f_capt_parent"],
                "particle_count": metrics["particle_count"],
                "bound_particle_count": metrics["bound_particle_count"],
                "captured_particle_count": metrics["captured_particle_count"],
                "mars_mass_kg": metrics["mars_mass_kg"],
                "mars_hill_radius_m": metrics["mars_hill_radius_m"],
                "method": metrics["method"],
                "unresolved_handling": metrics["unresolved_handling"],
                "notes": metrics["notes"],
            }
            rows.append(row)
            metrics_by_physical[snapshot_path.name] = {**meta, **metrics}
        except Exception as exc:
            rows.append(
                {
                    **meta,
                    "raw_hdf5_path": str(snapshot_path.resolve()),
                    "fof_linking_length": choose_preferred_linking_length(snapshot_path.name, bound_rows),
                    "target_mass_kg": "",
                    "bound_mass_kg": "",
                    "captured_mass_kg": "",
                    "f_bnd_parent": "",
                    "f_capt_parent": "",
                    "particle_count": "",
                    "bound_particle_count": "",
                    "captured_particle_count": "",
                    "mars_mass_kg": "",
                    "mars_hill_radius_m": mars_hill_radius_m,
                    "method": "particle_level_raw_snapshot_parent_normalized_failed",
                    "unresolved_handling": "Raw particle-level extraction failed before classification.",
                    "notes": "Particle-level extraction failed: {}".format(exc),
                }
            )
    rows.sort(key=lambda item: item["physical_file"])
    return rows, metrics_by_physical


def corrected_bound_rows(bound_rows, metrics_by_physical):
    corrected_rows = []
    for row in bound_rows:
        metrics = metrics_by_physical.get(row["physical_file"])
        corrected = dict(row)
        if metrics:
            corrected["target_mass_kg"] = format_float(metrics["target_mass_kg"])
            corrected["bound_mass_kg"] = format_float(metrics["bound_mass_kg"])
            corrected["bound_mass_fraction"] = format_float(metrics["f_bnd_parent"])
            corrected["captured_mass_kg"] = format_float(metrics["captured_mass_kg"])
            corrected["captured_mass_fraction"] = format_float(metrics["f_capt_parent"])
            corrected["unbound_mass_kg"] = format_float(metrics["unbound_mass_kg"])
            corrected["unbound_mass_fraction"] = format_float(metrics["unbound_mass_fraction_parent"])
            corrected["raw_hdf5_path"] = metrics["raw_hdf5_path"]
            corrected["method"] = metrics["method"]
            corrected["unresolved_handling"] = metrics["unresolved_handling"]
            corrected["normalization_notes"] = (
                "Corrected bound_mass_fraction uses bound_mass_kg / target_mass_kg from raw particle-level Mars-centric orbit tests."
            )
        corrected_rows.append(corrected)
    return corrected_rows


def physical_file_from_fof_row(row):
    filename = row.get("filename", "")
    match = FOF_BASE_RE.match(filename)
    if not match:
        return ""
    return match.group("base") + ".hdf5"


def corrected_fof_rows(fof_rows, bound_rows, metrics_by_physical):
    bound_lookup = {
        (
            row.get("physical_file", ""),
            row.get("fof_linking_length", ""),
            row.get("timestep", ""),
            row.get("mass_code", ""),
            row.get("resolution_code", ""),
            row.get("periapsis_code", ""),
            row.get("velocity_code", ""),
            row.get("spin_code", ""),
        ): row
        for row in bound_rows
    }
    extra_fields = [
        "fof_file",
        "physical_file",
        "bound_fragment_count",
        "bound_mass_kg_existing_fof_csv",
        "bound_mass_fraction_existing_fof_csv",
        "largest_bound_fragment_mass_kg",
        "unbound_fragment_count",
        "unbound_mass_kg_existing_fof_csv",
        "unbound_mass_fraction_existing_fof_csv",
        "largest_unbound_fragment_mass_kg",
        "target_mass_kg",
        "bound_mass_kg",
        "bound_mass_fraction",
        "captured_mass_kg",
        "captured_mass_fraction",
        "unbound_mass_kg",
        "unbound_mass_fraction",
        "raw_hdf5_path",
        "method",
        "unresolved_handling",
    ]
    corrected_rows = []
    for row in fof_rows:
        physical_file = physical_file_from_fof_row(row)
        key = (
            physical_file,
            row.get("fof_linking_length", ""),
            row.get("timestep", ""),
            row.get("mass_code", ""),
            row.get("resolution_code", ""),
            row.get("periapsis_code", ""),
            row.get("velocity_code", ""),
            row.get("spin_code", ""),
        )
        bound_row = bound_lookup.get(key, {})
        metrics = metrics_by_physical.get(physical_file)
        corrected = dict(row)
        corrected["fof_file"] = bound_row.get("fof_file", "")
        corrected["physical_file"] = bound_row.get("physical_file", physical_file)
        corrected["bound_fragment_count"] = bound_row.get("bound_fragment_count", "")
        corrected["bound_mass_kg_existing_fof_csv"] = bound_row.get("bound_mass_kg", "")
        corrected["bound_mass_fraction_existing_fof_csv"] = bound_row.get("bound_mass_fraction", "")
        corrected["largest_bound_fragment_mass_kg"] = bound_row.get("largest_bound_fragment_mass_kg", "")
        corrected["unbound_fragment_count"] = bound_row.get("unbound_fragment_count", "")
        corrected["unbound_mass_kg_existing_fof_csv"] = bound_row.get("unbound_mass_kg", "")
        corrected["unbound_mass_fraction_existing_fof_csv"] = bound_row.get("unbound_mass_fraction", "")
        corrected["largest_unbound_fragment_mass_kg"] = bound_row.get("largest_unbound_fragment_mass_kg", "")
        if metrics:
            corrected["target_mass_kg"] = format_float(metrics["target_mass_kg"])
            corrected["bound_mass_kg"] = format_float(metrics["bound_mass_kg"])
            corrected["bound_mass_fraction"] = format_float(metrics["f_bnd_parent"])
            corrected["captured_mass_kg"] = format_float(metrics["captured_mass_kg"])
            corrected["captured_mass_fraction"] = format_float(metrics["f_capt_parent"])
            corrected["unbound_mass_kg"] = format_float(metrics["unbound_mass_kg"])
            corrected["unbound_mass_fraction"] = format_float(metrics["unbound_mass_fraction_parent"])
            corrected["raw_hdf5_path"] = metrics["raw_hdf5_path"]
            corrected["method"] = metrics["method"]
            corrected["unresolved_handling"] = metrics["unresolved_handling"]
        corrected_rows.append(corrected)
    return corrected_rows, extra_fields


def validation_rows(metrics_by_physical):
    lookup = defaultdict(dict)
    for physical_file, row in metrics_by_physical.items():
        lookup[
            (
                row["mass_code"],
                row["resolution_code"],
                row["periapsis_code"],
                row["velocity_code"],
                row["spin_code"],
            )
        ] = row

    rows = []
    for paper_row in PAPER_TABLE2_ROWS:
        ours = lookup.get(
            (
                paper_row["mass_code"],
                paper_row["resolution_code"],
                paper_row["periapsis_code"],
                paper_row["velocity_code"],
                paper_row["spin_code"],
            )
        )
        our_f_bnd = ours["f_bnd_parent"] if ours else math.nan
        our_f_capt = ours["f_capt_parent"] if ours else math.nan
        rows.append(
            {
                "scenario": paper_row["scenario"],
                "paper_f_bnd": paper_row["paper_f_bnd"],
                "our_f_bnd_parent": our_f_bnd if ours else "",
                "difference_f_bnd": (our_f_bnd - paper_row["paper_f_bnd"]) if ours else "",
                "paper_f_capt": paper_row["paper_f_capt"],
                "our_f_capt_parent": our_f_capt if ours else "",
                "difference_f_capt": (our_f_capt - paper_row["paper_f_capt"]) if ours else "",
            }
        )
    return rows


def build_summary(paper_rows, bound_rows, fof_rows, validation):
    matched_validation = [row for row in validation if row["our_f_bnd_parent"] != ""]
    validation_diff_bnd = [float(row["difference_f_bnd"]) for row in matched_validation]
    validation_diff_capt = [float(row["difference_f_capt"]) for row in matched_validation]
    fiducial = next((row for row in validation if row["scenario"] == "A2000_n65_r16_v00_no_spin"), None)

    lines = [
        "Corrected BMF extraction summary",
        "",
        "Exact formulas",
        "f_bnd_parent = bound_mass_kg / target_mass_kg",
        "f_capt_parent = captured_mass_kg / target_mass_kg",
        "",
        "Extraction level",
        "Paper-style mass fractions are particle-level from raw physical HDF5 snapshots.",
        "FoF-grouped diagnostics are not regenerated from raw data in this pass because FoF HDF5 catalogues are not available locally.",
        "",
        "Bound/capture classification",
        "Bound if specific orbital energy epsilon = v^2 / 2 - G M_Mars / r is negative in the Mars-centric frame.",
        "Apoapsis computed for bound particles as Q = a (1 + e), with a = -G M_Mars / (2 epsilon).",
        f"Capture requires bound orbit and Q < R_Hill, using R_Hill = {DEFAULT_MARS_HILL_RADIUS_M:.6e} m.",
        "",
        "Material accounting",
        "Denominator: all asteroid particles in each raw physical snapshot.",
        "Numerator for f_bnd_parent: all asteroid particles with negative Mars-centric orbital energy.",
        "Numerator for f_capt_parent: all asteroid particles with negative Mars-centric orbital energy and apoapsis below the Mars Hill radius.",
        "Unresolved/background/unassigned particles: included for the particle-level metrics because the raw physical snapshots contain the asteroid particles directly and no FoF filtering is applied.",
        "",
        "Approximation status",
        "Paper-style parent-normalized bound/captured fractions are an exact particle-level reconstruction under the raw-snapshot assumptions above.",
        "FoF-level fragment columns in corrected bound_outcomes.csv and corrected fof_outcomes.csv remain inherited approximations from the existing CSV pipeline because FoF HDF5 group catalogues were not available for re-extraction.",
        "",
        "Row counts",
        f"paper_metric_rows={len(paper_rows)}",
        f"corrected_bound_outcome_rows={len(bound_rows)}",
        f"corrected_fof_outcome_rows={len(fof_rows)}",
        f"validation_rows_total={len(validation)}",
        f"validation_rows_matched_to_archive={len(matched_validation)}",
        "",
        "Missing-data counts",
        f"paper_metric_missing_bound_mass_rows={sum(1 for row in paper_rows if row['bound_mass_kg'] in ('', None))}",
        f"paper_metric_missing_captured_mass_rows={sum(1 for row in paper_rows if row['captured_mass_kg'] in ('', None))}",
        f"bound_outcome_rows_missing_raw_match={sum(1 for row in bound_rows if row.get('target_mass_kg', '') == '')}",
        f"fof_outcome_rows_missing_raw_match={sum(1 for row in fof_rows if row.get('target_mass_kg', '') == '')}",
        "",
        "Validation",
        f"validation_mean_signed_difference_f_bnd={np.mean(validation_diff_bnd):.6f}" if validation_diff_bnd else "validation_mean_signed_difference_f_bnd=n/a",
        f"validation_mean_signed_difference_f_capt={np.mean(validation_diff_capt):.6f}" if validation_diff_capt else "validation_mean_signed_difference_f_capt=n/a",
        f"validation_max_abs_difference_f_bnd={np.max(np.abs(validation_diff_bnd)):.6f}" if validation_diff_bnd else "validation_max_abs_difference_f_bnd=n/a",
        f"validation_max_abs_difference_f_capt={np.max(np.abs(validation_diff_capt)):.6f}" if validation_diff_capt else "validation_max_abs_difference_f_capt=n/a",
    ]
    if fiducial:
        lines.extend(
            [
                f"fiducial_case_scenario={fiducial['scenario']}",
                f"fiducial_paper_f_bnd={fiducial['paper_f_bnd']}",
                f"fiducial_our_f_bnd_parent={fiducial['our_f_bnd_parent']}",
                f"fiducial_difference_f_bnd={fiducial['difference_f_bnd']}",
                f"fiducial_paper_f_capt={fiducial['paper_f_capt']}",
                f"fiducial_our_f_capt_parent={fiducial['our_f_capt_parent']}",
                f"fiducial_difference_f_capt={fiducial['difference_f_capt']}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    raw_dir = args.raw_dir if args.raw_dir.is_absolute() else repo_root / args.raw_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    bound_path = args.bound_outcomes if args.bound_outcomes.is_absolute() else repo_root / args.bound_outcomes
    fof_path = args.fof_outcomes if args.fof_outcomes.is_absolute() else repo_root / args.fof_outcomes
    output_dir.mkdir(parents=True, exist_ok=True)

    bound_rows_original, bound_fields = read_csv_rows(bound_path)
    fof_rows_original, fof_fields = read_csv_rows(fof_path)
    raw_files = sorted(path for path in raw_dir.glob("*.hdf5") if "_fof_" not in path.name)

    paper_rows, metrics_by_physical = build_paper_rows(
        raw_files=raw_files,
        bound_rows=bound_rows_original,
        fallback_mars_mass_kg=args.mars_mass_kg,
        mars_hill_radius_m=args.mars_hill_radius_m,
    )
    corrected_bound = corrected_bound_rows(bound_rows_original, metrics_by_physical)
    corrected_fof, fof_extra_fields = corrected_fof_rows(fof_rows_original, bound_rows_original, metrics_by_physical)
    validation = validation_rows(metrics_by_physical)

    paper_fields = [
        "physical_file",
        "raw_hdf5_path",
        "mass_code",
        "resolution_code",
        "periapsis_code",
        "velocity_code",
        "spin_code",
        "timestep",
        "fof_linking_length",
        "target_mass_kg",
        "bound_mass_kg",
        "captured_mass_kg",
        "f_bnd_parent",
        "f_capt_parent",
        "particle_count",
        "bound_particle_count",
        "captured_particle_count",
        "mars_mass_kg",
        "mars_hill_radius_m",
        "method",
        "unresolved_handling",
        "notes",
    ]
    bound_output_fields = list(bound_fields)
    for field in ["target_mass_kg", "captured_mass_kg", "captured_mass_fraction", "raw_hdf5_path", "method", "unresolved_handling", "normalization_notes"]:
        if field not in bound_output_fields:
            bound_output_fields.append(field)
    fof_output_fields = list(fof_fields)
    for field in fof_extra_fields:
        if field not in fof_output_fields:
            fof_output_fields.append(field)

    write_csv(output_dir / "corrected_bmf_paper_metrics.csv", paper_rows, paper_fields)
    write_csv(output_dir / "bound_outcomes.csv", corrected_bound, bound_output_fields)
    write_csv(output_dir / "fof_outcomes.csv", corrected_fof, fof_output_fields)
    write_csv(
        output_dir / "corrected_bmf_validation_cases.csv",
        validation,
        ["scenario", "paper_f_bnd", "our_f_bnd_parent", "difference_f_bnd", "paper_f_capt", "our_f_capt_parent", "difference_f_capt"],
    )
    (output_dir / "corrected_bmf_extraction_summary.txt").write_text(
        build_summary(paper_rows, corrected_bound, corrected_fof, validation),
        encoding="utf-8",
    )

    print(f"Processed raw snapshots: {len(raw_files)}")
    print(f"Wrote: {output_dir / 'corrected_bmf_paper_metrics.csv'}")
    print(f"Wrote: {output_dir / 'bound_outcomes.csv'}")
    print(f"Wrote: {output_dir / 'fof_outcomes.csv'}")
    print(f"Wrote: {output_dir / 'corrected_bmf_validation_cases.csv'}")
    print(f"Wrote: {output_dir / 'corrected_bmf_extraction_summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
