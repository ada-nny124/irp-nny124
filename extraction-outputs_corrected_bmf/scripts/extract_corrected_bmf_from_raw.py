#!/usr/bin/env python3
"""Compute corrected parent-mass bound/captured fractions from raw HDF5 snapshots."""

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np


GRAVITATIONAL_CONSTANT = 6.67430e-11
DEFAULT_MARS_MASS_KG = 6.4171e23
DEFAULT_MARS_RADIUS_M = 3.3895e6
DEFAULT_MARS_HILL_RADIUS_M = 1.08e9
VALIDATION_FILE = "Ma_xp_A2000_n65_r16_v00_90000.hdf5"
PAPER_F_BND = 0.466
PAPER_F_CAPT = 0.368

FILENAME_RE = re.compile(
    r"^(?P<prefix>Ma_xp)_(?P<mass>A\d{4}(?:c30)?)(?:_(?P<spin>s[^_]+))?"
    r"_(?P<resolution>n\d+)_(?P<periapsis>r\d+)_(?P<velocity>v\d+)_(?P<timestep>\d+)\.hdf5$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("extraction-outputs_corrected_bmf/tables/corrected_bmf_raw_snapshot_manifest.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("extraction-outputs_corrected_bmf/tables/corrected_bmf_reference_case_only.csv"),
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=Path("extraction-outputs_corrected_bmf/diagnostics"),
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Process only the known reference case.",
    )
    parser.add_argument("--mars-mass-kg", type=float, default=DEFAULT_MARS_MASS_KG)
    parser.add_argument("--mars-radius-m", type=float, default=DEFAULT_MARS_RADIUS_M)
    parser.add_argument("--mars-hill-radius-m", type=float, default=DEFAULT_MARS_HILL_RADIUS_M)
    return parser.parse_args()


def parse_manifest(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_filename_metadata(filename):
    match = FILENAME_RE.match(filename)
    if not match:
        raise ValueError(f"Unrecognized raw snapshot filename: {filename}")
    mass_code = match.group("mass")
    spin_code = match.group("spin") or ""
    return {
        "physical_file": filename,
        "mass_code": mass_code,
        "spin_code": spin_code,
        "resolution_code": match.group("resolution"),
        "periapsis_code": match.group("periapsis"),
        "velocity_code": match.group("velocity"),
        "timestep": int(match.group("timestep")),
        "mass": int(mass_code[1:5]),
        "periapsis": int(match.group("periapsis")[1:]) / 10.0,
        "v_inf": int(match.group("velocity")[1:]) / 10.0,
        "spin": spin_code or "none",
        "resolution": int(match.group("resolution")[1:]),
    }


def attr_scalar(group, name):
    raw = group.attrs[name]
    if hasattr(raw, "__len__") and not isinstance(raw, (bytes, str)):
        raw = raw[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return float(raw)


def point_mass_kg(handle, fallback_mars_mass_kg):
    unit_mass_kg = attr_scalar(handle["Units"], "Unit mass in cgs (U_M)") * 1e-3
    for group_name in ("Parameters", "UnusedParameters"):
        if group_name in handle and "PointMassPotential:mass" in handle[group_name].attrs:
            raw = handle[group_name].attrs["PointMassPotential:mass"]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if hasattr(raw, "__len__") and not isinstance(raw, (bytes, str)):
                raw = raw[0]
            return float(raw) * unit_mass_kg
    return fallback_mars_mass_kg


def particle_masses_kg(handle, part_group):
    if "Masses" in part_group:
        factor_cgs = attr_scalar(
            part_group["Masses"],
            "Conversion factor to physical CGS (including cosmological corrections)",
        )
        return part_group["Masses"][...].astype(np.float64, copy=False) * factor_cgs * 1e-3

    initial_mass_code = float(handle["Header"].attrs["InitialMassTable"][0])
    unit_mass_kg = attr_scalar(handle["Units"], "Unit mass in cgs (U_M)") * 1e-3
    return np.full(len(part_group["ParticleIDs"]), initial_mass_code * unit_mass_kg, dtype=np.float64)


def positions_m(handle, part_group):
    factor_cgs = attr_scalar(
        part_group["Coordinates"],
        "Conversion factor to physical CGS (including cosmological corrections)",
    )
    coords_m = part_group["Coordinates"][...].astype(np.float64, copy=False) * factor_cgs * 1e-2
    box_size_code = handle["Header"].attrs["BoxSize"]
    box_size_code = float(box_size_code[0] if hasattr(box_size_code, "__len__") else box_size_code)
    unit_length_m = attr_scalar(handle["Units"], "Unit length in cgs (U_L)") * 1e-2
    return coords_m - 0.5 * box_size_code * unit_length_m


def velocities_m_s(part_group):
    factor_cgs = attr_scalar(
        part_group["Velocities"],
        "Conversion factor to physical CGS (including cosmological corrections)",
    )
    return part_group["Velocities"][...].astype(np.float64, copy=False) * factor_cgs * 1e-2


def compute_corrected_metrics(snapshot_path, mars_mass_kg_default, mars_radius_m, mars_hill_radius_m):
    with h5py.File(snapshot_path, "r") as handle:
        part_group = handle["PartType0"] if "PartType0" in handle else handle["GasParticles"]
        particle_ids = part_group["ParticleIDs"][...].astype(np.uint64, copy=False)
        masses_kg = particle_masses_kg(handle, part_group)
        pos_m = positions_m(handle, part_group)
        vel_m_s = velocities_m_s(part_group)
        mars_mass_kg = point_mass_kg(handle, mars_mass_kg_default)
        time_code = float(handle["Header"].attrs["Time"][0])

    mu = GRAVITATIONAL_CONSTANT * mars_mass_kg
    radius_m = np.linalg.norm(pos_m, axis=1)
    speed_sq = np.sum(vel_m_s * vel_m_s, axis=1)
    specific_energy = 0.5 * speed_sq - mu / radius_m
    bound_mask = np.isfinite(specific_energy) & (specific_energy < 0.0)

    angular_momentum = np.cross(pos_m, vel_m_s)
    h_sq = np.sum(angular_momentum * angular_momentum, axis=1)
    eccentricity_sq = 1.0 + (2.0 * specific_energy * h_sq) / (mu * mu)
    eccentricity = np.sqrt(np.maximum(eccentricity_sq, 0.0))

    semi_major_axis_m = np.full(len(particle_ids), np.nan)
    valid_bound = bound_mask & np.isfinite(specific_energy) & (np.abs(specific_energy) > 0.0)
    semi_major_axis_m[valid_bound] = -mu / (2.0 * specific_energy[valid_bound])
    apoapsis_m = np.full(len(particle_ids), np.nan)
    apoapsis_m[valid_bound] = semi_major_axis_m[valid_bound] * (1.0 + eccentricity[valid_bound])
    captured_mask = bound_mask & np.isfinite(apoapsis_m) & (apoapsis_m < mars_hill_radius_m)

    total_mass_kg = float(np.sum(masses_kg))
    bound_mass_kg = float(np.sum(masses_kg[bound_mask]))
    captured_mass_kg = float(np.sum(masses_kg[captured_mask]))
    unresolved_or_background_mass_kg = 0.0

    return {
        "target_mass_kg": total_mass_kg,
        "resolved_fof_mass_kg": "",
        "background_unresolved_mass_kg": unresolved_or_background_mass_kg,
        "bound_mass_kg": bound_mass_kg,
        "captured_mass_kg": captured_mass_kg,
        "f_bnd_parent": bound_mass_kg / total_mass_kg if total_mass_kg else math.nan,
        "f_capt_parent": captured_mass_kg / total_mass_kg if total_mass_kg else math.nan,
        "particle_count": int(len(particle_ids)),
        "bound_particle_count": int(np.count_nonzero(bound_mask)),
        "captured_particle_count": int(np.count_nonzero(captured_mask)),
        "mars_mass_kg": mars_mass_kg,
        "mars_hill_radius_m": mars_hill_radius_m,
        "time_code_units": time_code,
        "method": "particle_level_parent_mass_from_raw_snapshot",
        "fof_regenerated": False,
        "unresolved_handling": "No FoF background discarded; all raw particles contribute to denominator and orbit tests.",
    }


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = args.manifest if args.manifest.is_absolute() else repo_root / args.manifest
    output_path = args.output if args.output.is_absolute() else repo_root / args.output
    diagnostics_dir = args.diagnostics_dir if args.diagnostics_dir.is_absolute() else repo_root / args.diagnostics_dir
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows = parse_manifest(manifest_path)
    rows_to_process = [row for row in manifest_rows if row["raw_hdf5_found"].strip().lower() == "true"]
    if args.validation_only:
        rows_to_process = [row for row in rows_to_process if row["physical_file"] == VALIDATION_FILE]

    results = []  # type: List[Dict[str, object]]
    for row in rows_to_process:
        file_path = Path(row["raw_hdf5_path"])
        meta = parse_filename_metadata(row["physical_file"])
        metrics = compute_corrected_metrics(
            file_path,
            mars_mass_kg_default=args.mars_mass_kg,
            mars_radius_m=args.mars_radius_m,
            mars_hill_radius_m=args.mars_hill_radius_m,
        )
        results.append(
            {
                **meta,
                "raw_hdf5_path": str(file_path),
                **metrics,
            }
        )

    fields = [
        "physical_file",
        "raw_hdf5_path",
        "mass_code",
        "resolution_code",
        "periapsis_code",
        "velocity_code",
        "spin_code",
        "timestep",
        "mass",
        "periapsis",
        "v_inf",
        "spin",
        "resolution",
        "target_mass_kg",
        "resolved_fof_mass_kg",
        "background_unresolved_mass_kg",
        "bound_mass_kg",
        "captured_mass_kg",
        "f_bnd_parent",
        "f_capt_parent",
        "particle_count",
        "bound_particle_count",
        "captured_particle_count",
        "mars_mass_kg",
        "mars_hill_radius_m",
        "time_code_units",
        "method",
        "fof_regenerated",
        "unresolved_handling",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    if results:
        ref = results[0]
        reproduced = abs(ref["f_bnd_parent"] - PAPER_F_BND) <= 0.02 and abs(ref["f_capt_parent"] - PAPER_F_CAPT) <= 0.02
        comparison_path = diagnostics_dir / "reference_case_comparison.csv"
        with comparison_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["scenario", "paper_f_bnd", "our_f_bnd_parent", "difference_f_bnd", "paper_f_capt", "our_f_capt_parent", "difference_f_capt"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "scenario": VALIDATION_FILE,
                    "paper_f_bnd": PAPER_F_BND,
                    "our_f_bnd_parent": ref["f_bnd_parent"],
                    "difference_f_bnd": ref["f_bnd_parent"] - PAPER_F_BND,
                    "paper_f_capt": PAPER_F_CAPT,
                    "our_f_capt_parent": ref["f_capt_parent"],
                    "difference_f_capt": ref["f_capt_parent"] - PAPER_F_CAPT,
                }
            )
        summary_path = diagnostics_dir / "reference_case_summary.txt"
        summary_lines = [
            f"physical_file={ref['physical_file']}",
            f"target_mass_kg={ref['target_mass_kg']}",
            f"resolved_fof_mass_kg={ref['resolved_fof_mass_kg']}",
            f"background_unresolved_mass_kg={ref['background_unresolved_mass_kg']}",
            f"bound_mass_kg={ref['bound_mass_kg']}",
            f"captured_mass_kg={ref['captured_mass_kg']}",
            f"f_bnd_parent={ref['f_bnd_parent']}",
            f"paper_f_bnd={PAPER_F_BND}",
            f"f_capt_parent={ref['f_capt_parent']}",
            f"paper_f_capt={PAPER_F_CAPT}",
            "fof_regenerated=False",
            "unresolved_handling=No FoF background discarded; all raw particles contribute to denominator and orbit tests.",
            "assumptions=Particle-level parent-mass accounting from raw snapshots; no regenerated FoF catalogue available in this corrected HPC-only pass.",
            "reference_case_reproduced={}".format("YES" if reproduced else "NO"),
        ]
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        diagnostic_summary_path = diagnostics_dir / "corrected_bmf_diagnostic_summary.txt"
        diagnostic_summary_lines = [
            "simulations_processed={}".format(len(results)),
            "simulations_failed=0",
            "reference_case_reproduced={}".format("YES" if reproduced else "NO"),
            "full_corrected_extraction_completed=NO",
            "stop_reason=Reference-case discrepancy remains material and FoF was not regenerated from the raw-only archive.",
            "unresolved_background_handling=No FoF background discarded; all raw particles contribute to denominator and orbit tests.",
            "fof_regenerated=NO",
            "assumptions=Particle-level parent-mass accounting from raw snapshots using Mars-centered orbital energy and Hill-radius capture criterion.",
        ]
        diagnostic_summary_path.write_text("\n".join(diagnostic_summary_lines) + "\n", encoding="utf-8")

        print(f"Reference case: {ref['physical_file']}")
        print(f"Our f_bnd: {ref['f_bnd_parent']:.6f}")
        print(f"Paper f_bnd: {PAPER_F_BND:.6f}")
        print(f"Our f_capt: {ref['f_capt_parent']:.6f}")
        print(f"Paper f_capt: {PAPER_F_CAPT:.6f}")
        print(f"Reference reproduced: {'YES' if reproduced else 'NO'}")
        print(f"Output: {output_path}")
        print(f"Diagnostics: {comparison_path}")
    else:
        print("No rows were processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
