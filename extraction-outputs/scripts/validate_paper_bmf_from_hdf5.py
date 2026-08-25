#!/usr/bin/env python3
"""Validate paper-style bound/captured mass fractions from a raw SPH snapshot."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import h5py
import numpy as np


GRAVITATIONAL_CONSTANT = 6.67430e-11
DEFAULT_MARS_MASS_KG = 6.4171e23
DEFAULT_MARS_RADIUS_M = 3.3895e6
DEFAULT_MARS_HILL_RADIUS_M = 1.08e9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="Raw non-FoF HDF5 snapshot to validate.")
    parser.add_argument("--mars-mass-kg", type=float, default=DEFAULT_MARS_MASS_KG)
    parser.add_argument("--mars-radius-m", type=float, default=DEFAULT_MARS_RADIUS_M)
    parser.add_argument("--mars-hill-radius-m", type=float, default=DEFAULT_MARS_HILL_RADIUS_M)
    return parser.parse_args()


def _attr_scalar(group: h5py.Group, name: str) -> float:
    raw = group.attrs[name]
    if hasattr(raw, "__len__") and not isinstance(raw, (bytes, str)):
        raw = raw[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return float(raw)


def _point_mass_kg(handle: h5py.File, fallback_mars_mass_kg: float) -> float:
    unit_mass_kg = _attr_scalar(handle["Units"], "Unit mass in cgs (U_M)") * 1e-3
    for group_name in ("Parameters", "UnusedParameters"):
        if group_name in handle and "PointMassPotential:mass" in handle[group_name].attrs:
            raw = handle[group_name].attrs["PointMassPotential:mass"]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if hasattr(raw, "__len__") and not isinstance(raw, (bytes, str)):
                raw = raw[0]
            return float(raw) * unit_mass_kg
    return fallback_mars_mass_kg


def _particle_masses_kg(handle: h5py.File, part_group: h5py.Group) -> np.ndarray:
    if "Masses" in part_group:
        factor_cgs = _attr_scalar(
            part_group["Masses"],
            "Conversion factor to physical CGS (including cosmological corrections)",
        )
        return part_group["Masses"][...].astype(np.float64, copy=False) * factor_cgs * 1e-3

    initial_mass_table = handle["Header"].attrs["InitialMassTable"]
    initial_mass_code = float(initial_mass_table[0])
    unit_mass_kg = _attr_scalar(handle["Units"], "Unit mass in cgs (U_M)") * 1e-3
    particle_mass_kg = initial_mass_code * unit_mass_kg
    return np.full(len(part_group["ParticleIDs"]), particle_mass_kg, dtype=np.float64)


def _positions_m(handle: h5py.File, part_group: h5py.Group) -> np.ndarray:
    factor_cgs = _attr_scalar(
        part_group["Coordinates"],
        "Conversion factor to physical CGS (including cosmological corrections)",
    )
    coords_m = part_group["Coordinates"][...].astype(np.float64, copy=False) * factor_cgs * 1e-2
    box_size_code = handle["Header"].attrs["BoxSize"]
    box_size_code = float(box_size_code[0] if hasattr(box_size_code, "__len__") else box_size_code)
    unit_length_m = _attr_scalar(handle["Units"], "Unit length in cgs (U_L)") * 1e-2
    return coords_m - 0.5 * box_size_code * unit_length_m


def _velocities_m_s(part_group: h5py.Group) -> np.ndarray:
    factor_cgs = _attr_scalar(
        part_group["Velocities"],
        "Conversion factor to physical CGS (including cosmological corrections)",
    )
    return part_group["Velocities"][...].astype(np.float64, copy=False) * factor_cgs * 1e-2


def main() -> int:
    args = parse_args()
    with h5py.File(args.snapshot, "r") as handle:
        part_group = handle["PartType0"] if "PartType0" in handle else handle["GasParticles"]
        particle_ids = part_group["ParticleIDs"][...].astype(np.uint64, copy=False)
        positions_m = _positions_m(handle, part_group)
        velocities_m_s = _velocities_m_s(part_group)
        masses_kg = _particle_masses_kg(handle, part_group)
        mars_mass_kg = _point_mass_kg(handle, args.mars_mass_kg)
        time_code = handle["Header"].attrs["Time"]
        time_code = float(time_code[0] if hasattr(time_code, "__len__") else time_code)

    radius_m = np.linalg.norm(positions_m, axis=1)
    speed_sq_m2_s2 = np.sum(velocities_m_s * velocities_m_s, axis=1)
    mu_m3_s2 = GRAVITATIONAL_CONSTANT * mars_mass_kg
    specific_energy_j_kg = 0.5 * speed_sq_m2_s2 - mu_m3_s2 / radius_m
    bound_mask = np.isfinite(specific_energy_j_kg) & (specific_energy_j_kg < 0.0)

    angular_momentum = np.cross(positions_m, velocities_m_s)
    h_sq = np.sum(angular_momentum * angular_momentum, axis=1)
    eccentricity_sq = 1.0 + (2.0 * specific_energy_j_kg * h_sq) / (mu_m3_s2 * mu_m3_s2)
    eccentricity = np.sqrt(np.maximum(eccentricity_sq, 0.0))
    semi_major_axis_m = np.full(len(particle_ids), np.nan)
    valid_bound = bound_mask & np.isfinite(specific_energy_j_kg) & (np.abs(specific_energy_j_kg) > 0.0)
    semi_major_axis_m[valid_bound] = -mu_m3_s2 / (2.0 * specific_energy_j_kg[valid_bound])
    apoapsis_m = np.full(len(particle_ids), np.nan)
    apoapsis_m[valid_bound] = semi_major_axis_m[valid_bound] * (1.0 + eccentricity[valid_bound])
    captured_mask = bound_mask & np.isfinite(apoapsis_m) & (apoapsis_m < args.mars_hill_radius_m)

    total_mass_kg = float(np.sum(masses_kg))
    bound_mass_kg = float(np.sum(masses_kg[bound_mask]))
    captured_mass_kg = float(np.sum(masses_kg[captured_mask]))
    escaping_mass_kg = total_mass_kg - bound_mass_kg
    bound_not_captured_mass_kg = bound_mass_kg - captured_mass_kg

    print(f"snapshot: {args.snapshot}")
    print(f"time_code_units: {time_code}")
    print(f"particle_count: {len(particle_ids)}")
    print(f"mars_mass_kg: {mars_mass_kg:.6e}")
    print(f"mars_radius_m: {args.mars_radius_m:.6e}")
    print(f"mars_hill_radius_m: {args.mars_hill_radius_m:.6e}")
    print(f"total_mass_kg: {total_mass_kg:.6e}")
    print(f"bound_mass_kg: {bound_mass_kg:.6e}")
    print(f"captured_mass_kg: {captured_mass_kg:.6e}")
    print(f"escaping_mass_kg: {escaping_mass_kg:.6e}")
    print(f"bound_not_captured_mass_kg: {bound_not_captured_mass_kg:.6e}")
    print(f"f_bnd: {bound_mass_kg / total_mass_kg:.6f}")
    print(f"f_capt: {captured_mass_kg / total_mass_kg:.6f}")
    print(f"bound_particle_count: {int(np.count_nonzero(bound_mask))}")
    print(f"captured_particle_count: {int(np.count_nonzero(captured_mask))}")

    bound_apoapsis_rm = apoapsis_m[bound_mask] / args.mars_radius_m
    if bound_apoapsis_rm.size:
        quantiles = np.nanquantile(bound_apoapsis_rm, [0.1, 0.5, 0.9, 0.99])
        print(
            "bound_apoapsis_Rm_quantiles:"
            f" q10={quantiles[0]:.3f} q50={quantiles[1]:.3f} q90={quantiles[2]:.3f} q99={quantiles[3]:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
