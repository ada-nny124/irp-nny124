#!/usr/bin/env python3

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np


BACKGROUND_GROUP_ID = 2147483647
FALLBACK_GM_MARS_M3_S2 = 4.282837e13
ROOT = Path("/rds/general/user/nny124/ephemeral/martian_moons_data")
OUTPUT_DIR = ROOT / "outputs"
FOF_PATTERN = re.compile(r"^(?P<base>.+?)_fof_(?P<linking>[^_]+)_(?P<fof_step>[^_]+)\.hdf5$")
CODE_PATTERN = re.compile(
    r"^Ma_xp_"
    r"(?P<mass_code>A[^_]+)"
    r"(?:_(?P<spin_code>s[^_]+))?"
    r"_(?P<resolution_code>n[^_]+)"
    r"_(?P<periapsis_code>r[^_]+)"
    r"_(?P<velocity_code>v[^_]+)"
    r"_(?P<timestep>\d+)$"
)


@dataclass
class MatchedPair:
    fof_path: Path
    physical_path: Optional[Path]
    fof_linking_length: str


class ExtractionStop(RuntimeError):
    pass


def decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == (1,):
        return decode_attr(value[0])
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_float_attr(source, key: str) -> Optional[float]:
    if key not in source:
        return None
    value = decode_attr(source[key])
    if isinstance(value, str):
        return float(value.strip())
    if isinstance(value, (float, int)):
        return float(value)
    return None


def list_matched_pairs(root: Path) -> Tuple[List[MatchedPair], int]:
    fof_files = sorted(root.glob("*_fof_*.hdf5"))
    matched_pairs = []  # type: List[MatchedPair]
    for fof_path in fof_files:
        match = FOF_PATTERN.match(fof_path.name)
        if not match:
            matched_pairs.append(MatchedPair(fof_path=fof_path, physical_path=None, fof_linking_length=""))
            continue
        physical_path = root / f"{match.group('base')}.hdf5"
        matched_pairs.append(
            MatchedPair(
                fof_path=fof_path,
                physical_path=physical_path if physical_path.exists() else None,
                fof_linking_length=match.group("linking"),
            )
        )
    return matched_pairs, len(fof_files)


def parse_codes(physical_name: str) -> Dict[str, str]:
    stem = Path(physical_name).stem
    match = CODE_PATTERN.match(stem)
    if not match:
        raise ExtractionStop(f"Could not parse simulation codes from physical file name: {physical_name}")
    values = match.groupdict()
    values["spin_code"] = values["spin_code"] or ""
    return values


def dataset_to_si(dataset) -> np.ndarray:
    factor_cgs = parse_float_attr(dataset.attrs, "Conversion factor to physical CGS (including cosmological corrections)")
    if factor_cgs is None:
        raise ExtractionStop(f"Missing CGS conversion factor for dataset {dataset.name}")
    if dataset.name.endswith("/Coordinates") or dataset.name.endswith("/Velocities"):
        factor_si = factor_cgs * 1e-2
    elif dataset.name.endswith("/Masses"):
        factor_si = factor_cgs * 1e-3
    else:
        raise ExtractionStop(f"Unsupported SI conversion for dataset {dataset.name}")
    return dataset[...].astype(np.float64, copy=False) * factor_si


def nonfof_mass_source(physical_h5: h5py.File, particle_count: int) -> Tuple[np.ndarray, str]:
    if "PartType0/Masses" in physical_h5:
        return dataset_to_si(physical_h5["PartType0/Masses"]), "dataset"

    header = physical_h5.get("Header")
    units = physical_h5.get("Units")
    if header is None or units is None:
        raise ExtractionStop("Missing non-FoF Header/Units metadata needed to infer masses")

    initial_mass_table = header.attrs.get("InitialMassTable")
    if initial_mass_table is None or len(initial_mass_table) == 0:
        raise ExtractionStop("Missing Header/InitialMassTable needed to infer non-FoF masses")

    mass_code = float(initial_mass_table[0])
    if not math.isfinite(mass_code) or mass_code <= 0:
        raise ExtractionStop("Header/InitialMassTable[0] is non-positive; cannot infer non-FoF masses")

    unit_mass_cgs = parse_float_attr(units.attrs, "Unit mass in cgs (U_M)")
    if unit_mass_cgs is None:
        raise ExtractionStop("Missing Units/Unit mass in cgs (U_M) needed to infer non-FoF masses")

    particle_mass_kg = mass_code * unit_mass_cgs * 1e-3
    masses = np.full(particle_count, particle_mass_kg, dtype=np.float64)
    return masses, "header_initial_mass_table"


def gm_mars_from_metadata(physical_h5: h5py.File) -> Tuple[float, str]:
    try:
        cgs_group = physical_h5["PhysicalConstants/CGS"]
        params = physical_h5["Parameters"].attrs
        units = physical_h5["Units"].attrs
    except KeyError:
        return FALLBACK_GM_MARS_M3_S2, "fallback_constant"

    newton_g_cgs = parse_float_attr(cgs_group.attrs, "newton_G")
    point_mass_code = parse_float_attr(params, "PointMassPotential:mass")
    unit_mass_cgs = parse_float_attr(units, "Unit mass in cgs (U_M)")
    if None in (newton_g_cgs, point_mass_code, unit_mass_cgs):
        return FALLBACK_GM_MARS_M3_S2, "fallback_constant"

    gm_cgs = newton_g_cgs * point_mass_code * unit_mass_cgs
    gm_si = gm_cgs * 1e-6
    if not math.isfinite(gm_si) or gm_si <= 0:
        return FALLBACK_GM_MARS_M3_S2, "fallback_constant"
    return gm_si, "metadata_point_mass"


def extract_pair(
    pair: MatchedPair,
    require_success: bool,
) -> Tuple[List[Dict[str, object]], Dict[str, object], Dict[str, object]]:
    log_row = {
        "fof_file": pair.fof_path.name,
        "physical_file": pair.physical_path.name if pair.physical_path else "",
        "matched_physical_file_found": bool(pair.physical_path),
        "readable": False,
        "particle_count_match": False,
        "nonfof_velocity_max_abs": "",
        "nonfof_velocity_nonzero_count": "",
        "status": "",
        "error_message": "",
    }
    if pair.physical_path is None:
        log_row["status"] = "missing_physical_file"
        log_row["error_message"] = "Matching non-FoF physical snapshot was not found"
        if require_success:
            raise ExtractionStop(log_row["error_message"])
        return [], {}, log_row

    try:
        with h5py.File(pair.fof_path, "r") as fof_h5, h5py.File(pair.physical_path, "r") as phys_h5:
            fof_group_ids = fof_h5["PartType0/FOFGroupIDs"][...].astype(np.int64, copy=False)

            if "PartType0/Coordinates" not in phys_h5 or "PartType0/Velocities" not in phys_h5:
                raise ExtractionStop("Matching non-FoF snapshot is missing Coordinates or Velocities")

            coords_m = dataset_to_si(phys_h5["PartType0/Coordinates"])
            vels_m_s = dataset_to_si(phys_h5["PartType0/Velocities"])
            masses_kg, mass_source = nonfof_mass_source(phys_h5, coords_m.shape[0])
            gm_mars_m3_s2, gm_source = gm_mars_from_metadata(phys_h5)

            log_row["readable"] = True
            log_row["particle_count_match"] = len(fof_group_ids) == len(coords_m) == len(vels_m_s) == len(masses_kg)
            if not log_row["particle_count_match"]:
                raise ExtractionStop(
                    "Particle count mismatch between FoF group IDs and non-FoF particle fields"
                )

            velocity_max_abs = float(np.max(np.abs(vels_m_s))) if vels_m_s.size else 0.0
            velocity_nonzero_count = int(np.count_nonzero(np.any(vels_m_s != 0.0, axis=1))) if vels_m_s.size else 0
            log_row["nonfof_velocity_max_abs"] = velocity_max_abs
            log_row["nonfof_velocity_nonzero_count"] = velocity_nonzero_count

            if velocity_max_abs <= 0.0 or velocity_nonzero_count <= 0:
                raise ExtractionStop("Non-FoF velocities are all zero; stopping to avoid FoF zero-velocity misuse")

            valid_mask = fof_group_ids != BACKGROUND_GROUP_ID
            valid_ids = fof_group_ids[valid_mask]
            if valid_ids.size == 0:
                log_row["status"] = "success_no_fragments"
                codes = parse_codes(pair.physical_path.name)
                outcome_row = {
                    "fof_file": pair.fof_path.name,
                    "physical_file": pair.physical_path.name,
                    **codes,
                    "fof_linking_length": pair.fof_linking_length,
                    "n_fragments": 0,
                    "largest_fragment_particle_count": 0,
                    "total_fragment_mass_kg": 0.0,
                    "bound_fragment_count": 0,
                    "bound_mass_kg": 0.0,
                    "bound_mass_fraction": 0.0,
                    "largest_bound_fragment_mass_kg": 0.0,
                    "unbound_fragment_count": 0,
                    "unbound_mass_kg": 0.0,
                    "unbound_mass_fraction": 0.0,
                    "largest_unbound_fragment_mass_kg": 0.0,
                }
                return [], outcome_row, log_row

            valid_coords = coords_m[valid_mask]
            valid_vels = vels_m_s[valid_mask]
            valid_masses = masses_kg[valid_mask]

            unique_ids, inverse = np.unique(valid_ids, return_inverse=True)
            mass_sums = np.bincount(inverse, weights=valid_masses)
            count_sums = np.bincount(inverse)

            x_sum = np.bincount(inverse, weights=valid_masses * valid_coords[:, 0])
            y_sum = np.bincount(inverse, weights=valid_masses * valid_coords[:, 1])
            z_sum = np.bincount(inverse, weights=valid_masses * valid_coords[:, 2])
            vx_sum = np.bincount(inverse, weights=valid_masses * valid_vels[:, 0])
            vy_sum = np.bincount(inverse, weights=valid_masses * valid_vels[:, 1])
            vz_sum = np.bincount(inverse, weights=valid_masses * valid_vels[:, 2])

            com_x = x_sum / mass_sums
            com_y = y_sum / mass_sums
            com_z = z_sum / mass_sums
            com_vx = vx_sum / mass_sums
            com_vy = vy_sum / mass_sums
            com_vz = vz_sum / mass_sums

            com_r = np.sqrt(com_x**2 + com_y**2 + com_z**2)
            com_speed = np.sqrt(com_vx**2 + com_vy**2 + com_vz**2)
            specific_energy = 0.5 * com_speed**2 - gm_mars_m3_s2 / com_r
            is_bound = specific_energy < 0.0

            codes = parse_codes(pair.physical_path.name)
            fragment_rows = []  # type: List[Dict[str, object]]
            for i, group_id in enumerate(unique_ids):
                fragment_rows.append(
                    {
                        "fof_file": pair.fof_path.name,
                        "physical_file": pair.physical_path.name,
                        "group_id": int(group_id),
                        **codes,
                        "fof_linking_length": pair.fof_linking_length,
                        "fragment_particle_count": int(count_sums[i]),
                        "fragment_mass_kg": float(mass_sums[i]),
                        "com_x_m": float(com_x[i]),
                        "com_y_m": float(com_y[i]),
                        "com_z_m": float(com_z[i]),
                        "com_vx_m_s": float(com_vx[i]),
                        "com_vy_m_s": float(com_vy[i]),
                        "com_vz_m_s": float(com_vz[i]),
                        "com_r_m": float(com_r[i]),
                        "com_speed_m_s": float(com_speed[i]),
                        "specific_energy_J_kg": float(specific_energy[i]),
                        "is_bound": bool(is_bound[i]),
                    }
                )

            bound_mask = is_bound
            bound_masses = mass_sums[bound_mask]
            unbound_masses = mass_sums[~bound_mask]
            total_fragment_mass = float(np.sum(mass_sums))
            bound_mass = float(np.sum(bound_masses))
            unbound_mass = float(np.sum(unbound_masses))

            outcome_row = {
                "fof_file": pair.fof_path.name,
                "physical_file": pair.physical_path.name,
                **codes,
                "fof_linking_length": pair.fof_linking_length,
                "n_fragments": int(len(unique_ids)),
                "largest_fragment_particle_count": int(np.max(count_sums)) if count_sums.size else 0,
                "total_fragment_mass_kg": total_fragment_mass,
                "bound_fragment_count": int(np.count_nonzero(bound_mask)),
                "bound_mass_kg": bound_mass,
                "bound_mass_fraction": bound_mass / total_fragment_mass if total_fragment_mass > 0 else 0.0,
                "largest_bound_fragment_mass_kg": float(np.max(bound_masses)) if bound_masses.size else 0.0,
                "unbound_fragment_count": int(np.count_nonzero(~bound_mask)),
                "unbound_mass_kg": unbound_mass,
                "unbound_mass_fraction": unbound_mass / total_fragment_mass if total_fragment_mass > 0 else 0.0,
                "largest_unbound_fragment_mass_kg": float(np.max(unbound_masses)) if unbound_masses.size else 0.0,
            }

            log_row["status"] = f"success_mass_source={mass_source};gm_source={gm_source}"
            return fragment_rows, outcome_row, log_row
    except Exception as exc:
        log_row["status"] = "error"
        log_row["error_message"] = str(exc)
        if require_success:
            raise ExtractionStop(f"{pair.fof_path.name}: {exc}") from exc
        return [], {}, log_row


def write_csv(path: Path, fieldnames: Iterable[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_examples(label: str, rows: List[Dict[str, object]], limit: int = 5) -> None:
    print(label)
    if not rows:
        print("  none")
        return
    for row in rows[:limit]:
        print(f"  {row}")


def run_phase(
    pairs: List[MatchedPair],
    require_success: bool,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    fragment_rows = []  # type: List[Dict[str, object]]
    outcome_rows = []  # type: List[Dict[str, object]]
    log_rows = []  # type: List[Dict[str, object]]
    for pair in pairs:
        pair_fragments, pair_outcome, pair_log = extract_pair(pair, require_success=require_success)
        fragment_rows.extend(pair_fragments)
        if pair_outcome:
            outcome_rows.append(pair_outcome)
        log_rows.append(pair_log)
    return fragment_rows, outcome_rows, log_rows


def main() -> int:
    matched_pairs, total_fof_files = list_matched_pairs(ROOT)
    matched_existing_pairs = [pair for pair in matched_pairs if pair.physical_path is not None]

    if not matched_existing_pairs:
        print("No matched FoF/non-FoF pairs were found; stopping.")
        return 1

    smoke_pairs = matched_existing_pairs[:5]
    smoke_fragments, smoke_outcomes, smoke_logs = run_phase(smoke_pairs, require_success=True)

    smoke_nonzero_pairs = sum(
        1
        for row in smoke_logs
        if isinstance(row["nonfof_velocity_nonzero_count"], int) and row["nonfof_velocity_nonzero_count"] > 0
    )
    print(f"Smoke test matched pairs found: {len(matched_existing_pairs)}")
    print(f"Smoke test pairs with nonzero non-FoF velocities: {smoke_nonzero_pairs} / {len(smoke_pairs)}")
    print_examples("Example fragment rows:", smoke_fragments, limit=5)
    print_examples("Example simulation outcome rows:", smoke_outcomes, limit=5)

    if smoke_nonzero_pairs == 0:
        print("Smoke test found only zero-valued non-FoF velocities; stopping immediately.")
        return 1

    all_fragments, all_outcomes, all_logs = run_phase(matched_pairs, require_success=False)

    fragment_columns = [
        "fof_file",
        "physical_file",
        "group_id",
        "mass_code",
        "resolution_code",
        "periapsis_code",
        "velocity_code",
        "spin_code",
        "timestep",
        "fof_linking_length",
        "fragment_particle_count",
        "fragment_mass_kg",
        "com_x_m",
        "com_y_m",
        "com_z_m",
        "com_vx_m_s",
        "com_vy_m_s",
        "com_vz_m_s",
        "com_r_m",
        "com_speed_m_s",
        "specific_energy_J_kg",
        "is_bound",
    ]
    outcome_columns = [
        "fof_file",
        "physical_file",
        "mass_code",
        "resolution_code",
        "periapsis_code",
        "velocity_code",
        "spin_code",
        "timestep",
        "fof_linking_length",
        "n_fragments",
        "largest_fragment_particle_count",
        "total_fragment_mass_kg",
        "bound_fragment_count",
        "bound_mass_kg",
        "bound_mass_fraction",
        "largest_bound_fragment_mass_kg",
        "unbound_fragment_count",
        "unbound_mass_kg",
        "unbound_mass_fraction",
        "largest_unbound_fragment_mass_kg",
    ]
    log_columns = [
        "fof_file",
        "physical_file",
        "matched_physical_file_found",
        "readable",
        "particle_count_match",
        "nonfof_velocity_max_abs",
        "nonfof_velocity_nonzero_count",
        "status",
        "error_message",
    ]

    fragment_path = OUTPUT_DIR / "fragment_orbital_catalog.csv"
    outcome_path = OUTPUT_DIR / "bound_outcomes.csv"
    log_path = OUTPUT_DIR / "bound_unbound_extraction_log.csv"

    write_csv(fragment_path, fragment_columns, all_fragments)
    write_csv(outcome_path, outcome_columns, all_outcomes)
    write_csv(log_path, log_columns, all_logs)

    success_logs = [row for row in all_logs if str(row["status"]).startswith("success")]
    skipped_logs = [row for row in all_logs if not str(row["status"]).startswith("success")]
    reason_counts: dict[str, int] = {}
    for row in skipped_logs:
        reason = str(row["status"])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    print(f"Total FoF files: {total_fof_files}")
    print(f"Matched FoF/non-FoF pairs: {len(matched_existing_pairs)}")
    print(f"Successfully extracted pairs: {len(success_logs)}")
    if skipped_logs:
        print("Skipped pairs and reasons:")
        for reason, count in sorted(reason_counts.items()):
            print(f"  {reason}: {count}")
    else:
        print("Skipped pairs and reasons: none")

    print(f"Output file paths: {fragment_path}, {outcome_path}, {log_path}")
    ready_for_ml = len(all_outcomes) > 0 and len(success_logs) > 0
    print(f"bound_outcomes.csv ready for ML modelling: {'yes' if ready_for_ml else 'no'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExtractionStop as exc:
        print(f"Stopped: {exc}")
        raise SystemExit(1)
