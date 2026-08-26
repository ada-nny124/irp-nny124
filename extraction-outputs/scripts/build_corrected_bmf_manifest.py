#!/usr/bin/env python3
"""Match current ML physical simulations to raw HDF5 snapshots for corrected BMF extraction."""

import csv
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
BOUND_OUTCOMES_PATH = REPO_ROOT / "extraction-outputs" / "tables" / "bound_outcomes.csv"
RAW_DATA_DIR = REPO_ROOT / "martian_moon_data"
OUTPUT_DIR = REPO_ROOT / "extraction-outputs" / "tables"


def load_expected_physical_rows():
    with BOUND_OUTCOMES_PATH.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    dedup = {}  # type: Dict[str, Dict[str, object]]
    for row in rows:
        physical_file = row["physical_file"]
        if physical_file in dedup:
            continue
        dedup[physical_file] = {
            "physical_file": physical_file,
            "mass_code": row["mass_code"],
            "resolution_code": row["resolution_code"],
            "periapsis_code": row["periapsis_code"],
            "velocity_code": row["velocity_code"],
            "spin_code": row["spin_code"],
            "timestep": row["timestep"],
            "expected_in_current_ml_dataset": True,
        }
    return sorted(dedup.values(), key=lambda item: str(item["physical_file"]))


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    expected_rows = load_expected_physical_rows()
    raw_files = sorted(path for path in RAW_DATA_DIR.glob("*.hdf5"))
    raw_lookup = {path.name: path for path in raw_files}

    matched_rows = []  # type: List[Dict[str, object]]
    missing_rows = []  # type: List[Dict[str, object]]
    for row in expected_rows:
        physical_file = str(row["physical_file"])
        match = raw_lookup.get(physical_file)
        out_row = dict(row)
        out_row["raw_hdf5_found"] = bool(match)
        out_row["raw_hdf5_path"] = str(match.resolve()) if match else ""
        out_row["raw_hdf5_size_bytes"] = match.stat().st_size if match else ""
        if match:
            matched_rows.append(out_row)
        else:
            missing_rows.append(out_row)

    expected_names = {str(row["physical_file"]) for row in expected_rows}
    extra_raw_rows = [
        {
            "physical_file": path.name,
            "raw_hdf5_path": str(path.resolve()),
            "raw_hdf5_size_bytes": path.stat().st_size,
            "expected_in_current_ml_dataset": False,
        }
        for path in raw_files
        if path.name not in expected_names
    ]

    manifest_path = OUTPUT_DIR / "corrected_bmf_raw_snapshot_manifest.csv"
    missing_path = OUTPUT_DIR / "corrected_bmf_missing_expected_raw_snapshots.csv"
    extra_path = OUTPUT_DIR / "corrected_bmf_extra_raw_snapshots_not_in_current_ml_dataset.csv"
    summary_path = OUTPUT_DIR / "corrected_bmf_manifest_summary.txt"

    manifest_fields = [
        "physical_file",
        "mass_code",
        "resolution_code",
        "periapsis_code",
        "velocity_code",
        "spin_code",
        "timestep",
        "expected_in_current_ml_dataset",
        "raw_hdf5_found",
        "raw_hdf5_path",
        "raw_hdf5_size_bytes",
    ]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(matched_rows + missing_rows)

    with missing_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(missing_rows)

    extra_fields = ["physical_file", "raw_hdf5_path", "raw_hdf5_size_bytes", "expected_in_current_ml_dataset"]
    with extra_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=extra_fields)
        writer.writeheader()
        writer.writerows(extra_raw_rows)

    total_raw_size = sum(path.stat().st_size for path in raw_files)
    summary_lines = [
        f"expected_physical_simulations={len(expected_rows)}",
        f"matched_raw_hdf5_simulations={len(matched_rows)}",
        f"missing_expected_raw_hdf5_simulations={len(missing_rows)}",
        f"raw_hdf5_files_found={len(raw_files)}",
        f"extra_raw_hdf5_files_not_in_current_ml_dataset={len(extra_raw_rows)}",
        f"raw_hdf5_total_size_bytes={total_raw_size}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Expected physical simulations: {len(expected_rows)}")
    print(f"Matched raw HDF5 simulations: {len(matched_rows)}")
    print(f"Missing simulations: {len(missing_rows)}")
    print(f"Extra raw snapshots: {len(extra_raw_rows)}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
