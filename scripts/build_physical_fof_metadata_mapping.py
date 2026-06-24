#!/usr/bin/env python3
"""Build a physical-to-FoF metadata mapping and cleaned bound-mass dataset."""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py


PHYSICAL_PATTERN = re.compile(
    r"^(?P<scenario_name>"
    r"Ma_xp_"
    r"(?P<mass_code>A\d{4}(?:c30)?)"
    r"(?:_(?P<spin_code>s[^_]+))?"
    r"_(?P<resolution_code>n[^_]+)"
    r"_(?P<periapsis_code>r[^_]+)"
    r"_(?P<velocity_code>v[^_]+)"
    r"_(?P<timestep>\d+)"
    r")\.hdf5$"
)

BOUND_NUMERIC_COLUMNS = (
    "n_fragments",
    "total_fragment_mass_kg",
    "bound_fragment_count",
    "bound_mass_kg",
    "bound_mass_fraction",
    "largest_bound_fragment_mass_kg",
    "unbound_fragment_count",
    "unbound_mass_kg",
    "unbound_mass_fraction",
    "largest_unbound_fragment_mass_kg",
)

DEDUP_GROUP_COLUMNS = (
    "mass_code",
    "periapsis_code",
    "velocity_code",
    "spin_code",
    "resolution_code",
    "timestep",
    "fof_linking_length",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bound-outcomes", default="outputs/bound_outcomes.csv")
    parser.add_argument(
        "--physical-root",
        default="/rds/general/ephemeral/user/nny124/ephemeral/martian_moons_data",
    )
    parser.add_argument(
        "--mapping-out",
        default="outputs/physical_fof_metadata_mapping.csv",
    )
    parser.add_argument(
        "--cleaned-out",
        default="outputs/plots/kegerreis_figure6_cleaned_dataset.csv",
    )
    return parser.parse_args()


def decode_attr(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        try:
            return str(value.item())
        except Exception:
            pass
    return str(value)


def parse_physical_filename(filename: str) -> Dict[str, str]:
    match = PHYSICAL_PATTERN.match(filename)
    if not match:
        raise ValueError(f"Could not parse physical filename: {filename}")

    values = match.groupdict()
    values["spin_code"] = values.get("spin_code") or ""
    return values


def parse_spin_orientation(spin_code: str) -> str:
    if not spin_code:
        return "no_spin"
    suffix = spin_code[4:] if len(spin_code) > 4 else "z"
    if suffix == "z":
        return "prograde_z"
    if suffix == "mz":
        return "retrograde_z"
    if suffix in {"x", "mx", "y", "my"}:
        return "equatorial"
    return "other"


def parse_spin_period_hr(spin_code: str) -> str:
    if not spin_code:
        return ""
    digits = spin_code[1:4]
    if not digits.isdigit():
        return ""
    return f"{int(digits) / 10.0:.1f}"


def code_to_decimal(code: str, prefix: str) -> str:
    if not code or not code.startswith(prefix):
        return ""
    digits = code[len(prefix) :]
    if not digits.isdigit():
        return ""
    return f"{int(digits) / 10.0:.1f}"


def read_snapshot_metadata(path: Path) -> Dict[str, str]:
    with h5py.File(path, "r") as handle:
        header = handle["Header"]
        part_counts = header.attrs.get("NumPart_ThisFile")
        particle_count = ""
        if part_counts is not None and len(part_counts) > 0:
            particle_count = str(int(part_counts[0]))

        return {
            "physical_snapshot_date": decode_attr(header.attrs.get("SnapshotDate", "")),
            "physical_output_type": decode_attr(header.attrs.get("OutputType", "")),
            "physical_run_name": decode_attr(header.attrs.get("RunName", "")),
            "physical_particle_count": particle_count,
        }


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_numeric(values: List[float]) -> Tuple[float, float, float]:
    mean_value = sum(values) / len(values)
    return mean_value, min(values), max(values)


def main() -> None:
    args = parse_args()
    bound_outcomes_path = Path(args.bound_outcomes)
    physical_root = Path(args.physical_root)
    mapping_out = Path(args.mapping_out)
    cleaned_out = Path(args.cleaned_out)

    source_rows = load_rows(bound_outcomes_path)
    mapping_rows: List[Dict[str, object]] = []
    valid_rows: List[Dict[str, object]] = []

    invalid_removed = 0

    for row in source_rows:
        mapping_row: Dict[str, object] = dict(row)
        physical_file = row["physical_file"]
        physical_path = physical_root / physical_file
        mapping_row["physical_path"] = str(physical_path)
        mapping_row["physical_exists"] = physical_path.exists()
        mapping_row["fof_filename_exists_in_source"] = False
        mapping_row["metadata_status"] = "unmapped"
        mapping_row["metadata_conflicts"] = ""
        mapping_row["spin_orientation"] = ""
        mapping_row["spin_period_hr"] = ""
        mapping_row["periapsis_rm"] = ""
        mapping_row["v_inf_kms"] = ""
        mapping_row["v_inf_source"] = ""
        mapping_row["physical_snapshot_date"] = ""
        mapping_row["physical_output_type"] = ""
        mapping_row["physical_run_name"] = ""
        mapping_row["physical_particle_count"] = ""

        if not physical_path.exists():
            mapping_row["metadata_status"] = "missing_physical_snapshot"
            mapping_rows.append(mapping_row)
            invalid_removed += 1
            continue

        try:
            parsed = parse_physical_filename(physical_file)
            snapshot_metadata = read_snapshot_metadata(physical_path)
        except Exception as exc:
            mapping_row["metadata_status"] = f"invalid_physical_snapshot:{exc}"
            mapping_rows.append(mapping_row)
            invalid_removed += 1
            continue

        conflicts = []
        for column in (
            "mass_code",
            "spin_code",
            "resolution_code",
            "periapsis_code",
            "velocity_code",
            "timestep",
        ):
            parsed_value = parsed.get(column, "")
            current_value = row.get(column, "")
            if str(current_value) != str(parsed_value):
                conflicts.append(f"{column}:{current_value}!={parsed_value}")
            mapping_row[column] = parsed_value

        mapping_row.update(snapshot_metadata)
        mapping_row["scenario_name"] = parsed["scenario_name"]
        mapping_row["spin_orientation"] = parse_spin_orientation(parsed["spin_code"])
        mapping_row["spin_period_hr"] = parse_spin_period_hr(parsed["spin_code"])
        mapping_row["periapsis_rm"] = code_to_decimal(parsed["periapsis_code"], "r")
        mapping_row["v_inf_kms"] = code_to_decimal(parsed["velocity_code"], "v")
        mapping_row["v_inf_source"] = "physical_snapshot_filename"
        mapping_row["metadata_conflicts"] = ";".join(conflicts)
        mapping_row["metadata_status"] = "mapped" if not conflicts else "mapped_with_conflicts"

        mapping_rows.append(mapping_row)
        if conflicts:
            invalid_removed += 1
            continue

        valid_rows.append(mapping_row)

    grouped_rows: Dict[Tuple[str, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in valid_rows:
        key = tuple(str(row[column]) for column in DEDUP_GROUP_COLUMNS)
        grouped_rows[key].append(row)

    cleaned_rows: List[Dict[str, object]] = []
    duplicate_rows_collapsed = 0

    for rows in grouped_rows.values():
        duplicate_rows_collapsed += max(0, len(rows) - 1)

        base = dict(rows[0])
        cleaned_row: Dict[str, object] = {
            column: base.get(column, "")
            for column in (
                "mass_code",
                "spin_code",
                "spin_orientation",
                "spin_period_hr",
                "resolution_code",
                "periapsis_code",
                "periapsis_rm",
                "velocity_code",
                "v_inf_kms",
                "v_inf_source",
                "timestep",
                "fof_linking_length",
                "physical_output_type",
                "physical_snapshot_date",
            )
        }
        cleaned_row["physical_file"] = base.get("physical_file", "")
        cleaned_row["scenario_name"] = base.get("scenario_name", "")
        cleaned_row["fof_file"] = ";".join(sorted({str(item["fof_file"]) for item in rows}))
        cleaned_row["row_count_aggregated"] = len(rows)
        cleaned_row["physical_file_count"] = len({str(item["physical_file"]) for item in rows})

        for column in BOUND_NUMERIC_COLUMNS:
            numeric_values = [float(item[column]) for item in rows]
            mean_value, min_value, max_value = aggregate_numeric(numeric_values)
            cleaned_row[column] = f"{mean_value:.12g}"
            cleaned_row[f"{column}_min"] = f"{min_value:.12g}"
            cleaned_row[f"{column}_max"] = f"{max_value:.12g}"

        cleaned_rows.append(cleaned_row)

    cleaned_rows.sort(
        key=lambda row: (
            row["mass_code"],
            float(row["v_inf_kms"]) if row["v_inf_kms"] else -1.0,
            row["spin_code"],
            float(row["periapsis_rm"]) if row["periapsis_rm"] else -1.0,
            row["resolution_code"],
            row["fof_linking_length"],
        )
    )

    mapping_fieldnames = list(mapping_rows[0].keys()) if mapping_rows else []
    cleaned_fieldnames = list(cleaned_rows[0].keys()) if cleaned_rows else []
    write_csv(mapping_out, mapping_rows, mapping_fieldnames)
    write_csv(cleaned_out, cleaned_rows, cleaned_fieldnames)

    print(f"Mapped rows: {len(valid_rows)}")
    print(f"Deduplicated rows collapsed: {duplicate_rows_collapsed}")
    print(f"Removed invalid rows: {invalid_removed}")
    print(f"Cleaned rows saved: {len(cleaned_rows)}")
    print(f"Mapping CSV: {mapping_out}")
    print(f"Cleaned CSV: {cleaned_out}")


if __name__ == "__main__":
    main()
