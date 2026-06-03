#!/usr/bin/env python3
"""Train grouped baseline models for run-level bound outcome prediction."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


FILENAME_RE = re.compile(
    r"^(?P<prefix>Ma_xp)_(?P<mass>A\d{4}(?:c30)?)(?:_(?P<spin>s\d{3}[A-Za-z]*))?"
    r"_n(?P<resolution>\d+)_r(?P<periapsis>\d+)_v(?P<velocity>\d+)"
    r"_(?P<timestep>\d+)"
    r"_fof_(?P<linking_length>[0-9.]+)_"
    r"(?P<chunk>\d+)\.hdf5$"
)
BASE_FEATURE_COLUMNS = [
    "mass_log10_kg",
    "particle_log10",
    "periapsis_Rm",
    "v_inf_kms",
    "spin_period_hr",
    "spin_axis",
    "has_explicit_spin",
    "special_case_code",
    "timestep",
    "fof_linking_length",
]
FEATURE_SET_COLUMNS = {
    "with_fof_linking_length": BASE_FEATURE_COLUMNS,
    "without_fof_linking_length": [column for column in BASE_FEATURE_COLUMNS if column != "fof_linking_length"],
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    frame: pd.DataFrame
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("outputs/bound_outcomes.csv"),
        help="Bound outcome table with one row per FoF run.",
    )
    parser.add_argument(
        "--ml-dir",
        type=Path,
        default=Path("ml/bound_outcomes"),
        help="Output directory for bound outcome ML artifacts.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


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


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    parsed = frame["fof_file"].map(parse_simulation_filename).apply(pd.Series)
    for column in parsed.columns:
        if column not in frame.columns:
            frame[column] = parsed[column]

    frame["mass_log10_kg"] = pd.to_numeric(frame["mass_value"], errors="coerce") / 100.0
    resolution_values = pd.to_numeric(frame["resolution_value"], errors="coerce")
    frame["particle_log10"] = resolution_values.map(lambda x: np.nan if pd.isna(x) else np.log10(x))
    frame["periapsis_Rm"] = pd.to_numeric(frame["periapsis_value"], errors="coerce") / 10.0
    frame["v_inf_kms"] = pd.to_numeric(frame["velocity_value"], errors="coerce") / 10.0
    frame["spin_period_hr"] = pd.to_numeric(frame["spin_value"], errors="coerce") / 10.0
    frame["has_explicit_spin"] = (
        frame["has_explicit_spin"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
    )
    frame["spin_axis"] = frame["spin_axis"].fillna("none").replace("", "none")
    frame["special_case_code"] = frame["special_case_code"].fillna("").replace("", "none")
    frame["has_any_bound_mass"] = pd.to_numeric(frame["bound_mass_fraction"], errors="coerce") > 0
    return frame


def build_dataset_specs(df: pd.DataFrame) -> list[DatasetSpec]:
    full = DatasetSpec(
        name="all_successful_runs",
        frame=df.copy(),
        description="All successful rows from outputs/bound_outcomes.csv.",
    )
    positive = df[pd.to_numeric(df["bound_mass_fraction"], errors="coerce") > 0].copy()
    positive_spec = DatasetSpec(
        name="positive_bound_runs",
        frame=positive,
        description="Only runs with bound_mass_fraction > 0.",
    )
    return [full, positive_spec]


def write_dataset_summary(dataset_specs: list[DatasetSpec], output_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for spec in dataset_specs:
        frame = spec.frame
        groups = frame["physical_file"].nunique(dropna=True)
        mixed_group_count = 0
        if "has_any_bound_mass" in frame.columns and not frame.empty:
            mixed_group_count = int(frame.groupby("physical_file")["has_any_bound_mass"].nunique().gt(1).sum())
        rows.append(
            {
                "dataset": spec.name,
                "rows": len(frame),
                "unique_physical_files": groups,
                "mean_bound_mass_fraction": pd.to_numeric(frame["bound_mass_fraction"], errors="coerce").mean(),
                "positive_bound_share": frame["has_any_bound_mass"].mean() if "has_any_bound_mass" in frame.columns and len(frame) else pd.NA,
                "mixed_label_physical_files": mixed_group_count,
                "description": spec.description,
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> int:
    args = parse_args()
    ml_dir = args.ml_dir
    tables_dir = ml_dir / "tables"
    ensure_dir(tables_dir)

    df = add_engineered_features(load_dataset(args.dataset))
    dataset_specs = build_dataset_specs(df)
    write_dataset_summary(dataset_specs, tables_dir / "dataset_summaries.csv")

    print(f"Loaded {len(df)} successful bound outcome rows from {args.dataset}")
    print(f"Wrote dataset summary to {tables_dir / 'dataset_summaries.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
