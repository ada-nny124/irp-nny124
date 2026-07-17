#!/usr/bin/env python3
"""Train a physics-structured tabular surrogate for SPH-derived outcomes."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


RANDOM_STATE = 42
N_SPLITS = 5
MARS_MU_KM3_S2 = 4.282837e4
MARS_RADIUS_KM = 3389.5
OUTPUT_ROOT = Path("ml/physics_structured_surrogate")
TABLES_DIR = OUTPUT_ROOT / "tables"
PLOTS_DIR = OUTPUT_ROOT / "plots"
MODELS_DIR = OUTPUT_ROOT / "models"
FOLD_ASSIGNMENTS_PATH = TABLES_DIR / "fold_assignments.csv"
PROMOTED_MODEL_INFO_PATH = TABLES_DIR / "promoted_model_info.json"
PRIMARY_TARGET = "bound_mass_fraction"
SECONDARY_TARGETS = ["n_fragments", "largest_fragment_mass_kg", "largest_fragment_particle_count"]
FILENAME_RE = re.compile(
    r"^(?P<prefix>Ma_xp)_(?P<mass>A\d{4}(?:c30)?)(?:_(?P<spin>s\d{3}[A-Za-z]*))?"
    r"_n(?P<resolution>\d+)_r(?P<periapsis>\d+)_v(?P<velocity>\d+)"
    r"_(?P<timestep>\d+)"
    r"_fof_(?P<linking_length>[0-9.]+)_"
    r"(?P<chunk>\d+)\.hdf5$"
)


@dataclass(frozen=True)
class StagePaths:
    root: Path
    tables: Path
    plots: Path
    models: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("extraction_outputs/bound_outcomes.csv"),
        help="Bound outcome table used as the canonical surrogate dataset.",
    )
    parser.add_argument(
        "--stage",
        choices=[
            "baseline",
            "tuning",
            "fof_compare",
            "feature_ablation",
            "target_transforms",
            "trust",
            "diagnostics",
            "all",
        ],
        default="all",
        help="Pipeline stage to run.",
    )
    return parser.parse_args()


def ensure_output_dirs() -> StagePaths:
    for path in [OUTPUT_ROOT, TABLES_DIR, PLOTS_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    return StagePaths(root=OUTPUT_ROOT, tables=TABLES_DIR, plots=PLOTS_DIR, models=MODELS_DIR)


def build_group_folds(frame: pd.DataFrame, groups: pd.Series, output_path: Path = FOLD_ASSIGNMENTS_PATH) -> pd.DataFrame:
    splitter = GroupKFold(n_splits=min(N_SPLITS, groups.nunique()))
    fold_assignments = np.full(len(frame), -1, dtype=int)
    for fold_index, (_, test_idx) in enumerate(splitter.split(frame, groups=groups)):
        fold_assignments[test_idx] = fold_index
    fold_frame = frame.loc[:, ["physical_file"]].copy()
    fold_frame["row_index"] = frame.index.to_numpy()
    fold_frame["fold_index"] = fold_assignments
    fold_frame.to_csv(output_path, index=False)
    return fold_frame


def write_promoted_model_info(payload: dict[str, object], output_path: Path = PROMOTED_MODEL_INFO_PATH) -> None:
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_simulation_filename(filename: str) -> dict[str, object]:
    match = FILENAME_RE.match(filename)
    if not match:
        raise ValueError(f"Unrecognized FoF filename pattern: {filename}")

    mass_code = match.group("mass")
    spin_code = match.group("spin") or ""
    spin_axis = spin_code[4:] if len(spin_code) > 4 else ""
    spin_value = spin_code[1:4] if spin_code else ""
    resolution_value = int(match.group("resolution"))
    periapsis_value = int(match.group("periapsis"))
    velocity_value = int(match.group("velocity"))
    linking_length = float(match.group("linking_length"))

    return {
        "mass_code": mass_code,
        "mass_value": int(mass_code[1:5]),
        "special_case_code": "c30" if mass_code.endswith("c30") else "",
        "spin_code": spin_code,
        "spin_value": int(spin_value) if spin_value else np.nan,
        "spin_axis": spin_axis or "none",
        "has_explicit_spin": bool(spin_code),
        "resolution_code": f"n{resolution_value}",
        "resolution_value": resolution_value,
        "periapsis_code": f"r{periapsis_value}",
        "periapsis_value": periapsis_value,
        "velocity_code": f"v{velocity_value:02d}",
        "velocity_value": velocity_value,
        "timestep": int(match.group("timestep")),
        "fof_linking_length": linking_length,
        "chunk_index": int(match.group("chunk")),
    }


def build_canonical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    parsed = frame["fof_file"].map(parse_simulation_filename).apply(pd.Series)
    canonical = frame.copy()
    for column in parsed.columns:
        if column not in canonical.columns:
            canonical[column] = parsed[column]

    canonical["mass_log10_kg"] = pd.to_numeric(canonical["mass_value"], errors="coerce") / 100.0
    canonical["target_mass_kg"] = 10 ** canonical["mass_log10_kg"]
    canonical["particle_log10"] = np.log10(pd.to_numeric(canonical["resolution_value"], errors="coerce"))
    canonical["periapsis_Rm"] = pd.to_numeric(canonical["periapsis_value"], errors="coerce") / 10.0
    canonical["v_inf_kms"] = pd.to_numeric(canonical["velocity_value"], errors="coerce") / 10.0
    canonical["spin_period_hr"] = pd.to_numeric(canonical["spin_value"], errors="coerce") / 10.0
    canonical["spin_axis"] = canonical["spin_axis"].fillna("none").replace("", "none")
    canonical["special_case_code"] = canonical["special_case_code"].fillna("").replace("", "none")
    canonical["has_explicit_spin"] = canonical["has_explicit_spin"].fillna(False).astype(bool)
    canonical["has_spin"] = canonical["has_explicit_spin"].astype(int)
    canonical["bound_mass_fraction_ge_0_1"] = pd.to_numeric(canonical["bound_mass_fraction"], errors="coerce") >= 0.1
    largest_bound = pd.to_numeric(canonical["largest_bound_fragment_mass_kg"], errors="coerce")
    largest_unbound = pd.to_numeric(canonical["largest_unbound_fragment_mass_kg"], errors="coerce")
    canonical["largest_fragment_mass_kg"] = np.maximum(largest_bound.fillna(-np.inf), largest_unbound.fillna(-np.inf))
    canonical["largest_fragment_mass_kg"] = canonical["largest_fragment_mass_kg"].replace(-np.inf, np.nan)
    canonical["largest_fragment_mass_fraction"] = canonical["largest_fragment_mass_kg"] / canonical["target_mass_kg"]
    return canonical


def load_canonical_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    return build_canonical_frame(frame)
