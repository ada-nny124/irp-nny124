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
