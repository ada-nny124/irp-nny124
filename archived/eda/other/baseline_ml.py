"""Baseline ML scaffold for manifest-derived features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

TARGET_COLUMNS = [
    "fragment_count",
    "largest_fragment_mass",
    "total_bound_mass",
    "bound_mass_fraction",
    "disk_mass",
    "debris_spread",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build baseline features from manifest-only metadata."""
    feature_columns = [
        "mass_log10_kg",
        "particle_log10",
        "periapsis_Rm",
        "v_inf_kms",
        "timestep",
        "fof_linking",
        "file_index",
        "has_spin",
        "spin_period_hr",
        "model_prefix",
        "A_code",
        "spin_raw",
        "spin_direction",
        "special_case",
        "n_code",
        "r_code",
        "v_code",
    ]
    available = [column for column in feature_columns if column in df.columns]
    features = df[available].copy()
    features["has_spin"] = features.get("has_spin", False).fillna(False).astype(int)
    for column in features.columns:
        if features[column].dtype == object:
            features[column] = features[column].fillna("missing")
    return pd.get_dummies(features, dummy_na=False)


def load_targets(path: Path | None) -> pd.DataFrame | None:
    """Load target data when outcome extraction is available."""
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Target file not found: {path}")
    targets = pd.read_csv(path)
    available = [column for column in TARGET_COLUMNS if column in targets.columns]
    if not available:
        raise ValueError(
            "Target file does not contain any supported physical outcome columns: "
            + ", ".join(TARGET_COLUMNS)
        )
    return targets[available]


def train_baseline_model(X: pd.DataFrame, y: pd.DataFrame) -> None:
    """Placeholder model training hook for future outcome prediction."""
    raise NotImplementedError(
        "Model training is intentionally deferred until real physical outcome targets are extracted. "
        "Planned models: DummyRegressor, Ridge, RandomForestRegressor, GradientBoostingRegressor."
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/manifest.csv"))
    parser.add_argument("--targets", type=Path, help="Optional target table for future modeling.")
    return parser.parse_args()


def main() -> int:
    """Load the manifest and prepare for future baseline modeling."""
    args = parse_args()
    if not args.manifest.is_file():
        print(f"Error: Manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.manifest)
    X = build_features(df)

    try:
        y = load_targets(args.targets)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Prepared manifest-derived feature matrix with shape {X.shape}.")
    if y is None:
        print("No physical outcome target available yet. Run outcome extraction first.")
        return 0

    print(f"Loaded target table with shape {y.shape}.")
    try:
        train_baseline_model(X, y)
    except NotImplementedError as exc:
        print(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
