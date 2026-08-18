"""Run local EDA from a filename manifest."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "irp_matplotlib_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib.pyplot as plt
import pandas as pd

SUMMARY_SPECS = {
    "mass": "A_code",
    "resolution": "n_code",
    "periapsis": "r_code",
    "velocity": "v_code",
    "spin": "spin_label",
    "timestep": "timestep",
    "fof_linking": "fof_linking",
}


def require_manifest(path: Path) -> pd.DataFrame:
    """Load the manifest CSV or raise a clear error."""
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Manifest is empty: {path}")
    return df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add display-friendly columns for EDA."""
    df = df.copy()
    special = df["special_case"].fillna("")
    spin = df["spin_raw"].fillna("")
    df["spin_label"] = "no_spin"
    df.loc[spin != "", "spin_label"] = spin[spin != ""]
    df.loc[special != "", "spin_label"] = "special_" + special[special != ""]
    return df


def save_count_summary(df: pd.DataFrame, column: str, label: str, outputs_dir: Path) -> pd.DataFrame:
    """Save counts for one manifest column."""
    summary = (
        df[column]
        .fillna("missing")
        .value_counts(dropna=False)
        .rename_axis(column)
        .reset_index(name="count")
        .sort_values(by=[column], kind="stable")
    )
    output_path = outputs_dir / f"summary_{label}.csv"
    summary.to_csv(output_path, index=False)
    print(f"\nCounts by {label}:")
    print(summary.to_string(index=False))
    return summary


def plot_bar(summary: pd.DataFrame, x_col: str, title: str, output_path: Path) -> None:
    """Save a simple bar plot."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(summary[x_col].astype(str), summary["count"], color="#4c78a8")
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, row_col: str, col_col: str, title: str, output_path: Path) -> None:
    """Save a heatmap using only pandas and matplotlib."""
    pivot = pd.crosstab(df[row_col].fillna("missing"), df[col_col].fillna("missing"))
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(pivot.values, aspect="auto", cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel(col_col)
    ax.set_ylabel(row_col)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(value) for value in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(value) for value in pivot.index])
    fig.colorbar(image, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/manifest.csv"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--plots-dir", type=Path, default=Path("plots"))
    return parser.parse_args()


def main() -> int:
    """Run manifest-based EDA and save summaries and plots."""
    args = parse_args()
    try:
        df = prepare_dataframe(require_manifest(args.manifest))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, pd.DataFrame] = {}
    for label, column in SUMMARY_SPECS.items():
        summaries[label] = save_count_summary(df, column, label, args.outputs_dir)

    plot_bar(summaries["mass"], "A_code", "Count by Mass Code", args.plots_dir / "count_by_mass.png")
    plot_bar(
        summaries["resolution"], "n_code", "Count by Resolution", args.plots_dir / "count_by_resolution.png"
    )
    plot_bar(
        summaries["periapsis"], "r_code", "Count by Periapsis", args.plots_dir / "count_by_periapsis.png"
    )
    plot_bar(summaries["velocity"], "v_code", "Count by Velocity", args.plots_dir / "count_by_velocity.png")
    plot_bar(summaries["spin"], "spin_label", "Count by Spin", args.plots_dir / "count_by_spin.png")
    plot_bar(
        summaries["timestep"], "timestep", "Count by Timestep", args.plots_dir / "count_by_timestep.png"
    )
    plot_heatmap(
        df,
        "A_code",
        "r_code",
        "Mass Code vs Periapsis",
        args.plots_dir / "heatmap_mass_vs_periapsis.png",
    )
    plot_heatmap(
        df,
        "r_code",
        "v_code",
        "Periapsis vs Velocity",
        args.plots_dir / "heatmap_periapsis_vs_velocity.png",
    )

    print(f"\nSaved summary CSVs to {args.outputs_dir}")
    print(f"Saved plots to {args.plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
