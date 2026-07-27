#!/usr/bin/env python3
"""Create a clean feature-importance plot using only raw and engineered inputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "ml" / "physics_structured_surrogate" / "tables" / "physics_feature_importance.csv"
OUTPUT_DIR = ROOT / "report" / "figures"
OUTPUT_PNG = OUTPUT_DIR / "physics_feature_importance_inputs_only.png"
OUTPUT_MD = ROOT / "report" / "physics_feature_importance_inputs_only.md"

RAW_FEATURES = {
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
}

ENGINEERED_FEATURES = {
    "encounter_eccentricity_proxy",
    "v_inf_squared",
    "periapsis_inverse",
    "angular_momentum_proxy",
    "spin_frequency_hr_inv",
    "has_spin",
    "particle_mass_proxy",
    "mass_resolution_interaction",
    "largest_fragment_mass_fraction",
}

EXCLUDED_FEATURES = {
    "largest_fragment_mass_fraction",  # outcome-derived
    "fof_linking_length",              # post-processing sensitivity
    "particle_mass_proxy",             # mass-derived proxy duplicates the base mass signal
    "special_case_code",               # not a main physical parameter
    "timestep",                        # constant / unimportant here
    "has_explicit_spin",               # low-information helper flag
    "has_spin",                        # low-information helper flag
}

MIN_IMPORTANCE = 0.01


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    subset = df[df["ablation_name"] == "physics_with_fof"].copy()
    subset = subset[~subset["feature"].isin(EXCLUDED_FEATURES)].copy()
    subset = subset[subset["importance_mean"] >= MIN_IMPORTANCE].copy()

    subset["feature_type"] = subset["feature"].map(
        lambda feature: "engineered" if feature in ENGINEERED_FEATURES else "raw"
    )
    subset = subset.sort_values("importance_mean", ascending=True)

    colors = subset["feature_type"].map({"raw": "#1f4e79", "engineered": "#c44536"})

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.barh(subset["feature"], subset["importance_mean"], color=colors)
    ax.set_title("Promoted BMF model: raw and engineered input feature importances", fontsize=15, fontweight="semibold")
    ax.set_xlabel("Permutation importance (mean decrease in grouped-CV $R^2$)")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.25)

    legend_handles = [
        Patch(facecolor="#1f4e79", label="raw input"),
        Patch(facecolor="#c44536", label="physics-derived engineered input"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=180)
    plt.close(fig)

    note = "\n".join(
        [
            "# Input-Only Feature Importance Plot",
            "",
            f"- Figure: `{OUTPUT_PNG}`",
            "- Source: `physics_with_fof` promoted-model importance table",
            "- Included: raw input features and physics-derived engineered features only",
            "- Excluded:",
            "  - `largest_fragment_mass_fraction` because it is outcome-derived",
            "  - `fof_linking_length` because it is post-processing sensitivity rather than a physical encounter input",
            "  - `particle_mass_proxy` because it is a mass-derived proxy rather than an independent encounter parameter",
            "  - helper or negligible features below the display threshold",
            f"- Display threshold: `importance_mean >= {MIN_IMPORTANCE}`",
        ]
    )
    OUTPUT_MD.write_text(note + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
