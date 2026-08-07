#!/usr/bin/env python3
"""Create a raw-input-only feature-importance plot excluding FoF and resolution."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "ml" / "physics_structured_surrogate" / "tables" / "physics_feature_importance.csv"
OUTPUT_DIR = ROOT / "report" / "figures"
OUTPUT_PNG = OUTPUT_DIR / "physics_feature_importance_inputs_only.png"
OUTPUT_MD = ROOT / "report" / "physics_feature_importance_inputs_only.md"

RAW_FEATURES = {
    "mass_log10_kg",
    "periapsis_Rm",
    "v_inf_kms",
    "spin_period_hr",
    "spin_axis",
}

EXCLUDED_FEATURES = {
    "particle_log10",        # resolution proxy
    "fof_linking_length",    # post-processing sensitivity
    "has_explicit_spin",     # helper flag rather than a physical knob
    "special_case_code",     # archive-specific label rather than a physical knob
    "timestep",              # constant / unimportant here
}

MIN_IMPORTANCE = 0.01
DISPLAY_LABELS = {
    "mass_log10_kg": "Target mass (log10 kg)",
    "periapsis_Rm": "Periapsis ($R_{Mars}$)",
    "v_inf_kms": "Encounter speed ($v_{\\infty}$, km/s)",
    "spin_axis": "Spin axis",
    "spin_period_hr": "Spin period (hr)",
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    subset = df[df["ablation_name"] == "original_without_fof"].copy()
    subset = subset[~subset["feature"].isin(EXCLUDED_FEATURES)].copy()
    subset = subset[subset["feature"].isin(RAW_FEATURES)].copy()
    subset = subset[subset["importance_mean"] >= MIN_IMPORTANCE].copy()
    subset = subset.sort_values("importance_mean", ascending=True)
    subset["display_label"] = subset["feature"].map(DISPLAY_LABELS).fillna(subset["feature"])

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.barh(subset["display_label"], subset["importance_mean"], color="#1f4e79")
    ax.set_title("Raw physical input feature importances", fontsize=15, fontweight="semibold")
    ax.set_xlabel("Permutation importance (mean decrease in grouped-CV $R^2$)")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=180)
    plt.close(fig)

    note = "\n".join(
        [
            "# Input-Only Feature Importance Plot",
            "",
            f"- Figure: `{OUTPUT_PNG}`",
            "- Source: `original_without_fof` raw-input random-forest importance table",
            "- Permutation importance procedure:",
            "  1. Fit the raw-input random-forest model on the full evaluation design matrix.",
            "  2. Measure the baseline evaluation score (`R^2` here) with the true feature columns intact.",
            "  3. For one feature at a time, randomly shuffle that column across rows so its relationship to the target is broken while every other column is left unchanged.",
            "  4. Recompute the score after each shuffle and record the score drop relative to the baseline.",
            "  5. Repeat the shuffle several times and report the mean score decrease as the plotted importance.",
            "- Interpretation: a larger positive bar means the model loses more predictive skill when that feature is destroyed, so the fitted model was relying on it more heavily on this dataset.",
            "- Included: raw physical input features only",
            "- Excluded:",
            "  - `fof_linking_length` because it is post-processing sensitivity rather than a physical encounter input",
            "  - `particle_log10` because it is the resolution proxy",
            "  - helper flags and archive-specific labels that are not direct physical encounter inputs",
            "  - negligible features below the display threshold",
            f"- Display threshold: `importance_mean >= {MIN_IMPORTANCE}`",
        ]
    )
    OUTPUT_MD.write_text(note + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
