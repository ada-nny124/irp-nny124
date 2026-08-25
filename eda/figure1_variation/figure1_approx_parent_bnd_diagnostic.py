from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
BOUND_SOURCE = ROOT / "extraction-outputs" / "tables" / "bound_outcomes.csv"
FOF_SOURCE = ROOT / "extraction-outputs" / "tables" / "fof_outcomes.csv"

DIAGNOSTIC_CSV = SCRIPT_DIR / "figure1_approx_parent_bnd_diagnostic.csv"
SUBSET_PLOT = SCRIPT_DIR / "figure1_approx_parent_bnd_paper_reference_subset.png"
COMPARISON_CSV = SCRIPT_DIR / "figure1_paper_reference_table2_approx_parent_bnd_comparison.csv"
SUMMARY_MD = SCRIPT_DIR / "approx_parent_bnd_diagnostic_summary.md"

PAPER_TABLE2_ROWS = [
    {"periapsis_Rm": 1.1, "v_inf_kms": 0.0, "paper_f_bnd": 0.526},
    {"periapsis_Rm": 1.2, "v_inf_kms": 0.0, "paper_f_bnd": 0.516},
    {"periapsis_Rm": 1.3, "v_inf_kms": 0.0, "paper_f_bnd": 0.516},
    {"periapsis_Rm": 1.4, "v_inf_kms": 0.0, "paper_f_bnd": 0.513},
    {"periapsis_Rm": 1.5, "v_inf_kms": 0.0, "paper_f_bnd": 0.434},
    {"periapsis_Rm": 1.6, "v_inf_kms": 0.0, "paper_f_bnd": 0.466},
    {"periapsis_Rm": 1.7, "v_inf_kms": 0.0, "paper_f_bnd": 0.388},
    {"periapsis_Rm": 1.8, "v_inf_kms": 0.0, "paper_f_bnd": 0.238},
    {"periapsis_Rm": 1.9, "v_inf_kms": 0.0, "paper_f_bnd": 0.181},
    {"periapsis_Rm": 2.0, "v_inf_kms": 0.0, "paper_f_bnd": 0.000},
    {"periapsis_Rm": 2.2, "v_inf_kms": 0.0, "paper_f_bnd": 0.000},
    {"periapsis_Rm": 2.4, "v_inf_kms": 0.0, "paper_f_bnd": 0.000},
    {"periapsis_Rm": 1.2, "v_inf_kms": 0.2, "paper_f_bnd": 0.493},
    {"periapsis_Rm": 1.4, "v_inf_kms": 0.2, "paper_f_bnd": 0.452},
    {"periapsis_Rm": 1.6, "v_inf_kms": 0.2, "paper_f_bnd": 0.392},
    {"periapsis_Rm": 1.8, "v_inf_kms": 0.2, "paper_f_bnd": 0.237},
    {"periapsis_Rm": 2.0, "v_inf_kms": 0.2, "paper_f_bnd": 0.121},
    {"periapsis_Rm": 1.2, "v_inf_kms": 0.4, "paper_f_bnd": 0.426},
    {"periapsis_Rm": 1.4, "v_inf_kms": 0.4, "paper_f_bnd": 0.380},
    {"periapsis_Rm": 1.6, "v_inf_kms": 0.4, "paper_f_bnd": 0.334},
    {"periapsis_Rm": 1.8, "v_inf_kms": 0.4, "paper_f_bnd": 0.183},
    {"periapsis_Rm": 2.0, "v_inf_kms": 0.4, "paper_f_bnd": 0.000},
    {"periapsis_Rm": 1.2, "v_inf_kms": 0.6, "paper_f_bnd": 0.395},
    {"periapsis_Rm": 1.4, "v_inf_kms": 0.6, "paper_f_bnd": 0.321},
    {"periapsis_Rm": 1.6, "v_inf_kms": 0.6, "paper_f_bnd": 0.253},
    {"periapsis_Rm": 1.8, "v_inf_kms": 0.6, "paper_f_bnd": 0.112},
    {"periapsis_Rm": 2.0, "v_inf_kms": 0.6, "paper_f_bnd": 0.000},
]

VELOCITY_COLORS = {
    0.0: "#1f77b4",
    0.2: "#2ca02c",
    0.4: "#ffbf00",
    0.6: "#ff7f0e",
    0.8: "#ff4d4d",
    1.0: "#c266ff",
    1.2: "#7f7f7f",
    1.4: "#bcbd22",
    1.6: "#17becf",
}


def parse_numeric_code(series: pd.Series, pattern: str, scale: float = 1.0) -> pd.Series:
    extracted = series.fillna("").astype(str).str.extract(pattern)[0]
    return pd.to_numeric(extracted, errors="coerce") / scale


def load_bound_frame() -> pd.DataFrame:
    frame = pd.read_csv(BOUND_SOURCE, low_memory=False)
    frame["periapsis_Rm"] = parse_numeric_code(frame["periapsis_code"], r"r(\d+)", 10.0)
    frame["v_inf_kms"] = parse_numeric_code(frame["velocity_code"], r"v(\d+)", 10.0)
    frame["mass_log10_kg"] = parse_numeric_code(frame["mass_code"], r"A(\d+)", 100.0)
    frame["target_mass_kg"] = np.power(10.0, frame["mass_log10_kg"])
    frame["asteroid_mass_kg"] = frame["target_mass_kg"]

    spin_code = frame["spin_code"].fillna("").astype(str)
    frame["spin_orientation"] = "no_spin"
    frame.loc[spin_code.str.contains("mz"), "spin_orientation"] = "retrograde_z"
    frame.loc[spin_code.str.contains("x") | spin_code.str.contains("y"), "spin_orientation"] = "equatorial"
    frame.loc[spin_code.str.contains("z") & ~spin_code.str.contains("mz"), "spin_orientation"] = "prograde_z"

    for column in ["bound_mass_fraction", "bound_mass_kg", "target_mass_kg", "fof_linking_length"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["approx_f_bnd_parent"] = frame["bound_mass_kg"] / frame["target_mass_kg"]
    return frame


def load_fof_frame() -> pd.DataFrame:
    return pd.read_csv(FOF_SOURCE, low_memory=False)


def build_diagnostic_csv(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "physical_file",
        "mass_code",
        "resolution_code",
        "periapsis_code",
        "velocity_code",
        "spin_code",
        "asteroid_mass_kg",
        "periapsis_Rm",
        "v_inf_kms",
        "spin_orientation",
        "fof_linking_length",
        "bound_mass_fraction",
        "bound_mass_kg",
        "target_mass_kg",
        "approx_f_bnd_parent",
    ]
    return frame.loc[:, columns].copy()


def build_paper_reference_subset(frame: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    subset = frame.loc[
        (frame["mass_code"] == "A2000")
        & (frame["resolution_code"] == "n65")
        & (frame["spin_orientation"] == "no_spin")
        & frame["periapsis_Rm"].notna()
        & frame["v_inf_kms"].notna()
        & frame["approx_f_bnd_parent"].notna()
    ].copy()

    duplicate_count = subset.duplicated(subset=["periapsis_Rm", "v_inf_kms"], keep=False).sum()
    applied_fof_filter = False
    if duplicate_count > 0:
        subset = subset.loc[np.isclose(subset["fof_linking_length"], 0.0040)].copy()
        applied_fof_filter = True

    subset = subset.sort_values(["v_inf_kms", "periapsis_Rm"]).copy()
    return subset, applied_fof_filter


def build_comparison_table(subset: pd.DataFrame) -> pd.DataFrame:
    paper = pd.DataFrame(PAPER_TABLE2_ROWS)
    archive = subset[
        [
            "periapsis_Rm",
            "v_inf_kms",
            "approx_f_bnd_parent",
            "bound_mass_fraction",
        ]
    ].copy()
    comparison = paper.merge(archive, on=["periapsis_Rm", "v_inf_kms"], how="left")
    comparison["difference"] = comparison["approx_f_bnd_parent"] - comparison["paper_f_bnd"]
    comparison["current_project_metric"] = comparison["bound_mass_fraction"]
    comparison["scenario"] = comparison.apply(
        lambda row: f"q={row['periapsis_Rm']:.1f} R_Mars, v_inf={row['v_inf_kms']:.1f} km s^-1, mass=10^20 kg, no spin",
        axis=1,
    )
    return comparison[
        [
            "scenario",
            "paper_f_bnd",
            "approx_f_bnd_parent",
            "difference",
            "current_project_metric",
        ]
    ].copy()


def draw_subset_plot(subset: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.3), dpi=220)
    for velocity, velocity_rows in subset.groupby("v_inf_kms", sort=True):
        color = VELOCITY_COLORS.get(float(velocity), "#333333")
        velocity_rows = velocity_rows.sort_values("periapsis_Rm")
        ax.plot(
            velocity_rows["periapsis_Rm"],
            velocity_rows["approx_f_bnd_parent"],
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=4.5,
            label=f"v∞ = {velocity:g} km s$^{{-1}}$",
        )

    ax.set_title("Approximate parent-normalized bound fraction", fontsize=12)
    ax.set_xlabel(r"Periapsis ($R_{\mathrm{Mars}}$)", fontsize=11)
    ax.set_ylabel("Approximate parent-normalized bound fraction", fontsize=11)
    ax.set_xlim(1.1, 2.4)
    ax.set_ylim(bottom=0.0)
    ax.set_xticks([1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4])
    ax.grid(True, color="#e6e6e6", linewidth=0.8, alpha=0.8)
    ax.set_facecolor("white")
    ax.legend(loc="upper right", fontsize=8, frameon=True, title="Saved rows")
    fig.text(
        0.5,
        0.02,
        "Approximate comparison to paper reference. Saved resolved-FoF bound mass only; unresolved/background material is excluded.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    fig.savefig(SUBSET_PLOT, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    bound_frame: pd.DataFrame,
    fof_frame: pd.DataFrame,
    subset: pd.DataFrame,
    comparison: pd.DataFrame,
    applied_fof_filter: bool,
) -> None:
    matched = comparison.dropna(subset=["approx_f_bnd_parent"]).copy()
    diff = matched["difference"].dropna()
    current_minus_approx = (
        matched["current_project_metric"] - matched["approx_f_bnd_parent"]
    ).dropna()
    capture_columns_present = {
        "captured_mass_fraction",
        "mean_bound_fragment_apoapsis_Rm",
        "bound_metrics_available",
    }.issubset(fof_frame.columns)

    lines = [
        "# Approximate parent-normalized bound-fraction diagnostic",
        "",
        "This diagnostic leaves the project target unchanged and adds a separate approximate paper-comparison path.",
        "",
        "## Definition",
        "",
        "`approx_f_bnd_parent = bound_mass_kg / target_mass_kg`",
        "",
        "## Why this is closer to the paper's `f_bnd`",
        "",
        "- It uses the original asteroid mass in the denominator rather than resolved fragment mass.",
        "- That makes it closer in normalization to the paper's bound fraction than the current project metric `bound_mass_fraction`.",
        "",
        "## Why it is still not identical",
        "",
        "- The numerator still comes from saved resolved-FoF bound mass only.",
        "- Unresolved/background material cannot be recovered from the saved extraction outputs.",
        "- This is not equivalent to paper `f_capt`.",
        "- Exact `f_capt` cannot be reconstructed from saved outputs without raw particle/orbit HDF5 data.",
        "",
        "## Paper-reference subset used here",
        "",
        "- Mass: `10^20 kg` (`A2000`)",
        "- Spin: no spin",
        "- Resolution: `n65`",
        f"- FoF filter applied for de-duplication: {'yes, 0.004 only' if applied_fof_filter else 'no, exact saved rows were already unique'}",
        f"- Saved comparison rows in subset: {len(subset)}",
        "",
        "## Numerical comparison against paper-reference rows",
        "",
        f"- Matched scenarios: {len(matched)}",
        f"- Mean absolute difference vs paper `f_bnd`: {diff.abs().mean():.4f}" if not diff.empty else "- Mean absolute difference vs paper `f_bnd`: n/a",
        f"- Max absolute difference vs paper `f_bnd`: {diff.abs().max():.4f}" if not diff.empty else "- Max absolute difference vs paper `f_bnd`: n/a",
        f"- Mean signed difference (`approx_f_bnd_parent - paper_f_bnd`): {diff.mean():.4f}" if not diff.empty else "- Mean signed difference (`approx_f_bnd_parent - paper_f_bnd`): n/a",
        (
            f"- Range of (`bound_mass_fraction - approx_f_bnd_parent`) on matched rows: "
            f"{current_minus_approx.min():.4f} to {current_minus_approx.max():.4f}"
        )
        if not current_minus_approx.empty
        else "- Range of (`bound_mass_fraction - approx_f_bnd_parent`) on matched rows: n/a",
        "",
        "## Source checks",
        "",
        f"- Diagnostic CSV rows: {len(bound_frame)}",
        f"- `approx_f_bnd_parent` non-null rows: {int(bound_frame['approx_f_bnd_parent'].notna().sum())}",
        f"- Saved `fof_outcomes.csv` has capture-specific columns needed for exact `f_capt` reconstruction: {'yes' if capture_columns_present else 'no'}",
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    bound_frame = load_bound_frame()
    fof_frame = load_fof_frame()

    diagnostic = build_diagnostic_csv(bound_frame)
    diagnostic.to_csv(DIAGNOSTIC_CSV, index=False)

    subset, applied_fof_filter = build_paper_reference_subset(bound_frame)
    draw_subset_plot(subset)

    comparison = build_comparison_table(subset)
    comparison.to_csv(COMPARISON_CSV, index=False)

    write_summary(bound_frame, fof_frame, subset, comparison, applied_fof_filter)


if __name__ == "__main__":
    main()
