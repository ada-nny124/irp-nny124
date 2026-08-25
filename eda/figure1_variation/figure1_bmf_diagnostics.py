from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
BOUND_SOURCE = ROOT / "extraction-outputs" / "tables" / "bound_outcomes.csv"
FOF_SOURCE = ROOT / "extraction-outputs" / "tables" / "fof_outcomes.csv"

CURRENT_GROUPED_PLOT = SCRIPT_DIR / "figure1_diag_current_metric_grouped.png"
PARENT_GROUPED_PLOT = SCRIPT_DIR / "figure1_diag_f_bnd_parent_grouped.png"
FIDUCIAL_COMPARE_PLOT = SCRIPT_DIR / "figure1_diag_fiducial_nospin_compare.png"
PAPER_COMPARE_PLOT = SCRIPT_DIR / "figure1_diag_paper_reference_table2_compare.png"
NORMALIZATION_REPORT = SCRIPT_DIR / "figure1_bmf_normalization_audit.txt"
TABLE_COMPARE_CSV = SCRIPT_DIR / "figure1_paper_reference_table2_comparison.csv"

PERI_RANGE = (1.1, 3.0)
PERI_TICKS = [1.1, 1.3, 1.5, 1.7, 1.9, 2.2, 2.6, 3.0]
SPIN_MARKERS = {
    "no_spin": "o",
    "equatorial": "s",
    "prograde_z": "^",
    "retrograde_z": "D",
}
SPIN_LINESTYLES = {
    "no_spin": "solid",
    "equatorial": (0, (6, 2)),
    "prograde_z": (0, (2, 2)),
    "retrograde_z": (0, (9, 3, 2, 3)),
}
SPIN_LABELS = {
    "no_spin": "No spin",
    "equatorial": "Equatorial spin",
    "prograde_z": "Prograde z-spin",
    "retrograde_z": "Retrograde z-spin",
}
VELOCITY_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
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

# Table 2 values extracted from the paper text layer for the fiducial 10^20 kg, no-spin subset.
PAPER_TABLE2_ROWS = [
    {"periapsis_Rm": 1.1, "v_inf_kms": 0.0, "paper_f_bnd": 0.526, "paper_f_capt": 0.479},
    {"periapsis_Rm": 1.2, "v_inf_kms": 0.0, "paper_f_bnd": 0.516, "paper_f_capt": 0.444},
    {"periapsis_Rm": 1.3, "v_inf_kms": 0.0, "paper_f_bnd": 0.516, "paper_f_capt": 0.424},
    {"periapsis_Rm": 1.4, "v_inf_kms": 0.0, "paper_f_bnd": 0.513, "paper_f_capt": 0.442},
    {"periapsis_Rm": 1.5, "v_inf_kms": 0.0, "paper_f_bnd": 0.434, "paper_f_capt": 0.428},
    {"periapsis_Rm": 1.6, "v_inf_kms": 0.0, "paper_f_bnd": 0.466, "paper_f_capt": 0.368},
    {"periapsis_Rm": 1.7, "v_inf_kms": 0.0, "paper_f_bnd": 0.388, "paper_f_capt": 0.336},
    {"periapsis_Rm": 1.8, "v_inf_kms": 0.0, "paper_f_bnd": 0.238, "paper_f_capt": 0.235},
    {"periapsis_Rm": 1.9, "v_inf_kms": 0.0, "paper_f_bnd": 0.181, "paper_f_capt": 0.181},
    {"periapsis_Rm": 2.0, "v_inf_kms": 0.0, "paper_f_bnd": 0.000, "paper_f_capt": 0.000},
    {"periapsis_Rm": 2.2, "v_inf_kms": 0.0, "paper_f_bnd": 0.000, "paper_f_capt": 0.000},
    {"periapsis_Rm": 2.4, "v_inf_kms": 0.0, "paper_f_bnd": 0.000, "paper_f_capt": 0.000},
    {"periapsis_Rm": 1.2, "v_inf_kms": 0.2, "paper_f_bnd": 0.493, "paper_f_capt": 0.441},
    {"periapsis_Rm": 1.4, "v_inf_kms": 0.2, "paper_f_bnd": 0.452, "paper_f_capt": 0.405},
    {"periapsis_Rm": 1.6, "v_inf_kms": 0.2, "paper_f_bnd": 0.392, "paper_f_capt": 0.352},
    {"periapsis_Rm": 1.8, "v_inf_kms": 0.2, "paper_f_bnd": 0.237, "paper_f_capt": 0.188},
    {"periapsis_Rm": 2.0, "v_inf_kms": 0.2, "paper_f_bnd": 0.121, "paper_f_capt": 0.000},
    {"periapsis_Rm": 1.2, "v_inf_kms": 0.4, "paper_f_bnd": 0.426, "paper_f_capt": 0.344},
    {"periapsis_Rm": 1.4, "v_inf_kms": 0.4, "paper_f_bnd": 0.380, "paper_f_capt": 0.330},
    {"periapsis_Rm": 1.6, "v_inf_kms": 0.4, "paper_f_bnd": 0.334, "paper_f_capt": 0.298},
    {"periapsis_Rm": 1.8, "v_inf_kms": 0.4, "paper_f_bnd": 0.183, "paper_f_capt": 0.168},
    {"periapsis_Rm": 2.0, "v_inf_kms": 0.4, "paper_f_bnd": 0.000, "paper_f_capt": 0.000},
    {"periapsis_Rm": 1.2, "v_inf_kms": 0.6, "paper_f_bnd": 0.395, "paper_f_capt": 0.278},
    {"periapsis_Rm": 1.4, "v_inf_kms": 0.6, "paper_f_bnd": 0.321, "paper_f_capt": 0.243},
    {"periapsis_Rm": 1.6, "v_inf_kms": 0.6, "paper_f_bnd": 0.253, "paper_f_capt": 0.213},
    {"periapsis_Rm": 1.8, "v_inf_kms": 0.6, "paper_f_bnd": 0.112, "paper_f_capt": 0.087},
    {"periapsis_Rm": 2.0, "v_inf_kms": 0.6, "paper_f_bnd": 0.000, "paper_f_capt": 0.000},
]


def parse_numeric_code(series: pd.Series, pattern: str, scale: float = 1.0) -> pd.Series:
    extracted = series.fillna("").astype(str).str.extract(pattern)[0]
    return pd.to_numeric(extracted, errors="coerce") / scale


def load_bound_frame() -> pd.DataFrame:
    frame = pd.read_csv(BOUND_SOURCE, low_memory=False)
    frame["periapsis_Rm"] = parse_numeric_code(frame["periapsis_code"], r"r(\d+)", 10.0)
    frame["v_inf_kms"] = parse_numeric_code(frame["velocity_code"], r"v(\d+)", 10.0)
    frame["mass_log10_kg"] = parse_numeric_code(frame["mass_code"], r"A(\d+)", 100.0)
    frame["target_mass_kg"] = np.power(10.0, frame["mass_log10_kg"])

    spin_code = frame["spin_code"].fillna("").astype(str)
    frame["spin_orientation"] = "no_spin"
    frame.loc[spin_code.str.contains("mz"), "spin_orientation"] = "retrograde_z"
    frame.loc[spin_code.str.contains("x") | spin_code.str.contains("y"), "spin_orientation"] = "equatorial"
    frame.loc[spin_code.str.contains("z") & ~spin_code.str.contains("mz"), "spin_orientation"] = "prograde_z"

    for column in ["bound_mass_fraction", "bound_mass_kg", "total_fragment_mass_kg", "fof_linking_length"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["f_bnd_parent"] = frame["bound_mass_kg"] / frame["target_mass_kg"]
    frame["fragment_to_target_mass_ratio"] = frame["total_fragment_mass_kg"] / frame["target_mass_kg"]
    return frame


def load_fof_frame() -> pd.DataFrame:
    return pd.read_csv(FOF_SOURCE, low_memory=False)


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#e6e6e6", linewidth=0.8, alpha=0.8)
    ax.set_facecolor("white")


def grouped_metric_panel(frame: pd.DataFrame, metric: str, mass_code: str | None = None) -> pd.DataFrame:
    panel = frame.loc[
        frame["periapsis_Rm"].between(*PERI_RANGE, inclusive="both")
        & frame[metric].notna()
        & frame["v_inf_kms"].notna()
    ].copy()
    if mass_code is not None:
        panel = panel.loc[panel["mass_code"] == mass_code].copy()
    return (
        panel.groupby(["periapsis_Rm", "v_inf_kms", "spin_orientation"], as_index=False)
        .agg(metric_median=(metric, "median"), raw_row_count=(metric, "size"))
        .sort_values(["v_inf_kms", "spin_orientation", "periapsis_Rm"])
    )


def add_legends(ax: plt.Axes) -> None:
    velocity_handles = [
        Line2D([0], [0], color=VELOCITY_COLORS[v], linewidth=2.0, label=f"{v:g}")
        for v in VELOCITY_VALUES
        if v in VELOCITY_COLORS
    ]
    spin_handles = [
        Line2D(
            [0],
            [0],
            color="#666666",
            linestyle=SPIN_LINESTYLES[name],
            marker=SPIN_MARKERS[name],
            markerfacecolor="white",
            markeredgecolor="#666666",
            markersize=6,
            linewidth=1.6,
            label=SPIN_LABELS[name],
        )
        for name in ["no_spin", "equatorial", "prograde_z", "retrograde_z"]
    ]
    leg1 = ax.legend(
        handles=velocity_handles,
        loc="upper right",
        frameon=True,
        title="v∞ (km s$^{-1}$)",
        title_fontsize=9,
        fontsize=8,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=spin_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.38),
        frameon=True,
        title="spin",
        title_fontsize=9,
        fontsize=8,
    )


def draw_grouped_panel(ax: plt.Axes, grouped: pd.DataFrame, title: str, ylabel: str) -> None:
    for (velocity, spin_name), subset in grouped.groupby(["v_inf_kms", "spin_orientation"], sort=True):
        color = VELOCITY_COLORS.get(float(velocity), "#333333")
        marker = SPIN_MARKERS.get(spin_name, "o")
        linestyle = SPIN_LINESTYLES.get(spin_name, "solid")
        if len(subset) > 1:
            ax.plot(
                subset["periapsis_Rm"],
                subset["metric_median"],
                color=color,
                linewidth=1.6,
                linestyle=linestyle,
                alpha=0.9,
                zorder=2,
            )
        ax.scatter(
            subset["periapsis_Rm"],
            subset["metric_median"],
            color=[color],
            marker=marker,
            s=30,
            linewidths=0.45,
            edgecolors="white",
            alpha=0.95,
            zorder=3,
        )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel(r"Periapsis ($R_{\mathrm{Mars}}$)", fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*PERI_RANGE)
    ax.set_xticks(PERI_TICKS)
    ax.tick_params(axis="x", labelsize=9, pad=4)
    style_axes(ax)
    add_legends(ax)


def save_grouped_plot(frame: pd.DataFrame, metric: str, output_path: Path, title: str, ylabel: str) -> None:
    grouped = grouped_metric_panel(frame, metric)
    fig, ax = plt.subplots(figsize=(7.0, 5.1), dpi=220)
    draw_grouped_panel(ax, grouped, title, ylabel)
    y_max = max(0.32, float(grouped["metric_median"].max()) * 1.08 if not grouped.empty else 0.32)
    ax.set_ylim(0.0, y_max)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_table2_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    paper = pd.DataFrame(PAPER_TABLE2_ROWS)
    archive = frame.loc[
        (frame["mass_code"] == "A2000")
        & (frame["resolution_code"] == "n65")
        & np.isclose(frame["fof_linking_length"], 0.0040)
        & (frame["spin_orientation"] == "no_spin")
    ].copy()
    archive = archive[
        [
            "periapsis_Rm",
            "v_inf_kms",
            "bound_mass_fraction",
            "f_bnd_parent",
            "bound_mass_kg",
            "total_fragment_mass_kg",
            "target_mass_kg",
        ]
    ]
    merged = paper.merge(archive, on=["periapsis_Rm", "v_inf_kms"], how="left")
    merged["paper_minus_current_metric"] = merged["paper_f_bnd"] - merged["bound_mass_fraction"]
    merged["paper_minus_f_bnd_parent"] = merged["paper_f_bnd"] - merged["f_bnd_parent"]
    merged["scenario"] = merged.apply(
        lambda row: f"q={row['periapsis_Rm']:.1f} R_Mars, v_inf={row['v_inf_kms']:.1f} km s^-1, mass=10^20 kg, no spin",
        axis=1,
    )
    return merged[
        [
            "scenario",
            "periapsis_Rm",
            "v_inf_kms",
            "paper_f_bnd",
            "f_bnd_parent",
            "paper_minus_f_bnd_parent",
            "paper_f_capt",
            "bound_mass_fraction",
            "paper_minus_current_metric",
            "bound_mass_kg",
            "total_fragment_mass_kg",
            "target_mass_kg",
        ]
    ]


def draw_fiducial_compare(frame: pd.DataFrame) -> None:
    subset = frame.loc[
        (frame["mass_code"] == "A2000")
        & (frame["resolution_code"] == "n65")
        & np.isclose(frame["fof_linking_length"], 0.0040)
        & (frame["spin_orientation"] == "no_spin")
        & frame["periapsis_Rm"].between(1.1, 2.4, inclusive="both")
    ].copy()
    subset = subset.sort_values(["v_inf_kms", "periapsis_Rm"])

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), dpi=220, sharex=True, sharey=True)
    panels = [
        ("bound_mass_fraction", "Current metric: bound mass / resolved fragment mass"),
        ("f_bnd_parent", "Parent-normalized diagnostic: bound mass / target mass"),
    ]
    for ax, (metric, title) in zip(axes, panels):
        for velocity, velocity_rows in subset.groupby("v_inf_kms", sort=True):
            color = VELOCITY_COLORS.get(float(velocity), "#333333")
            ax.plot(
                velocity_rows["periapsis_Rm"],
                velocity_rows[metric],
                color=color,
                linewidth=1.8,
                marker="o",
                markersize=4.2,
                label=f"{velocity:g} km s$^{{-1}}$",
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"Periapsis ($R_{\mathrm{Mars}}$)")
        ax.set_xlim(1.1, 2.4)
        ax.set_xticks([1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4])
        ax.set_ylabel("Mass fraction")
        style_axes(ax)
    axes[1].legend(loc="upper right", fontsize=8, frameon=True, title="v∞")
    fig.suptitle(r"Exact archive curves for $10^{20}$ kg, no spin, n65, FoF = 0.004", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIDUCIAL_COMPARE_PLOT, bbox_inches="tight")
    plt.close(fig)


def draw_paper_compare(comparison: pd.DataFrame) -> None:
    plot_rows = comparison.dropna(subset=["f_bnd_parent"]).copy()
    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=220)
    for velocity, subset in plot_rows.groupby("v_inf_kms", sort=True):
        color = VELOCITY_COLORS.get(float(velocity), "#333333")
        subset = subset.sort_values("periapsis_Rm")
        ax.plot(
            subset["periapsis_Rm"],
            subset["paper_f_bnd"],
            color=color,
            linewidth=1.8,
            linestyle="--",
            marker="o",
            markersize=4.0,
            label=f"Paper f_bnd, v∞={velocity:g}",
        )
        ax.plot(
            subset["periapsis_Rm"],
            subset["f_bnd_parent"],
            color=color,
            linewidth=1.8,
            linestyle="-",
            marker="s",
            markersize=3.8,
            label=f"Our f_bnd_parent, v∞={velocity:g}",
        )
    ax.set_title(r"Paper reference Table 2 vs local archive: $10^{20}$ kg, no spin", fontsize=12)
    ax.set_xlabel(r"Periapsis ($R_{\mathrm{Mars}}$)")
    ax.set_ylabel("Bound mass fraction")
    ax.set_xlim(1.1, 2.4)
    ax.set_ylim(0.0, 0.6)
    ax.set_xticks([1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4])
    style_axes(ax)
    ax.legend(loc="upper right", fontsize=7.5, frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(PAPER_COMPARE_PLOT, bbox_inches="tight")
    plt.close(fig)


def write_normalization_report(bound_frame: pd.DataFrame, fof_frame: pd.DataFrame, comparison: pd.DataFrame) -> None:
    ratio = bound_frame["fragment_to_target_mass_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    within_1pct = float((ratio.sub(1.0).abs() <= 0.01).mean()) if not ratio.empty else float("nan")
    within_5pct = float((ratio.sub(1.0).abs() <= 0.05).mean()) if not ratio.empty else float("nan")
    worst_rows = (
        bound_frame.assign(abs_delta=lambda df: (df["fragment_to_target_mass_ratio"] - 1.0).abs())
        .sort_values("abs_delta", ascending=False)
        .head(8)
    )
    fof_has_capture = {"captured_mass_fraction", "mean_bound_fragment_apoapsis_Rm", "bound_metrics_available"}.issubset(
        fof_frame.columns
    )
    current_minus_parent = (comparison["bound_mass_fraction"] - comparison["f_bnd_parent"]).dropna()
    with NORMALIZATION_REPORT.open("w", encoding="utf-8") as handle:
        handle.write("Figure 1 BMF diagnostic audit\n")
        handle.write("============================\n\n")
        handle.write("Current stored training/report metric in bound_outcomes.csv:\n")
        handle.write("bound_mass_fraction = bound_mass_kg / total_fragment_mass_kg\n")
        handle.write("where total_fragment_mass_kg is the mass in resolved FoF fragments only.\n\n")
        handle.write("Parent-normalized diagnostic used here:\n")
        handle.write("f_bnd_parent = bound_mass_kg / target_mass_kg\n\n")
        handle.write("Saved fof_outcomes.csv capture-capable columns present locally: ")
        handle.write("yes\n" if fof_has_capture else "no\n")
        if not fof_has_capture:
            handle.write(
                "The local saved fof_outcomes.csv does not preserve apoapsis/Hill-sphere capture columns, so exact f_capt_parent cannot be reconstructed from saved outputs alone.\n\n"
            )
        handle.write("Resolved-fragment mass vs target mass summary:\n")
        handle.write(ratio.describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]).to_string())
        handle.write("\n")
        handle.write(f"within 1% of target mass: {within_1pct:.3f}\n")
        handle.write(f"within 5% of target mass: {within_5pct:.3f}\n\n")
        if not current_minus_parent.empty:
            handle.write("Current metric minus parent-normalized diagnostic for matched fiducial Table 2 rows:\n")
            handle.write(current_minus_parent.describe().to_string())
            handle.write("\n\n")
        handle.write("Largest fragment-to-target mismatches in saved bound_outcomes.csv:\n")
        handle.write(
            worst_rows[
                [
                    "mass_code",
                    "resolution_code",
                    "periapsis_code",
                    "velocity_code",
                    "spin_code",
                    "fof_linking_length",
                    "total_fragment_mass_kg",
                    "target_mass_kg",
                    "fragment_to_target_mass_ratio",
                    "bound_mass_fraction",
                    "f_bnd_parent",
                ]
            ].to_string(index=False)
        )
        handle.write("\n")


def main() -> None:
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    bound_frame = load_bound_frame()
    fof_frame = load_fof_frame()

    save_grouped_plot(
        bound_frame,
        "bound_mass_fraction",
        CURRENT_GROUPED_PLOT,
        "Grouped median of current bound_mass_fraction",
        "Median bound mass fraction",
    )
    save_grouped_plot(
        bound_frame,
        "f_bnd_parent",
        PARENT_GROUPED_PLOT,
        "Grouped median of parent-normalized bound mass fraction",
        "Median f_bnd_parent",
    )

    comparison = build_table2_comparison(bound_frame)
    comparison.to_csv(TABLE_COMPARE_CSV, index=False)

    draw_fiducial_compare(bound_frame)
    draw_paper_compare(comparison)
    write_normalization_report(bound_frame, fof_frame, comparison)


if __name__ == "__main__":
    main()
