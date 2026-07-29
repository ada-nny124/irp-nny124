#!/usr/bin/env python3
"""Build a slide-ready trustability figure from existing diagnostics outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "report"
SLICE_ROOT = REPORT_ROOT / "slice_diagnostics_20260716"
TABLES_DIR = SLICE_ROOT / "tables"
PLOTS_DIR = SLICE_ROOT / "plots"
OUTPUT_DIR = REPORT_ROOT / "figures"

COVERAGE_TABLE = TABLES_DIR / "coverage_mass_vs_periapsis.csv"
REGRESSION_PREDICTIONS = TABLES_DIR / "regression_oof_predictions.csv"
METRICS_TABLE = TABLES_DIR / "model_metrics_summary.csv"
OUTPUT_PNG = OUTPUT_DIR / "model_trust_parameter_support.png"
OUTPUT_MD = REPORT_ROOT / "trustability_slide_notes.md"

PRIMARY_TARGET = "bound_mass_fraction"
PRIMARY_MODEL = "random_forest"


def load_coverage() -> pd.DataFrame:
    coverage = pd.read_csv(COVERAGE_TABLE)
    coverage = coverage.set_index("mass_log10_kg")
    coverage.columns = [float(column) for column in coverage.columns]
    coverage.index = [float(index) for index in coverage.index]
    return coverage.sort_index().sort_index(axis=1)


def load_error_table() -> pd.DataFrame:
    predictions = pd.read_csv(REGRESSION_PREDICTIONS)
    subset = predictions[(predictions["target"] == PRIMARY_TARGET) & (predictions["model"] == PRIMARY_MODEL)].copy()
    subset["abs_error"] = pd.to_numeric(subset["residual"], errors="coerce").abs()
    table = subset.pivot_table(index="mass_log10_kg", columns="periapsis_Rm", values="abs_error", aggfunc="mean")
    table.columns = [float(column) for column in table.columns]
    table.index = [float(index) for index in table.index]
    return table.sort_index().sort_index(axis=1)


def load_summary_text() -> tuple[str, str]:
    metrics = pd.read_csv(METRICS_TABLE)
    row = metrics[(metrics["task"] == "regression") & (metrics["target"] == PRIMARY_TARGET) & (metrics["model"] == PRIMARY_MODEL)].iloc[0]
    coverage = load_coverage()
    occupied = int((coverage > 0).sum().sum())
    total = int(coverage.size)
    subtitle = (
        f"Grouped held-out trust view for {PRIMARY_TARGET.replace('_', ' ')} "
        f"using {PRIMARY_MODEL.replace('_', ' ')} "
        f"(R²={row['r2']:.3f}, MAE={row['mae']:.4f}, occupied bins={occupied}/{total})"
    )
    footer = (
        "Left: SPH support count by mass and periapsis. "
        "Right: mean out-of-fold absolute error on the same grid. "
        "Dense interior bins support interpolation; sparse or edge bins are weaker."
    )
    return subtitle, footer


def draw_heatmap(
    ax: plt.Axes,
    table: pd.DataFrame,
    title: str,
    cmap: str,
    cbar_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
    *,
    distinguish_zero_and_missing: bool = False,
) -> None:
    values = table.to_numpy(dtype=float)
    cmap_obj = colormaps.get_cmap(cmap).copy()
    image_kwargs: dict[str, object] = {
        "aspect": "auto",
        "origin": "lower",
        "cmap": cmap_obj,
        "vmin": vmin,
        "vmax": vmax,
    }
    if distinguish_zero_and_missing:
        cmap_obj.set_bad("#cfecc7")
        cmap_obj.set_under("#e8f1fb")
        image_kwargs["vmin"] = 0.5 if vmin is None else max(vmin, 0.5)
    values = np.ma.masked_invalid(values)
    image = ax.imshow(values, **image_kwargs)
    ax.set_title(title, fontsize=13, fontweight="semibold")
    ax.set_xlabel("Periapsis ($R_{Mars}$)")
    ax.set_ylabel("Mass log10(kg)")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels([f"{value:.1f}" for value in table.columns], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels([f"{value:.1f}" for value in table.index], fontsize=9)
    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)


def write_notes(subtitle: str, footer: str) -> None:
    text = "\n".join(
        [
            "# Trustability Slide Notes",
            "",
            "## Title",
            "Model Trust Depends on Parameter-Space Support",
            "",
            "## Subtitle",
            "Coverage and held-out error show where predictions are supported by the SPH archive",
            "",
            "## Main interpretation",
            "- The model is most trustworthy where the SPH archive densely covers the parameter space and out-of-fold error stays low.",
            "- It is less trustworthy near sparse or edge regions, even if a prediction can still be produced.",
            "- Trust comes from grouped held-out performance plus local data support, not from curve smoothness alone.",
            "",
            "## Speaker version",
            "> This slide is the real trust argument. The left panel shows where the model actually has SPH support. The right panel shows held-out error in those same regions. Where coverage is dense and error remains low, the model is suitable for screening and interpolation. Where coverage is sparse or near the domain edge, the prediction becomes weaker and should be treated cautiously or deferred to SPH.",
            "",
            "## Asset summary",
            f"- Figure: `{OUTPUT_PNG}`",
            f"- Supporting source figure 1: `{PLOTS_DIR / 'parameter_coverage_heatmaps.png'}`",
            f"- Supporting source figure 2: `{PLOTS_DIR / 'coverage_vs_error_heatmaps.png'}`",
            f"- Plot subtitle used in asset: {subtitle}",
            f"- Footer used in asset: {footer}",
        ]
    )
    OUTPUT_MD.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    coverage = load_coverage()
    error_table = load_error_table().reindex(index=coverage.index, columns=coverage.columns)
    subtitle, footer = load_summary_text()

    fig = plt.figure(figsize=(15, 7.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[12, 1.2], hspace=0.28, wspace=0.20)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    ax_footer = fig.add_subplot(gs[1, :])
    ax_footer.axis("off")

    draw_heatmap(
        ax_left,
        coverage,
        "SPH support: coverage by mass and periapsis",
        "Blues",
        "Runs",
        distinguish_zero_and_missing=True,
    )
    draw_heatmap(ax_right, error_table, "Held-out reliability: mean |error| on same grid", "OrRd", "|actual - predicted|")

    fig.suptitle("Model Trust Depends on Parameter-Space Support", fontsize=20, fontweight="bold", y=0.98)
    fig.text(0.5, 0.935, subtitle, ha="center", va="center", fontsize=11)
    ax_footer.text(0.0, 0.75, footer, fontsize=11, ha="left", va="center")
    ax_footer.text(
        0.0,
        0.20,
        "Grouped validation by physical_file reduces leakage across related simulations.",
        fontsize=10,
        ha="left",
        va="center",
        color="#333333",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)

    write_notes(subtitle, footer)


if __name__ == "__main__":
    main()
