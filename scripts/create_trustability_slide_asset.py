#!/usr/bin/env python3
"""Build a slide-ready multi-row trustability figure from current surrogate outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from train_physics_structured_surrogate import add_physics_features, load_canonical_dataset


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "report"
OUTPUT_DIR = REPORT_ROOT / "figures"
OUTPUT_PNG = OUTPUT_DIR / "model_trust_parameter_support.png"
OUTPUT_MD = REPORT_ROOT / "trustability_slide_notes.md"

SURROGATE_ROOT = ROOT / "ml" / "physics_structured_surrogate"
SURROGATE_TABLES = SURROGATE_ROOT / "tables"
DATASET_PATH = ROOT / "extraction_outputs" / "bound_outcomes.csv"
PREDICTIONS_PATH = SURROGATE_TABLES / "predictions_with_trust_flags.csv"

PRIMARY_TARGET = "bound_mass_fraction"
NO_DATA_COLOR = "#BDBDBD"
SUPPORT_BOUNDS = [0, 1, 5, 10, 25, 50, 100, 201]
SUPPORT_TICK_POSITIONS = [0.5, 3, 7.5, 17, 37, 74.5, 150]
SUPPORT_TICK_LABELS = ["0", "1–4", "5–9", "10–24", "25–49", "50–99", "100–200"]

ROW_SPECS = [
    {
        "row_col": "mass_log10_kg",
        "col_col": "periapsis_Rm",
        "row_label": "Mass log10(kg)",
        "col_label": "Periapsis ($R_{Mars}$)",
        "title": "Mass × periapsis",
    },
    {
        "row_col": "asteroid_radius_km",
        "col_col": "v_inf_kms",
        "row_label": "Asteroid radius (km)",
        "col_label": "$v_{\\infty}$ (km/s)",
        "title": "Radius × velocity",
    },
    {
        "row_col": "spin_period_hr",
        "col_col": "periapsis_Rm",
        "row_label": "Spin period (hr)",
        "col_label": "Periapsis ($R_{Mars}$)",
        "title": "Spin × periapsis",
    },
]


def load_frame() -> pd.DataFrame:
    return add_physics_features(load_canonical_dataset(DATASET_PATH))


def load_predictions() -> pd.DataFrame:
    predictions = pd.read_csv(PREDICTIONS_PATH)
    predictions["abs_error"] = pd.to_numeric(predictions["residual"], errors="coerce").abs()
    return predictions


def merged_predictions(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    extra_cols = [
        "fof_file",
        "physical_file",
        "asteroid_radius_km",
        "mass_log10_kg",
        "periapsis_Rm",
        "v_inf_kms",
        "spin_period_hr",
    ]
    keep = frame.loc[:, [column for column in extra_cols if column in frame.columns]].drop_duplicates(subset=["fof_file"])
    merged = predictions.merge(keep, on=["fof_file", "physical_file", "mass_log10_kg", "periapsis_Rm", "v_inf_kms", "spin_period_hr"], how="left")
    return merged


def _format_numeric_label(value: float, decimals: int) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}"
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def bin_labels(values: list[float]) -> list[str]:
    if not values:
        return []

    for decimals in range(0, 5):
        labels = [_format_numeric_label(float(value), decimals) for value in values]
        if len(set(labels)) == len(labels):
            return labels
    return [f"{float(value):.4f}".rstrip("0").rstrip(".") for value in values]


def pairwise_tables(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    row_col: str,
    col_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    support_frame = frame.dropna(subset=[row_col, col_col]).copy()
    error_frame = predictions[predictions["target"] == PRIMARY_TARGET].dropna(subset=[row_col, col_col]).copy()

    row_bins = sorted(pd.to_numeric(support_frame[row_col], errors="coerce").dropna().unique())
    col_bins = sorted(pd.to_numeric(support_frame[col_col], errors="coerce").dropna().unique())

    coverage = support_frame.pivot_table(index=row_col, columns=col_col, values="physical_file", aggfunc="count")
    coverage = coverage.reindex(index=row_bins, columns=col_bins)
    coverage = coverage.fillna(0)

    error = error_frame.pivot_table(index=row_col, columns=col_col, values="abs_error", aggfunc="mean")
    error = error.reindex(index=row_bins, columns=col_bins)

    coverage.index = pd.Index([float(value) for value in coverage.index], name=row_col)
    coverage.columns = pd.Index([float(value) for value in coverage.columns], name=col_col)
    error.index = pd.Index([float(value) for value in error.index], name=row_col)
    error.columns = pd.Index([float(value) for value in error.columns], name=col_col)
    return coverage, error


def finite_min_max(arrays: list[np.ndarray], *, default_min: float = 0.0, default_max: float = 1.0) -> tuple[float, float]:
    finite_values = []
    for arr in arrays:
        values = np.asarray(arr, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            finite_values.append(finite)
    if not finite_values:
        return default_min, default_max
    stacked = np.concatenate(finite_values)
    return float(np.min(stacked)), float(np.max(stacked))


def should_annotate_na(masked_values: np.ma.MaskedArray) -> bool:
    if masked_values.mask is np.ma.nomask:
        return False
    missing_count = int(np.sum(masked_values.mask))
    total_cells = masked_values.shape[0] * masked_values.shape[1]
    return 0 < missing_count <= 8 and total_cells <= 120


def draw_panel(
    ax: plt.Axes,
    table: pd.DataFrame,
    cmap_name: str,
    norm: Normalize,
    title: str,
    x_label: str,
    y_label: str,
    *,
    annotate_na: bool,
    missing_mask: np.ndarray | None = None,
) -> None:
    if cmap_name == "Blues":
        cmap = colormaps["Blues"].copy()
    else:
        cmap = colormaps["Reds"].copy()
    cmap.set_bad(NO_DATA_COLOR)
    values = table.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    if missing_mask is not None:
        mask = np.asarray(missing_mask, dtype=bool)
        if masked.mask is np.ma.nomask:
            masked.mask = mask
        else:
            masked.mask = np.logical_or(masked.mask, mask)
    image = ax.imshow(masked, cmap=cmap, norm=norm, origin="lower", aspect="equal")
    ax.set_title(title, fontsize=11.5, fontweight="semibold", pad=2)
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(bin_labels(list(table.columns)), rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(bin_labels(list(table.index)), fontsize=9.5)
    ax.set_xlabel(x_label, fontsize=10, labelpad=1)
    ax.set_ylabel(y_label, fontsize=10, labelpad=2)
    ax.set_xticks(np.arange(-0.5, len(table.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(table.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#555555")
    if annotate_na and masked.mask is not np.ma.nomask:
        for row_idx in range(masked.shape[0]):
            for col_idx in range(masked.shape[1]):
                if bool(masked.mask[row_idx, col_idx]):
                    ax.text(col_idx, row_idx, "N/A", ha="center", va="center", fontsize=8.5, color="#424242")
    return image


def write_notes() -> None:
    text = "\n".join(
        [
            "# Trustability Slide Notes",
            "",
            "## Title",
            "Model Trust Across Parameter-Space Support",
            "",
            "## Main interpretation",
            "- Blue shows how many SPH samples support each parameter-space cell.",
            "- Red shows grouped held-out mean absolute error on exactly the same cell grid.",
            "- Grey marks cells where no finite estimate is available.",
            "- The surrogate is most trustworthy where support is dense and held-out error remains low.",
            "",
            "## Asset summary",
            f"- Figure: `{OUTPUT_PNG}`",
            "- Caption: Blue shows SPH sample coverage. Red shows grouped held-out mean absolute error. Grey indicates that the corresponding quantity cannot be estimated; it does not represent zero.",
        ]
    )
    OUTPUT_MD.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_frame()
    predictions = merged_predictions(frame, load_predictions())

    pair_data: list[dict[str, object]] = []
    coverage_arrays: list[np.ndarray] = []
    error_arrays: list[np.ndarray] = []
    for spec in ROW_SPECS:
        coverage, error = pairwise_tables(frame, predictions, str(spec["row_col"]), str(spec["col_col"]))
        pair_data.append({"spec": spec, "coverage": coverage, "error": error})
        coverage_arrays.append(coverage.to_numpy(dtype=float))
        error_arrays.append(error.to_numpy(dtype=float))

    error_min, error_max = finite_min_max(error_arrays, default_min=0.0, default_max=1.0)
    coverage_cmap = colormaps["Blues"].copy()
    coverage_cmap.set_bad(NO_DATA_COLOR)
    coverage_norm = BoundaryNorm(SUPPORT_BOUNDS, ncolors=coverage_cmap.N, clip=True)
    error_norm = Normalize(vmin=min(0.0, error_min), vmax=error_max)

    fig = plt.figure(figsize=(16.2, 11.9))
    gs = fig.add_gridspec(
        nrows=len(ROW_SPECS) + 2,
        ncols=4,
        width_ratios=[1, 1, 0.045, 0.045],
        height_ratios=[0.035] + [1] * len(ROW_SPECS) + [0.16],
        hspace=0.52,
        wspace=0.18,
    )

    heading_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3])]
    for ax in heading_axes:
        ax.axis("off")
    heading_axes[0].text(0.5, 0.18, "SPH coverage", ha="center", va="center", fontsize=16, fontweight="bold")
    heading_axes[1].text(0.5, 0.18, "Held-out prediction error", ha="center", va="center", fontsize=16, fontweight="bold")

    last_cov_image = None
    last_err_image = None
    for row_idx, row_data in enumerate(pair_data, start=1):
        spec = row_data["spec"]
        coverage = row_data["coverage"]
        error = row_data["error"]
        ax_cov = fig.add_subplot(gs[row_idx, 0])
        ax_err = fig.add_subplot(gs[row_idx, 1])

        shared_title = str(spec["title"])
        annotate_cov = should_annotate_na(np.ma.masked_invalid(coverage.to_numpy(dtype=float)))
        annotate_err = should_annotate_na(np.ma.masked_invalid(error.to_numpy(dtype=float)))

        last_cov_image = draw_panel(
            ax_cov,
            coverage,
            "Blues",
            coverage_norm,
            shared_title,
            str(spec["col_label"]),
            str(spec["row_label"]),
            annotate_na=annotate_cov,
            missing_mask=None,
        )
        last_err_image = draw_panel(
            ax_err,
            error,
            "Reds",
            error_norm,
            "",
            str(spec["col_label"]),
            str(spec["row_label"]),
            annotate_na=annotate_err,
        )

    cax_blue = fig.add_subplot(gs[1:1 + len(ROW_SPECS), 2])
    cax_red = fig.add_subplot(gs[1:1 + len(ROW_SPECS), 3])
    blue_cbar = fig.colorbar(last_cov_image, cax=cax_blue)
    blue_cbar.set_label("SPH support count", fontsize=12, labelpad=10)
    blue_cbar.set_ticks(SUPPORT_TICK_POSITIONS)
    blue_cbar.set_ticklabels(SUPPORT_TICK_LABELS)
    red_cbar = fig.colorbar(last_err_image, cax=cax_red)
    red_cbar.set_label("Mean absolute error", fontsize=12, labelpad=10)

    caption_ax = fig.add_subplot(gs[-1, :2])
    caption_ax.axis("off")
    caption_ax.legend(
        handles=[Patch(facecolor=NO_DATA_COLOR, edgecolor="#666666", label="No data")],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        frameon=False,
        fontsize=11,
    )
    caption_ax.text(
        0.0,
        0.08,
        "Blue shows SPH sample coverage. Red shows grouped held-out mean absolute error. Grey indicates that the corresponding quantity cannot be estimated; it does not represent zero.",
        ha="left",
        va="center",
        fontsize=11,
        color="#333333",
    )

    fig.suptitle("Model Trust Across Parameter-Space Support", fontsize=22, fontweight="bold", y=0.94)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    write_notes()


if __name__ == "__main__":
    main()
