from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
BOUND_SOURCE = ROOT / "extraction-outputs" / "tables" / "bound_outcomes.csv"
PREDICTIONS_SOURCE = ROOT / "report-table-figure" / "tables" / "tuned_gb_oof_predictions.csv"
OUTPUT_PATH = SCRIPT_DIR / "figure5_used_in_report.png"

NO_DATA_COLOR = "#CFCFCF"
SUPPORT_BOUNDS = [0, 1, 5, 10, 25, 50, 100, 201]
SUPPORT_TICK_POSITIONS = [0.5, 3, 7.5, 17, 37, 74.5, 150]
SUPPORT_TICK_LABELS = ["0", "1-4", "5-9", "10-24", "25-49", "50-99", "100-200"]
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]


def parse_numeric_code(series: pd.Series, pattern: str, scale: float = 1.0) -> pd.Series:
    extracted = series.fillna("").astype(str).str.extract(pattern)[0]
    return pd.to_numeric(extracted, errors="coerce") / scale


def load_bound_frame() -> pd.DataFrame:
    frame = pd.read_csv(BOUND_SOURCE, low_memory=False)
    frame["mass_log10_kg"] = parse_numeric_code(frame["mass_code"], r"A(\d+)", 100.0)
    frame["periapsis_Rm"] = parse_numeric_code(frame["periapsis_code"], r"r(\d+)", 10.0)
    frame["v_inf_kms"] = parse_numeric_code(frame["velocity_code"], r"v(\d+)", 10.0)
    return frame


def load_predictions() -> pd.DataFrame:
    predictions = pd.read_csv(PREDICTIONS_SOURCE, low_memory=False)
    predictions["abs_error"] = pd.to_numeric(predictions["residual"], errors="coerce").abs()
    return predictions


def unique_physical_coverage(frame: pd.DataFrame, row_col: str, col_col: str) -> pd.DataFrame:
    support = frame.dropna(subset=[row_col, col_col]).copy()
    coverage = support.pivot_table(index=row_col, columns=col_col, values="physical_file", aggfunc=pd.Series.nunique)
    coverage = coverage.reindex(index=sorted(pd.to_numeric(coverage.index, errors="coerce").dropna().unique()))
    coverage = coverage.reindex(columns=sorted(pd.to_numeric(coverage.columns, errors="coerce").dropna().unique()))
    return coverage.fillna(0)


def mean_abs_error_table(predictions: pd.DataFrame, row_col: str, col_col: str) -> pd.DataFrame:
    error_frame = predictions.dropna(subset=[row_col, col_col, "abs_error"]).copy()
    error = error_frame.pivot_table(index=row_col, columns=col_col, values="abs_error", aggfunc="mean")
    error = error.reindex(
        index=sorted(pd.to_numeric(error.index, errors="coerce").dropna().unique()),
        columns=sorted(pd.to_numeric(error.columns, errors="coerce").dropna().unique()),
    )
    return error


def mean_bmf_table(frame: pd.DataFrame, row_col: str, col_col: str) -> pd.DataFrame:
    bmf_frame = frame.dropna(subset=[row_col, col_col, "bound_mass_fraction"]).copy()
    table = bmf_frame.pivot_table(index=row_col, columns=col_col, values="bound_mass_fraction", aggfunc="mean")
    table = table.reindex(
        index=sorted(pd.to_numeric(table.index, errors="coerce").dropna().unique()),
        columns=sorted(pd.to_numeric(table.columns, errors="coerce").dropna().unique()),
    )
    return table


def format_label(value: float | str) -> str:
    if isinstance(value, str):
        return value
    value = float(value)
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}"
    return f"{value:.1f}".rstrip("0").rstrip(".")


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def style_heatmap_axes(ax: plt.Axes, table: pd.DataFrame, x_label: str, y_label: str) -> None:
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels([format_label(value) for value in table.columns], rotation=45, ha="right", fontsize=8.2)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels([format_label(value) for value in table.index], fontsize=8.2)
    ax.set_xlabel(x_label, fontsize=9.6, labelpad=2)
    ax.set_ylabel(y_label, fontsize=9.6, labelpad=2)
    ax.set_xticks(np.arange(-0.5, len(table.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(table.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#555555")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(0.0, 1.06, label, transform=ax.transAxes, fontsize=11.2, fontweight="bold", ha="left", va="bottom", color="#222222", clip_on=False)


def text_color_for_rgba(rgba: tuple[float, float, float, float]) -> str:
    red, green, blue, _alpha = rgba
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.5 else "#111111"


def draw_coverage_panel(ax: plt.Axes, table: pd.DataFrame, title: str, x_label: str, y_label: str, panel_label: str, norm: BoundaryNorm):
    cmap = colormaps["Blues"].copy()
    cmap.set_bad(NO_DATA_COLOR)
    values = table.to_numpy(dtype=float)
    masked = np.ma.masked_where(values <= 0, values)
    image = ax.imshow(masked, cmap=cmap, norm=norm, origin="lower", aspect="auto")
    ax.set_title(title, fontsize=10.8, pad=12)
    style_heatmap_axes(ax, table, x_label, y_label)
    add_panel_label(ax, panel_label)
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            count = int(values[row_idx, col_idx])
            if count > 0:
                rgba = cmap(norm(values[row_idx, col_idx]))
                ax.text(col_idx, row_idx, f"{count}", ha="center", va="center", fontsize=7.4, color=text_color_for_rgba(rgba))
    return image


def draw_error_panel(ax: plt.Axes, table: pd.DataFrame, title: str, x_label: str, y_label: str, panel_label: str, norm: Normalize, *, text_rotation: float = 90, text_fontsize: float = 6.6):
    cmap = colormaps["Reds"].copy()
    cmap.set_bad(NO_DATA_COLOR)
    masked = np.ma.masked_invalid(table.to_numpy(dtype=float))
    image = ax.imshow(masked, cmap=cmap, norm=norm, origin="lower", aspect="auto")
    ax.set_title(title, fontsize=10.8, pad=12)
    style_heatmap_axes(ax, table, x_label, y_label)
    add_panel_label(ax, panel_label)
    for row_idx in range(masked.shape[0]):
        for col_idx in range(masked.shape[1]):
            if bool(masked.mask[row_idx, col_idx]):
                continue
            value = float(masked[row_idx, col_idx])
            rgba = cmap(norm(value))
            ax.text(col_idx, row_idx, format_percent(value), ha="center", va="center", rotation=text_rotation, fontsize=text_fontsize, color=text_color_for_rgba(rgba))
    return image


def draw_bmf_panel(ax: plt.Axes, mean_table: pd.DataFrame, count_table: pd.DataFrame, title: str, x_label: str, y_label: str, panel_label: str, norm: Normalize, *, text_rotation: float = 0):
    cmap = colormaps["viridis"].copy()
    cmap.set_bad(NO_DATA_COLOR)
    counts = count_table.reindex(index=mean_table.index, columns=mean_table.columns, fill_value=0).to_numpy(dtype=float)
    means = mean_table.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(means)
    image = ax.imshow(masked, cmap=cmap, norm=norm, origin="lower", aspect="auto")
    ax.set_title(title, fontsize=10.8, pad=12)
    style_heatmap_axes(ax, mean_table, x_label, y_label)
    add_panel_label(ax, panel_label)
    for row_idx in range(counts.shape[0]):
        for col_idx in range(counts.shape[1]):
            if int(counts[row_idx, col_idx]) <= 0:
                continue
            value = float(means[row_idx, col_idx])
            rgba = cmap(norm(value))
            ax.text(col_idx, row_idx, format_percent(value), ha="center", va="center", rotation=text_rotation, fontsize=6.8, color=text_color_for_rgba(rgba))
    return image


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    bound_frame = load_bound_frame()
    predictions = load_predictions()

    mass_peri_cov = unique_physical_coverage(bound_frame, "mass_log10_kg", "periapsis_Rm")
    mass_peri_err = mean_abs_error_table(predictions, "mass_log10_kg", "periapsis_Rm")
    mass_peri_bmf = mean_bmf_table(bound_frame, "mass_log10_kg", "periapsis_Rm")
    peri_vel_cov = unique_physical_coverage(bound_frame, "periapsis_Rm", "v_inf_kms")
    peri_vel_err = mean_abs_error_table(predictions, "periapsis_Rm", "v_inf_kms")
    peri_vel_bmf = mean_bmf_table(bound_frame, "periapsis_Rm", "v_inf_kms")

    support_norm = BoundaryNorm(SUPPORT_BOUNDS, ncolors=colormaps["Blues"].N, clip=True)
    all_error_values = np.concatenate([
        mass_peri_err.to_numpy(dtype=float)[np.isfinite(mass_peri_err.to_numpy(dtype=float))],
        peri_vel_err.to_numpy(dtype=float)[np.isfinite(peri_vel_err.to_numpy(dtype=float))],
    ])
    error_norm = Normalize(vmin=0.0, vmax=float(np.max(all_error_values)) if all_error_values.size else 0.1)
    all_bmf_values = np.concatenate([
        mass_peri_bmf.to_numpy(dtype=float)[np.isfinite(mass_peri_bmf.to_numpy(dtype=float))],
        peri_vel_bmf.to_numpy(dtype=float)[np.isfinite(peri_vel_bmf.to_numpy(dtype=float))],
    ])
    bmf_norm = Normalize(vmin=0.0, vmax=float(np.max(all_bmf_values)) if all_bmf_values.size else 0.3)

    fig = plt.figure(figsize=(21.6, 9.8), dpi=220)
    gs = fig.add_gridspec(nrows=2, ncols=6, width_ratios=[1, 1, 1, 0.07, 0.07, 0.07], height_ratios=[1, 1], hspace=0.48, wspace=0.42)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    support_image = draw_coverage_panel(ax_a, mass_peri_cov, "Mass-periapsis: unique simulations", "Periapsis ($R_{Mars}$)", "Mass log10 kg", PANEL_LABELS[0], support_norm)
    error_image = draw_error_panel(ax_b, mass_peri_err, "Mass-periapsis: mean |OOF residual|", "Periapsis ($R_{Mars}$)", "Mass log10 kg", PANEL_LABELS[1], error_norm)
    bmf_image = draw_bmf_panel(ax_c, mass_peri_bmf, mass_peri_cov, "Mass-periapsis:\nmean observed mass fraction", "Periapsis ($R_{Mars}$)", "Mass log10 kg", PANEL_LABELS[2], bmf_norm, text_rotation=90)
    draw_coverage_panel(ax_d, peri_vel_cov, "Periapsis-velocity: unique simulations", r"$v_{\infty}$ (km s$^{-1}$)", "Periapsis $R_{Mars}$", PANEL_LABELS[3], support_norm)
    draw_error_panel(ax_e, peri_vel_err, "Periapsis-velocity: mean |OOF residual|", r"$v_{\infty}$ (km s$^{-1}$)", "Periapsis $R_{Mars}$", PANEL_LABELS[4], error_norm, text_rotation=0, text_fontsize=5.8)
    draw_bmf_panel(ax_f, peri_vel_bmf, peri_vel_cov, "Periapsis-velocity:\nmean observed mass fraction", r"$v_{\infty}$ (km s$^{-1}$)", "Periapsis $R_{Mars}$", PANEL_LABELS[5], bmf_norm)

    cax_support = fig.add_subplot(gs[:, 3])
    cax_error = fig.add_subplot(gs[:, 4])
    cax_bmf = fig.add_subplot(gs[:, 5])
    support_cbar = fig.colorbar(support_image, cax=cax_support)
    support_cbar.set_ticks(SUPPORT_TICK_POSITIONS)
    support_cbar.set_ticklabels(SUPPORT_TICK_LABELS)
    error_cbar = fig.colorbar(error_image, cax=cax_error)
    bmf_cbar = fig.colorbar(bmf_image, cax=cax_bmf)
    support_cbar.ax.tick_params(labelsize=8.0)
    error_cbar.ax.tick_params(labelsize=8.0)
    bmf_cbar.ax.tick_params(labelsize=8.0)
    support_cbar.set_label("Unique physical simulations", fontsize=9.0, labelpad=10)
    error_cbar.set_label("Mean |held-out residual| (%)", fontsize=9.0, labelpad=18)
    bmf_cbar.set_label("Mean observed BMF (%)", fontsize=9.0, labelpad=18)
    error_cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value * 100:.0f}%"))
    bmf_cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value * 100:.0f}%"))
    for cbar in (support_cbar, error_cbar, bmf_cbar):
        cbar.ax.yaxis.set_label_position("left")

    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
