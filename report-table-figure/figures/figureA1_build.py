from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.model_training_scripts.helper_functions_ml import build_regression_pipeline, load_canonical_dataset


PRIMARY_TARGET = "captured_mass_fraction"
OUTPUT_ROOT_ENV = os.environ.get("PIPELINE_OUTPUT_ROOT")
OUTPUT_BASE = Path(OUTPUT_ROOT_ENV).resolve() if OUTPUT_ROOT_ENV else REPO_ROOT
DATASET_PATH = Path("extraction-outputs/tables/bound_outcomes.csv")
OUTPUT_PATH = OUTPUT_BASE / "report-table-figure" / "figures" / "figureA1_used_in_report.png"
GRID_POINTS = 20
BMF_YMAX_OVERRIDE: float | None = None
INTERPOLATION_COLOR = "#dbe8ff"
EXTRAPOLATION_COLOR = "#f3d3d3"
LINE_COLOR = "#3b80d0"
DOT_COLOR = "#7fb2e5"
SPH_COLOR = "#e46c0a"
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]
FEATURE_COLUMNS = [
    "mass_log10_kg",
    "periapsis_Rm",
    "v_inf_kms",
    "spin_period_hr",
    "spin_axis",
    "resolution_value",
    "fof_linking_length",
]
GB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.08,
    "max_depth": 3,
    "subsample": 0.8,
    "min_samples_leaf": 1,
    "random_state": 42,
}
NUMERIC_GRID_COLUMNS = [
    "mass_log10_kg",
    "target_mass_kg",
    "mass_value",
    "particle_log10",
    "resolution_value",
    "periapsis_Rm",
    "periapsis_value",
    "v_inf_kms",
    "velocity_value",
    "spin_period_hr",
    "spin_value",
    "timestep",
    "fof_linking_length",
]


def train_model(frame: pd.DataFrame) -> tuple[list[str], object]:
    pipeline = build_regression_pipeline(frame[FEATURE_COLUMNS], "gradient_boosting", GB_PARAMS)
    target = pd.to_numeric(frame[PRIMARY_TARGET], errors="coerce")
    model = pipeline.fit(frame[FEATURE_COLUMNS], target)
    return FEATURE_COLUMNS, model


def linspace_grid(
    base_row: pd.Series,
    parameter: str,
    x_min: float,
    x_max: float,
    points: int = GRID_POINTS,
) -> pd.DataFrame:
    x_values = np.linspace(x_min, x_max, points)
    grid = pd.concat([base_row.to_frame().T] * points, ignore_index=True)
    grid[parameter] = x_values

    if parameter == "mass_log10_kg":
        grid["target_mass_kg"] = np.power(10.0, grid["mass_log10_kg"].astype(float))
        grid["mass_value"] = np.round(grid["mass_log10_kg"].astype(float) * 100.0).astype(int)
        grid["mass_code"] = grid["mass_value"].map(lambda value: f"A{value:04d}")
    elif parameter == "periapsis_Rm":
        grid["periapsis_value"] = np.round(grid["periapsis_Rm"].astype(float) * 10.0).astype(int)
        grid["periapsis_code"] = grid["periapsis_value"].map(lambda value: f"r{value}")
    elif parameter == "v_inf_kms":
        grid["velocity_value"] = np.round(grid["v_inf_kms"].astype(float) * 10.0).astype(int)
        grid["velocity_code"] = grid["velocity_value"].map(lambda value: f"v{value:02d}")
    elif parameter == "spin_period_hr":
        spin_values = np.round(grid["spin_period_hr"].astype(float) * 10.0).astype(int)
        axis = str(base_row["spin_axis"])
        grid["spin_value"] = spin_values
        grid["spin_code"] = spin_values.map(lambda value: f"s{value:03d}{axis}")

    for column in NUMERIC_GRID_COLUMNS:
        if column in grid.columns:
            grid[column] = pd.to_numeric(grid[column], errors="coerce")
    grid[parameter] = x_values
    return grid


def build_prediction_grid(
    base_rows: pd.DataFrame,
    parameter: str,
    x_min: float,
    x_max: float,
    points: int = GRID_POINTS,
) -> pd.DataFrame:
    x_values = np.linspace(x_min, x_max, points)
    setting_rows = base_rows.loc[:, FEATURE_COLUMNS].drop_duplicates().reset_index(drop=True)
    grids: list[pd.DataFrame] = []
    for _, row in setting_rows.iterrows():
        grids.append(linspace_grid(row, parameter, x_min, x_max, points))
    grid = pd.concat(grids, ignore_index=True)
    grid["grid_x"] = np.tile(x_values, len(setting_rows))
    return grid


def collapse_observed_slice(slice_df: pd.DataFrame, parameter: str) -> pd.DataFrame:
    return (
        slice_df.groupby(parameter, as_index=False)
        .agg(observed_fraction=(PRIMARY_TARGET, "mean"), run_count=(PRIMARY_TARGET, "size"))
        .sort_values(parameter)
    )


def make_slice_specs(frame: pd.DataFrame) -> list[dict[str, object]]:
    no_spin = (frame["spin_axis"] == "none") & (frame["spin_period_hr"].isna())
    peri_mask = (
        (frame["mass_log10_kg"] == 20.0)
        & (frame["v_inf_kms"] == 0.0)
        & (frame["spin_axis"] == "z")
        & (frame["spin_period_hr"] == 4.7)
    )
    vel_mask = (
        (frame["mass_log10_kg"] == 20.0)
        & (frame["periapsis_Rm"] == 1.2)
        & no_spin
    )
    mass_mask = (
        (frame["periapsis_Rm"] == 1.2)
        & (frame["v_inf_kms"] == 0.0)
        & no_spin
    )
    spin_mask = (
        (frame["mass_log10_kg"] == 20.0)
        & (frame["periapsis_Rm"] == 1.2)
        & (frame["v_inf_kms"] == 0.0)
        & (frame["spin_axis"] == "z")
    )

    return [
        {
            "parameter": "periapsis_Rm",
            "title": "Periapsis",
            "x_label": r"Periapsis ($R_{\mathrm{Mars}}$)",
            "x_range": (0.5, 6.0),
            "fixed_text": "Fixed: mass = $10^{20}$ kg, $v_\\infty$ = 0 km s$^{-1}$,\nz-axis spin, period = 4.7 h",
            "mask": peri_mask,
            "base_selector": peri_mask & (frame["periapsis_Rm"] == 1.2),
        },
        {
            "parameter": "v_inf_kms",
            "title": "Encounter velocity",
            "x_label": r"$v_\infty$ (km s$^{-1}$)",
            "x_range": (0.0, 5.0),
            "fixed_text": "Fixed: mass = $10^{20}$ kg, periapsis = 1.2 $R_{\\mathrm{Mars}}$,\nno spin",
            "mask": vel_mask,
            "base_selector": vel_mask & (frame["v_inf_kms"] == 0.0),
        },
        {
            "parameter": "mass_log10_kg",
            "title": "Asteroid mass",
            "x_label": "Asteroid mass (kg)",
            "x_range": (16.0, 24.0),
            "fixed_text": "Fixed: periapsis = 1.2 $R_{\\mathrm{Mars}}$, $v_\\infty$ = 0 km s$^{-1}$,\nno spin",
            "mask": mass_mask,
            "base_selector": mass_mask & (frame["mass_log10_kg"] == 20.0),
        },
        {
            "parameter": "spin_period_hr",
            "title": "Spin period (z-axis)",
            "x_label": "Spin period (h)",
            "x_range": (0.5, 40.0),
            "fixed_text": "Fixed: mass = $10^{20}$ kg, periapsis = 1.2 $R_{\\mathrm{Mars}}$,\n$v_\\infty$ = 0 km s$^{-1}$, z-axis spin",
            "mask": spin_mask,
            "base_selector": spin_mask & (frame["spin_period_hr"] == 4.7),
        },
    ]


def render_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    feature_columns: list[str],
    model: object,
    spec: dict[str, object],
    panel_label: str,
) -> None:
    slice_df = frame.loc[spec["mask"]].sort_values(spec["parameter"]).copy()
    base_rows = frame.loc[spec["base_selector"], feature_columns].copy()
    grid = build_prediction_grid(base_rows, spec["parameter"], spec["x_range"][0], spec["x_range"][1])
    grid["predicted_fraction"] = np.clip(model.predict(grid[feature_columns]), 0.0, 1.0)
    collapsed = collapse_observed_slice(slice_df, spec["parameter"])
    synthetic_curve = grid.groupby("grid_x", as_index=False)["predicted_fraction"].mean().sort_values("grid_x")

    observed = np.sort(collapsed[spec["parameter"]].astype(float).unique())
    observed_min = float(observed.min())
    observed_max = float(observed.max())

    if spec["x_range"][0] < observed_min:
        ax.axvspan(spec["x_range"][0], observed_min, color=EXTRAPOLATION_COLOR, alpha=0.45, zorder=0)
    ax.axvspan(observed_min, observed_max, color=INTERPOLATION_COLOR, alpha=0.25, zorder=0)
    if observed_max < spec["x_range"][1]:
        ax.axvspan(observed_max, spec["x_range"][1], color=EXTRAPOLATION_COLOR, alpha=0.45, zorder=0)

    ax.plot(
        synthetic_curve["grid_x"],
        synthetic_curve["predicted_fraction"],
        color=LINE_COLOR,
        linewidth=2.0,
        label="Prediction line",
    )
    ax.scatter(
        synthetic_curve["grid_x"],
        synthetic_curve["predicted_fraction"],
        color=DOT_COLOR,
        s=12,
        label="Prediction dots",
        zorder=2,
    )
    ax.scatter(
        collapsed[spec["parameter"]],
        collapsed["observed_fraction"],
        color=SPH_COLOR,
        s=18,
        label="SPH results",
        zorder=3,
    )
    ax.set_title(spec["title"])
    ax.set_xlabel(spec["x_label"])
    ax.set_ylabel("Mass fraction")
    ax.set_xlim(*spec["x_range"])
    if spec["parameter"] == "mass_log10_kg":
        ticks = np.arange(16.0, 24.1, 2.0)
        ax.set_xticks(ticks)
        ax.set_xticklabels([fr"$10^{{{tick:g}}}$" for tick in ticks])
    y_upper = BMF_YMAX_OVERRIDE if BMF_YMAX_OVERRIDE is not None else 0.30
    ax.set_ylim(-0.02, y_upper)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.text(0.0, 1.03, panel_label, transform=ax.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold")
    ax.text(
        0.5,
        -0.31,
        spec["fixed_text"],
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw = load_canonical_dataset(DATASET_PATH)
    frame = raw.loc[raw[PRIMARY_TARGET].notna()].copy()
    feature_columns, model = train_model(frame)
    specs = make_slice_specs(frame)

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.2))
    for ax, spec, panel_label in zip(axes.flat, specs, PANEL_LABELS):
        render_panel(ax, frame, feature_columns, model, spec, panel_label)

    fig.tight_layout(h_pad=3.0, w_pad=2.2)
    fig.subplots_adjust(bottom=0.14)
    fig.savefig(OUTPUT_PATH, dpi=180)
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
