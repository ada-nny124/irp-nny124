from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.eda.train_physics_structured_surrogate import (
    PRIMARY_TARGET,
    add_physics_features,
    build_regression_pipeline,
    feature_columns_for_set,
    load_canonical_dataset,
)


DATASET_PATH = Path("extraction_outputs/bound_outcomes.csv")
OUTPUT_PATH = Path("report/figures/bmf_rf_conditional_profiles_250_linspace.png")
GRID_POINTS = 250
INTERPOLATION_COLOR = "#dbe8ff"
EXTRAPOLATION_COLOR = "#f3d3d3"
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


def fit_random_forest(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], object]:
    model_frame = add_physics_features(frame.copy())
    feature_columns = feature_columns_for_set("with_fof_linking_length", include_physics=True)
    valid = model_frame.loc[model_frame[PRIMARY_TARGET].notna()].copy()
    pipeline = build_regression_pipeline(valid[feature_columns], "random_forest")
    fitted = pipeline.fit(valid[feature_columns], pd.to_numeric(valid[PRIMARY_TARGET], errors="coerce"))
    return valid, feature_columns, fitted


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

    refreshed = add_physics_features(grid)
    refreshed[parameter] = x_values
    return refreshed


def make_slice_specs(frame: pd.DataFrame) -> list[dict[str, object]]:
    peri_mask = (
        (frame["mass_log10_kg"] == 20.0)
        & (frame["v_inf_kms"] == 0.0)
        & (frame["spin_axis"] == "z")
        & (frame["spin_period_hr"] == 4.7)
        & (frame["resolution_value"] == 65)
        & (frame["fof_linking_length"] == 0.004)
        & (frame["timestep"] == 90000)
    )
    vel_mask = (
        (frame["mass_log10_kg"] == 20.0)
        & (frame["periapsis_Rm"] == 1.2)
        & (frame["spin_axis"] == "z")
        & (frame["spin_period_hr"] == 4.7)
        & (frame["resolution_value"] == 65)
        & (frame["fof_linking_length"] == 0.004)
        & (frame["timestep"] == 90000)
    )
    mass_mask = (
        (frame["periapsis_Rm"] == 1.2)
        & (frame["v_inf_kms"] == 0.0)
        & (frame["spin_axis"] == "z")
        & (frame["spin_period_hr"] == 4.7)
        & (frame["resolution_value"] == 65)
        & (frame["timestep"] == 90000)
        & (frame["mass_log10_kg"].isin([19.0, 20.0]))
    )
    spin_mask = (
        (frame["mass_log10_kg"] == 20.0)
        & (frame["periapsis_Rm"] == 1.2)
        & (frame["v_inf_kms"] == 0.0)
        & (frame["spin_axis"] == "z")
        & (frame["resolution_value"] == 65)
        & (frame["fof_linking_length"] == 0.004)
        & (frame["timestep"] == 90000)
    )

    return [
        {
            "parameter": "periapsis_Rm",
            "title": "Periapsis",
            "x_label": r"Periapsis ($R_{\mathrm{Mars}}$)",
            "x_range": (1.0, 3.0),
            "fixed_text": "Fixed: mass=20.0, v_inf=0.0, spin=4.7 h,\naxis=z, n=65, FoF=0.004",
            "mask": peri_mask,
            "base_selector": peri_mask & (frame["periapsis_Rm"] == 1.2),
        },
        {
            "parameter": "v_inf_kms",
            "title": "Encounter velocity",
            "x_label": r"$v_\infty$ (km s$^{-1}$)",
            "x_range": (0.0, 1.8),
            "physical_min": 0.0,
            "fixed_text": "Fixed: mass=20.0, peri=1.2, spin=4.7 h,\naxis=z, n=65, FoF=0.004",
            "mask": vel_mask,
            "base_selector": vel_mask & (frame["v_inf_kms"] == 0.0),
        },
        {
            "parameter": "mass_log10_kg",
            "title": "Asteroid mass (sparse exact slice)",
            "x_label": r"$\log_{10}(M/\mathrm{kg})$",
            "x_range": (18.0, 21.0),
            "fixed_text": "Fixed: peri=1.2, v_inf=0.0, spin=4.7 h,\naxis=z, n=65",
            "mask": mass_mask,
            "base_selector": mass_mask & (frame["mass_log10_kg"] == 20.0),
        },
        {
            "parameter": "spin_period_hr",
            "title": "Spin period (z-axis)",
            "x_label": "Spin period (h)",
            "x_range": (2.5, 17.5),
            "fixed_text": "Fixed: mass=20.0, peri=1.2, v_inf=0.0,\naxis=z, n=65, FoF=0.004",
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
) -> None:
    slice_df = frame.loc[spec["mask"]].sort_values(spec["parameter"]).copy()
    base_row = frame.loc[spec["base_selector"]].iloc[0]
    grid = linspace_grid(base_row, spec["parameter"], spec["x_range"][0], spec["x_range"][1])
    predicted = np.clip(model.predict(grid[feature_columns]), 0.0, 1.0)
    observed = np.sort(slice_df[spec["parameter"]].astype(float).unique())
    observed_min = float(observed.min())
    observed_max = float(observed.max())

    if spec["x_range"][0] < observed_min:
        ax.axvspan(spec["x_range"][0], observed_min, color=EXTRAPOLATION_COLOR, alpha=0.45, zorder=0)
    ax.axvspan(observed_min, observed_max, color=INTERPOLATION_COLOR, alpha=0.25, zorder=0)
    if observed_max < spec["x_range"][1]:
        ax.axvspan(observed_max, spec["x_range"][1], color=EXTRAPOLATION_COLOR, alpha=0.45, zorder=0)

    ax.plot(
        grid[spec["parameter"]],
        predicted,
        color="#2a78c4",
        linewidth=2.0,
        label="RF conditional prediction",
    )
    ax.scatter(
        slice_df[spec["parameter"]],
        slice_df[PRIMARY_TARGET],
        color="#1f77b4",
        s=26,
        label="SPH slice",
        zorder=3,
    )
    ax.set_title(spec["title"])
    ax.set_xlabel(spec["x_label"])
    ax.set_ylabel("BMF")
    ax.set_xlim(*spec["x_range"])
    if spec["parameter"] == "v_inf_kms":
        ax.set_xlim(left=max(0.0, spec["x_range"][0]))
    ax.set_ylim(-0.02, 0.30)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.text(
        0.5,
        -0.31,
        f"{spec['fixed_text']}\nlinspace={GRID_POINTS}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw = load_canonical_dataset(DATASET_PATH)
    frame, feature_columns, model = fit_random_forest(raw)
    specs = make_slice_specs(frame)

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.2))
    for ax, spec in zip(axes.flat, specs):
        render_panel(ax, frame, feature_columns, model, spec)

    fig.tight_layout(h_pad=3.0, w_pad=2.2)
    fig.subplots_adjust(bottom=0.14, right=0.78)
    fig.text(
        0.865,
        0.60,
        "Background guide",
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
    fig.text(
        0.865,
        0.56,
        "Blue: interpolation\ninside sampled range",
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": INTERPOLATION_COLOR, "edgecolor": "none", "boxstyle": "round,pad=0.35"},
    )
    fig.text(
        0.865,
        0.44,
        "Red: extrapolation\noutside sampled range",
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": EXTRAPOLATION_COLOR, "edgecolor": "none", "boxstyle": "round,pad=0.35"},
    )
    fig.savefig(OUTPUT_PATH, dpi=180)
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
