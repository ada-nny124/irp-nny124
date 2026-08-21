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

from ml.main.helper_functions_ml import (
    PRIMARY_TARGET,
    add_physics_features,
    build_regression_pipeline,
    load_canonical_dataset,
)

DATASET_PATH = REPO_ROOT / "extraction_outputs/bound_outcomes.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
GRID_POINTS = 250
INTERPOLATION_COLOR = "#dbe8ff"
EXTRAPOLATION_COLOR = "#f3d3d3"

RAW_FEATURE_COLUMNS = [
    "mass_log10_kg",
    "periapsis_Rm",
    "v_inf_kms",
    "spin_period_hr",
    "spin_axis",
    "resolution_value",
    "fof_linking_length",
]

PHYSICS_FEATURE_COLUMNS = [
    "v_inf_squared",
    "periapsis_inverse",
    "angular_momentum_proxy",
    "spin_frequency_hr_inv",
    "asteroid_radius_km",
    "encounter_eccentricity_proxy",
    "time_within_2_mars_radii_hr",
    "time_within_tidal_disruption_hr",
]

RAW_RF_PARAMS = {
    "n_estimators": 500,
    "max_features": 0.8,
    "min_samples_leaf": 2,
    "max_depth": None,
    "random_state": 42,
    "n_jobs": -1,
}

TUNED_RF_PARAMS = {
    "n_estimators": 500,
    "max_features": 0.8,
    "min_samples_leaf": 1,
    "max_depth": 10,
    "random_state": 42,
    "n_jobs": -1,
}

RAW_GB_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "max_depth": 3,
    "subsample": 1.0,
    "min_samples_leaf": 2,
    "random_state": 42,
}

TUNED_GB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.08,
    "max_depth": 3,
    "subsample": 0.8,
    "min_samples_leaf": 1,
    "random_state": 42,
}

NUMERIC_GRID_COLUMNS = [
    "mass_log10_kg", "target_mass_kg", "mass_value", "particle_log10",
    "resolution_value", "periapsis_Rm", "periapsis_value", "v_inf_kms",
    "velocity_value", "spin_period_hr", "spin_value", "timestep",
    "fof_linking_length",
]

MODEL_CONFIGS = {
    "rf_raw": ("random_forest", RAW_RF_PARAMS, RAW_FEATURE_COLUMNS),
    "rf_tuned": ("random_forest", TUNED_RF_PARAMS, RAW_FEATURE_COLUMNS),
    "rf_physics": ("random_forest", TUNED_RF_PARAMS, RAW_FEATURE_COLUMNS + PHYSICS_FEATURE_COLUMNS),
    "gb_raw": ("gradient_boosting", RAW_GB_PARAMS, RAW_FEATURE_COLUMNS),
    "gb_tuned": ("gradient_boosting", TUNED_GB_PARAMS, RAW_FEATURE_COLUMNS),
    "gb_physics": ("gradient_boosting", TUNED_GB_PARAMS, RAW_FEATURE_COLUMNS + PHYSICS_FEATURE_COLUMNS),
}


def fit_model(model_key: str, frame: pd.DataFrame):
    model_name, params, feature_columns = MODEL_CONFIGS[model_key]
    pipeline = build_regression_pipeline(frame[feature_columns], model_name, params)
    target = pd.to_numeric(frame[PRIMARY_TARGET], errors="coerce")
    return feature_columns, pipeline.fit(frame[feature_columns], target)


def linspace_grid(base_row: pd.Series, parameter: str, x_min: float, x_max: float, grid_points: int = GRID_POINTS) -> pd.DataFrame:
    x_values = np.linspace(x_min, x_max, grid_points)
    grid = pd.concat([base_row.to_frame().T] * grid_points, ignore_index=True)
    grid[parameter] = x_values
    if parameter == "mass_log10_kg":
        grid["target_mass_kg"] = np.power(10.0, grid[parameter].astype(float))
        grid["mass_value"] = np.round(grid[parameter].astype(float) * 100.0).astype(int)
    elif parameter == "periapsis_Rm":
        grid["periapsis_value"] = np.round(grid[parameter].astype(float) * 10.0).astype(int)
    elif parameter == "v_inf_kms":
        grid["velocity_value"] = np.round(grid[parameter].astype(float) * 10.0).astype(int)
    elif parameter == "spin_period_hr":
        grid["spin_value"] = np.round(grid[parameter].astype(float) * 10.0).astype(int)
    for column in NUMERIC_GRID_COLUMNS:
        if column in grid.columns:
            grid[column] = pd.to_numeric(grid[column], errors="coerce")
    refreshed = add_physics_features(grid)
    refreshed[parameter] = x_values
    return refreshed


def make_slice_specs(frame: pd.DataFrame) -> list[dict[str, object]]:
    common = (
        (frame["spin_axis"] == "z")
        & (frame["resolution_value"] == 65)
        & (frame["fof_linking_length"] == 0.004)
        & (frame["timestep"] == 90000)
    )
    peri = common & (frame["mass_log10_kg"] == 20.0) & (frame["v_inf_kms"] == 0.0) & (frame["spin_period_hr"] == 4.7)
    velocity = common & (frame["mass_log10_kg"] == 20.0) & (frame["periapsis_Rm"] == 1.2) & (frame["spin_period_hr"] == 4.7)
    mass = (
        (frame["periapsis_Rm"] == 1.2)
        & (frame["v_inf_kms"] == 0.0)
        & (frame["spin_axis"] == "z")
        & (frame["spin_period_hr"] == 4.7)
        & (frame["resolution_value"] == 65)
        & (frame["timestep"] == 90000)
        & frame["mass_log10_kg"].isin([19.0, 20.0])
    )
    spin = common & (frame["mass_log10_kg"] == 20.0) & (frame["periapsis_Rm"] == 1.2) & (frame["v_inf_kms"] == 0.0)
    return [
        {"parameter": "periapsis_Rm", "title": "Periapsis", "x_label": r"Periapsis ($R_{\mathrm{Mars}}$)", "x_range": (1.0, 3.0), "fixed_text": "Fixed: mass=20.0, v_inf=0.0, spin=4.7 h, axis=z, n=65, FoF=0.004", "mask": peri, "base_selector": peri & (frame["periapsis_Rm"] == 1.2)},
        {"parameter": "v_inf_kms", "title": "Encounter velocity", "x_label": r"$v_\infty$ (km s$^{-1}$)", "x_range": (0.0, 1.8), "fixed_text": "Fixed: mass=20.0, peri=1.2, spin=4.7 h, axis=z, n=65, FoF=0.004", "mask": velocity, "base_selector": velocity & (frame["v_inf_kms"] == 0.0)},
        {"parameter": "mass_log10_kg", "title": "Asteroid mass", "x_label": r"$\log_{10}(M/\mathrm{kg})$", "x_range": (18.0, 21.0), "fixed_text": "Fixed: peri=1.2, v_inf=0.0, spin=4.7 h, axis=z, n=65; FoF varies", "mask": mass, "base_selector": mass & (frame["mass_log10_kg"] == 20.0)},
        {"parameter": "spin_period_hr", "title": "Spin period (z-axis)", "x_label": "Spin period (h)", "x_range": (2.5, 17.5), "fixed_text": "Fixed: mass=20.0, peri=1.2, v_inf=0.0, axis=z, n=65, FoF=0.004", "mask": spin, "base_selector": spin & (frame["spin_period_hr"] == 4.7)},
    ]


def render_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    feature_columns: list[str],
    model: object,
    spec: dict[str, object],
    *,
    grid_points: int = GRID_POINTS,
    show_prediction_dots: bool = False,
) -> None:
    slice_df = frame.loc[spec["mask"]].sort_values(spec["parameter"])
    base_row = frame.loc[spec["base_selector"]].iloc[0]
    grid = linspace_grid(base_row, spec["parameter"], *spec["x_range"], grid_points=grid_points)
    predicted = np.clip(model.predict(grid[feature_columns]), 0.0, 1.0)
    observed = np.sort(slice_df[spec["parameter"]].astype(float).unique())
    observed_min, observed_max = float(observed.min()), float(observed.max())
    if spec["x_range"][0] < observed_min:
        ax.axvspan(spec["x_range"][0], observed_min, color=EXTRAPOLATION_COLOR, alpha=0.45, zorder=0)
    ax.axvspan(observed_min, observed_max, color=INTERPOLATION_COLOR, alpha=0.25, zorder=0)
    if observed_max < spec["x_range"][1]:
        ax.axvspan(observed_max, spec["x_range"][1], color=EXTRAPOLATION_COLOR, alpha=0.45, zorder=0)
    ax.plot(
        grid[spec["parameter"]],
        predicted,
        color="#2a78c4",
        linewidth=2.2,
        alpha=0.9,
        zorder=1,
        label="Prediction line",
    )
    if show_prediction_dots:
        ax.scatter(
            grid[spec["parameter"]],
            predicted,
            color="#2a78c4",
            s=12,
            alpha=0.55,
            zorder=2,
            label="Prediction dots",
        )
    ax.scatter(
        slice_df[spec["parameter"]],
        slice_df[PRIMARY_TARGET],
        color="#d95f02",
        s=26,
        zorder=3,
        label="SPH results",
    )
    ax.set_title(spec["title"])
    ax.set_xlabel(spec["x_label"])
    ax.set_ylabel("BMF")
    ax.set_xlim(*spec["x_range"])
    ax.set_ylim(-0.02, 0.30)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="best")
    footer = f"{spec['fixed_text']}\nlinspace={grid_points}"
    if show_prediction_dots:
        footer += " | prediction dots shown"
    ax.text(0.5, -0.22, footer, transform=ax.transAxes, ha="center", va="top", fontsize=8)


def make_figure(model_key: str, *, suffix: str = "", grid_points: int = GRID_POINTS, show_prediction_dots: bool = False) -> Path:
    raw = load_canonical_dataset(DATASET_PATH)
    frame = add_physics_features(raw.copy())
    frame = frame.loc[frame[PRIMARY_TARGET].notna()].copy()
    feature_columns, model = fit_model(model_key, frame)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4))
    for ax, spec in zip(axes.flat, make_slice_specs(frame)):
        render_panel(
            ax,
            frame,
            feature_columns,
            model,
            spec,
            grid_points=grid_points,
            show_prediction_dots=show_prediction_dots,
        )
    fig.tight_layout(h_pad=2.6, w_pad=2.2)
    output_path = OUTPUT_DIR / f"figure3_{model_key}{suffix}.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    for model_key in MODEL_CONFIGS:
        print(make_figure(model_key))
    print(make_figure("gb_tuned", suffix="_dots250", grid_points=250, show_prediction_dots=True))
    print(make_figure("gb_tuned", suffix="_dots50", grid_points=50, show_prediction_dots=True))
    print(make_figure("gb_tuned", suffix="_dots20", grid_points=20, show_prediction_dots=True))
    print(f"Created {len(MODEL_CONFIGS)} figure variants in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
