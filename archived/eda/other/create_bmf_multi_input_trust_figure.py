#!/usr/bin/env python3
"""Create a clean multi-panel BMF trustability figure for classification models."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_slice_diagnostics import FEATURE_COLUMNS, classification_models, load_dataset


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "report" / "figures"
OUTPUT_PNG = OUTPUT_DIR / "bmf_multi_input_trustability.png"
OUTPUT_MD = ROOT / "report" / "bmf_multi_input_trustability_notes.md"
PREDICTIONS_CSV = ROOT / "report" / "slice_diagnostics_20260716" / "tables" / "classification_oof_predictions.csv"

TARGET = "bound_mass_fraction_ge_0_1"
MODEL_ORDER = ["logistic_regression", "random_forest", "gradient_boosting"]
MODEL_COLORS = {
    "logistic_regression": "#2ca02c",
    "random_forest": "#1f77b4",
    "gradient_boosting": "#d62728",
}
MODEL_MARKERS = {
    "logistic_regression": "o",
    "random_forest": "^",
    "gradient_boosting": "D",
}
MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
}


def choose_best_slice(df: pd.DataFrame, varying: str, fixed_cols: list[str]) -> tuple[dict[str, object], pd.DataFrame]:
    counts = df.groupby(fixed_cols)[varying].nunique().reset_index(name="n_unique").sort_values("n_unique", ascending=False)
    best = counts.iloc[0].to_dict()
    mask = pd.Series(True, index=df.index)
    for col in fixed_cols:
        mask &= df[col] == best[col]
    return best, df.loc[mask].copy()


def add_background(ax: plt.Axes, observed: np.ndarray) -> None:
    observed = np.sort(np.unique(observed))
    lo = float(observed.min())
    hi = float(observed.max())
    x_min, x_max = ax.get_xlim()
    ax.axvspan(x_min, lo, color="#f3d3d3", alpha=0.45, zorder=0)
    ax.axvspan(lo, hi, color="#dbe8ff", alpha=0.25, zorder=0)
    ax.axvspan(hi, x_max, color="#f3d3d3", alpha=0.45, zorder=0)


def fit_models(df: pd.DataFrame) -> dict[str, object]:
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].astype(bool)
    models = classification_models(X)
    return {name: model.fit(X, y) for name, model in models.items()}


def load_oof_predictions() -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS_CSV)


def draw_panel(
    ax: plt.Axes,
    varying: str,
    x_label: str,
    fixed_cols: list[str],
    df: pd.DataFrame,
    fitted_models: dict[str, object],
    oof: pd.DataFrame,
    title: str,
    fixed_label: str,
) -> tuple[float, float]:
    best, slice_df = choose_best_slice(df, varying, fixed_cols)
    observed = np.sort(slice_df[varying].unique())
    step = np.median(np.diff(observed)) if len(observed) > 1 else 0.2
    grid_values = np.linspace(max(0.0, observed.min() - step), observed.max() + step, 250)
    grid = pd.concat([slice_df.iloc[[0]].copy()] * len(grid_values), ignore_index=True)

    if varying == "periapsis_Rm":
        grid["periapsis_Rm"] = grid_values
        grid["periapsis_value"] = np.round(grid_values * 10.0).astype(int)
        grid["periapsis_code"] = grid["periapsis_value"].map(lambda value: f"r{value}")
    elif varying == "v_inf_kms":
        grid["v_inf_kms"] = grid_values
        grid["velocity_value"] = np.round(grid_values * 10.0).astype(int)
        grid["velocity_code"] = grid["velocity_value"].map(lambda value: f"v{value:02d}")
    else:
        raise ValueError(f"Unsupported varying axis: {varying}")

    slice_oof = oof.copy()
    for col, value in best.items():
        if col != "n_unique":
            slice_oof = slice_oof[slice_oof[col] == value]

    ax.set_xlim(grid_values.min(), grid_values.max())
    ax.set_ylim(-0.05, 1.05)
    add_background(ax, observed)
    ax.scatter(slice_df[varying], slice_df[TARGET].astype(int), color="black", s=28, label="Actual BMF ≥ 0.1", zorder=5)

    for model_name in MODEL_ORDER:
        probs = fitted_models[model_name].predict_proba(grid[FEATURE_COLUMNS])[:, 1]
        ax.plot(grid_values, probs, color=MODEL_COLORS[model_name], linewidth=2.2, label=f"{MODEL_LABELS[model_name]} curve")

        model_oof = slice_oof[slice_oof["model"] == model_name]
        ax.scatter(
            model_oof[varying],
            model_oof["predicted_probability"],
            color=MODEL_COLORS[model_name],
            marker=MODEL_MARKERS[model_name],
            s=42,
            alpha=0.95,
            label=f"{MODEL_LABELS[model_name]} OOF",
            zorder=6,
        )

    ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1.0)
    ax.set_title(title, fontsize=14, fontweight="semibold")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Predicted probability")
    ax.text(
        0.02,
        0.97,
        fixed_label.format(**best),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    return float(observed.min()), float(observed.max())


def write_notes(peri_range: tuple[float, float], vel_range: tuple[float, float]) -> None:
    text = "\n".join(
        [
            "# BMF Multi-Input Trustability Notes",
            "",
            f"- Figure: `{OUTPUT_PNG}`",
            f"- Periapsis interpolation range: `{peri_range[0]:.1f}` to `{peri_range[1]:.1f}` R_Mars",
            f"- Velocity interpolation range: `{vel_range[0]:.1f}` to `{vel_range[1]:.1f}` km s^-1",
            "- Blue background denotes interpolation inside the sampled range for that exact slice.",
            "- Red background denotes extrapolation outside the sampled range for that exact slice.",
        ]
    )
    OUTPUT_MD.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset().copy()
    fitted_models = fit_models(df)
    oof = load_oof_predictions()

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8), constrained_layout=True)

    peri_range = draw_panel(
        axes[0],
        varying="periapsis_Rm",
        x_label="Periapsis ($R_{Mars}$)",
        fixed_cols=["mass_code", "resolution_code", "velocity_code", "spin_code", "timestep", "fof_linking_length"],
        df=df,
        fitted_models=fitted_models,
        oof=oof,
        title="Threshold BMF vs periapsis",
        fixed_label="Fixed: mass={mass_code}, vel={velocity_code}, spin={spin_code}",
    )
    vel_range = draw_panel(
        axes[1],
        varying="v_inf_kms",
        x_label="$v_{\\infty}$ (km s$^{-1}$)",
        fixed_cols=["mass_code", "resolution_code", "periapsis_code", "spin_code", "timestep", "fof_linking_length"],
        df=df,
        fitted_models=fitted_models,
        oof=oof,
        title="Threshold BMF vs velocity at infinity",
        fixed_label="Fixed: mass={mass_code}, peri={periapsis_code}, spin={spin_code}",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Bound Mass Fraction Threshold: Multiple Inputs and Interpolation/Extrapolation Ranges", fontsize=18, fontweight="bold")

    fig.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)
    write_notes(peri_range, vel_range)


if __name__ == "__main__":
    main()
