#!/usr/bin/env python3
"""Generate one-parameter interpolation and extrapolation diagnostics for BMF."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_physics_structured_surrogate import (
    OUTPUT_ROOT,
    PLOTS_DIR,
    PRIMARY_TARGET,
    TABLES_DIR,
    MARS_RADIUS_KM,
    PROXIMITY_DISTANCE_RM,
    add_physics_features,
    asteroid_radius_km,
    build_group_folds,
    determine_promoted_model,
    eccentricity_proxy,
    evaluate_model_config_oof,
    feature_columns_for_set,
    load_canonical_dataset,
    tidal_disruption_radius_rm,
    time_inside_radius_hours,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = OUTPUT_ROOT / "reports"
INTERP_REPORT_PATH = REPORTS_DIR / "interpolation_diagnostics.md"
INTERP_SUMMARY_PATH = TABLES_DIR / "interpolation_case_summary.csv"
MASS_195_PATH = TABLES_DIR / "mass_19_5_case_summary.csv"
PROFILE_TABLE_PATH = TABLES_DIR / "one_parameter_profiles.csv"
PROFILE_PLOT_DIR = PLOTS_DIR / "interpolation_profiles"
MASS_195_PLOT = PROFILE_PLOT_DIR / "mass_19_5_interpolation_cases.png"


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    display_name: str
    core_column: str
    ordered: bool
    x_label: str
    tick_formatter: str = "numeric"
    min_step: float = 0.0
    use_codes_for_groups: bool = False


PARAMETER_SPECS = [
    ParameterSpec("mass", "Mass", "mass_log10_kg", True, "Asteroid mass ($\\log_{10}$ kg)", min_step=0.5),
    ParameterSpec("periapsis", "Periapsis", "periapsis_Rm", True, "Periapsis ($R_{Mars}$)", min_step=0.1),
    ParameterSpec("velocity", "Velocity", "v_inf_kms", True, "$v_\\infty$ (km s$^{-1}$)", min_step=0.2),
    ParameterSpec("spin_period", "Spin Period", "spin_period_hr", True, "Spin period (hr)", min_step=0.1),
    ParameterSpec("resolution", "Resolution", "resolution_value", True, "Resolution value $n$", tick_formatter="int", min_step=5.0),
    ParameterSpec("fof", "FoF Linking Length", "fof_linking_length", True, "FoF linking length", min_step=0.0001),
    ParameterSpec("spin_axis", "Spin Axis", "spin_axis", False, "Spin axis"),
]

GROUP_COLUMNS = [
    "mass_log10_kg",
    "periapsis_Rm",
    "v_inf_kms",
    "spin_period_hr",
    "spin_axis",
    "resolution_value",
    "fof_linking_length",
    "special_case_code",
    "timestep",
]


def ensure_dirs() -> None:
    PROFILE_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def safe_feature_columns(promoted: dict[str, object]) -> list[str]:
    columns = feature_columns_for_set(str(promoted["feature_set"]), bool(promoted["include_physics_features"]))
    return [column for column in columns if column != "largest_fragment_mass_fraction"]


def prepare_frame() -> pd.DataFrame:
    frame = add_physics_features(load_canonical_dataset(ROOT / "extraction_outputs" / "bound_outcomes.csv"))
    valid = frame[frame[PRIMARY_TARGET].notna()].copy()
    valid["resolution_value"] = pd.to_numeric(valid["resolution_value"], errors="coerce")
    valid["mass_log10_kg"] = pd.to_numeric(valid["mass_log10_kg"], errors="coerce")
    valid["periapsis_Rm"] = pd.to_numeric(valid["periapsis_Rm"], errors="coerce")
    valid["v_inf_kms"] = pd.to_numeric(valid["v_inf_kms"], errors="coerce")
    valid["spin_period_hr"] = pd.to_numeric(valid["spin_period_hr"], errors="coerce")
    valid["fof_linking_length"] = pd.to_numeric(valid["fof_linking_length"], errors="coerce")
    return valid


def recompute_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["target_mass_kg"] = np.power(10.0, pd.to_numeric(enriched["mass_log10_kg"], errors="coerce"))
    enriched["particle_log10"] = np.log10(pd.to_numeric(enriched["resolution_value"], errors="coerce"))
    enriched["encounter_eccentricity_proxy"] = eccentricity_proxy(enriched["periapsis_Rm"], enriched["v_inf_kms"])
    enriched["v_inf_squared"] = np.square(pd.to_numeric(enriched["v_inf_kms"], errors="coerce"))
    with np.errstate(divide="ignore", invalid="ignore"):
        enriched["periapsis_inverse"] = 1.0 / pd.to_numeric(enriched["periapsis_Rm"], errors="coerce")
        enriched["spin_frequency_hr_inv"] = 1.0 / pd.to_numeric(enriched["spin_period_hr"], errors="coerce")
    enriched["angular_momentum_proxy"] = pd.to_numeric(enriched["periapsis_Rm"], errors="coerce") * pd.to_numeric(
        enriched["v_inf_kms"], errors="coerce"
    )
    enriched["asteroid_radius_km"] = asteroid_radius_km(enriched["target_mass_kg"])
    tidal_threshold_rm = tidal_disruption_radius_rm()
    enriched["time_within_2_mars_radii_hr"] = [
        time_inside_radius_hours(float(peri), float(vel), PROXIMITY_DISTANCE_RM)
        for peri, vel in zip(enriched["periapsis_Rm"], enriched["v_inf_kms"])
    ]
    enriched["time_within_tidal_disruption_hr"] = [
        time_inside_radius_hours(float(peri), float(vel), tidal_threshold_rm)
        for peri, vel in zip(enriched["periapsis_Rm"], enriched["v_inf_kms"])
    ]
    enriched["has_spin"] = pd.to_numeric(enriched["has_explicit_spin"], errors="coerce").fillna(0).astype(int)
    enriched["particle_mass_proxy"] = enriched["target_mass_kg"] / pd.to_numeric(enriched["resolution_value"], errors="coerce")
    enriched["mass_resolution_interaction"] = pd.to_numeric(enriched["mass_log10_kg"], errors="coerce") - pd.to_numeric(
        enriched["particle_log10"], errors="coerce"
    )
    return enriched.replace([np.inf, -np.inf], np.nan)


def base_group_columns(spec: ParameterSpec) -> list[str]:
    return [column for column in GROUP_COLUMNS if column != spec.core_column]


def choose_anchor_case(frame: pd.DataFrame, spec: ParameterSpec) -> pd.Series:
    group_cols = base_group_columns(spec)
    work = frame.copy()
    if spec.name == "spin_period":
        work = work[work["has_explicit_spin"]].copy()
    if spec.name == "spin_axis":
        work = work[work["spin_period_hr"].notna()].copy()
    summary = (
        work.groupby(group_cols, dropna=False)[spec.core_column]
        .nunique(dropna=True)
        .reset_index(name="n_unique")
        .merge(work.groupby(group_cols, dropna=False).size().reset_index(name="n_rows"), on=group_cols, how="left")
    )
    best = summary.sort_values(["n_unique", "n_rows"], ascending=[False, False]).iloc[0]
    mask = pd.Series(True, index=work.index)
    for column in group_cols:
        value = best[column]
        if pd.isna(value):
            mask &= work[column].isna()
        else:
            mask &= work[column] == value
    anchor = work.loc[mask].sort_values(spec.core_column).iloc[0]
    return anchor


def anchor_mask(frame: pd.DataFrame, anchor: pd.Series, spec: ParameterSpec) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column in base_group_columns(spec):
        value = anchor[column]
        if pd.isna(value):
            mask &= frame[column].isna()
        else:
            mask &= frame[column] == value
    return mask


def classify_domain(values: np.ndarray, observed: np.ndarray) -> list[str]:
    observed = np.sort(np.unique(observed))
    if len(observed) == 0:
        return ["unsupported"] * len(values)
    lo = float(observed.min())
    hi = float(observed.max())
    labels: list[str] = []
    for value in values:
        if value < lo or value > hi:
            labels.append("extrapolation")
        elif np.isclose(observed, value, atol=1e-9).any():
            labels.append("observed")
        else:
            labels.append("interpolation")
    return labels


def build_numeric_grid(anchor: pd.Series, observed_values: np.ndarray, spec: ParameterSpec, points: int = 250) -> pd.DataFrame:
    observed_values = np.sort(np.unique(observed_values.astype(float)))
    step = spec.min_step
    if len(observed_values) > 1:
        diffs = np.diff(observed_values)
        step = max(spec.min_step, float(np.nanmedian(diffs)))
    x_min = observed_values.min() - step
    x_max = observed_values.max() + step
    grid_values = np.linspace(x_min, x_max, points)
    grid = pd.DataFrame([anchor.to_dict()] * points)
    grid[spec.core_column] = grid_values
    if spec.name == "resolution":
        grid["particle_log10"] = np.log10(grid["resolution_value"])
    if spec.name == "mass":
        grid["target_mass_kg"] = np.power(10.0, grid["mass_log10_kg"])
    grid = recompute_engineered_features(grid)
    grid["domain_type"] = classify_domain(grid_values, observed_values)
    return grid


def plot_background(ax: plt.Axes, observed_values: np.ndarray) -> None:
    observed_values = np.sort(np.unique(observed_values.astype(float)))
    lo = observed_values.min()
    hi = observed_values.max()
    ax.axvspan(ax.get_xlim()[0], lo, color="#f3d3d3", alpha=0.45, zorder=0)
    ax.axvspan(lo, hi, color="#dbe8ff", alpha=0.25, zorder=0)
    ax.axvspan(hi, ax.get_xlim()[1], color="#f3d3d3", alpha=0.45, zorder=0)


def anchor_description(anchor: pd.Series, spec: ParameterSpec) -> str:
    parts = []
    for column in base_group_columns(spec):
        value = anchor[column]
        if column == "mass_log10_kg":
            parts.append(f"mass={value:.1f}")
        elif column == "periapsis_Rm":
            parts.append(f"peri={value:.1f}")
        elif column == "v_inf_kms":
            parts.append(f"v_inf={value:.1f}")
        elif column == "spin_period_hr":
            parts.append(f"spin={value:.1f} hr" if pd.notna(value) else "spin=none")
        elif column == "resolution_value":
            parts.append(f"n={int(value)}")
        elif column == "fof_linking_length":
            parts.append(f"FoF={value:.4f}")
        elif column == "spin_axis":
            parts.append(f"axis={value}")
    return ", ".join(parts)


def format_xticks(ax: plt.Axes, spec: ParameterSpec, observed_values: np.ndarray) -> None:
    observed_values = np.sort(np.unique(observed_values))
    ax.set_xticks(observed_values)
    if spec.tick_formatter == "int":
        ax.set_xticklabels([str(int(round(value))) for value in observed_values], rotation=35, ha="right")
    else:
        ax.set_xticklabels([f"{value:g}" for value in observed_values], rotation=35, ha="right")


def create_numeric_profile(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    fitted_model,
    spec: ParameterSpec,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, Path]:
    anchor = choose_anchor_case(frame, spec)
    subset = frame.loc[anchor_mask(frame, anchor, spec)].copy().sort_values(spec.core_column)
    observed_values = subset[spec.core_column].to_numpy(dtype=float)
    grid = build_numeric_grid(anchor, observed_values, spec)
    grid["predicted"] = fitted_model.predict(grid[feature_columns])

    pred_subset = predictions.loc[anchor_mask(predictions, anchor, spec)].copy().sort_values(spec.core_column)
    actual_agg = subset.groupby(spec.core_column, dropna=False)[PRIMARY_TARGET].mean().reset_index()
    pred_agg = pred_subset.groupby(spec.core_column, dropna=False)["predicted"].mean().reset_index()
    merged = actual_agg.merge(pred_agg, on=spec.core_column, how="outer").sort_values(spec.core_column)
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.plot(grid[spec.core_column], grid["predicted"], color="#d62728", linewidth=2.2, label="Inference-safe full-model curve")
    ax.scatter(merged[spec.core_column], merged[PRIMARY_TARGET], color="black", s=40, zorder=6, label="SPH simulation")
    ax.scatter(merged[spec.core_column], merged["predicted"], color="#1f77b4", marker="D", s=48, zorder=7, label="Grouped OOF prediction")
    ax.set_xlim(grid[spec.core_column].min(), grid[spec.core_column].max())
    plot_background(ax, observed_values)
    ax.set_xlabel(spec.x_label)
    ax.set_ylabel("Bound mass fraction")
    ax.set_title(f"{spec.display_name} sweep with all other parameters fixed")
    ax.text(0.0, 1.02, anchor_description(anchor, spec), transform=ax.transAxes, fontsize=9, color="#444444")
    ax.grid(axis="y", alpha=0.25)
    format_xticks(ax, spec, observed_values)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = PROFILE_PLOT_DIR / f"{spec.name}_interpolation_profile.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    summary = pd.DataFrame(
        {
            "parameter": spec.name,
            "display_name": spec.display_name,
            "anchor_case": anchor_description(anchor, spec),
            "x_value": merged[spec.core_column].to_numpy(),
            "actual_bmf": merged[PRIMARY_TARGET].to_numpy(),
            "oof_predicted_bmf": merged["predicted"].to_numpy(),
            "abs_error": np.abs(merged[PRIMARY_TARGET] - merged["predicted"]).to_numpy(),
            "domain_type": classify_domain(merged[spec.core_column].to_numpy(dtype=float), observed_values),
            "plot_path": out_path.as_posix(),
        }
    )
    return summary, out_path


def create_spin_axis_profile(frame: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, Path]:
    spec = next(item for item in PARAMETER_SPECS if item.name == "spin_axis")
    anchor = choose_anchor_case(frame, spec)
    subset = frame.loc[anchor_mask(frame, anchor, spec)].copy()
    pred_subset = predictions.loc[anchor_mask(predictions, anchor, spec)].copy()
    order = list(pd.Index(subset["spin_axis"]).drop_duplicates())
    actual = subset.groupby("spin_axis")[PRIMARY_TARGET].mean().reindex(order)
    pred = pred_subset.groupby("spin_axis")["predicted"].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    x = np.arange(len(order))
    width = 0.35
    ax.bar(x - width / 2.0, actual.to_numpy(), width=width, color="black", alpha=0.75, label="SPH mean")
    ax.bar(x + width / 2.0, pred.to_numpy(), width=width, color="#1f77b4", alpha=0.75, label="Grouped OOF mean")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Bound mass fraction")
    ax.set_xlabel(spec.x_label)
    ax.set_title("Spin-axis categorical sweep with all other parameters fixed")
    ax.text(0.0, 1.02, anchor_description(anchor, spec), transform=ax.transAxes, fontsize=9, color="#444444")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = PROFILE_PLOT_DIR / "spin_axis_interpolation_profile.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    summary = pd.DataFrame(
        {
            "parameter": spec.name,
            "display_name": spec.display_name,
            "anchor_case": anchor_description(anchor, spec),
            "x_value": order,
            "actual_bmf": actual.to_numpy(),
            "oof_predicted_bmf": pred.to_numpy(),
            "abs_error": np.abs(actual.to_numpy() - pred.to_numpy()),
            "domain_type": ["observed_category"] * len(order),
            "plot_path": out_path.as_posix(),
        }
    )
    return summary, out_path


def create_mass_19_5_plot(frame: pd.DataFrame, predictions: pd.DataFrame, fitted_model, feature_columns: list[str]) -> pd.DataFrame:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    rows: list[dict[str, object]] = []
    common = {
        "v_inf_kms": 0.0,
        "spin_axis": "none",
        "resolution_value": 65,
        "fof_linking_length": 0.0020,
        "special_case_code": "none",
        "timestep": 90000,
    }
    for ax, peri in zip(axes, [1.2, 1.6]):
        subset = frame.copy()
        for column, value in common.items():
            subset = subset[subset[column] == value]
        subset = subset[np.isclose(subset["periapsis_Rm"], peri)].copy().sort_values("mass_log10_kg")
        pred_subset = predictions.copy()
        for column, value in common.items():
            pred_subset = pred_subset[pred_subset[column] == value]
        pred_subset = pred_subset[np.isclose(pred_subset["periapsis_Rm"], peri)].copy().sort_values("mass_log10_kg")
        actual_agg = subset.groupby("mass_log10_kg", dropna=False)[PRIMARY_TARGET].mean().reset_index()
        pred_agg = pred_subset.groupby("mass_log10_kg", dropna=False)["predicted"].mean().reset_index()
        merged = actual_agg.merge(pred_agg, on="mass_log10_kg", how="outer").sort_values("mass_log10_kg")
        observed_mass = merged["mass_log10_kg"].to_numpy(dtype=float)

        anchor = subset.iloc[0].copy()
        mass_grid = np.linspace(observed_mass.min() - 0.5, observed_mass.max() + 0.5, 250)
        grid = pd.DataFrame([anchor.to_dict()] * len(mass_grid))
        grid["mass_log10_kg"] = mass_grid
        grid = recompute_engineered_features(grid)
        grid["predicted"] = fitted_model.predict(grid[feature_columns])

        ax.plot(grid["mass_log10_kg"], grid["predicted"], color="#d62728", linewidth=2.2, label="Inference-safe full-model curve")
        ax.scatter(merged["mass_log10_kg"], merged[PRIMARY_TARGET], color="black", s=40, label="SPH simulation", zorder=6)
        ax.scatter(merged["mass_log10_kg"], merged["predicted"], color="#1f77b4", marker="D", s=50, label="Grouped OOF prediction", zorder=7)
        ax.axvspan(ax.get_xlim()[0], observed_mass.min(), color="#f3d3d3", alpha=0.45, zorder=0)
        ax.axvspan(observed_mass.min(), observed_mass.max(), color="#dbe8ff", alpha=0.25, zorder=0)
        ax.axvspan(observed_mass.max(), ax.get_xlim()[1], color="#f3d3d3", alpha=0.45, zorder=0)
        ax.set_title(f"Periapsis {peri:.1f} $R_{{Mars}}$")
        ax.set_xlabel("Asteroid mass ($\\log_{10}$ kg)")
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(observed_mass)
        ax.set_xticklabels([f"{value:.1f}" for value in observed_mass])
        for _, row in merged.iterrows():
            rows.append(
                {
                    "periapsis_Rm": peri,
                    "mass_log10_kg": float(row["mass_log10_kg"]),
                    "actual_bmf": float(row[PRIMARY_TARGET]),
                    "oof_predicted_bmf": float(row["predicted"]),
                    "abs_error": float(abs(row[PRIMARY_TARGET] - row["predicted"])),
                    "is_target_mass_19_5": bool(math.isclose(float(row["mass_log10_kg"]), 19.5)),
                    "anchor_case": json.dumps(common, sort_keys=True),
                }
            )
    axes[0].set_ylabel("Bound mass fraction")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
    fig.suptitle("Mass interpolation check around $\\log_{10}(M)=19.5$")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    fig.savefig(MASS_195_PLOT, dpi=220)
    plt.close(fig)
    summary = pd.DataFrame(rows)
    summary.to_csv(MASS_195_PATH, index=False)
    return summary


def build_report(profile_summary: pd.DataFrame, mass_195: pd.DataFrame, promoted: dict[str, object], feature_columns: list[str]) -> None:
    parameter_rollup = (
        profile_summary.groupby(["parameter", "display_name", "plot_path"], dropna=False)
        .agg(
            n_observed_points=("x_value", "size"),
            mean_abs_error=("abs_error", "mean"),
            max_abs_error=("abs_error", "max"),
        )
        .reset_index()
    )
    parameter_rollup.to_csv(INTERP_SUMMARY_PATH, index=False)
    lines = [
        "# Interpolation and Extrapolation Diagnostics",
        "",
        "- Date: `2026-08-05`",
        f"- Dataset: `extraction_outputs/bound_outcomes.csv`",
        f"- Model for these plots: inference-safe version of the promoted `{promoted['model_name']}` surrogate",
        f"- Inference-safe feature columns: `{', '.join(feature_columns)}`",
        "",
        "## What this diagnostic is testing",
        "",
        "These plots vary one input at a time while holding the others fixed on an observed archive slice.",
        "For ordered parameters, blue shading marks the part of the one-dimensional slice that lies inside the observed support for that exact anchor case, and red shading marks the local one-dimensional extrapolation beyond the sampled slice endpoints.",
        "For `spin_axis`, interpolation and extrapolation do not apply because the parameter is categorical rather than ordered.",
        "",
        "## Does the model interpolate at mass = 19.5?",
        "",
        "Not in the strict sense of an unseen-mass test, because mass `19.5` is already present in the archive.",
        "What we can test is whether the grouped held-out model predicts those `19.5` cases correctly when they are withheld by `physical_file`.",
        "",
    ]
    target_rows = mass_195[mass_195["is_target_mass_19_5"]].copy().sort_values("periapsis_Rm")
    for _, row in target_rows.iterrows():
        lines.append(
            f"- `periapsis={row['periapsis_Rm']:.1f}`, `mass=19.5`: actual `BMF={row['actual_bmf']:.4f}`, predicted `BMF={row['oof_predicted_bmf']:.4f}`, absolute error `{row['abs_error']:.4f}`"
        )
    lines.extend(
        [
            "",
            "Neighbouring observed masses on the same fixed slices are included in the case plot so you can directly compare whether `19.5` sits on the local trend or breaks it.",
            f"Case figure: `ml/physics_structured_surrogate/plots/interpolation_profiles/{MASS_195_PLOT.name}`",
            "",
            f"![Mass 19.5 interpolation cases](../plots/interpolation_profiles/{MASS_195_PLOT.name})",
            "",
            "## One-parameter profile discussion",
            "",
        ]
    )
    for parameter in profile_summary["parameter"].unique():
        sub = profile_summary[profile_summary["parameter"] == parameter].copy()
        mean_error = float(sub["abs_error"].mean())
        worst = sub.sort_values("abs_error", ascending=False).iloc[0]
        if len(sub) < 2:
            lines.append(
                f"- `{parameter}`: only `{len(sub)}` observed point exists on the matched anchor slice, so this archive does not support a true one-parameter interpolation check here. Plot: `{worst['plot_path']}`"
            )
        else:
            lines.append(
                f"- `{parameter}`: mean absolute error across the observed anchor slice is `{mean_error:.4f}`. Worst observed point on that slice is `{worst['x_value']}` with error `{worst['abs_error']:.4f}`. Plot: `{worst['plot_path']}`"
            )
        lines.append("")
        lines.append(f"![{sub['display_name'].iloc[0]} profile](../plots/interpolation_profiles/{Path(str(worst['plot_path'])).name})")
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "The important distinction is local support, not just whether a value lies inside the global training range.",
            "A point can be globally in-range and still behave badly if the exact local slice is sparse or if the target changes sharply across that slice.",
            "That is exactly the issue around `mass = 19.5`: it is in-range and observed, but it sits on a thin, low-periapsis, zero-velocity corner where the model can still miss the local retained-mass ordering.",
            "So the answer to “should it interpolate at 19.5?” is: it should be easier than a true extrapolation problem, but in the current archive it still does not do so well on that exact local slice.",
            "",
            "For the ordered parameters, the full-model curve can look smooth in red extrapolation zones, but those sections are not directly validated by observed SPH points on the same fixed slice.",
            "Those red segments should be treated as trend extensions, not as tested physical predictions.",
        ]
    )
    INTERP_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    promoted = determine_promoted_model(ROOT / "extraction_outputs" / "bound_outcomes.csv")
    feature_columns = safe_feature_columns(promoted)
    frame = prepare_frame()
    fold_assignments_path = TABLES_DIR / "fold_assignments.csv"
    fold_assignments = pd.read_csv(fold_assignments_path) if fold_assignments_path.exists() else build_group_folds(frame, frame["physical_file"].astype(str))
    _, predictions, fitted_model = evaluate_model_config_oof(
        frame,
        PRIMARY_TARGET,
        feature_columns,
        fold_assignments,
        str(promoted["model_name"]),
        promoted["params"],
    )

    profile_frames: list[pd.DataFrame] = []
    for spec in PARAMETER_SPECS:
        if spec.name == "spin_axis":
            summary, _ = create_spin_axis_profile(frame, predictions)
        else:
            summary, _ = create_numeric_profile(frame, predictions, fitted_model, spec, feature_columns)
        profile_frames.append(summary)
    profile_summary = pd.concat(profile_frames, ignore_index=True)
    profile_summary.to_csv(PROFILE_TABLE_PATH, index=False)
    mass_195 = create_mass_19_5_plot(frame, predictions, fitted_model, feature_columns)
    build_report(profile_summary, mass_195, promoted, feature_columns)


if __name__ == "__main__":
    main()
