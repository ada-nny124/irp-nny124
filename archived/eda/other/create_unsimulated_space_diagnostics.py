#!/usr/bin/env python3
"""Unsimulated-space interpolation diagnostics for the BMF surrogate."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from train_physics_structured_surrogate import (
    OUTPUT_ROOT,
    PLOTS_DIR,
    PRIMARY_TARGET,
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
REPORT_PATH = OUTPUT_ROOT / "INTERPOLATION_IN_UNSIMULATED_SPACE.md"
UNSIMULATED_CSV = OUTPUT_ROOT / "unsimulated_interpolation_predictions.csv"
MASS_195_CSV = OUTPUT_ROOT / "unsimulated_mass_19_5_cases.csv"
RECOMMENDED_CSV = OUTPUT_ROOT / "recommended_new_sph_runs.csv"
TABLES_DIR = OUTPUT_ROOT / "tables"
PLOT_DIR = PLOTS_DIR / "unsimulated_space"

INPUT_COLUMNS = [
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
EXACT_INPUT_COLUMNS = INPUT_COLUMNS.copy()
PHYSICAL_SUPPORT_COLUMNS = [column for column in INPUT_COLUMNS if column != "fof_linking_length"]
NUMERIC_SUPPORT_COLUMNS = ["mass_log10_kg", "periapsis_Rm", "v_inf_kms", "spin_period_filled", "resolution_value"]
CATEGORICAL_SUPPORT_COLUMNS = ["spin_axis", "special_case_code"]
MODEL_NAMES = ["gradient_boosting", "random_forest", "ridge"]


@dataclass(frozen=True)
class PairSpec:
    name: str
    x_col: str
    y_col: str
    title: str


PAIR_SPECS = [
    PairSpec("mass_periapsis", "mass_log10_kg", "periapsis_Rm", "Mass × periapsis"),
    PairSpec("mass_velocity", "mass_log10_kg", "v_inf_kms", "Mass × velocity"),
    PairSpec("periapsis_velocity", "periapsis_Rm", "v_inf_kms", "Periapsis × velocity"),
    PairSpec("mass_spin_period", "mass_log10_kg", "spin_period_hr", "Mass × spin period"),
    PairSpec("periapsis_spin_period", "periapsis_Rm", "spin_period_hr", "Periapsis × spin period"),
]


def ensure_dirs() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


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
    valid["spin_axis"] = valid["spin_axis"].fillna("none")
    valid["special_case_code"] = valid["special_case_code"].fillna("none")
    valid["spin_period_filled"] = valid["spin_period_hr"].fillna(0.0)
    return valid


def fit_models(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, dict[str, object], dict[str, pd.DataFrame]]:
    fold_assignments_path = TABLES_DIR / "fold_assignments.csv"
    fold_assignments = pd.read_csv(fold_assignments_path) if fold_assignments_path.exists() else build_group_folds(
        frame, frame["physical_file"].astype(str)
    )
    metric_frames: list[pd.DataFrame] = []
    fitted_models: dict[str, object] = {}
    prediction_frames: dict[str, pd.DataFrame] = {}
    for model_name in MODEL_NAMES:
        metrics, predictions, fitted = evaluate_model_config_oof(
            frame,
            PRIMARY_TARGET,
            feature_columns,
            fold_assignments,
            model_name,
            None,
        )
        metric_frames.append(metrics)
        fitted_models[model_name] = fitted
        prediction_frames[model_name] = predictions
    metrics_frame = pd.concat(metric_frames, ignore_index=True)
    return metrics_frame, fitted_models, prediction_frames


def unique_exact_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[EXACT_INPUT_COLUMNS].drop_duplicates().reset_index(drop=True)


def unique_physical_support(frame: pd.DataFrame) -> pd.DataFrame:
    support = frame[PHYSICAL_SUPPORT_COLUMNS + ["physical_file"]].drop_duplicates().reset_index(drop=True)
    support["spin_period_filled"] = support["spin_period_hr"].fillna(0.0)
    return support


def build_support_geometry(support: pd.DataFrame) -> dict[str, object]:
    numeric = support[NUMERIC_SUPPORT_COLUMNS].to_numpy(dtype=float)
    scaler = StandardScaler().fit(numeric)
    scaled_numeric = scaler.transform(numeric)
    cat = pd.get_dummies(support[CATEGORICAL_SUPPORT_COLUMNS].astype(str), prefix=CATEGORICAL_SUPPORT_COLUMNS)
    support_matrix = np.hstack([scaled_numeric, cat.to_numpy(dtype=float)])
    nn = NearestNeighbors(metric="euclidean")
    nn.fit(support_matrix)
    observed_distances = nn.kneighbors(support_matrix, n_neighbors=min(6, len(support_matrix)), return_distance=True)[0]
    k5 = observed_distances[:, min(5, observed_distances.shape[1] - 1)]
    radius = float(np.quantile(k5, 0.75))
    hull = Delaunay(scaled_numeric) if len(support) > scaled_numeric.shape[1] else None
    return {
        "scaler": scaler,
        "cat_columns": cat.columns.tolist(),
        "nn": nn,
        "support_matrix": support_matrix,
        "radius": radius,
        "hull": hull,
    }


def encode_support_points(points: pd.DataFrame, geometry: dict[str, object]) -> np.ndarray:
    numeric = points[NUMERIC_SUPPORT_COLUMNS].to_numpy(dtype=float)
    scaled_numeric = geometry["scaler"].transform(numeric)
    cat = pd.get_dummies(points[CATEGORICAL_SUPPORT_COLUMNS].astype(str), prefix=CATEGORICAL_SUPPORT_COLUMNS)
    cat = cat.reindex(columns=geometry["cat_columns"], fill_value=0)
    return np.hstack([scaled_numeric, cat.to_numpy(dtype=float)])


def inside_convex_hull(points: pd.DataFrame, geometry: dict[str, object]) -> np.ndarray:
    hull = geometry["hull"]
    if hull is None:
        return np.zeros(len(points), dtype=bool)
    scaled_numeric = geometry["scaler"].transform(points[NUMERIC_SUPPORT_COLUMNS].to_numpy(dtype=float))
    return hull.find_simplex(scaled_numeric) >= 0


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
        time_inside_radius_hours(float(peri), float(vel), 2.0) for peri, vel in zip(enriched["periapsis_Rm"], enriched["v_inf_kms"])
    ]
    enriched["time_within_tidal_disruption_hr"] = [
        time_inside_radius_hours(float(peri), float(vel), tidal_threshold_rm)
        for peri, vel in zip(enriched["periapsis_Rm"], enriched["v_inf_kms"])
    ]
    enriched["has_spin"] = enriched["has_explicit_spin"].fillna(False).astype(int)
    enriched["particle_mass_proxy"] = enriched["target_mass_kg"] / pd.to_numeric(enriched["resolution_value"], errors="coerce")
    enriched["mass_resolution_interaction"] = pd.to_numeric(enriched["mass_log10_kg"], errors="coerce") - pd.to_numeric(
        enriched["particle_log10"], errors="coerce"
    )
    enriched["spin_period_filled"] = enriched["spin_period_hr"].fillna(0.0)
    return enriched.replace([np.inf, -np.inf], np.nan)


def choose_pair_anchor(frame: pd.DataFrame, spec: PairSpec) -> pd.Series:
    fixed_columns = [column for column in INPUT_COLUMNS if column not in {spec.x_col, spec.y_col}]
    work = frame.copy()
    if spec.y_col == "spin_period_hr" or spec.x_col == "spin_period_hr":
        work = work[work["has_explicit_spin"]].copy()
    grouped = (
        work.groupby(fixed_columns, dropna=False)
        .apply(lambda subset: subset[[spec.x_col, spec.y_col]].drop_duplicates().shape[0])
        .reset_index(name="n_observed_cells")
    )
    best = grouped.sort_values(["n_observed_cells"], ascending=False).iloc[0]
    mask = pd.Series(True, index=work.index)
    for column in fixed_columns:
        value = best[column]
        if pd.isna(value):
            mask &= work[column].isna()
        else:
            mask &= work[column] == value
    return work.loc[mask].iloc[0]


def observed_axis_values(frame: pd.DataFrame, column: str) -> list[float]:
    values = sorted(pd.to_numeric(frame[column], errors="coerce").dropna().unique().tolist())
    return [float(value) for value in values]


def midpoint_values(values: list[float]) -> list[float]:
    mids: list[float] = []
    for low, high in zip(values[:-1], values[1:]):
        mids.append(round((low + high) / 2.0, 6))
    return mids


def build_anchor_points(anchor: pd.Series, x_col: str, y_col: str, x_values: list[float], y_values: list[float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for x_value in x_values:
        for y_value in y_values:
            row = anchor[frame_like_columns(anchor)].to_dict()
            row[x_col] = x_value
            row[y_col] = y_value
            rows.append(row)
    points = pd.DataFrame(rows)
    points["has_explicit_spin"] = points["spin_axis"].astype(str) != "none"
    points["spin_period_filled"] = points["spin_period_hr"].fillna(0.0)
    return recompute_engineered_features(points)


def frame_like_columns(anchor: pd.Series) -> list[str]:
    return [column for column in anchor.index if column not in {"bound_mass_fraction", "target_mass_kg", "particle_log10"}]


def exact_combo_exists(points: pd.DataFrame, observed_exact: pd.DataFrame) -> np.ndarray:
    observed_keys = set(tuple(row) for row in observed_exact[EXACT_INPUT_COLUMNS].itertuples(index=False, name=None))
    return np.array([tuple(row) in observed_keys for row in points[EXACT_INPUT_COLUMNS].itertuples(index=False, name=None)])


def categorical_supported(points: pd.DataFrame, support: pd.DataFrame) -> np.ndarray:
    observed_cats = set(tuple(row) for row in support[CATEGORICAL_SUPPORT_COLUMNS].astype(str).itertuples(index=False, name=None))
    return np.array(
        [tuple(row) in observed_cats for row in points[CATEGORICAL_SUPPORT_COLUMNS].astype(str).itertuples(index=False, name=None)]
    )


def parameter_ranges(frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return {column: (float(frame[column].min()), float(frame[column].max())) for column in ["mass_log10_kg", "periapsis_Rm", "v_inf_kms", "resolution_value", "fof_linking_length"]}


def inside_ranges(points: pd.DataFrame, ranges: dict[str, tuple[float, float]]) -> np.ndarray:
    mask = np.ones(len(points), dtype=bool)
    for column, (low, high) in ranges.items():
        mask &= pd.to_numeric(points[column], errors="coerce").between(low, high, inclusive="both").to_numpy()
    spin_vals = sorted(points["spin_period_hr"].dropna().unique())
    mask &= ~((points["spin_axis"].astype(str) == "none") & points["spin_period_hr"].notna()).to_numpy()
    mask &= ~((points["spin_axis"].astype(str) != "none") & points["spin_period_hr"].isna()).to_numpy()
    return mask


def support_metrics(points: pd.DataFrame, support: pd.DataFrame, geometry: dict[str, object]) -> pd.DataFrame:
    encoded = encode_support_points(points, geometry)
    n_neighbors = min(5, len(support))
    distances, indices = geometry["nn"].kneighbors(encoded, n_neighbors=n_neighbors, return_distance=True)
    counts = (distances <= geometry["radius"]).sum(axis=1)
    nearest_files = []
    for row_indices in indices[:, :3]:
        nearest_files.append("; ".join(support.iloc[row_indices]["physical_file"].astype(str).tolist()))
    return pd.DataFrame(
        {
            "nearest_distance": distances[:, 0],
            "distance_k3_mean": distances[:, : min(3, n_neighbors)].mean(axis=1),
            "distance_k5_mean": distances.mean(axis=1),
            "local_support_count": counts,
            "nearest_supporting_physical_files": nearest_files,
        }
    )


def classify_points(
    points: pd.DataFrame,
    support: pd.DataFrame,
    geometry: dict[str, object],
    ranges: dict[str, tuple[float, float]],
    x_col: str,
    y_col: str,
    anchor_slice: pd.DataFrame,
) -> pd.DataFrame:
    supported_cats = categorical_supported(points, support)
    in_ranges = inside_ranges(points, ranges)
    in_hull = inside_convex_hull(points, geometry)
    metrics = support_metrics(points, support, geometry)
    slice_x = pd.to_numeric(anchor_slice[x_col], errors="coerce").dropna()
    slice_y = pd.to_numeric(anchor_slice[y_col], errors="coerce").dropna()
    x_low, x_high = float(slice_x.min()), float(slice_x.max())
    y_low, y_high = float(slice_y.min()), float(slice_y.max())
    edge_flag = (
        np.isclose(pd.to_numeric(points[x_col], errors="coerce"), x_low)
        | np.isclose(pd.to_numeric(points[x_col], errors="coerce"), x_high)
        | np.isclose(pd.to_numeric(points[y_col], errors="coerce"), y_low)
        | np.isclose(pd.to_numeric(points[y_col], errors="coerce"), y_high)
    )
    class_labels: list[str] = []
    for idx in range(len(points)):
        if not supported_cats[idx]:
            class_labels.append("unsupported categorical combination")
        elif not in_ranges[idx]:
            class_labels.append("extrapolation")
        elif not in_hull[idx]:
            class_labels.append("extrapolation")
        elif edge_flag[idx]:
            class_labels.append("boundary interpolation")
        elif metrics.loc[idx, "local_support_count"] >= 8 and metrics.loc[idx, "nearest_distance"] <= geometry["radius"]:
            class_labels.append("dense interpolation")
        else:
            class_labels.append("sparse interpolation")
    output = points.copy()
    output["interpolation_class"] = class_labels
    output["inside_parameter_ranges"] = in_ranges
    output["inside_convex_hull"] = in_hull
    output = pd.concat([output.reset_index(drop=True), metrics.reset_index(drop=True)], axis=1)
    return output


def predict_with_model_family(points: pd.DataFrame, feature_columns: list[str], fitted_models: dict[str, object]) -> pd.DataFrame:
    result = points.copy()
    model_preds: dict[str, np.ndarray] = {}
    for model_name, model in fitted_models.items():
        preds = np.asarray(model.predict(points[feature_columns]), dtype=float)
        result[f"pred_{model_name}"] = preds
        model_preds[model_name] = preds
    pred_matrix = np.vstack([model_preds[name] for name in MODEL_NAMES]).T
    result["predicted_bmf"] = result["pred_gradient_boosting"]
    result["prediction_mean"] = pred_matrix.mean(axis=1)
    result["prediction_min"] = pred_matrix.min(axis=1)
    result["prediction_max"] = pred_matrix.max(axis=1)
    result["prediction_std"] = pred_matrix.std(axis=1, ddof=0)
    result["prediction_range"] = result["prediction_max"] - result["prediction_min"]
    return result


def compute_local_prediction_variability(points: pd.DataFrame, feature_columns: list[str], promoted_model, frame: pd.DataFrame) -> pd.Series:
    observed_values = {
        "mass_log10_kg": observed_axis_values(frame, "mass_log10_kg"),
        "periapsis_Rm": observed_axis_values(frame, "periapsis_Rm"),
        "v_inf_kms": observed_axis_values(frame, "v_inf_kms"),
        "spin_period_hr": observed_axis_values(frame[frame["has_explicit_spin"]], "spin_period_hr"),
        "resolution_value": observed_axis_values(frame, "resolution_value"),
        "fof_linking_length": observed_axis_values(frame, "fof_linking_length"),
    }
    variability: list[float] = []
    for _, row in points.iterrows():
        neighbour_preds: list[float] = []
        for column, values in observed_values.items():
            current = float(row[column]) if pd.notna(row[column]) else math.nan
            if not math.isfinite(current) or len(values) == 0:
                continue
            lower = max([value for value in values if value < current], default=None)
            upper = min([value for value in values if value > current], default=None)
            for candidate in [lower, upper]:
                if candidate is None:
                    continue
                probe = pd.DataFrame([row.to_dict()])
                probe[column] = candidate
                probe = recompute_engineered_features(probe)
                neighbour_preds.append(float(promoted_model.predict(probe[feature_columns])[0]))
        variability.append(float(np.std(neighbour_preds, ddof=0)) if neighbour_preds else 0.0)
    return pd.Series(variability)


def assign_trust(points: pd.DataFrame) -> pd.DataFrame:
    trust: list[str] = []
    near_zero_boundary = (
        ((points["prediction_min"] <= 0.02) & (points["prediction_max"] >= 0.02))
        | points["prediction_mean"].between(0.02, 0.08, inclusive="both")
    )
    points["near_zero_boundary"] = near_zero_boundary
    for _, row in points.iterrows():
        if row["interpolation_class"] in {"extrapolation", "unsupported categorical combination"}:
            trust.append("low")
        elif row["prediction_range"] > 0.08 or row["local_prediction_variability"] > 0.05:
            trust.append("low")
        elif row["interpolation_class"] == "dense interpolation" and row["local_support_count"] >= 8 and row["prediction_range"] <= 0.03:
            trust.append("high")
        else:
            trust.append("medium")
    points["trust_level"] = trust
    return points


def add_source_metadata(points: pd.DataFrame, source: str, anchor: pd.Series | None, pair_name: str | None = None) -> pd.DataFrame:
    output = points.copy()
    output["source"] = source
    output["pair_name"] = pair_name or ""
    output["anchor_case"] = json.dumps({column: anchor[column] for column in INPUT_COLUMNS if column in anchor.index}, default=str) if anchor is not None else ""
    return output


def create_pair_maps(points: pd.DataFrame, spec: PairSpec, anchor: pd.Series, frame: pd.DataFrame) -> list[Path]:
    created: list[Path] = []
    x_vals = observed_axis_values(frame, spec.x_col)
    y_vals = observed_axis_values(frame, spec.y_col)
    existing = points[points["is_exact_observed"]].copy()
    missing = points[~points["is_exact_observed"]].copy()
    metrics = [
        ("predicted_bmf", "Predicted BMF in unsimulated cells", "viridis"),
        ("local_support_count", "Nearby unique physical simulations", "YlGnBu"),
        ("prediction_range", "Model disagreement range", "magma"),
    ]
    for metric, title, cmap in metrics:
        fig, ax = plt.subplots(figsize=(9.5, 7.0))
        values = missing[metric].to_numpy(dtype=float)
        norm = Normalize(vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)) if np.nanmax(values) > np.nanmin(values) else float(np.nanmin(values) + 1.0))
        cmap_obj = plt.get_cmap(cmap)
        for x in x_vals:
            for y in y_vals:
                cell = points[
                    np.isclose(pd.to_numeric(points[spec.x_col], errors="coerce"), x)
                    & np.isclose(pd.to_numeric(points[spec.y_col], errors="coerce"), y)
                ]
                if cell.empty:
                    continue
                row = cell.iloc[0]
                rect = Rectangle((x - 0.045, y - 0.045), 0.09, 0.09, facecolor="#d9d9d9", edgecolor="white", linewidth=0.7)
                if not bool(row["is_exact_observed"]):
                    rect.set_facecolor(cmap_obj(norm(float(row[metric]))))
                    if row["interpolation_class"] == "sparse interpolation":
                        rect.set_hatch("//")
                    elif row["interpolation_class"] == "boundary interpolation":
                        rect.set_edgecolor("#ff8c00")
                        rect.set_linewidth(2.0)
                    elif row["interpolation_class"] == "extrapolation":
                        rect.set_hatch("xx")
                    elif row["interpolation_class"] == "unsupported categorical combination":
                        rect.set_hatch("..")
                ax.add_patch(rect)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(metric.replace("_", " "))
        ax.set_xlim(min(x_vals) - 0.1, max(x_vals) + 0.1)
        ax.set_ylim(min(y_vals) - 0.1, max(y_vals) + 0.1)
        ax.set_xticks(x_vals)
        ax.set_yticks(y_vals)
        ax.set_xlabel(spec.x_col)
        ax.set_ylabel(spec.y_col)
        ax.set_title(f"{spec.title}: {title}\nGrey = existing SPH simulation, coloured = unsimulated model prediction")
        fig.tight_layout()
        out_path = PLOT_DIR / f"{spec.name}_{metric}.png"
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        created.append(out_path)
    return created


def one_parameter_grid(values: list[float]) -> list[float]:
    return sorted(set(values + midpoint_values(values)))


def create_one_parameter_sweeps(
    frame: pd.DataFrame,
    feature_columns: list[str],
    promoted_model,
    support: pd.DataFrame,
    geometry: dict[str, object],
    ranges: dict[str, tuple[float, float]],
) -> tuple[pd.DataFrame, list[Path], pd.DataFrame]:
    sweep_rows: list[pd.DataFrame] = []
    smoothness_rows: list[dict[str, object]] = []
    created: list[Path] = []
    sweep_specs = [
        ("mass_log10_kg", "Mass", "Asteroid mass ($\\log_{10}$ kg)"),
        ("periapsis_Rm", "Periapsis", "Periapsis ($R_{Mars}$)"),
        ("v_inf_kms", "Velocity", "$v_\\infty$ (km s$^{-1}$)"),
        ("spin_period_hr", "Spin Period", "Spin period (hr)"),
        ("resolution_value", "Resolution", "Resolution value $n$"),
        ("fof_linking_length", "FoF Linking Length", "FoF linking length"),
    ]
    for column, label, x_label in sweep_specs:
        work = frame[frame["has_explicit_spin"]].copy() if column == "spin_period_hr" else frame.copy()
        fixed_columns = [col for col in INPUT_COLUMNS if col != column]
        grouped = work.groupby(fixed_columns, dropna=False)[column].nunique(dropna=True).reset_index(name="n_unique")
        best = grouped.sort_values(["n_unique"], ascending=False).iloc[0]
        mask = pd.Series(True, index=work.index)
        for fixed in fixed_columns:
            value = best[fixed]
            if pd.isna(value):
                mask &= work[fixed].isna()
            else:
                mask &= work[fixed] == value
        anchor_slice = work.loc[mask].copy().sort_values(column)
        if anchor_slice.empty:
            continue
        anchor = anchor_slice.iloc[0]
        observed_values = sorted(pd.to_numeric(anchor_slice[column], errors="coerce").dropna().unique().tolist())
        grid_values = one_parameter_grid([float(value) for value in observed_values])
        if len(grid_values) >= 2:
            step = grid_values[1] - grid_values[0]
            grid_values = [grid_values[0] - step] + grid_values + [grid_values[-1] + step]
        rows = []
        for value in grid_values:
            row = anchor[frame_like_columns(anchor)].to_dict()
            row[column] = value
            rows.append(row)
        grid = pd.DataFrame(rows)
        grid["has_explicit_spin"] = grid["spin_axis"].astype(str) != "none"
        grid = recompute_engineered_features(grid)
        grid = classify_points(grid, support, geometry, ranges, column, column, anchor_slice[[column]].rename(columns={column: column}))
        grid["is_exact_observed"] = exact_combo_exists(grid, unique_exact_inputs(frame))
        grid = predict_with_model_family(grid, feature_columns, {"gradient_boosting": promoted_model, "random_forest": promoted_model, "ridge": promoted_model})
        # overwrite disagreement placeholders with promoted-only for sweeps
        promoted_preds = np.asarray(promoted_model.predict(grid[feature_columns]), dtype=float)
        grid["predicted_bmf"] = promoted_preds
        diffs = np.diff(promoted_preds)
        x_diffs = np.diff(pd.to_numeric(grid[column], errors="coerce").to_numpy(dtype=float))
        slopes = np.divide(diffs, x_diffs, out=np.zeros_like(diffs), where=x_diffs != 0)
        smoothness_rows.append(
            {
                "parameter": column,
                "anchor_case": json.dumps({fixed: best[fixed] for fixed in fixed_columns}, default=str),
                "largest_prediction_jump": float(np.max(np.abs(diffs))) if len(diffs) else 0.0,
                "largest_local_slope": float(np.max(np.abs(slopes))) if len(slopes) else 0.0,
                "monotonicity_comment": "expected increasing" if column == "periapsis_Rm" else ("expected decreasing" if column == "v_inf_kms" else "no strong monotonic prior"),
            }
        )
        plot_grid = grid.copy()
        fig, ax = plt.subplots(figsize=(9.5, 5.8))
        ax.plot(plot_grid[column], plot_grid["predicted_bmf"], color="#d62728", linewidth=2.0, label="Promoted GB prediction")
        observed_points = plot_grid[plot_grid["is_exact_observed"]]
        unsim_points = plot_grid[~plot_grid["is_exact_observed"] & plot_grid["interpolation_class"].isin(["dense interpolation", "sparse interpolation", "boundary interpolation"])]
        extrap_points = plot_grid[plot_grid["interpolation_class"] == "extrapolation"]
        ax.scatter(observed_points[column], observed_points["predicted_bmf"], color="black", s=35, label="Exact observed inputs")
        ax.scatter(unsim_points[column], unsim_points["predicted_bmf"], color="#1f77b4", marker="D", s=40, label="Unsimulated interpolation")
        if not extrap_points.empty:
            ax.scatter(extrap_points[column], extrap_points["predicted_bmf"], color="#ff8c00", marker="x", s=50, label="Extrapolation")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Predicted BMF")
        ax.set_title(f"{label} sweep: observed inputs vs unsimulated predictions")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out_path = PLOT_DIR / f"sweep_{column}.png"
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        created.append(out_path)
        plot_grid["source"] = "one_parameter_sweep"
        plot_grid["pair_name"] = ""
        plot_grid["anchor_case"] = json.dumps({fixed: best[fixed] for fixed in fixed_columns}, default=str)
        sweep_rows.append(plot_grid)
    return pd.concat(sweep_rows, ignore_index=True), created, pd.DataFrame(smoothness_rows)


def mass_19_5_unsimulated_cases(
    frame: pd.DataFrame,
    support: pd.DataFrame,
    geometry: dict[str, object],
    ranges: dict[str, tuple[float, float]],
    feature_columns: list[str],
    fitted_models: dict[str, object],
) -> tuple[pd.DataFrame, list[Path]]:
    created: list[Path] = []
    base = {
        "mass_log10_kg": 19.5,
        "spin_axis": "none",
        "spin_period_hr": math.nan,
        "resolution_value": 65,
        "fof_linking_length": 0.0020,
        "special_case_code": "none",
        "timestep": 90000,
    }
    exact_observed = unique_exact_inputs(frame)
    velocities = [0.0, 0.4, 0.8]
    peri_values = observed_axis_values(frame, "periapsis_Rm")
    rows = []
    for velocity in velocities:
        for peri in peri_values:
            row = base.copy()
            row["v_inf_kms"] = velocity
            row["periapsis_Rm"] = peri
            rows.append(row)
    peri_plot = pd.DataFrame(rows)
    peri_plot["has_explicit_spin"] = False
    peri_plot = recompute_engineered_features(peri_plot)
    peri_plot["is_exact_observed"] = exact_combo_exists(peri_plot, exact_observed)
    peri_plot = classify_points(peri_plot, support, geometry, ranges, "periapsis_Rm", "v_inf_kms", frame)
    peri_plot = predict_with_model_family(peri_plot, feature_columns, fitted_models)
    peri_plot["local_prediction_variability"] = compute_local_prediction_variability(peri_plot, feature_columns, fitted_models["gradient_boosting"], frame)
    peri_plot = assign_trust(peri_plot)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {0.0: "#1f77b4", 0.4: "#2ca02c", 0.8: "#d62728"}
    for velocity in velocities:
        subset = peri_plot[peri_plot["v_inf_kms"] == velocity].sort_values("periapsis_Rm")
        ax.plot(subset["periapsis_Rm"], subset["predicted_bmf"], color=colors[velocity], linewidth=2.0, label=f"v_inf={velocity:.1f}")
        observed = subset[subset["is_exact_observed"]]
        missing = subset[~subset["is_exact_observed"]]
        ax.scatter(observed["periapsis_Rm"], observed["predicted_bmf"], color=colors[velocity], edgecolor="black", s=50, marker="o")
        ax.scatter(missing["periapsis_Rm"], missing["predicted_bmf"], color=colors[velocity], s=40, marker="D", alpha=0.8)
    ax.set_xlabel("Periapsis ($R_{Mars}$)")
    ax.set_ylabel("Predicted BMF")
    ax.set_title("Mass = 19.5 unsimulated predictions vs periapsis")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = PLOT_DIR / "mass_19_5_vs_periapsis.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    created.append(out_path)

    peri_fixed = [1.2, 1.6, 2.0]
    vel_values = observed_axis_values(frame, "v_inf_kms")
    rows = []
    for peri in peri_fixed:
        for velocity in vel_values:
            row = base.copy()
            row["periapsis_Rm"] = peri
            row["v_inf_kms"] = velocity
            rows.append(row)
    vel_plot = pd.DataFrame(rows)
    vel_plot["has_explicit_spin"] = False
    vel_plot = recompute_engineered_features(vel_plot)
    vel_plot["is_exact_observed"] = exact_combo_exists(vel_plot, exact_observed)
    vel_plot = classify_points(vel_plot, support, geometry, ranges, "v_inf_kms", "periapsis_Rm", frame)
    vel_plot = predict_with_model_family(vel_plot, feature_columns, fitted_models)
    vel_plot["local_prediction_variability"] = compute_local_prediction_variability(vel_plot, feature_columns, fitted_models["gradient_boosting"], frame)
    vel_plot = assign_trust(vel_plot)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {1.2: "#1f77b4", 1.6: "#ff7f0e", 2.0: "#2ca02c"}
    for peri in peri_fixed:
        subset = vel_plot[vel_plot["periapsis_Rm"] == peri].sort_values("v_inf_kms")
        ax.plot(subset["v_inf_kms"], subset["predicted_bmf"], color=colors[peri], linewidth=2.0, label=f"peri={peri:.1f}")
        observed = subset[subset["is_exact_observed"]]
        missing = subset[~subset["is_exact_observed"]]
        ax.scatter(observed["v_inf_kms"], observed["predicted_bmf"], color=colors[peri], edgecolor="black", s=50, marker="o")
        ax.scatter(missing["v_inf_kms"], missing["predicted_bmf"], color=colors[peri], s=40, marker="D", alpha=0.8)
    ax.set_xlabel("$v_\\infty$ (km s$^{-1}$)")
    ax.set_ylabel("Predicted BMF")
    ax.set_title("Mass = 19.5 unsimulated predictions vs velocity")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = PLOT_DIR / "mass_19_5_vs_velocity.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    created.append(out_path)

    combined = pd.concat([peri_plot.assign(source="mass19.5_peri"), vel_plot.assign(source="mass19.5_velocity")], ignore_index=True)
    combined = combined[~combined["is_exact_observed"]].copy()
    combined.to_csv(MASS_195_CSV, index=False)
    return combined, created


def select_recommended_runs(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    def pick(sub: pd.DataFrame, reason: str) -> None:
        if sub.empty:
            return
        row = sub.iloc[0].copy()
        row["reason"] = reason
        rows.append(row)
    pick(
        points[(points["trust_level"] == "high") & (points["interpolation_class"] == "dense interpolation")].sort_values(
            ["prediction_range", "nearest_distance"]
        ),
        "High-confidence dense interpolation point for a clean surrogate interpolation check.",
    )
    pick(
        points[(points["interpolation_class"] == "sparse interpolation")].sort_values(["prediction_range", "nearest_distance"], ascending=[False, False]),
        "Sparse interpolation point to test whether the surrogate still behaves well with thinner local support.",
    )
    pick(
        points[points["near_zero_boundary"]].sort_values(["prediction_range", "local_prediction_variability"], ascending=[False, False]),
        "Near zero/non-zero transition where a new SPH run would test whether the surrogate handles retention onset correctly.",
    )
    pick(
        points.sort_values("prediction_range", ascending=False),
        "High model-disagreement point where model family spread indicates elevated risk.",
    )
    pick(
        points[(points["mass_log10_kg"] == 19.5) & (~points["is_exact_observed"])].sort_values(
            ["trust_level", "prediction_range", "nearest_distance"], ascending=[True, False, False]
        ),
        "Mass = 19.5 missing-input validation point motivated directly by supervisor questions.",
    )
    result = pd.DataFrame(rows).drop_duplicates(subset=EXACT_INPUT_COLUMNS)
    keep_cols = EXACT_INPUT_COLUMNS + [
        "predicted_bmf",
        "prediction_min",
        "prediction_max",
        "prediction_std",
        "prediction_range",
        "interpolation_class",
        "nearest_distance",
        "local_support_count",
        "inside_parameter_ranges",
        "inside_convex_hull",
        "near_zero_boundary",
        "trust_level",
        "reason",
    ]
    result = result[keep_cols]
    result.to_csv(RECOMMENDED_CSV, index=False)
    return result


def write_report(
    metrics: pd.DataFrame,
    pair_points: list[pd.DataFrame],
    sweep_summary: pd.DataFrame,
    smoothness: pd.DataFrame,
    mass195: pd.DataFrame,
    recommended: pd.DataFrame,
    created_plots: list[Path],
) -> None:
    all_points = pd.concat(pair_points + [mass195], ignore_index=True)
    class_counts = all_points["interpolation_class"].value_counts().to_dict()
    lines = [
        "# Unsimulated-Space Interpolation Diagnostics",
        "",
        "This analysis is separate from the held-out validation diagnostics.",
        "Existing grouped held-out cases still measure validation error on known SPH simulations.",
        "The unsimulated cases analysed here have no known true error. Coverage, local support, convex-hull checks, model disagreement, and smoothness only estimate risk, not correctness.",
        "",
        "## Questions Answered",
        "",
        "1. Can the model generate predictions for unsimulated combinations?",
        "2. Which unsimulated combinations are genuine interpolation rather than extrapolation?",
        "3. Where is interpolation densely supported, sparse, or near a transition?",
        "4. Does the prediction vary smoothly between known simulations?",
        "5. Where do models disagree?",
        "6. Which new SPH simulations would best validate the surrogate?",
        "",
        "## Model Setup",
        "",
        "- Dataset: `extraction_outputs/bound_outcomes.csv`",
        "- Target: `bound_mass_fraction`",
        "- Promoted inference-safe model used for primary predictions: gradient boosting surrogate without outcome-derived features",
        "- Companion model-disagreement set: gradient boosting, random forest, and ridge",
        "- Grouping variable for fitted pipelines: `physical_file`",
        "",
        "### Reference grouped-CV metrics on observed SPH rows",
    ]
    for _, row in metrics.sort_values("model").iterrows():
        lines.append(f"- `{row['model']}`: `R² = {row['r2']:.4f}`, `MAE = {row['mae']:.4f}`")
    lines.extend(
        [
            "",
            "## Interpolation Class Definitions",
            "",
            "- `dense interpolation`: inside parameter ranges, categorically supported, inside the numeric convex hull, and locally well supported by nearby unique physical simulations.",
            "- `sparse interpolation`: inside parameter ranges and hull, but with thinner nearby support.",
            "- `boundary interpolation`: inside parameter ranges, but lying on the edge of the local observed slice for at least one varied parameter.",
            "- `extrapolation`: outside the numeric convex hull or outside observed parameter ranges.",
            "- `unsupported categorical combination`: categorical state not represented in the archive support set.",
            "",
            "## Overall Unsimulated-Point Summary",
        ]
    )
    for key in ["dense interpolation", "sparse interpolation", "boundary interpolation", "extrapolation", "unsupported categorical combination"]:
        lines.append(f"- `{key}`: `{class_counts.get(key, 0)}` points")
    lines.extend(
        [
            "",
            "## Gap-Focused 2D Maps",
            "",
            "Grey cells are exact SPH simulations that already exist in the archive.",
            "Coloured cells are unsimulated model predictions.",
            "Sparse interpolation cells use hatching. Boundary interpolation cells use orange borders. Extrapolation cells use cross-hatching.",
            "",
        ]
    )
    map_groups = {
        "mass_periapsis": "Mass × periapsis",
        "mass_velocity": "Mass × velocity",
        "periapsis_velocity": "Periapsis × velocity",
    }
    for prefix, title in map_groups.items():
        lines.append(f"### {title}")
        for suffix in ["predicted_bmf", "local_support_count", "prediction_range"]:
            path = PLOT_DIR / f"{prefix}_{suffix}.png"
            lines.append(f"![{title} {suffix}](plots/unsimulated_space/{path.name})")
            lines.append("")
    lines.extend(
        [
            "## One-Parameter Smoothness Diagnostics",
            "",
            "These sweeps vary one physical parameter at a time, hold the others fixed at an observed anchor case, and distinguish exact observed inputs from unsimulated interpolation points.",
            "A smooth-looking curve does not prove physical correctness; it only shows how the surrogate behaves between or beyond known points.",
            "",
        ]
    )
    for _, row in smoothness.iterrows():
        path = PLOT_DIR / f"sweep_{row['parameter']}.png"
        lines.append(
            f"- `{row['parameter']}`: largest adjacent prediction jump `{row['largest_prediction_jump']:.4f}`, largest local slope `{row['largest_local_slope']:.4f}`, interpretation `{row['monotonicity_comment']}`."
        )
        if path.exists():
            lines.append(f"![{row['parameter']} sweep](plots/unsimulated_space/{path.name})")
            lines.append("")
    lines.extend(
        [
            "## Specific `mass = 19.5` Investigation",
            "",
            "Case A is the existing `mass = 19.5` SPH simulations already covered by held-out validation diagnostics.",
            "Case B is the genuinely missing-input prediction set at `mass = 19.5`, analysed below without claiming true error.",
            "",
            f"- Unsimulated `mass = 19.5` rows saved in `{MASS_195_CSV.as_posix()}`",
            "- These points are classified by support and interpolation risk, not by error.",
            "",
            "![Mass 19.5 vs periapsis](plots/unsimulated_space/mass_19_5_vs_periapsis.png)",
            "",
            "![Mass 19.5 vs velocity](plots/unsimulated_space/mass_19_5_vs_velocity.png)",
            "",
        ]
    )
    if not mass195.empty:
        dense195 = mass195["interpolation_class"].value_counts().to_dict()
        for key, value in dense195.items():
            lines.append(f"- `mass = 19.5` `{key}` points: `{value}`")
    lines.extend(
        [
            "",
            "## Where Models Disagree",
            "",
            "Prediction spread here means model disagreement across gradient boosting, random forest, and ridge. It is not a calibrated posterior uncertainty.",
            f"- Highest disagreement in the generated unsimulated set: `{all_points['prediction_range'].max():.4f}`",
            f"- Median disagreement in the generated unsimulated set: `{all_points['prediction_range'].median():.4f}`",
            "",
            "## Trust Guidance",
            "",
            "- High trust: dense interpolation with strong local support and low model disagreement.",
            "- Medium trust: interpolation with either weaker support or moderate disagreement.",
            "- Low trust: extrapolation, unsupported categorical combinations, high disagreement, or high local variability.",
            "",
            "## Recommended New SPH Validation Runs",
            "",
            "These are candidate missing-input simulations chosen to test the surrogate in complementary regimes.",
        ]
    )
    for _, row in recommended.iterrows():
        lines.append(
            f"- `mass={row['mass_log10_kg']:.1f}`, `peri={row['periapsis_Rm']:.1f}`, `v_inf={row['v_inf_kms']:.1f}`, `spin={row['spin_axis']}`, `resolution={int(row['resolution_value'])}`, `FoF={row['fof_linking_length']:.4f}`: predicted `BMF={row['predicted_bmf']:.4f}`, class `{row['interpolation_class']}`, support `{int(row['local_support_count'])}`, disagreement `{row['prediction_range']:.4f}`. {row['reason']}"
        )
    lines.extend(
        [
            "",
            "## Final Answer",
            "",
            "Yes, the surrogate can generate predictions for unsimulated combinations.",
            "Some of those points are genuine interpolation in supported regions, while others are sparse, boundary-adjacent, or effectively extrapolative in the physical input space.",
            "The most defensible missing-input predictions are the dense-interpolation points with low disagreement and strong local support.",
            "The least defensible are sparse or extrapolative points near sharp zero/non-zero transitions or where model disagreement is large.",
            "Only new SPH simulations can validate whether any genuinely unsimulated prediction is actually correct.",
            "",
            "## Files",
            "",
            f"- Report: `{REPORT_PATH.as_posix()}`",
            f"- Main CSV: `{UNSIMULATED_CSV.as_posix()}`",
            f"- Mass 19.5 CSV: `{MASS_195_CSV.as_posix()}`",
            f"- Recommended runs CSV: `{RECOMMENDED_CSV.as_posix()}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    promoted = determine_promoted_model(ROOT / "extraction_outputs" / "bound_outcomes.csv")
    feature_columns = safe_feature_columns(promoted)
    frame = prepare_frame()
    exact_observed = unique_exact_inputs(frame)
    support = unique_physical_support(frame)
    geometry = build_support_geometry(support)
    ranges = parameter_ranges(frame)
    metrics, fitted_models, prediction_frames = fit_models(frame, feature_columns)

    pair_outputs: list[pd.DataFrame] = []
    created_plots: list[Path] = []
    for spec in PAIR_SPECS:
        anchor = choose_pair_anchor(frame, spec)
        x_values = observed_axis_values(frame, spec.x_col)
        y_values = observed_axis_values(frame, spec.y_col)
        grid = build_anchor_points(anchor, spec.x_col, spec.y_col, x_values, y_values)
        anchor_mask = pd.Series(True, index=frame.index)
        for column in INPUT_COLUMNS:
            if column in {spec.x_col, spec.y_col}:
                continue
            value = anchor[column]
            if pd.isna(value):
                anchor_mask &= frame[column].isna()
            else:
                anchor_mask &= frame[column] == value
        anchor_slice = frame.loc[anchor_mask].copy()
        grid["is_exact_observed"] = exact_combo_exists(grid, exact_observed)
        grid = classify_points(grid, support, geometry, ranges, spec.x_col, spec.y_col, anchor_slice)
        grid = predict_with_model_family(grid, feature_columns, fitted_models)
        grid["local_prediction_variability"] = compute_local_prediction_variability(grid, feature_columns, fitted_models["gradient_boosting"], frame)
        grid = assign_trust(grid)
        grid = add_source_metadata(grid, "pair_grid", anchor, spec.name)
        pair_outputs.append(grid)
        if spec.name in {"mass_periapsis", "mass_velocity", "periapsis_velocity"}:
            created_plots.extend(create_pair_maps(grid, spec, anchor, frame))

    sweep_points, sweep_plots, smoothness = create_one_parameter_sweeps(
        frame, feature_columns, fitted_models["gradient_boosting"], support, geometry, ranges
    )
    sweep_points["local_prediction_variability"] = compute_local_prediction_variability(
        sweep_points, feature_columns, fitted_models["gradient_boosting"], frame
    )
    sweep_points = assign_trust(sweep_points)
    created_plots.extend(sweep_plots)

    mass195, mass195_plots = mass_19_5_unsimulated_cases(frame, support, geometry, ranges, feature_columns, fitted_models)
    created_plots.extend(mass195_plots)

    all_unsimulated = pd.concat(pair_outputs + [sweep_points, mass195], ignore_index=True)
    all_unsimulated = all_unsimulated[~all_unsimulated["is_exact_observed"]].copy()
    keep_cols = EXACT_INPUT_COLUMNS + [
        "source",
        "pair_name",
        "anchor_case",
        "predicted_bmf",
        "prediction_mean",
        "prediction_min",
        "prediction_max",
        "prediction_std",
        "prediction_range",
        "interpolation_class",
        "nearest_distance",
        "distance_k3_mean",
        "distance_k5_mean",
        "local_support_count",
        "nearest_supporting_physical_files",
        "inside_parameter_ranges",
        "inside_convex_hull",
        "near_zero_boundary",
        "local_prediction_variability",
        "trust_level",
        "pred_gradient_boosting",
        "pred_random_forest",
        "pred_ridge",
    ]
    all_unsimulated[keep_cols].to_csv(UNSIMULATED_CSV, index=False)

    recommended = select_recommended_runs(all_unsimulated)
    write_report(metrics, pair_outputs, sweep_points, smoothness, mass195, recommended, created_plots)


if __name__ == "__main__":
    main()
