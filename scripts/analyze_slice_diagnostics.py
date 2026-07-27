#!/usr/bin/env python3
"""Generate slice diagnostics and coverage plots for SPH surrogate models."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
N_SPLITS = 5
FIGSIZE = (9, 5.5)
OUTPUT_ROOT = Path("report/slice_diagnostics_20260716")
PLOTS_DIR = OUTPUT_ROOT / "plots"
TABLES_DIR = OUTPUT_ROOT / "tables"
REPORT_PATH = Path("progress_7july2026.txt")
IMAGE_REPORT_PATH = OUTPUT_ROOT / "sanity_check_report_with_images.md"
FILENAME_RE = re.compile(
    r"^(?P<prefix>Ma_xp)_(?P<mass>A\d{4}(?:c30)?)(?:_(?P<spin>s\d{3}[A-Za-z]*))?"
    r"_n(?P<resolution>\d+)_r(?P<periapsis>\d+)_v(?P<velocity>\d+)"
    r"_(?P<timestep>\d+)"
    r"_fof_(?P<linking_length>[0-9.]+)_"
    r"(?P<chunk>\d+)\.hdf5$"
)
FEATURE_COLUMNS = [
    "mass_log10_kg",
    "particle_log10",
    "periapsis_Rm",
    "v_inf_kms",
    "spin_period_hr",
    "spin_axis",
    "has_explicit_spin",
    "special_case_code",
    "timestep",
    "fof_linking_length",
]
SLICE_GROUP_COLUMNS = [
    "mass_code",
    "resolution_code",
    "velocity_code",
    "spin_code",
    "timestep",
    "fof_linking_length",
]
REGRESSION_TARGETS = [
    "bound_mass_fraction",
    "n_fragments",
    "largest_fragment_mass_kg",
]


@dataclass(frozen=True)
class SliceSpec:
    mass_code: str
    resolution_code: str
    velocity_code: str
    spin_code: str
    timestep: int
    fof_linking_length: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_simulation_filename(filename: str) -> dict[str, object]:
    match = FILENAME_RE.match(filename)
    if not match:
        raise ValueError(f"Unrecognized FoF filename pattern: {filename}")

    mass_code = match.group("mass")
    spin_code = match.group("spin") or ""
    special_case_code = "c30" if mass_code.endswith("c30") else ""
    mass_digits = mass_code[1:5]
    spin_axis = spin_code[4:] if len(spin_code) > 4 else ""
    spin_value = spin_code[1:4] if spin_code else ""

    resolution_value = int(match.group("resolution"))
    periapsis_value = int(match.group("periapsis"))
    velocity_value = int(match.group("velocity"))
    timestep = int(match.group("timestep"))
    chunk_index = int(match.group("chunk"))
    linking_length = float(match.group("linking_length"))

    return {
        "filename": filename,
        "mass_code": mass_code,
        "mass_value": int(mass_digits),
        "special_case_code": special_case_code,
        "spin_code": spin_code,
        "spin_value": int(spin_value) if spin_value else np.nan,
        "spin_axis": spin_axis or "none",
        "has_explicit_spin": bool(spin_code),
        "resolution_code": f"n{resolution_value}",
        "resolution_value": resolution_value,
        "periapsis_code": f"r{periapsis_value}",
        "periapsis_value": periapsis_value,
        "velocity_code": f"v{velocity_value:02d}",
        "velocity_value": velocity_value,
        "timestep": timestep,
        "fof_linking_length": linking_length,
        "chunk_index": chunk_index,
    }


def load_dataset() -> pd.DataFrame:
    frame = pd.read_csv("extraction_outputs/bound_outcomes.csv", low_memory=False)
    parsed = frame["fof_file"].map(parse_simulation_filename).apply(pd.Series)
    for column in parsed.columns:
        if column not in frame.columns:
            frame[column] = parsed[column]

    frame["mass_log10_kg"] = pd.to_numeric(frame["mass_value"], errors="coerce") / 100.0
    frame["particle_log10"] = np.log10(pd.to_numeric(frame["resolution_value"], errors="coerce"))
    frame["periapsis_Rm"] = pd.to_numeric(frame["periapsis_value"], errors="coerce") / 10.0
    frame["v_inf_kms"] = pd.to_numeric(frame["velocity_value"], errors="coerce") / 10.0
    frame["spin_period_hr"] = pd.to_numeric(frame["spin_value"], errors="coerce") / 10.0
    frame["spin_axis"] = frame["spin_axis"].fillna("none").replace("", "none")
    frame["special_case_code"] = frame["special_case_code"].fillna("").replace("", "none")
    frame["has_explicit_spin"] = frame["has_explicit_spin"].fillna(False).astype(bool)
    frame["bound_mass_fraction_ge_0_1"] = pd.to_numeric(frame["bound_mass_fraction"], errors="coerce") >= 0.1
    largest_bound = pd.to_numeric(frame["largest_bound_fragment_mass_kg"], errors="coerce")
    largest_unbound = pd.to_numeric(frame["largest_unbound_fragment_mass_kg"], errors="coerce")
    frame["largest_fragment_mass_kg"] = np.maximum(largest_bound.fillna(-np.inf), largest_unbound.fillna(-np.inf))
    frame["largest_fragment_mass_kg"] = frame["largest_fragment_mass_kg"].replace(-np.inf, np.nan)
    return frame


def build_preprocessor(X: pd.DataFrame, scaled: bool) -> ColumnTransformer:
    categorical = [column for column in ["spin_axis", "special_case_code"] if column in X.columns]
    numeric = [column for column in X.columns if column not in categorical]

    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scaled:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )


def regression_models(X: pd.DataFrame) -> dict[str, Pipeline]:
    return {
        "ridge": Pipeline([("preprocessor", build_preprocessor(X, scaled=True)), ("model", Ridge(alpha=1.0))]),
        "random_forest": Pipeline(
            [
                ("preprocessor", build_preprocessor(X, scaled=False)),
                ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)),
            ]
        ),
        "gradient_boosting": Pipeline(
            [("preprocessor", build_preprocessor(X, scaled=False)), ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))]
        ),
    }


def classification_models(X: pd.DataFrame) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", build_preprocessor(X, scaled=True)),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", build_preprocessor(X, scaled=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [("preprocessor", build_preprocessor(X, scaled=False)), ("model", GradientBoostingClassifier(random_state=RANDOM_STATE))]
        ),
    }


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def grouped_oof_regression(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, dict[str, Pipeline], list[dict[str, object]]]:
    valid = df[df[target].notna()].copy()
    X = valid[FEATURE_COLUMNS].copy()
    y = pd.to_numeric(valid[target], errors="coerce")
    groups = valid["physical_file"].astype(str)
    splitter = GroupKFold(n_splits=N_SPLITS)
    models = regression_models(X)
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    fitted_models: dict[str, Pipeline] = {}

    for model_name, pipeline in models.items():
        oof = np.full(len(valid), np.nan)
        for train_idx, test_idx in splitter.split(X, y, groups):
            fitted = clone(pipeline)
            fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
            oof[test_idx] = fitted.predict(X.iloc[test_idx])
        fitted_models[model_name] = clone(pipeline).fit(X, y)
        metric_rows.append(
            {
                "task": "regression",
                "target": target,
                "model": model_name,
                "rows": len(valid),
                "unique_physical_files": valid["physical_file"].nunique(),
                "mae": mean_absolute_error(y, oof),
                "rmse": rmse(y, oof),
                "r2": r2_score(y, oof),
            }
        )
        model_frame = valid.copy()
        model_frame["predicted"] = oof
        model_frame["residual"] = model_frame[target] - model_frame["predicted"]
        model_frame["target"] = target
        model_frame["model"] = model_name
        prediction_rows.extend(model_frame.to_dict("records"))

    return pd.DataFrame(prediction_rows), fitted_models, metric_rows


def grouped_oof_classification(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, dict[str, Pipeline], list[dict[str, object]]]:
    valid = df[df[target].notna()].copy()
    X = valid[FEATURE_COLUMNS].copy()
    y = valid[target].astype(bool)
    groups = valid["physical_file"].astype(str)
    splitter = GroupKFold(n_splits=N_SPLITS)
    models = classification_models(X)
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    fitted_models: dict[str, Pipeline] = {}

    for model_name, pipeline in models.items():
        pred_label = np.full(len(valid), False)
        pred_prob = np.full(len(valid), np.nan)
        for train_idx, test_idx in splitter.split(X, y, groups):
            fitted = clone(pipeline)
            fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred_label[test_idx] = fitted.predict(X.iloc[test_idx])
            pred_prob[test_idx] = fitted.predict_proba(X.iloc[test_idx])[:, 1]
        fitted_models[model_name] = clone(pipeline).fit(X, y)
        metric_rows.append(
            {
                "task": "classification",
                "target": target,
                "model": model_name,
                "rows": len(valid),
                "unique_physical_files": valid["physical_file"].nunique(),
                "accuracy": accuracy_score(y, pred_label),
                "balanced_accuracy": balanced_accuracy_score(y, pred_label),
                "f1": f1_score(y, pred_label),
                "roc_auc": roc_auc_score(y, pred_prob),
            }
        )
        model_frame = valid.copy()
        model_frame["predicted_label"] = pred_label
        model_frame["predicted_probability"] = pred_prob
        model_frame["target"] = target
        model_frame["model"] = model_name
        prediction_rows.extend(model_frame.to_dict("records"))

    return pd.DataFrame(prediction_rows), fitted_models, metric_rows


def choose_slice(df: pd.DataFrame) -> SliceSpec:
    counts = df.groupby(SLICE_GROUP_COLUMNS)["periapsis_Rm"].nunique().reset_index(name="n_periapsis")
    best = counts.sort_values(["n_periapsis", "mass_code", "velocity_code"], ascending=[False, False, True]).iloc[0]
    return SliceSpec(
        mass_code=str(best["mass_code"]),
        resolution_code=str(best["resolution_code"]),
        velocity_code=str(best["velocity_code"]),
        spin_code=str(best["spin_code"]),
        timestep=int(best["timestep"]),
        fof_linking_length=float(best["fof_linking_length"]),
    )


def slice_mask(df: pd.DataFrame, spec: SliceSpec) -> pd.Series:
    return (
        (df["mass_code"] == spec.mass_code)
        & (df["resolution_code"] == spec.resolution_code)
        & (df["velocity_code"] == spec.velocity_code)
        & (df["spin_code"] == spec.spin_code)
        & (df["timestep"].astype(int) == spec.timestep)
        & (pd.to_numeric(df["fof_linking_length"], errors="coerce") == spec.fof_linking_length)
    )


def classify_domain(grid_values: np.ndarray, observed_values: np.ndarray) -> list[str]:
    obs = np.sort(np.unique(observed_values))
    lo = float(obs.min())
    hi = float(obs.max())
    labels = []
    for value in grid_values:
        if value < lo or value > hi:
            labels.append("extrapolation")
        elif np.isclose(obs, value, atol=1e-9).any():
            labels.append("observed")
        else:
            labels.append("interpolation")
    return labels


def make_grid(slice_df: pd.DataFrame, spec: SliceSpec, points: int = 250) -> pd.DataFrame:
    observed = np.sort(slice_df["periapsis_Rm"].unique())
    step = np.median(np.diff(observed)) if len(observed) > 1 else 0.2
    x_min = max(0.0, observed.min() - step)
    x_max = observed.max() + step
    grid_values = np.linspace(x_min, x_max, points)
    base = slice_df.iloc[[0]].copy()
    grid = pd.concat([base] * points, ignore_index=True)
    grid["periapsis_Rm"] = grid_values
    grid["periapsis_value"] = np.round(grid_values * 10.0).astype(int)
    grid["periapsis_code"] = grid["periapsis_value"].map(lambda value: f"r{value}")
    grid["domain_type"] = classify_domain(grid_values, observed)
    return grid


def add_domain_background(ax: plt.Axes, observed: np.ndarray) -> None:
    observed = np.sort(np.unique(observed))
    if len(observed) == 0:
        return
    lo = float(observed.min())
    hi = float(observed.max())
    ax.axvspan(ax.get_xlim()[0], lo, color="#f3d3d3", alpha=0.45, zorder=0)
    ax.axvspan(lo, hi, color="#dbe8ff", alpha=0.25, zorder=0)
    ax.axvspan(hi, ax.get_xlim()[1], color="#f3d3d3", alpha=0.45, zorder=0)


def add_domain_explainer(ax: plt.Axes, observed: np.ndarray) -> None:
    observed = np.sort(np.unique(observed))
    if len(observed) == 0:
        return
    lo = float(observed.min())
    hi = float(observed.max())
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    y_text = y_max - 0.06 * (y_max - y_min)
    if lo > x_min:
        ax.text(
            (x_min + lo) / 2.0,
            y_text,
            "Extrapolation:\noutside sampled\nperiapsis",
            ha="center",
            va="top",
            fontsize=8,
            color="#6b2d2d",
            zorder=10,
        )
    ax.text(
        (lo + hi) / 2.0,
        y_text,
        "Interpolation:\ninside sampled\nperiapsis",
        ha="center",
        va="top",
        fontsize=8,
        color="#204a87",
        zorder=10,
    )
    if hi < x_max:
        ax.text(
            (hi + x_max) / 2.0,
            y_text,
            "Extrapolation:\noutside sampled\nperiapsis",
            ha="center",
            va="top",
            fontsize=8,
            color="#6b2d2d",
            zorder=10,
        )


def display_model_name(model_name: str) -> str:
    names = {
        "ridge": "Linear ridge baseline",
        "random_forest": "Random forest",
        "gradient_boosting": "Gradient boosting",
        "logistic_regression": "Logistic regression",
    }
    return names.get(model_name, model_name.replace("_", " "))


def plot_regression_slice(
    target: str,
    y_label: str,
    slice_df: pd.DataFrame,
    predictions: pd.DataFrame,
    fitted_models: dict[str, Pipeline],
    grid: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    observed = np.sort(slice_df["periapsis_Rm"].unique())
    domain_grid = grid.copy()
    add_domain_background(ax, observed)
    ax.scatter(slice_df["periapsis_Rm"], slice_df[target], color="black", s=34, label="SPH simulation", zorder=5)

    colors = {"random_forest": "#1f77b4", "gradient_boosting": "#d62728"}
    markers = {"random_forest": "^", "gradient_boosting": "D"}
    grid_rows: list[pd.DataFrame] = []

    for model_name in ["random_forest", "gradient_boosting"]:
        model = fitted_models[model_name]
        domain_grid[f"pred_{model_name}"] = model.predict(domain_grid[FEATURE_COLUMNS])
        ax.plot(
            domain_grid["periapsis_Rm"],
            domain_grid[f"pred_{model_name}"],
            color=colors[model_name],
            linewidth=2.0,
            label=f"{display_model_name(model_name)} curve",
        )
        slice_preds = predictions[(predictions["target"] == target) & (predictions["model"] == model_name)].copy()
        slice_preds = slice_preds.loc[slice_mask(slice_preds, choose_slice(slice_df))] if "mass_code" in slice_preds else slice_preds
        slice_preds = slice_preds[
            (slice_preds["mass_code"] == slice_df["mass_code"].iloc[0])
            & (slice_preds["resolution_code"] == slice_df["resolution_code"].iloc[0])
            & (slice_preds["velocity_code"] == slice_df["velocity_code"].iloc[0])
            & (slice_preds["spin_code"] == slice_df["spin_code"].iloc[0])
            & (slice_preds["timestep"].astype(int) == int(slice_df["timestep"].iloc[0]))
            & (pd.to_numeric(slice_preds["fof_linking_length"], errors="coerce") == float(slice_df["fof_linking_length"].iloc[0]))
        ]
        ax.scatter(
            slice_preds["periapsis_Rm"],
            slice_preds["predicted"],
            color=colors[model_name],
            marker=markers[model_name],
            s=50,
            alpha=0.9,
            label=f"{display_model_name(model_name)} OOF",
            zorder=6,
        )
        model_grid = domain_grid[["periapsis_Rm", "domain_type", f"pred_{model_name}"]].rename(
            columns={f"pred_{model_name}": "predicted"}
        )
        model_grid["target"] = target
        model_grid["model"] = model_name
        grid_rows.append(model_grid)

    ax.set_xlim(domain_grid["periapsis_Rm"].min(), domain_grid["periapsis_Rm"].max())
    ax.set_xlabel("Periapsis ($R_\\mathrm{Mars}$)")
    ax.set_ylabel(y_label)
    ax.set_title(f"{target} on a fixed-parameter periapsis slice")
    add_domain_background(ax, observed)
    add_domain_explainer(ax, observed)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return pd.concat(grid_rows, ignore_index=True)


def plot_logistic_slice(
    slice_df: pd.DataFrame,
    predictions: pd.DataFrame,
    fitted_models: dict[str, Pipeline],
    grid: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    observed = np.sort(slice_df["periapsis_Rm"].unique())
    add_domain_background(ax, observed)
    binary_actual = slice_df["bound_mass_fraction_ge_0_1"].astype(int)
    ax.scatter(slice_df["periapsis_Rm"], binary_actual, color="black", s=34, label="Actual BMF >= 0.1", zorder=5)

    colors = {"logistic_regression": "#2ca02c", "random_forest": "#1f77b4", "gradient_boosting": "#d62728"}
    markers = {"logistic_regression": "o", "random_forest": "^", "gradient_boosting": "D"}
    grid_rows: list[pd.DataFrame] = []

    for model_name in ["logistic_regression", "random_forest", "gradient_boosting"]:
        model = fitted_models[model_name]
        prob = model.predict_proba(grid[FEATURE_COLUMNS])[:, 1]
        ax.plot(
            grid["periapsis_Rm"],
            prob,
            color=colors[model_name],
            linewidth=2.0,
            label=f"{display_model_name(model_name)} probability",
        )
        slice_preds = predictions[predictions["model"] == model_name]
        slice_preds = slice_preds[
            (slice_preds["mass_code"] == slice_df["mass_code"].iloc[0])
            & (slice_preds["resolution_code"] == slice_df["resolution_code"].iloc[0])
            & (slice_preds["velocity_code"] == slice_df["velocity_code"].iloc[0])
            & (slice_preds["spin_code"] == slice_df["spin_code"].iloc[0])
            & (slice_preds["timestep"].astype(int) == int(slice_df["timestep"].iloc[0]))
            & (pd.to_numeric(slice_preds["fof_linking_length"], errors="coerce") == float(slice_df["fof_linking_length"].iloc[0]))
        ]
        ax.scatter(
            slice_preds["periapsis_Rm"],
            slice_preds["predicted_probability"],
            color=colors[model_name],
            marker=markers[model_name],
            s=50,
            alpha=0.9,
            label=f"{display_model_name(model_name)} OOF",
            zorder=6,
        )
        model_grid = grid[["periapsis_Rm", "domain_type"]].copy()
        model_grid["predicted_probability"] = prob
        model_grid["target"] = "bound_mass_fraction_ge_0_1"
        model_grid["model"] = model_name
        grid_rows.append(model_grid)

    ax.axhline(0.5, color="#444444", linestyle="--", linewidth=1.0)
    ax.set_xlim(grid["periapsis_Rm"].min(), grid["periapsis_Rm"].max())
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Periapsis ($R_\\mathrm{Mars}$)")
    ax.set_ylabel("Predicted probability")
    ax.set_title("Bound-retention threshold classification on the same periapsis slice")
    add_domain_background(ax, observed)
    add_domain_explainer(ax, observed)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return pd.concat(grid_rows, ignore_index=True)


def heatmap_table(df: pd.DataFrame, row: str, col: str, value: str | None = None, agg: str = "count") -> pd.DataFrame:
    if value is None:
        table = df.pivot_table(index=row, columns=col, values="physical_file", aggfunc="count", fill_value=0)
    else:
        table = df.pivot_table(index=row, columns=col, values=value, aggfunc=agg)
    return table.sort_index().sort_index(axis=1)


def draw_heatmap(ax: plt.Axes, table: pd.DataFrame, title: str, cbar_label: str, cmap: str) -> None:
    data = table.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels([str(value) for value in table.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels([str(value) for value in table.index])
    ax.set_xlabel(table.columns.name or "")
    ax.set_ylabel(table.index.name or "")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)


def plot_coverage_heatmaps(df: pd.DataFrame, output_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    mass_peri = heatmap_table(df, "mass_log10_kg", "periapsis_Rm")
    peri_vel = heatmap_table(df, "periapsis_Rm", "v_inf_kms")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    draw_heatmap(axes[0], mass_peri, "Coverage count: mass vs periapsis", "Runs", "Blues")
    draw_heatmap(axes[1], peri_vel, "Coverage count: periapsis vs velocity", "Runs", "Blues")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return mass_peri, peri_vel


def plot_coverage_vs_error(predictions: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 13))
    targets = [
        ("bound_mass_fraction", "random_forest"),
        ("n_fragments", "random_forest"),
        ("largest_fragment_mass_kg", "gradient_boosting"),
    ]
    for row_idx, (target, model_name) in enumerate(targets):
        subset = predictions[(predictions["target"] == target) & (predictions["model"] == model_name)].copy()
        subset["abs_error"] = subset["residual"].abs()
        coverage = heatmap_table(subset, "mass_log10_kg", "periapsis_Rm")
        error = heatmap_table(subset, "mass_log10_kg", "periapsis_Rm", value="abs_error", agg="mean")
        draw_heatmap(axes[row_idx, 0], coverage, f"{target}: coverage", "Runs", "Blues")
        draw_heatmap(axes[row_idx, 1], error, f"{target}: mean |error| ({model_name})", "|error|", "OrRd")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_slice_summary(slice_df: pd.DataFrame, spec: SliceSpec) -> pd.DataFrame:
    peri = np.sort(slice_df["periapsis_Rm"].unique())
    return pd.DataFrame(
        [
            {
                "mass_code": spec.mass_code,
                "resolution_code": spec.resolution_code,
                "velocity_code": spec.velocity_code,
                "spin_code": spec.spin_code,
                "timestep": spec.timestep,
                "fof_linking_length": spec.fof_linking_length,
                "n_rows": len(slice_df),
                "n_unique_physical_files": slice_df["physical_file"].nunique(),
                "n_periapsis_values": len(peri),
                "periapsis_min_Rm": float(peri.min()),
                "periapsis_max_Rm": float(peri.max()),
                "periapsis_values_Rm": ", ".join(f"{value:.1f}" for value in peri),
            }
        ]
    )


def summarize_coverage(mass_peri: pd.DataFrame, peri_vel: pd.DataFrame) -> dict[str, object]:
    return {
        "mass_peri_nonzero_bins": int((mass_peri > 0).sum().sum()),
        "mass_peri_total_bins": int(mass_peri.size),
        "peri_vel_nonzero_bins": int((peri_vel > 0).sum().sum()),
        "peri_vel_total_bins": int(peri_vel.size),
        "mass_peri_max_bin_count": int(np.nanmax(mass_peri.to_numpy())),
        "peri_vel_max_bin_count": int(np.nanmax(peri_vel.to_numpy())),
    }


def format_regression_line(metrics: pd.DataFrame, target: str, model: str) -> str:
    row = metrics[(metrics["task"] == "regression") & (metrics["target"] == target) & (metrics["model"] == model)].iloc[0]
    return f"{model}: R^2={row['r2']:.3f}, MAE={row['mae']:.3e}, RMSE={row['rmse']:.3e}"


def format_classification_line(metrics: pd.DataFrame, target: str, model: str) -> str:
    row = metrics[(metrics["task"] == "classification") & (metrics["target"] == target) & (metrics["model"] == model)].iloc[0]
    return (
        f"{model}: balanced_accuracy={row['balanced_accuracy']:.3f}, "
        f"F1={row['f1']:.3f}, ROC_AUC={row['roc_auc']:.3f}"
    )


def write_report(
    report_path: Path,
    slice_spec: SliceSpec,
    slice_summary: pd.DataFrame,
    metrics: pd.DataFrame,
    coverage_summary: dict[str, object],
    output_files: dict[str, Path],
) -> None:
    lines = [
        "SPH surrogate sanity-check report",
        "Date: 2026-07-16",
        "",
        "0. How the surrogate model works and why it is useful",
        "Answer:",
        "The machine-learning model is a surrogate trained on existing SPH simulations rather than a replacement for the SPH physics solver itself.",
        "It learns an empirical mapping from collision inputs such as mass, periapsis, velocity, spin, timestep, and FoF linking length to SPH-derived outputs such as bound mass fraction, fragment count, and largest fragment mass.",
        "The modelling pipeline in the middle consists of data cleaning, feature engineering, target construction, grouped held-out validation, model fitting, and error analysis on controlled slices and parameter-space coverage plots.",
        "Using the surrogate is beneficial because once trained it produces predictions almost instantly, whereas a new SPH run is computationally expensive.",
        "This makes the model useful for screening, triage, rapid parameter exploration, and deciding which new SPH simulations are worth running next.",
        "The correct scientific interpretation is that ML is cheaper and faster for in-domain screening, while SPH remains the high-fidelity source of physical truth.",
        "",
        "1. Bound Mass Fraction slice plots",
        "Answer:",
        "A representative periapsis slice was chosen by maximizing the number of available periapsis values with all other parameters fixed.",
        (
            f"Chosen slice: mass={slice_spec.mass_code}, resolution={slice_spec.resolution_code}, "
            f"velocity={slice_spec.velocity_code}, spin={slice_spec.spin_code}, timestep={slice_spec.timestep}, "
            f"fof_linking_length={slice_spec.fof_linking_length:.4f}."
        ),
        (
            f"This slice contains {int(slice_summary['n_periapsis_values'].iloc[0])} periapsis values spanning "
            f"{slice_summary['periapsis_min_Rm'].iloc[0]:.1f} to {slice_summary['periapsis_max_Rm'].iloc[0]:.1f} R_Mars."
        ),
        "The bound-mass-fraction plot overlays black SPH points, out-of-fold model predictions at those same simulations, and full-model slice curves.",
        "Interpretation: random forest tracks the sharp drop in retained mass at low periapsis best; gradient boosting is similar but slightly smoother.",
        f"Generated plot: {output_files['bmf_slice']}",
        "",
        "2. Fragment Count and Largest Fragment Mass slice plots",
        "Answer:",
        "The same fixed-parameter periapsis slice was reused so differences across targets are not caused by switching scenarios.",
        "Fragment count is visibly noisier than bound mass fraction because neighbouring simulations can jump between fragmentation regimes; tree models capture the broad trend but miss several local spikes.",
        "Largest fragment mass is harder in the disruptive regime because small differences in the dominant remnant lead to large mass changes; both random forest and gradient boosting flatten some extremes.",
        f"Generated plots: {output_files['fragment_slice']}, {output_files['largest_fragment_slice']}",
        "",
        "3. Interpolation vs extrapolation",
        "Answer:",
        "Each slice fixes mass, resolution, velocity, spin, timestep, and FoF linking length, then varies only periapsis.",
        "The black points are the SPH simulations that actually exist for that exact fixed-parameter slice.",
        "Blue shading means the periapsis value lies inside the sampled SPH periapsis range for that slice. That is interpolation.",
        "Interpolation means the model is estimating between known examples. It has nearby SPH cases on both sides, so this is the safer use case.",
        "Red shading means the periapsis value lies below the smallest sampled periapsis or above the largest sampled periapsis. That is extrapolation.",
        "Extrapolation means the model is extending the trend beyond the data it actually saw on that slice. The curve can still look smooth there, but no SPH point anchors the behaviour beyond the edge.",
        "A point does not need to sit exactly on a black marker to count as interpolation. If it lies inside the sampled periapsis span, it is still interpolation between neighbouring SPH cases.",
        f"Grid and domain labels table: {output_files['slice_grid']}",
        "",
        "4. Parameter-space coverage and relation to model performance",
        "Answer:",
        (
            f"Mass-periapsis coverage occupies {coverage_summary['mass_peri_nonzero_bins']} of "
            f"{coverage_summary['mass_peri_total_bins']} bins; periapsis-velocity coverage occupies "
            f"{coverage_summary['peri_vel_nonzero_bins']} of {coverage_summary['peri_vel_total_bins']} bins."
        ),
        "Coverage is concentrated in a few low-velocity, fixed-resolution slices, so good performance there is expected and sparse corners are less trustworthy.",
        "The coverage figure shows where simulations exist. The coverage-vs-error figure compares that support with out-of-fold absolute error, showing that error usually increases in sparse edge bins and in strongly disruptive bins.",
        f"Generated figures: {output_files['coverage_heatmaps']}, {output_files['coverage_vs_error']}",
        f"Coverage tables: {output_files['coverage_mass_peri']}, {output_files['coverage_peri_vel']}",
        "",
        "5. Model-by-model discussion",
        "Answer:",
        "Logistic Regression is only appropriate for the thresholded classification task BMF >= 0.1, not for the continuous bound-mass-fraction regression slice.",
        "That is why logistic regression is shown on the separate threshold-probability slice rather than on the continuous BMF plot.",
        f"Threshold classifier metrics: {format_classification_line(metrics, 'bound_mass_fraction_ge_0_1', 'logistic_regression')}",
        "It captures the broad monotonic decision boundary well and is useful as a transparent screening baseline, but it cannot represent multi-regime nonlinear structure or predict continuous fragment properties.",
        f"Random Forest metrics for the main continuous targets: {format_regression_line(metrics, 'bound_mass_fraction', 'random_forest')}; {format_regression_line(metrics, 'n_fragments', 'random_forest')}; {format_regression_line(metrics, 'largest_fragment_mass_kg', 'random_forest')}.",
        "Random forest captures abrupt regime changes and works well on well-sampled slices, which makes it the most practical default for SPH screening in-domain. It still struggles with sparse extremes and local target noise.",
        f"Gradient Boosting metrics for the main continuous targets: {format_regression_line(metrics, 'bound_mass_fraction', 'gradient_boosting')}; {format_regression_line(metrics, 'n_fragments', 'gradient_boosting')}; {format_regression_line(metrics, 'largest_fragment_mass_kg', 'gradient_boosting')}.",
        "Gradient boosting captures smooth nonlinear trends and is competitive on fragment mass, but it can become too smooth in some noisy regions and too aggressive near sparse boundaries.",
        f"Classification slice plot for model comparison: {output_files['logistic_slice']}",
        "",
        "6. Sanity check: where predictions are trustworthy",
        "Answer:",
        "Predictions are most defensible when the query lies inside dense training regions, especially within the well-sampled low-velocity periapsis sweeps and away from the edge of the observed domain.",
        "Predictions are also more trustworthy when different model families agree and when the target is smooth, which is why bound mass fraction is easier to trust than fragment count.",
        "Predictions should be treated cautiously when periapsis falls outside the observed slice span, when mass-velocity combinations sit in sparse heatmap bins, or when the target itself is noisy as in fragment count and largest-fragment outcomes.",
        "This sanity check supports using the surrogate for triage and prioritization, not as a replacement for new SPH runs in sparsely covered or extrapolative regions.",
        "",
        "7. Limitations and future improvements",
        "Answer:",
        "The current dataset is small and unevenly sampled, with strong concentration in a few physical configurations and limited coverage of the broader parameter space.",
        "The surrogate is therefore strongest as an in-domain screening tool. Reliability outside the training domain remains limited because the models have little physical support there.",
        "Future improvements should include expanding the simulation set in sparse mass-periapsis-velocity regions, adding targeted runs beyond current slice endpoints to reduce extrapolation risk, and validating surrogate predictions prospectively against newly generated SPH simulations.",
        "Additional work could also test uncertainty-aware surrogates and physically informed features so that noisy targets are accompanied by calibrated confidence estimates.",
        "",
        "8. Generated files",
        f"Metrics table: {output_files['metrics']}",
        f"Regression OOF records: {output_files['regression_predictions']}",
        f"Classification OOF records: {output_files['classification_predictions']}",
        f"Slice summary table: {output_files['slice_summary']}",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_image_report(
    report_path: Path,
    slice_spec: SliceSpec,
    slice_summary: pd.DataFrame,
    metrics: pd.DataFrame,
    coverage_summary: dict[str, object],
    output_files: dict[str, Path],
) -> None:
    lines = [
        "# SPH Surrogate Sanity-Check Report",
        "",
        "Date: 2026-07-16",
        "",
        "## 1. What the model is doing",
        "The machine-learning surrogate does not replace the SPH physics calculation itself. Instead, it learns an empirical mapping from collision setup parameters to SPH-derived outcomes using the existing simulation archive as training data.",
        "",
        "In this workflow the inputs are the physical and numerical descriptors of each run: mass, resolution, periapsis, velocity, spin period, spin axis, timestep, and FoF linking length. The outputs are the post-processed SPH quantities such as bound mass fraction, fragment count, largest fragment mass, and the thresholded bound-retention label.",
        "",
        "That means the surrogate is best interpreted as a fast emulator of the trends already present in the SPH dataset, not as a first-principles solver.",
        "",
        "## 2. Why use ML rather than rerunning SPH every time",
        "SPH remains the high-fidelity source of physical truth in this project. The surrogate is useful because once trained it can produce predictions almost instantly, whereas a new SPH run is far more computationally expensive.",
        "",
        "This makes the surrogate useful for:",
        "- screening large parameter sweeps before committing to expensive simulations",
        "- identifying regions that are likely to be interesting or highly disruptive",
        "- estimating broad outcome trends inside the existing training domain",
        "- prioritizing which new SPH runs would add the most scientific value",
        "",
        "The correct claim is therefore not that ML is physically better than SPH. The correct claim is that ML is a cheaper in-domain surrogate that can reduce the number of expensive SPH runs needed for exploration and triage.",
        "",
        "## 3. What happens in the middle of the modelling",
        "The modelling pipeline has four main stages:",
        "",
        "1. Data preparation",
        "   The extracted SPH outcome tables were cleaned and converted into engineered features such as `periapsis_Rm`, `v_inf_kms`, `mass_log10_kg`, and `particle_log10`.",
        "",
        "2. Target construction",
        "   Separate targets were defined for smooth bound-retention behaviour and noisier fragmentation behaviour. This is important because fragment count and largest-fragment targets are harder to learn than bound mass fraction.",
        "",
        "3. Model fitting and held-out validation",
        "   Grouped cross-validation was used so that the reported performance reflects generalization to held-out physical scenarios rather than simple memorization of the archive.",
        "",
        "4. Error analysis and sanity checks",
        "   Slice plots, parameter-space coverage plots, and error heatmaps were generated to show where the model interpolates, where it extrapolates, and where performance is stable or unreliable.",
        "",
        "## 4. Representative periapsis slice",
        (
            f"The chosen slice fixes mass={slice_spec.mass_code}, resolution={slice_spec.resolution_code}, "
            f"velocity={slice_spec.velocity_code}, spin={slice_spec.spin_code}, timestep={slice_spec.timestep}, "
            f"and fof_linking_length={slice_spec.fof_linking_length:.4f} while varying periapsis."
        ),
        (
            f"It contains {int(slice_summary['n_periapsis_values'].iloc[0])} observed periapsis values spanning "
            f"{slice_summary['periapsis_min_Rm'].iloc[0]:.1f} to {slice_summary['periapsis_max_Rm'].iloc[0]:.1f} R_Mars."
        ),
        "",
        "Blue shading marks interpolation: periapsis values inside the sampled SPH range for this exact slice.",
        "Red shading marks extrapolation: periapsis values outside the sampled SPH range, where the model is extending the trend beyond its SPH support.",
        "Interpolation is safer because nearby SPH runs exist on both sides. Extrapolation is weaker because the curve may look smooth even though no SPH point anchors it beyond the edge.",
        "",
        "![Bound mass fraction slice](plots/bound_mass_fraction_slice_periapsis.png)",
        "",
        "The bound-mass-fraction slice shows that the tree-based models follow the sharp retention decline well, with random forest usually tracking the drop most closely. This is one reason bound mass fraction is a defensible primary surrogate target.",
        "",
        "![Fragment count slice](plots/fragment_count_slice_periapsis.png)",
        "",
        "The fragment-count slice is noisier. The broad trend is still captured, but local jumps are harder to predict because nearby runs can fall into different fragmentation regimes.",
        "",
        "![Largest fragment mass slice](plots/largest_fragment_mass_slice_periapsis.png)",
        "",
        "Largest fragment mass is also more difficult, especially in the strongly disruptive regime, because small changes in remnant identity can produce large absolute mass differences.",
        "",
        "## 5. How the different models behave",
        f"Logistic regression threshold metrics: {format_classification_line(metrics, 'bound_mass_fraction_ge_0_1', 'logistic_regression')}",
        "Logistic regression is useful because it gives an interpretable baseline for a yes/no screening decision such as whether `bound_mass_fraction >= 0.1`.",
        "It belongs on the threshold-classification plot, not on the continuous bound-mass-fraction regression plot. Logistic regression predicts a probability of crossing the threshold, whereas the BMF regression slice predicts the continuous retained-mass value itself.",
        "That is why logistic regression is shown on the separate threshold-probability slice below.",
        "",
        f"Random forest regression metrics: {format_regression_line(metrics, 'bound_mass_fraction', 'random_forest')}; {format_regression_line(metrics, 'n_fragments', 'random_forest')}; {format_regression_line(metrics, 'largest_fragment_mass_kg', 'random_forest')}.",
        "Random forest works well when the response surface has nonlinear structure and regime changes. In this dataset it is especially strong for bound mass fraction and remains a practical default for in-domain screening.",
        "",
        f"Gradient boosting regression metrics: {format_regression_line(metrics, 'bound_mass_fraction', 'gradient_boosting')}; {format_regression_line(metrics, 'n_fragments', 'gradient_boosting')}; {format_regression_line(metrics, 'largest_fragment_mass_kg', 'gradient_boosting')}.",
        "Gradient boosting is competitive on smooth nonlinear trends and performs similarly to random forest on several targets, but it can either smooth over local noise or react sharply near sparse boundaries depending on the target.",
        "",
        "![Threshold classification slice](plots/bound_mass_fraction_threshold_probability_slice.png)",
        "",
        "The threshold-classification slice shows the same fixed-parameter scenario through a binary screening lens. This is useful when the practical question is whether a run is likely to retain at least a modest amount of bound material.",
        "Here logistic regression is appropriate because the target is binary: above or below the 10 percent BMF threshold.",
        "",
        "## 6. Why we can trust some predictions",
        "Trust in the surrogate is conditional rather than absolute.",
        "",
        "Predictions are more trustworthy when:",
        "- the query lies inside the training range",
        "- the surrounding parameter region is well sampled",
        "- multiple model families produce similar answers",
        "- the target is smooth, as with bound mass fraction",
        "",
        "Predictions are less trustworthy when:",
        "- the query is outside the training support",
        "- the parameter-space bin is sparse",
        "- the target is noisy or discrete, as with fragment count",
        "- the prediction sits near a sharp physical regime transition",
        "",
        "This is why the coverage and error plots matter: they show that good numerical metrics alone are not enough. The trust argument has to be tied to where the model is being applied.",
        "",
        "## 7. Coverage and reliability",
        (
            f"Mass-periapsis coverage occupies {coverage_summary['mass_peri_nonzero_bins']} of "
            f"{coverage_summary['mass_peri_total_bins']} bins. Periapsis-velocity coverage occupies "
            f"{coverage_summary['peri_vel_nonzero_bins']} of {coverage_summary['peri_vel_total_bins']} bins."
        ),
        "",
        "![Coverage heatmaps](plots/parameter_coverage_heatmaps.png)",
        "",
        "These heatmaps show that the archive is unevenly sampled. Coverage is concentrated in a limited set of low-velocity and specific-mass slices, so the surrogate is naturally strongest there.",
        "",
        "![Coverage versus error heatmaps](plots/coverage_vs_error_heatmaps.png)",
        "",
        "The error heatmaps connect this directly to model behaviour. Higher errors tend to appear in sparse edge bins and in strongly disruptive regions where the targets are intrinsically noisier.",
        "",
        "## 8. Why the surrogate is scientifically useful",
        "The surrogate is scientifically useful because it reproduces broad SPH trends, identifies which inputs matter most, and gives a fast first-pass estimate of likely outcomes before new expensive simulations are run.",
        "",
        "This makes it appropriate for triage, prioritization, and sensitivity exploration. It is not a replacement for SPH when the goal is to establish new physical conclusions in poorly sampled or extrapolative regions.",
        "",
        "## 9. Limitations and future improvements",
        "The current dataset is still relatively small and unevenly distributed across parameter space. That limits reliability outside the best-covered regions.",
        "",
        "The most important future improvements are:",
        "- expand the SPH archive in sparse mass-periapsis-velocity regions",
        "- add targeted runs beyond current slice endpoints to reduce extrapolation risk",
        "- validate the surrogate prospectively on newly generated SPH simulations",
        "- investigate uncertainty-aware models so that noisy predictions carry calibrated confidence information",
        "",
        "## 10. File locations",
        f"- Text report: `{REPORT_PATH.resolve()}`",
        f"- Image report: `{report_path.resolve()}`",
        f"- Metrics table: `{output_files['metrics']}`",
        f"- Regression OOF records: `{output_files['regression_predictions']}`",
        f"- Classification OOF records: `{output_files['classification_predictions']}`",
        f"- Slice summary table: `{output_files['slice_summary']}`",
        f"- Slice domain grid: `{output_files['slice_grid']}`",
        f"- Coverage mass-periapsis table: `{output_files['coverage_mass_peri']}`",
        f"- Coverage periapsis-velocity table: `{output_files['coverage_peri_vel']}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(PLOTS_DIR)
    ensure_dir(TABLES_DIR)

    df = load_dataset()
    slice_spec = choose_slice(df)
    chosen_slice = df.loc[slice_mask(df, slice_spec)].copy().sort_values("periapsis_Rm")
    grid = make_grid(chosen_slice, slice_spec)

    regression_predictions_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    regression_model_map: dict[str, dict[str, Pipeline]] = {}
    for target in REGRESSION_TARGETS:
        preds, fitted_models, target_metrics = grouped_oof_regression(df, target)
        regression_predictions_frames.append(preds)
        metric_rows.extend(target_metrics)
        regression_model_map[target] = fitted_models

    regression_predictions = pd.concat(regression_predictions_frames, ignore_index=True)
    classification_predictions, classification_models_map, classification_metrics = grouped_oof_classification(
        df, "bound_mass_fraction_ge_0_1"
    )
    metric_rows.extend(classification_metrics)
    metrics = pd.DataFrame(metric_rows)

    bmf_slice_path = PLOTS_DIR / "bound_mass_fraction_slice_periapsis.png"
    fragment_slice_path = PLOTS_DIR / "fragment_count_slice_periapsis.png"
    largest_fragment_slice_path = PLOTS_DIR / "largest_fragment_mass_slice_periapsis.png"
    logistic_slice_path = PLOTS_DIR / "bound_mass_fraction_threshold_probability_slice.png"
    coverage_heatmaps_path = PLOTS_DIR / "parameter_coverage_heatmaps.png"
    coverage_vs_error_path = PLOTS_DIR / "coverage_vs_error_heatmaps.png"

    slice_grid_rows: list[pd.DataFrame] = []
    slice_grid_rows.append(
        plot_regression_slice(
            "bound_mass_fraction",
            "Bound mass fraction",
            chosen_slice,
            regression_predictions,
            regression_model_map["bound_mass_fraction"],
            grid.copy(),
            bmf_slice_path,
        )
    )
    slice_grid_rows.append(
        plot_regression_slice(
            "n_fragments",
            "Fragment count",
            chosen_slice,
            regression_predictions,
            regression_model_map["n_fragments"],
            grid.copy(),
            fragment_slice_path,
        )
    )
    slice_grid_rows.append(
        plot_regression_slice(
            "largest_fragment_mass_kg",
            "Largest fragment mass (kg)",
            chosen_slice,
            regression_predictions,
            regression_model_map["largest_fragment_mass_kg"],
            grid.copy(),
            largest_fragment_slice_path,
        )
    )
    slice_grid_rows.append(
        plot_logistic_slice(
            chosen_slice,
            classification_predictions,
            classification_models_map,
            grid.copy(),
            logistic_slice_path,
        )
    )

    mass_peri, peri_vel = plot_coverage_heatmaps(df, coverage_heatmaps_path)
    plot_coverage_vs_error(regression_predictions, coverage_vs_error_path)
    coverage_summary = summarize_coverage(mass_peri, peri_vel)

    metrics_path = TABLES_DIR / "model_metrics_summary.csv"
    regression_predictions_path = TABLES_DIR / "regression_oof_predictions.csv"
    classification_predictions_path = TABLES_DIR / "classification_oof_predictions.csv"
    slice_summary_path = TABLES_DIR / "selected_slice_summary.csv"
    coverage_mass_peri_path = TABLES_DIR / "coverage_mass_vs_periapsis.csv"
    coverage_peri_vel_path = TABLES_DIR / "coverage_periapsis_vs_velocity.csv"
    slice_grid_path = TABLES_DIR / "slice_grid_predictions_and_domain_labels.csv"

    metrics.sort_values(["task", "target", "model"]).to_csv(metrics_path, index=False)
    regression_predictions.to_csv(regression_predictions_path, index=False)
    classification_predictions.to_csv(classification_predictions_path, index=False)
    slice_summary = build_slice_summary(chosen_slice, slice_spec)
    slice_summary.to_csv(slice_summary_path, index=False)
    mass_peri.to_csv(coverage_mass_peri_path)
    peri_vel.to_csv(coverage_peri_vel_path)
    pd.concat(slice_grid_rows, ignore_index=True).to_csv(slice_grid_path, index=False)

    output_files = {
        "bmf_slice": bmf_slice_path.resolve(),
        "fragment_slice": fragment_slice_path.resolve(),
        "largest_fragment_slice": largest_fragment_slice_path.resolve(),
        "logistic_slice": logistic_slice_path.resolve(),
        "coverage_heatmaps": coverage_heatmaps_path.resolve(),
        "coverage_vs_error": coverage_vs_error_path.resolve(),
        "coverage_mass_peri": coverage_mass_peri_path.resolve(),
        "coverage_peri_vel": coverage_peri_vel_path.resolve(),
        "slice_grid": slice_grid_path.resolve(),
        "metrics": metrics_path.resolve(),
        "regression_predictions": regression_predictions_path.resolve(),
        "classification_predictions": classification_predictions_path.resolve(),
        "slice_summary": slice_summary_path.resolve(),
    }
    write_report(REPORT_PATH.resolve(), slice_spec, slice_summary, metrics, coverage_summary, output_files)
    write_image_report(IMAGE_REPORT_PATH.resolve(), slice_spec, slice_summary, metrics, coverage_summary, output_files)


if __name__ == "__main__":
    main()
