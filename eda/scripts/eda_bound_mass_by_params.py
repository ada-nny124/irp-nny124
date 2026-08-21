#!/usr/bin/env python3
"""EDA: bound-outcome metrics vs simulation parameters (Figure-6-style plots).

Produces one multi-line plot per target metric showing how the outcome
varies with periapsis distance while colouring/styling by every other
parameter (velocity, mass, spin axis, spin period, resolution).

Run:
    python eda/scripts/eda_bound_mass_by_params.py \
      --outcomes outputs/bound_outcomes.csv \
      --eda-dir eda/bound_eda
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── filename parser ──────────────────────────────────────────────────────────
FILENAME_RE = re.compile(
    r"^(?:Ma_xp_)?(?P<mass>A\d{4}(?:c30)?)(?:_(?P<spin>s\d{3}[A-Za-z]*))?"
    r"_n(?P<resolution>\d+)_r(?P<periapsis>\d+)_v(?P<velocity>\d+)"
    r"_(?P<timestep>\d+)"
    r"_fof_(?P<ll>[0-9.]+)_\d+\.hdf5$"
)


def parse_filename(filename: str) -> dict:
    fname = Path(filename).name
    m = FILENAME_RE.match(fname)
    if not m:
        return {}
    mass_str = m.group("mass")
    spin_str = m.group("spin") or ""
    spin_axis = spin_str[4:] if len(spin_str) > 4 else "none"
    spin_val  = int(spin_str[1:4]) / 10.0 if spin_str else float("nan")
    return {
        "mass_log10": int(mass_str[1:5]) / 100.0,
        "resolution": int(m.group("resolution")),
        "periapsis_Rm": int(m.group("periapsis")) / 10.0,
        "v_inf_kms": int(m.group("velocity")) / 10.0,
        "spin_axis": spin_axis,
        "spin_period_hr": spin_val,
        "fof_ll": float(m.group("ll")),
        "timestep": int(m.group("timestep")),
    }


# ── helper ───────────────────────────────────────────────────────────────────

def load_and_enrich(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    parsed = df["fof_file"].map(parse_filename).apply(pd.Series)
    for col in parsed.columns:
        if col not in df.columns:
            df[col] = parsed[col]
    df["bound_mass_fraction"]           = pd.to_numeric(df["bound_mass_fraction"], errors="coerce")
    df["bound_fragment_count"]          = pd.to_numeric(df["bound_fragment_count"], errors="coerce")
    df["largest_bound_fragment_mass_kg"]= pd.to_numeric(df["largest_bound_fragment_mass_kg"], errors="coerce")
    # bound/unbound ratio (avoid div-by-zero)
    denom = pd.to_numeric(df.get("unbound_mass_fraction", pd.Series(dtype=float)), errors="coerce").replace(0, float("nan"))
    df["bound_unbound_ratio"] = df["bound_mass_fraction"] / denom
    return df


# Choose a representative linking-length to reduce noise
def best_ll_subset(df: pd.DataFrame) -> pd.DataFrame:
    mode = df["fof_ll"].mode()
    if mode.empty:
        return df
    # prefer the most common non-zero linking length
    ll_counts = df["fof_ll"].value_counts()
    ll_use = ll_counts[ll_counts.index > 0].idxmax() if (ll_counts.index > 0).any() else mode.iloc[0]
    return df[df["fof_ll"] == ll_use].copy()


# ── plotting helpers ─────────────────────────────────────────────────────────

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]

LS_MAP = {0.0: "-", 0.5: "--", 1.0: "-."}   # for spin analogy labels


def _cmap_colors(n: int):
    cmap = plt.get_cmap("tab10")
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def figure6_style_plot(
    df: pd.DataFrame,
    groupby_col: str,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    legend_title: str,
    output_path: Path,
    log_y: bool = False,
    label_fmt: str = "{}",
) -> None:
    """Line plot of mean(y_col) vs x_col, one line per groupby_col value."""
    groups = sorted(df[groupby_col].dropna().unique())
    if not groups:
        return
    colors = _cmap_colors(len(groups))

    fig, ax = plt.subplots(figsize=(9, 6))
    for color, gval in zip(colors, groups):
        sub = df[df[groupby_col] == gval].copy()
        agg = sub.groupby(x_col, sort=True)[y_col].mean().dropna()
        if agg.empty:
            continue
        ax.plot(
            agg.index, agg.values,
            marker="o", markersize=5,
            linewidth=1.8, color=color,
            label=label_fmt.format(gval),
        )

    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(title=legend_title, fontsize=9, title_fontsize=10, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_all_params_plot(
    df: pd.DataFrame,
    y_col: str,
    ylabel: str,
    y_label_short: str,
    plots_dir: Path,
    log_y: bool = False,
) -> None:
    """For a given y_col, create one plot per groupby parameter."""
    param_specs = [
        ("v_inf_kms",      "v_inf (km s⁻¹)",   "{:.1f}"),
        ("mass_log10",     "log₁₀(mass / kg)",  "{:.1f}"),
        ("spin_axis",      "Spin axis",          "{}"),
        ("spin_period_hr", "Spin period (hr)",   "{:.1f}"),
        ("resolution",     "Resolution (n)",     "{:g}"),
    ]
    for param_col, param_label, fmt in param_specs:
        if param_col not in df.columns:
            continue
        sub = df.copy()
        if sub[param_col].dtype == object:
            pass  # string categories fine
        else:
            sub = sub.dropna(subset=[param_col])
        if sub.empty or sub[param_col].nunique() < 2:
            continue
        fname = f"by_param_{y_label_short}_vs_periapsis_by_{param_col}.png"
        figure6_style_plot(
            df=sub,
            groupby_col=param_col,
            x_col="periapsis_Rm",
            y_col=y_col,
            xlabel="Periapsis (R♂)",
            ylabel=ylabel,
            title=f"{ylabel} vs Periapsis — grouped by {param_label}",
            legend_title=param_label,
            output_path=plots_dir / fname,
            log_y=log_y,
            label_fmt=fmt,
        )
        print(f"  saved {fname}")


# ── combined all-params subplot (Figure 6 analogue) ─────────────────────────

def make_combined_subplot(
    df: pd.DataFrame,
    y_col: str,
    ylabel: str,
    y_label_short: str,
    plots_dir: Path,
    log_y: bool = False,
) -> None:
    """Single figure with one subplot per groupby parameter (2×3 grid)."""
    param_specs = [
        ("v_inf_kms",      "v_∞ (km s⁻¹)"),
        ("mass_log10",     "log₁₀(mass / kg)"),
        ("spin_axis",      "Spin axis"),
        ("spin_period_hr", "Spin period (hr)"),
        ("resolution",     "Resolution n"),
    ]
    valid_specs = [
        (col, lbl) for col, lbl in param_specs
        if col in df.columns and df[col].dropna().nunique() >= 2
    ]
    if not valid_specs:
        return

    ncols = 3
    nrows = int(np.ceil(len(valid_specs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 4.5), squeeze=False)
    axes_flat = axes.flatten()

    for ax_idx, (param_col, param_label) in enumerate(valid_specs):
        ax = axes_flat[ax_idx]
        sub = df.dropna(subset=[param_col]).copy()
        groups = sorted(sub[param_col].unique())
        colors = _cmap_colors(len(groups))
        for color, gval in zip(colors, groups):
            part = sub[sub[param_col] == gval]
            agg = part.groupby("periapsis_Rm", sort=True)[y_col].mean().dropna()
            if agg.empty:
                continue
            fmt = "{:.1f}" if isinstance(gval, float) else "{}"
            ax.plot(agg.index, agg.values, marker="o", markersize=4,
                    linewidth=1.6, color=color, label=fmt.format(gval))
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Periapsis (R♂)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(param_label, fontsize=11, fontweight="bold")
        ax.legend(title=param_label, fontsize=7, title_fontsize=8, framealpha=0.8, ncol=2)
        ax.grid(True, alpha=0.3)

    # hide unused axes
    for ax_idx in range(len(valid_specs), len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    fig.suptitle(f"{ylabel} vs Periapsis by All Parameters", fontsize=14, y=1.01)
    fig.tight_layout()
    fname = f"combined_all_params_{y_label_short}_vs_periapsis.png"
    fig.savefig(plots_dir / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outcomes", default="outputs/bound_outcomes.csv")
    p.add_argument("--eda-dir",  default="eda/bound_eda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    plots_dir = Path(args.eda_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_enrich(Path(args.outcomes))
    df = best_ll_subset(df)
    print(f"Loaded {len(df)} rows (single linking-length subset).")

    targets = [
        ("bound_mass_fraction",            "Bound Mass Fraction",             "bmf",  False),
        ("bound_fragment_count",           "Bound Fragment Count",            "bfc",  False),
        ("bound_unbound_ratio",            "Bound/Unbound Mass Ratio",        "bur",  False),
        ("largest_bound_fragment_mass_kg", "Largest Bound Fragment Mass (kg)","lbfm", False),
    ]

    for y_col, ylabel, short, log_y in targets:
        if y_col not in df.columns or df[y_col].dropna().empty:
            print(f"  skipping {y_col} (no data)")
            continue
        print(f"\n=== {ylabel} ===")
        make_all_params_plot(df, y_col, ylabel, short, plots_dir, log_y=log_y)
        make_combined_subplot(df, y_col, ylabel, short, plots_dir, log_y=log_y)

    print("\nDone — plots written to", plots_dir)


if __name__ == "__main__":
    main()
