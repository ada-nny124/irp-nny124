#!/usr/bin/env python3
"""Kegerreis-style retained mass fraction vs periapsis plot.

Reproduces the style of Kegerreis et al. (2024) Fig. 6:
  - x-axis : periapsis (R♂)
  - y-axis  : bound mass fraction  [log scale, zeros dropped]
  - colour  : v_∞ (km s⁻¹)
  - linestyle: spin category  (no-spin | prograde-z | retrograde-z | equatorial)

Dataset: outputs/bound_outcomes_dedup.csv  (deduplicated, one row per condition)

Only the "standard" resolution is included for the no-spin runs (controlled
subset); spin runs are plotted separately with a faint background style.

Run:
    python scripts/eda/plot_kegerreis_bmf.py \
        --outcomes outputs/bound_outcomes_dedup.csv \
        --out      eda/bound_eda/plots/kegerreis_bmf_vs_periapsis.png
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

# ── colour palette matching Kegerreis (blue → green → yellow → orange → red)
VELOCITY_COLORS = {
    0.0: "#1565C0",   # deep blue
    0.2: "#2E7D32",   # dark green
    0.4: "#F9A825",   # amber / yellow
    0.6: "#EF6C00",   # orange
    0.8: "#B71C1C",   # dark red
    1.0: "#6A1B9A",   # purple  (extra velocities beyond Kegerreis range)
    1.2: "#00838F",
    1.4: "#37474F",
    1.5: "#880E4F",
    1.6: "#004D40",
    2.0: "#212121",
}

# ── line-style mapping for spin category
SPIN_LINESTYLE = {
    "no_spin":     "-",        # solid   — no spin
    "prograde_z":  "--",       # dashed  — prograde z (spin aligned with orbit)
    "retro_z":     "-.",       # dash-dot — retrograde z
    "equatorial":  ":",        # dotted  — x / y axis spin
}

SPIN_LABEL = {
    "no_spin":     "No spin  (Lz = 0)",
    "prograde_z":  "Prograde-z  (Lz > 0)",
    "retro_z":     "Retrograde-z  (Lz < 0)",
    "equatorial":  "Equatorial (x / y axis)",
}

BMF_FLOOR = 1e-4   # below this → treated as zero and excluded from log-plot


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw code columns to physical values."""
    df = df.copy()
    df["periapsis_Rm"] = df["periapsis_code"].str[1:].astype(int) / 10.0
    df["v_inf_kms"]    = df["velocity_code"].str[1:].astype(int) / 10.0

    def _spin_cat(code) -> str:
        if pd.isna(code) or str(code).strip() in ("", "none"):
            return "no_spin"
        m = re.match(r"s(\d{3})(.*)", str(code))
        if not m:
            return "no_spin"
        axis = m.group(2) or "z"
        if "mz" in axis:
            return "retro_z"
        if axis == "z":
            return "prograde_z"
        return "equatorial"   # x, y, mx, my, etc.

    df["spin_cat"] = df["spin_code"].apply(_spin_cat)
    return df


def aggregate_bmf(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Mean BMF per (periapsis, group); drop zero-BMF points for log plot."""
    agg = (
        df.groupby(group_cols, sort=True, dropna=False)["bound_mass_fraction"]
        .mean()
        .reset_index()
        .rename(columns={"bound_mass_fraction": "bmf_mean"})
    )
    # keep rows where at least one simulation had BMF above the floor
    agg = agg[agg["bmf_mean"] > BMF_FLOOR]
    return agg


# ── main plot ─────────────────────────────────────────────────────────────────

def plot_kegerreis(df: pd.DataFrame, out_path: Path) -> None:
    """Draw the full Kegerreis-style figure."""
    fig, ax = plt.subplots(figsize=(8, 6))

    velocities  = sorted(df["v_inf_kms"].unique())
    spin_cats   = ["no_spin", "prograde_z", "retro_z", "equatorial"]

    # ── draw lines ─────────────────────────────────────────────────────────
    legend_v_handles   = {}   # colour legend (velocity)
    legend_s_handles   = {}   # linestyle legend (spin)

    for v in velocities:
        color = VELOCITY_COLORS.get(v, "grey")
        for scat in spin_cats:
            sub = df[(df["v_inf_kms"] == v) & (df["spin_cat"] == scat)].copy()
            agg = aggregate_bmf(sub, ["periapsis_Rm"])
            if agg.empty:
                continue

            ls = SPIN_LINESTYLE[scat]
            lw = 1.8 if scat == "no_spin" else 1.4
            alpha = 1.0 if scat == "no_spin" else 0.85

            line, = ax.plot(
                agg["periapsis_Rm"], agg["bmf_mean"],
                color=color, linestyle=ls, linewidth=lw, alpha=alpha,
                marker="o", markersize=3.5, markerfacecolor=color,
            )
            # collect handle for velocity legend (once per velocity)
            if v not in legend_v_handles:
                legend_v_handles[v] = matplotlib.lines.Line2D(
                    [], [], color=color, linestyle="-", linewidth=2,
                    label=f"{v:.1f}",
                )
            # collect handle for spin-style legend (once per spin category)
            if scat not in legend_s_handles and not agg.empty:
                legend_s_handles[scat] = matplotlib.lines.Line2D(
                    [], [], color="black", linestyle=ls, linewidth=2,
                    label=SPIN_LABEL[scat],
                )

    # ── log scale & zero handling ───────────────────────────────────────────
    ax.set_yscale("log")
    ax.set_ylim(BMF_FLOOR * 0.5, 1.2)

    # Custom y-tick labels so 0.1, 0.2 … look clean
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, _: f"{y:g}" if y >= 0.1 else f"{y:.0e}"
    ))

    # ── axes labels & styling ───────────────────────────────────────────────
    ax.set_xlabel(r"Periapsis ($R_{\!\!\mathcal{d}}$)", fontsize=13)
    ax.set_ylabel("Bound Mass Fraction", fontsize=13)
    ax.set_title(
        "Retained Mass Fraction vs Periapsis\n"
        r"(Kegerreis-style; $v_\infty$ colour, spin linestyle)",
        fontsize=12,
    )
    ax.grid(True, which="both", alpha=0.2, linestyle="--")
    ax.set_xlim(df["periapsis_Rm"].min() - 0.05, df["periapsis_Rm"].max() + 0.05)

    # ── two-part legend ─────────────────────────────────────────────────────
    # velocity legend (colour)
    leg_v = ax.legend(
        handles=list(legend_v_handles.values()),
        title=r"$v_\infty$  (km s$^{-1}$)",
        loc="lower left",
        fontsize=8.5,
        title_fontsize=9,
        framealpha=0.9,
        borderpad=0.6,
    )
    ax.add_artist(leg_v)

    # spin legend (linestyle) — placed upper right
    if legend_s_handles:
        ax.legend(
            handles=[legend_s_handles[k] for k in spin_cats if k in legend_s_handles],
            title="Spin category",
            loc="upper right",
            fontsize=8.5,
            title_fontsize=9,
            framealpha=0.9,
            borderpad=0.6,
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── second figure: no-spin only (cleanest Kegerreis analogue) ────────────────

def plot_kegerreis_nospin(df: pd.DataFrame, out_path: Path) -> None:
    """No-spin runs only — exact Kegerreis analogue (velocity only)."""
    sub = df[df["spin_cat"] == "no_spin"].copy()
    velocities = sorted(sub["v_inf_kms"].unique())

    fig, ax = plt.subplots(figsize=(7, 5.5))

    for v in velocities:
        vsub = sub[sub["v_inf_kms"] == v]
        agg  = aggregate_bmf(vsub, ["periapsis_Rm"])
        if agg.empty:
            continue
        color = VELOCITY_COLORS.get(v, "grey")
        ax.plot(
            agg["periapsis_Rm"], agg["bmf_mean"],
            color=color, linestyle="-", linewidth=2.0,
            marker="o", markersize=4, markerfacecolor=color,
            label=f"{v:.1f}",
        )

    ax.set_yscale("log")
    ax.set_ylim(BMF_FLOOR * 0.5, 1.2)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, _: f"{y:g}" if y >= 0.1 else f"{y:.0e}"
    ))
    ax.set_xlabel(r"Periapsis ($R_{\!\!\mathcal{d}}$)", fontsize=13)
    ax.set_ylabel("Bound Mass Fraction", fontsize=13)
    ax.set_title(
        "Retained Mass Fraction vs Periapsis — No-spin Runs\n"
        r"(Kegerreis-style; colour = $v_\infty$)",
        fontsize=12,
    )
    ax.legend(
        title=r"$v_\infty$  (km s$^{-1}$)",
        fontsize=9, title_fontsize=10,
        framealpha=0.9, loc="lower left",
    )
    ax.grid(True, which="both", alpha=0.2, linestyle="--")
    ax.set_xlim(sub["periapsis_Rm"].min() - 0.05, sub["periapsis_Rm"].max() + 0.05)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── third figure: z-spin comparison (no-spin vs prograde vs retrograde) ───────

def plot_kegerreis_spin_comparison(df: pd.DataFrame, out_path: Path) -> None:
    """Fix velocity to the best-represented values; vary spin category."""

    # pick velocities with enough coverage (≥3 periapsis points)
    def _count_peri(v, scat):
        return aggregate_bmf(
            df[(df["v_inf_kms"] == v) & (df["spin_cat"] == scat)],
            ["periapsis_Rm"],
        ).shape[0]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    velocities  = sorted(df["v_inf_kms"].unique())
    spin_cats   = ["no_spin", "prograde_z", "retro_z"]

    legend_v_handles = {}
    legend_s_handles = {}

    for v in velocities:
        color = VELOCITY_COLORS.get(v, "grey")
        for scat in spin_cats:
            sub = df[(df["v_inf_kms"] == v) & (df["spin_cat"] == scat)]
            agg = aggregate_bmf(sub, ["periapsis_Rm"])
            if len(agg) < 2:
                continue
            ls = SPIN_LINESTYLE[scat]
            ax.plot(
                agg["periapsis_Rm"], agg["bmf_mean"],
                color=color, linestyle=ls, linewidth=1.8,
                marker="o", markersize=3.5, alpha=0.9,
            )
            if v not in legend_v_handles:
                legend_v_handles[v] = matplotlib.lines.Line2D(
                    [], [], color=color, linestyle="-", linewidth=2,
                    label=f"{v:.1f}",
                )
            if scat not in legend_s_handles:
                legend_s_handles[scat] = matplotlib.lines.Line2D(
                    [], [], color="black", linestyle=ls, linewidth=2,
                    label=SPIN_LABEL[scat],
                )

    ax.set_yscale("log")
    ax.set_ylim(BMF_FLOOR * 0.5, 1.2)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, _: f"{y:g}" if y >= 0.1 else f"{y:.0e}"
    ))
    ax.set_xlabel(r"Periapsis ($R_{\!\!\mathcal{d}}$)", fontsize=13)
    ax.set_ylabel("Bound Mass Fraction", fontsize=13)
    ax.set_title(
        "Retained Mass Fraction — Spin Effect\n"
        r"(colour = $v_\infty$, linestyle = spin)",
        fontsize=12,
    )
    ax.grid(True, which="both", alpha=0.2, linestyle="--")

    leg_v = ax.legend(
        handles=list(legend_v_handles.values()),
        title=r"$v_\infty$ (km s$^{-1}$)",
        loc="lower left", fontsize=8.5, title_fontsize=9, framealpha=0.9,
    )
    ax.add_artist(leg_v)
    if legend_s_handles:
        ax.legend(
            handles=[legend_s_handles[k] for k in spin_cats if k in legend_s_handles],
            title="Spin",
            loc="upper right", fontsize=8.5, title_fontsize=9, framealpha=0.9,
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outcomes", default="outputs/bound_outcomes_dedup.csv")
    p.add_argument("--out-dir",  default="eda/bound_eda/plots")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    out_dir = Path(args.out_dir)
    df_raw  = pd.read_csv(args.outcomes, low_memory=False)
    df      = parse_codes(df_raw)

    print(f"Loaded {len(df)} rows | "
          f"BMF > 0: {(df['bound_mass_fraction'] > BMF_FLOOR).sum()}")

    # 1. Full plot (all velocities × all spin categories)
    plot_kegerreis(
        df,
        out_dir / "kegerreis_bmf_vs_periapsis_full.png",
    )

    # 2. No-spin only (cleanest Kegerreis analogue)
    plot_kegerreis_nospin(
        df,
        out_dir / "kegerreis_bmf_vs_periapsis_nospin.png",
    )

    # 3. Spin comparison (no-spin / prograde-z / retrograde-z)
    plot_kegerreis_spin_comparison(
        df,
        out_dir / "kegerreis_bmf_vs_periapsis_spin_effect.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
