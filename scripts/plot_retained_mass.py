#!/usr/bin/env python3
"""Retained mass fraction vs periapsis — aggregated controlled subset.

Controlled subset:
  - mass         : A2000  (fixed)
  - resolution   : n65    (fixed)
  - timestep     : 90000  (only value; fixed)
  - FoF link len : 0.004  (fixed)

Aggregation:
  Before plotting, group by (periapsis_Rm, v_inf_kms, spin_cat, spin_period)
  and take the MEDIAN BMF.  This ensures one averaged point per x-position.
  Spin period is NEVER mixed inside a single line.

Figures produced:
  1. No-spin only  — colour = v_inf_kms
  2. Spin comparison — 2-panel (period=3.0 hr | period=4.7 hr);
     within each panel: linestyle = spin orientation, colour = v_inf_kms
  3. Spin-period effect at v=0 — colour = spin period, prograde-z only
     (no-spin baseline shown as dashed black)

Zero-BMF rows are omitted for log scale; a note is added to every plot.
Only groups with >= MIN_POINTS periapsis values after aggregation are drawn.

Run:
    python scripts/plot_retained_mass.py \\
        --outcomes outputs/bound_outcomes_dedup.csv \\
        --out-dir  eda/bound_eda/plots
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# ── constants ─────────────────────────────────────────────────────────────────

MASS_FILTER   = "A2000"
RES_FILTER    = "n65"
FOF_FILTER    = 0.004

BMF_FLOOR  = 1e-4   # rows below this are omitted (log scale)
MIN_POINTS = 2      # minimum periapsis values per line after aggregation

# spin periods to use for the comparison figure (best-covered)
PRIMARY_PERIODS = [3.0, 4.7]

# ── colour palettes ───────────────────────────────────────────────────────────

V_COLORS = {
    0.0: "#1565C0",
    0.2: "#2E7D32",
    0.4: "#558B2F",
    0.6: "#F9A825",
    0.8: "#EF6C00",
    1.0: "#B71C1C",
    1.2: "#6A1B9A",
    1.4: "#00695C",
    1.6: "#37474F",
}

PERIOD_COLORS = {
    3.0:  "#E53935",
    3.5:  "#FB8C00",
    3.6:  "#FDD835",
    4.7:  "#43A047",
    8.6:  "#1E88E5",
    17.0: "#8E24AA",
}

SPIN_LS = {
    "no_spin":    "-",
    "prograde_z": "--",
    "retro_z":    "-.",
    "equatorial": ":",
}
SPIN_LABEL = {
    "no_spin":    r"No spin ($L_z = 0$)",
    "prograde_z": r"Prograde-$z$",
    "retro_z":    r"Retrograde-$z$",
    "equatorial": r"Equatorial ($x$/$y$)",
}


# ── helpers ───────────────────────────────────────────────────────────────────

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
    return "equatorial"


def _spin_period(code) -> float | None:
    if pd.isna(code) or str(code).strip() in ("", "none"):
        return None
    m = re.match(r"s(\d{3})", str(code))
    return int(m.group(1)) / 10.0 if m else None


def load_controlled(path: str) -> pd.DataFrame:
    """Load and filter to the controlled subset; parse codes; drop zero BMF."""
    df = pd.read_csv(path, low_memory=False)
    df = df[
        (df["mass_code"]          == MASS_FILTER) &
        (df["resolution_code"]    == RES_FILTER)  &
        (df["fof_linking_length"] == FOF_FILTER)
    ].copy()
    df["periapsis_Rm"] = df["periapsis_code"].str[1:].astype(int) / 10.0
    df["v_inf_kms"]    = df["velocity_code"].str[1:].astype(int)  / 10.0
    df["spin_cat"]     = df["spin_code"].apply(_spin_cat)
    df["spin_period"]  = df["spin_code"].apply(_spin_period)
    # keep raw zeros in the data; we'll drop them only during plotting
    return df


def aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Median BMF per group; then drop rows below BMF_FLOOR."""
    agg = (
        df.groupby(group_cols, sort=True, dropna=False)["bound_mass_fraction"]
        .median()
        .reset_index()
        .rename(columns={"bound_mass_fraction": "bmf"})
    )
    return agg[agg["bmf"] > BMF_FLOOR].copy()


def _style_ax(ax: plt.Axes) -> None:
    ax.set_yscale("log")
    ax.set_ylim(BMF_FLOOR * 0.6, 1.2)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, _: f"{y:.0%}" if y >= 0.01 else f"{y:.1e}"
    ))
    ax.set_xlabel(r"Periapsis ($R_\mathcal{d}$)", fontsize=11)
    ax.set_ylabel("Bound Mass Fraction", fontsize=11)
    ax.grid(True, which="both", alpha=0.18, linestyle="--", linewidth=0.7)
    ax.annotate(
        "Zero BMF values omitted for log scale",
        xy=(0.01, 0.01), xycoords="axes fraction",
        fontsize=7.5, color="#666666", ha="left", va="bottom",
    )


def _subset_title_suffix() -> str:
    return (
        f"Subset: mass={MASS_FILTER}, res={RES_FILTER}, "
        f"fof_ll={FOF_FILTER}, t=90000"
    )


# ── Figure 1: no-spin only ────────────────────────────────────────────────────

def fig_nospin(df: pd.DataFrame, out: Path) -> None:
    sub  = df[df["spin_cat"] == "no_spin"].copy()
    agg  = aggregate(sub, ["periapsis_Rm", "v_inf_kms"])

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for v, grp in agg.groupby("v_inf_kms"):
        grp = grp.sort_values("periapsis_Rm")
        if len(grp) < MIN_POINTS:
            continue
        color = V_COLORS.get(v, "grey")
        ax.plot(
            grp["periapsis_Rm"], grp["bmf"],
            color=color, linestyle="-", linewidth=2.0,
            marker="o", markersize=5.5,
            markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.6,
            label=f"{v:.1f}",
        )

    ax.set_title(
        "Retained Mass Fraction vs Periapsis — No-spin Runs\n"
        + _subset_title_suffix(),
        fontsize=11,
    )
    _style_ax(ax)
    ax.legend(
        title=r"$v_\infty$ (km s$^{-1}$)", fontsize=9,
        title_fontsize=9.5, framealpha=0.9, loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ── Figure 2: spin comparison — separate panels per spin period ───────────────

def fig_spin_comparison(df: pd.DataFrame, out: Path,
                        periods: list[float] = PRIMARY_PERIODS) -> None:
    """One sub-panel per spin period; within each panel colour=v_inf, ls=spin."""
    cats   = ["no_spin", "prograde_z", "retro_z"]
    n_pan  = len(periods)
    fig, axes = plt.subplots(1, n_pan, figsize=(5.5 * n_pan, 6), sharey=True)
    if n_pan == 1:
        axes = [axes]

    # pre-aggregate no-spin (no spin_period, shared across panels)
    nospin_agg = aggregate(
        df[df["spin_cat"] == "no_spin"],
        ["periapsis_Rm", "v_inf_kms"],
    )

    legend_v = {}
    legend_s = {}

    for ax, period in zip(axes, periods):
        # spin rows: fix to this period
        spin_sub = df[
            (df["spin_cat"].isin(["prograde_z", "retro_z"])) &
            (df["spin_period"] == period)
        ]
        spin_agg = aggregate(spin_sub, ["periapsis_Rm", "v_inf_kms", "spin_cat"])

        velocities = sorted(
            set(nospin_agg["v_inf_kms"].unique()) |
            set(spin_agg["v_inf_kms"].unique())
        )

        for v in velocities:
            color = V_COLORS.get(v, "grey")

            # no-spin (solid)
            ns = nospin_agg[nospin_agg["v_inf_kms"] == v].sort_values("periapsis_Rm")
            if len(ns) >= MIN_POINTS:
                ax.plot(
                    ns["periapsis_Rm"], ns["bmf"],
                    color=color, linestyle="-", linewidth=1.8,
                    marker="o", markersize=4.5,
                    markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=1.3,
                )
                legend_v.setdefault(v, mlines.Line2D(
                    [], [], color=color, linestyle="-", linewidth=2, label=f"{v:.1f}"
                ))
                legend_s.setdefault("no_spin", mlines.Line2D(
                    [], [], color="black", linestyle="-", linewidth=2,
                    label=SPIN_LABEL["no_spin"]
                ))

            # prograde-z (dashed)
            for cat, ls_key in [("prograde_z", "prograde_z"), ("retro_z", "retro_z")]:
                pts = spin_agg[
                    (spin_agg["v_inf_kms"] == v) &
                    (spin_agg["spin_cat"] == cat)
                ].sort_values("periapsis_Rm")
                if len(pts) < MIN_POINTS:
                    continue
                ls = SPIN_LS[cat]
                ax.plot(
                    pts["periapsis_Rm"], pts["bmf"],
                    color=color, linestyle=ls, linewidth=1.8,
                    marker="o", markersize=4.5,
                    markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=1.3,
                )
                legend_v.setdefault(v, mlines.Line2D(
                    [], [], color=color, linestyle="-", linewidth=2, label=f"{v:.1f}"
                ))
                legend_s.setdefault(cat, mlines.Line2D(
                    [], [], color="black", linestyle=ls, linewidth=2,
                    label=SPIN_LABEL[cat]
                ))

        period_str = (
            "No spin / "
            + " / ".join([f"spin period {period:.1f} hr"])
        )
        ax.set_title(f"Spin period = {period:.1f} hr", fontsize=10.5)
        _style_ax(ax)
        if ax is not axes[0]:
            ax.set_ylabel("")

    fig.suptitle(
        "Retained Mass Fraction vs Periapsis — Spin Orientation Effect\n"
        + _subset_title_suffix(),
        fontsize=11, y=1.01,
    )

    # shared legends — attach to last axis
    leg_v = axes[-1].legend(
        handles=sorted(legend_v.values(), key=lambda h: float(h.get_label())),
        title=r"$v_\infty$ (km s$^{-1}$)",
        loc="upper right", fontsize=8.5, title_fontsize=9,
        framealpha=0.9, borderpad=0.5,
    )
    axes[-1].add_artist(leg_v)

    if legend_s:
        axes[-1].legend(
            handles=[legend_s[k] for k in cats if k in legend_s],
            title="Spin orientation",
            loc="lower left", fontsize=8.5, title_fontsize=9,
            framealpha=0.9, borderpad=0.5,
        )

    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ── Figure 3: spin-period effect at v=0, prograde-z ──────────────────────────

def fig_spin_period(df: pd.DataFrame, out: Path) -> None:
    """Prograde-z only at v=0; colour = spin period; no-spin as baseline."""
    prog = df[
        (df["v_inf_kms"] == 0.0) &
        (df["spin_cat"] == "prograde_z")
    ]
    agg = aggregate(prog, ["periapsis_Rm", "spin_period"])

    # no-spin baseline at v=0
    ns_agg = aggregate(
        df[(df["v_inf_kms"] == 0.0) & (df["spin_cat"] == "no_spin")],
        ["periapsis_Rm"],
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # baseline
    if len(ns_agg) >= MIN_POINTS:
        ax.plot(
            ns_agg["periapsis_Rm"], ns_agg["bmf"],
            color="black", linestyle="-", linewidth=2.2,
            marker="D", markersize=5,
            markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.5,
            label="No spin (baseline)",
        )

    periods = sorted(agg["spin_period"].dropna().unique())
    for period in periods:
        pts = agg[agg["spin_period"] == period].sort_values("periapsis_Rm")
        if len(pts) < MIN_POINTS:
            continue
        color = PERIOD_COLORS.get(period, "grey")
        ax.plot(
            pts["periapsis_Rm"], pts["bmf"],
            color=color, linestyle="--", linewidth=1.9,
            marker="s", markersize=5,
            markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.4,
            label=f"{period:.1f} hr",
        )

    ax.set_title(
        r"Retained Mass Fraction — Prograde-$z$ Spin Period Effect"
        "\n"
        r"($v_\infty = 0$ km s$^{-1}$)  ·  " + _subset_title_suffix(),
        fontsize=11,
    )
    _style_ax(ax)
    ax.legend(
        title="Spin period",
        fontsize=9, title_fontsize=9.5,
        framealpha=0.9, loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outcomes", default="outputs/bound_outcomes_dedup.csv")
    p.add_argument("--out-dir",  default="eda/bound_eda/plots")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_controlled(args.outcomes)
    n_raw = len(df)
    n_pos = (df["bound_mass_fraction"] > BMF_FLOOR).sum()
    print(
        f"Controlled subset: {n_raw} rows total | "
        f"{n_pos} with BMF > {BMF_FLOOR}  "
        f"(mass={MASS_FILTER}, res={RES_FILTER}, fof_ll={FOF_FILTER})"
    )

    fig_nospin(df,         out_dir / "retained_mass_nospin.png")
    fig_spin_comparison(df, out_dir / "retained_mass_spin_effect.png")
    fig_spin_period(df,    out_dir / "retained_mass_spin_period.png")

    print("\nAll done.")


if __name__ == "__main__":
    main()
