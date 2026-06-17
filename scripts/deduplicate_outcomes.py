#!/usr/bin/env python3
"""Create a deduplicated version of bound_outcomes.csv for ML.

Some physical conditions (same mass / periapsis / velocity / spin)
were run at multiple resolutions to test numerical convergence.
This script:
  1. Identifies duplicate physical conditions (same mass, periapsis,
     velocity, spin_code, timestep) across different resolution codes.
  2. Keeps only one representative row per physical condition
     (the highest resolution by default, or specified via --keep).
  3. Also deduplicates rows with the same physical_file but different
     fof_linking_length by keeping one preferred linking length.
  4. Writes the result to outputs/bound_outcomes_dedup.csv.

Run:
    python scripts/deduplicate_outcomes.py \
      --outcomes  outputs/bound_outcomes.csv \
      --out       outputs/bound_outcomes_dedup.csv \
      --keep      highest_resolution \
      --ll-mode   most_common
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

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
    spin_str = m.group("spin") or ""
    return {
        "p_mass":       m.group("mass"),
        "p_spin":       spin_str,
        "p_resolution": int(m.group("resolution")),
        "p_periapsis":  int(m.group("periapsis")),
        "p_velocity":   int(m.group("velocity")),
        "p_timestep":   int(m.group("timestep")),
        "p_ll":         float(m.group("ll")),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outcomes", default="outputs/bound_outcomes.csv")
    p.add_argument("--out",      default="outputs/bound_outcomes_dedup.csv")
    p.add_argument(
        "--keep",
        choices=["highest_resolution", "lowest_resolution", "first"],
        default="highest_resolution",
        help="Which resolution to keep when duplicates exist.",
    )
    p.add_argument(
        "--ll-mode",
        choices=["most_common", "median", "first"],
        default="most_common",
        help="Which FoF linking length to keep per physical file.",
    )
    return p.parse_args()


def pick_ll(df: pd.DataFrame, mode: str) -> float:
    if mode == "most_common":
        counts = df["p_ll"].value_counts()
        return float(counts.index[0])
    if mode == "median":
        return float(df["p_ll"].median())
    return float(df["p_ll"].iloc[0])


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.outcomes, low_memory=False)
    n_raw = len(df)
    print(f"Loaded {n_raw} rows from {args.outcomes}")

    # ── parse filenames ──────────────────────────────────────────────────────
    parsed = df["fof_file"].map(parse_filename).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)

    unparseable = df["p_mass"].isna().sum()
    if unparseable:
        print(f"  WARNING: {unparseable} rows could not be parsed — they will be kept as-is.")

    # ── Step 1: deduplicate linking length per physical file ─────────────────
    # A physical_file is the same physical snapshot; different ll rows are
    # post-processing variants. Keep one ll per physical_file.
    print("\nStep 1: Select one FoF linking-length per physical snapshot …")
    ll_rows = []
    for pfile, group in df.groupby("physical_file", dropna=False):
        if group["p_ll"].isna().all():
            ll_rows.append(group)
            continue
        chosen_ll = pick_ll(group.dropna(subset=["p_ll"]), args.ll_mode)
        kept = group[group["p_ll"] == chosen_ll]
        if kept.empty:
            kept = group.head(1)
        ll_rows.append(kept)
    df_ll = pd.concat(ll_rows, ignore_index=True)
    print(f"  Rows after ll dedup: {len(df_ll)}  (removed {n_raw - len(df_ll)})")

    # ── Step 2: identify duplicate physical conditions at different resolutions
    print("\nStep 2: Identify resolution duplicates …")
    # Physical identity columns (excluding resolution)
    id_cols = ["p_mass", "p_spin", "p_periapsis", "p_velocity", "p_timestep"]
    id_cols_present = [c for c in id_cols if c in df_ll.columns]

    df_parsed = df_ll.dropna(subset=["p_mass"])     # parseable rows
    df_unparseable = df_ll[df_ll["p_mass"].isna()]   # keep these unchanged

    dup_groups = df_parsed.groupby(id_cols_present, dropna=False)
    multi = dup_groups.filter(lambda g: g["p_resolution"].nunique() > 1)
    n_multi_conditions = multi.groupby(id_cols_present, dropna=False).ngroups
    print(f"  Physical conditions run at multiple resolutions: {n_multi_conditions}")

    # For each condition, keep one resolution
    res_rows = []
    for keys, group in dup_groups:
        if group["p_resolution"].nunique() <= 1:
            res_rows.append(group)
            continue
        if args.keep == "highest_resolution":
            best_res = group["p_resolution"].max()
        elif args.keep == "lowest_resolution":
            best_res = group["p_resolution"].min()
        else:
            best_res = group["p_resolution"].iloc[0]
        kept = group[group["p_resolution"] == best_res]
        if kept.empty:
            kept = group.head(1)
        res_rows.append(kept)

    df_dedup = pd.concat(res_rows + [df_unparseable], ignore_index=True)
    n_removed = len(df_ll) - len(df_dedup)
    print(f"  Rows after resolution dedup: {len(df_dedup)}  (removed {n_removed})")

    # ── Report ───────────────────────────────────────────────────────────────
    # Drop the helper columns before saving
    helper_cols = [c for c in df_dedup.columns if c.startswith("p_")]
    df_out = df_dedup.drop(columns=helper_cols, errors="ignore")

    print(f"\nSummary:")
    print(f"  Original rows : {n_raw}")
    print(f"  After ll dedup: {len(df_ll)}")
    print(f"  After res dedup: {len(df_dedup)}")
    print(f"  Total removed : {n_raw - len(df_dedup)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved deduplicated data to {out_path}")

    # ── Write a duplicate report ─────────────────────────────────────────────
    report_path = out_path.with_name(out_path.stem + "_duplicate_report.csv")
    dup_report = (
        multi.groupby(id_cols_present, dropna=False)["p_resolution"]
        .apply(lambda s: ",".join(str(v) for v in sorted(s.unique())))
        .reset_index()
        .rename(columns={"p_resolution": "resolutions_present"})
    )
    dup_report.to_csv(report_path, index=False)
    print(f"Duplicate report saved to {report_path}")


if __name__ == "__main__":
    main()
