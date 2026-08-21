#!/usr/bin/env python3
"""Inspect sampled Martian-moons FoF HDF5 files and write a schema summary CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from extract_fof_outcomes import SCHEMA_FIELDS, select_hdf5_files, summarize_hdf5_schema, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Directory containing FoF HDF5 files.")
    parser.add_argument(
        "--output",
        default="extraction-outputs/tables/hdf5_schema_summary.csv",
        help="Schema summary CSV path.",
    )
    parser.add_argument("--limit", type=int, default=3, help="Number of HDF5 files to sample.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    rows = summarize_hdf5_schema(select_hdf5_files(data_dir, args.limit))
    write_csv(output_path, rows, SCHEMA_FIELDS)
    print(f"Wrote {len(rows)} schema rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
