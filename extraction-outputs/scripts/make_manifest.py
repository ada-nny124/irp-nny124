#!/usr/bin/env python3
"""Create a filename-derived manifest for Martian-moons FoF HDF5 files."""

from __future__ import annotations

import argparse
from pathlib import Path

from extract_fof_outcomes import MANIFEST_FIELDS, build_manifest_rows, select_hdf5_files, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Directory containing FoF HDF5 files.")
    parser.add_argument("--output", default="outputs/manifest.csv", help="Manifest CSV path.")
    parser.add_argument("--limit", type=int, default=None, help="Only include the first N files after sorting.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    rows = build_manifest_rows(select_hdf5_files(data_dir, args.limit))
    write_csv(output_path, rows, MANIFEST_FIELDS)
    print(f"Wrote {len(rows)} manifest rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
