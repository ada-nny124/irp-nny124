"""Inspect the structure of a local HDF5 file without loading full arrays."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    import h5py
except ImportError:  # pragma: no cover - import guard for optional dependency
    h5py = None


def collect_schema_rows(h5_file: "h5py.File") -> list[dict[str, str]]:
    """Collect dataset and group metadata recursively."""
    rows: list[dict[str, str]] = []

    def visitor(name: str, obj: "h5py.Dataset | h5py.Group") -> None:
        if isinstance(obj, h5py.Dataset):
            rows.append(
                {
                    "path": name,
                    "kind": "dataset",
                    "shape": str(obj.shape),
                    "dtype": str(obj.dtype),
                }
            )
        else:
            rows.append({"path": name, "kind": "group", "shape": "", "dtype": ""})

    h5_file.visititems(visitor)
    return rows


def write_schema(rows: list[dict[str, str]], output_path: Path) -> None:
    """Write schema rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "kind", "shape", "dtype"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True, help="Path to a local HDF5 file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/hdf5_schema_summary.csv"),
        help="CSV output path for schema summary.",
    )
    return parser.parse_args()


def main() -> int:
    """Inspect a local HDF5 schema."""
    args = parse_args()
    if h5py is None:
        print("Error: h5py is required for schema inspection. Install it in your local environment first.", file=sys.stderr)
        return 1
    if not args.file.is_file():
        print(f"Error: HDF5 file not found: {args.file}", file=sys.stderr)
        return 1

    with h5py.File(args.file, "r") as handle:
        top_level_groups = list(handle.keys())
        rows = collect_schema_rows(handle)

    print("Top-level groups/datasets:")
    for name in top_level_groups:
        print(f"- {name}")
    print("\nRecursive schema listing:")
    for row in rows:
        suffix = f" shape={row['shape']} dtype={row['dtype']}" if row["kind"] == "dataset" else ""
        print(f"- {row['kind']}: {row['path']}{suffix}")

    write_schema(rows, args.output)
    print(f"\nSaved schema summary to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
