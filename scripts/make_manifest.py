"""Build a manifest CSV from local or remote HDF5 filenames."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

OUTPUT_COLUMNS = [
    "filename",
    "model_prefix",
    "A_code",
    "mass_log10_kg",
    "has_spin",
    "spin_raw",
    "spin_period_hr",
    "spin_direction",
    "special_case",
    "n_code",
    "particle_log10",
    "r_code",
    "periapsis_Rm",
    "v_code",
    "v_inf_kms",
    "timestep",
    "fof_linking",
    "file_index",
    "parsed_ok",
]

A_TOKEN_RE = re.compile(r"^(A\d{4})([A-Za-z0-9]+)?$")
SPIN_TOKEN_RE = re.compile(r"^s(?P<period>\d{3})(?P<direction>m?[xyz])?$")
N_TOKEN_RE = re.compile(r"^(n\d{2})$")
R_TOKEN_RE = re.compile(r"^(r\d{2})$")
V_TOKEN_RE = re.compile(r"^(v\d{2})$")


def _empty_record(filename: str) -> dict[str, object]:
    return {column: None for column in OUTPUT_COLUMNS} | {"filename": filename, "parsed_ok": False}


def _parse_a_token(token: str) -> tuple[str | None, float | None, str | None]:
    match = A_TOKEN_RE.match(token)
    if not match:
        return None, None, None
    a_code = match.group(1)
    suffix = match.group(2)
    mass_log10_kg = int(a_code[1:]) / 100.0
    return a_code, mass_log10_kg, suffix


def _parse_spin_token(token: str) -> tuple[str, float, str | None] | None:
    match = SPIN_TOKEN_RE.match(token)
    if not match:
        return None
    period = int(match.group("period")) / 10.0
    direction = match.group("direction")
    if direction == "mz":
        direction = "-z"
    return token, period, direction


def parse_filename(raw_name: str) -> dict[str, object]:
    """Parse one HDF5 filename into manifest fields."""
    filename = Path(raw_name.strip()).name
    record = _empty_record(filename)
    if not filename:
        return record

    stem = filename
    for suffix in (".hdf5", ".h5"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    tokens = stem.split("_")
    a_index = next((idx for idx, token in enumerate(tokens) if A_TOKEN_RE.match(token)), None)
    if a_index is None:
        return record

    model_prefix = "_".join(tokens[:a_index])
    a_code, mass_log10_kg, special_case = _parse_a_token(tokens[a_index])
    if not model_prefix or a_code is None:
        return record

    cursor = a_index + 1
    has_spin = False
    spin_raw = None
    spin_period_hr = None
    spin_direction = None
    if cursor < len(tokens):
        spin_info = _parse_spin_token(tokens[cursor])
        if spin_info is not None:
            has_spin = True
            spin_raw, spin_period_hr, spin_direction = spin_info
            cursor += 1

    required_count = 7
    if len(tokens) - cursor < required_count:
        record.update(
            {
                "model_prefix": model_prefix,
                "A_code": a_code,
                "mass_log10_kg": mass_log10_kg,
                "has_spin": has_spin,
                "spin_raw": spin_raw,
                "spin_period_hr": spin_period_hr,
                "spin_direction": spin_direction,
                "special_case": special_case,
            }
        )
        return record

    n_code = tokens[cursor]
    r_code = tokens[cursor + 1]
    v_code = tokens[cursor + 2]
    timestep_token = tokens[cursor + 3]
    fof_token = tokens[cursor + 4]
    fof_linking_token = tokens[cursor + 5]
    file_index_token = tokens[cursor + 6]

    if not (N_TOKEN_RE.match(n_code) and R_TOKEN_RE.match(r_code) and V_TOKEN_RE.match(v_code)):
        return record | {
            "model_prefix": model_prefix,
            "A_code": a_code,
            "mass_log10_kg": mass_log10_kg,
            "has_spin": has_spin,
            "spin_raw": spin_raw,
            "spin_period_hr": spin_period_hr,
            "spin_direction": spin_direction,
            "special_case": special_case,
        }

    try:
        timestep = int(timestep_token)
        fof_linking = float(fof_linking_token)
        file_index = int(file_index_token)
    except ValueError:
        return record

    if fof_token != "fof":
        return record

    record.update(
        {
            "filename": filename,
            "model_prefix": model_prefix,
            "A_code": a_code,
            "mass_log10_kg": mass_log10_kg,
            "has_spin": has_spin,
            "spin_raw": spin_raw,
            "spin_period_hr": spin_period_hr,
            "spin_direction": spin_direction,
            "special_case": special_case,
            "n_code": n_code,
            "particle_log10": int(n_code[1:]) / 10.0,
            "r_code": r_code,
            "periapsis_Rm": int(r_code[1:]) / 10.0,
            "v_code": v_code,
            "v_inf_kms": int(v_code[1:]) / 10.0,
            "timestep": timestep,
            "fof_linking": fof_linking,
            "file_index": file_index,
            "parsed_ok": True,
        }
    )
    return record


def load_filenames_from_file(path: Path) -> list[str]:
    """Load filenames from a local text file."""
    if not path.is_file():
        raise FileNotFoundError(f"Filename list not found: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_filenames_from_dir(path: Path) -> list[str]:
    """Load filenames from a local directory."""
    if not path.is_dir():
        raise NotADirectoryError(f"Input directory not found: {path}")
    files = sorted(
        [item.name for item in path.iterdir() if item.is_file() and item.suffix.lower() in {".hdf5", ".h5"}]
    )
    return files


def load_filenames_over_ssh(host: str, remote_dir: str) -> list[str]:
    """List remote HDF5 filenames over SSH without downloading data."""
    remote_command = (
        "find "
        + subprocess.list2cmdline([remote_dir]).replace('"', "'")
        + " -maxdepth 1 -type f \\( -name '*.hdf5' -o -name '*.h5' \\) -printf '%f\\n' | sort"
    )
    command = ["ssh", host, remote_command]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("`ssh` command not found. Install SSH client tools first.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() or "SSH command failed."
        raise RuntimeError(
            "Unable to list remote filenames over SSH. "
            "Check your ~/.ssh/config host alias, SSH keys, and remote directory.\n"
            f"SSH stderr: {stderr}"
        ) from exc
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def resolve_filenames(args: argparse.Namespace) -> list[str]:
    """Resolve filenames from exactly one supported input source."""
    sources = [args.from_file is not None, args.from_dir is not None, args.ssh_host is not None or args.remote_dir is not None]
    if sum(sources) != 1:
        raise ValueError("Choose exactly one input source: --from-file, --from-dir, or --ssh-host with --remote-dir.")

    if args.from_file is not None:
        return load_filenames_from_file(args.from_file)

    if args.from_dir is not None:
        return load_filenames_from_dir(args.from_dir)

    if not args.ssh_host or not args.remote_dir:
        raise ValueError("Remote manifest creation requires both --ssh-host and --remote-dir.")
    return load_filenames_over_ssh(args.ssh_host, args.remote_dir)


def write_manifest(rows: Iterable[dict[str, object]], output_path: Path) -> None:
    """Write manifest rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-file", type=Path, help="Text file containing one filename per line.")
    parser.add_argument("--from-dir", type=Path, help="Directory containing .hdf5 or .h5 files.")
    parser.add_argument("--ssh-host", help="SSH host alias from ~/.ssh/config.")
    parser.add_argument("--remote-dir", help="Remote directory to list over SSH.")
    parser.add_argument("--output", type=Path, default=Path("outputs/manifest.csv"))
    return parser.parse_args()


def main() -> int:
    """Create a manifest CSV from filename metadata."""
    args = parse_args()
    try:
        filenames = resolve_filenames(args)
        rows = [parse_filename(name) for name in filenames]
        write_manifest(rows, args.output)
    except (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parsed_ok = sum(bool(row["parsed_ok"]) for row in rows)
    print(f"Wrote manifest with {len(rows)} rows to {args.output}")
    print(f"Parsed successfully: {parsed_ok}/{len(rows)}")
    if parsed_ok != len(rows):
        print("Some filenames did not match the expected pattern. Inspect parsed_ok == False rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
