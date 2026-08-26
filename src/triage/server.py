"""Installable entrypoint for the local Mars flyby dashboard/API server."""

from __future__ import annotations

from .dashboard import main as run_dashboard


def main() -> None:
    run_dashboard()


if __name__ == "__main__":
    main()
