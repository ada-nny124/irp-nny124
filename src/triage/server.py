"""Installable entrypoint for the local Mars flyby dashboard/API server."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "app.py"


@lru_cache(maxsize=1)
def _load_app_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("triage_dashboard_script", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load dashboard script from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    _load_app_module().main()


if __name__ == "__main__":
    main()
