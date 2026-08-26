from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile

import pandas as pd


DISPLAY_REPLACEMENTS = (
    ("Bound Mass Fraction", "Mass Fraction"),
    ("bound mass fraction", "mass fraction"),
    ("Local spread in BMF", "Local spread in mass fraction"),
    ("ΔBMF", "ΔMass fraction"),
    ("observed BMF", "observed mass fraction"),
    ("Observed mean BMF = 0", "Observed mean mass fraction = 0"),
    ("Mean observed BMF", "Mean observed mass fraction"),
    ("Actual BMF", "Actual mass fraction"),
    ("BMF", "Mass Fraction"),
)


def replace_display_text(value):
    if not isinstance(value, str):
        return value
    text = value
    for old, new in DISPLAY_REPLACEMENTS:
        text = text.replace(old, new)
    return text


@contextmanager
def patched_text_labels():
    import matplotlib.text as matplotlib_text

    original_set_text = matplotlib_text.Text.set_text

    def wrapped_set_text(self, s):
        return original_set_text(self, replace_display_text(s))

    matplotlib_text.Text.set_text = wrapped_set_text
    try:
        yield
    finally:
        matplotlib_text.Text.set_text = original_set_text


@contextmanager
def aliased_bound_dataset(source_path: Path, target_column: str = "captured_mass_fraction"):
    frame = pd.read_csv(source_path, low_memory=False)
    if target_column not in frame.columns:
        raise KeyError(f"{source_path} is missing {target_column}.")
    frame["bound_mass_fraction"] = pd.to_numeric(frame[target_column], errors="coerce")
    with tempfile.TemporaryDirectory(prefix="cmf_bound_alias_") as tmpdir:
        temp_path = Path(tmpdir) / "bound_outcomes_alias.csv"
        frame.to_csv(temp_path, index=False)
        yield temp_path


@contextmanager
def aliased_oof_predictions(source_path: Path):
    frame = pd.read_csv(source_path, low_memory=False)
    if "target" in frame.columns:
        frame["target"] = "bound_mass_fraction"
    with tempfile.TemporaryDirectory(prefix="cmf_oof_alias_") as tmpdir:
        temp_path = Path(tmpdir) / "oof_predictions_alias.csv"
        frame.to_csv(temp_path, index=False)
        yield temp_path
