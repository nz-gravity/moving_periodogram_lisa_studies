"""Locate library source and historical products after the repository split."""

from __future__ import annotations

import json
from pathlib import Path

import log_psplines

STUDY_REPO = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(log_psplines.__file__).resolve().parent
LIBRARY_REPO = PACKAGE_ROOT.parents[1]


def source_path(relative: str | Path) -> Path:
    """Resolve a source ledger entry against the repository that owns it."""
    path = Path(relative)
    if path.parts[:2] == ("src", "log_psplines"):
        return PACKAGE_ROOT.joinpath(*path.parts[2:])
    return STUDY_REPO / path


def source_label(path: str | Path) -> str:
    """Keep stable relative ledger names for sources in either repository."""
    path = Path(path).resolve()
    if path.is_relative_to(PACKAGE_ROOT):
        return str(Path("src/log_psplines") / path.relative_to(PACKAGE_ROOT))
    return str(path.relative_to(STUDY_REPO))


def relocated_path(path: str | Path) -> Path:
    """Read old absolute product paths without rewriting archived evidence."""
    path = Path(path)
    old_studies = LIBRARY_REPO / "experiments/dw_wdm_small"
    if path.is_relative_to(old_studies):
        return (
            STUDY_REPO
            / "experiments/dw_wdm_small"
            / path.relative_to(old_studies)
        )
    return path


def read_json(path: str | Path):
    """Relocate product-path values in memory; leave hash-ledger keys intact."""

    def relocate(value):
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, str):
            relocated = relocated_path(value)
            return str(relocated) if relocated != Path(value) else value
        return value

    return relocate(json.loads(relocated_path(path).read_text()))
