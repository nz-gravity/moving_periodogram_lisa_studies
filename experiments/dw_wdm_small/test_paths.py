"""Source ownership and archived paths survive the repository split."""

import json
from pathlib import Path

import log_psplines

from experiments.dw_wdm_small.paths import (
    LIBRARY_REPO,
    STUDY_REPO,
    read_json,
    source_path,
)


def test_library_sources_are_resolved_from_the_installed_package():
    expected = (
        Path(log_psplines.__file__).resolve().parent / "inference/power.py"
    )
    assert source_path("src/log_psplines/inference/power.py") == expected
    assert source_path("experiments/dw_wdm_small/benchmark.py").is_file()


def test_archived_product_paths_are_relocated_without_rewriting_files(
    tmp_path,
):
    old = str(LIBRARY_REPO / "experiments/dw_wdm_small/results/example")
    path = tmp_path / "inventory.json"
    original = json.dumps(
        {
            "path": old,
            "hashes": {old: "sha"},
            "description": "",
            "notation": "a//b",
        }
    )
    path.write_text(original)
    loaded = read_json(path)
    assert loaded["path"] == str(
        STUDY_REPO / "experiments/dw_wdm_small/results/example"
    )
    assert list(loaded["hashes"]) == [old]
    assert loaded["description"] == ""
    assert loaded["notation"] == "a//b"
    assert path.read_text() == original
