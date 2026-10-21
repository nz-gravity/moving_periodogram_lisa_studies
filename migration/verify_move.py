"""Verify moved files against their original hashes and documented source edits."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Check versioned migration files; local result archives are not required.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "migration/library-study-move.json").read_text()
    )
    receipt = json.loads((root / "migration/verification.json").read_text())
    # A fresh clone intentionally omits the multi-gigabyte result archive.
    # Use the Git index so source-only verification checks exactly the files
    # that are included in a commit, without silently skipping missing files.
    tracked = (
        set(
            subprocess.check_output(
                ["git", "ls-files", "-z"], cwd=root, text=True
            ).split("\0")
        )
        if args.source_only
        else None
    )
    checked = 0
    for row in manifest["files"]:
        if tracked is not None and row["destination"] not in tracked:
            continue
        checked += 1
        path = root / row["destination"]
        if "symlink" in row:
            assert path.is_symlink() and path.exists(), path
            continue
        expected = (
            receipt["adapted_sources"]
            .get(row["destination"], {})
            .get("current_sha256", row["sha256"])
        )
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(chunk)
        assert checksum.hexdigest() == expected, path
    print(f"Verified {checked} moved files and links.")


if __name__ == "__main__":
    main()
