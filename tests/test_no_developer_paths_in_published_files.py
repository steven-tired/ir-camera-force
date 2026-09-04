"""Nothing this repository publishes may name a developer's home directory.

Before this repo went public, 50 tracked files carried absolute paths into one
home directory — and ~35 of them were not documentation but live `default=`
values in `experiments/`, so a clone did not merely read wrong instructions, it
ran against a directory that does not exist. `ir_force/data_paths.py` resolves
those now; this test is what stops them coming back.

`hardware/` is exempt. It is third-party code carried verbatim at a recorded
upstream commit (see THIRD_PARTY_NOTICES.md); the paths in it are the upstream
authors' and rewriting them would make the trees diverge from their upstream
for no gain.
"""

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Assembled rather than written literally, for the same reason mediapipe-so101's
#: `test_public_boundary.py` assembles its marker: a test that checks for a
#: string is itself a file containing that string, and would fail on its own
#: source.
HOME_PATH = re.compile("/" + "home" + r"/[A-Za-z0-9_.-]+/")

EXEMPT_TREES = ("hardware",)


def published_files():
    """Every tracked text file, minus the vendored upstream trees."""
    listed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.split("\0")
    for name in listed:
        if not name or name.split("/")[0] in EXEMPT_TREES:
            continue
        path = REPO / name
        try:
            yield name, path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue  # binary asset, or a file staged for deletion


def test_no_published_file_names_a_developer_home_directory():
    offenders = sorted(name for name, text in published_files() if HOME_PATH.search(text))
    assert not offenders, offenders


def test_the_scan_actually_reaches_the_trees_it_names():
    """Guards the guard: an empty scan passes the check above vacuously."""
    scanned = {name.split("/")[0] for name, _ in published_files()}
    assert {"ir_force", "experiments", "tests", "calibration", "docs"} <= scanned, scanned
