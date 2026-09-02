"""Make the public mediapipe-so101 packages importable for this repo's tests.

`ir-camera-force` depends one-way on the public repo (see
docs/PUBLIC_INTERFACE_LOCK.md). `pressurevision_integration` is not installed
into the environment at all, so without this the collection of three test
modules fails outright; `webcam_input` and `lerobot_teleoperator_so101_webcam`
would resolve through their editable installs, but prepending all three keeps
the answer the same either way — the sibling checkout wins, so these tests
always run against the source next to them rather than whatever happens to be
installed. That is the same rule mediapipe-so101's own `scripts/_common.sh`
applies.

Until 2026-09-02 this file said it existed to beat a stale editable install
pointing at the pre-split `hand-teleop/webcam-input/` tree. That install is
gone, and so is the tree. The mechanism stayed because the first reason above
is the real one.
"""

from pathlib import Path
import sys

PUBLIC_REPO = Path(__file__).resolve().parent.parent / "mediapipe-so101"

for src in (
    PUBLIC_REPO / "packages" / "webcam_input" / "src",
    PUBLIC_REPO / "packages" / "so101_teleop" / "src",
    PUBLIC_REPO / "integrations" / "pressurevision" / "src",
):
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
