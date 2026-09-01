"""Make the public mediapipe-so101 packages importable for this repo's tests.

`ir-camera-force` depends one-way on the public repo (see
docs/PUBLIC_INTERFACE_LOCK.md). This environment additionally still carries an
editable install pointing at the *pre-split* `hand-teleop/webcam-input/` tree,
which `ir_pressure_soak` refuses to load. Prepending the public checkout's src
dirs makes the sibling checkout win over that stale .pth entry, the same way
mediapipe-so101's own `scripts/_common.sh` does.
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
