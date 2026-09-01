"""What this repo is allowed to own.

The split gave the public repo the MediaPipe->SO-101 pipeline and the optional
PressureVision integration; `ir_force` owns the FLIR/Lepton/thermal line. A copy
of a public module landing here is the failure that would not announce itself:
imports keep working, tests keep passing, and the two copies drift until a fix
applied to one silently does not apply to the other.
"""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
IR_FORCE = REPO / "ir_force"

#: Public modules, by the names they have in mediapipe-so101. If one of these
#: appears under ir_force/ it was copied instead of imported.
PUBLIC_MODULES = [
    "ee_controller.py",
    "ee_control.py",
    "webcam_source.py",
    "wrist_estimator.py",
    "depth.py",
    "config_so101_webcam.py",
    "config_so101_webcam_ee.py",
    "so101_webcam.py",
    "so101_webcam_ee.py",
    "paths.py",
    "servo_pid.py",
]


@pytest.mark.parametrize("name", PUBLIC_MODULES)
def test_private_ir_package_does_not_own_public_module(name):
    assert not (IR_FORCE / name).exists(), f"ir_force/{name} duplicates a public module"


def test_pressurevision_modules_stayed_in_the_public_repo():
    """`pv_*` is the public PressureVision integration, not part of the IR line."""
    strays = sorted(p.relative_to(REPO).as_posix()
                    for p in IR_FORCE.rglob("pv_*.py"))
    assert strays == [], f"PressureVision modules belong to mediapipe-so101: {strays}"


def test_the_public_pv_recorder_did_not_come_along():
    strays = sorted(p.relative_to(REPO).as_posix()
                    for p in REPO.rglob("record_so101_pv_ee.py")
                    if "local" not in p.parts)
    assert strays == []


def test_ir_force_actually_has_content():
    """Guards the guards: every assertion above passes vacuously on an empty tree."""
    modules = list(IR_FORCE.glob("*.py"))
    assert len(modules) >= 10, f"only {len(modules)} modules in ir_force/"
