"""The launchers must run this checkout and refuse before touching hardware.

These are the first wrappers in this repository, and they start a robot that
moves on its own while PressureVision drives the grip. Two things matter: they
resolve their own paths (the public repo's wrapper test exists because a stale
installed copy once got exercised instead of the checkout), and a run with
nothing configured fails loudly rather than starting a camera or an arm.
"""

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
WRAPPERS = sorted(p for p in SCRIPTS.glob("*.sh") if p.name != "_common.sh")


def test_the_expected_wrappers_exist():
    assert {p.name for p in WRAPPERS} == {
        "run_pv_carton_soft_direct_apply.sh",
        "run_pv_carton_span_apply.sh",
    }


@pytest.mark.parametrize("wrapper", WRAPPERS, ids=lambda p: p.name)
def test_wrapper_is_executable(wrapper):
    assert os.access(wrapper, os.X_OK), f"{wrapper.name} is not executable"


@pytest.mark.parametrize("wrapper", WRAPPERS, ids=lambda p: p.name)
def test_wrapper_hardcodes_no_developer_path(wrapper):
    body = wrapper.read_text(encoding="utf-8")
    assert "/home/" not in body, wrapper.name


def test_the_common_file_hardcodes_no_developer_path():
    assert "/home/" not in (SCRIPTS / "_common.sh").read_text(encoding="utf-8")


@pytest.mark.parametrize("wrapper", WRAPPERS, ids=lambda p: p.name)
def test_wrapper_says_the_arm_moves(wrapper):
    body = wrapper.read_text(encoding="utf-8")
    assert "THE ARM MOVES" in body, wrapper.name


@pytest.mark.parametrize("wrapper", WRAPPERS, ids=lambda p: p.name)
def test_wrapper_refuses_without_fitted_levels(wrapper):
    """PV_LEVELS is the one thing that cannot be defaulted: a stale levels.json
    silently changes what a pressure reading means."""
    env = dict(os.environ)
    env.pop("PV_LEVELS", None)
    out = subprocess.run(
        ["bash", str(wrapper)], capture_output=True, text=True, env=env, timeout=60
    )
    assert out.returncode != 0
    assert "PV_LEVELS" in out.stderr


def test_the_public_repo_is_located_not_assumed(tmp_path):
    """The dependency is one-way and the public checkout may live anywhere;
    the wrappers must say so rather than fail obscurely."""
    env = dict(os.environ)
    env["MEDIAPIPE_SO101_DIR"] = str(tmp_path / "not-a-checkout")
    env.pop("PV_LEVELS", None)
    out = subprocess.run(
        ["bash", str(SCRIPTS / "run_pv_carton_soft_direct_apply.sh")],
        capture_output=True, text=True, env=env, timeout=60,
    )
    assert out.returncode == 2
    assert "MEDIAPIPE_SO101_DIR" in out.stderr
