"""Where this checkout looks for its data, and how to point that somewhere else.

Every dataset root and every reference to the surrounding workspace used to be
an absolute path into one developer's home directory. They are resolved here
instead, so a clone runs without editing anything, and so a machine that keeps
its recordings or its calibration runs elsewhere can say so once:

    export IR_FORCE_DATA_ROOT=/mnt/big/ir-datasets
    export IR_FORCE_WORKSPACE_ROOT=~/robots/hand-teleop

Unset, both fall back to what the absolute paths used to name: `local/datasets/`
inside this checkout (git-ignored), and the directory this checkout sits in.

Nothing here checks that a path exists. Callers that need a file which only a
recording rig has — a frozen calibration XML, a workspace launcher script —
should say so at their own call site, so the failure names the missing file
rather than a missing environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

CHECKOUT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT_ENV = "IR_FORCE_DATA_ROOT"
WORKSPACE_ROOT_ENV = "IR_FORCE_WORKSPACE_ROOT"
CALIBRATION_RUNS_ENV = "IR_FORCE_CALIBRATION_RUNS"


def _from_env(name: str, fallback: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else fallback


def data_root() -> Path:
    """Root holding every recorded dataset."""
    return _from_env(DATA_ROOT_ENV, CHECKOUT_ROOT / "local" / "datasets")


def dataset_root(name: str) -> Path:
    """Root of one named dataset, e.g. `dataset_root("ir_foam_compression")`."""
    return data_root() / name


def workspace_root() -> Path:
    """The meta-workspace this checkout sits in.

    Holds the sibling repositories and the shared `scripts/` launchers. Only
    the live-rig programs reach outside the checkout like this; the analysis
    programs do not.
    """
    return _from_env(WORKSPACE_ROOT_ENV, CHECKOUT_ROOT.parent)


def calibration_runs_root() -> Path:
    """Root of the dated RealSense-Lepton calibration runs.

    These are large, are produced by the calibration rig, and are not carried
    in this repository — see docs/CALIBRATION_PROVENANCE.md.
    """
    return _from_env(
        CALIBRATION_RUNS_ENV, workspace_root() / "thermal-project-calibration-runs"
    )
