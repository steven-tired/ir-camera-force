"""The thermal sensor is slower than the control loop, and that is not a fault.

Migrated from the worktree's controller-pressure tests, where it needed a fake
robot and a full IK pipeline to reach. The behaviour is the runtime's: a frame
the thermal camera has not published yet holds the current policy without
advancing state or filters, and only a genuinely stale frame latches a fault.

Confusing "not yet" with "gone" in either direction is a real failure: latching
on every pending frame would make the sensor useless at FLIR's cadence, while
never latching would keep commanding force from a frame that has expired.
"""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from ir_force.ir_capture import FrameSample, LatestFrameSource
from ir_force.ir_hand_calibration import ProjectionCalibration
from ir_force.ir_pressure import HandPressureEstimator, PressureConfig
from pressurevision_integration.pv_grip_controller import PressureVisionGripRuntime

CALIBRATION = ProjectionCalibration(
    coeff_x=(0.0, 160.0, 0.0, 0.0),
    coeff_y=(0.0, 0.0, 128.0, 0.0),
    rms_error_px=2.0,
    max_error_px=4.0,
    sample_count=12,
    image_size=(160, 128),
)


def _thermal_frame(value):
    pattern = np.indices((128, 160)).sum(axis=0) % 5
    gray = np.clip(value + pattern, 0, 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


class _ControlledSlowThermalSource:
    """A thermal camera that publishes only when told to."""

    def __init__(self):
        self._release = threading.Event()
        self._closed = threading.Event()
        self._read_count = 0

    def read(self):
        self._read_count += 1
        if self._read_count > 1:
            self._release.wait()
            self._release.clear()
        if self._closed.is_set():
            raise RuntimeError("thermal source closed")
        return FrameSample(
            t=time.perf_counter(),
            frame=_thermal_frame(20 if self._read_count == 1 else 32),
        )

    def publish_next(self):
        self._release.set()

    def close(self):
        self._closed.set()
        self._release.set()


def _wait_for_latest(latest, *, newer_than=None, timeout_s=1.0):
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            sample = latest.read()
        except RuntimeError:
            time.sleep(0.001)
            continue
        if newer_than is None or sample.t > newer_than:
            return sample
        time.sleep(0.001)
    raise AssertionError("thermal producer did not publish in time")


def _landmarks(pinch=0.04):
    """A landmark frame carrying image coordinates and depth.

    Deliberately duck-typed rather than `webcam_input.types.LandmarksData`: the
    public dataclass carries only landmarks and validity, while the thermal
    estimator needs the image projection and per-point depth to place its ROI.
    Using a stand-in keeps that requirement visible here instead of pushing IR
    fields into the public type.
    """
    points = np.zeros((21, 3), dtype=float)
    points[8, 0] = pinch
    image_xy = np.zeros((21, 2), dtype=float)
    image_xy[4] = [0.45, 0.50]
    image_xy[8] = [0.50, 0.52]
    return SimpleNamespace(
        landmarks=points,
        valid=True,
        image_xy=image_xy,
        depth_m=np.full(21, 0.6, dtype=float),
        observed_at_s=time.perf_counter(),
        frame_id=0,
    )


def _drive(runtime, command, *, base_gripper=60.0, pinch=0.04):
    """One control frame in shadow mode, returning the PV proposal.

    Shadow is the only mode the IR path was ever approved for, so the legacy
    smoother is what actually reaches the gripper; the proposal is computed
    beside it. Asserting on the proposal is asserting on the thing under test.
    """
    runtime.update(
        base_gripper=base_gripper,
        landmarks=_landmarks(pinch),
        pinch=pinch,
        enabled=True,
        current_command=command,
        observed_at_s=time.perf_counter(),
        smooth_legacy=lambda: command,
    )
    return runtime.last_pressure_control.proposed_gripper


def test_a_pending_thermal_frame_holds_and_only_a_stale_one_latches():
    thermal = _ControlledSlowThermalSource()
    latest = LatestFrameSource(thermal)
    first_sample = _wait_for_latest(latest)
    estimator = HandPressureEstimator(
        calibration=CALIBRATION,
        thermal_source=latest,
        config=PressureConfig(
            max_oak_age_s=0.20,
            max_thermal_age_s=0.10,
            max_pair_skew_s=0.20,
        ),
    )
    runtime = PressureVisionGripRuntime(
        estimator,
        initial_gripper=50.0,
        middle_gripper=50.0,
        pressure_shadow=True,
        pv_mapping="absolute",
    )
    try:
        # An open hand first: the estimator needs a no-contact frame to take
        # its pixel baseline from.
        baseline = _drive(runtime, 50.0, pinch=0.08)
        baseline_pixels = estimator._baseline.copy()

        # Nothing new published yet: hold, and do not touch the baseline.
        pending = _drive(runtime, baseline)

        assert runtime.last_pressure.status == "thermal_pending"
        assert runtime.last_pressure.fresh is False
        assert runtime.last_pressure_control.state == "armed"
        assert pending == pytest.approx(baseline)
        assert estimator._active is False
        np.testing.assert_array_equal(estimator._baseline, baseline_pixels)

        thermal.publish_next()
        _wait_for_latest(latest, newer_than=first_sample.t)
        active = _drive(runtime, pending)
        active_raw = runtime.pressure_raw_gripper
        active_proposal = runtime.last_pressure_control.proposed_gripper
        assert runtime.last_pressure.status == "active"

        for _ in range(3):
            retained = _drive(runtime, active)
            assert retained == pytest.approx(active)
            assert runtime.last_pressure.status == "thermal_pending"
            assert runtime.last_pressure_control.state == "armed"
            assert runtime.pressure_raw_gripper == active_raw
            assert runtime.last_pressure_control.proposed_gripper == active_proposal
            assert estimator._active is True
            np.testing.assert_array_equal(estimator._baseline, baseline_pixels)

        time.sleep(0.12)
        stale = _drive(runtime, active)

        assert runtime.last_pressure.status == "thermal_stale"
        assert runtime.last_pressure.available is False
        assert runtime.last_pressure_control.state == "fault_latched"
        assert stale == pytest.approx(active)
    finally:
        runtime.close()
