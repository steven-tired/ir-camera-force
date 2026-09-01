import time

import numpy as np
import pytest

from ir_force.ir_capture import FrameSample
from ir_force.ir_hand_calibration import ProjectionCalibration
from ir_force.ir_pressure import (
    HandPressureEstimator,
    PressureConfig,
    PressureReading,
    lepton_pressure_config,
)
from ir_force.types import LandmarksData


class FakeThermal:
    def __init__(self, frames):
        self.frames = list(frames)

    def read(self):
        frame = self.frames.pop(0)
        if isinstance(frame, FrameSample):
            return frame
        return FrameSample(t=time.perf_counter(), frame=frame)


class RaisingThermal:
    def read(self):
        raise RuntimeError("thermal read failed")


def _calibration():
    return ProjectionCalibration(
        coeff_x=(0.0, 160.0, 0.0, 0.0),
        coeff_y=(0.0, 0.0, 128.0, 0.0),
        rms_error_px=2.0,
        max_error_px=4.0,
        sample_count=12,
        image_size=(160, 128),
    )


_DEFAULT_TIMESTAMP = object()


def _landmarks(*, observed_at_s=_DEFAULT_TIMESTAMP, frame_id=0):
    if observed_at_s is _DEFAULT_TIMESTAMP:
        observed_at_s = time.perf_counter()
    image_xy = np.zeros((21, 2), dtype=float)
    depth_m = np.full((21,), 0.6, dtype=float)
    image_xy[4] = [0.45, 0.50]
    image_xy[8] = [0.50, 0.52]
    return LandmarksData(
        np.zeros((21, 3)),
        True,
        image_xy=image_xy,
        depth_m=depth_m,
        observed_at_s=observed_at_s,
        frame_id=frame_id,
    )


def _frame(value):
    pattern = np.indices((128, 160)).sum(axis=0) % 5
    gray = np.clip(value + pattern, 0, 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def test_estimator_updates_baseline_while_inactive_then_reports_pressure_when_active():
    frames = [_frame(20), _frame(20), _frame(32)]
    estimator = HandPressureEstimator(
        calibration=_calibration(),
        thermal_source=FakeThermal(frames),
        config=PressureConfig(near_contact_pinch=0.05, full_scale_delta=12.0),
    )

    inactive = estimator.update(_landmarks(), pinch=0.08, enabled=True)
    active_entry = estimator.update(_landmarks(), pinch=0.04, enabled=True)
    active = estimator.update(_landmarks(), pinch=0.04, enabled=True)

    assert inactive.active is False
    assert inactive.available is True
    assert inactive.roi_mode == "tips"
    assert active_entry.active is True
    assert active_entry.roi_mode == "tips"
    assert active.active is True
    assert active.pressure_0_1 == 1.0
    assert active.roi_mode == "tips"


def test_estimator_returns_fallback_when_oak_metadata_is_missing():
    estimator = HandPressureEstimator(
        calibration=_calibration(),
        thermal_source=FakeThermal([_frame(20)]),
        config=PressureConfig(),
    )

    reading = estimator.update(
        LandmarksData(
            np.zeros((21, 3)),
            True,
            observed_at_s=time.perf_counter(),
            frame_id=0,
        ),
        pinch=0.04,
        enabled=True,
    )

    assert reading.available is False
    assert reading.status == "missing_oak_metadata"
    assert reading.pressure_0_1 == 0.0
    assert reading.roi_mode == "missing_oak_metadata"


def test_pressure_reading_keeps_existing_positional_construction_with_optional_timing():
    reading = PressureReading(0.0, False, 0.0, False, "oak_stale", None)

    assert reading.oak_observed_at_s is None
    assert reading.thermal_observed_at_s is None
    assert reading.sensor_skew_s is None
    assert reading.oak_age_s is None
    assert reading.thermal_age_s is None
    assert reading.roi_mode is None


def test_pressure_reading_accepts_roi_mode_as_optional_trailing_positional_field():
    reading = PressureReading(
        0.0,
        False,
        1.0,
        True,
        "baseline",
        None,
        None,
        None,
        None,
        None,
        None,
        "tips",
    )

    assert reading.roi_mode == "tips"


@pytest.mark.parametrize(
    ("oak_t", "thermal_t"),
    [(99.8, 99.95), (99.95, 99.8)],
)
def test_estimator_accepts_exact_age_and_pair_skew_boundaries(monkeypatch, oak_t, thermal_t):
    monkeypatch.setattr(
        "ir_force.ir_pressure.time.perf_counter",
        lambda: 100.0,
    )
    estimator = HandPressureEstimator(
        calibration=_calibration(),
        thermal_source=FakeThermal([FrameSample(t=thermal_t, frame=_frame(20))]),
        config=PressureConfig(
            max_oak_age_s=0.20,
            max_thermal_age_s=0.20,
            max_pair_skew_s=0.15,
        ),
    )

    reading = estimator.update(_landmarks(observed_at_s=oak_t), pinch=0.08, enabled=True)

    assert reading.available is True
    assert reading.status == "baseline"
    assert reading.sensor_skew_s == pytest.approx(0.15)


@pytest.mark.parametrize("newer_sensor", ["oak", "thermal"])
def test_estimator_accepts_reconstructed_boundaries_at_perf_counter_magnitude(
    monkeypatch,
    newer_sensor,
):
    now_s = 1_000_000.0
    older_t = now_s - 0.20
    newer_t = older_t + 0.15
    oak_t, thermal_t = (
        (newer_t, older_t) if newer_sensor == "oak" else (older_t, newer_t)
    )
    monkeypatch.setattr(
        "ir_force.ir_pressure.time.perf_counter",
        lambda: now_s,
    )
    estimator = HandPressureEstimator(
        calibration=_calibration(),
        thermal_source=FakeThermal([FrameSample(t=thermal_t, frame=_frame(20))]),
        config=PressureConfig(
            max_oak_age_s=0.20,
            max_thermal_age_s=0.20,
            max_pair_skew_s=0.15,
        ),
    )

    reading = estimator.update(_landmarks(observed_at_s=oak_t), pinch=0.08, enabled=True)

    assert abs(oak_t - thermal_t) > 0.15
    assert reading.available is True
    assert reading.status == "baseline"


@pytest.mark.parametrize(
    ("oak_t", "thermal_t"),
    [(99.8, 99.950001), (99.950001, 99.8)],
)
def test_estimator_rejects_pair_skew_just_over_boundary_in_both_directions(
    monkeypatch,
    oak_t,
    thermal_t,
):
    monkeypatch.setattr(
        "ir_force.ir_pressure.time.perf_counter",
        lambda: 100.0,
    )
    estimator = HandPressureEstimator(
        calibration=_calibration(),
        thermal_source=FakeThermal([FrameSample(t=thermal_t, frame=_frame(20))]),
        config=PressureConfig(
            max_oak_age_s=0.20,
            max_thermal_age_s=0.20,
            max_pair_skew_s=0.15,
        ),
    )

    reading = estimator.update(_landmarks(observed_at_s=oak_t), pinch=0.08, enabled=True)

    assert reading.available is False
    assert reading.status == "sensor_skew"


@pytest.mark.parametrize("newer_sensor", ["oak", "thermal"])
def test_estimator_rejects_one_nanosecond_over_skew_at_perf_counter_magnitude(
    monkeypatch,
    newer_sensor,
):
    now_s = 1_000_000.0
    newer_t = now_s - 0.01
    older_t = newer_t - (0.15 + 1e-9)
    oak_t, thermal_t = (
        (newer_t, older_t) if newer_sensor == "oak" else (older_t, newer_t)
    )
    monkeypatch.setattr(
        "ir_force.ir_pressure.time.perf_counter",
        lambda: now_s,
    )
    estimator = HandPressureEstimator(
        calibration=_calibration(),
        thermal_source=FakeThermal([FrameSample(t=thermal_t, frame=_frame(20))]),
        config=PressureConfig(
            max_oak_age_s=0.20,
            max_thermal_age_s=0.20,
            max_pair_skew_s=0.15,
        ),
    )

    reading = estimator.update(_landmarks(observed_at_s=oak_t), pinch=0.08, enabled=True)

    assert reading.available is False
    assert reading.status == "sensor_skew"


@pytest.mark.parametrize("oak_t", [None, np.nan, 100.001, 99.799999])
def test_estimator_rejects_missing_nonfinite_future_and_stale_oak_timestamps(
    monkeypatch,
    oak_t,
):
    monkeypatch.setattr(
        "ir_force.ir_pressure.time.perf_counter",
        lambda: 100.0,
    )
    estimator = HandPressureEstimator(
        calibration=_calibration(),
        thermal_source=FakeThermal([FrameSample(t=99.95, frame=_frame(20))]),
        config=PressureConfig(max_oak_age_s=0.20),
    )

    reading = estimator.update(_landmarks(observed_at_s=oak_t), pinch=0.08, enabled=True)

    assert reading.available is False
    assert reading.status == "oak_stale"


@pytest.mark.parametrize("thermal_t", [np.nan, 100.001, 99.799999])
def test_estimator_rejects_nonfinite_future_and_stale_thermal_timestamps(
    monkeypatch,
    thermal_t,
):
    monkeypatch.setattr(
        "ir_force.ir_pressure.time.perf_counter",
        lambda: 100.0,
    )
    estimator = HandPressureEstimator(
        calibration=_calibration(),
        thermal_source=FakeThermal([FrameSample(t=thermal_t, frame=_frame(20))]),
        config=PressureConfig(max_thermal_age_s=0.20),
    )

    reading = estimator.update(_landmarks(observed_at_s=99.95), pinch=0.08, enabled=True)

    assert reading.available is False
    assert reading.status == "thermal_stale"


def test_estimator_treats_retained_thermal_timestamp_as_pending(monkeypatch):
    monkeypatch.setattr(
        "ir_force.ir_pressure.time.perf_counter",
        lambda: 100.0,
    )
    estimator = HandPressureEstimator(
        calibration=_calibration(),
        thermal_source=FakeThermal(
            [
                FrameSample(t=99.95, frame=_frame(20)),
                FrameSample(t=99.95, frame=_frame(21)),
            ]
        ),
    )
    estimator.update(_landmarks(observed_at_s=99.95, frame_id=0), pinch=0.08, enabled=True)

    reading = estimator.update(
        _landmarks(observed_at_s=99.96, frame_id=1),
        pinch=0.08,
        enabled=True,
    )

    assert reading.available is True
    assert reading.status == "thermal_pending"
    assert reading.fresh is False


def test_estimator_keeps_duplicate_timestamp_pending_until_new_frame_arrives(monkeypatch):
    monkeypatch.setattr(
        "ir_force.ir_pressure.time.perf_counter",
        lambda: 100.0,
    )
    estimator = HandPressureEstimator(
        calibration=_calibration(),
        thermal_source=FakeThermal(
            [
                FrameSample(t=99.95, frame=_frame(20)),
                FrameSample(t=99.95, frame=_frame(21)),
                FrameSample(t=99.95, frame=_frame(22)),
                FrameSample(t=99.96, frame=_frame(23)),
            ]
        ),
    )

    first = estimator.update(
        _landmarks(observed_at_s=99.95, frame_id=0),
        pinch=0.08,
        enabled=True,
    )
    duplicate = estimator.update(
        _landmarks(observed_at_s=99.96, frame_id=1),
        pinch=0.08,
        enabled=True,
    )
    repeated_duplicate = estimator.update(
        _landmarks(observed_at_s=99.97, frame_id=2),
        pinch=0.08,
        enabled=True,
    )
    newer = estimator.update(
        _landmarks(observed_at_s=99.97, frame_id=3),
        pinch=0.08,
        enabled=True,
    )

    assert first.status == "baseline"
    assert first.fresh is True
    assert duplicate.status == "thermal_pending"
    assert duplicate.available is True
    assert duplicate.fresh is False
    assert repeated_duplicate.status == "thermal_pending"
    assert repeated_duplicate.available is True
    assert repeated_duplicate.fresh is False
    assert newer.status == "baseline"
    assert newer.available is True
    assert newer.fresh is True


def test_estimator_rejects_regressed_thermal_timestamp(monkeypatch):
    monkeypatch.setattr(
        "ir_force.ir_pressure.time.perf_counter",
        lambda: 100.0,
    )
    estimator = HandPressureEstimator(
        calibration=_calibration(),
        thermal_source=FakeThermal(
            [
                FrameSample(t=99.95, frame=_frame(20)),
                FrameSample(t=99.94, frame=_frame(21)),
            ]
        ),
    )
    estimator.update(
        _landmarks(observed_at_s=99.95, frame_id=0),
        pinch=0.08,
        enabled=True,
    )

    reading = estimator.update(
        _landmarks(observed_at_s=99.96, frame_id=1),
        pinch=0.08,
        enabled=True,
    )

    assert reading.available is False
    assert reading.status == "thermal_stale"


@pytest.mark.parametrize(
    ("oak_t", "thermal_t", "status"),
    [
        (99.70, 99.96, "oak_stale"),
        (99.96, 99.70, "thermal_stale"),
        (99.80, 99.96, "sensor_skew"),
    ],
)
def test_timing_rejections_reset_baseline_and_return_unavailable(
    monkeypatch,
    oak_t,
    thermal_t,
    status,
):
    monkeypatch.setattr(
        "ir_force.ir_pressure.time.perf_counter",
        lambda: 100.0,
    )
    estimator = HandPressureEstimator(
        calibration=_calibration(),
        thermal_source=FakeThermal(
            [
                FrameSample(t=99.95, frame=_frame(20)),
                FrameSample(t=thermal_t, frame=_frame(21)),
            ]
        ),
        config=PressureConfig(
            max_oak_age_s=0.20,
            max_thermal_age_s=0.20,
            max_pair_skew_s=0.15,
        ),
    )
    estimator.update(_landmarks(observed_at_s=99.95), pinch=0.08, enabled=True)
    assert estimator._baseline is not None

    reading = estimator.update(_landmarks(observed_at_s=oak_t), pinch=0.08, enabled=True)

    assert reading.available is False
    assert reading.status == status
    assert estimator._baseline is None


def test_estimator_rejects_runtime_thermal_shape_mismatch(monkeypatch):
    now = time.perf_counter()
    estimator = HandPressureEstimator(
        calibration=_calibration(),
        thermal_source=FakeThermal(
            [FrameSample(t=now, frame=np.ones((120, 160, 3), dtype=np.uint8) * 20)]
        ),
    )

    reading = estimator.update(_landmarks(observed_at_s=now), pinch=0.08, enabled=True)

    assert reading.available is False
    assert reading.status == "thermal_shape_mismatch"


def test_estimator_returns_projection_out_of_fov_for_invalid_tip_projection():
    landmarks = _landmarks()
    landmarks.image_xy[4] = [-0.01, 0.50]
    estimator = HandPressureEstimator(
        calibration=_calibration(),
        thermal_source=FakeThermal([_frame(20)]),
    )

    reading = estimator.update(landmarks, pinch=0.08, enabled=True)

    assert reading.available is False
    assert reading.status == "projection_out_of_fov"
    assert reading.roi_mode == "projection_out_of_fov"


def test_estimator_inactive_when_hand_control_is_disabled():
    estimator = HandPressureEstimator(
        calibration=_calibration(),
        thermal_source=FakeThermal([_frame(20)]),
        config=PressureConfig(),
    )

    reading = estimator.update(_landmarks(), pinch=0.04, enabled=False)

    assert reading.active is False
    assert reading.available is True
    assert reading.status == "disabled"


def test_estimator_returns_unavailable_when_thermal_read_raises():
    estimator = HandPressureEstimator(
        calibration=_calibration(),
        thermal_source=RaisingThermal(),
        config=PressureConfig(),
    )

    reading = estimator.update(_landmarks(), pinch=0.04, enabled=True)

    assert reading.active is False
    assert reading.available is False
    assert reading.status == "thermal_unavailable"


def test_estimator_returns_unavailable_for_covered_low_contrast_frame():
    estimator = HandPressureEstimator(
        calibration=_calibration(),
        thermal_source=FakeThermal([np.zeros((128, 160, 3), dtype=np.uint8)]),
    )

    reading = estimator.update(_landmarks(), pinch=0.04, enabled=True)

    assert reading.available is False
    assert reading.status == "thermal_low_contrast"


def test_estimator_returns_unavailable_for_stale_timestamp():
    class StaleThermal:
        def read(self):
            return FrameSample(t=time.perf_counter() - 2.0, frame=_frame(20))

    estimator = HandPressureEstimator(calibration=_calibration(), thermal_source=StaleThermal())

    reading = estimator.update(_landmarks(), pinch=0.04, enabled=True)

    assert reading.available is False
    assert reading.status == "thermal_stale"


def test_estimator_returns_unavailable_when_frame_stream_freezes():
    frame = _frame(20)
    estimator = HandPressureEstimator(
        calibration=_calibration(),
        thermal_source=FakeThermal([frame, frame.copy(), frame.copy()]),
        config=PressureConfig(max_repeated_frames=1),
    )

    estimator.update(_landmarks(), pinch=0.08, enabled=True)
    estimator.update(_landmarks(), pinch=0.08, enabled=True)
    reading = estimator.update(_landmarks(), pinch=0.08, enabled=True)

    assert reading.available is False
    assert reading.status == "thermal_stale"


def test_estimator_enforces_configured_minimum_roi_quality():
    landmarks = _landmarks()
    landmarks.depth_m[8] = np.nan
    estimator = HandPressureEstimator(
        calibration=_calibration(),
        thermal_source=FakeThermal([_frame(20)]),
        config=PressureConfig(min_quality=0.75),
    )

    reading = estimator.update(landmarks, pinch=0.04, enabled=True)

    assert reading.available is False
    assert reading.status == "low_quality"
    assert reading.quality == 0.5
    assert reading.roi_mode == "single_tip"


def test_estimator_requires_an_inactive_baseline_before_pressure_is_available():
    estimator = HandPressureEstimator(
        calibration=_calibration(),
        thermal_source=FakeThermal([_frame(20)]),
    )

    reading = estimator.update(_landmarks(), pinch=0.04, enabled=True)

    assert reading.active is True
    assert reading.available is False
    assert reading.status == "active_no_baseline"


def test_estimator_does_not_replace_baseline_when_active_roi_moves():
    frames = [_frame(20), _frame(30), _frame(32)]
    estimator = HandPressureEstimator(calibration=_calibration(), thermal_source=FakeThermal(frames))
    original = _landmarks()
    moved = _landmarks()
    moved.image_xy[4, 0] += 0.05
    moved.image_xy[8, 0] += 0.05

    estimator.update(original, pinch=0.08, enabled=True)
    moved_reading = estimator.update(moved, pinch=0.04, enabled=True)
    recovered = estimator.update(original, pinch=0.04, enabled=True)

    assert moved_reading.available is False
    assert moved_reading.status == "active_no_baseline"
    assert recovered.available is True
    assert recovered.pressure_0_1 > 0.0


def test_estimator_reset_requires_a_new_inactive_baseline():
    estimator = HandPressureEstimator(
        calibration=_calibration(),
        thermal_source=FakeThermal([_frame(20), _frame(30), _frame(32)]),
    )
    estimator.update(_landmarks(), pinch=0.08, enabled=True)
    assert estimator.update(_landmarks(), pinch=0.04, enabled=True).available is True

    estimator.reset()
    reading = estimator.update(_landmarks(), pinch=0.04, enabled=True)

    assert reading.available is False
    assert reading.status == "active_no_baseline"


def test_estimator_reset_clears_frozen_frame_history():
    frame = _frame(20)
    estimator = HandPressureEstimator(
        calibration=_calibration(),
        thermal_source=FakeThermal([frame, frame.copy(), frame.copy()]),
        config=PressureConfig(max_repeated_frames=1),
    )
    estimator.update(_landmarks(), pinch=0.08, enabled=True)
    estimator.update(_landmarks(), pinch=0.08, enabled=True)

    estimator.reset()
    reading = estimator.update(_landmarks(), pinch=0.08, enabled=True)

    assert reading.available is True
    assert reading.status == "baseline"


def test_estimator_close_releases_thermal_source_once():
    class CloseableThermal:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    thermal = CloseableThermal()
    estimator = HandPressureEstimator(calibration=_calibration(), thermal_source=thermal)

    estimator.close()
    estimator.close()

    assert thermal.close_calls == 1


# ---------------------------------------------------------------------------
# Blob ROI mode (Lepton path: raw uint16 frames, no projection calibration)
# ---------------------------------------------------------------------------


def _lepton_frame(background=8000, blob_delta=0):
    pattern = np.indices((120, 160)).sum(axis=0) % 5
    frame = (background + pattern).astype(np.uint16)
    if blob_delta:
        frame[50:62, 70:90] += blob_delta
    return frame


def test_lepton_pressure_config_defaults():
    config = lepton_pressure_config()
    assert config.roi_mode == "blob"
    assert config.thermal_image_size == (160, 120)
    assert config.max_thermal_age_s >= 0.3


def test_blob_mode_reports_pressure_without_calibration():
    frames = [_lepton_frame(blob_delta=300), _lepton_frame(blob_delta=300)]
    estimator = HandPressureEstimator(
        calibration=None,
        thermal_source=FakeThermal(frames),
        config=lepton_pressure_config(),
    )

    baseline = estimator.update(_landmarks(), pinch=0.08, enabled=True)
    active = estimator.update(_landmarks(), pinch=0.04, enabled=True)

    assert baseline.status == "baseline"
    assert baseline.roi_mode == "blob"
    assert active.status == "active"
    assert active.roi_mode == "blob"
    assert active.roi is not None
    assert 0.0 < active.pressure_0_1 <= 1.0


def test_blob_mode_reports_no_hotspot_when_hand_absent():
    estimator = HandPressureEstimator(
        calibration=None,
        thermal_source=FakeThermal([_lepton_frame()]),
        config=lepton_pressure_config(),
    )

    reading = estimator.update(_landmarks(), pinch=0.04, enabled=True)

    assert reading.status == "blob_no_hotspot"
    assert reading.available is False


def test_blob_mode_checks_frame_shape_against_config():
    wrong_shape = np.full((128, 160), 8000, dtype=np.uint16)
    estimator = HandPressureEstimator(
        calibration=None,
        thermal_source=FakeThermal([wrong_shape]),
        config=lepton_pressure_config(),
    )

    reading = estimator.update(_landmarks(), pinch=0.04, enabled=True)

    assert reading.status == "thermal_shape_mismatch"


def test_projection_mode_requires_calibration():
    with pytest.raises(ValueError):
        HandPressureEstimator(
            calibration=None,
            thermal_source=FakeThermal([]),
            config=PressureConfig(),
        )
