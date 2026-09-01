import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import calibrate_oak_flir_hand_pressure as calibration_script
from calibrate_oak_flir_hand_pressure import (
    append_projection_sample,
    build_arg_parser,
    build_calibration_preview,
    find_warm_blob_center,
    load_projection_samples,
    preview_click_to_thermal_point,
)
from ir_force.ir_capture import FrameSample
from ir_force.ir_hand_calibration import (
    ProjectionCalibration,
    ProjectionSample,
)
from ir_force.types import LandmarksData


def _runtime_acceptable_blob_frame():
    frame = np.full((12, 16, 3), 40, dtype=np.uint8)
    frame[4:8, 9:13] = 120
    frame[5:7, 10:12] = 255
    return frame


def test_find_warm_blob_center_returns_centroid_of_hot_patch():
    frame = _runtime_acceptable_blob_frame()

    center = find_warm_blob_center(frame, min_area_px=4)

    assert center is not None
    assert abs(center[0] - 10.5) < 0.1
    assert abs(center[1] - 5.5) < 0.1


def test_find_warm_blob_center_rejects_tiny_hot_patch():
    frame = np.full((12, 16, 3), 40, dtype=np.uint8)
    frame[4:8, 9:13] = 120
    frame[5, 10] = 255

    assert find_warm_blob_center(frame, min_area_px=4) is None


def test_find_warm_blob_center_rejects_uniform_nonzero_runtime_frame():
    frame = np.full((128, 160, 3), 40, dtype=np.uint8)

    assert find_warm_blob_center(frame) is None


def test_find_warm_blob_center_rejects_reviewer_low_contrast_plateau_frame():
    frame = np.full((128, 160, 3), 40, dtype=np.uint8)
    frame[63:65, 79:81] = 42

    gray = frame[:, :, 0]
    assert np.percentile(gray, 95.0) - np.percentile(gray, 5.0) == 0.0
    assert find_warm_blob_center(frame) is None


def test_find_warm_blob_center_ignores_threshold_plateau_and_centers_hot_region():
    frame = np.full((128, 160, 3), 40, dtype=np.uint8)
    frame[30:70, 20:60] = 120
    frame[40:60, 25:45] = 200

    center = find_warm_blob_center(frame)

    assert center == pytest.approx((34.5, 49.5))
    assert center != pytest.approx((79.5, 63.5))


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_find_warm_blob_center_rejects_nonfinite_input(nonfinite):
    frame = _runtime_acceptable_blob_frame().astype(np.float32)
    frame[0, 0] = nonfinite

    assert find_warm_blob_center(frame) is None


def test_projection_sample_csv_round_trip(tmp_path: Path):
    path = tmp_path / "samples.csv"
    sample = ProjectionSample(oak_x=0.4, oak_y=0.5, oak_z=0.6, ir_x=70.0, ir_y=50.0)

    append_projection_sample(path, sample)
    append_projection_sample(path, sample)

    assert load_projection_samples(path) == [sample, sample]


def test_calibration_preview_shows_oak_and_scaled_thermal_side_by_side():
    hand = np.zeros((60, 80, 3), dtype=np.uint8)
    thermal = np.full((30, 40, 3), 40, dtype=np.uint8)
    thermal[10:20, 15:25] = 120
    thermal[13:17, 18:22] = 255
    image_xy = np.zeros((21, 2), dtype=float)
    image_xy[4] = [0.5, 0.5]
    landmarks = type(
        "Landmarks",
        (),
        {"valid": True, "image_xy": image_xy, "depth_m": np.full(21, 0.5)},
    )()

    preview = build_calibration_preview(hand, thermal, landmarks)

    assert preview.shape == (60, 160, 3)
    assert tuple(preview[30, 40]) == (0, 255, 0)
    assert tuple(preview[30, 120]) == (0, 255, 255)


def test_calibration_preview_autoscales_raw_uint16_lepton_frame():
    hand = np.zeros((60, 80, 3), dtype=np.uint8)
    thermal = np.full((120, 160), 30_000, dtype=np.uint16)
    thermal[30:90, 50:110] = 31_000
    landmarks = type(
        "Landmarks",
        (),
        {"valid": False, "image_xy": None, "depth_m": None},
    )()

    preview = build_calibration_preview(hand, thermal, landmarks)

    assert preview.shape == (60, 160, 3)
    assert preview.dtype == np.uint8
    assert int(preview[:, 80:, :].max()) > int(preview[:, 80:, :].min())


def _snapshot(observed_at_s, *, hand_depth_m=0.6, thumb_depth_m=None):
    image_xy = np.zeros((21, 2), dtype=float)
    image_xy[4] = [0.4, 0.5]
    depth_m = np.full(21, hand_depth_m)
    if thumb_depth_m is not None:
        depth_m[4] = thumb_depth_m
    landmarks = LandmarksData(
        np.zeros((21, 3)),
        True,
        image_xy=image_xy,
        depth_m=depth_m,
        observed_at_s=observed_at_s,
        frame_id=3,
    )
    return SimpleNamespace(
        preview_frame=np.zeros((12, 16, 3), dtype=np.uint8),
        wrist=None,
        landmarks=landmarks,
        observed_at_s=observed_at_s,
        frame_id=3,
    )


def _thermal_sample(observed_at_s):
    return FrameSample(t=observed_at_s, frame=_runtime_acceptable_blob_frame())


def _pair_projection_sample():
    pair = getattr(calibration_script, "pair_projection_sample", None)
    assert callable(pair), "timestamp-aware calibration pairing helper is missing"
    return pair


def test_projection_sample_csv_stores_timing_diagnostics_without_changing_fit_rows(tmp_path):
    path = tmp_path / "samples.csv"
    sample = ProjectionSample(oak_x=0.4, oak_y=0.5, oak_z=0.6, ir_x=70.0, ir_y=50.0)

    append_projection_sample(
        path,
        sample,
        oak_observed_at_s=10.0,
        thermal_observed_at_s=10.1,
        sensor_skew_s=0.1,
    )

    row = next(csv.DictReader(path.open(newline="", encoding="utf-8")))
    assert float(row["oak_observed_at_s"]) == 10.0
    assert float(row["thermal_observed_at_s"]) == 10.1
    assert float(row["sensor_skew_s"]) == 0.1
    assert load_projection_samples(path) == [sample]


def test_appending_timed_sample_upgrades_existing_coordinate_only_csv(tmp_path):
    path = tmp_path / "samples.csv"
    legacy = ProjectionSample(oak_x=0.1, oak_y=0.2, oak_z=0.3, ir_x=4.0, ir_y=5.0)
    current = ProjectionSample(oak_x=0.4, oak_y=0.5, oak_z=0.6, ir_x=7.0, ir_y=8.0)
    path.write_text("oak_x,oak_y,oak_z,ir_x,ir_y\n0.1,0.2,0.3,4.0,5.0\n", encoding="utf-8")

    append_projection_sample(
        path,
        current,
        oak_observed_at_s=10.0,
        thermal_observed_at_s=10.1,
        sensor_skew_s=0.1,
    )

    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    assert rows[0]["oak_observed_at_s"] == ""
    assert rows[1]["sensor_skew_s"] == "0.1"
    assert load_projection_samples(path) == [legacy, current]


def test_calibration_pair_uses_one_snapshot_and_records_host_timing():
    pair = _pair_projection_sample()(
        _snapshot(99.90),
        _thermal_sample(99.95),
        now_s=100.0,
        max_oak_age_s=0.20,
        max_thermal_age_s=0.20,
        max_pair_skew_s=0.15,
    )

    assert pair is not None
    assert pair.sample.oak_x == 0.4
    assert pair.oak_observed_at_s == 99.90
    assert pair.thermal_observed_at_s == 99.95
    assert pair.sensor_skew_s == pytest.approx(0.05)


def test_calibration_pair_uses_manual_thermal_point_and_working_depth_gate():
    pair = _pair_projection_sample()(
        _snapshot(99.90, hand_depth_m=0.3),
        _thermal_sample(99.95),
        ir_point=(7.0, 4.0),
        now_s=100.0,
        min_hand_depth_m=0.2,
        max_hand_depth_m=0.5,
    )

    assert pair is not None
    assert pair.sample.ir_x == 7.0
    assert pair.sample.ir_y == 4.0

    fallback = _pair_projection_sample()(
        _snapshot(99.90, hand_depth_m=0.32, thumb_depth_m=0.8),
        _thermal_sample(99.95),
        ir_point=(7.0, 4.0),
        now_s=100.0,
        min_hand_depth_m=0.2,
        max_hand_depth_m=0.5,
    )
    assert fallback is not None
    assert fallback.sample.oak_z == pytest.approx(0.32)

    rejected = _pair_projection_sample()(
        _snapshot(99.90, hand_depth_m=0.798),
        _thermal_sample(99.95),
        ir_point=(7.0, 4.0),
        now_s=100.0,
        min_hand_depth_m=0.2,
        max_hand_depth_m=0.5,
    )
    assert rejected is None


def test_preview_click_maps_scaled_thermal_half_back_to_sensor_pixels():
    point = preview_click_to_thermal_point(
        (640 + 4 * 70, 4 * 40),
        hand_shape=(480, 640),
        thermal_shape=(120, 160),
    )

    assert point == pytest.approx((70.0, 40.0))
    assert (
        preview_click_to_thermal_point(
            (639, 160),
            hand_shape=(480, 640),
            thermal_shape=(120, 160),
        )
        is None
    )


@pytest.mark.parametrize("newer_sensor", ["oak", "thermal"])
def test_calibration_pair_accepts_reconstructed_boundaries_at_perf_counter_magnitude(
    newer_sensor,
):
    now_s = 1_000_000.0
    older_t = now_s - 0.20
    newer_t = older_t + 0.15
    oak_t, thermal_t = (
        (newer_t, older_t) if newer_sensor == "oak" else (older_t, newer_t)
    )

    pair = _pair_projection_sample()(
        _snapshot(oak_t),
        _thermal_sample(thermal_t),
        now_s=now_s,
        max_oak_age_s=0.20,
        max_thermal_age_s=0.20,
        max_pair_skew_s=0.15,
    )

    assert abs(oak_t - thermal_t) > 0.15
    assert pair is not None


@pytest.mark.parametrize("newer_sensor", ["oak", "thermal"])
def test_calibration_pair_rejects_one_nanosecond_over_skew_at_perf_counter_magnitude(
    newer_sensor,
):
    now_s = 1_000_000.0
    newer_t = now_s - 0.01
    older_t = newer_t - (0.15 + 1e-9)
    oak_t, thermal_t = (
        (newer_t, older_t) if newer_sensor == "oak" else (older_t, newer_t)
    )

    pair = _pair_projection_sample()(
        _snapshot(oak_t),
        _thermal_sample(thermal_t),
        now_s=now_s,
        max_oak_age_s=0.20,
        max_thermal_age_s=0.20,
        max_pair_skew_s=0.15,
    )

    assert pair is None


@pytest.mark.parametrize(
    ("oak_t", "thermal_t"),
    [
        (None, 99.95),
        (np.nan, 99.95),
        (100.01, 99.95),
        (99.79, 99.95),
        (99.95, np.nan),
        (99.95, 100.01),
        (99.95, 99.79),
        (99.80, 99.96),
        (99.96, 99.80),
    ],
)
def test_calibration_pair_rejects_absent_nonfinite_stale_future_and_skewed_timing(
    oak_t,
    thermal_t,
):
    pair = _pair_projection_sample()(
        _snapshot(oak_t),
        _thermal_sample(thermal_t),
        now_s=100.0,
        max_oak_age_s=0.20,
        max_thermal_age_s=0.20,
        max_pair_skew_s=0.15,
    )

    assert pair is None


def test_calibration_pair_skew_flag_defaults_to_provisional_150_ms():
    args = build_arg_parser().parse_args([])

    assert args.max_pair_skew_ms == 150.0


def test_lepton_cli_defaults_to_port_8080_and_locked_three_pixel_gate():
    args = build_arg_parser().parse_args(["--lepton-udp"])

    assert args.lepton_udp == 8080
    assert args.hand_camera == "realsense"
    assert args.realsense_serial is None
    assert args.max_rms_px == 3.0
    assert args.max_error_px == 3.0
    assert args.min_hand_depth_m == 0.20
    assert args.max_hand_depth_m == 0.90


def test_lepton_calibration_target_uses_160x120_and_distinct_artifacts():
    target_factory = getattr(calibration_script, "calibration_target", None)
    assert callable(target_factory), "Lepton calibration target selection is missing"

    target = target_factory(SimpleNamespace(lepton_udp=9000))

    assert target.label == "Lepton"
    assert target.image_size == (160, 120)
    assert target.hand_label == "RealSense"
    assert target.samples_filename == "realsense_lepton_hand_pressure_samples.csv"
    assert target.projection_filename == "realsense_lepton_hand_pressure_projection.json"
    assert target.error_report_filename == "realsense_lepton_hand_pressure_error_report.json"


def test_lepton_gate_cannot_be_relaxed_from_cli():
    args = build_arg_parser().parse_args(
        ["--lepton-udp", "--max-rms-px", "3.1", "--max-error-px", "4.0"]
    )
    target = calibration_script.calibration_target(args)

    with pytest.raises(ValueError, match="Lepton reprojection gate is locked at 3.00 px"):
        calibration_script.validate_gate_configuration(args, target)


def test_error_report_records_each_residual_and_escalate_decision(tmp_path):
    samples = [
        ProjectionSample(0.1, 0.2, 0.3, 10.0, 20.0),
        ProjectionSample(0.2, 0.3, 0.4, 11.0, 21.0),
    ]
    calibration = ProjectionCalibration(
        coeff_x=(0.0, 100.0, 0.0, 0.0),
        coeff_y=(0.0, 0.0, 100.0, 0.0),
        rms_error_px=2.5,
        max_error_px=3.6,
        sample_count=2,
        image_size=(160, 120),
    )
    path = tmp_path / "error-report.json"

    calibration_script.save_projection_error_report(
        path,
        calibration=calibration,
        samples=samples,
        hand_label="RealSense",
        thermal_label="Lepton",
        gate_px=3.0,
        min_hand_depth_m=0.2,
        max_hand_depth_m=0.9,
    )

    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["decision"] == "ESCALATE"
    assert report["accepted"] is False
    assert report["sample_count"] == 2
    assert report["gate_px"] == 3.0
    assert report["working_depth_m"] == {"min": 0.2, "max": 0.9}
    assert report["method"] == "manual_thermal_point_depth_affine"
    assert len(report["samples"]) == 2
    assert report["samples"][0]["predicted_ir_xy"] == pytest.approx([10.0, 20.0])
    assert report["samples"][0]["error_px"] == pytest.approx(0.0)


def test_build_hand_source_selects_realsense_with_optional_serial(monkeypatch):
    created = []

    class FakeRealSenseHandSource:
        def __init__(self, *, serial):
            created.append(serial)

    monkeypatch.setattr(calibration_script, "RealSenseHandSource", FakeRealSenseHandSource)
    factory = getattr(calibration_script, "build_hand_source", None)
    assert callable(factory), "hand source selection helper is missing"

    source = factory(
        SimpleNamespace(hand_camera="realsense", realsense_serial="233522078685")
    )

    assert isinstance(source, FakeRealSenseHandSource)
    assert created == ["233522078685"]


def test_calibration_hand_selection_accepts_either_single_hand():
    select = getattr(calibration_script, "select_calibration_hand", None)
    assert callable(select)
    right = object()
    left = object()

    assert select(right, left) is right
    assert select(None, left) is left
    assert select(None, None) is None


def test_build_thermal_source_selects_lepton_udp(monkeypatch):
    created = []

    class FakeLepton:
        def __init__(self, *, port):
            created.append(port)

    monkeypatch.setattr(calibration_script, "LeptonUDPSource", FakeLepton, raising=False)
    factory = getattr(calibration_script, "build_thermal_source", None)
    assert callable(factory), "thermal source selection helper is missing"

    source = factory(SimpleNamespace(lepton_udp=9000, thermal="/dev/unused"))

    assert isinstance(source, FakeLepton)
    assert created == [9000]


def test_calibration_pair_capture_reads_slow_thermal_before_fast_hand_source():
    events = []
    thermal_sample = object()
    hand_sample = object()
    thermal = SimpleNamespace(read=lambda: events.append("thermal") or thermal_sample)
    hand = SimpleNamespace(latest_sample=lambda: events.append("hand") or hand_sample)
    capture = getattr(calibration_script, "read_calibration_pair", None)
    assert callable(capture)

    actual_hand, actual_thermal = capture(hand, thermal)

    assert events == ["thermal", "hand"]
    assert actual_hand is hand_sample
    assert actual_thermal is thermal_sample


@pytest.mark.parametrize("failure_stage", ["start_hand", "named_window", "resize_window"])
def test_main_startup_failure_attempts_all_relevant_cleanup(
    monkeypatch,
    tmp_path,
    failure_stage,
):
    events = []
    args = SimpleNamespace(
        thermal="/dev/fake-thermal",
        lepton_udp=None,
        hand_camera="realsense",
        realsense_serial=None,
        out_dir=tmp_path / "calibration",
        min_samples=12,
        max_rms_px=3.0,
        max_error_px=3.0,
        max_oak_age_ms=200.0,
        max_thermal_age_ms=200.0,
        max_pair_skew_ms=150.0,
    )

    class FakeSource:
        def start(self):
            events.append("hand-start")
            if failure_stage == "start_hand":
                raise RuntimeError("start_hand failed")

        def stop(self):
            events.append("hand-stop")

    class FakeThermal:
        def close(self):
            events.append("thermal-close")

    def named_window(*_args):
        events.append("window-create")
        if failure_stage == "named_window":
            raise RuntimeError("namedWindow failed")

    def resize_window(*_args):
        events.append("window-resize")
        if failure_stage == "resize_window":
            raise RuntimeError("resizeWindow failed")

    monkeypatch.setattr(
        calibration_script,
        "build_arg_parser",
        lambda: SimpleNamespace(parse_args=lambda: args),
    )
    monkeypatch.setattr(calibration_script, "build_hand_source", lambda _args: FakeSource())
    monkeypatch.setattr(
        calibration_script,
        "OpenCVCameraSource",
        lambda _path: FakeThermal(),
    )
    monkeypatch.setattr(calibration_script.cv2, "namedWindow", named_window)
    monkeypatch.setattr(calibration_script.cv2, "resizeWindow", resize_window)
    monkeypatch.setattr(
        calibration_script.cv2,
        "destroyAllWindows",
        lambda: events.append("windows-close"),
    )

    with pytest.raises(RuntimeError, match="failed"):
        calibration_script.main()

    assert "hand-stop" in events
    assert "thermal-close" in events
    if failure_stage != "start_hand":
        assert "windows-close" in events
