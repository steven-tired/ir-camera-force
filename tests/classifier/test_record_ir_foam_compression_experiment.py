from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from ir_force.classifier.ir_capture import FrameSample, OAKFrameSample

REGION_ARGS = [
    "--thermal-foam-bbox", "68,40,28,30",
    "--thermal-foam-roi", "75,48,14,18",
    "--thermal-left-contact-roi", "68,48,6,18",
    "--thermal-right-contact-roi", "90,48,6,18",
    "--thermal-background-roi", "5,5,15,15",
    "--thermal-room-reference-roi", "15,15,12,12",
    "--thermal-warm-reference-roi", "130,15,12,12",
    "--oak-left-marker-roi", "180,90,150,140",
    "--oak-right-marker-roi", "360,100,80,100",
]

VISIBLE_MARKER_ARGS = [
    "--flir-visible-left-marker-roi", "600,260,180,220",
    "--flir-visible-right-marker-roi", "940,240,180,220",
]


def test_parser_requires_frozen_regions_and_uses_geometry_targets_not_force_labels(tmp_path):
    module = importlib.import_module("record_ir_foam_compression_experiment")
    args = module._parse_args(
        [
            "--session-id", "S01",
            "--participant-id", "ZK",
            "--object-id", "foam",
            "--rep", "1",
            "--recording-index", "1",
            "--root", str(tmp_path),
            *REGION_ARGS,
        ]
    )

    assert args.thermal == "/dev/video21"
    assert args.oak_fps == 10.0
    assert args.target_tolerance_pct == 3.0
    assert args.gate_stable_s == 1.0
    assert args.d0_settle_s == 2.0
    assert args.preflight_settle_s == 2.0
    assert args.marker_max_gray == 110
    assert args.min_reference_span == 5.0
    assert args.recording_index == 1
    assert not hasattr(args, "force_newton")
    assert args.palette.endswith("tools/flirone-v4l2/palettes/Iron2.raw")


def test_visible_marker_tracking_requires_rgb_recording_and_is_written_to_metadata(tmp_path):
    module = importlib.import_module("record_ir_foam_compression_experiment")
    args = module._parse_args(
        [
            "--session-id", "S01",
            "--participant-id", "ZK",
            "--object-id", "foam",
            "--rep", "1",
            "--recording-index", "1",
            "--root", str(tmp_path),
            "--thermal-roi-tracking", "flir-visible-markers",
            *REGION_ARGS,
            *VISIBLE_MARKER_ARGS,
        ]
    )

    paths, _plan, _regions = module._prepare_trial(args)
    metadata = json.loads(paths.metadata_path.read_text())

    assert args.record_flir_visible is True
    assert metadata["thermal_roi_tracking"]["mode"] == "flir-visible-markers"
    assert metadata["thermal_roi_tracking"]["alignment_assumption"] == "frame_normalized_visible_to_thermal"
    assert metadata["thermal_roi_tracking"]["flir_visible_marker_regions"] == {
        "left": [600, 260, 180, 220],
        "right": [940, 240, 180, 220],
    }


def test_auto_rep_selects_the_first_unused_trial_number(tmp_path):
    module = importlib.import_module("record_ir_foam_compression_experiment")
    args = module._parse_args(
        [
            "--session-id", "S01",
            "--participant-id", "ZK",
            "--object-id", "foam",
            "--rep", "1",
            "--recording-index", "1",
            "--root", str(tmp_path),
            *REGION_ARGS,
        ]
    )
    existing = tmp_path / "trials" / "foam-compression_s01_foam_zk_rep01"
    existing.mkdir(parents=True)

    assert module._next_available_rep(args) == 2


def test_parser_accepts_auto_rep(tmp_path):
    module = importlib.import_module("record_ir_foam_compression_experiment")

    args = module._parse_args(
        [
            "--session-id", "S01",
            "--participant-id", "ZK",
            "--rep", "auto",
            "--recording-index", "1",
            "--root", str(tmp_path),
            *REGION_ARGS,
        ]
    )

    assert args.rep is None


def test_reference_preflight_rejects_a_two_bin_span():
    module = importlib.import_module("record_ir_foam_compression_experiment")

    assert not module._reference_span_is_adequate(-2.0, min_span=5.0)
    assert module._reference_span_is_adequate(-5.0, min_span=5.0)
    assert module._reference_span_is_adequate(5.0, min_span=5.0)


def test_d0_stability_rejects_a_release_calibration_contaminated_by_squeeze():
    module = importlib.import_module("record_ir_foam_compression_experiment")

    assert 0.0 < module._d0_relative_span([152.8, 153.0, 153.1, 152.9, 153.0]) < 0.005
    assert not module._d0_is_stable(
        [110.9, 122.7, 128.4, 152.1, 153.0, 155.3, 158.9, 161.7, 180.5],
        max_relative_span=0.05,
    )


def test_d0_post_settle_distances_exclude_camera_startup_samples():
    module = importlib.import_module("record_ir_foam_compression_experiment")

    distances = module._d0_post_settle_distances(
        [(0.0, 114.1), (1.9, 114.2), (2.0, 147.3), (2.1, 147.4), (9.9, 147.5)],
        settle_s=2.0,
    )

    assert distances == [147.3, 147.4, 147.5]
    assert module._d0_is_stable(distances, max_relative_span=0.05)


def test_projects_median_visible_markers_into_the_thermal_baseline():
    module = importlib.import_module("record_ir_foam_compression_experiment")
    shared = importlib.import_module("ir_force.classifier.ir_foam_compression")
    markers = [
        shared.MarkerObservation((355.0, 270.0), (1085.0, 270.0), 100.0, 100.0),
        shared.MarkerObservation((360.0, 270.0), (1080.0, 270.0), 100.0, 100.0),
        shared.MarkerObservation((365.0, 270.0), (1075.0, 270.0), 100.0, 100.0),
    ]

    baseline = module._thermal_marker_baseline_from_visible(
        markers,
        visible_shape=(1080, 1440),
        thermal_shape=(128, 160),
    )

    assert baseline.left_xy == pytest.approx((40.0, 32.0))
    assert baseline.right_xy == pytest.approx((120.0, 32.0))


def test_preflight_marker_stats_exclude_oak_startup_before_evaluation():
    module = importlib.import_module("record_ir_foam_compression_experiment")

    frame_count, marker_count, longest_missing_s = module._preflight_marker_stats(
        [(0.1, False), (0.8, False), (1.6, False), (1.9, False), (2.1, True), (2.4, True), (2.8, True)],
        settle_s=2.0,
        end_elapsed_s=3.0,
    )

    assert (frame_count, marker_count, longest_missing_s) == (3, 3, 0.0)


def test_release_uses_a_wider_tolerance_without_changing_compression_targets():
    module = importlib.import_module("record_ir_foam_compression_experiment")
    shared = importlib.import_module("ir_force.classifier.ir_foam_compression")
    args = SimpleNamespace(target_tolerance_pct=2.0, release_tolerance_pct=5.0)

    assert module._step_target_tolerance(shared.FoamCompressionStep("drift", "baseline", "R", 30.0, 0, 0), args) == 5.0
    assert module._step_target_tolerance(shared.FoamCompressionStep("steady", "target", "C20", 6.0, 1, 1), args) == 2.0


def test_prepare_trial_writes_frozen_rois_and_prespecified_primary_feature(tmp_path):
    module = importlib.import_module("record_ir_foam_compression_experiment")
    args = module._parse_args(
        [
            "--session-id", "S01",
            "--participant-id", "ZK",
            "--object-id", "foam block",
            "--rep", "2",
            "--recording-index", "2",
            "--root", str(tmp_path),
            *REGION_ARGS,
        ]
    )

    paths, plan, regions = module._prepare_trial(args)
    metadata = json.loads(paths.metadata_path.read_text())

    assert paths.trial_id == "foam-compression_s01_foam-block_zk_rep02"
    assert metadata["objective_force_measurement"] is False
    assert metadata["compression_reference"] == "oak_black_marker_distance"
    assert metadata["primary_ir_feature"] == "foam_center_norm"
    assert metadata["thermal_regions"] == regions.metadata()
    assert metadata["recording_plan"] == [module._step_metadata(step) for step in plan]
    assert metadata["recording_index"] == 2
    assert not (paths.root / "force.csv").exists()


def test_frame_rows_record_marker_compression_and_preregistered_thermal_feature():
    module = importlib.import_module("record_ir_foam_compression_experiment")
    shared = importlib.import_module("ir_force.classifier.ir_foam_compression")
    regions = shared.FrozenThermalRegions(
        foam_bbox=shared.PixelROI(68, 40, 28, 30),
        foam_center=shared.PixelROI(75, 48, 14, 18),
        left_contact=shared.PixelROI(68, 48, 6, 18),
        right_contact=shared.PixelROI(90, 48, 6, 18),
        background=shared.PixelROI(5, 5, 15, 15),
        room_reference=shared.PixelROI(15, 15, 12, 12),
        warm_reference=shared.PixelROI(130, 15, 12, 12),
    )
    scalar = np.zeros((128, 160), dtype=np.float32)
    scalar[regions.room_reference.slices()] = 10.0
    scalar[regions.warm_reference.slices()] = 30.0
    scalar[regions.foam_center.slices()] = 15.0
    marker = shared.MarkerObservation((40.0, 30.0), (100.0, 30.0), 50.0, 50.0)
    step = shared.FoamCompressionStep("steady_state", "target", "C20", 6.0, 1, 1)

    telemetry, features = module._frame_rows(
        frame_index=7,
        protocol_elapsed_s=12.5,
        thermal_timestamp=100.0,
        oak_timestamp=100.001,
        step=step,
        capture_phase="stable_hold",
        step_elapsed_s=2.0,
        action_attempt=1,
        marker=marker,
        d0_px=75.0,
        gate_stable_s=1.2,
        regions=regions,
        scalar=scalar,
        thermal_sha1="abc",
        frozen_frame=False,
    )

    assert telemetry["compression_pct"] == 20.0
    assert telemetry["marker_distance_px"] == 60.0
    assert telemetry["gate_in_range"] is True
    assert "force_newton" not in telemetry
    assert features["foam_center_norm"] == 0.25
    assert features["thermal_frame_sha1"] == "abc"


def test_frame_rows_use_registered_foam_regions_and_save_registration_diagnostics():
    module = importlib.import_module("record_ir_foam_compression_experiment")
    shared = importlib.import_module("ir_force.classifier.ir_foam_compression")
    regions = shared.FrozenThermalRegions(
        foam_bbox=shared.PixelROI(68, 40, 28, 30),
        foam_center=shared.PixelROI(75, 48, 14, 18),
        left_contact=shared.PixelROI(68, 48, 6, 18),
        right_contact=shared.PixelROI(90, 48, 6, 18),
        background=shared.PixelROI(5, 5, 15, 15),
        room_reference=shared.PixelROI(15, 15, 12, 12),
        warm_reference=shared.PixelROI(130, 15, 12, 12),
    )
    active_regions = shared.FrozenThermalRegions(
        foam_bbox=shared.PixelROI(73, 40, 28, 30),
        foam_center=shared.PixelROI(80, 48, 14, 18),
        left_contact=shared.PixelROI(73, 48, 6, 18),
        right_contact=shared.PixelROI(95, 48, 6, 18),
        background=regions.background,
        room_reference=regions.room_reference,
        warm_reference=regions.warm_reference,
    )
    scalar = np.zeros((128, 160), dtype=np.float32)
    scalar[regions.room_reference.slices()] = 10.0
    scalar[regions.warm_reference.slices()] = 30.0
    scalar[regions.foam_center.slices()] = 15.0
    scalar[active_regions.foam_center.slices()] = 22.0
    step = shared.FoamCompressionStep("steady_state", "target", "C20", 6.0, 1, 1)
    visible_marker = shared.MarkerObservation((600.0, 300.0), (1000.0, 300.0), 50.0, 50.0)

    telemetry, features = module._frame_rows(
        frame_index=7,
        protocol_elapsed_s=12.5,
        thermal_timestamp=100.0,
        oak_timestamp=100.001,
        visible_timestamp=100.002,
        step=step,
        capture_phase="stable_hold",
        step_elapsed_s=2.0,
        action_attempt=1,
        marker=None,
        d0_px=None,
        gate_stable_s=1.2,
        regions=regions,
        active_regions=active_regions,
        scalar=scalar,
        thermal_sha1="abc",
        frozen_frame=False,
        registration={
            "mode": "flir-visible-markers",
            "valid": True,
            "scale": 1.0,
            "rotation_deg": 0.0,
            "translation_x": 5.0,
            "translation_y": 0.0,
            "visible_marker": visible_marker,
        },
    )

    assert telemetry["t_flir_visible"] == 100.002
    assert telemetry["flir_visible_marker_detected"] is True
    assert telemetry["thermal_roi_registration_valid"] is True
    assert telemetry["thermal_roi_translation_x"] == 5.0
    assert telemetry["thermal_foam_center_roi_x"] == 80
    assert features["foam_center_norm"] == pytest.approx(0.6)


def test_capture_sample_writes_synchronized_raw_streams_and_csv_rows(tmp_path):
    module = importlib.import_module("record_ir_foam_compression_experiment")
    shared = importlib.import_module("ir_force.classifier.ir_foam_compression")
    args = module._parse_args(
        [
            "--session-id", "S01",
            "--participant-id", "ZK",
            "--object-id", "foam",
            "--rep", "1",
            "--recording-index", "1",
            "--root", str(tmp_path),
            *REGION_ARGS,
        ]
    )
    paths, plan, regions = module._prepare_trial(args)
    thermal_frame = np.zeros((128, 160, 3), dtype=np.uint8)
    thermal_frame[regions.room_reference.slices()] = 10
    thermal_frame[regions.warm_reference.slices()] = 30
    thermal_frame[regions.foam_center.slices()] = 15
    oak_frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    cv2 = importlib.import_module("cv2")
    cv2.circle(oak_frame, (220, 140), 6, (0, 0, 0), -1)
    cv2.circle(oak_frame, (380, 140), 6, (0, 0, 0), -1)

    class Thermal:
        def read(self):
            return FrameSample(t=10.0, frame=thermal_frame)

    class Oak:
        def read(self):
            return OAKFrameSample(t=10.001, frame=oak_frame, depth=np.full((480, 640), 500, dtype=np.uint16))

    runtime = module._CaptureRuntime(
        paths=paths,
        regions=regions,
        thermal=Thermal(),
        oak=Oak(),
        visible=None,
        left_marker_roi=args.oak_left_marker_roi,
        right_marker_roi=args.oak_right_marker_roi,
        palette=None,
        invert_palette=False,
        marker_max_gray=60,
        marker_min_area_px=10,
        protocol_start=0.0,
    )
    sample = module._capture_sample(
        runtime,
        step=plan[1],
        capture_phase="approach",
        step_elapsed_s=0.2,
        action_attempt=1,
        d0_px=200.0,
        gate_stable_s=0.0,
        now=0.5,
    )

    assert sample.compression_pct == 20.0
    assert (paths.thermal_dir / "frame_000000.png").exists()
    assert (paths.root / "oak_rgb" / "frame_000000.png").exists()
    assert (paths.root / "oak_depth" / "frame_000000.png").exists()
    assert paths.telemetry_csv.read_text().count("\n") == 2
    assert (paths.root / "frame_features.csv").read_text().count("\n") == 2


def test_capture_sample_tracks_thermal_regions_from_flir_visible_markers(tmp_path):
    module = importlib.import_module("record_ir_foam_compression_experiment")
    shared = importlib.import_module("ir_force.classifier.ir_foam_compression")
    args = module._parse_args(
        [
            "--session-id", "S01",
            "--participant-id", "ZK",
            "--object-id", "foam",
            "--rep", "1",
            "--recording-index", "1",
            "--root", str(tmp_path),
            "--thermal-roi-tracking", "flir-visible-markers",
            *REGION_ARGS,
            "--flir-visible-left-marker-roi", "300,200,220,150",
            "--flir-visible-right-marker-roi", "920,200,220,150",
        ]
    )
    paths, plan, regions = module._prepare_trial(args)
    thermal_frame = np.zeros((128, 160, 3), dtype=np.uint8)
    thermal_frame[regions.room_reference.slices()] = 10
    thermal_frame[regions.warm_reference.slices()] = 30
    thermal_frame[regions.foam_center.slices()] = 15
    oak_frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    visible_frame = np.full((1080, 1440, 3), 255, dtype=np.uint8)
    cv2 = importlib.import_module("cv2")
    cv2.circle(oak_frame, (220, 140), 6, (0, 0, 0), -1)
    cv2.circle(oak_frame, (380, 140), 6, (0, 0, 0), -1)
    cv2.circle(visible_frame, (405, 270), 8, (0, 0, 0), -1)
    cv2.circle(visible_frame, (1035, 270), 8, (0, 0, 0), -1)

    class Thermal:
        def read(self):
            return FrameSample(t=10.0, frame=thermal_frame)

    class Oak:
        def read(self):
            return OAKFrameSample(t=10.001, frame=oak_frame, depth=np.full((480, 640), 500, dtype=np.uint16))

    class Visible:
        def read(self):
            return FrameSample(t=10.002, frame=visible_frame)

    runtime = module._CaptureRuntime(
        paths=paths,
        regions=regions,
        thermal=Thermal(),
        oak=Oak(),
        visible=Visible(),
        left_marker_roi=args.oak_left_marker_roi,
        right_marker_roi=args.oak_right_marker_roi,
        palette=None,
        invert_palette=False,
        marker_max_gray=60,
        marker_min_area_px=10,
        protocol_start=0.0,
        roi_tracking_mode="flir-visible-markers",
        visible_left_marker_roi=args.flir_visible_left_marker_roi,
        visible_right_marker_roi=args.flir_visible_right_marker_roi,
        visible_marker_max_gray=60,
        visible_marker_min_area_px=10,
        thermal_marker_baseline=shared.MarkerObservation((40.0, 32.0), (120.0, 32.0), 1.0, 1.0),
    )

    sample = module._capture_sample(
        runtime,
        step=plan[1],
        capture_phase="approach",
        step_elapsed_s=0.2,
        action_attempt=1,
        d0_px=200.0,
        gate_stable_s=0.0,
        now=0.5,
    )

    assert sample.visible_marker is not None
    assert sample.thermal_roi_registration_valid is True
    assert (paths.flir_visible_dir / "frame_000000.png").exists()
    assert runtime.last_feature_regions is not None
    assert runtime.last_feature_regions.foam_center.x == 76
