from __future__ import annotations

import importlib
import csv
import json
import time

import cv2
import numpy as np

from ir_force.classifier.ir_capture import FrameSample, OAKFrameSample
from ir_force.classifier.ir_features import ThermalROI, compute_baseline


class FakeFrameSource:
    def __init__(self, value):
        self.value = value
        self.count = 0

    def read(self):
        self.count += 1
        frame = np.full((4, 6), self.value + self.count, dtype=np.uint8)
        return FrameSample(t=float(self.count), frame=frame)


class FakeOAKFrameSource:
    def __init__(self, value):
        self.value = value
        self.count = 0

    def read(self):
        self.count += 1
        frame = np.full((4, 6, 3), self.value + self.count, dtype=np.uint8)
        depth = np.full((4, 6), 700 + self.count, dtype=np.uint16)
        return OAKFrameSample(t=float(self.count), frame=frame, depth=depth)


def test_hard_classifier_parser_defaults_to_randomized_subjective_squeeze_protocol():
    module = importlib.import_module("record_ir_hard_classifier_experiment")

    args = module._parse_args(
        [
            "--session-id",
            "S01",
            "--participant-id",
            "ZK",
            "--object-id",
            "foam",
            "--rep",
            "1",
        ]
    )

    assert args.thermal == "/dev/video21"
    assert not hasattr(args, "bird")
    assert args.oak_fps == 10.0
    assert args.flir_visible == "/dev/video20"
    assert args.record_flir_visible is False
    assert args.root.endswith("ir_hard_classifier")
    assert args.block_type == "fixed_posture"
    assert args.squeeze_percents == (0.0, 25.0, 50.0, 75.0)
    assert args.sequences == 8
    assert args.hold_s == 3.0
    assert args.release_s == 1.0
    assert args.pre_baseline_s == 3.0
    assert args.thermal_roi == "25,35,115,80"
    assert args.squeeze_percents == (0.0, 25.0, 50.0, 75.0)
    assert not hasattr(args, "fmax_n")
    assert not hasattr(args, "hard_fraction")


def test_prepare_trial_writes_subjective_squeeze_metadata_and_events_without_force_labels(tmp_path):
    module = importlib.import_module("record_ir_hard_classifier_experiment")

    args = module._parse_args(
        [
            "--session-id",
            "S01",
            "--participant-id",
            "ZK",
            "--object-id",
            "foam block",
            "--rep",
            "2",
            "--sequences",
            "2",
            "--root",
            str(tmp_path),
            "--seed",
            "4",
            "--posture-condition",
            "neutral",
            "--notes",
            "same foam, no scale",
        ]
    )

    spec, paths, steps = module._prepare_trial(args)

    metadata = json.loads(paths.metadata_path.read_text())
    assert paths.trial_id == "oak-squeeze_s01_fixed-posture_foam-block_zk_rep02"
    assert spec.block_type == "fixed_posture"
    assert metadata["experiment_kind"] == "ir_oak_squeeze_proxy"
    assert metadata["recording_mode"] == "randomized_subjective_squeeze_targets"
    assert metadata["primary_task"] == "oak_squeeze_proxy_vs_ir"
    assert metadata["primary_comparison"] == "oak_proxy_plus_ir_vs_oak_proxy_only"
    assert metadata["label_source"] == "oak_visual_proxy_pending"
    assert metadata["objective_force_measurement"] is False
    assert metadata["target_squeeze_percents"] == [0.0, 25.0, 50.0, 75.0]
    assert metadata["oak_rgb_size"] == [640, 480]
    assert metadata["oak_depth_unit"] == "millimeter"
    assert metadata["oak_fps"] == 10.0
    assert metadata["thermal_roi"] == "25,35,115,80"
    assert metadata["posture_condition"] == "neutral"
    assert metadata["notes"] == "same foam, no scale"
    assert not (paths.root / "force.csv").exists()
    assert "target_force_percents" not in metadata
    assert "ir_persistence_after_release" in metadata["analysis_tasks"]
    assert "false_hard_after_release" not in metadata["analysis_tasks"]
    assert len(steps) == 8

    event_rows = module._event_rows(steps)
    assert event_rows[0]["event_type"] == "target_start"
    assert event_rows[0]["block_type"] == "fixed_posture"
    assert event_rows[0]["target_squeeze_percent"] in {0.0, 25.0, 50.0, 75.0}
    assert event_rows[0]["posture_condition"] == "neutral"
    assert "target_force_newton" not in event_rows[0]


def test_capture_classifier_window_writes_frame_features_without_force_label(tmp_path):
    module = importlib.import_module("record_ir_hard_classifier_experiment")
    args = module._parse_args(
        [
            "--session-id",
            "S01",
            "--participant-id",
            "ZK",
            "--object-id",
            "foam",
            "--rep",
            "1",
            "--root",
            str(tmp_path),
        ]
    )
    spec, paths, steps = module._prepare_trial(args)
    baseline = compute_baseline(
        [np.full((4, 6), 10, dtype=np.uint8), np.full((4, 6), 12, dtype=np.uint8)],
        roi=ThermalROI(x=1, y=1, width=3, height=2),
    )

    frames, _samples, previous_p98, _previous_raw = module._capture_classifier_window(
        paths=paths,
        thermal=FakeFrameSource(20),
        oak=FakeOAKFrameSource(30),
        flir_visible=None,
        spec=spec,
        step=steps[0],
        phase="target_hold",
        duration_s=0.01,
        fps=10,
        protocol_start=time.perf_counter(),
        baseline=baseline,
        reference_patches=(),
        previous_frame_p98=None,
        previous_raw_frame=None,
    )

    assert frames == 1
    assert previous_p98 is not None
    telemetry_rows = list(csv.DictReader(paths.telemetry_csv.open()))
    feature_rows = list(csv.DictReader((paths.root / "frame_features.csv").open()))
    assert telemetry_rows[0]["phase"] == "target_hold"
    assert telemetry_rows[0]["target_squeeze_percent"] == str(steps[0].target_squeeze_percent)
    assert "target_force_newton" not in telemetry_rows[0]
    assert feature_rows[0]["phase"] == "target_hold"
    assert feature_rows[0]["delta_mean"] != ""
    assert "target_force_newton" not in feature_rows[0]
    assert telemetry_rows[0]["t_oak"] == "1.0"
    assert (paths.root / "oak_rgb" / "frame_000000.png").exists()
    depth = cv2.imread(str(paths.root / "oak_depth" / "frame_000000.png"), cv2.IMREAD_UNCHANGED)
    assert depth.dtype == np.uint16
    assert int(depth[0, 0]) == 701
