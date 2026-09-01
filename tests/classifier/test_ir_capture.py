import csv

import numpy as np

from ir_force.classifier.ir_capture import (
    CaptureSources,
    FrameSample,
    OAKCameraSource,
    capture_setup_snapshot,
    record_capture_window,
    record_labeled_camera_window,
)
from ir_force.classifier.ir_dataset import HandPressureTrialSpec, create_hand_pressure_trial_paths
from ir_force.classifier.ir_dataset import TrialSpec, create_trial_paths
from lerobot_teleoperator_so101_webcam.gripper_hardware import TelemetrySnapshot


class FakeFrameSource:
    def __init__(self, value):
        self.value = value
        self.count = 0

    def read(self):
        self.count += 1
        frame = np.full((4, 5), self.value, dtype=np.uint8)
        return FrameSample(t=float(self.count), frame=frame)


class FakeOAKCamera:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def read(self):
        rgb = np.full((3, 4, 3), 17, dtype=np.uint8)
        depth = np.full((3, 4), 725, dtype=np.uint16)
        return rgb, depth

    def stop(self):
        self.stopped = True


def test_oak_camera_source_returns_rgb_and_aligned_depth_and_stops_camera():
    camera = FakeOAKCamera()

    source = OAKCameraSource(camera=camera)
    sample = source.read()
    source.close()

    assert camera.started is True
    assert camera.stopped is True
    assert sample.frame.shape == (3, 4, 3)
    assert sample.frame.dtype == np.uint8
    assert sample.depth.shape == (3, 4)
    assert sample.depth.dtype == np.uint16


class FakeTelemetrySource:
    def __init__(self, goal_gripper_pos=80.0):
        self.count = 0
        self.goal_gripper_pos = goal_gripper_pos

    def read(self):
        self.count += 1
        return TelemetrySnapshot(
            t=float(self.count),
            gripper_pos=90.0 - self.count,
            goal_gripper_pos=self.goal_gripper_pos,
            present_current=10 + self.count,
            present_load=2 + self.count,
            present_temperature=30,
        )


def test_record_capture_window_writes_frames_and_telemetry(tmp_path):
    paths = create_trial_paths(tmp_path, TrialSpec("foam", "soft", "low", 1))
    sources = CaptureSources(
        thermal=FakeFrameSource(7),
        bird=FakeFrameSource(8),
        flir_visible=None,
        telemetry=FakeTelemetrySource(),
    )

    frames = record_capture_window(paths, sources, duration_s=0.21, fps=10)

    assert frames == 3
    assert sorted(path.name for path in paths.thermal_dir.glob("*.png")) == [
        "frame_000000.png",
        "frame_000001.png",
        "frame_000002.png",
    ]
    rows = list(csv.DictReader(paths.telemetry_csv.open()))
    assert rows[0]["present_current"] == "11"
    assert rows[2]["gripper_pos"] == "87.0"


def test_record_capture_window_reuses_same_csv_header_across_windows(tmp_path):
    paths = create_trial_paths(tmp_path, TrialSpec("foam", "soft", "low", 1))

    baseline_sources = CaptureSources(
        thermal=FakeFrameSource(7),
        bird=FakeFrameSource(8),
        flir_visible=None,
        telemetry=FakeTelemetrySource(goal_gripper_pos=100.0),
    )
    hold_sources = CaptureSources(
        thermal=FakeFrameSource(9),
        bird=FakeFrameSource(10),
        flir_visible=None,
        telemetry=FakeTelemetrySource(goal_gripper_pos=80.0),
    )

    baseline_frames = record_capture_window(paths, baseline_sources, duration_s=0.01, fps=10)
    hold_frames = record_capture_window(paths, hold_sources, duration_s=0.01, fps=10)

    assert baseline_frames == 1
    assert hold_frames == 1
    rows = list(csv.DictReader(paths.telemetry_csv.open()))
    assert rows[0]["goal_gripper_pos"] == "100.0"
    assert rows[1]["goal_gripper_pos"] == "80.0"
    assert rows[0].keys() == rows[1].keys()
    assert sorted(path.name for path in paths.thermal_dir.glob("*.png")) == [
        "frame_000000.png",
        "frame_000001.png",
    ]


def test_record_capture_window_runs_control_callback_before_sampling(tmp_path):
    paths = create_trial_paths(tmp_path, TrialSpec("foam", "soft", "low", 1))
    events: list[str] = []

    class EventTelemetrySource(FakeTelemetrySource):
        def read(self):
            events.append("telemetry")
            return super().read()

    sources = CaptureSources(
        thermal=FakeFrameSource(7),
        bird=FakeFrameSource(8),
        flir_visible=None,
        telemetry=EventTelemetrySource(),
    )

    def before_sample(_elapsed_s: float) -> None:
        events.append("before")

    record_capture_window(paths, sources, duration_s=0.01, fps=10, before_sample=before_sample)

    assert events == ["before", "telemetry"]


def test_record_labeled_camera_window_writes_sweep_phase_and_progress(tmp_path):
    spec = HandPressureTrialSpec("brick", "fingertip", 1)
    paths = create_hand_pressure_trial_paths(tmp_path, spec)

    baseline_frames = record_labeled_camera_window(
        paths,
        thermal=FakeFrameSource(6),
        bird=FakeFrameSource(7),
        flir_visible=None,
        duration_s=0.01,
        fps=10,
        labels={
            "surface": spec.surface,
            "contact": spec.contact,
            "phase": "baseline",
        },
    )
    sweep_frames = record_labeled_camera_window(
        paths,
        thermal=FakeFrameSource(7),
        bird=FakeFrameSource(8),
        flir_visible=None,
        duration_s=0.21,
        fps=10,
        progress_duration_s=0.21,
        labels={
            "surface": spec.surface,
            "contact": spec.contact,
            "phase": "pressure_sweep",
        },
    )

    assert baseline_frames == 1
    assert sweep_frames == 3
    assert sorted(path.name for path in paths.thermal_dir.glob("*.png")) == [
        "frame_000000.png",
        "frame_000001.png",
        "frame_000002.png",
        "frame_000003.png",
    ]
    rows = list(csv.DictReader(paths.telemetry_csv.open()))
    assert rows[0]["surface"] == "brick"
    assert rows[0]["contact"] == "fingertip"
    assert rows[0]["phase"] == "baseline"
    assert rows[0]["sweep_progress"] == ""
    sweep_progress = [float(row["sweep_progress"]) for row in rows[1:]]
    assert all(row["phase"] == "pressure_sweep" for row in rows[1:])
    assert 0.0 <= sweep_progress[0] <= 0.1
    assert sweep_progress[-1] > sweep_progress[0]
    assert sweep_progress[-1] <= 1.0
    assert "pressure_level" not in rows[0]
    assert "force_kg" not in rows[0]
    assert rows[3]["frame"] == "3"


def test_record_labeled_camera_window_can_record_release_progress(tmp_path):
    spec = HandPressureTrialSpec("foam", "whole hand", 1)
    paths = create_hand_pressure_trial_paths(tmp_path, spec)

    frames = record_labeled_camera_window(
        paths,
        thermal=FakeFrameSource(7),
        bird=FakeFrameSource(8),
        flir_visible=None,
        duration_s=0.21,
        fps=10,
        progress_duration_s=0.21,
        progress_start=1.0,
        progress_end=0.0,
        labels={
            "surface": spec.surface,
            "contact": spec.contact,
            "phase": "release",
        },
    )

    assert frames == 3
    rows = list(csv.DictReader(paths.telemetry_csv.open()))
    progress = [float(row["sweep_progress"]) for row in rows]
    assert all(row["phase"] == "release" for row in rows)
    assert 0.9 <= progress[0] <= 1.0
    assert progress[-1] < progress[0]
    assert progress[-1] >= 0.0


def test_capture_setup_snapshot_writes_thermal_bird_and_flir_visible(tmp_path, monkeypatch):
    paths = create_trial_paths(tmp_path, TrialSpec("foam", "soft", "high", 1))
    written: list[str] = []

    def fake_write_frame(path, frame):
        written.append(path.name)

    monkeypatch.setattr("ir_force.classifier.ir_capture._write_frame", fake_write_frame)

    capture_setup_snapshot(
        paths,
        thermal=FakeFrameSource(7),
        bird=FakeFrameSource(8),
        flir_visible=FakeFrameSource(9),
    )

    assert written == ["thermal.png", "bird.png", "flir_visible.png"]
