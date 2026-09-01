from __future__ import annotations

import csv
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from ir_force.classifier.ir_dataset import TrialPaths, append_telemetry_row
from lerobot_teleoperator_so101_webcam.gripper_hardware import TelemetrySnapshot


TELEMETRY_FIELDS = (
    "frame",
    "t_capture",
    "t_thermal",
    "t_bird",
    "t_flir_visible",
    "t",
    "gripper_pos",
    "goal_gripper_pos",
    "present_current",
    "present_load",
    "present_temperature",
)

LABELED_CAMERA_FIELDS = (
    "frame",
    "t_capture",
    "t_thermal",
    "t_bird",
    "t_flir_visible",
    "surface",
    "contact",
    "phase",
    "sweep_progress",
)


@dataclass(frozen=True)
class FrameSample:
    t: float
    frame: np.ndarray


@dataclass(frozen=True)
class OAKFrameSample(FrameSample):
    depth: np.ndarray


class FrameSource(Protocol):
    def read(self) -> FrameSample:
        ...


class TelemetrySource(Protocol):
    def read(self) -> TelemetrySnapshot:
        ...


@dataclass(frozen=True)
class CaptureSources:
    thermal: FrameSource
    bird: FrameSource
    flir_visible: FrameSource | None
    telemetry: TelemetrySource


class OpenCVCameraSource:
    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera {path}")

    def read(self) -> FrameSample:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"could not read camera {self.path}")
        return FrameSample(t=time.perf_counter(), frame=frame)

    def close(self) -> None:
        self.cap.release()


class OAKCameraSource:
    """Adapter for the shared OAK-D RGB plus aligned stereo-depth camera."""

    def __init__(
        self,
        *,
        rgb_size: tuple[int, int] = (640, 480),
        fps: float = 10.0,
        camera: Any | None = None,
    ):
        if fps <= 0:
            raise ValueError("fps must be positive")
        if camera is None:
            from webcam_input.oak_camera import OAKCamera

            camera = OAKCamera(rgb_size=rgb_size, fps=int(fps))
        self.camera = camera
        self.camera.start()

    def read(self) -> OAKFrameSample:
        frame, depth = self.camera.read()
        if frame is None or depth is None:
            raise RuntimeError("could not read OAK RGB/depth frame")
        return OAKFrameSample(t=time.perf_counter(), frame=frame, depth=depth)

    def close(self) -> None:
        self.camera.stop()


def _write_frame(path: Path, frame: np.ndarray) -> None:
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"could not write frame {path}")


def capture_setup_snapshot(
    paths: TrialPaths,
    *,
    thermal: FrameSource,
    bird: FrameSource,
    flir_visible: FrameSource | None,
) -> None:
    _write_frame(paths.preflight_dir / "thermal.png", thermal.read().frame)
    _write_frame(paths.preflight_dir / "bird.png", bird.read().frame)
    if flir_visible is not None:
        _write_frame(paths.preflight_dir / "flir_visible.png", flir_visible.read().frame)


def _existing_telemetry_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _next_frame_index(paths: TrialPaths) -> int:
    counts = [
        len(list(paths.thermal_dir.glob("frame_*.png"))),
        len(list(paths.bird_dir.glob("frame_*.png"))),
        len(list(paths.flir_visible_dir.glob("frame_*.png"))),
        _existing_telemetry_rows(paths.telemetry_csv),
    ]
    return max(counts)


def _telemetry_row(
    *,
    frame_index: int,
    t_capture: float,
    thermal: FrameSample,
    bird: FrameSample,
    visible: FrameSample | None,
    telemetry: TelemetrySnapshot,
) -> dict[str, object]:
    return {
        "frame": frame_index,
        "t_capture": round(t_capture, 6),
        "t_thermal": round(thermal.t, 6),
        "t_bird": round(bird.t, 6),
        "t_flir_visible": round(visible.t, 6) if visible is not None else "",
        "t": round(telemetry.t, 6),
        "gripper_pos": float(telemetry.gripper_pos),
        "goal_gripper_pos": float(telemetry.goal_gripper_pos),
        "present_current": telemetry.present_current,
        "present_load": telemetry.present_load,
        "present_temperature": telemetry.present_temperature,
    }


def _labeled_camera_row(
    *,
    frame_index: int,
    t_capture: float,
    thermal: FrameSample,
    bird: FrameSample,
    visible: FrameSample | None,
    labels: dict[str, object],
    sweep_progress: float | None,
) -> dict[str, object]:
    return {
        "frame": frame_index,
        "t_capture": round(t_capture, 6),
        "t_thermal": round(thermal.t, 6),
        "t_bird": round(bird.t, 6),
        "t_flir_visible": round(visible.t, 6) if visible is not None else "",
        "surface": labels["surface"],
        "contact": labels["contact"],
        "phase": labels["phase"],
        "sweep_progress": "" if sweep_progress is None else round(sweep_progress, 6),
    }


def record_labeled_camera_window(
    paths: TrialPaths,
    *,
    thermal: FrameSource,
    bird: FrameSource,
    flir_visible: FrameSource | None,
    duration_s: float,
    fps: float,
    labels: dict[str, object],
    progress_duration_s: float | None = None,
    progress_start: float = 0.0,
    progress_end: float = 1.0,
) -> int:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if duration_s < 0:
        raise ValueError("duration_s must be non-negative")
    if progress_duration_s is not None and progress_duration_s <= 0:
        raise ValueError("progress_duration_s must be positive when set")

    period = 1.0 / fps
    window_start = time.perf_counter()
    next_capture = window_start
    deadline = window_start + duration_s
    frame_index = _next_frame_index(paths)
    frames_written = 0

    while next_capture <= deadline + 1e-9:
        now = time.perf_counter()
        if now < next_capture:
            time.sleep(next_capture - now)

        elapsed_s = time.perf_counter() - window_start
        thermal_sample = thermal.read()
        bird_sample = bird.read()
        visible_sample = flir_visible.read() if flir_visible is not None else None
        sweep_progress = None
        if progress_duration_s is not None:
            fraction = min(max(elapsed_s / progress_duration_s, 0.0), 1.0)
            sweep_progress = progress_start + fraction * (progress_end - progress_start)

        _write_frame(paths.thermal_dir / f"frame_{frame_index:06d}.png", thermal_sample.frame)
        _write_frame(paths.bird_dir / f"frame_{frame_index:06d}.png", bird_sample.frame)
        if visible_sample is not None:
            _write_frame(paths.flir_visible_dir / f"frame_{frame_index:06d}.png", visible_sample.frame)

        row = _labeled_camera_row(
            frame_index=frame_index,
            t_capture=elapsed_s,
            thermal=thermal_sample,
            bird=bird_sample,
            visible=visible_sample,
            labels=labels,
            sweep_progress=sweep_progress,
        )
        assert tuple(row.keys()) == LABELED_CAMERA_FIELDS
        append_telemetry_row(paths.telemetry_csv, row)

        frames_written += 1
        frame_index += 1
        next_capture += period

    return frames_written


def record_capture_window(
    paths: TrialPaths,
    sources: CaptureSources,
    duration_s: float,
    fps: float,
    before_sample: Callable[[float], None] | None = None,
) -> int:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if duration_s < 0:
        raise ValueError("duration_s must be non-negative")

    period = 1.0 / fps
    window_start = time.perf_counter()
    next_capture = window_start
    deadline = window_start + duration_s
    frame_index = _next_frame_index(paths)
    frames_written = 0

    while next_capture <= deadline + 1e-9:
        now = time.perf_counter()
        if now < next_capture:
            time.sleep(next_capture - now)

        frame_started = time.perf_counter()
        elapsed_s = frame_started - window_start
        if before_sample is not None:
            before_sample(elapsed_s)
        thermal = sources.thermal.read()
        bird = sources.bird.read()
        visible = sources.flir_visible.read() if sources.flir_visible is not None else None
        telemetry = sources.telemetry.read()

        _write_frame(paths.thermal_dir / f"frame_{frame_index:06d}.png", thermal.frame)
        _write_frame(paths.bird_dir / f"frame_{frame_index:06d}.png", bird.frame)
        if visible is not None:
            _write_frame(paths.flir_visible_dir / f"frame_{frame_index:06d}.png", visible.frame)

        row = _telemetry_row(
            frame_index=frame_index,
            t_capture=elapsed_s,
            thermal=thermal,
            bird=bird,
            visible=visible,
            telemetry=telemetry,
        )
        assert tuple(row.keys()) == TELEMETRY_FIELDS
        append_telemetry_row(paths.telemetry_csv, row)

        frames_written += 1
        frame_index += 1
        next_capture += period

    return frames_written
