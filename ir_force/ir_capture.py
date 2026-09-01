from __future__ import annotations

import csv
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from ir_force.ir_dataset import TrialPaths, append_telemetry_row
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
class LeptonTelemetry:
    frame_counter: int
    packet_timestamp_ms: int
    ffc_desired: bool
    ffc_state: str
    ffc_in_progress: bool
    since_last_ffc_s: float
    tlinear_enabled: bool
    tlinear_resolution_k: float | None


@dataclass(frozen=True)
class FrameSample:
    """Frame plus host-monotonic read-completion time, not sensor exposure time."""

    t: float
    frame: np.ndarray
    temperature_c: np.ndarray | None = None
    lepton_telemetry: LeptonTelemetry | None = None


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


class FrameUnavailableError(RuntimeError):
    pass


LEPTON_PACKET_SIZE = 164
LEPTON_PACKETS_PER_SEGMENT = 60
LEPTON_SEGMENT_BYTES = LEPTON_PACKET_SIZE * LEPTON_PACKETS_PER_SEGMENT
LEPTON_TELEMETRY_PACKETS_PER_SEGMENT = 61
LEPTON_TELEMETRY_SEGMENT_BYTES = LEPTON_PACKET_SIZE * LEPTON_TELEMETRY_PACKETS_PER_SEGMENT
LEPTON_SEGMENT_COUNT = 4
LEPTON_FRAME_SHAPE = (120, 160)
_LEPTON_FFC_STATES = ("never_commanded", "imminent", "in_progress", "complete")


class LeptonFrameAssembler:
    """Assemble 160x120 uint16 frames from the Pi streamer's VoSPI segment datagrams.

    Image-only segments contain 60 packets. Telemetry-footer segments contain 61;
    the first three segments carry 61 image half-rows, while segment four carries
    57 image half-rows followed by telemetry rows A, B, C, and padding. Segments
    must arrive strictly as 1, 2, 3, 4; any gap, reordering, or 60/61-packet mode
    change discards the partial frame rather than mixing adjacent frames.
    """

    def __init__(self) -> None:
        self._segments: dict[int, np.ndarray] = {}
        self._segment_bytes: int | None = None
        self._expected_segment = 1
        self.last_telemetry: LeptonTelemetry | None = None

    def _reset_partial(self) -> None:
        self._segments.clear()
        self._segment_bytes = None
        self._expected_segment = 1

    @staticmethod
    def _u32(words: np.ndarray, offset: int) -> int:
        return int(words[offset]) | (int(words[offset + 1]) << 16)

    @classmethod
    def _parse_telemetry(cls, packets: np.ndarray) -> LeptonTelemetry:
        words = packets[:, 4:].copy().view(">u2").astype(np.uint16)
        row_a = words[0]
        row_c = words[2]
        time_ms = cls._u32(row_a, 1)
        status = cls._u32(row_a, 3)
        last_ffc_ms = cls._u32(row_a, 30)
        ffc_state_index = (status >> 4) & 0x03
        resolution = {0: 0.1, 1: 0.01}.get(int(row_c[49]))
        return LeptonTelemetry(
            frame_counter=cls._u32(row_a, 20),
            packet_timestamp_ms=time_ms,
            ffc_desired=bool((status >> 3) & 0x01),
            ffc_state=_LEPTON_FFC_STATES[ffc_state_index],
            ffc_in_progress=ffc_state_index in (1, 2),
            since_last_ffc_s=((time_ms - last_ffc_ms) & 0xFFFFFFFF) / 1000.0,
            tlinear_enabled=int(row_c[48]) == 1,
            tlinear_resolution_k=resolution,
        )

    def add_datagram(self, data: bytes) -> np.ndarray | None:
        if len(data) not in (LEPTON_SEGMENT_BYTES, LEPTON_TELEMETRY_SEGMENT_BYTES):
            self._reset_partial()
            return None
        packet_count = len(data) // LEPTON_PACKET_SIZE
        packets = np.frombuffer(data, dtype=np.uint8).reshape(packet_count, LEPTON_PACKET_SIZE)
        packet_numbers = ((packets[:, 0] & 0x0F).astype(np.uint16) << 8) | packets[:, 1]
        if not np.array_equal(packet_numbers, np.arange(packet_count, dtype=np.uint16)):
            self._reset_partial()
            return None
        segment = (data[20 * LEPTON_PACKET_SIZE] >> 4) & 0x0F
        if not 1 <= segment <= LEPTON_SEGMENT_COUNT:
            self._reset_partial()
            return None

        if segment == 1:
            self._reset_partial()
            self._segment_bytes = len(data)
        elif segment != self._expected_segment or len(data) != self._segment_bytes:
            self._reset_partial()
            return None

        self._segments[segment] = packets.copy()
        self._expected_segment = segment + 1
        if segment < LEPTON_SEGMENT_COUNT:
            return None

        if self._segment_bytes == LEPTON_TELEMETRY_SEGMENT_BYTES:
            pixel_packets = np.concatenate(
                [self._segments[s] for s in range(1, 4)] + [self._segments[4][:57]]
            )
            self.last_telemetry = self._parse_telemetry(self._segments[4][57:61])
        else:
            pixel_packets = np.concatenate(
                [self._segments[s] for s in range(1, LEPTON_SEGMENT_COUNT + 1)]
            )
            self.last_telemetry = None
        pixels = pixel_packets[:, 4:].copy().view(">u2").astype(np.uint16)
        frame = np.hstack([pixels[0::2], pixels[1::2]])
        self._reset_partial()
        return frame


class LeptonUDPSource:
    """FrameSource reading raw Lepton 3.x frames streamed over UDP by the Pi."""

    def __init__(self, *, bind_ip: str = "0.0.0.0", port: int = 8080, timeout_s: float = 2.0):
        self.timeout_s = timeout_s
        self._assembler = LeptonFrameAssembler()
        self._sock: socket.socket | None = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(
            socket.SOL_SOCKET, socket.SO_RCVBUF, LEPTON_TELEMETRY_SEGMENT_BYTES * 16
        )
        self._sock.bind((bind_ip, port))
        self.port = self._sock.getsockname()[1]

    def read(self) -> FrameSample:
        if self._sock is None:
            raise FrameUnavailableError("lepton udp source is closed")
        deadline = time.perf_counter() + self.timeout_s
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise FrameUnavailableError(
                    f"no complete Lepton frame within {self.timeout_s:.1f}s on udp:{self.port}"
                )
            self._sock.settimeout(remaining)
            try:
                data, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError as exc:
                raise FrameUnavailableError(f"lepton udp socket error: {exc}") from exc
            frame = self._assembler.add_datagram(data)
            if frame is not None:
                telemetry = self._assembler.last_telemetry
                temperature_c = None
                if (
                    telemetry is not None
                    and telemetry.tlinear_enabled
                    and telemetry.tlinear_resolution_k is not None
                ):
                    temperature_c = (
                        frame.astype(np.float32) * telemetry.tlinear_resolution_k - 273.15
                    )
                return FrameSample(
                    t=time.perf_counter(),
                    frame=frame,
                    temperature_c=temperature_c,
                    lepton_telemetry=telemetry,
                )

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None


class LatestFrameSource:
    """Publish the newest sample from a blocking source without blocking consumers."""

    _CLOSE_TIMEOUT_S = 1.0

    def __init__(self, source: FrameSource):
        self.source = source
        self._lock = threading.Lock()
        self._latest: FrameSample | None = None
        self._error: Exception | None = None
        self._running = True
        self._closed = False
        self._source_close_called = False
        self._source_close_error: Exception | None = None
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()

    def _produce(self) -> None:
        while True:
            with self._lock:
                if not self._running:
                    return
            try:
                sample = self.source.read()
            except Exception as exc:
                with self._lock:
                    if self._running:
                        self._error = exc
                return
            with self._lock:
                if not self._running:
                    return
                self._latest = sample

    def read(self) -> FrameSample:
        with self._lock:
            if not self._running:
                raise FrameUnavailableError("latest frame source is closed")
            if self._error is not None:
                raise self._error
            if self._latest is None:
                raise FrameUnavailableError("latest frame unavailable before first producer sample")
            return self._latest

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._running = False
            close_source = not self._source_close_called
            self._source_close_called = True

        if close_source:
            close = getattr(self.source, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    self._source_close_error = exc

        self._thread.join(timeout=self._CLOSE_TIMEOUT_S)
        if self._thread.is_alive():
            raise RuntimeError(
                f"latest frame producer thread did not terminate within {self._CLOSE_TIMEOUT_S:.1f}s"
            ) from self._source_close_error
        if self._source_close_error is not None:
            raise self._source_close_error

        with self._lock:
            self._closed = True


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
