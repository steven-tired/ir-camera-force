import csv
from pathlib import Path
import socket
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import ir_force.ir_capture as ir_capture

from ir_force.ir_capture import (
    CaptureSources,
    FrameSample,
    FrameUnavailableError,
    LEPTON_FRAME_SHAPE,
    LEPTON_PACKET_SIZE,
    LEPTON_SEGMENT_BYTES,
    LEPTON_TELEMETRY_SEGMENT_BYTES,
    LeptonFrameAssembler,
    LeptonTelemetry,
    LeptonUDPSource,
    capture_setup_snapshot,
    record_capture_window,
    record_labeled_camera_window,
)
from ir_force.ir_dataset import HandPressureTrialSpec, create_hand_pressure_trial_paths
from ir_force.ir_dataset import TrialSpec, create_trial_paths
from lerobot_teleoperator_so101_webcam.gripper_hardware import TelemetrySnapshot


class FakeFrameSource:
    def __init__(self, value):
        self.value = value
        self.count = 0

    def read(self):
        self.count += 1
        frame = np.full((4, 5), self.value, dtype=np.uint8)
        return FrameSample(t=float(self.count), frame=frame)


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

    monkeypatch.setattr("ir_force.ir_capture._write_frame", fake_write_frame)

    capture_setup_snapshot(
        paths,
        thermal=FakeFrameSource(7),
        bird=FakeFrameSource(8),
        flir_visible=FakeFrameSource(9),
    )

    assert written == ["thermal.png", "bird.png", "flir_visible.png"]


def _latest_frame_source_type():
    source_type = getattr(ir_capture, "LatestFrameSource", None)
    assert source_type is not None, "pressure runtime latest-frame wrapper is missing"
    return source_type


class _ControlledFrameSource:
    def __init__(self, outcome):
        self.outcome = outcome
        self.read_started = threading.Event()
        self.allow_read = threading.Event()
        self.read_finished = threading.Event()
        self.close_calls = 0

    def read(self):
        self.read_started.set()
        self.allow_read.wait(timeout=1.0)
        try:
            if isinstance(self.outcome, Exception):
                raise self.outcome
            return self.outcome
        finally:
            self.read_finished.set()

    def close(self):
        self.close_calls += 1
        self.allow_read.set()


class _NonUnblockingFrameSource:
    def __init__(self):
        self.read_started = threading.Event()
        self.allow_read = threading.Event()
        self.close_calls = 0

    def read(self):
        self.read_started.set()
        self.allow_read.wait()
        return FrameSample(t=12.5, frame=np.ones((2, 3), np.uint8))

    def close(self):
        self.close_calls += 1


def _wait_for_read(source, *, timeout_s=1.0):
    deadline = time.perf_counter() + timeout_s
    while True:
        try:
            return source.read()
        except RuntimeError as exc:
            if "unavailable" not in str(exc).lower() or time.perf_counter() >= deadline:
                raise
            time.sleep(0.001)


def test_latest_frame_source_returns_unavailable_before_first_frame_without_blocking():
    wrapped = _ControlledFrameSource(FrameSample(t=12.5, frame=np.ones((2, 3), np.uint8)))
    source = _latest_frame_source_type()(wrapped)
    assert wrapped.read_started.wait(timeout=1.0)

    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="(?i)unavailable"):
        source.read()
    elapsed_s = time.perf_counter() - started

    source.close()
    assert elapsed_s < 0.05


def test_latest_frame_source_retains_producer_timestamp_and_closes_once():
    expected = FrameSample(t=12.5, frame=np.ones((2, 3), np.uint8))
    wrapped = _ControlledFrameSource(expected)
    source = _latest_frame_source_type()(wrapped)
    assert wrapped.read_started.wait(timeout=1.0)
    wrapped.allow_read.set()
    assert wrapped.read_finished.wait(timeout=1.0)

    actual = _wait_for_read(source)
    source.close()

    with pytest.raises(RuntimeError, match="(?i)closed"):
        source.read()

    source.close()

    assert actual.t == 12.5
    np.testing.assert_array_equal(actual.frame, expected.frame)
    assert wrapped.close_calls == 1


def test_latest_frame_source_close_reports_producer_that_does_not_terminate():
    wrapped = _NonUnblockingFrameSource()
    source = _latest_frame_source_type()(wrapped)
    assert wrapped.read_started.wait(timeout=1.0)

    started = time.perf_counter()
    try:
        with pytest.raises(RuntimeError, match="producer thread did not terminate"):
            source.close()
        assert time.perf_counter() - started < 1.5
        assert source._thread.is_alive()
        with pytest.raises(RuntimeError, match="(?i)closed"):
            source.read()

        wrapped.allow_read.set()
        source.close()
        source.close()
        assert not source._thread.is_alive()
        assert wrapped.close_calls == 1
    finally:
        wrapped.allow_read.set()
        source._thread.join(timeout=1.0)


def test_latest_frame_source_exposes_producer_errors():
    wrapped = _ControlledFrameSource(RuntimeError("producer failed"))
    source = _latest_frame_source_type()(wrapped)
    assert wrapped.read_started.wait(timeout=1.0)
    wrapped.allow_read.set()
    assert wrapped.read_finished.wait(timeout=1.0)

    with pytest.raises(RuntimeError, match="producer failed"):
        source.read()

    source.close()


def test_live_pressure_builder_explicitly_uses_latest_frame_wrapper(tmp_path, monkeypatch):
    sys.modules.setdefault("mediapipe", SimpleNamespace(solutions=SimpleNamespace()))
    import teleop_viz_ee
    from ir_force.ir_hand_calibration import (
        ProjectionCalibration,
        save_projection_calibration,
    )

    path = Path(tmp_path) / "projection.json"
    save_projection_calibration(
        path,
        ProjectionCalibration(
            (0.0, 160.0, 0.0, 0.0),
            (0.0, 0.0, 128.0, 0.0),
            1.0,
            2.0,
            12,
        ),
    )
    wrapped = []

    class FakeThermal:
        def __init__(self, thermal_path):
            self.thermal_path = thermal_path

    class FakeLatestFrameSource:
        def __init__(self, source):
            self.source = source
            wrapped.append(self)

    monkeypatch.setattr(teleop_viz_ee, "OpenCVCameraSource", FakeThermal)
    monkeypatch.setattr(
        teleop_viz_ee,
        "LatestFrameSource",
        FakeLatestFrameSource,
        raising=False,
    )

    estimator = teleop_viz_ee.build_ir_pressure_source(
        enabled=True,
        calibration_path=str(path),
        thermal_path="/dev/video21",
    )

    assert estimator.thermal_source is wrapped[0]
    assert wrapped[0].source.thermal_path == "/dev/video21"


# ---------------------------------------------------------------------------
# Lepton UDP source (VoSPI-over-UDP from the Pi streamer)
# ---------------------------------------------------------------------------


def _lepton_test_frame() -> np.ndarray:
    rows = np.arange(120, dtype=np.uint16)[:, None]
    cols = np.arange(160, dtype=np.uint16)[None, :]
    return (rows * 200 + cols).astype(np.uint16)


def _segment_datagram(frame: np.ndarray, segment: int) -> bytes:
    rows = frame[30 * (segment - 1) : 30 * segment]
    packets = []
    for j in range(60):
        row = rows[j // 2]
        half = row[80 * (j % 2) : 80 * (j % 2) + 80]
        first = (segment << 4) if j == 20 else 0
        header = bytes((first, j, 0, 0))
        packets.append(header + half.astype(">u2").tobytes())
    datagram = b"".join(packets)
    assert len(datagram) == LEPTON_SEGMENT_BYTES
    return datagram


def _telemetry_footer_datagrams(
    frame: np.ndarray,
    *,
    time_ms: int = 125_000,
    last_ffc_ms: int = 120_000,
    frame_counter: int = 17,
    ffc_desired: bool = False,
    ffc_state: int = 3,
    tlinear_enabled: bool = True,
    tlinear_resolution: int = 1,
) -> list[bytes]:
    pixel_packets = []
    for row in frame:
        pixel_packets.extend((row[:80], row[80:]))

    row_a = np.zeros(80, dtype=np.uint16)
    row_a[1:3] = (time_ms & 0xFFFF, time_ms >> 16)
    status = (int(ffc_desired) << 3) | (ffc_state << 4)
    row_a[3:5] = (status & 0xFFFF, status >> 16)
    row_a[20:22] = (frame_counter & 0xFFFF, frame_counter >> 16)
    row_a[30:32] = (last_ffc_ms & 0xFFFF, last_ffc_ms >> 16)

    row_b = np.zeros(80, dtype=np.uint16)
    row_c = np.zeros(80, dtype=np.uint16)
    row_c[48] = int(tlinear_enabled)
    row_c[49] = tlinear_resolution
    padding = np.zeros(80, dtype=np.uint16)
    telemetry_packets = (row_a, row_b, row_c, padding)

    datagrams = []
    pixel_offset = 0
    for segment in range(1, 5):
        packet_payloads = []
        pixel_count = 61 if segment < 4 else 57
        packet_payloads.extend(pixel_packets[pixel_offset : pixel_offset + pixel_count])
        pixel_offset += pixel_count
        if segment == 4:
            packet_payloads.extend(telemetry_packets)

        packets = []
        for packet_number, payload in enumerate(packet_payloads):
            first = (segment << 4) if packet_number == 20 else 0
            header = bytes((first, packet_number, 0, 0))
            packets.append(header + np.asarray(payload, dtype=">u2").tobytes())
        datagram = b"".join(packets)
        assert len(datagram) == LEPTON_TELEMETRY_SEGMENT_BYTES
        datagrams.append(datagram)

    assert pixel_offset == 240
    return datagrams


def test_lepton_assembler_assembles_in_order_segments():
    frame = _lepton_test_frame()
    assembler = LeptonFrameAssembler()
    results = [assembler.add_datagram(_segment_datagram(frame, s)) for s in (1, 2, 3, 4)]
    assert results[:3] == [None, None, None]
    assembled = results[3]
    assert assembled.shape == LEPTON_FRAME_SHAPE
    assert assembled.dtype == np.uint16
    np.testing.assert_array_equal(assembled, frame)


def test_lepton_assembler_discards_out_of_order_segments():
    frame = _lepton_test_frame()
    assembler = LeptonFrameAssembler()
    results = [assembler.add_datagram(_segment_datagram(frame, s)) for s in (2, 1, 4, 3)]
    assert results == [None, None, None, None]

    results = [assembler.add_datagram(_segment_datagram(frame, s)) for s in (1, 2, 3, 4)]
    assert results[:3] == [None, None, None]
    np.testing.assert_array_equal(results[3], frame)


@pytest.mark.parametrize("missing_segment", (1, 2, 3, 4))
def test_lepton_assembler_never_mixes_frames_after_a_missing_segment(missing_segment):
    stale = _lepton_test_frame()
    fresh = stale + 1000
    assembler = LeptonFrameAssembler()

    for segment in (1, 2, 3, 4):
        if segment != missing_segment:
            assert assembler.add_datagram(_segment_datagram(stale, segment)) is None

    results = [assembler.add_datagram(_segment_datagram(fresh, s)) for s in (1, 2, 3, 4)]
    completed = [result for result in results if result is not None]
    assert len(completed) == 1
    np.testing.assert_array_equal(completed[0], fresh)


def test_lepton_assembler_restarts_on_repeated_segment_after_loss():
    stale = _lepton_test_frame()
    fresh = stale + 1000
    assembler = LeptonFrameAssembler()
    # Frame N loses segment 3: only 1, 2, 4 arrive.
    for s in (1, 2, 4):
        assert assembler.add_datagram(_segment_datagram(stale, s)) is None
    # Frame N+1 arrives complete; repeated segment 1 must discard the stale partial.
    results = [assembler.add_datagram(_segment_datagram(fresh, s)) for s in (1, 2, 3, 4)]
    assembled = results[3]
    assert assembled is not None
    np.testing.assert_array_equal(assembled, fresh)


def test_lepton_assembler_ignores_wrong_size_and_bad_segment():
    frame = _lepton_test_frame()
    assembler = LeptonFrameAssembler()
    assert assembler.add_datagram(b"short") is None
    bad_segment = bytearray(_segment_datagram(frame, 1))
    bad_segment[20 * LEPTON_PACKET_SIZE] = 0x50  # segment 5: invalid
    assert assembler.add_datagram(bytes(bad_segment)) is None
    results = [assembler.add_datagram(_segment_datagram(frame, s)) for s in (1, 2, 3, 4)]
    assert results[3] is not None


def test_lepton_assembler_parses_footer_telemetry_without_losing_image_rows():
    frame = _lepton_test_frame() + 30_000
    assembler = LeptonFrameAssembler()
    results = [assembler.add_datagram(data) for data in _telemetry_footer_datagrams(frame)]

    assembled = results[-1]
    assert assembled is not None
    np.testing.assert_array_equal(assembled, frame)
    assert assembler.last_telemetry == LeptonTelemetry(
        frame_counter=17,
        packet_timestamp_ms=125_000,
        ffc_desired=False,
        ffc_state="complete",
        ffc_in_progress=False,
        since_last_ffc_s=5.0,
        tlinear_enabled=True,
        tlinear_resolution_k=0.01,
    )


def test_lepton_assembler_preserves_raw_footer_packet_timestamp_ms():
    frame = _lepton_test_frame() + 30_000
    assembler = LeptonFrameAssembler()
    packet_timestamp_ms = 0xF1234567

    for data in _telemetry_footer_datagrams(frame, time_ms=packet_timestamp_ms):
        assembler.add_datagram(data)

    assert assembler.last_telemetry is not None
    assert assembler.last_telemetry.packet_timestamp_ms == packet_timestamp_ms


def test_lepton_assembler_never_mixes_image_only_and_telemetry_segments():
    frame = _lepton_test_frame() + 30_000
    assembler = LeptonFrameAssembler()
    telemetry = _telemetry_footer_datagrams(frame)

    assert assembler.add_datagram(_segment_datagram(frame, 1)) is None
    assert assembler.add_datagram(_segment_datagram(frame, 2)) is None
    assert assembler.add_datagram(telemetry[2]) is None
    assert assembler.add_datagram(telemetry[3]) is None
    results = [assembler.add_datagram(data) for data in telemetry]

    completed = [result for result in results if result is not None]
    assert len(completed) == 1
    np.testing.assert_array_equal(completed[0], frame)


def test_lepton_udp_source_reads_frame_from_loopback_socket():
    frame = _lepton_test_frame()
    source = LeptonUDPSource(bind_ip="127.0.0.1", port=0, timeout_s=2.0)
    try:
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for s in (1, 2, 3, 4):
                sender.sendto(_segment_datagram(frame, s), ("127.0.0.1", source.port))
        finally:
            sender.close()
        sample = source.read()
        assert isinstance(sample, FrameSample)
        assert sample.t > 0.0
        np.testing.assert_array_equal(sample.frame, frame)
    finally:
        source.close()


def test_lepton_udp_source_exposes_tlinear_celsius_and_frame_telemetry():
    frame = np.full(LEPTON_FRAME_SHAPE, 30_000, dtype=np.uint16)
    source = LeptonUDPSource(bind_ip="127.0.0.1", port=0, timeout_s=2.0)
    try:
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for datagram in _telemetry_footer_datagrams(frame, ffc_state=2):
                sender.sendto(datagram, ("127.0.0.1", source.port))
        finally:
            sender.close()

        sample = source.read()
        np.testing.assert_array_equal(sample.frame, frame)
        assert sample.temperature_c is not None
        assert sample.temperature_c.dtype == np.float32
        np.testing.assert_allclose(sample.temperature_c, 26.85, atol=1e-4)
        assert sample.lepton_telemetry is not None
        assert sample.lepton_telemetry.ffc_in_progress is True
    finally:
        source.close()


def test_lepton_udp_source_read_times_out_as_frame_unavailable():
    source = LeptonUDPSource(bind_ip="127.0.0.1", port=0, timeout_s=0.1)
    try:
        started = time.perf_counter()
        with pytest.raises(FrameUnavailableError):
            source.read()
        assert time.perf_counter() - started < 2.0
    finally:
        source.close()


def test_lepton_udp_source_read_after_close_raises():
    source = LeptonUDPSource(bind_ip="127.0.0.1", port=0, timeout_s=0.1)
    source.close()
    with pytest.raises(FrameUnavailableError):
        source.read()
