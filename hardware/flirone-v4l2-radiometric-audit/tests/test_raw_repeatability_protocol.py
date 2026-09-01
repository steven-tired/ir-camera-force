from __future__ import annotations

import json
import csv

import pytest

from raw_repeatability_protocol import ProtocolPhase


def test_dynamic_protocol_has_two_predeclared_hot_hand_cycles():
    protocol = __import__("raw_repeatability_protocol")

    phases = protocol.build_protocol("dynamic")

    assert [(phase.name, phase.duration_s) for phase in phases] == [
        ("baseline_01", 15.0),
        ("hot_hand_01", 15.0),
        ("recovery_01", 15.0),
        ("baseline_02", 15.0),
        ("hot_hand_02", 15.0),
        ("recovery_02", 15.0),
    ]


def test_fixed_display_run_requires_ordered_raw_count_bounds(tmp_path):
    protocol = __import__("raw_repeatability_protocol")

    with pytest.raises(ValueError, match="fixed display mode"):
        protocol.prepare_run(
            tmp_path / "session",
            run_id="dynamic_fixed",
            mode="dynamic",
            target_raw_roi="10,10,10,10",
            control_raw_roi="30,10,10,10",
            display_mode="fixed",
        )


def test_prepare_run_writes_raw_and_event_contract(tmp_path):
    protocol = __import__("raw_repeatability_protocol")

    run_dir = protocol.prepare_run(
        tmp_path / "session",
        run_id="restart_01",
        mode="restart",
        target_raw_roi="10,10,10,10",
        control_raw_roi="30,10,10,10",
        display_mode="dynamic",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert (run_dir / "raw").is_dir()
    assert manifest["mode"] == "restart"
    assert manifest["target_raw_roi"] == {"x": 10, "y": 10, "width": 10, "height": 10}
    assert manifest["display"] == {"mode": "dynamic", "raw_low": None, "raw_high": None}


def test_event_writer_records_monotonic_event_rows(tmp_path):
    recorder = __import__("record_raw_repeatability_events")
    events_path = tmp_path / "events.csv"
    events_path.write_text("timestamp_ns,event_type,phase,run_id\n")

    recorder.append_event(events_path, timestamp_ns=123, event_type="phase_start", phase="baseline_01", run_id="dynamic")

    with events_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"timestamp_ns": "123", "event_type": "phase_start", "phase": "baseline_01", "run_id": "dynamic"}]


def test_bridge_command_exports_raw_and_uses_fixed_count_range(tmp_path):
    recorder = __import__("record_raw_repeatability_events")

    command = recorder.build_bridge_command(
        audit_root=tmp_path / "audit",
        run_dir=tmp_path / "session" / "runs" / "dynamic_fixed",
        duration_s=90.0,
        display_mode="fixed",
        fixed_raw_low=3400,
        fixed_raw_high=3700,
    )

    assert command == [
        "sudo",
        "timeout",
        "--signal=INT",
        "--kill-after=5s",
        "120s",
        str(tmp_path / "audit" / "flirone"),
        str(tmp_path / "audit" / "palettes" / "Iron2.raw"),
        "--raw-dir",
        str(tmp_path / "session" / "runs" / "dynamic_fixed" / "raw"),
        "--raw-frame-limit",
        "0",
        "--fixed-raw-low",
        "3400",
        "--fixed-raw-high",
        "3700",
    ]


def test_capture_protocol_writes_rgb_rows_and_phase_events(tmp_path):
    recorder = __import__("record_raw_repeatability_events")

    class FakeClock:
        def __init__(self):
            self.seconds = 10.0

        def monotonic(self):
            return self.seconds

        def monotonic_ns(self):
            return round(self.seconds * 1_000_000_000)

        def sleep(self, seconds):
            self.seconds += seconds

    class FakeCapture:
        def read(self):
            return True, object()

    written = []
    run_dir = tmp_path / "run"
    (run_dir / "rgb").mkdir(parents=True)
    (run_dir / "events.csv").write_text("timestamp_ns,event_type,phase,run_id\n")
    (run_dir / "rgb_frames.csv").write_text("frame_index,timestamp_ns,file\n")
    clock = FakeClock()

    frame_count = recorder.capture_protocol(
        capture=FakeCapture(),
        run_dir=run_dir,
        run_id="dynamic",
        phases=[
            ProtocolPhase("baseline", 0.21, "hold"),
            ProtocolPhase("hot_hand", 0.21, "hold"),
        ],
        fps=10.0,
        monotonic=clock.monotonic,
        monotonic_ns=clock.monotonic_ns,
        sleep=clock.sleep,
        write_image=lambda path, image: written.append((path, image)) or True,
    )

    assert frame_count == 6
    assert [path.name for path, _ in written] == [f"frame_{index:06d}.png" for index in range(6)]
    with (run_dir / "events.csv").open(newline="") as handle:
        events = list(csv.DictReader(handle))
    assert [(event["event_type"], event["phase"]) for event in events] == [
        ("run_start", ""),
        ("phase_start", "baseline"),
        ("phase_end", "baseline"),
        ("phase_start", "hot_hand"),
        ("phase_end", "hot_hand"),
        ("run_end", ""),
    ]
    with (run_dir / "rgb_frames.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["file"] for row in rows] == [f"rgb/frame_{index:06d}.png" for index in range(6)]


def test_capture_protocol_records_and_recovers_from_one_transient_rgb_read_failure(tmp_path):
    recorder = __import__("record_raw_repeatability_events")

    class FakeClock:
        def __init__(self):
            self.seconds = 10.0

        def monotonic(self):
            return self.seconds

        def monotonic_ns(self):
            return round(self.seconds * 1_000_000_000)

        def sleep(self, seconds):
            self.seconds += seconds

    class OneFailureCapture:
        def __init__(self):
            self.read_count = 0

        def read(self):
            self.read_count += 1
            return (False, None) if self.read_count == 1 else (True, object())

    run_dir = tmp_path / "run"
    (run_dir / "rgb").mkdir(parents=True)
    (run_dir / "events.csv").write_text("timestamp_ns,event_type,phase,run_id\n")
    (run_dir / "rgb_frames.csv").write_text("frame_index,timestamp_ns,file\n")
    clock = FakeClock()

    frame_count = recorder.capture_protocol(
        capture=OneFailureCapture(),
        run_dir=run_dir,
        run_id="restart",
        phases=[ProtocolPhase("observation", 0.21, "hold")],
        fps=10.0,
        max_rgb_gap_s=0.5,
        monotonic=clock.monotonic,
        monotonic_ns=clock.monotonic_ns,
        sleep=clock.sleep,
        write_image=lambda path, image: True,
    )

    assert frame_count == 2
    with (run_dir / "events.csv").open(newline="") as handle:
        events = list(csv.DictReader(handle))
    assert ("rgb_read_failure", "observation") in [(event["event_type"], event["phase"]) for event in events]


def test_capture_protocol_stops_after_a_sustained_rgb_gap(tmp_path):
    recorder = __import__("record_raw_repeatability_events")

    class FakeClock:
        def __init__(self):
            self.seconds = 10.0

        def monotonic(self):
            return self.seconds

        def monotonic_ns(self):
            return round(self.seconds * 1_000_000_000)

        def sleep(self, seconds):
            self.seconds += seconds

    class FailedCapture:
        def read(self):
            return False, None

    run_dir = tmp_path / "run"
    (run_dir / "rgb").mkdir(parents=True)
    (run_dir / "events.csv").write_text("timestamp_ns,event_type,phase,run_id\n")
    (run_dir / "rgb_frames.csv").write_text("frame_index,timestamp_ns,file\n")
    clock = FakeClock()

    with pytest.raises(RuntimeError, match="RGB stream was unavailable"):
        recorder.capture_protocol(
            capture=FailedCapture(),
            run_dir=run_dir,
            run_id="restart",
            phases=[ProtocolPhase("observation", 1.0, "hold")],
            fps=10.0,
            max_rgb_gap_s=0.15,
            monotonic=clock.monotonic,
            monotonic_ns=clock.monotonic_ns,
            sleep=clock.sleep,
            write_image=lambda path, image: True,
        )


def test_recorder_cli_requires_explicit_session_and_two_raw_rois():
    recorder = __import__("record_raw_repeatability_events")

    args = recorder.parse_args(
        [
            "--session-root",
            "/tmp/session",
            "--run-id",
            "ffc_01",
            "--mode",
            "ffc",
            "--target-raw-roi",
            "10,10,10,10",
            "--control-raw-roi",
            "30,10,10,10",
        ]
    )

    assert args.session_root.name == "session"
    assert args.run_id == "ffc_01"
    assert args.thermal == "/dev/video21"
    assert args.fps == 10.0


def test_recorder_requires_an_explicit_ready_confirmation():
    recorder = __import__("record_raw_repeatability_events")

    with pytest.raises(RuntimeError, match="READY"):
        recorder.require_ready(lambda _: "almost")
    recorder.require_ready(lambda _: "READY")


def test_recorder_waits_for_raw_metadata_after_ready_before_capture(tmp_path):
    recorder = __import__("record_raw_repeatability_events")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    elapsed_s = 0.0

    def monotonic():
        return elapsed_s

    def sleep(seconds):
        nonlocal elapsed_s
        elapsed_s += seconds
        if elapsed_s >= 0.5:
            (raw_dir / "raw_frame_000000.json").write_text("{}")

    first_raw_frame = recorder.wait_for_raw_export(
        raw_dir,
        timeout_s=2.0,
        poll_interval_s=0.5,
        monotonic=monotonic,
        sleep=sleep,
    )

    assert first_raw_frame.name == "raw_frame_000000.json"
    assert elapsed_s == 0.5


def test_recorder_rejects_ready_when_raw_export_never_appears(tmp_path):
    recorder = __import__("record_raw_repeatability_events")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    elapsed_s = 0.0

    def monotonic():
        return elapsed_s

    def sleep(seconds):
        nonlocal elapsed_s
        elapsed_s += seconds

    with pytest.raises(RuntimeError, match="raw metadata"):
        recorder.wait_for_raw_export(
            raw_dir,
            timeout_s=1.0,
            poll_interval_s=0.5,
            monotonic=monotonic,
            sleep=sleep,
        )
