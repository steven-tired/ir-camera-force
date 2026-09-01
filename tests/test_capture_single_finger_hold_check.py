import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import capture_single_finger_hold_check as runner
from ir_force.single_finger_click_roi import (
    rois_from_clicks,
)


TIP = (80.0, 93.0)
ALONG = (80.0, 60.0)
REFERENCE = (30.0, 110.0)
CLICKS = [TIP, ALONG, REFERENCE]


def _finger_frame(*, shift_x=0, distal_effect=0):
    frame = np.full((120, 160), 29_000, dtype=np.int32)
    frame[20:100, 0:30] = 29_450
    frame[20:55, 60 + shift_x : 100 + shift_x] = 29_600
    frame[55:95, 74 + shift_x : 86 + shift_x] = 29_620
    frame[79:90, 74 + shift_x : 86 + shift_x] += distal_effect
    return frame.astype(np.uint16)


class _FakeThermalSource:
    def __init__(self, frames):
        self._frames = list(frames)
        self._index = 0
        self.closed = False

    def read(self):
        frame = self._frames[min(self._index, len(self._frames) - 1)]
        self._index += 1
        return SimpleNamespace(
            t=float(self._index),
            frame=frame,
            temperature_c=None,
            lepton_telemetry=SimpleNamespace(
                frame_counter=self._index,
                ffc_desired=False,
            ),
        )

    def close(self):
        self.closed = True


class _KeySource:
    def __init__(self, keys):
        self._keys = list(keys)
        self.lines = []
        self.anchors = []
        self.points = []

    def __call__(self, thermal_counts, lines, anchor=None, points=()):
        self.lines.append(tuple(str(line) for line in lines))
        self.anchors.append(anchor)
        self.points.append(list(points))
        return self._keys.pop(0) if self._keys else -1


class _FakeClicks:
    """Reveal the clicks progressively so the prompt sequence is exercised."""

    def __init__(self, points, *, reveal_after=0):
        self._points = list(points)
        self._reveal_after = reveal_after
        self._calls = 0
        self.resets = 0
        self.undos = 0

    def take(self):
        self._calls += 1
        return list(self._points) if self._calls > self._reveal_after else []

    def reset(self):
        self.resets += 1

    def undo(self):
        self.undos += 1


class _StallingThermalSource(_FakeThermalSource):
    """Raise FrameUnavailableError on the reads listed in ``stall_on``."""

    def __init__(self, frames, stall_on):
        super().__init__(frames)
        self._stall_on = set(stall_on)
        self._reads = 0

    def read(self):
        self._reads += 1
        if self._reads in self._stall_on:
            raise runner.FrameUnavailableError("no complete Lepton frame")
        return super().read()


def _args(tmp_path, **overrides):
    values = {
        "session_dir": tmp_path / "single_finger_hold_check_01",
        "surface_material": "black electrical tape on desk",
        "phase_seconds": 30.0,
        "rounds": 2,
        "kinds": ["press", "press"],
        "lepton_port": 8080,
        "manual_ffc": False,
        "load_note": None,
    }
    values.update(overrides)
    if "kinds" not in overrides:
        values["kinds"] = ["press"] * values["rounds"]
    return SimpleNamespace(**values)


def test_parse_args_rejects_a_session_name_outside_the_hold_check_family(
    tmp_path,
):
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--session-dir",
                str(tmp_path / "single_finger_surface_press_curve_01"),
                "--surface-material",
                "tape",
            ]
        )


def test_parse_args_refuses_to_reopen_an_existing_session(tmp_path):
    session = tmp_path / "single_finger_hold_check_02"
    session.mkdir()

    with pytest.raises(SystemExit):
        runner.parse_args(
            ["--session-dir", str(session), "--surface-material", "tape"]
        )


def test_phase_at_walks_the_aba_round_and_then_ends():
    assert runner.phase_at(0.0, 30.0) == ("LIGHT_A", 0.0)
    assert runner.phase_at(30.0, 30.0) == ("HARD", 0.0)
    assert runner.phase_at(65.0, 30.0)[0] == "LIGHT_B"
    assert runner.phase_at(90.0, 30.0) is None


def test_the_round_has_no_lift_phase():
    """hold_check_01's OFF phase was unusable: the ROI left the finger."""
    assert runner.PHASES == ("LIGHT_A", "HARD", "LIGHT_B")


def test_collect_anchor_prompts_for_three_clicks_then_starts_on_space():
    source = _FakeThermalSource([_finger_frame()] * 8)
    keys = _KeySource([-1, -1, ord(" ")])
    clicks = _FakeClicks(CLICKS, reveal_after=2)

    collected = runner.collect_anchor(
        thermal_source=source,
        key_source=keys,
        click_source=clicks,
        round_index=0,
        rounds=2,
    )

    assert collected is not None
    anchor, frame = collected
    assert anchor["distal_pixel_count"] >= 20
    assert frame.dtype == np.uint16
    assert "click 1/3" in keys.lines[0][1]
    assert keys.anchors[-1] is not None


def test_a_further_click_starts_a_new_set_instead_of_being_ignored():
    """The first version appended forever and always read back points 0-2, so
    the ROI froze wherever the operator first clicked."""
    clicks = runner.MouseClicks.__new__(runner.MouseClicks)
    clicks.points = []
    scale = runner.PREVIEW_SCALE

    for x, y in ((40, 40), (60, 60), (80, 80)):
        clicks._on_event(cv2.EVENT_LBUTTONDOWN, x * scale, y * scale, 0, None)
    first_set = clicks.take()
    clicks._on_event(cv2.EVENT_LBUTTONDOWN, 12 * scale, 34 * scale, 0, None)

    assert first_set == [(40.0, 40.0), (60.0, 60.0), (80.0, 80.0)]
    assert clicks.take() == [(12.0, 34.0)]


def test_undo_drops_only_the_last_click():
    clicks = runner.MouseClicks.__new__(runner.MouseClicks)
    clicks.points = []
    scale = runner.PREVIEW_SCALE
    for x, y in ((40, 40), (60, 60)):
        clicks._on_event(cv2.EVENT_LBUTTONDOWN, x * scale, y * scale, 0, None)

    clicks.undo()

    assert clicks.take() == [(40.0, 40.0)]


def test_collect_anchor_forwards_clicks_to_the_preview():
    source = _FakeThermalSource([_finger_frame()] * 4)
    keys = _KeySource([ord("q")])

    runner.collect_anchor(
        thermal_source=source,
        key_source=keys,
        click_source=_FakeClicks([TIP, ALONG]),
        round_index=0,
        rounds=1,
    )

    assert keys.points[0] == [TIP, ALONG]


def test_collect_anchor_undoes_a_click_on_z():
    source = _FakeThermalSource([_finger_frame()] * 4)
    clicks = _FakeClicks(CLICKS)

    runner.collect_anchor(
        thermal_source=source,
        key_source=_KeySource([ord("z"), ord("q")]),
        click_source=clicks,
        round_index=0,
        rounds=1,
    )

    assert clicks.undos == 1


def test_collect_anchor_survives_a_lepton_stall():
    source = _StallingThermalSource([_finger_frame()] * 6, stall_on={1, 2})
    keys = _KeySource([-1, ord(" ")])

    collected = runner.collect_anchor(
        thermal_source=source,
        key_source=keys,
        click_source=_FakeClicks(CLICKS),
        round_index=0,
        rounds=1,
    )

    assert collected is not None
    assert "LEPTON STREAM STALLED" not in keys.lines[-1][0]


def test_a_stall_mid_round_is_recorded_as_a_gap_and_the_round_continues(
    tmp_path,
):
    """hold_check_03 lost rounds 2 and 3 to a 2 s dropout during the rest."""
    anchor = rois_from_clicks(_finger_frame(), TIP, ALONG, REFERENCE)
    source = _StallingThermalSource([_finger_frame()] * 6, stall_on={2})
    archive = runner.ThermalOnlyArchive(tmp_path / "session")
    stream = (tmp_path / "capture.jsonl").open("x", encoding="utf-8")
    try:
        captured = runner.capture_round(
            thermal_source=source,
            anchor=anchor,
            anchor_frame=_finger_frame(),
            archive=archive,
            stream=stream,
            round_index=0,
            start_frame_index=0,
            phase_seconds=5.0,
            rounds=1,
            clock=iter([0.0, 0.0, 3.0, 6.0, 11.0, 16.0]).__next__,
            key_source=_KeySource([-1] * 5),
        )
    finally:
        stream.close()

    assert captured["stream_gap_count"] == 1
    assert len(captured["rows"]) == 3
    written = [
        json.loads(line)
        for line in (tmp_path / "capture.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["row_type"] for row in written].count("stream_gap") == 1


def test_collect_anchor_rejects_bad_clicks_and_lets_the_operator_retry():
    source = _FakeThermalSource([_finger_frame()] * 6)
    keys = _KeySource([-1, ord("q")])
    # third click lands on the finger, so the reference patch is invalid
    clicks = _FakeClicks([TIP, ALONG, (80.0, 80.0)])

    collected = runner.collect_anchor(
        thermal_source=source,
        key_source=keys,
        click_source=clicks,
        round_index=0,
        rounds=2,
    )

    assert collected is None
    assert "rejected: reference_patch_touches_finger" in keys.lines[0][1]
    # The rejected clicks stay on screen so the operator can see what was
    # wrong; clicking again starts a new set.
    assert keys.points[0] == [TIP, ALONG, (80.0, 80.0)]
    assert keys.anchors[0] is None


def test_round_summary_scores_the_press_against_both_light_phases():
    def rows_for(light_a, hard, light_b):
        rows = []
        for phase, value in (
            ("LIGHT_A", light_a),
            ("HARD", hard),
            ("LIGHT_B", light_b),
        ):
            rows.append(
                {
                    "phase": phase,
                    "tracked": {"primary_signal_count": value},
                }
            )
        return rows

    recovered = runner.round_summary(rows_for(50.0, 0.0, 50.0))
    assert recovered["aba_effect_count"] == -50.0
    assert recovered["return_recovery_ratio"] == 0.0

    # A monotone drift never comes back, so the effect is inflated and the
    # recovery ratio exposes it.
    drifting = runner.round_summary(rows_for(50.0, 0.0, -50.0))
    assert drifting["aba_effect_count"] == 0.0
    assert drifting["return_recovery_ratio"] == 2.0


def test_run_session_records_two_clicked_and_tracked_aba_rounds(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "MIN_VALID_FRAMES_PER_PHASE", 1)
    args = _args(tmp_path, phase_seconds=5.0, rounds=2)
    light = _finger_frame()
    hard = _finger_frame(distal_effect=-60)
    # anchor, LIGHT_A, HARD, LIGHT_B, loop-exit, rest, then the same again
    source = _FakeThermalSource(
        [light, light, hard, light, light, light]
        + [light, light, hard, light, light]
    )
    # one clock read per loop entry, plus the round start
    times = iter(
        [0.0, 0.0, 6.0, 11.0, 16.0, 0.0, 0.0, 6.0, 11.0, 16.0]
    ).__next__
    keys = _KeySource(
        [ord(" ")] + [-1] * 3 + [ord(" ")] + [ord(" ")] + [-1] * 3
    )

    manifest = runner.run_session(
        args,
        thermal_source_factory=lambda: source,
        key_source=keys,
        click_source_factory=lambda: _FakeClicks(CLICKS),
        clock=times,
        manual_ffc=lambda: None,
        sleep=lambda seconds: None,
    )

    assert manifest["status"] == "complete"
    assert manifest["rounds_recorded"] == 2
    assert manifest["role"] == "sanity_check_not_preregistered"
    assert manifest["signal_verdict"] == "not_a_formal_result"
    assert manifest["controller_or_robot_actuation"] is False
    assert manifest["force_ground_truth"] is False
    assert source.closed is True

    first = manifest["rounds"][0]["phase_summary"]
    assert first["LIGHT_A"]["valid_frames"] == 1
    assert first["HARD"]["median_count"] < first["LIGHT_A"]["median_count"]
    assert first["aba_effect_count"] < -40
    assert manifest["median_aba_effect_count"] < -40

    assert (args.session_dir / "figures/aba_rounds.png").is_file()
    assert (args.session_dir / "figures/roi_round1.png").is_file()
    assert (args.session_dir / "figures/roi_round2.png").is_file()

    rows = [
        json.loads(line)
        for line in (args.session_dir / "capture.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    metadata = rows[0]
    assert metadata["experiment_identity"] == "single_finger_hold_check_v2"
    assert metadata["modalities"] == ["lepton_thermal"]
    assert metadata["d435_used"] is False
    assert metadata["roi_method"] == "operator_clicked_tip_axis_reference"
    assert metadata["roi_tracking"] == "template_ncc_translation"
    anchors = [row for row in rows if row["row_type"] == "round_anchor"]
    assert len(anchors) == 2
    assert anchors[0]["overlay_figure"] == "roi_round1.png"
    frames = [row for row in rows if row["row_type"] == "frame"]
    assert [row["phase"] for row in frames[:3]] == list(runner.PHASES)


def test_round_diagnostics_surface_the_ffc_drift_and_shift(tmp_path):
    rows = [
        {
            "phase": "HARD",
            "frame_median_count": 29_400.0,
            "ffc_desired": False,
            "tracked": {"shift_magnitude_px": 1.0, "reference_count": 29_200.0},
        },
        {
            "phase": "HARD",
            "frame_median_count": 29_100.0,
            "ffc_desired": True,
            "tracked": {"shift_magnitude_px": 3.0, "reference_count": 29_180.0},
        },
    ]

    diagnostics = runner.round_diagnostics(rows)

    assert diagnostics["ffc_desired_fraction"] == 0.5
    assert diagnostics["frame_median_drift_count"] == -300.0
    assert diagnostics["median_shift_px_by_phase"]["HARD"] == 2.0
    assert diagnostics["median_shift_px_by_phase"]["LIGHT_A"] is None
    assert diagnostics["median_reference_count_by_phase"]["HARD"] == 29_190.0


def test_manual_ffc_runs_before_every_round_not_only_the_first(
    tmp_path, monkeypatch
):
    """hold_check_04 FFC'd once: rounds 2 and 3 ran at 96% and 100%
    ffc_desired."""
    monkeypatch.setattr(runner, "MIN_VALID_FRAMES_PER_PHASE", 1)
    args = _args(tmp_path, phase_seconds=5.0, rounds=2, manual_ffc=True)
    light = _finger_frame()
    hard = _finger_frame(distal_effect=-60)
    calls = []
    sources = []

    def factory():
        source = _FakeThermalSource(
            [light, light, hard, light, light, light] * 2
        )
        sources.append(source)
        return source

    manifest = runner.run_session(
        args,
        thermal_source_factory=factory,
        key_source=_KeySource(
            [ord(" ")] + [-1] * 3 + [ord(" ")] + [ord(" ")] + [-1] * 3
        ),
        click_source_factory=lambda: _FakeClicks(CLICKS),
        clock=iter([0.0, 0.0, 6.0, 11.0, 16.0] * 2).__next__,
        manual_ffc=lambda: calls.append("ffc"),
        sleep=lambda seconds: calls.append("guard"),
    )

    assert manifest["status"] == "complete"
    assert calls == ["ffc", "guard", "ffc", "guard"]
    assert len(sources) == 2
    assert sources[0].closed is True
    assert sources[1].closed is True
    assert manifest["rounds"][0]["diagnostics"]["ffc_desired_fraction"] == 0.0


def test_run_session_runs_manual_ffc_and_guards_before_the_first_round(
    tmp_path,
):
    args = _args(tmp_path, phase_seconds=5.0, rounds=1, manual_ffc=True)
    calls = []

    manifest = runner.run_session(
        args,
        thermal_source_factory=lambda: _FakeThermalSource(
            [_finger_frame()] * 4
        ),
        key_source=_KeySource([ord("q")]),
        click_source_factory=lambda: _FakeClicks(CLICKS),
        clock=iter([0.0, 0.0]).__next__,
        manual_ffc=lambda: calls.append("ffc"),
        sleep=lambda seconds: calls.append(("sleep", seconds)),
    )

    assert calls == ["ffc", ("sleep", runner.FFC_GUARD_S)]
    assert manifest["status"] == "aborted"
    assert manifest["rounds_recorded"] == 0


def test_source_declares_no_robot_controller_or_gripper_dependency():
    source = Path(runner.__file__).read_text(encoding="utf-8")

    for forbidden in (
        "ee_controller",
        "teleop_viz_ee",
        "record_so101_ee",
        "SO101",
        "pyrealsense2",
        "mediapipe",
    ):
        assert forbidden not in source


def test_a_round_without_enough_frames_per_phase_is_not_counted_as_valid(
    tmp_path,
):
    """hold_check_05 round 3 lost the stream at the LIGHT_A/HARD boundary and
    recorded zero LIGHT_B frames, yet the session reported complete."""
    args = _args(tmp_path, phase_seconds=5.0, rounds=1)
    light = _finger_frame()

    manifest = runner.run_session(
        args,
        thermal_source_factory=lambda: _FakeThermalSource([light] * 8),
        key_source=_KeySource([ord(" ")] + [-1] * 3),
        click_source_factory=lambda: _FakeClicks(CLICKS),
        clock=iter([0.0, 0.0, 6.0, 11.0, 16.0]).__next__,
        manual_ffc=lambda: None,
        sleep=lambda seconds: None,
    )

    assert manifest["rounds_recorded"] == 1
    assert manifest["valid_round_count"] == 0
    assert manifest["status"] == "incomplete"
    assert manifest["rounds"][0]["phase_summary"]["valid"] is False
    assert manifest["median_aba_effect_count"] is None


def test_a_control_round_asks_for_movement_without_extra_force(tmp_path):
    """Press and geometry are confounded: every press shifts the ROI 2-4 px on
    a 10 px wide finger, and the shift distributions barely overlap."""
    assert "do NOT press harder" in runner.PHASE_CUES["control"]["HARD"]
    assert runner.PHASE_CUES["press"]["HARD"] != (
        runner.PHASE_CUES["control"]["HARD"]
    )

    parsed = runner.parse_args(
        [
            "--session-dir",
            str(tmp_path / "single_finger_hold_check_09"),
            "--surface-material",
            "tape",
            "--kinds",
            "press,control,press,control",
        ]
    )

    assert parsed.kinds == ["press", "control", "press", "control"]
    assert parsed.rounds == 4


def test_unknown_round_kinds_are_rejected(tmp_path):
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--session-dir",
                str(tmp_path / "single_finger_hold_check_09"),
                "--surface-material",
                "tape",
                "--kinds",
                "press,squeeze",
            ]
        )


def test_the_round_stops_once_the_stream_has_been_gone_long_enough(tmp_path):
    anchor = rois_from_clicks(_finger_frame(), TIP, ALONG, REFERENCE)
    source = _StallingThermalSource(
        [_finger_frame()] * 4,
        stall_on=set(range(1, 40)),
    )
    archive = runner.ThermalOnlyArchive(tmp_path / "session")
    stream = (tmp_path / "capture.jsonl").open("x", encoding="utf-8")
    try:
        captured = runner.capture_round(
            thermal_source=source,
            anchor=anchor,
            anchor_frame=_finger_frame(),
            archive=archive,
            stream=stream,
            round_index=0,
            start_frame_index=0,
            phase_seconds=5.0,
            rounds=1,
            clock=lambda: 0.0,
            key_source=_KeySource([]),
        )
    finally:
        stream.close()

    assert captured["stream_lost"] is True
    assert captured["stream_gap_count"] == runner.MAX_CONSECUTIVE_STREAM_GAPS
    assert captured["rows"] == []
