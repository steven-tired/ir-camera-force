import ast
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import capture_single_finger_curve as runner
from ir_force.single_finger_curve_protocol import (
    TrialSpec,
    trial_integrity,
)


def _required_args(tmp_path):
    photo = tmp_path / "surface.jpg"
    photo.write_bytes(b"surface")
    return [
        "--session-dir",
        str(tmp_path / "single_finger_surface_press_curve_01"),
        "--surface-material",
        "rigid plastic",
        "--surface-photo",
        str(photo),
        "--preview",
        "--manual-ffc",
    ]


def test_cli_requires_bounded_session_surface_preview_and_manual_ffc(tmp_path):
    args = runner.parse_args(_required_args(tmp_path))

    assert args.session_dir.name == "single_finger_surface_press_curve_01"
    assert args.surface_material == "rigid plastic"
    assert args.lepton_port == 8080
    for missing in ("--preview", "--manual-ffc"):
        argv = _required_args(tmp_path)
        argv.remove(missing)
        with pytest.raises(SystemExit, match="2"):
            runner.parse_args(argv)


def test_cli_rejects_wrong_or_existing_session_but_allows_missing_photo(tmp_path):
    argv = _required_args(tmp_path)
    argv[1] = str(tmp_path / "wrong_name")
    with pytest.raises(SystemExit, match="2"):
        runner.parse_args(argv)

    existing = tmp_path / "single_finger_surface_press_curve_02"
    existing.mkdir()
    argv = _required_args(tmp_path)
    argv[1] = str(existing)
    with pytest.raises(SystemExit, match="2"):
        runner.parse_args(argv)

    argv = _required_args(tmp_path)
    argv[5] = str(tmp_path / "missing.jpg")
    args = runner.parse_args(argv)
    assert args.surface_photo is None


def test_entry_point_has_no_robot_or_controller_imports():
    tree = ast.parse(Path(runner.__file__).read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")

    forbidden = ("gripper_hardware", "record_so101", "gripper", "controller", "deploy")
    assert not [
        module
        for module in imported
        if any(token in module for token in forbidden)
    ]


def test_json_writer_preserves_booleans_instead_of_coercing_them_to_integers():
    assert runner._json_value(True) is True
    assert runner._json_value(False) is False


@pytest.mark.parametrize(
    ("phase", "condition", "expected"),
    [
        ("A1", "null", "A1: LIGHT CONTACT"),
        ("X", "null", "X/null: KEEP LIGHT CONTACT"),
        ("X", "press", "X/press: PRESS HARD"),
        ("A2", "press", "A2: LIGHT CONTACT"),
        ("A3", "null", "A3: LIFT - NO CONTACT"),
        ("REST", "press", "REST: NO CONTACT"),
    ],
)
def test_phase_cues_are_fixed(phase, condition, expected):
    assert runner.phase_cue(phase, condition) == expected


def test_reserves_run_only_after_all_primary_blocks_and_until_six_valid():
    realized = runner.realized_block_indices(
        [True, True, False, True, True, True, True, True]
    )

    assert realized == [0, 1, 2, 3, 4, 5, 6]
    assert runner.realized_block_indices(
        [True, False, False, True, True, True, True, False]
    ) == list(range(8))


class _FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


class _ThermalSource:
    def __init__(self, clock):
        self.clock = clock
        self.read_count = 0

    def read(self):
        self.clock.value += 0.1
        self.read_count += 1
        telemetry = SimpleNamespace(
            ffc_desired=False,
            ffc_state="complete",
            ffc_in_progress=False,
        )
        return SimpleNamespace(
            t=self.clock.value,
            frame=np.full((120, 160), self.read_count, dtype=np.uint16),
            lepton_telemetry=telemetry,
        )


class _Archive:
    def __init__(self):
        self.frame_indices = []

    def capture(self, *, frame_index, **_kwargs):
        self.frame_indices.append(frame_index)
        return {"thermal_uint16": f"raw/{frame_index:06d}.png"}


def _frame_builder(_raw, thermal, _hands, _projection, trial):
    return {
        "row_type": "frame",
        **{key: value for key, value in trial.items() if key != "now_s"},
        "thermal_host_s": thermal.t,
        "tracking_valid": True,
        "ffc_in_progress": False,
        "ffc_state": "complete",
        "artifact_write_ok": True,
        "primary_signal_count": 999999.0,
    }


def test_trial_records_every_in_window_frame_and_advances_phases_automatically():
    clock = _FakeClock()
    thermal = _ThermalSource(clock)
    archive = _Archive()
    stream = io.StringIO()
    raw = SimpleNamespace(
        color_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth_z16=np.zeros((4, 4), dtype=np.uint16),
        color_frame_number=1,
        depth_frame_number=1,
    )

    result = runner.capture_trial_frames(
        spec=TrialSpec(0, "press", 0, False),
        trial_index=0,
        start_frame_index=0,
        raw_reader=lambda: raw,
        thermal_source=thermal,
        hands=object(),
        projection_context={},
        archive=archive,
        stream=stream,
        clock=clock,
        key_source=lambda *_args: -1,
        frame_builder=_frame_builder,
    )

    rows = result["rows"]
    assert result["aborted"] is False
    assert len(rows) == len(archive.frame_indices)
    assert len(rows) >= 190
    assert {row["phase"] for row in rows} == {"A1", "X", "A2", "A3"}
    first_by_phase = {
        phase: next(row for row in rows if row["phase"] == phase)
        for phase in ("A1", "X", "A2", "A3")
    }
    assert first_by_phase["X"]["global_elapsed_s"] >= 5.0
    assert first_by_phase["A2"]["global_elapsed_s"] >= 10.0
    assert first_by_phase["A3"]["global_elapsed_s"] >= 15.0
    assert all(row["global_elapsed_s"] < 20.0 for row in rows)
    assert trial_integrity(rows)["valid"] is True


def test_ready_gate_requires_complete_d435_to_thermal_tracking_chain():
    raw = SimpleNamespace(
        color_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
    )
    thermal = SimpleNamespace(
        frame=np.zeros((120, 160), dtype=np.uint16),
    )
    tracking_rows = iter(
        (
            {
                "tracking_valid": False,
                "tracking_reasons": ["TIP:color_to_depth_sdk_no_match"],
            },
            {
                "tracking_valid": True,
                "tracking_reasons": [],
            },
        )
    )
    displayed = []

    def frame_builder(*_args):
        return next(tracking_rows)

    ready = runner._wait_for_space(
        spec=TrialSpec(0, "press", 0, False),
        raw_reader=lambda: raw,
        thermal_source=SimpleNamespace(read=lambda: thermal),
        hands=object(),
        projection_context={"projection": "required"},
        key_source=lambda _color, _thermal, lines: (
            displayed.append(lines) or ord(" ")
        ),
        readiness_mode="d435_and_thermal",
        frame_builder=frame_builder,
        thermal_anchor_builder=lambda _frames: {
            "finger_width_px": 12.0,
        },
        thermal_window_size=1,
    )

    assert ready is True
    assert len(displayed) == 2
    assert "NOT READY" in displayed[0][1]
    assert "TIP:color_to_depth_sdk_no_match" in displayed[0][1]
    assert "READY" in displayed[1][1]


def test_default_ready_gate_records_the_d435_chain_without_gating_on_it():
    """Session 01 lost every frame to the D435 chain at this working distance.

    The v2 primary value is thermal-only, so the failure is recorded as a
    diagnostic instead of blocking the capture.
    """
    raw = SimpleNamespace(color_rgb=np.zeros((4, 4, 3), dtype=np.uint8))
    thermal = SimpleNamespace(frame=np.zeros((120, 160), dtype=np.uint16))
    displayed = []

    ready = runner._wait_for_space(
        spec=TrialSpec(0, "press", 0, False),
        raw_reader=lambda: raw,
        thermal_source=SimpleNamespace(read=lambda: thermal),
        hands=object(),
        projection_context={},
        key_source=lambda _color, _thermal, lines: (
            displayed.append(lines) or ord(" ")
        ),
        frame_builder=lambda *_args: {
            "tracking_valid": False,
            "tracking_reasons": ["TIP:color_to_depth_sdk_no_match"],
        },
        thermal_anchor_builder=lambda _frames: {"finger_width_px": 12.0},
        thermal_window_size=1,
    )

    assert ready is True
    assert len(displayed) == 1
    assert "READY" in displayed[0][1]
    assert "not gating" in displayed[0][1]
    assert "TIP:color_to_depth_sdk_no_match" in displayed[0][1]


def test_thermal_width_still_gates_when_the_d435_chain_is_not_gating():
    raw = SimpleNamespace(color_rgb=np.zeros((4, 4, 3), dtype=np.uint8))
    thermal = SimpleNamespace(frame=np.zeros((120, 160), dtype=np.uint16))
    widths = iter((8.0, 12.0))
    displayed = []

    ready = runner._wait_for_space(
        spec=TrialSpec(0, "press", 0, False),
        raw_reader=lambda: raw,
        thermal_source=SimpleNamespace(read=lambda: thermal),
        hands=object(),
        projection_context={},
        key_source=lambda _color, _thermal, lines: (
            displayed.append(lines) or ord(" ")
        ),
        frame_builder=lambda *_args: {
            "tracking_valid": False,
            "tracking_reasons": ["TIP:color_to_depth_sdk_no_match"],
        },
        thermal_anchor_builder=lambda _frames: {
            "finger_width_px": next(widths),
        },
        thermal_window_size=1,
    )

    assert ready is True
    assert len(displayed) == 2
    assert "NOT READY" in displayed[0][1]
    assert "thermal width 8.0px" in displayed[0][1]


def test_ready_gate_requires_approximately_ten_thermal_pixels_of_finger():
    raw = SimpleNamespace(
        color_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
    )
    thermal = SimpleNamespace(
        frame=np.zeros((120, 160), dtype=np.uint16),
    )
    widths = iter((8.0, 12.0))
    displayed = []

    ready = runner._wait_for_space(
        spec=TrialSpec(0, "press", 0, False),
        raw_reader=lambda: raw,
        thermal_source=SimpleNamespace(read=lambda: thermal),
        hands=object(),
        projection_context={},
        key_source=lambda _color, _thermal, lines: (
            displayed.append(lines) or ord(" ")
        ),
        frame_builder=lambda *_args: {
            "tracking_valid": True,
            "tracking_reasons": [],
        },
        thermal_anchor_builder=lambda _frames: {
            "finger_width_px": next(widths),
        },
        thermal_window_size=1,
    )

    assert ready is True
    assert len(displayed) == 2
    assert "thermal width 8.0px" in displayed[0][1]
    assert "NOT READY" in displayed[0][1]
    assert "thermal width 12.0px" in displayed[1][1]
