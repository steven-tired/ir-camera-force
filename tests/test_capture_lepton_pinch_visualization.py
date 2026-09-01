import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import capture_lepton_pinch_visualization as capture


def _valid_argv(session_dir):
    return [
        "--frames",
        "900",
        "--session-dir",
        str(session_dir),
        "--lepton-port",
        "8080",
        "--manual-ffc",
        "--preview",
    ]


def _contact_null_argv(session_dir):
    return [*_valid_argv(session_dir), "--contact-only"]


def test_capture_cli_requires_visualization_identity_preview_and_manual_ffc(
    tmp_path,
):
    session_dir = tmp_path / "stage1e_tip_pinch_visualization_01"

    args = capture.parse_args(_valid_argv(session_dir))

    assert args.frames == 900
    assert args.session_dir == session_dir
    assert args.lepton_port == 8080
    assert args.preview is True
    assert args.manual_ffc is True
    for missing in ("--preview", "--manual-ffc"):
        argv = _valid_argv(session_dir)
        argv.remove(missing)
        with pytest.raises(SystemExit, match="2"):
            capture.parse_args(argv)
    for invalid_name in (
        "stage1e_tip_pinch_signal_06",
        "stage1e_tip_pinch_visualization_attempt01",
        "visualization_01",
    ):
        argv = _valid_argv(tmp_path / invalid_name)
        with pytest.raises(SystemExit, match="2"):
            capture.parse_args(argv)


def test_contact_null_cli_changes_only_the_physical_middle_instruction(
    tmp_path,
):
    session_dir = tmp_path / "stage1e_tip_pinch_contact_null_01"

    args = capture.parse_args(_contact_null_argv(session_dir))

    assert args.contact_only is True
    assert args.session_dir == session_dir
    assert capture._contact_only_instruction(
        {"phase": "prepare_press_hard"}
    ) == "KEEP JUST TOUCH - SPACE WHEN READY"
    assert capture._contact_only_instruction(
        {"phase": "record_press_hard"}
    ) == "HOLD JUST TOUCH - NO PRESS"
    assert capture._contact_only_instruction(
        {"phase": "record_just_touch"}
    ) == "HOLD JUST TOUCH"
    assert capture._contact_only_display_label(
        {"phase": "record_press_hard", "label": "press"}
    ) == "contact"
    with pytest.raises(SystemExit, match="2"):
        capture.parse_args(_valid_argv(session_dir))


def test_capture_session_directory_is_created_exclusively(tmp_path):
    session_dir = tmp_path / "stage1e_tip_pinch_visualization_01"

    capture.create_session_dir(session_dir)

    assert session_dir.is_dir()
    with pytest.raises(FileExistsError):
        capture.create_session_dir(session_dir)


def test_capture_main_runs_frozen_protocol_then_render_and_analysis(
    monkeypatch,
    tmp_path,
):
    session_dir = tmp_path / "stage1e_tip_pinch_visualization_01"
    fake_rs = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)
    monkeypatch.setattr(
        capture.runner,
        "_run_manual_ffc",
        lambda: "Manual FFC complete",
    )
    archive = object()
    monkeypatch.setattr(capture, "FrameArchive", lambda path: (
        archive if path == session_dir else pytest.fail("wrong session")
    ))
    runner_calls = []

    def run_shadow(**kwargs):
        runner_calls.append(kwargs)
        return {
            "pinch_signal_protocol_completed": True,
            "software_gate_accepted": 90,
        }

    monkeypatch.setattr(capture.runner, "run_shadow", run_shadow)
    render_calls = []
    monkeypatch.setattr(
        capture,
        "render_session",
        lambda jsonl, session: render_calls.append((jsonl, session)),
    )
    analysis_calls = []
    monkeypatch.setattr(
        capture.visualization_analyzer,
        "main",
        lambda argv: analysis_calls.append(argv) or 0,
    )

    result = capture.main(_valid_argv(session_dir))

    assert result == 0
    assert session_dir.is_dir()
    assert len(runner_calls) == 1
    call = runner_calls[0]
    assert call["attempts"] == 900
    assert call["output_path"] == session_dir / "capture.jsonl"
    assert call["rs_module"] is fake_rs
    assert call["preview"] is True
    assert call["manual_ffc_before_start"] is True
    assert call["diagnose_inward_samples"] is False
    assert call["pinch_signal_trial"] is True
    assert call["attempt_artifact_writer"] is archive
    assert callable(call["raw_source_factory"])
    assert callable(call["thermal_source_factory"])
    assert call["hands_factory"] is capture.runner._default_hands_factory
    assert render_calls == [
        (session_dir / "capture.jsonl", session_dir)
    ]
    assert analysis_calls == [
        [
            "--input",
            str(session_dir / "capture.jsonl"),
            "--output",
            str(session_dir / "descriptive_analysis.json"),
        ]
    ]


def test_capture_entry_imports_no_robot_or_control_modules():
    tree = ast.parse(Path(capture.__file__).read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    assert not any(
        forbidden in module
        for module in imported
        for forbidden in (
            "control",
            "deploy",
            "record_so101",
            "teleop_viz",
            "ir_robot",
        )
    )
