from __future__ import annotations

import importlib
import json


def test_hand_pressure_parser_defaults_to_safe_camera_only_capture():
    module = importlib.import_module("record_ir_hand_pressure_trial")

    args = module._parse_args(
        [
            "--surface",
            "brick",
            "--contact",
            "fingertip",
            "--rep",
            "1",
            "--bird",
            "/tmp/bird",
        ]
    )

    assert args.thermal == "/dev/video21"
    assert args.flir_visible == "/dev/video20"
    assert args.record_flir_visible is False
    assert args.baseline_s == 2.0
    assert args.press_s == 5.0
    assert args.fps == 10.0
    assert not hasattr(args, "pressure_level")
    assert not hasattr(args, "force_kg")


def test_hand_pressure_parser_can_default_to_foam_hand_experiment():
    module = importlib.import_module("record_ir_hand_pressure_trial")

    args = module._parse_args(
        [
            "--rep",
            "3",
            "--bird",
            "/tmp/bird",
            "--notes",
            "bird camera corrected",
        ]
    )

    assert args.surface == "foam"
    assert args.contact == "whole hand"
    assert args.hold_s == 1.0
    assert args.release_s == 5.0
    assert args.notes == "bird camera corrected"
    assert not hasattr(args, "pressure_level")
    assert not hasattr(args, "force_kg")


def test_prepare_trial_writes_hand_pressure_metadata(tmp_path):
    module = importlib.import_module("record_ir_hand_pressure_trial")

    args = module._parse_args(
        [
            "--surface",
            "red brick",
            "--contact",
            "index fingertip",
            "--rep",
            "2",
            "--bird",
            "/tmp/bird",
            "--root",
            str(tmp_path),
        ]
    )

    _spec, paths = module._prepare_trial(args)

    meta = json.loads(paths.metadata_path.read_text())
    assert paths.trial_id == "hand-pressure_red-brick_index-fingertip_sweep_rep02"
    assert meta["surface"] == "red brick"
    assert meta["contact"] == "index fingertip"
    assert meta["recording_mode"] == "continuous_pressure_sweep"
    assert "pressure_level" not in meta
    assert "force_kg" not in meta
    assert meta["thermal_path"] == "/dev/video21"


def test_prepare_trial_writes_release_and_notes_metadata(tmp_path):
    module = importlib.import_module("record_ir_hand_pressure_trial")

    args = module._parse_args(
        [
            "--rep",
            "4",
            "--bird",
            "/tmp/bird",
            "--root",
            str(tmp_path),
            "--hold-s",
            "2.5",
            "--release-s",
            "6.5",
            "--notes",
            "foam centered in bird view",
        ]
    )

    _spec, paths = module._prepare_trial(args)

    meta = json.loads(paths.metadata_path.read_text())
    assert paths.trial_id == "hand-pressure_foam_whole-hand_sweep_rep04"
    assert meta["surface"] == "foam"
    assert meta["contact"] == "whole hand"
    assert meta["hold_s"] == 2.5
    assert meta["release_s"] == 6.5
    assert meta["notes"] == "foam centered in bird view"
    assert "pressure_level" not in meta
    assert "force_kg" not in meta
