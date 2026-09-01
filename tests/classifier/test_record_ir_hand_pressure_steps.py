from __future__ import annotations

import importlib
import json


def test_step_parser_defaults_to_hysteresis_protocol():
    module = importlib.import_module("record_ir_hand_pressure_steps")

    args = module._parse_args(
        [
            "--rep",
            "1",
        ]
    )

    assert args.thermal == "/dev/video21"
    assert args.bird == ""
    assert args.flir_visible == "/dev/video20"
    assert args.record_flir_visible is False
    assert args.root.endswith("ir_hand_pressure_hysteresis")
    assert args.levels == ("zero", "light", "medium", "hard", "medium", "light", "zero")
    assert args.cycles == 5
    assert args.hold_s == 3.0
    assert args.pre_baseline_s == 3.0
    assert not hasattr(args, "force_kg")
    assert not hasattr(args, "pressure_level")


def test_step_plan_marks_up_down_hysteresis_levels():
    module = importlib.import_module("record_ir_hand_pressure_steps")

    steps = module._build_step_plan(("zero", "light", "medium", "hard", "medium", "light", "zero"), 1)

    assert [step.direction for step in steps] == ["baseline", "up", "up", "up", "down", "down", "down"]
    assert [step.value for step in steps] == [0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0]
    assert steps[2].name == "cycle01_step02_medium"
    assert steps[4].name == "cycle01_step04_medium"


def test_prepare_trial_writes_step_metadata_without_force_labels(tmp_path):
    module = importlib.import_module("record_ir_hand_pressure_steps")

    args = module._parse_args(
        [
            "--rep",
            "2",
            "--root",
            str(tmp_path),
            "--notes",
            "metronome stepped squeeze",
        ]
    )

    _spec, paths, steps = module._prepare_trial(args)

    metadata = json.loads(paths.metadata_path.read_text())
    assert paths.trial_id == "hand-pressure_foam_whole-hand_sweep_rep02"
    assert metadata["recording_mode"] == "stepped_squeeze_hysteresis"
    assert metadata["bird_path"] == ""
    assert metadata["record_bird"] is False
    assert metadata["levels"] == ["zero", "light", "medium", "hard", "medium", "light", "zero"]
    assert metadata["cycles"] == 5
    assert metadata["analysis_targets"] == [
        "lag_after_level_change",
        "up_down_hysteresis_at_same_level",
        "zero_level_drift",
        "ir_change_after_aperture_plateau",
    ]
    assert metadata["notes"] == "metronome stepped squeeze"
    assert "force_kg" not in metadata
    assert "pressure_level" not in metadata
    assert len(steps) == 35


def test_prepare_trial_records_optional_bird_path(tmp_path):
    module = importlib.import_module("record_ir_hand_pressure_steps")

    args = module._parse_args(
        [
            "--rep",
            "3",
            "--bird",
            "/dev/video-test",
            "--root",
            str(tmp_path),
        ]
    )

    _spec, paths, _steps = module._prepare_trial(args)

    metadata = json.loads(paths.metadata_path.read_text())
    assert metadata["bird_path"] == "/dev/video-test"
    assert metadata["record_bird"] is True
