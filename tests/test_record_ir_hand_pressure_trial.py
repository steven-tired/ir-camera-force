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


def test_hand_pressure_parser_lepton_udp_disables_flir_visible_by_default():
    module = importlib.import_module("record_ir_hand_pressure_trial")

    args = module._parse_args(["--rep", "1", "--bird", "/tmp/bird", "--lepton-udp"])

    assert args.lepton_udp == 8080
    assert args.flir_visible == ""
    assert module._thermal_source_label(args) == "lepton-udp:8080"

    explicit = module._parse_args(
        ["--rep", "1", "--bird", "/tmp/bird", "--lepton-udp", "9000"]
    )
    assert explicit.lepton_udp == 9000

    default = module._parse_args(["--rep", "1", "--bird", "/tmp/bird"])
    assert default.lepton_udp is None
    assert default.flir_visible == "/dev/video20"
    assert module._thermal_source_label(default) == "/dev/video21"


def test_prepare_trial_records_lepton_thermal_source_in_metadata(tmp_path):
    module = importlib.import_module("record_ir_hand_pressure_trial")

    args = module._parse_args(
        [
            "--rep",
            "1",
            "--bird",
            "/tmp/bird",
            "--lepton-udp",
            "--root",
            str(tmp_path),
        ]
    )
    _, paths = module._prepare_trial(args)

    meta = json.loads(paths.metadata_path.read_text())
    assert meta["thermal_path"] == "lepton-udp:8080"
    assert meta["thermal_stream_kind"] == "lepton_raw_uint16_counts"
    assert meta["flir_visible_path"] == ""


def test_build_thermal_source_uses_lepton_udp_when_requested(monkeypatch):
    module = importlib.import_module("record_ir_hand_pressure_trial")
    built = {}

    class FakeLepton:
        def __init__(self, *, port):
            built["port"] = port

    monkeypatch.setattr(module, "LeptonUDPSource", FakeLepton)
    args = module._parse_args(["--rep", "1", "--bird", "/tmp/bird", "--lepton-udp", "9001"])

    source = module._build_thermal_source(args)

    assert isinstance(source, FakeLepton)
    assert built["port"] == 9001
