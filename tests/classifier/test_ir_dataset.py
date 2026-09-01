import csv
import json
import pytest

from ir_force.classifier.ir_dataset import (
    DEFAULT_FLIR_VISIBLE_PATH,
    DEFAULT_THERMAL_PATH,
    HAND_PRESSURE_STREAM_KIND,
    THERMAL_STREAM_KIND,
    HandPressureTrialSpec,
    TrialSpec,
    append_telemetry_row,
    create_hand_pressure_trial_paths,
    create_trial_paths,
    ensure_fresh_trial,
    hand_pressure_trial_id,
    trial_id,
    write_hand_pressure_metadata,
    write_metadata,
)


def test_trial_id_is_stable_and_filesystem_safe():
    spec = TrialSpec(object_name="foam block", hardness="soft", grip_level="med", rep=2)
    assert trial_id(spec) == "foam-block_soft_med_rep02"


def test_warmed_trial_id_is_marked():
    spec = TrialSpec(
        object_name="wood block",
        hardness="solid",
        grip_level="high",
        rep=1,
        warmed=True,
    )
    assert trial_id(spec) == "wood-block_solid_high_rep01_warmed"


def test_xhigh_trial_id_is_supported_for_extra_probe_trial():
    spec = TrialSpec(object_name="foam block", hardness="soft", grip_level="xhigh", rep=1)

    assert trial_id(spec) == "foam-block_soft_xhigh_rep01"


def test_trial_id_supports_more_than_three_repeats():
    spec = TrialSpec(object_name="foam block", hardness="soft", grip_level="low", rep=4)

    assert trial_id(spec) == "foam-block_soft_low_rep04"


def test_sweep_trial_id_is_supported_for_continuous_capture():
    spec = TrialSpec(object_name="foam block", hardness="soft", grip_level="sweep", rep=5)

    assert trial_id(spec) == "foam-block_soft_sweep_rep05"


def test_hand_pressure_trial_id_and_metadata_describe_continuous_sweep(tmp_path):
    spec = HandPressureTrialSpec(
        surface="red brick",
        contact="index fingertip",
        rep=2,
    )
    paths = create_hand_pressure_trial_paths(tmp_path, spec)

    assert hand_pressure_trial_id(spec) == "hand-pressure_red-brick_index-fingertip_sweep_rep02"

    write_hand_pressure_metadata(
        paths,
        spec,
        {
            "recording_mode": "continuous_pressure_sweep",
            "thermal_roi": "25,35,115,80",
        },
    )

    meta = json.loads(paths.metadata_path.read_text())
    assert meta["experiment_kind"] == HAND_PRESSURE_STREAM_KIND
    assert meta["surface"] == "red brick"
    assert meta["contact"] == "index fingertip"
    assert meta["recording_mode"] == "continuous_pressure_sweep"
    assert "pressure_level" not in meta
    assert "force_kg" not in meta
    assert meta["thermal_roi"] == "25,35,115,80"
    assert meta["thermal_stream_kind"] == THERMAL_STREAM_KIND


@pytest.mark.parametrize(
    ("kwargs", "message"),
        [
            ({"surface": ""}, "surface"),
            ({"contact": ""}, "contact"),
            ({"rep": 0}, "rep"),
        ],
    )
def test_hand_pressure_trial_spec_rejects_invalid_values(kwargs, message):
    base = {
        "surface": "brick",
        "contact": "fingertip",
        "rep": 1,
    }
    base.update(kwargs)

    with pytest.raises(ValueError, match=message):
        HandPressureTrialSpec(**base)


def test_create_trial_paths_makes_directories(tmp_path):
    spec = TrialSpec(object_name="sponge", hardness="soft", grip_level="low", rep=1)
    paths = create_trial_paths(tmp_path, spec)
    assert paths.metadata_path.parent.exists()
    assert paths.thermal_dir.exists()
    assert paths.bird_dir.exists()
    assert paths.flir_visible_dir.exists()
    assert paths.preflight_dir.exists()
    assert paths.overlays_dir.exists()
    assert paths.plots_dir.exists()


def test_metadata_and_telemetry_are_written(tmp_path):
    spec = TrialSpec(object_name="plastic", hardness="solid", grip_level="high", rep=3)
    paths = create_trial_paths(tmp_path, spec)
    write_metadata(paths, spec, {"thermal_path": "/dev/video21"})
    meta = json.loads(paths.metadata_path.read_text())
    assert meta["object_name"] == "plastic"
    assert meta["hardness"] == "solid"
    assert meta["thermal_path"] == "/dev/video21"

    append_telemetry_row(paths.telemetry_csv, {"t": 0.0, "gripper_pos": 50.0, "present_current": 11})
    append_telemetry_row(paths.telemetry_csv, {"t": 0.1, "gripper_pos": 51.0, "present_current": 12})
    rows = list(csv.DictReader(paths.telemetry_csv.open()))
    assert rows[0]["gripper_pos"] == "50.0"
    assert rows[1]["present_current"] == "12"


def test_write_metadata_includes_default_acquisition_facts(tmp_path):
    spec = TrialSpec(object_name="foam cube", hardness="soft", grip_level="low", rep=1)
    paths = create_trial_paths(tmp_path, spec)

    write_metadata(paths, spec, {})

    meta = json.loads(paths.metadata_path.read_text())
    assert meta["thermal_path"] == DEFAULT_THERMAL_PATH
    assert meta["flir_visible_path"] == DEFAULT_FLIR_VISIBLE_PATH
    assert meta["thermal_stream_kind"] == THERMAL_STREAM_KIND


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"object_name": ""}, "object_name"),
        ({"hardness": "rubbery"}, "hardness"),
        ({"grip_level": "medium"}, "grip_level"),
        ({"rep": 0}, "rep"),
    ],
)
def test_trial_spec_rejects_invalid_values(kwargs, message):
    base = {"object_name": "foam cube", "hardness": "soft", "grip_level": "low", "rep": 1}
    base.update(kwargs)

    with pytest.raises(ValueError, match=message):
        TrialSpec(**base)


def test_append_telemetry_row_rejects_key_mismatch(tmp_path):
    csv_path = tmp_path / "telemetry.csv"
    append_telemetry_row(csv_path, {"t": 0.0, "gripper_pos": 50.0, "present_current": 11})

    with pytest.raises(ValueError, match="header"):
        append_telemetry_row(csv_path, {"t": 0.1, "gripper_pos": 51.0, "servo_temp": 42.0})


def test_ensure_fresh_trial_rejects_existing_capture_data(tmp_path):
    spec = TrialSpec(object_name="foam cube", hardness="soft", grip_level="high", rep=2)
    paths = create_trial_paths(tmp_path, spec)
    append_telemetry_row(paths.telemetry_csv, {"t": 0.0, "gripper_pos": 50.0, "present_current": 11})

    with pytest.raises(FileExistsError, match="already contains capture data"):
        ensure_fresh_trial(paths)
