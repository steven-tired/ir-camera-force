import pytest

from ir_force.classifier.ir_robot import (
    TelemetrySnapshot,
    choose_three_grip_targets,
    serialize_telemetry_snapshot,
    slow_close_waypoints,
    summarize_target_current,
)


def test_slow_close_waypoints_are_monotonic_and_include_target():
    assert slow_close_waypoints(100.0, 40.0, steps=4) == [85.0, 70.0, 55.0, 40.0]


def test_slow_close_rejects_nonpositive_steps():
    with pytest.raises(ValueError, match="steps must be positive"):
        slow_close_waypoints(100.0, 40.0, steps=0)


def test_summarize_target_current_uses_hold_window_samples():
    samples = [
        TelemetrySnapshot(
            t=0.0,
            gripper_pos=90,
            goal_gripper_pos=80,
            present_current=10,
            present_load=3,
            present_temperature=30,
        ),
        TelemetrySnapshot(
            t=1.0,
            gripper_pos=80,
            goal_gripper_pos=80,
            present_current=20,
            present_load=4,
            present_temperature=31,
        ),
        TelemetrySnapshot(
            t=2.0,
            gripper_pos=80,
            goal_gripper_pos=80,
            present_current=30,
            present_load=5,
            present_temperature=32,
        ),
    ]
    summary = summarize_target_current(samples)
    assert summary["mean_current"] == 20.0
    assert summary["max_current"] == 30.0
    assert summary["mean_load"] == 4.0


def test_serialize_telemetry_snapshot_preserves_raw_fields():
    sample = TelemetrySnapshot(
        t=0.25,
        gripper_pos=81.5,
        goal_gripper_pos=80.0,
        present_current=22,
        present_load=4,
        present_temperature=31,
    )
    assert serialize_telemetry_snapshot(sample, target=80.0, sample_index=3) == {
        "target": 80.0,
        "sample_index": 3,
        "t": 0.25,
        "gripper_pos": 81.5,
        "goal_gripper_pos": 80.0,
        "present_current": 22,
        "present_load": 4,
        "present_temperature": 31,
    }


def test_choose_three_grip_targets_requires_separated_currents():
    records = [
        {"target": 85.0, "mean_current": 10.0},
        {"target": 70.0, "mean_current": 18.0},
        {"target": 55.0, "mean_current": 31.0},
        {"target": 40.0, "mean_current": 45.0},
    ]
    assert choose_three_grip_targets(records, min_current_gap=10.0) == {
        "low": 70.0,
        "med": 55.0,
        "high": 40.0,
    }


def test_choose_three_grip_targets_fails_when_current_gaps_are_small():
    records = [
        {"target": 90.0, "mean_current": 10.0},
        {"target": 80.0, "mean_current": 12.0},
        {"target": 70.0, "mean_current": 13.0},
    ]
    with pytest.raises(ValueError, match="could not find three separated grip targets"):
        choose_three_grip_targets(records, min_current_gap=10.0)
