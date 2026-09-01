from __future__ import annotations

import importlib


def test_sweep_parser_defaults_to_continuous_close_protocol():
    module = importlib.import_module("record_ir_grip_sweep")

    args = module._parse_args(
        [
            "--object",
            "foam-block",
            "--hardness",
            "soft",
            "--rep",
            "5",
            "--bird",
            "/tmp/bird",
        ]
    )

    assert args.open_pos == 100.0
    assert args.target_pos == 10.0
    assert args.baseline_s == 2.0
    assert args.sweep_s == 12.0
    assert args.hold_s == 3.0
    assert args.release_pos == 90.0
    assert args.sweep_start_pos is None
    assert args.pre_baseline_settle_s == 0.0
    assert args.record_flir_visible is False


def test_sweep_start_defaults_to_open_position():
    module = importlib.import_module("record_ir_grip_sweep")

    args = module._parse_args(
        [
            "--object",
            "foam-block",
            "--hardness",
            "soft",
            "--rep",
            "5",
            "--bird",
            "/tmp/bird",
            "--open-pos",
            "55",
        ]
    )

    assert module._sweep_start_pos(args) == 55.0


def test_sweep_start_can_be_set_separately_from_placement_open():
    module = importlib.import_module("record_ir_grip_sweep")

    args = module._parse_args(
        [
            "--object",
            "foam-block",
            "--hardness",
            "soft",
            "--rep",
            "5",
            "--bird",
            "/tmp/bird",
            "--open-pos",
            "90",
            "--sweep-start-pos",
            "50",
            "--pre-baseline-settle-s",
            "1.5",
        ]
    )

    assert args.open_pos == 90.0
    assert module._sweep_start_pos(args) == 50.0
    assert args.pre_baseline_settle_s == 1.5


def test_sweep_goal_interpolates_and_clamps():
    module = importlib.import_module("record_ir_grip_sweep")

    assert module._sweep_goal(open_pos=100.0, target_pos=10.0, elapsed_s=-1.0, duration_s=12.0) == 100.0
    assert module._sweep_goal(open_pos=100.0, target_pos=10.0, elapsed_s=6.0, duration_s=12.0) == 55.0
    assert module._sweep_goal(open_pos=100.0, target_pos=10.0, elapsed_s=20.0, duration_s=12.0) == 10.0


def test_sweep_telemetry_source_tracks_updated_goal(monkeypatch):
    module = importlib.import_module("record_ir_grip_sweep")

    class FakeRobot:
        pass

    seen: list[float] = []

    def fake_read_gripper_telemetry(robot, goal_gripper_pos: float, t: float):
        seen.append(goal_gripper_pos)
        return object()

    monkeypatch.setattr(module, "read_gripper_telemetry", fake_read_gripper_telemetry)
    source = module.SweepRobotTelemetrySource(FakeRobot(), goal_gripper_pos=100.0, t0=0.0)

    source.set_goal(42.0)
    source.read()

    assert seen == [42.0]
