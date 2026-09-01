from __future__ import annotations

import importlib

import pytest


MODULE_NAMES = [
    "record_ir_grip_trial",
    "characterize_ir_grip_current",
]


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_cleanup_robot_opens_before_disconnect(module_name: str, monkeypatch: pytest.MonkeyPatch):
    module = importlib.import_module(module_name)
    events: list[object] = []

    class FakeRobot:
        def disconnect(self) -> None:
            events.append("disconnect")

    def fake_send_gripper(robot: FakeRobot, gripper_pos: float) -> None:
        events.append(("send", gripper_pos))

    monkeypatch.setattr(module, "_send_gripper", fake_send_gripper)

    module._cleanup_robot(FakeRobot(), open_pos=100.0)

    assert events == [("send", 100.0), "disconnect"]


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_cleanup_robot_warns_and_preserves_original_exception(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    module = importlib.import_module(module_name)
    events: list[str] = []

    class FakeRobot:
        def disconnect(self) -> None:
            events.append("disconnect")

    def fake_send_gripper(robot: FakeRobot, gripper_pos: float) -> None:
        raise RuntimeError("jammed")

    monkeypatch.setattr(module, "_send_gripper", fake_send_gripper)

    with pytest.raises(RuntimeError, match="boom"):
        try:
            raise RuntimeError("boom")
        finally:
            module._cleanup_robot(FakeRobot(), open_pos=100.0)

    assert events == ["disconnect"]
    assert "warning: failed to open gripper before disconnect: jammed" in capsys.readouterr().out


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_cleanup_robot_preserves_original_exception_when_disconnect_fails(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    module = importlib.import_module(module_name)

    class FakeRobot:
        def disconnect(self) -> None:
            raise RuntimeError("link down")

    def fake_send_gripper(robot: FakeRobot, gripper_pos: float) -> None:
        return None

    monkeypatch.setattr(module, "_send_gripper", fake_send_gripper)

    with pytest.raises(RuntimeError, match="boom"):
        try:
            raise RuntimeError("boom")
        finally:
            module._cleanup_robot(FakeRobot(), open_pos=100.0)

    assert "warning: failed to disconnect robot: link down" in capsys.readouterr().out


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_connect_robot_disables_torque_on_disconnect(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
):
    module = importlib.import_module(module_name)
    seen: dict[str, object] = {}

    class FakeFollower:
        def __init__(self, config):
            seen["config"] = config

        def connect(self, calibrate: bool) -> None:
            seen["calibrate"] = calibrate

    monkeypatch.setattr(module, "SOFollower", FakeFollower)

    module._connect_robot("/tmp/fake-port")

    config = seen["config"]
    assert config.port == "/tmp/fake-port"
    assert config.use_degrees is False
    assert config.disable_torque_on_disconnect is True
    assert seen["calibrate"] is False


def test_record_trial_release_position_defaults_to_90_while_open_stays_100():
    module = importlib.import_module("record_ir_grip_trial")

    args = module._parse_args(
        [
            "--object",
            "hard-block",
            "--hardness",
            "solid",
            "--grip-level",
            "low",
            "--rep",
            "1",
            "--bird",
            "/tmp/bird",
        ]
    )

    assert args.open_pos == 100.0
    assert args.release_pos == 90.0


def test_record_trial_accepts_xhigh_extra_probe_level():
    module = importlib.import_module("record_ir_grip_trial")

    args = module._parse_args(
        [
            "--object",
            "foam-block",
            "--hardness",
            "soft",
            "--grip-level",
            "xhigh",
            "--rep",
            "1",
            "--bird",
            "/tmp/bird",
        ]
    )

    assert module._requested_grip_levels(args) == ("xhigh",)


def test_record_trial_accepts_comma_separated_grip_level_sequence():
    module = importlib.import_module("record_ir_grip_trial")

    args = module._parse_args(
        [
            "--object",
            "foam-block",
            "--hardness",
            "soft",
            "--grip-level",
            "low,med,high,xhigh",
            "--rep",
            "2",
            "--bird",
            "/tmp/bird",
        ]
    )

    assert module._requested_grip_levels(args) == ("low", "med", "high", "xhigh")


def test_record_trial_accepts_space_separated_grip_level_sequence():
    module = importlib.import_module("record_ir_grip_trial")

    args = module._parse_args(
        [
            "--object",
            "foam-block",
            "--hardness",
            "soft",
            "--grip-level",
            "low",
            "med",
            "high",
            "xhigh",
            "--rep",
            "2",
            "--bird",
            "/tmp/bird",
        ]
    )

    assert module._requested_grip_levels(args) == ("low", "med", "high", "xhigh")


def test_record_trial_records_flir_visible_frames_only_when_requested():
    module = importlib.import_module("record_ir_grip_trial")

    default_args = module._parse_args(
        [
            "--object",
            "foam-block",
            "--hardness",
            "soft",
            "--grip-level",
            "low",
            "--rep",
            "2",
            "--bird",
            "/tmp/bird",
        ]
    )
    enabled_args = module._parse_args(
        [
            "--object",
            "foam-block",
            "--hardness",
            "soft",
            "--grip-level",
            "low",
            "--rep",
            "2",
            "--bird",
            "/tmp/bird",
            "--record-flir-visible",
        ]
    )
    visible = object()

    assert module._continuous_visible_source(default_args, visible) is None
    assert module._continuous_visible_source(enabled_args, visible) is visible


def test_record_trial_release_after_hold_sends_release_position_and_waits(
    monkeypatch: pytest.MonkeyPatch,
):
    module = importlib.import_module("record_ir_grip_trial")
    events: list[object] = []

    class FakeRobot:
        pass

    def fake_send_gripper(robot: FakeRobot, gripper_pos: float) -> None:
        events.append(("send", gripper_pos))

    def fake_sleep(seconds: float) -> None:
        events.append(("sleep", seconds))

    monkeypatch.setattr(module, "_send_gripper", fake_send_gripper)
    monkeypatch.setattr(module.time, "sleep", fake_sleep)

    module._release_after_hold(FakeRobot(), release_pos=90.0, release_settle_s=0.25)

    assert events == [("send", 90.0), ("sleep", 0.25)]
