from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SOFollower

from ir_force.ir_capture import (
    CaptureSources,
    OpenCVCameraSource,
    capture_setup_snapshot,
    record_capture_window,
)
from ir_force.ir_dataset import (
    GRIP_LEVELS,
    TrialSpec,
    create_trial_paths,
    ensure_fresh_trial,
    write_metadata,
)
from ir_force.ir_robot import read_gripper_telemetry, slow_close_waypoints


ARM_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14110850-if00"
ARM_ID = "so101_follower_1"


class RobotTelemetrySource:
    def __init__(self, robot: SOFollower, goal_gripper_pos: float, t0: float):
        self.robot = robot
        self.goal_gripper_pos = goal_gripper_pos
        self.t0 = t0

    def read(self):
        return read_gripper_telemetry(
            self.robot,
            self.goal_gripper_pos,
            time.perf_counter() - self.t0,
        )


def _connect_robot(port: str) -> SOFollower:
    robot = SOFollower(
        SO101FollowerConfig(
            port=port,
            id=ARM_ID,
            use_degrees=False,
            cameras={},
            disable_torque_on_disconnect=True,
        )
    )
    robot.connect(calibrate=False)
    return robot


def _send_gripper(robot: SOFollower, gripper_pos: float) -> None:
    observation = robot.get_observation()
    action = {key: float(value) for key, value in observation.items() if key.endswith(".pos")}
    action["gripper.pos"] = float(gripper_pos)
    robot.send_action(action)


def _cleanup_robot(robot: SOFollower, open_pos: float) -> None:
    pending_exception = sys.exc_info()[1]
    try:
        _send_gripper(robot, open_pos)
    except BaseException as exc:
        print(f"warning: failed to open gripper before disconnect: {exc}")
    try:
        robot.disconnect()
    except BaseException as exc:
        if pending_exception is None:
            raise
        print(f"warning: failed to disconnect robot: {exc}")


def _release_after_hold(robot: SOFollower, release_pos: float, release_settle_s: float) -> None:
    _send_gripper(robot, release_pos)
    if release_settle_s > 0:
        time.sleep(release_settle_s)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", required=True, dest="object_name")
    parser.add_argument("--hardness", required=True, choices=["soft", "solid"])
    parser.add_argument("--grip-level", required=True, nargs="+", metavar="LEVEL")
    parser.add_argument("--rep", required=True, type=int)
    parser.add_argument("--warmed", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--thermal", default="/dev/video21")
    parser.add_argument("--bird", required=True)
    parser.add_argument("--flir-visible", default="/dev/video20")
    parser.add_argument("--record-flir-visible", action="store_true")
    parser.add_argument(
        "--grip-targets",
        default="/home/zhuokai/hand-teleop/datasets/ir_grip_force_viability/grip_targets.json",
    )
    parser.add_argument("--root", default="/home/zhuokai/hand-teleop/datasets/ir_grip_force_viability")
    parser.add_argument("--port", default=ARM_PORT)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--baseline-s", type=float, default=2.0)
    parser.add_argument("--hold-s", type=float, default=8.0)
    parser.add_argument("--open-pos", type=float, default=100.0)
    parser.add_argument("--release-pos", type=float, default=90.0)
    parser.add_argument("--release-settle-s", type=float, default=0.5)
    parser.add_argument("--close-steps", type=int, default=30)
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def _requested_grip_levels(args: argparse.Namespace) -> tuple[str, ...]:
    levels: list[str] = []
    for token in args.grip_level:
        levels.extend(level.strip() for level in token.split(",") if level.strip())
    if not levels:
        raise ValueError("at least one grip level is required")
    invalid = [level for level in levels if level not in GRIP_LEVELS]
    if invalid:
        raise ValueError(f"invalid grip level(s): {', '.join(invalid)}; expected one of {', '.join(GRIP_LEVELS)}")
    return tuple(levels)


def _continuous_visible_source(args: argparse.Namespace, visible: OpenCVCameraSource | None):
    if args.record_flir_visible:
        return visible
    return None


def _prepare_trial(args: argparse.Namespace, level: str, target: float, sequence_index: int, sequence_count: int):
    spec = TrialSpec(args.object_name, args.hardness, level, args.rep, warmed=args.warmed)
    paths = create_trial_paths(Path(args.root), spec)
    if not args.append:
        ensure_fresh_trial(paths)
    write_metadata(
        paths,
        spec,
        {
            "thermal_path": args.thermal,
            "bird_path": args.bird,
            "flir_visible_path": args.flir_visible,
            "record_flir_visible": args.record_flir_visible,
            "target_gripper_pos": target,
            "baseline_s": args.baseline_s,
            "hold_s": args.hold_s,
            "fps": args.fps,
            "open_pos": args.open_pos,
            "release_pos": args.release_pos,
            "release_settle_s": args.release_settle_s,
            "sequence_index": sequence_index,
            "sequence_count": sequence_count,
        },
    )
    return spec, paths


def _record_one_trial(
    *,
    args: argparse.Namespace,
    paths,
    target: float,
    thermal: OpenCVCameraSource,
    bird: OpenCVCameraSource,
    visible: OpenCVCameraSource | None,
    robot: SOFollower,
) -> None:
    print("capturing camera preflight")
    capture_setup_snapshot(paths, thermal=thermal, bird=bird, flir_visible=visible)
    continuous_visible = _continuous_visible_source(args, visible)
    _send_gripper(robot, args.open_pos)
    t0 = time.perf_counter()
    baseline_sources = CaptureSources(
        thermal=thermal,
        bird=bird,
        flir_visible=continuous_visible,
        telemetry=RobotTelemetrySource(robot, args.open_pos, t0),
    )
    print("capturing baseline")
    record_capture_window(paths, baseline_sources, duration_s=args.baseline_s, fps=args.fps)

    print("slow closing")
    for waypoint in slow_close_waypoints(args.open_pos, target, args.close_steps):
        _send_gripper(robot, waypoint)
        time.sleep(0.04)

    hold_sources = CaptureSources(
        thermal=thermal,
        bird=bird,
        flir_visible=continuous_visible,
        telemetry=RobotTelemetrySource(robot, target, t0),
    )
    print("capturing hold")
    record_capture_window(paths, hold_sources, duration_s=args.hold_s, fps=args.fps)

    _release_after_hold(robot, args.release_pos, args.release_settle_s)
    print(f"saved trial to {paths.root}")


def main() -> None:
    args = _parse_args()
    levels = _requested_grip_levels(args)
    targets = json.loads(Path(args.grip_targets).read_text())["selected_targets"]
    prepared = [
        _prepare_trial(args, level, float(targets[level]), index, len(levels))
        for index, level in enumerate(levels, start=1)
    ]

    print(f"This script will move the SO-101 gripper for {len(prepared)} trial(s).")
    for _spec, paths in prepared:
        print(f"Trial: {paths.trial_id}")
        print(f"Target gripper position: {float(targets[_spec.grip_level])}")
    if input("Type YES to continue: ").strip() != "YES":
        raise SystemExit("aborted")

    thermal = OpenCVCameraSource(args.thermal)
    bird = OpenCVCameraSource(args.bird)
    visible = OpenCVCameraSource(args.flir_visible) if args.flir_visible else None
    robot: SOFollower | None = None
    try:
        robot = _connect_robot(args.port)
        for spec, paths in prepared:
            _record_one_trial(
                args=args,
                paths=paths,
                target=float(targets[spec.grip_level]),
                thermal=thermal,
                bird=bird,
                visible=visible,
                robot=robot,
            )
    finally:
        if robot is not None:
            _cleanup_robot(robot, args.release_pos)
        thermal.close()
        bird.close()
        if visible is not None:
            visible.close()


if __name__ == "__main__":
    main()
