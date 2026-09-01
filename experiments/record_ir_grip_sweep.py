from __future__ import annotations

import argparse
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
    SWEEP_GRIP_LEVEL,
    TrialSpec,
    create_trial_paths,
    ensure_fresh_trial,
    write_metadata,
)
from lerobot_teleoperator_so101_webcam.gripper_hardware import read_gripper_telemetry


ARM_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14110850-if00"
ARM_ID = "so101_follower_1"


class SweepRobotTelemetrySource:
    def __init__(self, robot: SOFollower, goal_gripper_pos: float, t0: float):
        self.robot = robot
        self.goal_gripper_pos = float(goal_gripper_pos)
        self.t0 = t0

    def set_goal(self, goal_gripper_pos: float) -> None:
        self.goal_gripper_pos = float(goal_gripper_pos)

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


def _sweep_goal(*, open_pos: float, target_pos: float, elapsed_s: float, duration_s: float) -> float:
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    progress = min(max(elapsed_s / duration_s, 0.0), 1.0)
    return round(open_pos + (target_pos - open_pos) * progress, 6)


def _continuous_visible_source(args: argparse.Namespace, visible: OpenCVCameraSource | None):
    if args.record_flir_visible:
        return visible
    return None


def _sweep_start_pos(args: argparse.Namespace) -> float:
    if args.sweep_start_pos is None:
        return float(args.open_pos)
    return float(args.sweep_start_pos)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", required=True, dest="object_name")
    parser.add_argument("--hardness", required=True, choices=["soft", "solid"])
    parser.add_argument("--rep", required=True, type=int)
    parser.add_argument("--warmed", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--thermal", default="/dev/video21")
    parser.add_argument("--bird", required=True)
    parser.add_argument("--flir-visible", default="/dev/video20")
    parser.add_argument("--record-flir-visible", action="store_true")
    parser.add_argument("--root", default="/home/zhuokai/hand-teleop/ir-camera-force/local/datasets/ir_grip_force_viability")
    parser.add_argument("--port", default=ARM_PORT)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--baseline-s", type=float, default=2.0)
    parser.add_argument("--sweep-s", type=float, default=12.0)
    parser.add_argument("--hold-s", type=float, default=3.0)
    parser.add_argument("--open-pos", type=float, default=100.0)
    parser.add_argument("--sweep-start-pos", type=float)
    parser.add_argument("--pre-baseline-settle-s", type=float, default=0.0)
    parser.add_argument("--target-pos", type=float, default=10.0)
    parser.add_argument("--release-pos", type=float, default=90.0)
    parser.add_argument("--release-settle-s", type=float, default=0.5)
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def _prepare_trial(args: argparse.Namespace):
    spec = TrialSpec(args.object_name, args.hardness, SWEEP_GRIP_LEVEL, args.rep, warmed=args.warmed)
    paths = create_trial_paths(Path(args.root), spec)
    sweep_start_pos = _sweep_start_pos(args)
    if not args.append:
        ensure_fresh_trial(paths)
    write_metadata(
        paths,
        spec,
        {
            "recording_mode": "sweep",
            "thermal_path": args.thermal,
            "bird_path": args.bird,
            "flir_visible_path": args.flir_visible,
            "record_flir_visible": args.record_flir_visible,
            "baseline_s": args.baseline_s,
            "sweep_s": args.sweep_s,
            "hold_s": args.hold_s,
            "fps": args.fps,
            "open_pos": args.open_pos,
            "sweep_start_pos": sweep_start_pos,
            "pre_baseline_settle_s": args.pre_baseline_settle_s,
            "target_gripper_pos": args.target_pos,
            "release_pos": args.release_pos,
            "release_settle_s": args.release_settle_s,
        },
    )
    return spec, paths


def _record_sweep_trial(
    *,
    args: argparse.Namespace,
    paths,
    thermal: OpenCVCameraSource,
    bird: OpenCVCameraSource,
    visible: OpenCVCameraSource | None,
    robot: SOFollower,
) -> None:
    print("capturing camera preflight")
    capture_setup_snapshot(paths, thermal=thermal, bird=bird, flir_visible=visible)
    continuous_visible = _continuous_visible_source(args, visible)
    sweep_start_pos = _sweep_start_pos(args)

    _send_gripper(robot, args.open_pos)
    if sweep_start_pos != args.open_pos:
        _send_gripper(robot, sweep_start_pos)
    if args.pre_baseline_settle_s > 0:
        time.sleep(args.pre_baseline_settle_s)

    t0 = time.perf_counter()
    telemetry = SweepRobotTelemetrySource(robot, sweep_start_pos, t0)

    sources = CaptureSources(
        thermal=thermal,
        bird=bird,
        flir_visible=continuous_visible,
        telemetry=telemetry,
    )

    print("capturing baseline")
    telemetry.set_goal(sweep_start_pos)
    record_capture_window(paths, sources, duration_s=args.baseline_s, fps=args.fps)

    print("sweeping closed")

    def update_sweep_goal(elapsed_s: float) -> None:
        goal = _sweep_goal(
            open_pos=sweep_start_pos,
            target_pos=args.target_pos,
            elapsed_s=elapsed_s,
            duration_s=args.sweep_s,
        )
        telemetry.set_goal(goal)
        _send_gripper(robot, goal)

    record_capture_window(
        paths,
        sources,
        duration_s=args.sweep_s,
        fps=args.fps,
        before_sample=update_sweep_goal,
    )

    if args.hold_s > 0:
        print("capturing terminal hold")
        telemetry.set_goal(args.target_pos)
        _send_gripper(robot, args.target_pos)
        record_capture_window(paths, sources, duration_s=args.hold_s, fps=args.fps)

    _release_after_hold(robot, args.release_pos, args.release_settle_s)
    print(f"saved sweep trial to {paths.root}")


def main() -> None:
    args = _parse_args()
    _spec, paths = _prepare_trial(args)

    print("This script will move the SO-101 gripper through one continuous sweep.")
    print(f"Trial: {paths.trial_id}")
    print(f"Open position: {args.open_pos}")
    print(f"Sweep start position: {_sweep_start_pos(args)}")
    print(f"Target position: {args.target_pos}")
    print(f"Sweep duration: {args.sweep_s}s")
    if input("Type YES to continue: ").strip() != "YES":
        raise SystemExit("aborted")

    thermal = OpenCVCameraSource(args.thermal)
    bird = OpenCVCameraSource(args.bird)
    visible = OpenCVCameraSource(args.flir_visible) if args.flir_visible else None
    robot: SOFollower | None = None
    try:
        robot = _connect_robot(args.port)
        _record_sweep_trial(
            args=args,
            paths=paths,
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
