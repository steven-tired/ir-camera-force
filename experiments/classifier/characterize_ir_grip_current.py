from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SOFollower

from ir_force.classifier.ir_robot import (
    TelemetrySnapshot,
    choose_three_grip_targets,
    read_gripper_telemetry,
    serialize_telemetry_snapshot,
    slow_close_waypoints,
    summarize_target_current,
)


ARM_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14110850-if00"
ARM_ID = "so101_follower_1"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=ARM_PORT)
    parser.add_argument("--targets", default="85,70,55,40,25")
    parser.add_argument("--open-pos", type=float, default=100.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--hold-s", type=float, default=2.0)
    parser.add_argument("--min-current-gap", type=float, default=10.0)
    parser.add_argument(
        "--out",
        default="/home/zhuokai/hand-teleop/datasets/ir_grip_force_viability/grip_targets.json",
    )
    args = parser.parse_args()

    print("This script will move only the SO-101 gripper. Keep fingers and objects clear.")
    if input("Type YES to continue: ").strip() != "YES":
        raise SystemExit("aborted")

    robot: SOFollower | None = None
    summaries: list[dict[str, float]] = []
    telemetry_samples: list[dict[str, float | int | None]] = []
    try:
        robot = _connect_robot(args.port)
        _send_gripper(robot, args.open_pos)
        time.sleep(1.0)
        for target in [float(value) for value in args.targets.split(",")]:
            samples: list[TelemetrySnapshot] = []
            for waypoint in slow_close_waypoints(args.open_pos, target, args.steps):
                _send_gripper(robot, waypoint)
                time.sleep(0.04)
            start = time.perf_counter()
            while time.perf_counter() - start < args.hold_s:
                _send_gripper(robot, target)
                samples.append(read_gripper_telemetry(robot, target, time.perf_counter() - start))
                time.sleep(0.1)
            telemetry_samples.extend(
                serialize_telemetry_snapshot(sample, target=target, sample_index=sample_index)
                for sample_index, sample in enumerate(samples)
            )
            summary = {"target": target, **summarize_target_current(samples)}
            print(summary)
            summaries.append(summary)
            _send_gripper(robot, args.open_pos)
            time.sleep(0.6)
    finally:
        if robot is not None:
            _cleanup_robot(robot, args.open_pos)

    selected = choose_three_grip_targets(summaries, min_current_gap=args.min_current_gap)
    output = {
        "selected_targets": selected,
        "summaries": summaries,
        "telemetry_samples": telemetry_samples,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
