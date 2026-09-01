from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


GRIPPER = "gripper"


@dataclass(frozen=True)
class TelemetrySnapshot:
    t: float
    gripper_pos: float
    goal_gripper_pos: float
    present_current: int | None
    present_load: int | None
    present_temperature: int | None


def slow_close_waypoints(start: float, target: float, steps: int) -> list[float]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    delta = (target - start) / steps
    return [round(start + delta * (i + 1), 6) for i in range(steps)]


def _numeric(values: list[int | float | None]) -> list[float]:
    return [float(value) for value in values if value is not None]


def summarize_target_current(samples: list[TelemetrySnapshot]) -> dict[str, float]:
    currents = _numeric([sample.present_current for sample in samples])
    loads = _numeric([sample.present_load for sample in samples])
    temperatures = _numeric([sample.present_temperature for sample in samples])
    return {
        "mean_current": mean(currents) if currents else 0.0,
        "max_current": max(currents) if currents else 0.0,
        "mean_load": mean(loads) if loads else 0.0,
        "max_temperature": max(temperatures) if temperatures else 0.0,
    }


def serialize_telemetry_snapshot(
    sample: TelemetrySnapshot,
    *,
    target: float,
    sample_index: int,
) -> dict[str, float | int | None]:
    return {
        "target": float(target),
        "sample_index": int(sample_index),
        "t": float(sample.t),
        "gripper_pos": float(sample.gripper_pos),
        "goal_gripper_pos": float(sample.goal_gripper_pos),
        "present_current": sample.present_current,
        "present_load": sample.present_load,
        "present_temperature": sample.present_temperature,
    }


def choose_three_grip_targets(records: list[dict[str, float]], min_current_gap: float) -> dict[str, float]:
    ordered = sorted(records, key=lambda record: record["mean_current"])
    selected: tuple[dict[str, float], dict[str, float], dict[str, float]] | None = None
    selected_key: tuple[float, float, float] | None = None
    for low_index, low in enumerate(ordered):
        for med_index, med in enumerate(ordered[low_index + 1 :], start=low_index + 1):
            if med["mean_current"] - low["mean_current"] < min_current_gap:
                continue
            for high in ordered[med_index + 1 :]:
                if high["mean_current"] - med["mean_current"] >= min_current_gap:
                    candidate = (low, med, high)
                    candidate_key = (
                        candidate[0]["mean_current"],
                        candidate[1]["mean_current"],
                        candidate[2]["mean_current"],
                    )
                    if selected_key is None or candidate_key > selected_key:
                        selected = candidate
                        selected_key = candidate_key

    if selected is None:
        raise ValueError("could not find three separated grip targets")

    return {
        "low": selected[0]["target"],
        "med": selected[1]["target"],
        "high": selected[2]["target"],
    }


def _read_reg(robot, reg: str, motor: str) -> int | None:
    try:
        return int(robot.bus.read(reg, motor, normalize=False, num_retry=5))
    except Exception:
        return None


def read_gripper_telemetry(robot, goal_gripper_pos: float, t: float) -> TelemetrySnapshot:
    observation = robot.get_observation()
    return TelemetrySnapshot(
        t=t,
        gripper_pos=float(observation["gripper.pos"]),
        goal_gripper_pos=float(goal_gripper_pos),
        present_current=_read_reg(robot, "Present_Current", GRIPPER),
        present_load=_read_reg(robot, "Present_Load", GRIPPER),
        present_temperature=_read_reg(robot, "Present_Temperature", GRIPPER),
    )
