"""Frozen protocol helpers for continuous single-finger thermal trials."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


PHASE_DURATION_S = 5.0
TRIAL_DURATION_S = 20.0
BIN_DURATION_S = 0.5
BINS_PER_PHASE = 10
MIN_VALID_FRAMES_PER_BIN = 2
MAX_THERMAL_GAP_S = 0.75
PHASES = ("A1", "X", "A2", "A3")
PRIMARY_BLOCKS = (
    ("null", "press"),
    ("press", "null"),
    ("press", "null"),
    ("null", "press"),
    ("null", "press"),
    ("press", "null"),
)
RESERVE_BLOCKS = (("null", "press"), ("press", "null"))


@dataclass(frozen=True)
class TrialSpec:
    block_index: int
    condition: str
    order_in_block: int
    reserve: bool


def scheduled_trial_specs() -> tuple[TrialSpec, ...]:
    specs = []
    for block_index, conditions in enumerate(
        PRIMARY_BLOCKS + RESERVE_BLOCKS
    ):
        for order_in_block, condition in enumerate(conditions):
            specs.append(
                TrialSpec(
                    block_index=block_index,
                    condition=condition,
                    order_in_block=order_in_block,
                    reserve=block_index >= len(PRIMARY_BLOCKS),
                )
            )
    return tuple(specs)


def _finite_nonnegative(value, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite and non-negative")
    value = float(value)
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return value


def phase_at(elapsed_s: float) -> str | None:
    elapsed_s = _finite_nonnegative(elapsed_s, label="elapsed_s")
    if elapsed_s >= TRIAL_DURATION_S:
        return None
    return PHASES[int(elapsed_s // PHASE_DURATION_S)]


def phase_elapsed(elapsed_s: float) -> float:
    elapsed_s = _finite_nonnegative(elapsed_s, label="elapsed_s")
    phase = phase_at(elapsed_s)
    if phase is None:
        raise ValueError("elapsed_s is outside the trial")
    return elapsed_s - PHASES.index(phase) * PHASE_DURATION_S


def global_elapsed(phase: str, elapsed_s: float) -> float:
    if phase not in PHASES:
        raise ValueError("unknown phase")
    elapsed_s = _finite_nonnegative(elapsed_s, label="elapsed_s")
    if elapsed_s >= PHASE_DURATION_S:
        raise ValueError("elapsed_s is outside the phase")
    return PHASES.index(phase) * PHASE_DURATION_S + elapsed_s


def persisted_flag_is_true(value) -> bool:
    return value is True or (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value == 1
    )


def trial_integrity(rows) -> dict:
    counts = {
        phase: [0] * BINS_PER_PHASE for phase in PHASES
    }
    invalid_phase = False
    invalid_phase_elapsed = False
    invalid_timestamp = False
    ffc_active = False
    artifact_failure = False
    timestamps = []

    for row in rows:
        if not isinstance(row, dict) or row.get("row_type") != "frame":
            continue
        ffc_active = ffc_active or bool(
            row.get("ffc_in_progress")
            or row.get("ffc_state") in ("imminent", "in_progress")
        )
        artifact_failure = artifact_failure or not persisted_flag_is_true(
            row.get("artifact_write_ok")
        )

        timestamp = row.get("thermal_host_s")
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not isfinite(float(timestamp))
        ):
            invalid_timestamp = True
        else:
            timestamps.append(float(timestamp))

        phase = row.get("phase")
        if phase not in PHASES:
            invalid_phase = True
            continue
        elapsed = row.get("phase_elapsed_s")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not isfinite(float(elapsed))
            or not 0.0 <= float(elapsed) < PHASE_DURATION_S
        ):
            invalid_phase_elapsed = True
            continue
        if persisted_flag_is_true(row.get("tracking_valid")):
            bin_index = min(
                int(float(elapsed) // BIN_DURATION_S),
                BINS_PER_PHASE - 1,
            )
            counts[phase][bin_index] += 1

    timestamp_gap = invalid_timestamp or any(
        current <= previous
        or current - previous > MAX_THERMAL_GAP_S
        for previous, current in zip(timestamps, timestamps[1:])
    )
    insufficient = [
        f"insufficient_tracking:{phase}:{bin_index}"
        for phase in PHASES
        for bin_index, count in enumerate(counts[phase])
        if count < MIN_VALID_FRAMES_PER_BIN
    ]
    reasons = []
    if ffc_active:
        reasons.append("ffc_active")
    if artifact_failure:
        reasons.append("required_artifact_write_failed")
    if timestamp_gap:
        reasons.append("thermal_timestamp_gap")
    if invalid_phase:
        reasons.append("invalid_phase")
    if invalid_phase_elapsed:
        reasons.append("invalid_phase_elapsed")
    reasons.extend(insufficient)
    return {
        "valid": not reasons,
        "reasons": reasons,
        "valid_frames_by_phase_bin": counts,
    }
