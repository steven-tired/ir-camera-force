from __future__ import annotations

import random
import re
from dataclasses import dataclass


VALID_BLOCK_TYPES = frozenset({"fixed_posture", "posture_control", "natural_posture"})


@dataclass(frozen=True)
class ForceLabelThresholds:
    hard_fraction: float = 0.70
    not_hard_fraction: float = 0.50

    def __post_init__(self) -> None:
        if not 0.0 <= self.not_hard_fraction < self.hard_fraction <= 1.0:
            raise ValueError("not_hard_fraction must be >= 0 and < hard_fraction <= 1")

    def metadata(self) -> dict[str, object]:
        return {
            "hard_fraction": self.hard_fraction,
            "not_hard_fraction": self.not_hard_fraction,
            "ambiguous_fraction_range": [self.not_hard_fraction, self.hard_fraction],
        }


@dataclass(frozen=True)
class HardClassifierTrialSpec:
    session_id: str
    block_type: str
    rep: int
    object_id: str
    participant_id: str

    def __post_init__(self) -> None:
        session_id = self.session_id.strip()
        object_id = self.object_id.strip()
        participant_id = self.participant_id.strip()
        if not session_id:
            raise ValueError("session_id must be non-empty")
        if self.block_type not in VALID_BLOCK_TYPES:
            raise ValueError(f"block_type must be one of {sorted(VALID_BLOCK_TYPES)}")
        if not 1 <= self.rep <= 99:
            raise ValueError("rep must be in the range 1..99")
        if not object_id:
            raise ValueError("object_id must be non-empty")
        if not participant_id:
            raise ValueError("participant_id must be non-empty")
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "participant_id", participant_id)


@dataclass(frozen=True)
class ForceTargetStep:
    sequence_id: int
    step_index: int
    name: str
    block_type: str
    target_force_percent: float
    target_force_newton: float
    posture_condition: str
    hold_s: float
    release_s: float


def _slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def hard_classifier_trial_id(spec: HardClassifierTrialSpec) -> str:
    return (
        f"hard-classifier_{_slug(spec.session_id)}_{_slug(spec.block_type)}_"
        f"{_slug(spec.object_id)}_{_slug(spec.participant_id)}_rep{spec.rep:02d}"
    )


def force_label(
    force_n: float,
    *,
    fmax_n: float,
    thresholds: ForceLabelThresholds | None = None,
) -> str:
    if fmax_n <= 0:
        raise ValueError("fmax_n must be positive")
    thresholds = thresholds or ForceLabelThresholds()
    fraction = force_n / fmax_n
    if fraction >= thresholds.hard_fraction:
        return "hard"
    if fraction <= thresholds.not_hard_fraction:
        return "not_hard"
    return "ambiguous"


def target_force_newton(target_force_percent: float, *, fmax_n: float) -> float:
    if fmax_n <= 0:
        raise ValueError("fmax_n must be positive")
    if target_force_percent < 0:
        raise ValueError("target_force_percent must be non-negative")
    return target_force_percent / 100.0 * fmax_n


def build_force_target_steps(
    *,
    target_percents: tuple[float, ...],
    sequences: int,
    hold_s: float,
    release_s: float,
    fmax_n: float,
    seed: int | None,
    block_type: str,
    posture_condition: str,
) -> list[ForceTargetStep]:
    if block_type not in VALID_BLOCK_TYPES:
        raise ValueError(f"block_type must be one of {sorted(VALID_BLOCK_TYPES)}")
    if sequences <= 0:
        raise ValueError("sequences must be positive")
    if len(target_percents) < 2:
        raise ValueError("target_percents must contain at least two values")
    if hold_s <= 0:
        raise ValueError("hold_s must be positive")
    if release_s < 0:
        raise ValueError("release_s must be non-negative")

    rng = random.Random(seed)
    steps: list[ForceTargetStep] = []
    for sequence_id in range(1, sequences + 1):
        sequence = [float(value) for value in target_percents]
        rng.shuffle(sequence)
        if sequence == sorted(sequence) and len(sequence) > 2:
            sequence = sequence[1:] + sequence[:1]
        for step_index, target_percent in enumerate(sequence):
            steps.append(
                ForceTargetStep(
                    sequence_id=sequence_id,
                    step_index=step_index,
                    name=f"sequence{sequence_id:02d}_step{step_index:02d}_{target_percent:g}pct",
                    block_type=block_type,
                    target_force_percent=target_percent,
                    target_force_newton=target_force_newton(target_percent, fmax_n=fmax_n),
                    posture_condition=posture_condition,
                    hold_s=hold_s,
                    release_s=release_s,
                )
            )
    return steps
