"""Shared protocol contract for guided raw-repeatability recording."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from raw_repeatability import RawROI


@dataclass(frozen=True)
class ProtocolPhase:
    name: str
    duration_s: float
    instruction: str


def parse_raw_roi(value: str) -> RawROI:
    try:
        x, y, width, height = (int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise ValueError("raw ROI must be x,y,width,height") from exc
    roi = RawROI(x=x, y=y, width=width, height=height)
    roi.validate_for(frame_width=80, frame_height=60)
    return roi


def build_protocol(mode: str) -> list[ProtocolPhase]:
    if mode == "ffc":
        return [ProtocolPhase("stationary", 240.0, "Keep the target and camera completely still.")]
    if mode == "restart":
        return [
            ProtocolPhase("warmup", 30.0, "Keep the target and camera completely still."),
            ProtocolPhase("stable", 20.0, "Keep the target and camera completely still."),
            ProtocolPhase("observation", 40.0, "Keep the target and camera completely still."),
        ]
    if mode == "dynamic":
        phases: list[ProtocolPhase] = []
        for cycle in (1, 2):
            phases.extend(
                [
                    ProtocolPhase(f"baseline_{cycle:02d}", 15.0, "Keep the hot hand outside the frame."),
                    ProtocolPhase(f"hot_hand_{cycle:02d}", 15.0, "Hold the hot hand in the non-target region."),
                    ProtocolPhase(f"recovery_{cycle:02d}", 15.0, "Move the hot hand fully out of the frame."),
                ]
            )
        return phases
    raise ValueError("mode must be one of: ffc, restart, dynamic")


def _display_contract(
    *,
    display_mode: str,
    fixed_raw_low: int | None,
    fixed_raw_high: int | None,
) -> dict[str, object]:
    if display_mode == "dynamic":
        if fixed_raw_low is not None or fixed_raw_high is not None:
            raise ValueError("dynamic display mode cannot set fixed raw-count bounds")
        return {"mode": "dynamic", "raw_low": None, "raw_high": None}
    if display_mode == "fixed":
        if fixed_raw_low is None or fixed_raw_high is None or fixed_raw_low >= fixed_raw_high:
            raise ValueError("fixed display mode requires ordered raw-count bounds")
        return {"mode": "fixed", "raw_low": fixed_raw_low, "raw_high": fixed_raw_high}
    raise ValueError("display_mode must be dynamic or fixed")


def prepare_run(
    session_root: Path,
    *,
    run_id: str,
    mode: str,
    target_raw_roi: str,
    control_raw_roi: str,
    display_mode: str,
    fixed_raw_low: int | None = None,
    fixed_raw_high: int | None = None,
) -> Path:
    if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in run_id):
        raise ValueError("run_id may only contain letters, numbers, underscores, and hyphens")
    run_dir = session_root / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    target = parse_raw_roi(target_raw_roi)
    control = parse_raw_roi(control_raw_roi)
    display = _display_contract(
        display_mode=display_mode,
        fixed_raw_low=fixed_raw_low,
        fixed_raw_high=fixed_raw_high,
    )
    phases = build_protocol(mode)
    (run_dir / "raw").mkdir(parents=True)
    (run_dir / "rgb").mkdir()
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "mode": mode,
        "target_raw_roi": asdict(target),
        "control_raw_roi": asdict(control),
        "display": display,
        "phases": [asdict(phase) for phase in phases],
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with (run_dir / "events.csv").open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=("timestamp_ns", "event_type", "phase", "run_id")).writeheader()
    with (run_dir / "rgb_frames.csv").open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=("frame_index", "timestamp_ns", "file")).writeheader()
    return run_dir
