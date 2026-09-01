from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ir_force.classifier.ir_capture import (
    FrameSample,
    FrameSource,
    OpenCVCameraSource,
    capture_setup_snapshot,
)
from ir_force.classifier.ir_dataset import (
    HandPressureTrialSpec,
    TrialPaths,
    append_telemetry_row,
    create_hand_pressure_trial_paths,
    ensure_fresh_trial,
    write_hand_pressure_metadata,
)


DEFAULT_ROOT = "/home/zhuokai/hand-teleop/datasets/ir_hand_pressure_hysteresis"
DEFAULT_LEVELS = "zero,light,medium,hard,medium,light,zero"
LEVEL_VALUES = {
    "zero": 0.0,
    "light": 1.0,
    "medium": 2.0,
    "hard": 3.0,
}

STEP_TELEMETRY_FIELDS = (
    "frame",
    "t_capture",
    "t_thermal",
    "t_bird",
    "t_flir_visible",
    "surface",
    "contact",
    "phase",
    "cycle",
    "step_index",
    "step_name",
    "squeeze_level",
    "squeeze_value",
    "direction",
    "step_elapsed_s",
    "step_progress",
)


@dataclass(frozen=True)
class StepSpec:
    cycle: int
    step_index: int
    name: str
    value: float
    direction: str


class DummyBirdSource:
    def __init__(self) -> None:
        self.frame = np.zeros((1, 1, 3), dtype=np.uint8)

    def read(self) -> FrameSample:
        return FrameSample(t=time.perf_counter(), frame=self.frame)


def _parse_levels(value: str) -> tuple[str, ...]:
    levels = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if len(levels) < 2:
        raise argparse.ArgumentTypeError("levels must contain at least two comma-separated items")
    unknown = [level for level in levels if level not in LEVEL_VALUES]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown squeeze levels: {', '.join(unknown)}")
    return levels


def _direction(previous_value: float | None, value: float) -> str:
    if previous_value is None:
        return "baseline" if value == 0 else "start"
    if value > previous_value:
        return "up"
    if value < previous_value:
        return "down"
    return "hold"


def _build_step_plan(levels: tuple[str, ...], cycles: int) -> list[StepSpec]:
    if cycles <= 0:
        raise ValueError("cycles must be positive")
    steps: list[StepSpec] = []
    previous_value: float | None = None
    for cycle in range(1, cycles + 1):
        for step_index, level in enumerate(levels):
            value = LEVEL_VALUES[level]
            steps.append(
                StepSpec(
                    cycle=cycle,
                    step_index=step_index,
                    name=f"cycle{cycle:02d}_step{step_index:02d}_{level}",
                    value=value,
                    direction=_direction(previous_value, value),
                )
            )
            previous_value = value
    return steps


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record camera-only stepped hand-pressure IR data for lag/hysteresis/drift checks."
    )
    parser.add_argument("--surface", default="foam", help="contact surface, e.g. foam")
    parser.add_argument("--contact", default="whole hand", help="hand contact, e.g. whole hand")
    parser.add_argument("--rep", required=True, type=int)
    parser.add_argument("--thermal", default="/dev/video21")
    parser.add_argument("--bird", default="", help="optional bird/OAK/aperture camera path")
    parser.add_argument("--flir-visible", default="/dev/video20")
    parser.add_argument("--record-flir-visible", action="store_true")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--levels", type=_parse_levels, default=_parse_levels(DEFAULT_LEVELS))
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--hold-s", type=float, default=3.0)
    parser.add_argument("--pre-baseline-s", type=float, default=3.0)
    parser.add_argument("--thermal-roi", default="", help="optional ROI note as x,y,width,height")
    parser.add_argument("--notes", default="")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--yes", action="store_true", help="skip YES prompt")
    parser.add_argument("--no-beep", action="store_true")
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def _write_frame(path: Path, frame) -> None:
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"could not write frame {path}")


def _existing_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _next_frame_index(paths: TrialPaths) -> int:
    counts = [
        len(list(paths.thermal_dir.glob("frame_*.png"))),
        len(list(paths.bird_dir.glob("frame_*.png"))),
        len(list(paths.flir_visible_dir.glob("frame_*.png"))),
        _existing_rows(paths.telemetry_csv),
    ]
    return max(counts)


def _prepare_trial(args: argparse.Namespace) -> tuple[HandPressureTrialSpec, TrialPaths, list[StepSpec]]:
    spec = HandPressureTrialSpec(surface=args.surface, contact=args.contact, rep=args.rep)
    paths = create_hand_pressure_trial_paths(Path(args.root), spec)
    if not args.append:
        ensure_fresh_trial(paths)
    step_plan = _build_step_plan(args.levels, args.cycles)
    write_hand_pressure_metadata(
        paths,
        spec,
        {
            "thermal_path": args.thermal,
            "bird_path": args.bird,
            "record_bird": bool(args.bird),
            "flir_visible_path": args.flir_visible,
            "record_flir_visible": args.record_flir_visible,
            "recording_mode": "stepped_squeeze_hysteresis",
            "fps": args.fps,
            "levels": list(args.levels),
            "level_values": LEVEL_VALUES,
            "cycles": args.cycles,
            "hold_s": args.hold_s,
            "pre_baseline_s": args.pre_baseline_s,
            "thermal_roi": args.thermal_roi,
            "notes": args.notes,
            "analysis_targets": [
                "lag_after_level_change",
                "up_down_hysteresis_at_same_level",
                "zero_level_drift",
                "ir_change_after_aperture_plateau",
            ],
        },
    )
    return spec, paths, step_plan


def _capture_step_window(
    *,
    paths: TrialPaths,
    thermal: FrameSource,
    bird: FrameSource,
    flir_visible: FrameSource | None,
    spec: HandPressureTrialSpec,
    step: StepSpec,
    duration_s: float,
    fps: float,
    protocol_start: float,
) -> int:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if duration_s < 0:
        raise ValueError("duration_s must be non-negative")

    period = 1.0 / fps
    window_start = time.perf_counter()
    next_capture = window_start
    deadline = window_start + duration_s
    frame_index = _next_frame_index(paths)
    frames_written = 0

    while next_capture <= deadline + 1e-9:
        now = time.perf_counter()
        if now < next_capture:
            time.sleep(next_capture - now)

        sample_started = time.perf_counter()
        step_elapsed = sample_started - window_start
        thermal_sample = thermal.read()
        bird_sample = bird.read()
        visible_sample = flir_visible.read() if flir_visible is not None else None

        _write_frame(paths.thermal_dir / f"frame_{frame_index:06d}.png", thermal_sample.frame)
        _write_frame(paths.bird_dir / f"frame_{frame_index:06d}.png", bird_sample.frame)
        if visible_sample is not None:
            _write_frame(paths.flir_visible_dir / f"frame_{frame_index:06d}.png", visible_sample.frame)

        row = {
            "frame": frame_index,
            "t_capture": round(sample_started - protocol_start, 6),
            "t_thermal": round(thermal_sample.t, 6),
            "t_bird": round(bird_sample.t, 6),
            "t_flir_visible": round(visible_sample.t, 6) if visible_sample is not None else "",
            "surface": spec.surface,
            "contact": spec.contact,
            "phase": "stepped_squeeze",
            "cycle": step.cycle,
            "step_index": step.step_index,
            "step_name": step.name,
            "squeeze_level": step.name.rsplit("_", 1)[-1],
            "squeeze_value": step.value,
            "direction": step.direction,
            "step_elapsed_s": round(step_elapsed, 6),
            "step_progress": round(min(max(step_elapsed / max(duration_s, 1e-9), 0.0), 1.0), 6),
        }
        assert tuple(row.keys()) == STEP_TELEMETRY_FIELDS
        append_telemetry_row(paths.telemetry_csv, row)

        frames_written += 1
        frame_index += 1
        next_capture += period

    return frames_written


def _print_protocol(step_plan: list[StepSpec], hold_s: float) -> None:
    print("Protocol:")
    for step in step_plan:
        print(
            f"  cycle {step.cycle:02d} step {step.step_index:02d}: "
            f"{step.name.rsplit('_', 1)[-1]} ({step.direction}), hold {hold_s:g}s"
        )


def _record_trial(
    *,
    args: argparse.Namespace,
    spec: HandPressureTrialSpec,
    paths: TrialPaths,
    step_plan: list[StepSpec],
    thermal: OpenCVCameraSource,
    bird: FrameSource,
    visible: OpenCVCameraSource | None,
) -> None:
    print("capturing camera preflight")
    capture_setup_snapshot(paths, thermal=thermal, bird=bird, flir_visible=visible)
    continuous_visible = visible if args.record_flir_visible else None

    if args.pre_baseline_s > 0:
        baseline_step = StepSpec(cycle=0, step_index=0, name="pre_baseline_zero", value=0.0, direction="baseline")
        print(f"PRE-BASELINE: no squeeze for {args.pre_baseline_s:g}s")
        protocol_start = time.perf_counter()
        _capture_step_window(
            paths=paths,
            thermal=thermal,
            bird=bird,
            flir_visible=continuous_visible,
            spec=spec,
            step=baseline_step,
            duration_s=args.pre_baseline_s,
            fps=args.fps,
            protocol_start=protocol_start,
        )
    else:
        protocol_start = time.perf_counter()

    for step in step_plan:
        level = step.name.rsplit("_", 1)[-1]
        message = (
            f"CYCLE {step.cycle:02d}/{args.cycles:02d} "
            f"STEP {step.step_index:02d}: squeeze level = {level.upper()} ({step.direction})"
        )
        print(message)
        if not args.no_beep:
            print("\a", end="", flush=True)
        _capture_step_window(
            paths=paths,
            thermal=thermal,
            bird=bird,
            flir_visible=continuous_visible,
            spec=spec,
            step=step,
            duration_s=args.hold_s,
            fps=args.fps,
            protocol_start=protocol_start,
        )
    print(f"saved stepped hand-pressure trial to {paths.root}")


def main() -> None:
    args = _parse_args()
    spec, paths, step_plan = _prepare_trial(args)

    print("This script records camera-only stepped hand-pressure IR data.")
    print("It does not connect to or move the SO-101 robot.")
    print(f"Trial: {paths.trial_id}")
    print(f"Surface/contact: {spec.surface} / {spec.contact}")
    print(f"Pre-baseline: zero squeeze for {args.pre_baseline_s:g}s")
    _print_protocol(step_plan, args.hold_s)
    if not args.yes and input("Type YES to continue: ").strip() != "YES":
        raise SystemExit("aborted")

    thermal = OpenCVCameraSource(args.thermal)
    bird = OpenCVCameraSource(args.bird) if args.bird else DummyBirdSource()
    visible = OpenCVCameraSource(args.flir_visible) if args.flir_visible else None
    try:
        _record_trial(
            args=args,
            spec=spec,
            paths=paths,
            step_plan=step_plan,
            thermal=thermal,
            bird=bird,
            visible=visible,
        )
    finally:
        thermal.close()
        if isinstance(bird, OpenCVCameraSource):
            bird.close()
        if visible is not None:
            visible.close()


if __name__ == "__main__":
    main()
