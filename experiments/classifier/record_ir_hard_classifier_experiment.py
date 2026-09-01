from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from ir_force.classifier.ir_capture import (
    FrameSample,
    FrameSource,
    OAKCameraSource,
    OAKFrameSample,
    OpenCVCameraSource,
)
from ir_force.classifier.ir_dataset import (
    TrialPaths,
    _create_trial_paths_for_id,
    append_telemetry_row,
    ensure_fresh_trial,
)
from ir_force.classifier.ir_features import (
    BaselineStats,
    ReferencePatch,
    ThermalROI,
    compute_baseline,
    extract_classifier_frame_features,
)
from ir_force.classifier.ir_hard_classifier import (
    HardClassifierTrialSpec,
    hard_classifier_trial_id,
)


DEFAULT_ROOT = "/home/zhuokai/hand-teleop/ir-camera-force/local/datasets/ir_hard_classifier"
DEFAULT_TARGET_PERCENTS = "0,25,50,75"
DEFAULT_THERMAL_ROI = "25,35,115,80"

STEP_TELEMETRY_FIELDS = (
    "frame",
    "t_capture",
    "t_thermal",
    "t_oak",
    "t_flir_visible",
    "session_id",
    "participant_id",
    "object_id",
    "block_type",
    "phase",
    "sequence_id",
    "step_index",
    "step_name",
    "target_squeeze_percent",
    "posture_condition",
    "step_elapsed_s",
    "step_progress",
)

EVENT_FIELDS = (
    "timestamp",
    "event_type",
    "block_type",
    "target_squeeze_percent",
    "posture_condition",
    "sequence_id",
    "step_index",
    "step_name",
)


@dataclass(frozen=True)
class SqueezePromptStep:
    sequence_id: int
    step_index: int
    name: str
    block_type: str
    target_squeeze_percent: float
    posture_condition: str
    hold_s: float
    release_s: float


def _parse_percent_list(value: str) -> tuple[float, ...]:
    try:
        percents = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if len(percents) < 2:
        raise argparse.ArgumentTypeError("squeeze percents must contain at least two comma-separated values")
    if any(percent < 0 for percent in percents):
        raise argparse.ArgumentTypeError("squeeze percents must be non-negative")
    return percents


def _parse_roi(value: str) -> ThermalROI:
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height")
    try:
        x, y, width, height = (int(part) for part in parts)
        return ThermalROI(x=x, y=y, width=width, height=height)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_reference_patches(value: str) -> tuple[ReferencePatch, ...]:
    if not value.strip():
        return ()
    patches: list[ReferencePatch] = []
    for raw_patch in value.split(";"):
        raw_patch = raw_patch.strip()
        if not raw_patch:
            continue
        if ":" not in raw_patch:
            raise argparse.ArgumentTypeError("reference patches must be name:x,y,width,height")
        name, roi_text = raw_patch.split(":", 1)
        patches.append(ReferencePatch(name=name.strip(), roi=_parse_roi(roi_text)))
    return tuple(patches)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record randomized FLIR/OAK data for offline squeeze-proxy analysis."
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--participant-id", required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--rep", required=True, type=int)
    parser.add_argument(
        "--block-type",
        default="fixed_posture",
        choices=("fixed_posture", "posture_control", "natural_posture"),
    )
    parser.add_argument("--posture-condition", default="neutral")
    parser.add_argument("--thermal", default="/dev/video21")
    parser.add_argument("--oak-fps", type=float, default=10.0)
    parser.add_argument("--flir-visible", default="/dev/video20")
    parser.add_argument("--record-flir-visible", action="store_true")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--squeeze-percents", type=_parse_percent_list, default=_parse_percent_list(DEFAULT_TARGET_PERCENTS))
    parser.add_argument("--sequences", type=int, default=8)
    parser.add_argument("--hold-s", type=float, default=3.0)
    parser.add_argument("--release-s", type=float, default=1.0)
    parser.add_argument("--pre-baseline-s", type=float, default=3.0)
    parser.add_argument("--thermal-roi", default=DEFAULT_THERMAL_ROI)
    parser.add_argument(
        "--reference-patches",
        default="",
        help="optional semicolon list like ref1:x,y,w,h;ref2:x,y,w,h for AGC/reference checks",
    )
    parser.add_argument("--seed", type=int, default=1)
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
        len(list((paths.root / "oak_rgb").glob("frame_*.png"))),
        len(list((paths.root / "oak_depth").glob("frame_*.png"))),
        len(list(paths.flir_visible_dir.glob("frame_*.png"))),
        _existing_rows(paths.telemetry_csv),
    ]
    return max(counts)


def _oak_rgb_dir(paths: TrialPaths) -> Path:
    directory = paths.root / "oak_rgb"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _oak_depth_dir(paths: TrialPaths) -> Path:
    directory = paths.root / "oak_depth"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _capture_setup_snapshot(
    paths: TrialPaths,
    *,
    thermal: FrameSource,
    oak: FrameSource,
    flir_visible: FrameSource | None,
) -> None:
    thermal_sample = thermal.read()
    oak_sample = oak.read()
    if not isinstance(oak_sample, OAKFrameSample):
        raise TypeError("OAK capture source must return OAKFrameSample")
    _write_frame(paths.preflight_dir / "thermal.png", thermal_sample.frame)
    _write_frame(paths.preflight_dir / "oak_rgb.png", oak_sample.frame)
    _write_frame(paths.preflight_dir / "oak_depth.png", oak_sample.depth)
    if flir_visible is not None:
        _write_frame(paths.preflight_dir / "flir_visible.png", flir_visible.read().frame)


def _write_empty_csv(path: Path, fieldnames: tuple[str, ...]) -> None:
    if path.exists():
        return
    with path.open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()


def _build_squeeze_prompt_steps(
    *,
    squeeze_percents: tuple[float, ...],
    sequences: int,
    hold_s: float,
    release_s: float,
    seed: int,
    block_type: str,
    posture_condition: str,
) -> list[SqueezePromptStep]:
    if sequences < 1:
        raise ValueError("sequences must be positive")
    if hold_s <= 0:
        raise ValueError("hold_s must be positive")
    if release_s < 0:
        raise ValueError("release_s must be non-negative")

    rng = random.Random(seed)
    steps: list[SqueezePromptStep] = []
    for sequence_id in range(1, sequences + 1):
        ordered = list(squeeze_percents)
        rng.shuffle(ordered)
        for step_index, target_squeeze_percent in enumerate(ordered):
            steps.append(
                SqueezePromptStep(
                    sequence_id=sequence_id,
                    step_index=step_index,
                    name=f"sequence{sequence_id:02d}_step{step_index:02d}_{target_squeeze_percent:g}pct",
                    block_type=block_type,
                    target_squeeze_percent=target_squeeze_percent,
                    posture_condition=posture_condition,
                    hold_s=hold_s,
                    release_s=release_s,
                )
            )
    return steps


def _oak_squeeze_trial_id(spec: HardClassifierTrialSpec) -> str:
    return hard_classifier_trial_id(spec).replace("hard-classifier", "oak-squeeze", 1)


def _event_rows(steps: list[SqueezePromptStep], timestamp: float | str = "") -> list[dict[str, object]]:
    return [
        {
            "timestamp": timestamp,
            "event_type": "target_start",
            "block_type": step.block_type,
            "target_squeeze_percent": step.target_squeeze_percent,
            "posture_condition": step.posture_condition,
            "sequence_id": step.sequence_id,
            "step_index": step.step_index,
            "step_name": step.name,
        }
        for step in steps
    ]


def _append_event(paths: TrialPaths, row: dict[str, object]) -> None:
    assert tuple(row.keys()) == EVENT_FIELDS
    append_telemetry_row(paths.root / "events.csv", row)


def _prepare_trial(args: argparse.Namespace) -> tuple[HardClassifierTrialSpec, TrialPaths, list[SqueezePromptStep]]:
    spec = HardClassifierTrialSpec(
        session_id=args.session_id,
        block_type=args.block_type,
        rep=args.rep,
        object_id=args.object_id,
        participant_id=args.participant_id,
    )
    paths = _create_trial_paths_for_id(Path(args.root), _oak_squeeze_trial_id(spec))
    if not args.append:
        ensure_fresh_trial(paths)
    steps = _build_squeeze_prompt_steps(
        squeeze_percents=args.squeeze_percents,
        sequences=args.sequences,
        hold_s=args.hold_s,
        release_s=args.release_s,
        seed=args.seed,
        block_type=args.block_type,
        posture_condition=args.posture_condition,
    )

    metadata = {
        "trial_id": paths.trial_id,
        "experiment_kind": "ir_oak_squeeze_proxy",
        "recording_mode": "randomized_subjective_squeeze_targets",
        "primary_task": "oak_squeeze_proxy_vs_ir",
        "primary_comparison": "oak_proxy_plus_ir_vs_oak_proxy_only",
        "label_source": "oak_visual_proxy_pending",
        "objective_force_measurement": False,
        **asdict(spec),
        "target_squeeze_percents": list(args.squeeze_percents),
        "sequences": args.sequences,
        "hold_s": args.hold_s,
        "release_s": args.release_s,
        "pre_baseline_s": args.pre_baseline_s,
        "thermal_path": args.thermal,
        "oak_rgb_size": [640, 480],
        "oak_depth_unit": "millimeter",
        "oak_fps": args.oak_fps,
        "flir_visible_path": args.flir_visible,
        "record_flir_visible": args.record_flir_visible,
        "thermal_roi": args.thermal_roi,
        "reference_patches": args.reference_patches,
        "posture_condition": args.posture_condition,
        "fps": args.fps,
        "seed": args.seed,
        "notes": args.notes,
        "analysis_tasks": [
            "contact_vs_no_contact",
            "oak_hand_foam_compression_proxy",
            "oak_proxy_plus_ir_vs_oak_proxy_only",
            "ir_persistence_after_release",
            "agc_jump_and_frozen_frame_gating",
        ],
    }
    paths.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    _write_empty_csv(
        paths.root / "gripper.csv",
        ("timestamp", "gripper_position", "gripper_aperture", "gripper_current", "gripper_load", "gripper_velocity"),
    )
    return spec, paths, steps


def _telemetry_row(
    *,
    frame_index: int,
    protocol_elapsed: float,
    thermal_sample: FrameSample,
    oak_sample: OAKFrameSample,
    visible_sample: FrameSample | None,
    spec: HardClassifierTrialSpec,
    step: SqueezePromptStep,
    phase: str,
    step_elapsed: float,
    duration_s: float,
) -> dict[str, object]:
    row = {
        "frame": frame_index,
        "t_capture": round(protocol_elapsed, 6),
        "t_thermal": round(thermal_sample.t, 6),
        "t_oak": round(oak_sample.t, 6),
        "t_flir_visible": round(visible_sample.t, 6) if visible_sample is not None else "",
        "session_id": spec.session_id,
        "participant_id": spec.participant_id,
        "object_id": spec.object_id,
        "block_type": spec.block_type,
        "phase": phase,
        "sequence_id": step.sequence_id,
        "step_index": step.step_index,
        "step_name": step.name,
        "target_squeeze_percent": step.target_squeeze_percent,
        "posture_condition": step.posture_condition,
        "step_elapsed_s": round(step_elapsed, 6),
        "step_progress": round(min(max(step_elapsed / max(duration_s, 1e-9), 0.0), 1.0), 6),
    }
    assert tuple(row.keys()) == STEP_TELEMETRY_FIELDS
    return row


def _capture_classifier_window(
    *,
    paths: TrialPaths,
    thermal: FrameSource,
    oak: FrameSource,
    flir_visible: FrameSource | None,
    spec: HardClassifierTrialSpec,
    step: SqueezePromptStep,
    phase: str,
    duration_s: float,
    fps: float,
    protocol_start: float,
    baseline: BaselineStats | None,
    reference_patches: tuple[ReferencePatch, ...],
    previous_frame_p98: float | None,
    previous_raw_frame: np.ndarray | None,
) -> tuple[int, list[np.ndarray], float | None, np.ndarray | None]:
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
    thermal_frames: list[np.ndarray] = []
    last_p98 = previous_frame_p98
    last_raw_frame = previous_raw_frame

    while next_capture <= deadline + 1e-9:
        now = time.perf_counter()
        if now < next_capture:
            time.sleep(next_capture - now)

        sample_started = time.perf_counter()
        step_elapsed = sample_started - window_start
        protocol_elapsed = sample_started - protocol_start
        thermal_sample = thermal.read()
        oak_sample = oak.read()
        if not isinstance(oak_sample, OAKFrameSample):
            raise TypeError("OAK capture source must return OAKFrameSample")
        visible_sample = flir_visible.read() if flir_visible is not None else None

        _write_frame(paths.thermal_dir / f"frame_{frame_index:06d}.png", thermal_sample.frame)
        _write_frame(_oak_rgb_dir(paths) / f"frame_{frame_index:06d}.png", oak_sample.frame)
        _write_frame(_oak_depth_dir(paths) / f"frame_{frame_index:06d}.png", oak_sample.depth)
        if visible_sample is not None:
            _write_frame(paths.flir_visible_dir / f"frame_{frame_index:06d}.png", visible_sample.frame)

        append_telemetry_row(
            paths.telemetry_csv,
            _telemetry_row(
                frame_index=frame_index,
                protocol_elapsed=protocol_elapsed,
                thermal_sample=thermal_sample,
                oak_sample=oak_sample,
                visible_sample=visible_sample,
                spec=spec,
                step=step,
                phase=phase,
                step_elapsed=step_elapsed,
                duration_s=duration_s,
            ),
        )
        thermal_frames.append(thermal_sample.frame.copy())

        if baseline is not None:
            feature_row = extract_classifier_frame_features(
                thermal_sample.frame,
                baseline,
                frame_id=frame_index,
                timestamp=protocol_elapsed,
                reference_patches=reference_patches,
                previous_frame_p98=last_p98,
            )
            if last_raw_frame is not None and last_raw_frame.shape == thermal_sample.frame.shape:
                feature_row["frozen_frame_flag"] = bool(np.array_equal(last_raw_frame, thermal_sample.frame))
            feature_row = {
                "phase": phase,
                "sequence_id": step.sequence_id,
                "step_index": step.step_index,
                "target_squeeze_percent": step.target_squeeze_percent,
                **feature_row,
            }
            append_telemetry_row(paths.root / "frame_features.csv", feature_row)
            last_p98 = float(feature_row["frame_p98"])
        last_raw_frame = thermal_sample.frame.copy()

        frames_written += 1
        frame_index += 1
        next_capture += period

    return frames_written, thermal_frames, last_p98, last_raw_frame


def _print_protocol(steps: list[SqueezePromptStep]) -> None:
    print("Protocol:")
    for step in steps:
        print(
            f"  seq {step.sequence_id:02d} step {step.step_index:02d}: "
            f"{step.target_squeeze_percent:g}% self-rated squeeze, "
            f"hold {step.hold_s:g}s, release {step.release_s:g}s"
        )


def _record_trial(
    *,
    args: argparse.Namespace,
    spec: HardClassifierTrialSpec,
    paths: TrialPaths,
    steps: list[SqueezePromptStep],
    thermal: OpenCVCameraSource,
    oak: OAKCameraSource,
    visible: OpenCVCameraSource | None,
) -> None:
    print("capturing camera preflight")
    _capture_setup_snapshot(paths, thermal=thermal, oak=oak, flir_visible=visible)
    continuous_visible = visible if args.record_flir_visible else None
    roi = _parse_roi(args.thermal_roi)
    reference_patches = _parse_reference_patches(args.reference_patches)
    protocol_start = time.perf_counter()
    previous_p98: float | None = None
    previous_raw_frame: np.ndarray | None = None

    baseline_step = SqueezePromptStep(
        sequence_id=0,
        step_index=0,
        name="pre_baseline_zero",
        block_type=spec.block_type,
        target_squeeze_percent=0.0,
        posture_condition=args.posture_condition,
        hold_s=args.pre_baseline_s,
        release_s=0.0,
    )
    print(f"PRE-BASELINE: no squeeze for {args.pre_baseline_s:g}s")
    _append_event(paths, {**_event_rows([baseline_step], timestamp=0.0)[0], "event_type": "baseline_start"})
    _baseline_count, baseline_frames, previous_p98, previous_raw_frame = _capture_classifier_window(
        paths=paths,
        thermal=thermal,
        oak=oak,
        flir_visible=continuous_visible,
        spec=spec,
        step=baseline_step,
        phase="pre_baseline",
        duration_s=args.pre_baseline_s,
        fps=args.fps,
        protocol_start=protocol_start,
        baseline=None,
        reference_patches=reference_patches,
        previous_frame_p98=previous_p98,
        previous_raw_frame=previous_raw_frame,
    )
    baseline = compute_baseline(baseline_frames, roi=roi)

    for step in steps:
        if not args.no_beep:
            print("\a", end="", flush=True)
        print(
            f"SEQ {step.sequence_id:02d} STEP {step.step_index:02d}: "
            f"target {step.target_squeeze_percent:g}% self-rated squeeze"
        )
        _append_event(paths, _event_rows([step], timestamp=round(time.perf_counter() - protocol_start, 6))[0])
        _frames, _samples, previous_p98, previous_raw_frame = _capture_classifier_window(
            paths=paths,
            thermal=thermal,
            oak=oak,
            flir_visible=continuous_visible,
            spec=spec,
            step=step,
            phase="target_hold",
            duration_s=step.hold_s,
            fps=args.fps,
            protocol_start=protocol_start,
            baseline=baseline,
            reference_patches=reference_patches,
            previous_frame_p98=previous_p98,
            previous_raw_frame=previous_raw_frame,
        )
        if step.release_s > 0:
            release_step = SqueezePromptStep(
                sequence_id=step.sequence_id,
                step_index=step.step_index,
                name=f"{step.name}_release",
                block_type=step.block_type,
                target_squeeze_percent=0.0,
                posture_condition=step.posture_condition,
                hold_s=step.release_s,
                release_s=0.0,
            )
            _append_event(
                paths,
                {
                    **_event_rows([release_step], timestamp=round(time.perf_counter() - protocol_start, 6))[0],
                    "event_type": "release_start",
                },
            )
            _frames, _samples, previous_p98, previous_raw_frame = _capture_classifier_window(
                paths=paths,
                thermal=thermal,
                oak=oak,
                flir_visible=continuous_visible,
                spec=spec,
                step=release_step,
                phase="release",
                duration_s=step.release_s,
                fps=args.fps,
                protocol_start=protocol_start,
                baseline=baseline,
                reference_patches=reference_patches,
                previous_frame_p98=previous_p98,
                previous_raw_frame=previous_raw_frame,
            )
    print(f"saved OAK squeeze-proxy trial to {paths.root}")


def main() -> None:
    args = _parse_args()
    spec, paths, steps = _prepare_trial(args)

    print("This script records randomized IR/OAK squeeze-proxy data.")
    print("It records subjective squeeze prompts only; no force labels are assigned.")
    print(f"Trial: {paths.trial_id}")
    print(f"Object/participant/session: {spec.object_id} / {spec.participant_id} / {spec.session_id}")
    _print_protocol(steps)
    if not args.yes and input("Type YES to continue: ").strip() != "YES":
        raise SystemExit("aborted")

    thermal = OpenCVCameraSource(args.thermal)
    oak = OAKCameraSource(fps=args.oak_fps)
    visible = OpenCVCameraSource(args.flir_visible) if args.flir_visible else None
    try:
        _record_trial(
            args=args,
            spec=spec,
            paths=paths,
            steps=steps,
            thermal=thermal,
            oak=oak,
            visible=visible,
        )
    finally:
        thermal.close()
        oak.close()
        if visible is not None:
            visible.close()


if __name__ == "__main__":
    main()
