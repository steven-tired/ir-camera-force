#!/usr/bin/env python3
"""Thermal-only A-B-A press sanity check for the single-finger signal.

This is NOT a preregistered experiment and produces NO formal verdict. It
records repeated LIGHT / HARD / LIGHT rounds on a fixed contact spot so the
operator can see whether a press effect exists, how large it is, and whether it
returns when the load is removed.

Why this shape:

- `single_finger_hold_check_01` measured the wrong place. The frozen v2 ROI
  rule thresholds the whole frame and takes the leftmost 3% of the blob, which
  put both ROIs on the back of the fist and reported a 0.25 count effect. With
  the fingertip located correctly the same frames give roughly -50 counts, in
  the predicted direction. ROIs are therefore operator-clicked and tracked by
  template matching, with no global segmentation.
- That session had no return phase, so a 30 s press could not be told apart
  from a monotone drift: LIGHT already fell at -1.14 counts/s and HARD at
  -2.23 counts/s. The A-B-A round is what separates the two.
- Its lift phase was unusable, since the ROI leaves the finger. Contact is now
  held throughout a round, and rounds are separated by an operator-paced rest
  so a taped contact patch can cool between presses.

Robot-free: no controller, teleop, recorder, gripper, or pressure-apply import,
and no D435i, MediaPipe, projection, or Stage 0 contract.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np


_CHECKOUT_ROOT = Path(__file__).resolve().parents[1]
if str(_CHECKOUT_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHECKOUT_ROOT))

from live_lepton_hand_shadow import _run_manual_ffc  # noqa: E402
from ir_force.ir_capture import (  # noqa: E402
    FrameUnavailableError,
    LeptonUDPSource,
)
from ir_force.single_finger_click_roi import (  # noqa: E402
    TemplateTracker,
    rois_from_clicks,
)
from ir_force.single_finger_curve_runtime import (  # noqa: E402
    _thermal_inferno_auto,
    _write_png_exclusive,
)


EXPERIMENT_IDENTITY = "single_finger_hold_check_v2"
SESSION_NAME_PATTERN = re.compile(r"single_finger_hold_check_\d{2}")
DEFAULT_PHASE_S = 30.0
DEFAULT_ROUNDS = 3
FFC_GUARD_S = 5.0
ROLLING_MEDIAN_FRAMES = 5
PREVIEW_SCALE = 4
PHASES = ("LIGHT_A", "HARD", "LIGHT_B")
ROUND_KINDS = ("press", "control")
PHASE_CUES = {
    "press": {
        "LIGHT_A": "LIGHT: rest on the spot, minimum force",
        "HARD": "HARD: press down hard, hold the finger still",
        "LIGHT_B": "LIGHT AGAIN: release to minimum force, stay in contact",
    },
    # The motion control. Pressing always moves the fingertip a few pixels, so
    # phase and geometry are confounded; this reproduces the movement at
    # constant light force to measure what the movement alone is worth.
    "control": {
        "LIGHT_A": "LIGHT: rest on the spot, minimum force",
        "HARD": "NUDGE: roll/slide the fingertip, do NOT press harder",
        "LIGHT_B": "LIGHT AGAIN: settle back, same minimum force",
    },
}
# A phase needs this many tracked frames before its median means anything.
MIN_VALID_FRAMES_PER_PHASE = 20
# Stop a round once the stream has been gone this long instead of burning the
# rest of the wall clock, as hold_check_05 round 3 did for 60 s.
MAX_CONSECUTIVE_STREAM_GAPS = 15
CLICK_PROMPTS = (
    "click 1/3: the FINGERTIP",
    "click 2/3: a point further UP THE SAME FINGER",
    "click 3/3: a surface patch OFF the finger (drift diagnostic)",
)


def _udp_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Lepton port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("Lepton port must be in [1, 65535]")
    return port


def _phase_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--phase-seconds must be a number") from exc
    if not 5.0 <= seconds <= 120.0:
        raise argparse.ArgumentTypeError("--phase-seconds must be in [5, 120]")
    return seconds


def _rounds(value: str) -> int:
    try:
        rounds = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--rounds must be an integer") from exc
    if not 1 <= rounds <= 12:
        raise argparse.ArgumentTypeError("--rounds must be in [1, 12]")
    return rounds


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Thermal-only A-B-A single-finger press sanity check.",
    )
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument("--surface-material", required=True)
    parser.add_argument(
        "--phase-seconds",
        type=_phase_seconds,
        default=DEFAULT_PHASE_S,
        help="duration of each of the three phases in a round (default 30 s)",
    )
    parser.add_argument(
        "--rounds",
        type=_rounds,
        default=DEFAULT_ROUNDS,
        help="number of LIGHT/HARD/LIGHT press rounds (default 3)",
    )
    parser.add_argument(
        "--kinds",
        default=None,
        help=(
            "comma-separated round kinds, e.g. press,control,press,control. "
            "A control round asks for the same fingertip movement at constant "
            "light force, which is the only way to tell a press response from "
            "the motion artifact it is confounded with. Overrides --rounds."
        ),
    )
    parser.add_argument("--lepton-port", type=_udp_port, default=8080)
    parser.add_argument(
        "--manual-ffc",
        action="store_true",
        help="run the approved Pi streamer manual FFC before the first round",
    )
    parser.add_argument(
        "--load-note",
        default=None,
        help="free-text record of any independent force reading, e.g. a scale",
    )
    args = parser.parse_args(argv)
    if not SESSION_NAME_PATTERN.fullmatch(args.session_dir.name):
        parser.error(
            "--session-dir basename must match single_finger_hold_check_NN"
        )
    if args.session_dir.exists():
        parser.error("--session-dir must not already exist")
    args.surface_material = args.surface_material.strip()
    if not args.surface_material:
        parser.error("--surface-material must not be blank")
    if args.kinds is None:
        args.kinds = ["press"] * args.rounds
    else:
        args.kinds = [kind.strip() for kind in args.kinds.split(",") if kind.strip()]
        if not args.kinds:
            parser.error("--kinds must not be blank")
        unknown = sorted(set(args.kinds) - set(ROUND_KINDS))
        if unknown:
            parser.error(f"unknown round kind(s): {', '.join(unknown)}")
        args.rounds = len(args.kinds)
    return args


def phase_at(elapsed_s: float, phase_seconds: float):
    """Return ``(phase, phase_elapsed_s)``, or ``None`` once a round is over."""
    if elapsed_s < 0.0:
        raise ValueError("elapsed_s must not be negative")
    index = int(elapsed_s // phase_seconds)
    if index >= len(PHASES):
        return None
    return PHASES[index], elapsed_s - index * phase_seconds


def _json_value(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if isfinite(float(value)) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_jsonl(stream, row) -> None:
    json.dump(_json_value(row), stream, sort_keys=True, allow_nan=False)
    stream.write("\n")
    stream.flush()


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _anchor_record(anchor: dict) -> dict:
    """The JSON-safe part of an anchor; masks stay out of the JSONL."""
    return {
        key: anchor[key]
        for key in (
            "tip_uv",
            "along_uv",
            "direction_uv",
            "reference_uv",
            "clicked_length_px",
            "finger_width_px",
            "distal_pixel_count",
            "proximal_pixel_count",
        )
    }


class ThermalOnlyArchive:
    """Lossless uint16 thermal frames plus a display-only Inferno rendering."""

    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=False)
        for relative in (
            Path("raw/thermal_uint16"),
            Path("rendered/thermal_inferno_auto"),
            Path("figures"),
        ):
            (self.session_dir / relative).mkdir(parents=True, exist_ok=False)

    def capture(self, *, frame_index: int, thermal_counts: np.ndarray) -> dict:
        frame = np.asarray(thermal_counts)
        if frame.shape != (120, 160) or frame.dtype != np.uint16:
            raise ValueError("thermal_frame_invalid")
        raw_relative = (
            Path("raw/thermal_uint16") / f"frame_{frame_index:06d}.png"
        ).as_posix()
        rendered_relative = (
            Path("rendered/thermal_inferno_auto")
            / f"frame_{frame_index:06d}.png"
        ).as_posix()
        _write_png_exclusive(self.session_dir / raw_relative, frame)
        _write_png_exclusive(
            self.session_dir / rendered_relative,
            _thermal_inferno_auto(frame),
        )
        return {
            "thermal_uint16": raw_relative,
            "thermal_inferno_auto": rendered_relative,
        }


def _preview(thermal_counts: np.ndarray, lines, anchor=None, points=()) -> np.ndarray:
    image = cv2.resize(
        _thermal_inferno_auto(thermal_counts),
        (160 * PREVIEW_SCALE, 120 * PREVIEW_SCALE),
        interpolation=cv2.INTER_NEAREST,
    )
    # Draw every click as it lands, so a click that does not register is
    # visible immediately rather than after a whole round is recorded.
    for index, point in enumerate(points):
        centre = (
            int(round(point[0] * PREVIEW_SCALE)),
            int(round(point[1] * PREVIEW_SCALE)),
        )
        color = ((0, 255, 255), (0, 200, 255), (255, 200, 80))[min(index, 2)]
        cv2.drawMarker(image, centre, color, cv2.MARKER_CROSS, 14, 2)
        cv2.putText(
            image,
            str(index + 1),
            (centre[0] + 8, centre[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    if anchor is not None:
        for key, color in (
            ("distal_mask", (100, 255, 100)),
            ("proximal_mask", (255, 255, 255)),
            ("reference_mask", (255, 200, 80)),
        ):
            mask = anchor.get(key)
            if mask is None:
                continue
            outline = cv2.resize(
                mask.astype(np.uint8) * 255,
                (160 * PREVIEW_SCALE, 120 * PREVIEW_SCALE),
                interpolation=cv2.INTER_NEAREST,
            )
            contours, _ = cv2.findContours(
                outline,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(image, contours, -1, color, 1)
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            str(line),
            (14, 28 + index * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return image


WINDOW = "single-finger press check (auto contrast, display only)"


def default_key_source(thermal_counts, lines, anchor=None, points=()) -> int:
    cv2.imshow(WINDOW, _preview(thermal_counts, lines, anchor, points))
    return cv2.waitKey(1) & 0xFF


class MouseClicks:
    """Collect preview clicks in native thermal pixel coordinates.

    A click once the set is full starts a new set rather than being ignored.
    The first version kept appending and always read back points 0-2, so the
    ROI froze at wherever the operator first clicked and could not be moved.
    """

    CLICKS_PER_SET = len(CLICK_PROMPTS)

    def __init__(self):
        self.points: list[tuple[float, float]] = []
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, self._on_event)

    def _on_event(self, event, x, y, _flags, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(self.points) >= self.CLICKS_PER_SET:
            self.points.clear()
        self.points.append((x / PREVIEW_SCALE, y / PREVIEW_SCALE))

    def undo(self) -> None:
        if self.points:
            self.points.pop()

    def take(self) -> list[tuple[float, float]]:
        return list(self.points)

    def reset(self) -> None:
        self.points.clear()


def read_or_none(thermal_source, last=None):
    """Read a thermal frame, tolerating a UDP dropout.

    `single_finger_hold_check_03` lost round 2 and 3 because a 2 s gap in the
    Pi stream during the untimed rest raised out of the whole session. Idle
    screens now keep showing the last frame and say so.
    """
    try:
        return thermal_source.read(), None
    except FrameUnavailableError as exc:
        return last, str(exc)


def collect_anchor(
    *,
    thermal_source,
    key_source,
    click_source,
    round_index: int,
    rounds: int,
    roi_builder=rois_from_clicks,
):
    """Let the operator click the fingertip, the finger axis, and a reference.

    Returns ``(anchor, frame)``, or ``None`` if the operator aborted. The
    preview marks every click and redraws the ROIs live, so the placement can
    be corrected before anything is recorded: a further click starts a new
    set, ``z`` undoes the last click, and ``r`` clears them all.
    """
    click_source.reset()
    blocker = None
    thermal = None
    while True:
        thermal, stream_error = read_or_none(thermal_source, thermal)
        if thermal is None:
            continue
        points = click_source.take()
        anchor = None
        if len(points) >= len(CLICK_PROMPTS):
            try:
                anchor = roi_builder(
                    thermal.frame,
                    points[0],
                    points[1],
                    points[2],
                )
                blocker = None
            except ValueError as exc:
                blocker = str(exc)
        prompt = (
            CLICK_PROMPTS[len(points)]
            if len(points) < len(CLICK_PROMPTS)
            else "SPACE: start   click again: new set   z: undo   r: clear"
        )
        lines = [
            f"round {round_index + 1}/{rounds}"
            + ("" if stream_error is None else "  [LEPTON STREAM STALLED]"),
            prompt if blocker is None else f"rejected: {blocker}",
            (
                f"finger width {anchor['finger_width_px']:.1f}px, "
                f"distal {anchor['distal_pixel_count']}px"
                if anchor is not None
                else f"{len(points)}/{len(CLICK_PROMPTS)} clicks   "
                "z: undo   r: clear   q: abort"
            ),
        ]
        key = key_source(thermal.frame, lines, anchor, points)
        if key in (ord("q"), ord("Q")):
            return None
        if key in (ord("r"), ord("R")):
            click_source.reset()
            blocker = None
        if key in (ord("z"), ord("Z")):
            click_source.undo()
            blocker = None
        if key == ord(" ") and anchor is not None:
            return anchor, thermal.frame


def capture_round(
    *,
    thermal_source,
    anchor,
    anchor_frame,
    archive,
    stream,
    round_index: int,
    start_frame_index: int,
    phase_seconds: float,
    rounds: int,
    kind: str = "press",
    clock=time.perf_counter,
    key_source=default_key_source,
    tracker_factory=TemplateTracker,
) -> dict:
    tracker = tracker_factory(anchor_frame, anchor)
    started_s = float(clock())
    frame_index = start_frame_index
    rows = []
    aborted = False
    stream_gaps = []
    consecutive_gaps = 0
    stream_lost = False
    while True:
        thermal, stream_error = read_or_none(thermal_source)
        now_s = float(clock())
        elapsed_s = now_s - started_s
        current = phase_at(elapsed_s, phase_seconds)
        if current is None:
            break
        phase, phase_elapsed_s = current
        if thermal is None:
            # A gap loses frames but not the round; the phases are wall-clock
            # bounded, so record it and keep going.
            gap = {
                "row_type": "stream_gap",
                "round_index": round_index,
                "phase": phase,
                "round_elapsed_s": elapsed_s,
                "error": stream_error,
            }
            stream_gaps.append(gap)
            _write_jsonl(stream, gap)
            consecutive_gaps += 1
            if consecutive_gaps >= MAX_CONSECUTIVE_STREAM_GAPS:
                stream_lost = True
                break
            continue
        consecutive_gaps = 0
        artifacts = archive.capture(
            frame_index=frame_index,
            thermal_counts=thermal.frame,
        )
        measured = tracker.measure(thermal.frame)
        telemetry = thermal.lepton_telemetry
        row = {
            "row_type": "frame",
            "frame_index": frame_index,
            "round_index": round_index,
            "phase": phase,
            "phase_elapsed_s": phase_elapsed_s,
            "round_elapsed_s": elapsed_s,
            "thermal_host_s": float(thermal.t),
            "frame_median_count": float(np.median(thermal.frame)),
            "lepton_frame_counter": (
                None if telemetry is None else int(telemetry.frame_counter)
            ),
            "ffc_desired": (
                None if telemetry is None else bool(telemetry.ffc_desired)
            ),
            "frame_artifacts": artifacts,
            "tracked": measured,
        }
        _write_jsonl(stream, row)
        rows.append(row)
        frame_index += 1
        key = key_source(
            thermal.frame,
            (
                f"round {round_index + 1}/{rounds} [{kind}]: "
                f"{PHASE_CUES[kind][phase]}",
                f"{phase_seconds - phase_elapsed_s:.1f} s left",
                (
                    f"tracking OK  score {measured['template_score']:.2f}  "
                    f"shift {measured['shift_magnitude_px']:.1f}px"
                    if measured["tracking_valid"]
                    else "tracking lost: "
                    + ", ".join(measured["tracking_reasons"])
                ),
                "q: abort",
            ),
            anchor,
        )
        if key in (ord("q"), ord("Q")):
            aborted = True
            break
    return {
        "rows": rows,
        "next_frame_index": frame_index,
        "aborted": aborted,
        "stream_gap_count": len(stream_gaps),
        "stream_lost": stream_lost,
    }


def wait_for_rest(*, thermal_source, key_source, round_index: int, rounds: int) -> bool:
    """Untimed operator-paced rest so a taped contact patch can cool."""
    thermal = None
    while True:
        thermal, stream_error = read_or_none(thermal_source, thermal)
        if thermal is None:
            continue
        key = key_source(
            thermal.frame,
            (
                f"round {round_index + 1}/{rounds} done"
                + ("" if stream_error is None else "  [LEPTON STREAM STALLED]"),
                "REST: lift the finger and let the surface cool",
                "SPACE: continue to the next round",
                "q: stop here",
            ),
            None,
        )
        if key in (ord("q"), ord("Q")):
            return False
        if key == ord(" "):
            return True


def round_summary(rows) -> dict:
    summary = {}
    for phase in PHASES:
        values = [
            float(row["tracked"]["primary_signal_count"])
            for row in rows
            if row["phase"] == phase
            and row["tracked"]["primary_signal_count"] is not None
        ]
        summary[phase] = {
            "valid_frames": len(values),
            "median_count": float(np.median(values)) if values else None,
        }
    light_a = summary["LIGHT_A"]["median_count"]
    light_b = summary["LIGHT_B"]["median_count"]
    hard = summary["HARD"]["median_count"]
    aba_effect = None
    recovery_ratio = None
    if None not in (light_a, light_b, hard):
        baseline = (light_a + light_b) / 2.0
        aba_effect = hard - baseline
        # 0 = fully recovered to the opening LIGHT, 1 = stayed at HARD.
        if hard != light_a:
            recovery_ratio = abs(light_b - light_a) / abs(hard - light_a)
    summary["aba_effect_count"] = aba_effect
    summary["return_recovery_ratio"] = recovery_ratio
    summary["valid"] = aba_effect is not None and all(
        summary[phase]["valid_frames"] >= MIN_VALID_FRAMES_PER_PHASE
        for phase in PHASES
    )
    return summary


def round_diagnostics(rows) -> dict:
    """Numbers that decide whether an ABA effect is believable.

    hold_check_04 produced a consistent +6 to +12 count effect while the frame
    median fell by up to 353 counts inside a round, `ffc_desired` was asserted
    on most frames of two rounds, and the ROI shifted 2-4 px under load on a
    10 px wide finger. None of that was visible without reopening the JSONL.
    """
    if not rows:
        return {}
    medians = [float(row["frame_median_count"]) for row in rows]
    ffc = [row["ffc_desired"] for row in rows]
    return {
        "ffc_desired_fraction": (
            float(sum(1 for value in ffc if value)) / len(ffc)
        ),
        "frame_median_drift_count": medians[-1] - medians[0],
        "median_shift_px_by_phase": {
            phase: (
                float(
                    np.median(
                        [
                            row["tracked"]["shift_magnitude_px"]
                            for row in rows
                            if row["phase"] == phase
                        ]
                    )
                )
                if any(row["phase"] == phase for row in rows)
                else None
            )
            for phase in PHASES
        },
        "median_reference_count_by_phase": {
            phase: (
                float(np.median(values))
                if (
                    values := [
                        row["tracked"]["reference_count"]
                        for row in rows
                        if row["phase"] == phase
                        and row["tracked"]["reference_count"] is not None
                    ]
                )
                else None
            )
            for phase in PHASES
        },
    }


def tracking_failure_counts(rows) -> dict:
    counts: dict[str, int] = {}
    for row in rows:
        for reason in row["tracked"]["tracking_reasons"] or []:
            counts[str(reason)] = counts.get(str(reason), 0) + 1
    return counts


def _rolling_median(values, width: int = ROLLING_MEDIAN_FRAMES):
    radius = width // 2
    smoothed = []
    for index in range(len(values)):
        window = [
            value
            for value in values[max(0, index - radius) : index + radius + 1]
            if value is not None
        ]
        smoothed.append(float(np.median(window)) if window else None)
    return smoothed


def plot_rounds(rows_by_round, path: Path, *, phase_seconds: float) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.8, max(len(rows_by_round), 1)))
    plotted = 0
    for index, rows in enumerate(rows_by_round):
        times = [float(row["round_elapsed_s"]) for row in rows]
        values = [row["tracked"]["primary_signal_count"] for row in rows]
        smooth = _rolling_median(values)
        keep = [i for i, value in enumerate(smooth) if value is not None]
        if not keep:
            continue
        plotted += 1
        axes[0].plot(
            [times[i] for i in keep],
            [smooth[i] for i in keep],
            color=colors[index],
            linewidth=1.6,
            label=f"round {index + 1}",
        )
        reference = [row["tracked"]["reference_count"] for row in rows]
        first = next((value for value in reference if value is not None), None)
        if first is not None:
            keep_ref = [i for i, value in enumerate(reference) if value is not None]
            axes[1].plot(
                [times[i] for i in keep_ref],
                [reference[i] - first for i in keep_ref],
                color=colors[index],
                linewidth=1.2,
            )
    axes[0].set_ylabel("distal - proximal (counts)")
    axes[0].set_title(
        "Single-finger A-B-A press check: "
        "median(distal) - median(proximal), rounds overlaid"
    )
    if plotted:
        axes[0].legend(loc="upper right", fontsize=8)
    axes[1].set_ylabel("surface reference\nchange (counts)")
    axes[1].set_xlabel("time since the start of the round (s)")
    axes[1].axhline(0.0, color="0.3", linewidth=0.8)
    for axis in axes:
        for index in range(1, len(PHASES)):
            axis.axvline(
                index * phase_seconds,
                color="0.35",
                linestyle="--",
                linewidth=0.8,
            )
    for index, phase in enumerate(PHASES):
        axes[0].annotate(
            phase,
            ((index + 0.5) * phase_seconds, 0.03),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="bottom",
            fontsize=11,
            color="0.25",
        )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return {"path": path.name, "plotted_rounds": plotted}


def plot_roi_overlay(anchor, anchor_frame, path: Path) -> dict:
    """Written at the start of every round, so a mislocated ROI is visible now.

    Session 01 was recorded in full before anyone noticed the ROIs were on the
    back of the fist.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 4.6))
    lower, upper = np.percentile(anchor_frame, (1.0, 99.0))
    axis.imshow(anchor_frame, cmap="inferno", vmin=lower, vmax=upper)
    for key, color, label in (
        ("distal_mask", "#33ff66", "distal"),
        ("proximal_mask", "#ffffff", "proximal"),
        ("reference_mask", "#33ccff", "surface reference"),
        ("finger_mask", "#ff5555", "local finger mask"),
    ):
        mask = anchor.get(key)
        if mask is None:
            continue
        axis.contour(
            mask.astype(float),
            levels=[0.5],
            colors=[color],
            linewidths=1.0 if key == "finger_mask" else 1.4,
        )
        axis.plot([], [], color=color, label=label)
    axis.plot(*anchor["tip_uv"], "co", markersize=5)
    axis.set_title(
        f"clicked ROIs, finger width {anchor['finger_width_px']:.1f}px",
        fontsize=10,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.legend(loc="lower right", fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return {"path": path.name}


def run_session(
    args,
    *,
    thermal_source_factory=None,
    key_source=default_key_source,
    click_source_factory=MouseClicks,
    clock=time.perf_counter,
    manual_ffc=_run_manual_ffc,
    sleep=time.sleep,
    round_plotter=plot_rounds,
    overlay_plotter=plot_roi_overlay,
) -> dict:
    thermal_source_factory = thermal_source_factory or (
        lambda: LeptonUDPSource(port=args.lepton_port)
    )
    archive = ThermalOnlyArchive(args.session_dir)
    capture_path = args.session_dir / "capture.jsonl"
    manifest_path = args.session_dir / "manifest.json"
    stream = capture_path.open("x", encoding="utf-8")
    thermal_source = None
    rows_by_round = []
    round_records = []
    frame_index = 0
    aborted = False
    error = None
    figure = None
    def start_round_stream(existing):
        """Re-FFC before every round.

        hold_check_04 ran one FFC at the start and then drifted: round 1 had
        `ffc_desired` on 0/370 frames, rounds 2 and 3 on 96% and 100%, with the
        frame median falling 353, 279 and 198 counts inside each round. The
        approved manual FFC path reconfigures the Pi streamer, so the UDP
        source is closed first and reopened after, exactly as
        capture_single_finger_curve does per block.
        """
        if not args.manual_ffc:
            return existing or thermal_source_factory()
        if existing is not None:
            existing.close()
        manual_ffc()
        sleep(FFC_GUARD_S)
        return thermal_source_factory()

    try:
        thermal_source = start_round_stream(None)
        click_source = click_source_factory()
        _write_jsonl(
            stream,
            {
                "row_type": "metadata",
                "experiment_identity": EXPERIMENT_IDENTITY,
                "schema_version": 2,
                "role": "sanity_check_not_preregistered",
                "safety_mode": "robot_free_no_actuation",
                "modalities": ["lepton_thermal"],
                "d435_used": False,
                "surface_material": args.surface_material,
                "load_note": args.load_note,
                "phases": list(PHASES),
                "phase_duration_s": args.phase_seconds,
                "rounds_requested": args.rounds,
                "round_kinds": list(args.kinds),
                "roi_method": "operator_clicked_tip_axis_reference",
                "roi_tracking": "template_ncc_translation",
                "primary_value": "median(distal) - median(proximal)",
                "raw_thermal_authority": "uint16_counts",
                "rendered_thermal_role": "display_only_auto_contrast",
                "manual_ffc_requested": bool(args.manual_ffc),
            },
        )
        for round_index, kind in enumerate(args.kinds):
            if round_index > 0:
                thermal_source = start_round_stream(thermal_source)
            collected = collect_anchor(
                thermal_source=thermal_source,
                key_source=key_source,
                click_source=click_source,
                round_index=round_index,
                rounds=args.rounds,
            )
            if collected is None:
                aborted = True
                break
            anchor, anchor_frame = collected
            overlay = overlay_plotter(
                anchor,
                anchor_frame,
                args.session_dir / "figures" / f"roi_round{round_index + 1}.png",
            )
            _write_jsonl(
                stream,
                {
                    "row_type": "round_anchor",
                    "round_index": round_index,
                    "kind": kind,
                    "anchor": _anchor_record(anchor),
                    "overlay_figure": overlay["path"],
                },
            )
            captured = capture_round(
                thermal_source=thermal_source,
                anchor=anchor,
                anchor_frame=anchor_frame,
                archive=archive,
                stream=stream,
                round_index=round_index,
                start_frame_index=frame_index,
                phase_seconds=args.phase_seconds,
                rounds=args.rounds,
                kind=kind,
                clock=clock,
                key_source=key_source,
            )
            frame_index = captured["next_frame_index"]
            rows_by_round.append(captured["rows"])
            record = {
                "round_index": round_index,
                "kind": kind,
                "stream_lost": captured["stream_lost"],
                "anchor": _anchor_record(anchor),
                "frame_count": len(captured["rows"]),
                "stream_gap_count": captured["stream_gap_count"],
                "diagnostics": round_diagnostics(captured["rows"]),
                "phase_summary": round_summary(captured["rows"]),
                "tracking_failure_counts": tracking_failure_counts(
                    captured["rows"]
                ),
            }
            round_records.append(record)
            _write_jsonl(stream, {"row_type": "round_summary", **record})
            if captured["aborted"]:
                aborted = True
                break
            if round_index + 1 < args.rounds and not wait_for_rest(
                thermal_source=thermal_source,
                key_source=key_source,
                round_index=round_index,
                rounds=args.rounds,
            ):
                aborted = True
                break
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        if thermal_source is not None:
            thermal_source.close()
        stream.close()
        cv2.destroyAllWindows()
        if rows_by_round:
            try:
                figure = round_plotter(
                    rows_by_round,
                    args.session_dir / "figures" / "aba_rounds.png",
                    phase_seconds=args.phase_seconds,
                )
            except Exception as exc:  # pragma: no cover - plotting is optional
                figure = {"error": repr(exc)}
        def effects_for(kind):
            return [
                record["phase_summary"]["aba_effect_count"]
                for record in round_records
                if record["kind"] == kind and record["phase_summary"]["valid"]
            ]

        press_effects = effects_for("press")
        control_effects = effects_for("control")
        effects = press_effects + control_effects
        valid_rounds = [
            record for record in round_records if record["phase_summary"]["valid"]
        ]
        manifest = {
            "experiment_identity": EXPERIMENT_IDENTITY,
            "role": "sanity_check_not_preregistered",
            "status": (
                "error"
                if error is not None
                else "aborted"
                if aborted
                else "complete"
                if len(valid_rounds) == args.rounds
                else "incomplete"
            ),
            "error": error,
            "surface_material": args.surface_material,
            "load_note": args.load_note,
            "phase_duration_s": args.phase_seconds,
            "rounds_requested": args.rounds,
            "round_kinds": list(args.kinds),
            "rounds_recorded": len(round_records),
            "valid_round_count": len(valid_rounds),
            "frame_row_count": frame_index,
            "capture_jsonl": "capture.jsonl",
            "capture_jsonl_sha256": (
                _file_sha256(capture_path) if capture_path.is_file() else None
            ),
            "figure": figure,
            "rounds": round_records,
            "median_aba_effect_count": (
                float(np.median(press_effects)) if press_effects else None
            ),
            "median_control_effect_count": (
                float(np.median(control_effects)) if control_effects else None
            ),
            "raw_thermal_png_count": len(
                list((args.session_dir / "raw/thermal_uint16").glob("*.png"))
            ),
            "force_ground_truth": False,
            "controller_or_robot_actuation": False,
            "stage1f_authority": False,
            "signal_verdict": "not_a_formal_result",
        }
        with manifest_path.open("x", encoding="utf-8") as output:
            json.dump(
                _json_value(manifest),
                output,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            output.write("\n")
    return manifest


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        manifest = run_session(args)
    except Exception as exc:
        print(f"press check failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "rounds_recorded": manifest["rounds_recorded"],
                "valid_round_count": manifest["valid_round_count"],
                "frame_row_count": manifest["frame_row_count"],
                "median_aba_effect_count": manifest["median_aba_effect_count"],
                "median_control_effect_count": manifest[
                    "median_control_effect_count"
                ],
                "per_round": [
                    {
                        "round": record["round_index"] + 1,
                        "kind": record["kind"],
                        "valid": record["phase_summary"]["valid"],
                        "aba_effect_count": record["phase_summary"][
                            "aba_effect_count"
                        ],
                        "return_recovery_ratio": record["phase_summary"][
                            "return_recovery_ratio"
                        ],
                        "ffc_desired_fraction": record["diagnostics"].get(
                            "ffc_desired_fraction"
                        ),
                        "frame_median_drift_count": record["diagnostics"].get(
                            "frame_median_drift_count"
                        ),
                    }
                    for record in manifest["rounds"]
                ],
                "session_dir": str(args.session_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
