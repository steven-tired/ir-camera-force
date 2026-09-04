#!/usr/bin/env python3
"""Bounded hand-associated robot-free D435i-to-Lepton shadow runner."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from math import floor, hypot, isfinite
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


_CHECKOUT_ROOT = Path(__file__).resolve().parents[1]
if str(_CHECKOUT_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHECKOUT_ROOT))

from live_lepton_projector_shadow import (
    FROZEN_XML,
    MAX_HOST_READ_COMPLETION_AGE_S,
    MAX_HOST_READ_COMPLETION_SKEW_S,
    SOURCE_TIME_LIMITATION,
    STAGE0_RUNTIME_JSON,
    _advance_pair_state,
    _load_stage0_contract,
    _pair_rejection_reasons,
    _pair_state,
    _projection_fields,
    _source_fields,
)
from ir_force.data_paths import workspace_root
from ir_force.realsense_camera import RealSenseRawProjectorCamera
from webcam_input.webcam_source import WebcamSource
from ir_force.pinch_geometry import compute_pinch_geometry
from ir_force.ir_capture import LeptonUDPSource
from ir_force.ir_thermal_projection import (
    FROZEN_EXTRINSIC_SHA256,
    load_frozen_thermal_geometry,
)
from ir_force.ir_thermal_sparse_projection import (
    project_raw_depth_samples_to_sparse_thermal,
)


MAX_ATTEMPTS = 900
MAX_FRAME_EVENTS_PER_ATTEMPT = 5
COLOR_WIDTH = 1280
COLOR_HEIGHT = 720
DEPTH_WIDTH = 1280
DEPTH_HEIGHT = 720
DEPTH_MIN_M = 0.20
DEPTH_MAX_M = 0.90
MAX_SDK_MATCH_ERROR_PX = 0.75
INWARD_DIAGNOSTIC_FRACTIONS = (0.25, 0.50)
PINCH_SIGNAL_GROUPS = 6
PINCH_SIGNAL_TARGET_VALID_SAMPLES = 5
PINCH_SIGNAL_PHASE_TIMEOUT_S = 10.0
MANUAL_FFC_COMMAND = [
    str(workspace_root() / "scripts" / "run_lepton_stream.sh"),
    "start",
]


def _attempt_count(value: str) -> int:
    try:
        attempts = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("frames must be an integer") from exc
    if not 1 <= attempts <= MAX_ATTEMPTS:
        raise argparse.ArgumentTypeError(
            f"frames must satisfy 1 <= N <= {MAX_ATTEMPTS}"
        )
    return attempts


def _udp_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Lepton port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("Lepton port must be in [1, 65535]")
    return port


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, type=_attempt_count)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lepton-port", type=_udp_port, default=8080)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--manual-ffc", action="store_true")
    parser.add_argument("--diagnose-inward-samples", action="store_true")
    parser.add_argument("--pinch-signal-trial", action="store_true")
    args = parser.parse_args(argv)
    if args.pinch_signal_trial:
        if not args.preview:
            parser.error("--pinch-signal-trial requires --preview")
        if not args.manual_ffc:
            parser.error("--pinch-signal-trial requires --manual-ffc")
        if args.diagnose_inward_samples:
            parser.error(
                "--pinch-signal-trial rejects --diagnose-inward-samples"
            )
    return args


def _run_manual_ffc(run_command=subprocess.run) -> str:
    result = run_command(
        MANUAL_FFC_COMMAND,
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        raise RuntimeError(
            f"manual FFC command failed with exit {result.returncode}: "
            f"{output.strip()}"
        )
    if "Manual FFC complete" not in output:
        raise RuntimeError("manual FFC completion marker missing")
    return output


def _normalized_color_pixel(normalized_xy) -> tuple[int, int]:
    x, y = (float(value) for value in normalized_xy)
    if not isfinite(x) or not isfinite(y):
        raise ValueError("normalized_color_nonfinite")
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ValueError("normalized_color_out_of_bounds")
    color_x = COLOR_WIDTH - 1 if x == 1.0 else floor(x * COLOR_WIDTH)
    color_y = COLOR_HEIGHT - 1 if y == 1.0 else floor(y * COLOR_HEIGHT)
    return color_x, color_y


def _pinch_signal_initial_state():
    return {
        "group_index": None,
        "phase": "warmup",
        "phase_started_s": None,
        "valid_samples": 0,
        "baseline_center_uv": None,
        "just_touch_centers": (),
    }


def _pinch_signal_unlock(state):
    if state != _pinch_signal_initial_state():
        raise ValueError("only warmup state can be unlocked")
    return {
        "group_index": 0,
        "phase": "prepare_just_touch",
        "phase_started_s": None,
        "valid_samples": 0,
        "baseline_center_uv": None,
        "just_touch_centers": (),
    }


def _pinch_signal_press_space(state, now_s):
    try:
        now_s = float(now_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("time must be finite") from exc
    if not isfinite(now_s):
        raise ValueError("time must be finite")
    phase = state["phase"]
    if phase == "prepare_just_touch":
        next_phase = "record_just_touch"
    elif phase == "prepare_press_hard":
        next_phase = "record_press_hard"
    elif phase == "prepare_return_touch":
        next_phase = "record_return_touch"
    elif phase == "rest":
        return {
            "group_index": state["group_index"] + 1,
            "phase": "prepare_just_touch",
            "phase_started_s": None,
            "valid_samples": 0,
            "baseline_center_uv": None,
            "just_touch_centers": (),
        }
    else:
        return dict(state)
    return {
        "group_index": state["group_index"],
        "phase": next_phase,
        "phase_started_s": now_s,
        "valid_samples": 0,
        "baseline_center_uv": state["baseline_center_uv"],
        "just_touch_centers": state["just_touch_centers"],
    }


def _pinch_signal_accept_sample(state, center_uv):
    phase = state["phase"]
    if not phase.startswith("record_"):
        raise ValueError("only a recording phase can accept a sample")
    try:
        center_uv = tuple(float(value) for value in center_uv)
    except (TypeError, ValueError) as exc:
        raise ValueError("pinch center must contain two finite values") from exc
    if len(center_uv) != 2 or not all(isfinite(value) for value in center_uv):
        raise ValueError("pinch center must contain two finite values")

    center_shift_px = None
    if phase != "record_just_touch":
        baseline = state["baseline_center_uv"]
        if baseline is None:
            raise ValueError("press and return phases require a baseline center")
        center_shift_px = hypot(
            center_uv[0] - baseline[0],
            center_uv[1] - baseline[1],
        )

    just_touch_centers = state["just_touch_centers"]
    if phase == "record_just_touch":
        just_touch_centers = (*just_touch_centers, center_uv)
    valid_samples = state["valid_samples"] + 1
    result = {
        "quota_accepted": True,
        "quota_reasons": [],
        "valid_samples_after": valid_samples,
        "pinch_center_shift_px": center_shift_px,
        "group_completed": False,
    }
    if valid_samples < PINCH_SIGNAL_TARGET_VALID_SAMPLES:
        return {
            **state,
            "valid_samples": valid_samples,
            "just_touch_centers": just_touch_centers,
        }, result

    if phase == "record_just_touch":
        next_phase = "prepare_press_hard"
        baseline_center_uv = tuple(
            float(np.median([center[index] for center in just_touch_centers]))
            for index in (0, 1)
        )
    elif phase == "record_press_hard":
        next_phase = "prepare_return_touch"
        baseline_center_uv = state["baseline_center_uv"]
    elif phase == "record_return_touch":
        result["group_completed"] = True
        if state["group_index"] + 1 >= PINCH_SIGNAL_GROUPS:
            next_phase = "complete"
        else:
            next_phase = "rest"
        baseline_center_uv = state["baseline_center_uv"]
    else:
        raise ValueError("invalid recording phase")
    return {
        "group_index": (
            None if next_phase == "complete" else state["group_index"]
        ),
        "phase": next_phase,
        "phase_started_s": None,
        "valid_samples": 0,
        "baseline_center_uv": baseline_center_uv,
        "just_touch_centers": just_touch_centers,
    }, result


def _pinch_signal_timed_out(state, now_s):
    cue = _pinch_signal_cue(state, now_s)
    return (
        cue["recording"]
        and cue["phase_elapsed_s"] >= PINCH_SIGNAL_PHASE_TIMEOUT_S
    )


def _pinch_signal_invalidate_group(state, *, stop):
    if not state["phase"].startswith("record_"):
        raise ValueError("only a recording phase can invalidate a group")
    if stop:
        next_phase = "blocked"
        group_index = None
    elif state["group_index"] + 1 >= PINCH_SIGNAL_GROUPS:
        next_phase = "complete"
        group_index = None
    else:
        next_phase = "rest"
        group_index = state["group_index"]
    return {
        "group_index": group_index,
        "phase": next_phase,
        "phase_started_s": None,
        "valid_samples": 0,
        "baseline_center_uv": state["baseline_center_uv"],
        "just_touch_centers": state["just_touch_centers"],
    }


def _pinch_signal_cue(state, now_s):
    try:
        now_s = float(now_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("time must be finite and non-decreasing") from exc
    if not isfinite(now_s):
        raise ValueError("time must be finite and non-decreasing")
    phase = state["phase"]
    recording = phase.startswith("record_")
    phase_elapsed_s = 0.0
    phase_remaining_s = 0.0
    if recording:
        started_s = state["phase_started_s"]
        if not isinstance(started_s, (int, float)):
            raise ValueError("recording state requires start time")
        phase_elapsed_s = now_s - float(started_s)
        if phase_elapsed_s < 0.0:
            raise ValueError("time must be finite and non-decreasing")
        phase_remaining_s = max(
            PINCH_SIGNAL_PHASE_TIMEOUT_S - phase_elapsed_s,
            0.0,
        )
    if phase in (
        "prepare_just_touch",
        "record_just_touch",
        "prepare_return_touch",
        "record_return_touch",
    ):
        label = "contact"
    elif phase in ("prepare_press_hard", "record_press_hard"):
        label = "press"
    else:
        label = None
    if phase in ("complete", "blocked"):
        return {
            "complete": phase == "complete",
            "blocked": phase == "blocked",
            "group_index": None,
            "label": None,
            "phase": phase,
            "recording": False,
            "phase_elapsed_s": 0.0,
            "phase_remaining_s": 0.0,
            "phase_timeout_s": PINCH_SIGNAL_PHASE_TIMEOUT_S,
            "valid_samples": 0,
            "target_valid_samples": PINCH_SIGNAL_TARGET_VALID_SAMPLES,
        }
    return {
        "complete": False,
        "blocked": False,
        "group_index": state["group_index"],
        "label": label,
        "phase": phase,
        "recording": recording,
        "phase_elapsed_s": phase_elapsed_s,
        "phase_remaining_s": phase_remaining_s,
        "phase_timeout_s": PINCH_SIGNAL_PHASE_TIMEOUT_S,
        "valid_samples": state["valid_samples"],
        "target_valid_samples": PINCH_SIGNAL_TARGET_VALID_SAMPLES,
    }


def _pinch_signal_instruction(cue):
    phase = cue["phase"]
    if phase == "warmup":
        return "SHOW RIGHT HAND"
    if phase == "prepare_just_touch":
        return "JUST TOUCH - SPACE WHEN READY"
    if phase == "record_just_touch":
        return "HOLD JUST TOUCH"
    if phase == "prepare_press_hard":
        return "PRESS HARD - SPACE WHEN READY"
    if phase == "record_press_hard":
        return "HOLD PRESS HARD"
    if phase == "prepare_return_touch":
        return "RETURN TO JUST TOUCH - SPACE WHEN READY"
    if phase == "record_return_touch":
        return "HOLD JUST TOUCH"
    if phase == "rest":
        return "SEPARATE AND REST - SPACE FOR NEXT GROUP"
    if phase == "complete":
        return "PROTOCOL COMPLETE"
    if phase == "blocked":
        return "PROTOCOL BLOCKED"
    raise ValueError("invalid pinch signal cue")


def _pinch_signal_display_label(cue):
    return cue["label"]


def _thermal_patch_statistics(frame, thermal_pixel):
    x, y = (int(value) for value in thermal_pixel)
    y0 = max(0, y - 1)
    y1 = min(frame.shape[0], y + 2)
    x0 = max(0, x - 1)
    x1 = min(frame.shape[1], x + 2)
    patch = np.asarray(frame[y0:y1, x0:x1], dtype=float)
    if patch.size == 0:
        raise ValueError("thermal pixel is outside frame")
    return {
        "thermal_patch_3x3_mean_count": float(np.mean(patch)),
        "thermal_patch_3x3_std_counts": float(np.std(patch)),
    }


def _thermal_patch_pixels(thermal_pixel, frame_shape):
    x, y = (int(value) for value in thermal_pixel)
    height, width = frame_shape
    return {
        (patch_x, patch_y)
        for patch_y in range(max(0, y - 1), min(height, y + 2))
        for patch_x in range(max(0, x - 1), min(width, x + 2))
    }


def _thermal_patches_overlap(thermal_pixels, frame_shape):
    if len(thermal_pixels) != 2:
        raise ValueError("exactly two thermal pixels are required")
    left, right = (
        _thermal_patch_pixels(pixel, frame_shape)
        for pixel in thermal_pixels
    )
    return bool(left & right)


def _thermal_pinch_center_uv(fingertips):
    by_label = {
        tip.get("label"): tip.get("thermal_uv")
        for tip in fingertips
        if tip.get("label") in ("thumb_tip", "index_tip")
    }
    if set(by_label) != {"thumb_tip", "index_tip"}:
        raise ValueError("two labelled fingertip thermal coordinates required")
    thumb = by_label["thumb_tip"]
    index = by_label["index_tip"]
    values = tuple(float(value) for value in (*thumb, *index))
    if len(values) != 4 or not all(isfinite(value) for value in values):
        raise ValueError("fingertip thermal coordinates must be finite")
    return (
        0.5 * (values[0] + values[2]),
        0.5 * (values[1] + values[3]),
    )


def _pinch_geometry_record(image_xy, fingertips):
    depth_m = np.full(21, np.nan, dtype=float)
    depth_index = {"thumb_tip": 4, "index_tip": 8}
    for tip in fingertips:
        index = depth_index.get(tip.get("label"))
        if index is not None:
            depth_m[index] = float(tip.get("depth_m", np.nan))
    result = compute_pinch_geometry(
        image_xy,
        depth_m,
        width_px=COLOR_WIDTH,
        height_px=COLOR_HEIGHT,
    )
    return {
        "valid": bool(result.valid),
        "reason": result.reason.value,
        "pinch_distance_2d_norm": float(result.pinch_distance_2d_norm),
        "pinch_depth_delta_m": float(result.pinch_depth_delta_m),
    }


def _inward_diagnostic_samples(image_xy):
    samples = []
    for label, tip_index, previous_index in (
        ("thumb_tip", 4, 3),
        ("index_tip", 8, 7),
    ):
        tip_xy = image_xy[tip_index]
        previous_xy = image_xy[previous_index]
        for fraction in INWARD_DIAGNOSTIC_FRACTIONS:
            inward_xy = tuple(
                float(tip + fraction * (previous - tip))
                for tip, previous in zip(tip_xy, previous_xy, strict=True)
            )
            samples.append((label, fraction, inward_xy))
    return samples


def _associate_color_to_raw_depth(
    *,
    label,
    normalized_xy,
    depth_z16,
    depth_sdk_buffer,
    rs_module,
    depth_scale_m,
    depth_intrinsics,
    color_intrinsics,
    color_to_depth_extrinsics,
    depth_to_color_extrinsics,
):
    normalized_values = []
    for value in normalized_xy:
        number = float(value)
        normalized_values.append(number if isfinite(number) else None)
    record = {
        "label": label,
        "normalized_color_xy": normalized_values,
    }
    try:
        color_pixel = _normalized_color_pixel(normalized_xy)
    except (TypeError, ValueError) as exc:
        reason = str(exc)
        if reason not in {
            "normalized_color_nonfinite",
            "normalized_color_out_of_bounds",
        }:
            reason = "normalized_color_invalid"
        return {**record, "status": "blocked", "reason": reason}
    record["color_pixel"] = list(color_pixel)
    if depth_sdk_buffer is None:
        return {
            **record,
            "status": "blocked",
            "reason": "color_to_depth_sdk_buffer_missing",
        }
    try:
        sdk_depth_uv = rs_module.rs2_project_color_pixel_to_depth_pixel(
            depth_sdk_buffer,
            depth_scale_m,
            DEPTH_MIN_M,
            DEPTH_MAX_M,
            depth_intrinsics,
            color_intrinsics,
            color_to_depth_extrinsics,
            depth_to_color_extrinsics,
            [float(color_pixel[0]), float(color_pixel[1])],
        )
    except Exception as exc:
        return {
            **record,
            "status": "blocked",
            "reason": "color_to_depth_sdk_failed",
            "error": repr(exc),
        }
    try:
        sdk_depth_uv = tuple(sdk_depth_uv)
    except TypeError:
        sdk_depth_uv = ()
    if len(sdk_depth_uv) != 2:
        return {
            **record,
            "status": "blocked",
            "reason": "color_to_depth_sdk_invalid",
        }
    if sdk_depth_uv == (-1.0, -1.0):
        return {**record, "status": "blocked", "reason": "color_to_depth_sdk_no_match"}
    if not all(
        isfinite(float(value)) for value in sdk_depth_uv
    ):
        return {
            **record,
            "status": "blocked",
            "reason": "color_to_depth_sdk_nonfinite",
            "sdk_depth_uv": [
                float(value) if isfinite(float(value)) else None
                for value in sdk_depth_uv
            ],
        }
    sdk_depth_uv = tuple(float(value) for value in sdk_depth_uv)
    record["sdk_depth_uv"] = list(sdk_depth_uv)
    if not (
        0.0 <= sdk_depth_uv[0] <= DEPTH_WIDTH - 1
        and 0.0 <= sdk_depth_uv[1] <= DEPTH_HEIGHT - 1
    ):
        return {
            **record,
            "status": "blocked",
            "reason": "color_to_depth_sdk_out_of_bounds",
        }
    depth_pixel = (
        floor(sdk_depth_uv[0]),
        floor(sdk_depth_uv[1]),
    )
    record["depth_pixel"] = list(depth_pixel)
    record["legacy_half_up_depth_pixel"] = [
        floor(sdk_depth_uv[0] + 0.5),
        floor(sdk_depth_uv[1] + 0.5),
    ]
    raw_depth = int(depth_z16[depth_pixel[1], depth_pixel[0]])
    record["raw_depth"] = raw_depth
    if raw_depth == 0:
        return {
            **record,
            "status": "blocked",
            "reason": "color_to_depth_zero_depth",
        }
    depth_m = raw_depth * depth_scale_m
    record["depth_m"] = depth_m
    if not DEPTH_MIN_M <= depth_m <= DEPTH_MAX_M:
        return {
            **record,
            "status": "blocked",
            "reason": "color_to_depth_depth_out_of_range",
        }
    try:
        sdk_depth_xyz = rs_module.rs2_deproject_pixel_to_point(
            depth_intrinsics,
            list(sdk_depth_uv),
            depth_m,
        )
        sdk_color_xyz = rs_module.rs2_transform_point_to_point(
            depth_to_color_extrinsics,
            sdk_depth_xyz,
        )
    except Exception as exc:
        return {
            **record,
            "status": "blocked",
            "reason": "color_to_depth_forward_geometry_failed",
            "error": repr(exc),
        }
    if (
        len(sdk_depth_xyz) != 3
        or not all(isfinite(float(value)) for value in sdk_depth_xyz)
        or float(sdk_depth_xyz[2]) <= 0.0
        or
        len(sdk_color_xyz) != 3
        or not all(isfinite(float(value)) for value in sdk_color_xyz)
        or float(sdk_color_xyz[2]) <= 0.0
    ):
        return {
            **record,
            "status": "blocked",
            "reason": "color_to_depth_forward_geometry_invalid",
        }
    try:
        sdk_reprojected_color_uv = rs_module.rs2_project_point_to_pixel(
            color_intrinsics,
            sdk_color_xyz,
        )
    except Exception as exc:
        return {
            **record,
            "status": "blocked",
            "reason": "color_to_depth_forward_geometry_failed",
            "error": repr(exc),
        }
    if len(sdk_reprojected_color_uv) != 2 or not all(
        isfinite(float(value)) for value in sdk_reprojected_color_uv
    ):
        return {
            **record,
            "status": "blocked",
            "reason": "color_to_depth_forward_geometry_invalid",
        }
    sdk_reprojected_color_uv = tuple(
        float(value) for value in sdk_reprojected_color_uv
    )
    record["sdk_reprojected_color_uv"] = list(sdk_reprojected_color_uv)
    sdk_match_error_px = hypot(
        sdk_reprojected_color_uv[0] - color_pixel[0],
        sdk_reprojected_color_uv[1] - color_pixel[1],
    )
    record["sdk_match_error_px"] = sdk_match_error_px
    if sdk_match_error_px > MAX_SDK_MATCH_ERROR_PX:
        return {
            **record,
            "status": "blocked",
            "reason": "color_to_depth_sdk_match_error_exceeded",
        }
    try:
        source_cell_depth_xyz = rs_module.rs2_deproject_pixel_to_point(
            depth_intrinsics,
            [float(depth_pixel[0]), float(depth_pixel[1])],
            depth_m,
        )
        source_cell_color_xyz = rs_module.rs2_transform_point_to_point(
            depth_to_color_extrinsics,
            source_cell_depth_xyz,
        )
    except Exception as exc:
        return {
            **record,
            "status": "blocked",
            "reason": "color_to_depth_forward_geometry_failed",
            "error": repr(exc),
        }
    if (
        len(source_cell_depth_xyz) != 3
        or not all(isfinite(float(value)) for value in source_cell_depth_xyz)
        or float(source_cell_depth_xyz[2]) <= 0.0
        or
        len(source_cell_color_xyz) != 3
        or not all(isfinite(float(value)) for value in source_cell_color_xyz)
        or float(source_cell_color_xyz[2]) <= 0.0
    ):
        return {
            **record,
            "status": "blocked",
            "reason": "color_to_depth_forward_geometry_invalid",
        }
    try:
        source_cell_reprojected_color_uv = (
            rs_module.rs2_project_point_to_pixel(
                color_intrinsics,
                source_cell_color_xyz,
            )
        )
    except Exception as exc:
        return {
            **record,
            "status": "blocked",
            "reason": "color_to_depth_forward_geometry_failed",
            "error": repr(exc),
        }
    if len(source_cell_reprojected_color_uv) != 2 or not all(
        isfinite(float(value)) for value in source_cell_reprojected_color_uv
    ):
        return {
            **record,
            "status": "blocked",
            "reason": "color_to_depth_forward_geometry_invalid",
        }
    record["source_cell_reprojected_color_uv"] = [
        float(value) for value in source_cell_reprojected_color_uv
    ]
    return {
        **record,
        "status": "ok",
    }


def _json_value(value):
    if isinstance(value, float):
        return value if isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_jsonl(stream, row):
    json.dump(_json_value(row), stream, sort_keys=True, allow_nan=False)
    stream.write("\n")
    stream.flush()


def _show_preview(
    cv2,
    color_rgb,
    row,
    attempts,
    *,
    pinch_signal_instruction=_pinch_signal_instruction,
    pinch_signal_display_label=_pinch_signal_display_label,
):
    frame = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    status = row["status"]
    cue = row.get("pinch_signal")
    attempt = row.get("attempt_index", attempts - 1) + 1
    status_color = (0, 220, 0) if status == "software_gate_accepted" else (0, 0, 255)
    cv2.putText(
        frame,
        f"{attempt}/{attempts} {status}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2,
        cv2.LINE_AA,
    )
    reasons = list(row.get("reasons", ()))
    if not reasons and cue is not None:
        reasons = list(cue.get("quota_reasons", ()))
    reasons = ",".join(reasons) or "none"
    cv2.putText(
        frame,
        reasons[:140],
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        status_color,
        1,
        cv2.LINE_AA,
    )
    depth_parts = []
    for tip in row.get("fingertips", ()):
        depth_m = tip.get("depth_m")
        if isinstance(depth_m, (int, float)) and isfinite(float(depth_m)):
            label = str(tip.get("label", "tip")).removesuffix("_tip")
            depth_parts.append(f"{label}={float(depth_m):.3f} m")
    depth_text = (
        f"tip depth: {' '.join(depth_parts)}"
        if depth_parts
        else "tip depth: unavailable"
    )
    cv2.putText(
        frame,
        depth_text,
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if cue is not None:
        cv2.putText(
            frame,
            pinch_signal_instruction(cue),
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        group_index = cue.get("group_index")
        if cue["phase"] == "warmup":
            countdown = "waiting for first accepted projection"
        elif cue["complete"]:
            countdown = "complete"
        elif cue.get("blocked"):
            countdown = "blocked"
        elif cue["recording"]:
            countdown = (
                f"group {group_index + 1}/{PINCH_SIGNAL_GROUPS} "
                f"{pinch_signal_display_label(cue)} valid "
                f"{cue.get('valid_samples_after', cue['valid_samples'])}/"
                f"{cue['target_valid_samples']} "
                f"timeout {cue['phase_remaining_s']:.1f}s"
            )
        else:
            countdown = (
                f"group {group_index + 1}/{PINCH_SIGNAL_GROUPS} "
                "waiting for SPACE"
            )
        cv2.putText(
            frame,
            countdown,
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        center_shift_px = row.get("pinch_center_shift_px")
        center_text = (
            "pinch center: baseline pending"
            if center_shift_px is None
            else f"pinch center shift: {float(center_shift_px):.2f} px"
        )
        cv2.putText(
            frame,
            center_text,
            (20, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    for tip in row.get("fingertips", ()):
        normalized_xy = tip.get("normalized_color_xy")
        if normalized_xy is None:
            continue
        try:
            pixel = _normalized_color_pixel(normalized_xy)
        except (TypeError, ValueError):
            continue
        color = (0, 220, 0) if tip["status"] == "ok" else (0, 165, 255)
        cv2.circle(frame, pixel, 10, color, 3)
        cv2.putText(
            frame,
            tip["label"],
            (pixel[0] + 12, pixel[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.imshow(
        (
            "Stage 1E pinch signal preview"
            if cue is not None
            else "Stage 1D hand shadow preview"
        ),
        frame,
    )
    return cv2.waitKey(1)


def _validated_attempts(attempts):
    if isinstance(attempts, bool) or not isinstance(attempts, int):
        raise ValueError("attempts must be an integer")
    if not 1 <= attempts <= MAX_ATTEMPTS:
        raise ValueError(f"attempts must satisfy 1 <= N <= {MAX_ATTEMPTS}")
    return attempts


def _sparse_pair_rejection_reason(sparse, depth_pixels):
    if sparse.collision_count != 0:
        return "fingertip_thermal_collision"
    if (
        sparse.status != "ok"
        or sparse.input_count != 2
        or sparse.accepted_count != 2
        or sparse.rejected_count != 0
        or len(sparse.winners) != 2
    ):
        return "fingertip_sparse_pair_incomplete"
    winner_sources = tuple(
        tuple(projected.source_depth_xy)
        for _thermal_xy, projected in sparse.winners
    )
    if len(set(winner_sources)) != 2 or set(winner_sources) != set(depth_pixels):
        return "fingertip_sparse_identity_mismatch"
    thermal_pixels = tuple(thermal_xy for thermal_xy, _projected in sparse.winners)
    if len(set(thermal_pixels)) != 2:
        return "fingertip_thermal_collision"
    return None


def run_shadow(
    *,
    attempts,
    output_path,
    rs_module,
    raw_source_factory,
    thermal_source_factory,
    hands_factory,
    preview=False,
    manual_ffc_before_start=False,
    diagnose_inward_samples=False,
    pinch_signal_trial=False,
    attempt_artifact_writer=None,
    pinch_signal_instruction=_pinch_signal_instruction,
    pinch_signal_display_label=_pinch_signal_display_label,
    clock=time.perf_counter,
    frozen_xml=FROZEN_XML,
    stage0_json=STAGE0_RUNTIME_JSON,
):
    attempts = _validated_attempts(attempts)
    if attempt_artifact_writer is not None and not pinch_signal_trial:
        raise ValueError(
            "attempt artifact archive requires pinch signal trial"
        )
    if pinch_signal_trial and not preview:
        raise ValueError("pinch signal trial requires preview")
    if pinch_signal_trial and not manual_ffc_before_start:
        raise ValueError("pinch signal trial requires manual FFC")
    if pinch_signal_trial and diagnose_inward_samples:
        raise ValueError("pinch signal trial rejects inward diagnostics")
    stage0_contract, stage0_hash = _load_stage0_contract(Path(stage0_json))
    R_tc, T_tc, thermal_K, thermal_D = load_frozen_thermal_geometry(frozen_xml)
    summary = {
        "frame_events": 0,
        "depth_reused_events": 0,
        "attempted": 0,
        "fresh_depth_attempts": 0,
        "association_eligible_attempts": 0,
        "software_gate_accepted": 0,
        "blocked": 0,
    }
    if pinch_signal_trial:
        summary.update(
            pinch_signal_started=False,
            pinch_signal_protocol_completed=False,
            pinch_signal_acquisition_blocked=False,
            pinch_signal_valid_groups=0,
            pinch_signal_invalid_groups=0,
            pinch_signal_phase_timeouts=0,
        )

    with ExitStack() as cleanup:
        preview_cv2 = None
        if preview:
            os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
            import cv2

            preview_cv2 = cv2
            cleanup.callback(preview_cv2.destroyAllWindows)
        stream = cleanup.enter_context(
            Path(output_path).open("x", encoding="utf-8")
        )
        raw_source = raw_source_factory()
        cleanup.callback(raw_source.stop)
        raw_source.start()
        if raw_source.runtime_metadata != stage0_contract:
            raise ValueError("Stage 0 runtime metadata mismatch")

        metadata = {
            "row_type": "metadata",
            "status": "ok",
            "schema_version": 4 if pinch_signal_trial else 2,
            "safety_mode": "robot_free_hand_shadow_only",
            "preview_enabled": bool(preview),
            "manual_ffc_before_start": bool(manual_ffc_before_start),
            "diagnose_inward_samples": bool(diagnose_inward_samples),
            "fingertips": ["thumb_tip", "index_tip"],
            "max_attempts": attempts,
            "attempt_accounting": "fresh_depth_frame",
            "duplicate_depth_rows": "frame_event_not_attempted",
            "depth_search_bounds_m": [DEPTH_MIN_M, DEPTH_MAX_M],
            "sdk_match_error_metric": "euclidean_color_pixels",
            "max_sdk_match_error_px": MAX_SDK_MATCH_ERROR_PX,
            "max_host_read_completion_age_s": MAX_HOST_READ_COMPLETION_AGE_S,
            "max_host_read_completion_skew_s": (
                MAX_HOST_READ_COMPLETION_SKEW_S
            ),
            "metadata_comparison": "exact_normalized_equality",
            "stage0_runtime_json": str(stage0_json),
            "stage0_runtime_sha256": stage0_hash,
            "frozen_xml": str(frozen_xml),
            "frozen_xml_sha256": FROZEN_EXTRINSIC_SHA256,
            "d435_runtime": raw_source.runtime_metadata,
            "thermal_shape": [120, 160],
            "limitations": [SOURCE_TIME_LIMITATION],
        }
        if pinch_signal_trial:
            metadata["pinch_signal_protocol"] = {
                "groups": PINCH_SIGNAL_GROUPS,
                "sequence": [
                    "just_touch",
                    "press_hard",
                    "return_just_touch",
                ],
                "target_valid_samples": (
                    PINCH_SIGNAL_TARGET_VALID_SAMPLES
                ),
                "phase_timeout_s": PINCH_SIGNAL_PHASE_TIMEOUT_S,
                "pinch_center_policy": "diagnostic_only",
                "phase_completion": "accepted_sample_quota",
                "advance_key": "space",
                "start_trigger": "first_software_gate_accepted",
            }
        if attempt_artifact_writer is not None:
            metadata["visualization_capture"] = (
                attempt_artifact_writer.metadata()
            )
        _write_jsonl(stream, metadata)

        thermal_source = thermal_source_factory()
        cleanup.callback(thermal_source.close)
        hands = hands_factory()
        cleanup.callback(hands.close)
        prior = None
        last_depth_frame_number = None
        attempt_index = 0
        last_progress_attempt = 0
        pinch_signal_state = _pinch_signal_initial_state()
        last_pinch_signal_transition = None

        def emit_pinch_signal_transition(cue):
            nonlocal last_pinch_signal_transition
            transition = (cue["group_index"], cue["phase"])
            if transition == last_pinch_signal_transition:
                return
            last_pinch_signal_transition = transition
            print(
                "Stage 1E cue: "
                f"{pinch_signal_instruction(cue)}"
                + (
                    ""
                    if cue["group_index"] is None
                    else (
                        f" [group {cue['group_index'] + 1}/"
                        f"{PINCH_SIGNAL_GROUPS}, "
                        f"{pinch_signal_display_label(cue)}]"
                    )
                ),
                file=sys.stderr,
                flush=True,
            )

        def show_preview_and_handle_key(row, color_rgb, now_s):
            nonlocal pinch_signal_state
            if preview_cv2 is None:
                return
            key = _show_preview(
                preview_cv2,
                color_rgb,
                row,
                attempts,
                pinch_signal_instruction=pinch_signal_instruction,
                pinch_signal_display_label=pinch_signal_display_label,
            )
            if not pinch_signal_trial or key != 32:
                return
            next_state = _pinch_signal_press_space(
                pinch_signal_state,
                now_s,
            )
            if next_state != pinch_signal_state:
                pinch_signal_state = next_state
                emit_pinch_signal_transition(
                    _pinch_signal_cue(pinch_signal_state, now_s)
                )

        def reject_pinch_signal_quota(row):
            cue = row.get("pinch_signal")
            if cue is None or not cue["recording"]:
                return
            cue.update(
                quota_accepted=False,
                quota_reasons=list(row.get("reasons", ())),
                valid_samples_after=cue["valid_samples"],
                pinch_center_shift_px=None,
                group_completed=False,
            )

        while attempt_index < attempts:
            if (
                summary["frame_events"]
                >= attempts * MAX_FRAME_EVENTS_PER_ATTEMPT
            ):
                raise RuntimeError(
                    "insufficient fresh depth frames within bounded "
                    "frame-event budget"
                )
            if (
                attempt_index
                and attempt_index % 30 == 0
                and attempt_index != last_progress_attempt
            ):
                print(
                    ("Stage 1E" if pinch_signal_trial else "Stage 1D")
                    + " progress: "
                    f"{summary['attempted']}/{attempts} attempts; "
                    f"accepted={summary['software_gate_accepted']} "
                    f"blocked={summary['blocked']}",
                    file=sys.stderr,
                    flush=True,
                )
                last_progress_attempt = attempt_index
            raw = raw_source.read()
            thermal = thermal_source.read()
            now_s = clock()
            summary["frame_events"] += 1
            pinch_signal_cue = None
            if pinch_signal_trial:
                pinch_signal_cue = _pinch_signal_cue(
                    pinch_signal_state,
                    now_s,
                )
                emit_pinch_signal_transition(pinch_signal_cue)
                if (
                    pinch_signal_cue["complete"]
                    or pinch_signal_cue["blocked"]
                ):
                    summary["pinch_signal_protocol_completed"] = bool(
                        pinch_signal_cue["complete"]
                        and summary["pinch_signal_valid_groups"]
                        >= max(1, PINCH_SIGNAL_GROUPS - 1)
                    )
                    summary["pinch_signal_acquisition_blocked"] = bool(
                        pinch_signal_cue["blocked"]
                        or not summary["pinch_signal_protocol_completed"]
                    )
                    completion_row = {
                        "row_type": "frame_event",
                        "event_index": summary["frame_events"] - 1,
                        "event": (
                            "pinch_signal_complete"
                            if summary["pinch_signal_protocol_completed"]
                            else "pinch_signal_blocked"
                        ),
                        "status": "not_attempted",
                        "reasons": (
                            []
                            if summary["pinch_signal_protocol_completed"]
                            else ["insufficient_valid_groups"]
                        ),
                        "pinch_signal": pinch_signal_cue,
                        "limitations": [SOURCE_TIME_LIMITATION],
                        **_source_fields(raw, thermal, now_s),
                    }
                    _write_jsonl(stream, completion_row)
                    show_preview_and_handle_key(
                        completion_row,
                        raw.color_rgb,
                        now_s,
                    )
                    break
                if _pinch_signal_timed_out(pinch_signal_state, now_s):
                    timeout_cue = dict(pinch_signal_cue)
                    timeout_cue.update(
                        quota_accepted=False,
                        quota_reasons=["pinch_signal_phase_timeout"],
                        valid_samples_after=timeout_cue["valid_samples"],
                        pinch_center_shift_px=None,
                        group_completed=False,
                    )
                    summary["pinch_signal_phase_timeouts"] += 1
                    summary["pinch_signal_invalid_groups"] += 1
                    stop = summary["pinch_signal_invalid_groups"] >= 2
                    pinch_signal_state = _pinch_signal_invalidate_group(
                        pinch_signal_state,
                        stop=stop,
                    )
                    timeout_row = {
                        "row_type": "frame_event",
                        "event_index": summary["frame_events"] - 1,
                        "event": "pinch_signal_phase_timeout",
                        "status": "not_attempted",
                        "reasons": ["pinch_signal_phase_timeout"],
                        "pinch_signal": timeout_cue,
                        "limitations": [SOURCE_TIME_LIMITATION],
                        **_source_fields(raw, thermal, now_s),
                    }
                    _write_jsonl(stream, timeout_row)
                    emit_pinch_signal_transition(
                        _pinch_signal_cue(pinch_signal_state, now_s)
                    )
                    show_preview_and_handle_key(
                        timeout_row,
                        raw.color_rgb,
                        now_s,
                    )
                    continue
            reasons = list(
                _pair_rejection_reasons(
                    raw,
                    thermal,
                    now_s=now_s,
                    prior=prior,
                )
            )
            if thermal.frame.shape != (120, 160) or thermal.frame.dtype != np.uint16:
                reasons.append("lepton_frame_invalid")
            current_state = _pair_state(raw, thermal)
            depth_frame_number = int(raw.depth_frame_number)
            depth_reused = (
                last_depth_frame_number is not None
                and depth_frame_number == last_depth_frame_number
            )
            prior = _advance_pair_state(prior, current_state)
            if depth_reused:
                event_reasons = [
                    (
                        "d435_depth_frame_reused"
                        if reason == "d435_depth_frame_non_increasing"
                        else reason
                    )
                    for reason in reasons
                ]
                if "d435_depth_frame_reused" not in event_reasons:
                    event_reasons.insert(0, "d435_depth_frame_reused")
                event_row = {
                    "row_type": "frame_event",
                    "event_index": summary["frame_events"] - 1,
                    "event": "depth_reused",
                    "status": "not_attempted",
                    "reasons": event_reasons,
                    "limitations": [SOURCE_TIME_LIMITATION],
                    **_source_fields(raw, thermal, now_s),
                }
                if pinch_signal_cue is not None:
                    event_row["pinch_signal"] = pinch_signal_cue
                summary["depth_reused_events"] += 1
                reject_pinch_signal_quota(event_row)
                _write_jsonl(stream, event_row)
                continue
            last_depth_frame_number = depth_frame_number
            row = {
                "row_type": "attempt",
                "attempt_index": attempt_index,
                "limitations": [SOURCE_TIME_LIMITATION],
                **_source_fields(raw, thermal, now_s),
            }
            if pinch_signal_cue is not None:
                row["pinch_signal"] = pinch_signal_cue
            if (
                pinch_signal_trial
                and thermal.frame.shape == (120, 160)
                and thermal.frame.dtype == np.uint16
            ):
                row["thermal_frame_median_count"] = float(
                    np.median(thermal.frame)
                )
            if attempt_artifact_writer is not None:
                row["frame_artifacts"] = attempt_artifact_writer.capture(
                    attempt_index=attempt_index,
                    thermal_counts=thermal.frame,
                    color_rgb=raw.color_rgb,
                    depth_z16=raw.depth_z16,
                )
            attempt_index += 1
            summary["attempted"] += 1
            summary["fresh_depth_attempts"] += 1
            if reasons:
                row.update(status="blocked", reasons=reasons)
                summary["blocked"] += 1
                reject_pinch_signal_quota(row)
                _write_jsonl(stream, row)
                show_preview_and_handle_key(row, raw.color_rgb, now_s)
                continue

            summary["association_eligible_attempts"] += 1
            hand_results = hands.process(raw.color_rgb)
            handedness = []
            for handed in getattr(
                hand_results, "multi_handedness", None
            ) or ():
                classifications = getattr(handed, "classification", None) or ()
                if classifications:
                    classification = classifications[0]
                    handedness.append(
                        {
                            "label": str(classification.label),
                            "score": float(classification.score),
                        }
                    )
            row["mediapipe"] = {
                "image_hand_count": len(
                    getattr(hand_results, "multi_hand_landmarks", None) or ()
                ),
                "world_hand_count": len(
                    getattr(
                        hand_results,
                        "multi_hand_world_landmarks",
                        None,
                    )
                    or ()
                ),
                "handedness": handedness,
            }
            right, _left = WebcamSource.split_results(hand_results)
            if right is None:
                row.update(
                    status="blocked",
                    reasons=["physical_right_hand_missing"],
                    fingertips=[],
                )
                summary["blocked"] += 1
                reject_pinch_signal_quota(row)
                _write_jsonl(stream, row)
                show_preview_and_handle_key(row, raw.color_rgb, now_s)
                continue

            image_xy = right[1]
            depth_sdk_buffer = (
                None
                if raw.depth_sdk_frame is None
                else raw.depth_sdk_frame.get_data()
            )
            fingertip_inputs = (
                ("thumb_tip", image_xy[4]),
                ("index_tip", image_xy[8]),
            )
            fingertips = [
                _associate_color_to_raw_depth(
                    label=label,
                    normalized_xy=normalized_xy,
                    depth_z16=raw.depth_z16,
                    depth_sdk_buffer=depth_sdk_buffer,
                    rs_module=rs_module,
                    depth_scale_m=raw_source.runtime_metadata["depth_scale_m"],
                    depth_intrinsics=raw_source.depth_intrinsics,
                    color_intrinsics=raw_source.color_intrinsics,
                    color_to_depth_extrinsics=(
                        raw_source.color_to_depth_extrinsics
                    ),
                    depth_to_color_extrinsics=(
                        raw_source.depth_to_color_extrinsics
                    ),
                )
                for label, normalized_xy in fingertip_inputs
            ]
            row["fingertips"] = fingertips
            if pinch_signal_trial:
                row["pinch_geometry"] = _pinch_geometry_record(
                    image_xy,
                    fingertips,
                )
            if diagnose_inward_samples:
                row["inward_association_diagnostics"] = [
                    {
                        **_associate_color_to_raw_depth(
                            label=label,
                            normalized_xy=normalized_xy,
                            depth_z16=raw.depth_z16,
                            depth_sdk_buffer=depth_sdk_buffer,
                            rs_module=rs_module,
                            depth_scale_m=raw_source.runtime_metadata[
                                "depth_scale_m"
                            ],
                            depth_intrinsics=raw_source.depth_intrinsics,
                            color_intrinsics=raw_source.color_intrinsics,
                            color_to_depth_extrinsics=(
                                raw_source.color_to_depth_extrinsics
                            ),
                            depth_to_color_extrinsics=(
                                raw_source.depth_to_color_extrinsics
                            ),
                        ),
                        "inward_fraction": fraction,
                    }
                    for label, fraction, normalized_xy in (
                        _inward_diagnostic_samples(image_xy)
                    )
                ]
            association_reasons = [
                tip["reason"] for tip in fingertips if tip["status"] != "ok"
            ]
            depth_pixels = [
                tuple(tip["depth_pixel"])
                for tip in fingertips
                if tip["status"] == "ok"
            ]
            if association_reasons:
                row.update(status="blocked", reasons=association_reasons)
                summary["blocked"] += 1
                reject_pinch_signal_quota(row)
                _write_jsonl(stream, row)
                show_preview_and_handle_key(row, raw.color_rgb, now_s)
                continue
            if len(set(depth_pixels)) != 2:
                row.update(
                    status="blocked",
                    reasons=["fingertip_depth_pixel_collision"],
                )
                summary["blocked"] += 1
                reject_pinch_signal_quota(row)
                _write_jsonl(stream, row)
                show_preview_and_handle_key(row, raw.color_rgb, now_s)
                continue

            samples = [
                (
                    tip["depth_pixel"][0],
                    tip["depth_pixel"][1],
                    tip["raw_depth"],
                )
                for tip in fingertips
            ]
            sparse = project_raw_depth_samples_to_sparse_thermal(
                samples=samples,
                rs_module=rs_module,
                depth_scale_m=raw_source.runtime_metadata["depth_scale_m"],
                depth_intrinsics=raw_source.depth_intrinsics,
                depth_to_color_extrinsics=(
                    raw_source.depth_to_color_extrinsics
                ),
                R_tc=R_tc,
                T_tc=T_tc,
                thermal_K=thermal_K,
                thermal_D=thermal_D,
            )
            row.update(_projection_fields(sparse, thermal.frame))
            winner_by_source = {
                tuple(projected.source_depth_xy): (thermal_xy, projected)
                for thermal_xy, projected in sparse.winners
            }
            sparse_reason = _sparse_pair_rejection_reason(
                sparse,
                depth_pixels,
            )
            if sparse_reason is not None:
                row.update(
                    status="blocked",
                    reasons=[sparse_reason],
                )
                summary["blocked"] += 1
                reject_pinch_signal_quota(row)
                _write_jsonl(stream, row)
                show_preview_and_handle_key(row, raw.color_rgb, now_s)
                continue

            for tip, depth_pixel in zip(fingertips, depth_pixels, strict=True):
                thermal_xy, projected = winner_by_source[depth_pixel]
                tip_fields = {
                    "thermal_uv": list(projected.thermal_uv),
                    "thermal_pixel": list(thermal_xy),
                    "thermal_raw_count": int(
                        thermal.frame[thermal_xy[1], thermal_xy[0]]
                    ),
                }
                if pinch_signal_trial:
                    tip_fields.update(
                        _thermal_patch_statistics(
                            thermal.frame,
                            thermal_xy,
                        )
                    )
                tip.update(
                    tip_fields
                )
            thermal_pinch_center_uv = None
            if pinch_signal_trial:
                thermal_pixels = [
                    tuple(tip["thermal_pixel"]) for tip in fingertips
                ]
                row["thermal_patches_overlap"] = _thermal_patches_overlap(
                    thermal_pixels,
                    thermal.frame.shape,
                )
                thermal_pinch_center_uv = _thermal_pinch_center_uv(
                    fingertips
                )
                row["thermal_pinch_center_uv"] = list(
                    thermal_pinch_center_uv
                )
            if pinch_signal_trial and pinch_signal_state["phase"] == "warmup":
                pinch_signal_state = _pinch_signal_unlock(
                    pinch_signal_state
                )
                summary["pinch_signal_started"] = True
                row["pinch_signal"] = _pinch_signal_cue(
                    pinch_signal_state,
                    now_s,
                )
                emit_pinch_signal_transition(row["pinch_signal"])
            row.update(status="software_gate_accepted", reasons=[])
            summary["software_gate_accepted"] += 1
            if (
                pinch_signal_trial
                and row["pinch_signal"]["recording"]
            ):
                pinch_signal_state, quota_result = (
                    _pinch_signal_accept_sample(
                        pinch_signal_state,
                        thermal_pinch_center_uv,
                    )
                )
                row["pinch_signal"].update(quota_result)
                row["pinch_center_shift_px"] = quota_result[
                    "pinch_center_shift_px"
                ]
                if quota_result["group_completed"]:
                    summary["pinch_signal_valid_groups"] += 1
                emit_pinch_signal_transition(
                    _pinch_signal_cue(pinch_signal_state, now_s)
                )
            _write_jsonl(stream, row)
            show_preview_and_handle_key(row, raw.color_rgb, now_s)
        if pinch_signal_trial:
            _write_jsonl(
                stream,
                {
                    "row_type": "summary",
                    "status": (
                        "ok"
                        if summary["pinch_signal_protocol_completed"]
                        else "blocked"
                    ),
                    **summary,
                },
            )
    return summary


def _default_hands_factory():
    import mediapipe as mp

    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    stage = "Stage 1E" if args.pinch_signal_trial else "Stage 1D"
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        print(f"raw D435i source blocked: {exc}", file=sys.stderr)
        return 1
    if args.manual_ffc:
        print(
            "Running approved Pi C++ manual FFC before capture...",
            file=sys.stderr,
            flush=True,
        )
        try:
            ffc_output = _run_manual_ffc()
        except Exception as exc:
            print(f"{stage} manual FFC blocked: {exc!r}", file=sys.stderr)
            return 1
        print(ffc_output.strip(), file=sys.stderr, flush=True)
    mode = (
        "preview window enabled"
        if args.preview
        else "headless JSONL-only; no preview window"
    )
    print(
        f"{stage} hand shadow: {mode}; no robot, controller, or actuation.",
        file=sys.stderr,
        flush=True,
    )
    if args.pinch_signal_trial:
        print(
            "Self-paced 6-group tip-to-tip protocol; SPACE starts each "
            "phase, which holds until 5 valid samples or a 10 s timeout. "
            "Follow the preview cues exactly.",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            "Show the physical RIGHT hand near the image centre and move the "
            "thumb and index finger.",
            file=sys.stderr,
            flush=True,
        )
    print(
        f"Collecting {args.frames} attempts -> {args.output}",
        file=sys.stderr,
        flush=True,
    )
    try:
        summary = run_shadow(
            attempts=args.frames,
            output_path=args.output,
            rs_module=rs,
            raw_source_factory=lambda: RealSenseRawProjectorCamera(
                rs_module=rs
            ),
            thermal_source_factory=lambda: LeptonUDPSource(
                port=args.lepton_port
            ),
            hands_factory=_default_hands_factory,
            preview=args.preview,
            manual_ffc_before_start=args.manual_ffc,
            diagnose_inward_samples=args.diagnose_inward_samples,
            pinch_signal_trial=args.pinch_signal_trial,
        )
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"{stage} hand shadow blocked: {exc!r}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    if args.pinch_signal_trial:
        return 0 if summary["pinch_signal_protocol_completed"] else 1
    return 0 if summary["software_gate_accepted"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
