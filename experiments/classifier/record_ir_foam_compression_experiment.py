from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from ir_force.classifier.ir_capture import (
    OAKCameraSource,
    OAKFrameSample,
    OpenCVCameraSource,
)
from ir_force.classifier.ir_dataset import (
    _create_trial_paths_for_id,
    append_telemetry_row,
    ensure_fresh_trial,
)
from ir_force.classifier.ir_features import load_palette, palette_index_image
from ir_force.classifier.ir_flir_registration import (
    project_marker_observation,
    similarity_transform_from_markers,
    track_foam_regions,
)
from ir_force.classifier.ir_foam_compression import (
    FrozenThermalRegions,
    MarkerObservation,
    PixelROI,
    build_recording_plan,
    compression_percent,
    detect_centered_dark_marker_pair,
    detect_marker_pair,
    HoldToleranceGate,
    reference_normalized_features,
    StableCompressionGate,
    thermal_frame_hash,
)


DEFAULT_ROOT = "/home/zhuokai/hand-teleop/ir-camera-force/local/datasets/ir_foam_compression"
DEFAULT_PALETTE = "/home/zhuokai/hand-teleop/ir-camera-force/hardware/flirone-v4l2/palettes/Iron2.raw"


def _parse_roi(value: str) -> PixelROI:
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height")
    try:
        return PixelROI(*(int(part) for part in parts))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_rep(value: str) -> int | None:
    if value.strip().lower() == "auto":
        return None
    try:
        rep = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rep must be an integer from 1 to 99, or 'auto'") from exc
    if not 1 <= rep <= 99:
        raise argparse.ArgumentTypeError("rep must be in the range 1..99")
    return rep


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record a fixed-geometry foam-compression experiment with FLIR and OAK markers."
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--participant-id", required=True)
    parser.add_argument("--object-id", default="foam")
    parser.add_argument(
        "--rep",
        required=True,
        type=_parse_rep,
        metavar="REP|auto",
        help="trial repetition 1..99, or auto to select the next unused repetition",
    )
    parser.add_argument("--recording-index", required=True, type=int, choices=(1, 2, 3))
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--thermal", default="/dev/video21")
    parser.add_argument("--flir-visible", default="/dev/video20")
    parser.add_argument("--record-flir-visible", action="store_true")
    parser.add_argument(
        "--thermal-roi-tracking",
        choices=("fixed", "flir-visible-markers"),
        default="fixed",
        help="keep thermal ROIs fixed, or move foam-attached ROIs from FLIR-visible black-dot markers",
    )
    parser.add_argument("--flir-visible-left-marker-roi", type=_parse_roi)
    parser.add_argument("--flir-visible-right-marker-roi", type=_parse_roi)
    parser.add_argument("--flir-visible-marker-max-gray", type=int, default=90)
    parser.add_argument("--flir-visible-marker-min-area-px", type=int, default=20)
    parser.add_argument("--flir-visible-marker-max-area-px", type=int, default=2000)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--oak-fps", type=float, default=10.0)
    parser.add_argument("--palette", default=DEFAULT_PALETTE)
    parser.add_argument("--invert-palette", action="store_true")
    parser.add_argument("--thermal-foam-bbox", required=True, type=_parse_roi)
    parser.add_argument("--thermal-foam-roi", required=True, type=_parse_roi)
    parser.add_argument("--thermal-left-contact-roi", required=True, type=_parse_roi)
    parser.add_argument("--thermal-right-contact-roi", required=True, type=_parse_roi)
    parser.add_argument("--thermal-background-roi", required=True, type=_parse_roi)
    parser.add_argument("--thermal-room-reference-roi", required=True, type=_parse_roi)
    parser.add_argument("--thermal-warm-reference-roi", required=True, type=_parse_roi)
    parser.add_argument("--oak-left-marker-roi", required=True, type=_parse_roi)
    parser.add_argument("--oak-right-marker-roi", required=True, type=_parse_roi)
    parser.add_argument("--target-tolerance-pct", type=float, default=3.0)
    parser.add_argument("--release-tolerance-pct", type=float, default=5.0)
    parser.add_argument("--gate-stable-s", type=float, default=1.0)
    parser.add_argument("--max-hold-gap-s", type=float, default=0.5)
    parser.add_argument("--max-reach-s", type=float, default=15.0)
    parser.add_argument("--d0-s", type=float, default=10.0)
    parser.add_argument(
        "--d0-settle-s",
        type=float,
        default=2.0,
        help="initial released-camera settling time to retain in raw data but exclude from d0",
    )
    parser.add_argument("--max-d0-relative-span", type=float, default=0.05)
    parser.add_argument("--preflight-s", type=float, default=5.0)
    parser.add_argument(
        "--preflight-settle-s",
        type=float,
        default=2.0,
        help="initial OAK settling time to show in preview but exclude from marker preflight checks",
    )
    parser.add_argument("--marker-max-gray", type=int, default=110)
    parser.add_argument("--marker-min-area-px", type=int, default=40)
    parser.add_argument("--min-reference-span", type=float, default=5.0)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--no-hand-geometry", dest="track_hand_geometry", action="store_false")
    parser.set_defaults(track_hand_geometry=True)
    parser.add_argument("--no-preview", dest="show_preview", action="store_false")
    parser.set_defaults(show_preview=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--notes", default="")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--yes", action="store_true", help="skip the final YES prompt")
    parser.add_argument("--no-beep", action="store_true")
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.thermal_roi_tracking == "flir-visible-markers":
        if args.flir_visible_left_marker_roi is None or args.flir_visible_right_marker_roi is None:
            parser.error(
                "--thermal-roi-tracking flir-visible-markers requires both "
                "--flir-visible-left-marker-roi and --flir-visible-right-marker-roi"
            )
        args.record_flir_visible = True
    return args


def _slug(value: str) -> str:
    return "-".join(part for part in "".join(char.lower() if char.isalnum() else " " for char in value).split())


def _trial_prefix(args: argparse.Namespace) -> str:
    return f"foam-compression_{_slug(args.session_id)}_{_slug(args.object_id)}_{_slug(args.participant_id)}"


def _trial_id(args: argparse.Namespace) -> str:
    if args.rep is None:
        raise ValueError("rep must be resolved before building a trial id")
    return f"{_trial_prefix(args)}_rep{args.rep:02d}"


def _next_available_rep(args: argparse.Namespace) -> int:
    trial_root = Path(args.root) / "trials"
    prefix = _trial_prefix(args)
    for rep in range(1, 100):
        if not (trial_root / f"{prefix}_rep{rep:02d}").exists():
            return rep
    raise RuntimeError(f"no unused rep remains for {prefix}; use a new session id")


def _resolve_auto_rep(args: argparse.Namespace) -> int:
    if args.rep is None:
        args.rep = _next_available_rep(args)
    return args.rep


def _reference_span_is_adequate(reference_span: float, *, min_span: float) -> bool:
    return math.isfinite(reference_span) and abs(reference_span) >= min_span


def _d0_relative_span(distances: list[float]) -> float:
    """Return the robust p95-p5 spread of a release calibration, relative to its median."""
    if not distances:
        return float("inf")
    values = np.asarray(distances, dtype=np.float64)
    center = float(np.median(values))
    if not math.isfinite(center) or center <= 0.0:
        return float("inf")
    spread = float(np.percentile(values, 95.0) - np.percentile(values, 5.0))
    return spread / center


def _d0_is_stable(distances: list[float], *, max_relative_span: float) -> bool:
    return math.isfinite(_d0_relative_span(distances)) and _d0_relative_span(distances) <= max_relative_span


def _d0_post_settle_distances(samples: list[tuple[float, float]], *, settle_s: float) -> list[float]:
    """Keep only valid d0 marker distances captured after camera settling."""
    return [distance for elapsed_s, distance in samples if elapsed_s >= settle_s]


def _preflight_marker_stats(
    samples: list[tuple[float, bool]],
    *,
    settle_s: float,
    end_elapsed_s: float,
) -> tuple[int, int, float]:
    """Summarize marker availability after the camera's initial settling interval."""
    evaluated = [(elapsed_s, detected) for elapsed_s, detected in samples if elapsed_s >= settle_s]
    marker_count = sum(detected for _elapsed_s, detected in evaluated)
    missing_since: float | None = None
    longest_missing_s = 0.0
    for elapsed_s, detected in evaluated:
        if detected:
            if missing_since is not None:
                longest_missing_s = max(longest_missing_s, elapsed_s - missing_since)
                missing_since = None
        elif missing_since is None:
            missing_since = elapsed_s
    if missing_since is not None:
        longest_missing_s = max(longest_missing_s, end_elapsed_s - missing_since)
    return len(evaluated), marker_count, longest_missing_s


def _step_target_tolerance(step, args: argparse.Namespace) -> float:
    return args.release_tolerance_pct if step.state == "R" else args.target_tolerance_pct


def _regions_from_args(args: argparse.Namespace) -> FrozenThermalRegions:
    return FrozenThermalRegions(
        foam_bbox=args.thermal_foam_bbox,
        foam_center=args.thermal_foam_roi,
        left_contact=args.thermal_left_contact_roi,
        right_contact=args.thermal_right_contact_roi,
        background=args.thermal_background_roi,
        room_reference=args.thermal_room_reference_roi,
        warm_reference=args.thermal_warm_reference_roi,
    )


def _step_metadata(step) -> dict[str, object]:
    return {
        "block": step.block,
        "phase": step.phase,
        "state": step.state,
        "target_compression_pct": step.target_compression_pct,
        "hold_s": step.hold_s,
        "sequence_id": step.sequence_id,
        "step_index": step.step_index,
        "pulse_index": step.pulse_index,
        "name": step.name,
    }


def _prepare_trial(args: argparse.Namespace):
    _resolve_auto_rep(args)
    if not 1 <= args.rep <= 99:
        raise ValueError("rep must be in the range 1..99")
    if args.fps <= 0 or args.oak_fps <= 0:
        raise ValueError("fps values must be positive")
    if (
        args.max_reach_s <= 0
        or args.d0_s <= 0
        or not 0 <= args.d0_settle_s < args.d0_s
        or args.preflight_s <= 0
        or not 0 <= args.preflight_settle_s < args.preflight_s
        or args.min_reference_span <= 0
        or args.max_d0_relative_span <= 0
        or args.target_tolerance_pct <= 0
        or args.release_tolerance_pct <= 0
        or args.max_hold_gap_s < 0
    ):
        raise ValueError("capture durations and tolerance thresholds must be positive")
    regions = _regions_from_args(args)
    issues = regions.preflight_issues((128, 160))
    if issues:
        raise ValueError("invalid frozen thermal regions: " + "; ".join(issues))
    plan = build_recording_plan(args.recording_index)
    paths = _create_trial_paths_for_id(Path(args.root), _trial_id(args))
    if not args.append:
        ensure_fresh_trial(paths)
    metadata = {
        "trial_id": paths.trial_id,
        "experiment_kind": "fixed_geometry_foam_compression",
        "objective_force_measurement": False,
        "compression_reference": "oak_black_marker_distance",
        "compression_formula": "100 * (d0_px - d_px) / d0_px",
        "primary_ir_feature": "foam_center_norm",
        "primary_ir_feature_definition": "(foam_center_palette_median - room_reference_palette_median) / (warm_reference_palette_median - room_reference_palette_median)",
        "thermal_stream_kind": "colorized_relative_intensity",
        "thermal_path": args.thermal,
        "flir_visible_path": args.flir_visible,
        "record_flir_visible": bool(args.record_flir_visible),
        "thermal_roi_tracking": {
            "mode": args.thermal_roi_tracking,
            "alignment_assumption": (
                "frame_normalized_visible_to_thermal"
                if args.thermal_roi_tracking == "flir-visible-markers"
                else "not_used"
            ),
            "flir_visible_marker_regions": (
                {
                    "left": args.flir_visible_left_marker_roi.as_list(),
                    "right": args.flir_visible_right_marker_roi.as_list(),
                }
                if args.thermal_roi_tracking == "flir-visible-markers"
                else None
            ),
            "marker_max_gray": args.flir_visible_marker_max_gray,
            "marker_min_area_px": args.flir_visible_marker_min_area_px,
            "marker_max_area_px": args.flir_visible_marker_max_area_px,
            "moving_regions": ["foam_bbox", "foam_center", "left_contact", "right_contact"],
            "fixed_regions": ["background", "room_reference", "warm_reference"],
        },
        "fps": args.fps,
        "oak_fps": args.oak_fps,
        "oak_rgb_size": [640, 480],
        "oak_depth_unit": "millimeter",
        "palette_path": args.palette,
        "invert_palette": bool(args.invert_palette),
        "thermal_regions": regions.metadata(),
        "oak_marker_regions": {
            "left": args.oak_left_marker_roi.as_list(),
            "right": args.oak_right_marker_roi.as_list(),
        },
        "target_tolerance_pct": args.target_tolerance_pct,
        "release_tolerance_pct": args.release_tolerance_pct,
        "gate_stable_s": args.gate_stable_s,
        "max_hold_gap_s": args.max_hold_gap_s,
        "max_reach_s": args.max_reach_s,
        "d0_s": args.d0_s,
        "d0_settle_s": args.d0_settle_s,
        "max_d0_relative_span": args.max_d0_relative_span,
        "marker_max_gray": args.marker_max_gray,
        "marker_min_area_px": args.marker_min_area_px,
        "preflight_s": args.preflight_s,
        "preflight_settle_s": args.preflight_settle_s,
        "min_reference_span": args.min_reference_span,
        "recording_index": args.recording_index,
        "recording_plan": [_step_metadata(step) for step in plan],
        "session_id": args.session_id,
        "participant_id": args.participant_id,
        "object_id": args.object_id,
        "rep": args.rep,
        "notes": args.notes,
        "analysis_preregistration": {
            "primary": "foam_center_norm",
            "secondary": ["left_contact_norm", "right_contact_norm", "background_norm"],
            "deduplicate": "thermal_frame_sha1",
            "stable_window": "last 3 seconds of valid gated hold",
            "not_permitted": ["post_hoc_roi_selection", "post_hoc_best_feature_selection"],
        },
    }
    paths.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return paths, plan, regions


def _frame_rows(
    *,
    frame_index: int,
    protocol_elapsed_s: float,
    thermal_timestamp: float,
    oak_timestamp: float,
    step,
    capture_phase: str,
    step_elapsed_s: float,
    action_attempt: int,
    marker: MarkerObservation | None,
    d0_px: float | None,
    gate_stable_s: float,
    regions: FrozenThermalRegions,
    scalar: np.ndarray,
    thermal_sha1: str,
    frozen_frame: bool,
    hand_metrics: dict[str, float | bool | None] | None = None,
    marker_depth_mm: tuple[float | None, float | None] = (None, None),
    target_tolerance_pct: float = 3.0,
    visible_timestamp: float | None = None,
    active_regions: FrozenThermalRegions | None = None,
    registration: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    active_regions = regions if active_regions is None else active_regions
    registration = registration or {}
    visible_marker = registration.get("visible_marker")
    if visible_marker is not None and not isinstance(visible_marker, MarkerObservation):
        raise TypeError("registration visible_marker must be a MarkerObservation")
    distance = marker.distance_px if marker is not None else None
    compression = compression_percent(distance, d0_px=d0_px) if distance is not None and d0_px is not None else None
    gate_in_range = (
        compression is not None
        and abs(compression - step.target_compression_pct) <= target_tolerance_pct
    )
    features = reference_normalized_features(
        scalar,
        active_regions,
        strict_preflight=active_regions is regions,
    )
    midpoint_x, midpoint_y = marker.midpoint_xy if marker is not None else (None, None)
    hand_metrics = hand_metrics or {}
    telemetry = {
        "frame": frame_index,
        "t_capture": protocol_elapsed_s,
        "t_thermal": thermal_timestamp,
        "t_oak": oak_timestamp,
        "t_flir_visible": visible_timestamp,
        "block": step.block,
        "phase": capture_phase,
        "state": step.state,
        "target_compression_pct": step.target_compression_pct,
        "sequence_id": step.sequence_id,
        "step_index": step.step_index,
        "step_name": step.name,
        "pulse_index": step.pulse_index,
        "action_attempt": action_attempt,
        "step_elapsed_s": step_elapsed_s,
        "marker_detected": marker is not None,
        "marker_left_x": marker.left_xy[0] if marker is not None else None,
        "marker_left_y": marker.left_xy[1] if marker is not None else None,
        "marker_right_x": marker.right_xy[0] if marker is not None else None,
        "marker_right_y": marker.right_xy[1] if marker is not None else None,
        "marker_distance_px": distance,
        "d0_px": d0_px,
        "compression_pct": compression,
        "foam_center_x": midpoint_x,
        "foam_center_y": midpoint_y,
        "foam_rotation_deg": marker.angle_deg if marker is not None else None,
        "left_marker_depth_mm": marker_depth_mm[0],
        "right_marker_depth_mm": marker_depth_mm[1],
        "oak_hand_detected": hand_metrics.get("oak_hand_detected", False),
        "oak_hand_area_px": hand_metrics.get("oak_hand_area_px"),
        "oak_hand_center_x": hand_metrics.get("oak_hand_center_x"),
        "oak_hand_center_y": hand_metrics.get("oak_hand_center_y"),
        "hand_to_foam_gap_px": hand_metrics.get("hand_to_foam_gap_px"),
        "thumb_index_aperture_px": hand_metrics.get("thumb_index_aperture_px"),
        "flir_visible_marker_detected": visible_marker is not None,
        "flir_visible_marker_left_x": visible_marker.left_xy[0] if visible_marker is not None else None,
        "flir_visible_marker_left_y": visible_marker.left_xy[1] if visible_marker is not None else None,
        "flir_visible_marker_right_x": visible_marker.right_xy[0] if visible_marker is not None else None,
        "flir_visible_marker_right_y": visible_marker.right_xy[1] if visible_marker is not None else None,
        "thermal_roi_tracking_mode": registration.get("mode", "fixed"),
        "thermal_roi_registration_valid": bool(registration.get("valid", False)),
        "thermal_roi_scale": registration.get("scale"),
        "thermal_roi_rotation_deg": registration.get("rotation_deg"),
        "thermal_roi_translation_x": registration.get("translation_x"),
        "thermal_roi_translation_y": registration.get("translation_y"),
        "thermal_foam_bbox_roi_x": active_regions.foam_bbox.x,
        "thermal_foam_bbox_roi_y": active_regions.foam_bbox.y,
        "thermal_foam_bbox_roi_width": active_regions.foam_bbox.width,
        "thermal_foam_bbox_roi_height": active_regions.foam_bbox.height,
        "thermal_foam_center_roi_x": active_regions.foam_center.x,
        "thermal_foam_center_roi_y": active_regions.foam_center.y,
        "thermal_foam_center_roi_width": active_regions.foam_center.width,
        "thermal_foam_center_roi_height": active_regions.foam_center.height,
        "thermal_left_contact_roi_x": active_regions.left_contact.x,
        "thermal_left_contact_roi_y": active_regions.left_contact.y,
        "thermal_right_contact_roi_x": active_regions.right_contact.x,
        "thermal_right_contact_roi_y": active_regions.right_contact.y,
        "gate_in_range": gate_in_range,
        "gate_stable_s": gate_stable_s,
        "frozen_frame_flag": frozen_frame,
    }
    feature_row = {
        "frame": frame_index,
        "timestamp": protocol_elapsed_s,
        "block": step.block,
        "phase": capture_phase,
        "state": step.state,
        "target_compression_pct": step.target_compression_pct,
        "sequence_id": step.sequence_id,
        "step_index": step.step_index,
        "action_attempt": action_attempt,
        "thermal_frame_sha1": thermal_sha1,
        "frozen_frame_flag": frozen_frame,
        "thermal_roi_tracking_mode": registration.get("mode", "fixed"),
        "thermal_roi_registration_valid": bool(registration.get("valid", False)),
        "thermal_foam_center_roi_x": active_regions.foam_center.x,
        "thermal_foam_center_roi_y": active_regions.foam_center.y,
        "thermal_foam_center_roi_width": active_regions.foam_center.width,
        "thermal_foam_center_roi_height": active_regions.foam_center.height,
        **features,
    }
    return telemetry, feature_row


@dataclass
class _CaptureRuntime:
    paths: object
    regions: FrozenThermalRegions
    thermal: object
    oak: object
    visible: object | None
    left_marker_roi: PixelROI
    right_marker_roi: PixelROI
    palette: np.ndarray | None
    invert_palette: bool
    marker_max_gray: int
    marker_min_area_px: int
    protocol_start: float
    hand_tracker: object | None = None
    target_tolerance_pct: float = 3.0
    roi_tracking_mode: str = "fixed"
    visible_left_marker_roi: PixelROI | None = None
    visible_right_marker_roi: PixelROI | None = None
    visible_marker_max_gray: int = 110
    visible_marker_min_area_px: int = 40
    visible_marker_max_area_px: int = 2_000
    thermal_marker_baseline: MarkerObservation | None = None
    frame_index: int = 0
    previous_thermal: np.ndarray | None = None
    last_thermal: np.ndarray | None = None
    last_oak: np.ndarray | None = None
    last_visible: np.ndarray | None = None
    last_visible_marker: MarkerObservation | None = None
    last_feature_regions: FrozenThermalRegions | None = None


@dataclass(frozen=True)
class _CapturedSample:
    protocol_elapsed_s: float
    marker: MarkerObservation | None
    compression_pct: float | None
    visible_marker: MarkerObservation | None
    thermal_roi_registration_valid: bool


class _HandGeometryTracker:
    """OAK-side nuisance covariates; compression remains marker-distance based."""

    def __init__(self) -> None:
        try:
            import mediapipe as mp
        except Exception as exc:  # pragma: no cover - host camera/runtime dependency
            raise RuntimeError(
                "MediaPipe is required to save hand geometry; use the configured local venv or pass "
                "--no-hand-geometry only for a preflight check."
            ) from exc
        self._cv2 = cv2
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

    def close(self) -> None:
        self._hands.close()

    def metrics(self, frame: np.ndarray, marker: MarkerObservation | None) -> dict[str, float | bool | None]:
        result = self._hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not result.multi_hand_landmarks:
            return {
                "oak_hand_detected": False,
                "oak_hand_area_px": None,
                "oak_hand_center_x": None,
                "oak_hand_center_y": None,
                "hand_to_foam_gap_px": None,
                "thumb_index_aperture_px": None,
            }
        height, width = frame.shape[:2]
        points = np.asarray(
            [(landmark.x * width, landmark.y * height) for landmark in result.multi_hand_landmarks[0].landmark],
            dtype=np.float32,
        )
        hull = cv2.convexHull(points)
        moments = cv2.moments(hull)
        if moments["m00"]:
            center_x = moments["m10"] / moments["m00"]
            center_y = moments["m01"] / moments["m00"]
        else:
            center_x, center_y = points.mean(axis=0)
        if marker is None:
            gap = None
        else:
            start = np.asarray(marker.left_xy, dtype=np.float32)
            end = np.asarray(marker.right_xy, dtype=np.float32)
            segment = end - start
            denominator = float(np.dot(segment, segment))
            if denominator == 0:
                gap = float(np.linalg.norm(points - start, axis=1).min())
            else:
                projection = np.clip(((points - start) @ segment) / denominator, 0.0, 1.0)
                closest = start + projection[:, None] * segment
                gap = float(np.linalg.norm(points - closest, axis=1).min())
        return {
            "oak_hand_detected": True,
            "oak_hand_area_px": float(cv2.contourArea(hull)),
            "oak_hand_center_x": float(center_x),
            "oak_hand_center_y": float(center_y),
            "hand_to_foam_gap_px": gap,
            "thumb_index_aperture_px": float(np.linalg.norm(points[4] - points[8])),
        }


def _marker_depth_mm(depth: np.ndarray, point: tuple[float, float] | None, radius_px: int = 2) -> float | None:
    if point is None:
        return None
    x, y = (int(round(value)) for value in point)
    y0, y1 = max(0, y - radius_px), min(depth.shape[0], y + radius_px + 1)
    x0, x1 = max(0, x - radius_px), min(depth.shape[1], x + radius_px + 1)
    values = depth[y0:y1, x0:x1]
    valid = values[values > 0]
    return float(np.median(valid)) if valid.size else None


def _write_frame(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"could not write {path}")


def _thermal_scalar(frame: np.ndarray, palette: np.ndarray | None, invert_palette: bool) -> np.ndarray:
    if palette is not None:
        return palette_index_image(frame, palette, invert=invert_palette)
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return frame.astype(np.float32)


def _detect_visible_marker(runtime: _CaptureRuntime, frame: np.ndarray | None) -> MarkerObservation | None:
    if (
        frame is None
        or runtime.roi_tracking_mode != "flir-visible-markers"
        or runtime.visible_left_marker_roi is None
        or runtime.visible_right_marker_roi is None
    ):
        return None
    return detect_centered_dark_marker_pair(
        frame,
        left_roi=runtime.visible_left_marker_roi,
        right_roi=runtime.visible_right_marker_roi,
        max_gray=runtime.visible_marker_max_gray,
        min_area_px=runtime.visible_marker_min_area_px,
        max_area_px=runtime.visible_marker_max_area_px,
    )


def _thermal_marker_baseline_from_visible(
    markers: list[MarkerObservation],
    *,
    visible_shape: tuple[int, int],
    thermal_shape: tuple[int, int],
) -> MarkerObservation:
    """Project the per-coordinate median of released-state RGB marker detections."""
    if not markers:
        raise ValueError("at least one visible marker observation is required")
    median_marker = MarkerObservation(
        left_xy=(
            float(median(marker.left_xy[0] for marker in markers)),
            float(median(marker.left_xy[1] for marker in markers)),
        ),
        right_xy=(
            float(median(marker.right_xy[0] for marker in markers)),
            float(median(marker.right_xy[1] for marker in markers)),
        ),
        left_area_px=float(median(marker.left_area_px for marker in markers)),
        right_area_px=float(median(marker.right_area_px for marker in markers)),
    )
    return project_marker_observation(
        median_marker,
        source_shape=visible_shape,
        destination_shape=thermal_shape,
    )


def _registered_regions_from_visible_marker(
    runtime: _CaptureRuntime,
    *,
    visible_marker: MarkerObservation | None,
    visible_shape: tuple[int, int] | None,
    thermal_shape: tuple[int, int],
) -> tuple[FrozenThermalRegions, dict[str, object]]:
    registration: dict[str, object] = {
        "mode": runtime.roi_tracking_mode,
        "valid": False,
        "scale": None,
        "rotation_deg": None,
        "translation_x": None,
        "translation_y": None,
        "visible_marker": visible_marker,
    }
    if runtime.roi_tracking_mode != "flir-visible-markers":
        return runtime.regions, registration
    if visible_marker is None or visible_shape is None or runtime.thermal_marker_baseline is None:
        return runtime.regions, registration
    current_thermal_marker = project_marker_observation(
        visible_marker,
        source_shape=visible_shape,
        destination_shape=thermal_shape,
    )
    try:
        transform = similarity_transform_from_markers(runtime.thermal_marker_baseline, current_thermal_marker)
        active_regions = track_foam_regions(runtime.regions, transform, frame_shape=thermal_shape)
    except ValueError:
        return runtime.regions, registration
    if active_regions.sampling_issues(thermal_shape):
        return runtime.regions, registration
    registration.update(
        {
            "valid": True,
            "scale": transform.scale,
            "rotation_deg": transform.angle_deg,
            "translation_x": transform.translation_xy[0],
            "translation_y": transform.translation_xy[1],
        }
    )
    return active_regions, registration


def _capture_sample(
    runtime: _CaptureRuntime,
    *,
    step,
    capture_phase: str,
    step_elapsed_s: float,
    action_attempt: int,
    d0_px: float | None,
    gate_stable_s: float,
    now: float | None = None,
) -> _CapturedSample:
    now = time.perf_counter() if now is None else now
    thermal_sample = runtime.thermal.read()
    oak_sample = runtime.oak.read()
    if not isinstance(oak_sample, OAKFrameSample):
        raise TypeError("OAK capture source must return OAKFrameSample")
    visible_sample = runtime.visible.read() if runtime.visible is not None else None
    frame_index = runtime.frame_index
    _write_frame(runtime.paths.thermal_dir / f"frame_{frame_index:06d}.png", thermal_sample.frame)
    _write_frame(runtime.paths.root / "oak_rgb" / f"frame_{frame_index:06d}.png", oak_sample.frame)
    _write_frame(runtime.paths.root / "oak_depth" / f"frame_{frame_index:06d}.png", oak_sample.depth)
    if visible_sample is not None:
        _write_frame(runtime.paths.flir_visible_dir / f"frame_{frame_index:06d}.png", visible_sample.frame)

    marker = detect_marker_pair(
        oak_sample.frame,
        left_roi=runtime.left_marker_roi,
        right_roi=runtime.right_marker_roi,
        max_gray=runtime.marker_max_gray,
        min_area_px=runtime.marker_min_area_px,
    )
    visible_marker = _detect_visible_marker(runtime, visible_sample.frame if visible_sample is not None else None)
    active_regions, registration = _registered_regions_from_visible_marker(
        runtime,
        visible_marker=visible_marker,
        visible_shape=visible_sample.frame.shape[:2] if visible_sample is not None else None,
        thermal_shape=thermal_sample.frame.shape[:2],
    )
    hand_metrics = runtime.hand_tracker.metrics(oak_sample.frame, marker) if runtime.hand_tracker is not None else None
    marker_depth_mm = (
        _marker_depth_mm(oak_sample.depth, marker.left_xy if marker is not None else None),
        _marker_depth_mm(oak_sample.depth, marker.right_xy if marker is not None else None),
    )
    scalar = _thermal_scalar(thermal_sample.frame, runtime.palette, runtime.invert_palette)
    frozen = runtime.previous_thermal is not None and np.array_equal(runtime.previous_thermal, thermal_sample.frame)
    telemetry, features = _frame_rows(
        frame_index=frame_index,
        protocol_elapsed_s=now - runtime.protocol_start,
        thermal_timestamp=thermal_sample.t,
        oak_timestamp=oak_sample.t,
        visible_timestamp=visible_sample.t if visible_sample is not None else None,
        step=step,
        capture_phase=capture_phase,
        step_elapsed_s=step_elapsed_s,
        action_attempt=action_attempt,
        marker=marker,
        d0_px=d0_px,
        gate_stable_s=gate_stable_s,
        regions=runtime.regions,
        active_regions=active_regions,
        scalar=scalar,
        thermal_sha1=thermal_frame_hash(thermal_sample.frame),
        frozen_frame=frozen,
        hand_metrics=hand_metrics,
        marker_depth_mm=marker_depth_mm,
        target_tolerance_pct=runtime.target_tolerance_pct,
        registration=registration,
    )
    append_telemetry_row(runtime.paths.telemetry_csv, telemetry)
    append_telemetry_row(runtime.paths.root / "frame_features.csv", features)
    runtime.previous_thermal = thermal_sample.frame.copy()
    runtime.last_thermal = thermal_sample.frame.copy()
    runtime.last_oak = oak_sample.frame.copy()
    runtime.last_visible = visible_sample.frame.copy() if visible_sample is not None else None
    runtime.last_visible_marker = visible_marker
    runtime.last_feature_regions = active_regions
    runtime.frame_index += 1
    return _CapturedSample(
        protocol_elapsed_s=float(telemetry["t_capture"]),
        marker=marker,
        compression_pct=telemetry["compression_pct"],
        visible_marker=visible_marker,
        thermal_roi_registration_valid=bool(registration["valid"]),
    )


EVENT_FIELDS = (
    "timestamp",
    "event_type",
    "reason",
    "block",
    "phase",
    "state",
    "target_compression_pct",
    "sequence_id",
    "step_index",
    "step_name",
    "pulse_index",
    "action_attempt",
)

STATE_INSTRUCTIONS = {
    "R": "fully release: both fingers at least 10 mm from foam; keep the hand in view and wrist fixed",
    "N": "near but no contact: fingers about 2-3 mm from foam; do not touch",
    "C0": "just contact: compression no more than 2%",
    "C10": "compress foam width by 10% using the OAK progress bar",
    "C20": "compress foam width by 20% using the OAK progress bar",
    "C30": "compress foam width by 30% using the OAK progress bar",
}


def _append_event(paths, *, timestamp: float, event_type: str, reason: str, step, action_attempt: int) -> None:
    row = {
        "timestamp": round(timestamp, 6),
        "event_type": event_type,
        "reason": reason,
        "block": step.block,
        "phase": step.phase,
        "state": step.state,
        "target_compression_pct": step.target_compression_pct,
        "sequence_id": step.sequence_id,
        "step_index": step.step_index,
        "step_name": step.name,
        "pulse_index": step.pulse_index,
        "action_attempt": action_attempt,
    }
    assert tuple(row) == EVENT_FIELDS
    append_telemetry_row(paths.root / "events.csv", row)


def _load_palette(path_text: str) -> np.ndarray | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"palette does not exist: {path}")
    palette = load_palette(path)[:, ::-1].copy()  # OpenCV frames are BGR.
    return palette


def _draw_thermal_regions(frame: np.ndarray, regions: FrozenThermalRegions) -> np.ndarray:
    out = frame.copy()
    colors = {
        "foam_bbox": (0, 255, 0),
        "foam_center": (255, 255, 255),
        "left_contact": (255, 0, 0),
        "right_contact": (0, 0, 255),
        "background": (255, 255, 0),
        "room_reference": (255, 0, 255),
        "warm_reference": (0, 255, 255),
    }
    for name, roi in (
        ("foam_bbox", regions.foam_bbox),
        ("foam_center", regions.foam_center),
        ("left_contact", regions.left_contact),
        ("right_contact", regions.right_contact),
        ("background", regions.background),
        ("room_reference", regions.room_reference),
        ("warm_reference", regions.warm_reference),
    ):
        cv2.rectangle(out, (roi.x, roi.y), (roi.x_end - 1, roi.y_end - 1), colors[name], 1)
    cv2.line(out, (0, regions.overlay_y_exclusive), (out.shape[1] - 1, regions.overlay_y_exclusive), (0, 0, 255), 1)
    return out


def _preview(runtime: _CaptureRuntime, *, step, compression_pct: float | None, gate_stable_s: float, enabled: bool) -> bool:
    if not enabled or runtime.last_thermal is None or runtime.last_oak is None:
        return True
    display_regions = runtime.last_feature_regions or runtime.regions
    thermal = cv2.resize(_draw_thermal_regions(runtime.last_thermal, display_regions), (640, 512), interpolation=cv2.INTER_NEAREST)
    oak = runtime.last_oak.copy()
    for color, roi in (((255, 0, 0), runtime.left_marker_roi), ((0, 0, 255), runtime.right_marker_roi)):
        cv2.rectangle(oak, (roi.x, roi.y), (roi.x_end - 1, roi.y_end - 1), color, 2)
    display_pct = float("nan") if compression_pct is None else compression_pct
    text = f"{step.state} target {step.target_compression_pct:.0f}% | now {display_pct:.1f}% | stable {gate_stable_s:.1f}s"
    cv2.putText(oak, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2, cv2.LINE_AA)
    bar_x, bar_y, bar_w, bar_h = 24, oak.shape[0] - 42, 280, 18
    cv2.rectangle(oak, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (230, 230, 230), 1)
    target_x = bar_x + round(bar_w * step.target_compression_pct / 35.0)
    cv2.line(oak, (target_x, bar_y - 4), (target_x, bar_y + bar_h + 4), (0, 0, 255), 2)
    if compression_pct is not None:
        current_x = bar_x + round(bar_w * np.clip(compression_pct, 0.0, 35.0) / 35.0)
        cv2.rectangle(oak, (bar_x, bar_y), (current_x, bar_y + bar_h), (0, 180, 0), -1)
    oak = cv2.resize(oak, (640, 480), interpolation=cv2.INTER_AREA)
    oak = cv2.copyMakeBorder(oak, 16, 16, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.imshow("foam compression: OAK (left) / FLIR frozen ROIs (right)", np.hstack((oak, thermal)))
    key = cv2.waitKey(1) & 0xFF
    return key not in (27, ord("q"))


def _preflight_output_dir(args: argparse.Namespace) -> Path:
    directory = Path(args.root) / "preflight" / _trial_id(args)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _reference_saturation_fraction(scalar: np.ndarray, roi: PixelROI, scalar_max: float) -> float:
    values = scalar[roi.slices()]
    return float(np.mean((values <= 1.0) | (values >= scalar_max - 1.0)))


def _run_preflight(
    *,
    args: argparse.Namespace,
    regions: FrozenThermalRegions,
    thermal,
    oak,
    visible=None,
    palette: np.ndarray | None,
) -> tuple[dict[str, object], Path]:
    output_dir = _preflight_output_dir(args)
    issues = regions.preflight_issues((128, 160))
    period = 1.0 / args.fps
    started = time.perf_counter()
    deadline = started + args.preflight_s
    next_capture = started
    marker_samples: list[tuple[float, bool]] = []
    visible_marker_samples: list[tuple[float, bool]] = []
    last_thermal = None
    last_oak = None
    last_marker = None
    last_visible = None
    last_visible_marker = None
    while next_capture <= deadline + 1e-9:
        now = time.perf_counter()
        if now < next_capture:
            time.sleep(next_capture - now)
        thermal_sample = thermal.read()
        oak_sample = oak.read()
        visible_sample = visible.read() if visible is not None else None
        if not isinstance(oak_sample, OAKFrameSample):
            raise TypeError("OAK capture source must return OAKFrameSample")
        marker = detect_marker_pair(
            oak_sample.frame,
            left_roi=args.oak_left_marker_roi,
            right_roi=args.oak_right_marker_roi,
            max_gray=args.marker_max_gray,
            min_area_px=args.marker_min_area_px,
        )
        marker_samples.append((time.perf_counter() - started, marker is not None))
        if args.thermal_roi_tracking == "flir-visible-markers":
            visible_marker = detect_centered_dark_marker_pair(
                visible_sample.frame if visible_sample is not None else np.empty((0, 0, 3), dtype=np.uint8),
                left_roi=args.flir_visible_left_marker_roi,
                right_roi=args.flir_visible_right_marker_roi,
                max_gray=args.flir_visible_marker_max_gray,
                min_area_px=args.flir_visible_marker_min_area_px,
                max_area_px=args.flir_visible_marker_max_area_px,
            )
            visible_marker_samples.append((time.perf_counter() - started, visible_marker is not None))
            last_visible_marker = visible_marker
        last_thermal, last_oak, last_marker = thermal_sample.frame, oak_sample.frame, marker
        last_visible = visible_sample.frame if visible_sample is not None else None
        preview_runtime = _CaptureRuntime(
            paths=None,
            regions=regions,
            thermal=None,
            oak=None,
            visible=None,
            left_marker_roi=args.oak_left_marker_roi,
            right_marker_roi=args.oak_right_marker_roi,
            palette=palette,
            invert_palette=args.invert_palette,
            marker_max_gray=args.marker_max_gray,
            marker_min_area_px=args.marker_min_area_px,
            protocol_start=started,
            last_thermal=thermal_sample.frame,
            last_oak=oak_sample.frame,
            target_tolerance_pct=args.target_tolerance_pct,
        )
        preview_step = build_recording_plan(args.recording_index)[0]
        compression = None
        if marker is not None:
            compression = 0.0
        if not _preview(preview_runtime, step=preview_step, compression_pct=compression, gate_stable_s=0.0, enabled=args.show_preview):
            issues.append("operator aborted preflight preview")
            break
        next_capture += period
    frame_count, marker_count, longest_missing_s = _preflight_marker_stats(
        marker_samples,
        settle_s=args.preflight_settle_s,
        end_elapsed_s=time.perf_counter() - started,
    )
    marker_rate = marker_count / frame_count if frame_count else 0.0
    if marker_rate < 0.95:
        issues.append(f"marker detection rate {marker_rate:.2%} is below 95%")
    if longest_missing_s > 0.5:
        issues.append(f"marker tracking was lost continuously for {longest_missing_s:.3f}s")
    visible_marker_rate = None
    visible_longest_missing_s = None
    visible_frame_count = 0
    visible_marker_count = 0
    if args.thermal_roi_tracking == "flir-visible-markers":
        visible_frame_count, visible_marker_count, visible_longest_missing_s = _preflight_marker_stats(
            visible_marker_samples,
            settle_s=args.preflight_settle_s,
            end_elapsed_s=time.perf_counter() - started,
        )
        visible_marker_rate = visible_marker_count / visible_frame_count if visible_frame_count else 0.0
        if visible_marker_rate < 0.95:
            issues.append(f"FLIR-visible marker detection rate {visible_marker_rate:.2%} is below 95%")
        if visible_longest_missing_s > 0.5:
            issues.append(
                "FLIR-visible marker tracking was lost continuously for "
                f"{visible_longest_missing_s:.3f}s"
            )
    reference_features: dict[str, float] = {}
    saturation: dict[str, float] = {}
    if last_thermal is None or last_oak is None:
        issues.append("no preflight frames captured")
    else:
        scalar = _thermal_scalar(last_thermal, palette, args.invert_palette)
        reference_features = reference_normalized_features(scalar, regions)
        scalar_max = float(len(palette) - 1) if palette is not None else 255.0
        saturation = {
            "room_reference": _reference_saturation_fraction(scalar, regions.room_reference, scalar_max),
            "warm_reference": _reference_saturation_fraction(scalar, regions.warm_reference, scalar_max),
        }
        reference_span = reference_features["reference_span"]
        if not _reference_span_is_adequate(reference_span, min_span=args.min_reference_span):
            issues.append(
                "room and warm reference patches differ by only "
                f"{reference_span:.2f} palette bins; require at least {args.min_reference_span:.2f}"
            )
        for name, fraction in saturation.items():
            if fraction > 0.1:
                issues.append(f"{name} is saturated in {fraction:.0%} of pixels")
        _write_frame(output_dir / "thermal.png", _draw_thermal_regions(last_thermal, regions))
        _write_frame(output_dir / "oak_rgb.png", last_oak)
        if last_visible is not None:
            visible_overlay = last_visible.copy()
            if args.thermal_roi_tracking == "flir-visible-markers":
                for color, roi in (
                    ((255, 0, 0), args.flir_visible_left_marker_roi),
                    ((0, 0, 255), args.flir_visible_right_marker_roi),
                ):
                    cv2.rectangle(visible_overlay, (roi.x, roi.y), (roi.x_end - 1, roi.y_end - 1), color, 3)
                if last_visible_marker is not None:
                    cv2.circle(visible_overlay, tuple(round(value) for value in last_visible_marker.left_xy), 8, (255, 0, 0), 3)
                    cv2.circle(visible_overlay, tuple(round(value) for value in last_visible_marker.right_xy), 8, (0, 0, 255), 3)
            _write_frame(output_dir / "flir_visible.png", visible_overlay)
        if last_marker is not None:
            overlay = last_oak.copy()
            cv2.rectangle(
                overlay,
                (args.oak_left_marker_roi.x, args.oak_left_marker_roi.y),
                (args.oak_left_marker_roi.x_end - 1, args.oak_left_marker_roi.y_end - 1),
                (255, 0, 0),
                2,
            )
            cv2.rectangle(
                overlay,
                (args.oak_right_marker_roi.x, args.oak_right_marker_roi.y),
                (args.oak_right_marker_roi.x_end - 1, args.oak_right_marker_roi.y_end - 1),
                (0, 0, 255),
                2,
            )
            cv2.circle(overlay, tuple(round(value) for value in last_marker.left_xy), 5, (255, 0, 0), 2)
            cv2.circle(overlay, tuple(round(value) for value in last_marker.right_xy), 5, (0, 0, 255), 2)
            _write_frame(output_dir / "oak_markers.png", overlay)
    report = {
        "automatic_issues": issues,
        "marker_detection_rate": marker_rate,
        "marker_evaluated_frame_count": frame_count,
        "marker_raw_frame_count": len(marker_samples),
        "longest_marker_loss_s": longest_missing_s,
        "flir_visible_marker_detection_rate": visible_marker_rate,
        "flir_visible_marker_evaluated_frame_count": visible_frame_count,
        "flir_visible_marker_raw_frame_count": len(visible_marker_samples),
        "flir_visible_longest_marker_loss_s": visible_longest_missing_s,
        "preflight_settle_s": args.preflight_settle_s,
        "reference_features": reference_features,
        "reference_saturation_fraction": saturation,
        "min_reference_span": args.min_reference_span,
        "last_marker": (
            {
                "left_xy": list(last_marker.left_xy),
                "right_xy": list(last_marker.right_xy),
                "distance_px": last_marker.distance_px,
            }
            if last_marker is not None
            else None
        ),
        "manual_checklist": [
            "no face, hair, torso, second hand, hot object, or reflective surface appears in FLIR",
            "foam and both markers remain fully visible and foam spans at least 24 thermal pixels",
            "both FLIR-visible black dots are clearly detected throughout the RGB preview",
            "both fingers can fully release while remaining in frame",
            "the fixture, reference patches, and cameras are rigid and will not move",
        ],
    }
    (output_dir / "preflight_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report, output_dir


def _sleep_until(timestamp: float) -> float:
    now = time.perf_counter()
    if now < timestamp:
        time.sleep(timestamp - now)
    return time.perf_counter()


def _record_d0(runtime: _CaptureRuntime, *, args, paths) -> float:
    step = type("CalibrationStep", (), {
        "block": "calibration",
        "phase": "calibration",
        "state": "R",
        "target_compression_pct": 0.0,
        "sequence_id": 0,
        "step_index": 0,
        "name": "d0_release_calibration",
        "pulse_index": 0,
    })()
    _append_event(paths, timestamp=time.perf_counter() - runtime.protocol_start, event_type="d0_start", reason="fully_release", step=step, action_attempt=1)
    distance_samples: list[tuple[float, float]] = []
    visible_marker_samples: list[tuple[float, MarkerObservation]] = []
    started = time.perf_counter()
    next_capture = started
    deadline = started + args.d0_s
    while next_capture <= deadline + 1e-9:
        now = _sleep_until(next_capture)
        sample = _capture_sample(
            runtime,
            step=step,
            capture_phase="d0_calibration",
            step_elapsed_s=now - started,
            action_attempt=1,
            d0_px=None,
            gate_stable_s=0.0,
            now=now,
        )
        if sample.marker is not None:
            distance_samples.append((now - started, sample.marker.distance_px))
        if sample.visible_marker is not None:
            visible_marker_samples.append((now - started, sample.visible_marker))
        if not _preview(runtime, step=step, compression_pct=None, gate_stable_s=0.0, enabled=args.show_preview):
            raise KeyboardInterrupt("operator aborted preview")
        next_capture += 1.0 / args.fps
    distances = _d0_post_settle_distances(distance_samples, settle_s=args.d0_settle_s)
    usable_s = args.d0_s - args.d0_settle_s
    expected_count = max(1, int(usable_s * args.fps * 0.9))
    if len(distances) < expected_count:
        raise RuntimeError(f"insufficient valid marker frames for d0: {len(distances)} < {expected_count}")
    relative_span = _d0_relative_span(distances)
    if not _d0_is_stable(distances, max_relative_span=args.max_d0_relative_span):
        _append_event(
            paths,
            timestamp=time.perf_counter() - runtime.protocol_start,
            event_type="d0_invalid",
            reason=(
                f"settle_s={args.d0_settle_s:.3f}; relative_span={relative_span:.3%}; "
                f"limit={args.max_d0_relative_span:.3%}"
            ),
            step=step,
            action_attempt=1,
        )
        raise RuntimeError(
            "d0 calibration was not stable: marker distance changed by "
            f"{relative_span:.1%} (limit {args.max_d0_relative_span:.1%}). "
            "Keep both fingers fully released and out of contact for the entire d0 countdown, then rerun with a new session id."
        )
    d0_px = float(median(distances))
    if runtime.roi_tracking_mode == "flir-visible-markers":
        visible_markers = [
            marker
            for elapsed_s, marker in visible_marker_samples
            if elapsed_s >= args.d0_settle_s
        ]
        if len(visible_markers) < expected_count or runtime.last_visible is None or runtime.last_thermal is None:
            raise RuntimeError(
                "insufficient FLIR-visible marker frames for thermal ROI registration during d0; "
                "check both RGB marker ROIs and rerun with a new session id"
            )
        runtime.thermal_marker_baseline = _thermal_marker_baseline_from_visible(
            visible_markers,
            visible_shape=runtime.last_visible.shape[:2],
            thermal_shape=runtime.last_thermal.shape[:2],
        )
        baseline = runtime.thermal_marker_baseline
        _append_event(
            paths,
            timestamp=time.perf_counter() - runtime.protocol_start,
            event_type="thermal_roi_baseline_complete",
            reason=(
                f"left=({baseline.left_xy[0]:.3f},{baseline.left_xy[1]:.3f}); "
                f"right=({baseline.right_xy[0]:.3f},{baseline.right_xy[1]:.3f}); samples={len(visible_markers)}"
            ),
            step=step,
            action_attempt=1,
        )
    _append_event(
        paths,
        timestamp=time.perf_counter() - runtime.protocol_start,
        event_type="d0_complete",
        reason=(
            f"d0_px={d0_px:.3f}; settle_s={args.d0_settle_s:.3f}; "
            f"samples={len(distances)}; relative_span={relative_span:.3%}"
        ),
        step=step,
        action_attempt=1,
    )
    return d0_px


def _record_step(runtime: _CaptureRuntime, *, args, paths, step, d0_px: float) -> None:
    tolerance_pct = _step_target_tolerance(step, args)
    runtime.target_tolerance_pct = tolerance_pct
    for attempt in range(1, args.max_attempts + 1):
        if not args.no_beep:
            print("\a", end="", flush=True)
        print(
            f"{step.block} | {step.name} | {STATE_INSTRUCTIONS[step.state]} | "
            f"target {step.target_compression_pct:.0f}% +/- {tolerance_pct:.0f}%"
        )
        _append_event(paths, timestamp=time.perf_counter() - runtime.protocol_start, event_type="action_prompt", reason=STATE_INSTRUCTIONS[step.state], step=step, action_attempt=attempt)
        gate = StableCompressionGate(
            target_pct=step.target_compression_pct,
            tolerance_pct=tolerance_pct,
            required_s=args.gate_stable_s,
        )
        approach_started = time.perf_counter()
        next_capture = approach_started
        reached = False
        while next_capture <= approach_started + args.max_reach_s + 1e-9:
            now = _sleep_until(next_capture)
            sample = _capture_sample(
                runtime,
                step=step,
                capture_phase="approach",
                step_elapsed_s=now - approach_started,
                action_attempt=attempt,
                d0_px=d0_px,
                gate_stable_s=gate.stable_seconds,
                now=now,
            )
            reached = gate.update(timestamp=sample.protocol_elapsed_s, compression_pct=sample.compression_pct)
            if not _preview(runtime, step=step, compression_pct=sample.compression_pct, gate_stable_s=gate.stable_seconds, enabled=args.show_preview):
                raise KeyboardInterrupt("operator aborted preview")
            if reached:
                break
            next_capture += 1.0 / args.fps
        if not reached:
            _append_event(paths, timestamp=time.perf_counter() - runtime.protocol_start, event_type="invalid", reason="target_not_reached", step=step, action_attempt=attempt)
            print("target was not held in tolerance; retrying")
            continue
        _append_event(paths, timestamp=time.perf_counter() - runtime.protocol_start, event_type="gated_hold_start", reason="target_stable", step=step, action_attempt=attempt)
        hold_started = time.perf_counter()
        next_capture = hold_started
        valid_hold = True
        hold_gate = HoldToleranceGate(
            target_pct=step.target_compression_pct,
            tolerance_pct=tolerance_pct,
            max_gap_s=args.max_hold_gap_s,
        )
        while next_capture <= hold_started + step.hold_s + 1e-9:
            now = _sleep_until(next_capture)
            sample = _capture_sample(
                runtime,
                step=step,
                capture_phase="stable_hold",
                step_elapsed_s=now - hold_started,
                action_attempt=attempt,
                d0_px=d0_px,
                gate_stable_s=gate.stable_seconds,
                now=now,
            )
            if not hold_gate.update(timestamp=sample.protocol_elapsed_s, compression_pct=sample.compression_pct):
                valid_hold = False
            if not _preview(runtime, step=step, compression_pct=sample.compression_pct, gate_stable_s=gate.stable_seconds, enabled=args.show_preview):
                raise KeyboardInterrupt("operator aborted preview")
            if not valid_hold:
                break
            next_capture += 1.0 / args.fps
        if valid_hold:
            _append_event(paths, timestamp=time.perf_counter() - runtime.protocol_start, event_type="step_complete", reason="valid_gated_hold", step=step, action_attempt=attempt)
            return
        _append_event(paths, timestamp=time.perf_counter() - runtime.protocol_start, event_type="invalid", reason="gate_lost_during_hold", step=step, action_attempt=attempt)
        print("target left the tolerance band during hold; retrying")
    raise RuntimeError(f"step remained invalid after {args.max_attempts} attempts: {step.name}")


def main() -> None:
    args = _parse_args()
    requested_auto_rep = args.rep is None
    _resolve_auto_rep(args)
    if requested_auto_rep:
        print(f"auto-selected rep {args.rep:02d}")
    if (
        args.target_tolerance_pct <= 0
        or args.release_tolerance_pct <= 0
        or args.gate_stable_s <= 0
        or args.max_hold_gap_s < 0
        or args.max_attempts <= 0
        or args.d0_s <= 0
        or not 0 <= args.d0_settle_s < args.d0_s
        or args.preflight_s <= 0
        or not 0 <= args.preflight_settle_s < args.preflight_s
        or args.min_reference_span <= 0
    ):
        raise ValueError("target tolerance, gate duration, max attempts, and minimum reference span must be positive")
    regions = _regions_from_args(args)
    palette = _load_palette(args.palette)
    thermal = OpenCVCameraSource(args.thermal)
    oak = OAKCameraSource(fps=args.oak_fps)
    visible = OpenCVCameraSource(args.flir_visible) if args.record_flir_visible else None
    hand_tracker = None
    try:
        report, preflight_dir = _run_preflight(
            args=args,
            regions=regions,
            thermal=thermal,
            oak=oak,
            visible=visible,
            palette=palette,
        )
        print(f"preflight saved to {preflight_dir}")
        if report["automatic_issues"]:
            for issue in report["automatic_issues"]:
                print(f"PRECHECK FAILED: {issue}")
            raise SystemExit("fix the setup and rerun --preflight-only")
        print("Manual preflight checklist:")
        for item in report["manual_checklist"]:
            print(f"- {item}")
        if args.preflight_only:
            print("preflight passed automatic checks; inspect saved snapshots before formal capture")
            return
        if not args.yes and input("Type READY after confirming the manual checklist: ").strip() != "READY":
            raise SystemExit("capture aborted")
        paths, plan, _regions = _prepare_trial(args)
        metadata = json.loads(paths.metadata_path.read_text())
        metadata["preflight_report_path"] = str(preflight_dir)
        metadata["preflight_automatic_issues"] = report["automatic_issues"]
        paths.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        if args.track_hand_geometry:
            hand_tracker = _HandGeometryTracker()
        runtime = _CaptureRuntime(
            paths=paths,
            regions=regions,
            thermal=thermal,
            oak=oak,
            visible=visible,
            left_marker_roi=args.oak_left_marker_roi,
            right_marker_roi=args.oak_right_marker_roi,
            palette=palette,
            invert_palette=args.invert_palette,
            marker_max_gray=args.marker_max_gray,
            marker_min_area_px=args.marker_min_area_px,
            protocol_start=time.perf_counter(),
            hand_tracker=hand_tracker,
            target_tolerance_pct=args.target_tolerance_pct,
            roi_tracking_mode=args.thermal_roi_tracking,
            visible_left_marker_roi=args.flir_visible_left_marker_roi,
            visible_right_marker_roi=args.flir_visible_right_marker_roi,
            visible_marker_max_gray=args.flir_visible_marker_max_gray,
            visible_marker_min_area_px=args.flir_visible_marker_min_area_px,
            visible_marker_max_area_px=args.flir_visible_marker_max_area_px,
        )
        print(f"calibrating d0 in fully released R state for {args.d0_s:g}s")
        d0_px = _record_d0(runtime, args=args, paths=paths)
        metadata = json.loads(paths.metadata_path.read_text())
        metadata["d0_px"] = d0_px
        if runtime.thermal_marker_baseline is not None:
            metadata["thermal_roi_tracking"]["thermal_baseline_marker"] = {
                "left_xy": list(runtime.thermal_marker_baseline.left_xy),
                "right_xy": list(runtime.thermal_marker_baseline.right_xy),
                "distance_px": runtime.thermal_marker_baseline.distance_px,
            }
        paths.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        print(f"d0 = {d0_px:.2f} px; beginning {len(plan)} fixed protocol steps")
        for step in plan:
            _record_step(runtime, args=args, paths=paths, step=step, d0_px=d0_px)
        print(f"saved foam-compression trial to {paths.root}")
    finally:
        if hand_tracker is not None:
            hand_tracker.close()
        thermal.close()
        oak.close()
        if visible is not None:
            visible.close()
        if args.show_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
