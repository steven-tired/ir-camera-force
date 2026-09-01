"""Temporally tracked thermal ROIs for single-finger Null/Press trials."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from ir_force.single_finger_curve_analysis import (
    ANALYSIS_BINS,
    ANALYSIS_START_S,
    BIN_DURATION_S,
    exact_cluster_test,
)
from ir_force.single_finger_curve_protocol import (
    persisted_flag_is_true,
)


FRAME_SHAPE = (120, 160)
SEGMENT_OFFSET_COUNT = 100.0
SEGMENT_MIN_AREA_PX = 200
SEGMENT_MIN_CENTROID_X = 60.0
DESK_SIZE_PX = 15
DISTAL_AXIAL_RANGE_PX = (5.0, 13.0)
PROXIMAL_AXIAL_RANGE_PX = (15.0, 23.0)
FINGER_HALF_WIDTH_PX = 3.5
MIN_FINGER_WIDTH_PX = 10.0
FINGER_WIDTH_TOLERANCE_PX = 0.05
MAX_CENTER_STEP_PX = 1.0
MAX_AREA_CHANGE_RATIO = 0.20
MIN_INTERIOR_SUPPORT_FRACTION = 0.80
A1_INITIALIZATION_END_S = 2.0
MIN_BIN_SAMPLES = 2
MIN_DESCRIPTIVE_PAIRS_PER_BIN = 3


def _validated_frames(frames) -> list[np.ndarray]:
    validated = []
    for frame in frames:
        frame = np.asarray(frame)
        if frame.shape != FRAME_SHAPE or frame.dtype != np.uint16:
            raise ValueError("thermal_frame_invalid")
        validated.append(frame)
    if not validated:
        raise ValueError("A1 initialization frames missing")
    return validated


def _largest_hand_mask(frame: np.ndarray) -> np.ndarray:
    threshold = float(np.median(frame)) + SEGMENT_OFFSET_COUNT
    binary = (frame >= threshold).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    candidates = [
        label
        for label in range(1, count)
        if stats[label, cv2.CC_STAT_AREA] >= SEGMENT_MIN_AREA_PX
        and centroids[label, 0] > SEGMENT_MIN_CENTROID_X
    ]
    if not candidates:
        raise ValueError("right_hot_hand_component_missing")
    component = max(
        candidates,
        key=lambda label: int(stats[label, cv2.CC_STAT_AREA]),
    )
    return labels == component


def _hand_axis(hand_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_pixels, x_pixels = np.nonzero(hand_mask)
    left_cutoff = float(np.percentile(x_pixels, 3.0))
    tip_band = x_pixels <= left_cutoff
    tip = np.asarray(
        [
            np.median(x_pixels[tip_band]),
            np.median(y_pixels[tip_band]),
        ],
        dtype=float,
    )
    center = np.asarray(
        [np.median(x_pixels), np.median(y_pixels)],
        dtype=float,
    )
    direction = center - tip
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("right_hot_hand_axis_invalid")
    return tip, direction / norm


def _finger_masks(
    hand_mask: np.ndarray,
    eroded_hand_mask: np.ndarray,
    tip: np.ndarray,
    direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    yy, xx = np.indices(FRAME_SHAPE)
    delta_x = xx - tip[0]
    delta_y = yy - tip[1]
    axial = delta_x * direction[0] + delta_y * direction[1]
    perpendicular = np.abs(
        delta_x * (-direction[1]) + delta_y * direction[0]
    )
    distance = cv2.distanceTransform(
        hand_mask.astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    radii = []
    for axial_position in np.linspace(
        DISTAL_AXIAL_RANGE_PX[0] + 1.0,
        DISTAL_AXIAL_RANGE_PX[1] - 1.0,
        7,
    ):
        center = np.rint(tip + axial_position * direction).astype(int)
        u, v = int(center[0]), int(center[1])
        if 0 <= u < FRAME_SHAPE[1] and 0 <= v < FRAME_SHAPE[0]:
            radius = float(distance[v, u])
            if radius > 0.0:
                radii.append(radius)
    if not radii:
        raise ValueError("finger_interior_missing")
    finger_width_px = 2.0 * float(np.median(radii))
    common = eroded_hand_mask & (perpendicular <= FINGER_HALF_WIDTH_PX)
    distal = (
        common
        & (axial >= DISTAL_AXIAL_RANGE_PX[0])
        & (axial < DISTAL_AXIAL_RANGE_PX[1])
    )
    proximal = (
        common
        & (axial >= PROXIMAL_AXIAL_RANGE_PX[0])
        & (axial < PROXIMAL_AXIAL_RANGE_PX[1])
    )
    if np.count_nonzero(distal) < 20:
        raise ValueError("distal_interior_pixels_insufficient")
    if np.count_nonzero(proximal) < 20:
        raise ValueError("proximal_interior_pixels_insufficient")
    return distal, proximal, finger_width_px


def _desk_mask(
    a1_frames: list[np.ndarray],
    hand_mask: np.ndarray,
) -> np.ndarray:
    radius = DESK_SIZE_PX // 2
    exclusion = cv2.dilate(
        hand_mask.astype(np.uint8),
        np.ones((15, 15), dtype=np.uint8),
    ).astype(bool)
    best = None
    for v in range(82, FRAME_SHAPE[0] - radius, 3):
        for u in range(radius, FRAME_SHAPE[1] - radius, 3):
            ys = slice(v - radius, v + radius + 1)
            xs = slice(u - radius, u + radius + 1)
            if np.any(exclusion[ys, xs]):
                continue
            patch_medians = [
                float(np.median(frame[ys, xs])) for frame in a1_frames
            ]
            spatial = float(
                np.median(
                    [
                        np.std(frame[ys, xs], dtype=np.float64)
                        for frame in a1_frames
                    ]
                )
            )
            temporal = float(np.std(patch_medians, dtype=np.float64))
            candidate = (temporal + 0.1 * spatial, v, u)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError("far_desk_roi_missing")
    _score, v, u = best
    desk = np.zeros(FRAME_SHAPE, dtype=bool)
    desk[
        v - radius : v + radius + 1,
        u - radius : u + radius + 1,
    ] = True
    return desk


def initialize_trial_anchor(a1_frames) -> dict:
    frames = _validated_frames(a1_frames)
    median_frame = np.median(np.stack(frames), axis=0)
    hand_mask = _largest_hand_mask(median_frame)
    eroded_hand_mask = cv2.erode(
        hand_mask.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    tip, direction = _hand_axis(hand_mask)
    distal, proximal, finger_width_px = _finger_masks(
        hand_mask,
        eroded_hand_mask,
        tip,
        direction,
    )
    return {
        "median_frame": median_frame,
        "hand_mask": hand_mask,
        "eroded_hand_mask": eroded_hand_mask,
        "tip_uv": tip,
        "direction_uv": direction,
        "distal_mask": distal,
        "proximal_mask": proximal,
        "desk_mask": _desk_mask(frames, hand_mask),
        "finger_width_px": finger_width_px,
    }


def _shift_mask(mask: np.ndarray, shift_u: int, shift_v: int) -> np.ndarray:
    shifted = np.zeros_like(mask)
    source_x0 = max(0, -shift_u)
    source_x1 = min(mask.shape[1], mask.shape[1] - shift_u)
    source_y0 = max(0, -shift_v)
    source_y1 = min(mask.shape[0], mask.shape[0] - shift_v)
    if source_x0 >= source_x1 or source_y0 >= source_y1:
        return shifted
    shifted[
        source_y0 + shift_v : source_y1 + shift_v,
        source_x0 + shift_u : source_x1 + shift_u,
    ] = mask[source_y0:source_y1, source_x0:source_x1]
    return shifted


def _mask_center(mask: np.ndarray) -> list[float]:
    yy, xx = np.nonzero(mask)
    if xx.size == 0:
        raise ValueError("roi_empty")
    return [float(np.mean(xx)), float(np.mean(yy))]


def _trimmed_mean(values: np.ndarray, fraction: float = 0.2) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    trim = int(np.floor(ordered.size * fraction))
    if trim and ordered.size > 2 * trim:
        ordered = ordered[trim:-trim]
    return float(np.mean(ordered))


class TrialTracker:
    def __init__(self, anchor: dict, *, enforce_stability_gates: bool = True):
        """Track the A1-frozen ROIs across a trial.

        ``enforce_stability_gates=False`` keeps measuring when the finger
        width, centre step, or component area gate is violated, and reports the
        violations in ``gate_violations`` instead. It exists only for
        diagnostic sanity checks; the frozen v2 primary analysis always
        enforces the gates.
        """
        self.anchor = anchor
        self.enforce_stability_gates = bool(enforce_stability_gates)
        self._shift = (0, 0)
        self._component_area = int(np.count_nonzero(anchor["hand_mask"]))

    def _best_shift(self, current_mask: np.ndarray) -> tuple[int, int]:
        previous_u, previous_v = self._shift
        best_score = -1.0
        best_shift = self._shift
        best_step = float("inf")
        for shift_v in range(previous_v - 4, previous_v + 5):
            for shift_u in range(previous_u - 4, previous_u + 5):
                shifted = _shift_mask(
                    self.anchor["hand_mask"],
                    shift_u,
                    shift_v,
                )
                intersection = int(np.count_nonzero(shifted & current_mask))
                union = int(np.count_nonzero(shifted | current_mask))
                score = 0.0 if union == 0 else intersection / union
                step = float(
                    np.hypot(
                        shift_u - previous_u,
                        shift_v - previous_v,
                    )
                )
                if score > best_score or (
                    score == best_score and step < best_step
                ):
                    best_score = score
                    best_shift = (shift_u, shift_v)
                    best_step = step
        return best_shift

    def _invalid(
        self,
        reason: str,
        *,
        shift: tuple[int, int],
        component_area: int | None,
        area_change_ratio: float | None,
    ) -> dict:
        return {
            "tracking_valid": False,
            "tracking_reasons": [reason],
            "gate_violations": [],
            "shift_uv": [int(shift[0]), int(shift[1])],
            "center_step_px": float(
                np.hypot(
                    shift[0] - self._shift[0],
                    shift[1] - self._shift[1],
                )
            ),
            "component_area_px": component_area,
            "area_change_ratio": area_change_ratio,
            "primary_signal_count": None,
            "distal_count": None,
            "proximal_count": None,
            "desk_count": None,
            "desk_uv": _mask_center(self.anchor["desk_mask"]),
        }

    def measure(self, frame) -> dict:
        frame = _validated_frames([frame])[0]
        try:
            current_mask = _largest_hand_mask(frame)
        except ValueError as exc:
            return self._invalid(
                str(exc),
                shift=self._shift,
                component_area=None,
                area_change_ratio=None,
            )
        shift = self._best_shift(current_mask)
        step = float(
            np.hypot(
                shift[0] - self._shift[0],
                shift[1] - self._shift[1],
            )
        )
        component_area = int(np.count_nonzero(current_mask))
        area_change_ratio = abs(
            component_area - self._component_area
        ) / self._component_area
        gate_violations = []
        if (
            self.anchor["finger_width_px"]
            < MIN_FINGER_WIDTH_PX - FINGER_WIDTH_TOLERANCE_PX
        ):
            gate_violations.append("finger_width_below_10px")
        if step > MAX_CENTER_STEP_PX:
            gate_violations.append("center_step_exceeded")
        if area_change_ratio > MAX_AREA_CHANGE_RATIO:
            gate_violations.append("component_area_jump")
        if gate_violations and self.enforce_stability_gates:
            return self._invalid(
                gate_violations[0],
                shift=shift,
                component_area=component_area,
                area_change_ratio=area_change_ratio,
            )
        distal = _shift_mask(
            self.anchor["distal_mask"],
            shift[0],
            shift[1],
        )
        proximal = _shift_mask(
            self.anchor["proximal_mask"],
            shift[0],
            shift[1],
        )
        for label, roi in (("distal", distal), ("proximal", proximal)):
            if np.any(roi[[0, -1], :]) or np.any(roi[:, [0, -1]]):
                return self._invalid(
                    f"{label}_roi_touches_frame_boundary",
                    shift=shift,
                    component_area=component_area,
                    area_change_ratio=area_change_ratio,
                )
            support = np.count_nonzero(roi & current_mask) / np.count_nonzero(
                roi
            )
            if support < MIN_INTERIOR_SUPPORT_FRACTION:
                return self._invalid(
                    f"{label}_interior_pixels_insufficient",
                    shift=shift,
                    component_area=component_area,
                    area_change_ratio=area_change_ratio,
                )
        distal_values = frame[distal]
        proximal_values = frame[proximal]
        desk_values = frame[self.anchor["desk_mask"]]
        distal_count = float(np.median(distal_values))
        proximal_count = float(np.median(proximal_values))
        self._shift = shift
        self._component_area = component_area
        return {
            "tracking_valid": True,
            "tracking_reasons": [],
            "gate_violations": gate_violations,
            "shift_uv": [int(shift[0]), int(shift[1])],
            "center_step_px": step,
            "component_area_px": component_area,
            "area_change_ratio": area_change_ratio,
            "distal_pixel_count": int(np.count_nonzero(distal)),
            "proximal_pixel_count": int(np.count_nonzero(proximal)),
            "distal_uv": _mask_center(distal),
            "proximal_uv": _mask_center(proximal),
            "desk_uv": _mask_center(self.anchor["desk_mask"]),
            "distal_count": distal_count,
            "proximal_count": proximal_count,
            "desk_count": float(np.median(desk_values)),
            "distal_trimmed_mean_count": _trimmed_mean(distal_values),
            "proximal_trimmed_mean_count": _trimmed_mean(proximal_values),
            "primary_signal_count": distal_count - proximal_count,
        }


def _rolling_median(values: list[float | None], width: int = 5) -> list:
    radius = width // 2
    output = []
    for index, value in enumerate(values):
        if value is None or not np.isfinite(value):
            output.append(None)
            continue
        window = [
            item
            for item in values[
                max(0, index - radius) : min(len(values), index + radius + 1)
            ]
            if item is not None and np.isfinite(item)
        ]
        output.append(float(np.median(window)))
    return output


def _bin_without_interpolation(
    samples: list[dict],
    field: str,
) -> tuple[list[float | None], list[int]]:
    values = []
    missing = []
    for bin_index in range(ANALYSIS_BINS):
        start = ANALYSIS_START_S + bin_index * BIN_DURATION_S
        end = start + BIN_DURATION_S
        in_bin = [
            float(sample[field])
            for sample in samples
            if sample.get(field) is not None
            and start <= sample["time_s"] < end
        ]
        if len(in_bin) >= MIN_BIN_SAMPLES:
            values.append(float(np.median(in_bin)))
        else:
            values.append(None)
            missing.append(bin_index)
    return values, missing


def _significant_clusters(test: dict, differences: np.ndarray) -> list[dict]:
    clusters = []
    for cluster in test["clusters"]:
        if cluster["p_corrected"] > 0.05:
            continue
        values = differences[
            :,
            cluster["start_bin"] : cluster["end_bin"] + 1,
        ]
        clusters.append(
            {
                **cluster,
                "start_s": (
                    ANALYSIS_START_S
                    + cluster["start_bin"] * BIN_DURATION_S
                ),
                "end_s": (
                    ANALYSIS_START_S
                    + (cluster["end_bin"] + 1) * BIN_DURATION_S
                ),
                "median_press_minus_null_count": float(np.median(values)),
            }
        )
    return clusters


def _phase_medians(differences: np.ndarray) -> dict[str, float]:
    return {
        "X": float(np.median(differences[:, 0:10])),
        "A2": float(np.median(differences[:, 10:20])),
        "A3": float(np.median(differences[:, 20:30])),
    }


def analyze_tracked_thermal(rows, *, frame_loader) -> dict:
    grouped: dict[tuple[int, str], list[tuple[dict, np.ndarray]]] = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("row_type") != "frame"
            or row.get("condition") not in ("null", "press")
            or row.get("block_index") not in range(6)
            or not persisted_flag_is_true(row.get("artifact_write_ok"))
        ):
            continue
        grouped.setdefault(
            (row["block_index"], row["condition"]),
            [],
        ).append((row, frame_loader(row)))
    expected = {
        (block, condition)
        for block in range(6)
        for condition in ("null", "press")
    }
    if set(grouped) != expected:
        raise ValueError("six complete primary block labels required")

    curves = []
    runtime_anchors = []
    tracking_failure_counts: Counter[str] = Counter()
    analysis_failure_counts: Counter[str] = Counter()
    invalid_frame_count = 0
    missing_bins = []
    binned = {
        "primary_signal_count": {"null": [], "press": []},
        "shift_magnitude_px": {"null": [], "press": []},
    }
    desk_changes = []
    complete_trials: dict[tuple[int, str], bool] = {}
    for block_index in range(6):
        for condition in ("null", "press"):
            trial = sorted(
                grouped[(block_index, condition)],
                key=lambda item: float(item[0]["global_elapsed_s"]),
            )
            a1_frames = [
                frame
                for row, frame in trial
                if row.get("phase") == "A1"
                and float(row["global_elapsed_s"]) < A1_INITIALIZATION_END_S
            ]
            try:
                anchor = initialize_trial_anchor(a1_frames)
            except ValueError as exc:
                reason = f"anchor:{exc}"
                tracking_failure_counts[reason] += len(trial)
                invalid_frame_count += len(trial)
                complete_trials[(block_index, condition)] = False
                curves.append(
                    {
                        "block_index": block_index,
                        "condition": condition,
                        "time_s": [
                            float(row["global_elapsed_s"])
                            for row, _frame in trial
                        ],
                        "normalized_count": [None] * len(trial),
                        "rolling_median_5_count": [None] * len(trial),
                        "desk_change_count": [None] * len(trial),
                        "tracking_valid": [False] * len(trial),
                        "tracking_failure": reason,
                    }
                )
                for field in binned:
                    binned[field][condition].append([None] * ANALYSIS_BINS)
                missing_bins.extend(
                    {
                        "block_index": block_index,
                        "condition": condition,
                        "field": "primary_signal_count",
                        "bin_index": bin_index,
                    }
                    for bin_index in range(ANALYSIS_BINS)
                )
                continue
            anchor_record = {
                **anchor,
                "block_index": block_index,
                "condition": condition,
            }
            runtime_anchors.append(anchor_record)
            tracker = TrialTracker(anchor)
            measured_samples = []
            for row, frame in trial:
                measured = tracker.measure(frame)
                reasons = measured["tracking_reasons"]
                tracking_failure_counts.update(reasons)
                invalid_frame_count += int(not measured["tracking_valid"])
                measured_samples.append(
                    {
                        "time_s": float(row["global_elapsed_s"]),
                        "phase": row["phase"],
                        "frame_index": row.get("frame_index"),
                        **measured,
                    }
                )
            baseline_samples = [
                sample
                for sample in measured_samples
                if sample["tracking_valid"]
                and 3.0 <= sample["time_s"] < 5.0
            ]
            if not baseline_samples:
                analysis_failure_counts["A1_baseline_missing"] += 1
                signal_baseline = None
                desk_baseline = None
            else:
                signal_baseline = float(
                    np.median(
                        [
                            sample["primary_signal_count"]
                            for sample in baseline_samples
                        ]
                    )
                )
                desk_baseline = float(
                    np.median(
                        [sample["desk_count"] for sample in baseline_samples]
                    )
                )
            normalized_samples = []
            for sample in measured_samples:
                valid = (
                    sample["tracking_valid"]
                    and signal_baseline is not None
                    and desk_baseline is not None
                )
                normalized_count = (
                    sample["primary_signal_count"] - signal_baseline
                    if valid
                    else None
                )
                desk_change = (
                    sample["desk_count"] - desk_baseline if valid else None
                )
                if desk_change is not None:
                    desk_changes.append(float(desk_change))
                normalized_samples.append(
                    {
                        **sample,
                        "normalized_count": normalized_count,
                        "desk_change_count": desk_change,
                        "shift_magnitude_px": (
                            float(np.hypot(*sample["shift_uv"]))
                            if sample["tracking_valid"]
                            else None
                        ),
                    }
                )
            normalized_values = [
                sample["normalized_count"] for sample in normalized_samples
            ]
            curve = {
                "block_index": block_index,
                "condition": condition,
                "baseline_count": signal_baseline,
                "desk_baseline_count": desk_baseline,
                "time_s": [
                    sample["time_s"] for sample in normalized_samples
                ],
                "normalized_count": normalized_values,
                "rolling_median_5_count": _rolling_median(
                    normalized_values,
                    width=5,
                ),
                "desk_change_count": [
                    sample["desk_change_count"]
                    for sample in normalized_samples
                ],
                "tracking_valid": [
                    sample["tracking_valid"]
                    for sample in normalized_samples
                ],
            }
            curves.append(curve)
            trial_missing = []
            for field in binned:
                values, missing = _bin_without_interpolation(
                    normalized_samples,
                    field,
                )
                binned[field][condition].append(values)
                if field == "primary_signal_count":
                    trial_missing = missing
                missing_bins.extend(
                    {
                        "block_index": block_index,
                        "condition": condition,
                        "field": field,
                        "bin_index": bin_index,
                    }
                    for bin_index in missing
                )
            complete_trials[(block_index, condition)] = (
                signal_baseline is not None and not trial_missing
            )

    complete_pair_count = sum(
        complete_trials.get((block, "null"), False)
        and complete_trials.get((block, "press"), False)
        for block in range(6)
    )
    primary_null = np.asarray(
        binned["primary_signal_count"]["null"],
        dtype=object,
    )
    primary_press = np.asarray(
        binned["primary_signal_count"]["press"],
        dtype=object,
    )
    primary_complete = (
        primary_null.shape == (6, ANALYSIS_BINS)
        and primary_press.shape == (6, ANALYSIS_BINS)
        and all(value is not None for value in primary_null.flat)
        and all(value is not None for value in primary_press.flat)
    )
    if primary_complete:
        differences = primary_press.astype(float) - primary_null.astype(float)
        primary_test = exact_cluster_test(differences)
        significant = _significant_clusters(primary_test, differences)
        phase_medians = _phase_medians(differences)
    else:
        differences = None
        primary_test = None
        significant = []
        phase_medians = None
    return {
        "analysis_role": "posthoc_tracked_roi_v2_not_preregistered",
        "formal_primary_verdict": "INCOMPLETE_FOR_PRIMARY_TEST",
        "method_version": "tracked_thermal_roi_v2",
        "selected_pair_count": 6,
        "complete_pair_count": int(complete_pair_count),
        "roi_method": {
            "anchor": "median_of_first_2s_A1",
            "hand_segmentation": "frame_median_plus_100_count",
            "erosion": "3x3_one_iteration",
            "primary": "median(distal_interior)-median(proximal_interior)",
            "distal_axial_range_px": list(DISTAL_AXIAL_RANGE_PX),
            "proximal_axial_range_px": list(PROXIMAL_AXIAL_RANGE_PX),
            "desk": "fixed_auto_selected_low_variance_15x15_diagnostic_only",
            "temporal_tracking": "baseline_mask_translation_max_iou",
        },
        "quality_gates": {
            "minimum_finger_width_px": MIN_FINGER_WIDTH_PX,
            "finger_width_numeric_tolerance_px": (
                FINGER_WIDTH_TOLERANCE_PX
            ),
            "maximum_center_step_px": MAX_CENTER_STEP_PX,
            "maximum_component_area_change_ratio": MAX_AREA_CHANGE_RATIO,
            "minimum_interior_support_fraction": (
                MIN_INTERIOR_SUPPORT_FRACTION
            ),
            "minimum_samples_per_0_5s_bin": MIN_BIN_SAMPLES,
            "missing_bin_policy": "no_interpolation_inference_incomplete",
        },
        "tracking_failure_counts": dict(
            sorted(tracking_failure_counts.items())
        ),
        "analysis_failure_counts": dict(
            sorted(analysis_failure_counts.items())
        ),
        "invalid_frame_count": int(invalid_frame_count),
        "missing_bin_count": len(
            {
                (
                    item["block_index"],
                    item["condition"],
                    item["bin_index"],
                )
                for item in missing_bins
                if item["field"] == "primary_signal_count"
            }
        ),
        "missing_bins": missing_bins,
        "curves": curves,
        "binned": binned,
        "primary": {
            "test": primary_test,
            "significant_clusters": significant,
            "median_press_minus_null_by_phase_count": phase_medians,
        },
        "desk_drift_diagnostic": {
            "role": "diagnostic_only_not_subtracted_from_primary",
            "maximum_absolute_change_count": (
                float(np.max(np.abs(desk_changes))) if desk_changes else None
            ),
            "median_absolute_change_count": (
                float(np.median(np.abs(desk_changes)))
                if desk_changes
                else None
            ),
        },
        "trial_anchors": runtime_anchors,
    }


def _phase_annotations(axis) -> None:
    for boundary in (5.0, 10.0, 15.0):
        axis.axvline(boundary, color="0.35", linewidth=0.8, linestyle="--")
    for x, label in zip(
        (2.5, 7.5, 12.5, 17.5),
        ("A1", "X", "A2", "A3"),
        strict=True,
    ):
        axis.text(
            x,
            0.98,
            label,
            ha="center",
            va="top",
            transform=axis.get_xaxis_transform(),
        )


def plot_raw_and_rolling_curves(result: dict, path: str | Path) -> dict:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    colors = {"null": "#2f6fb0", "press": "#e57b25"}
    figure, axis = plt.subplots(figsize=(11, 6))
    raw_count = 0
    rolling_count = 0
    for curve in result["curves"]:
        time_s = np.asarray(curve["time_s"], dtype=float)
        raw = np.asarray(
            [
                np.nan if value is None else value
                for value in curve["normalized_count"]
            ],
            dtype=float,
        )
        rolling = np.asarray(
            [
                np.nan if value is None else value
                for value in curve["rolling_median_5_count"]
            ],
            dtype=float,
        )
        axis.plot(
            time_s,
            raw,
            color=colors[curve["condition"]],
            alpha=0.18,
            linewidth=0.7,
        )
        raw_count += 1
        axis.plot(
            time_s,
            rolling,
            color=colors[curve["condition"]],
            alpha=0.78,
            linewidth=1.2,
        )
        rolling_count += 1
    _phase_annotations(axis)
    axis.set(
        xlim=(0.0, 20.0),
        xlabel="Time from A1 start (s)",
        ylabel="Tracked distal - proximal, A1-subtracted (count)",
        title=(
            "POSTHOC TRACKED ROI V2 — raw faint, 5-frame median solid\n"
            "Null blue, Press orange; rolling curves are display-only"
        ),
    )
    axis.grid(alpha=0.18)
    figure.tight_layout()
    figure.savefig(Path(path), dpi=180)
    plt.close(figure)
    return {
        "raw_curve_count": raw_count,
        "rolling_curve_count": rolling_count,
    }


def plot_paired_median(result: dict, path: str | Path) -> dict:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    null = np.asarray(
        [
            [np.nan if value is None else value for value in pair]
            for pair in result["binned"]["primary_signal_count"]["null"]
        ],
        dtype=float,
    )
    press = np.asarray(
        [
            [np.nan if value is None else value for value in pair]
            for pair in result["binned"]["primary_signal_count"]["press"]
        ],
        dtype=float,
    )
    differences = press - null
    times = ANALYSIS_START_S + (
        np.arange(ANALYSIS_BINS) + 0.5
    ) * BIN_DURATION_S
    figure, axis = plt.subplots(figsize=(10, 5))
    for pair in differences:
        axis.plot(times, pair, color="#7755aa", alpha=0.4, linewidth=1.0)
    paired_median = np.full(ANALYSIS_BINS, np.nan, dtype=float)
    pair_count_by_bin = []
    for bin_index in range(ANALYSIS_BINS):
        values = differences[:, bin_index]
        finite = values[np.isfinite(values)]
        pair_count_by_bin.append(int(finite.size))
        if finite.size >= MIN_DESCRIPTIVE_PAIRS_PER_BIN:
            paired_median[bin_index] = float(np.median(finite))
    axis.plot(
        times,
        paired_median,
        color="black",
        linewidth=2.0,
        label="paired median",
    )
    for boundary in (10.0, 15.0):
        axis.axvline(boundary, color="0.4", linestyle="--", linewidth=0.8)
    axis.axhline(0.0, color="0.3", linewidth=0.8)
    axis.set(
        xlim=(5.0, 20.0),
        xlabel="Time from A1 start (s)",
        ylabel="Press - Null (count)",
        title=(
            "Tracked ROI v2 paired Press - Null\n"
            "median shown only where at least 3 pairs are available"
        ),
    )
    axis.legend()
    axis.grid(alpha=0.2)
    count_axis = axis.twinx()
    count_axis.step(
        times,
        pair_count_by_bin,
        where="mid",
        color="0.55",
        alpha=0.5,
        linewidth=0.8,
    )
    count_axis.set(
        ylabel="available pairs",
        ylim=(0, 6.5),
        yticks=range(0, 7),
    )
    figure.tight_layout()
    figure.savefig(Path(path), dpi=180)
    plt.close(figure)
    return {
        "paired_curve_count": int(differences.shape[0]),
        "pair_count_by_bin": pair_count_by_bin,
        "minimum_pairs_for_displayed_median": (
            MIN_DESCRIPTIVE_PAIRS_PER_BIN
        ),
    }


def plot_tracking_overlay(anchor: dict, path: str | Path) -> dict:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.imshow(anchor["median_frame"], cmap="inferno")
    specifications = (
        ("distal_mask", "#5df28b", "distal interior"),
        ("proximal_mask", "#ffffff", "proximal interior"),
        ("desk_mask", "#65d1ff", "desk diagnostic"),
    )
    for field, color, label in specifications:
        axis.contour(
            anchor[field].astype(float),
            levels=[0.5],
            colors=[color],
            linewidths=1.8,
        )
        center = _mask_center(anchor[field])
        axis.scatter([], [], edgecolor=color, facecolor="none", label=label)
        axis.text(center[0], center[1], label.split()[0], color=color)
    axis.set(
        title="Tracked ROI v2 A1-frozen masks",
        xlim=(0, FRAME_SHAPE[1] - 1),
        ylim=(FRAME_SHAPE[0] - 1, 0),
    )
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(Path(path), dpi=180)
    plt.close(figure)
    return {
        "distal_pixel_count": int(np.count_nonzero(anchor["distal_mask"])),
        "proximal_pixel_count": int(np.count_nonzero(anchor["proximal_mask"])),
        "desk_pixel_count": int(np.count_nonzero(anchor["desk_mask"])),
    }
