#!/usr/bin/env python3
"""Frozen Stage 1E attempt-05 position-diagnostic signal analysis."""

from __future__ import annotations

import argparse
import json
from math import hypot, isfinite
from pathlib import Path

import numpy as np


EXPECTED_STAGE0_SHA256 = (
    "22d41109dcaefb29ad770fb5715c35dfd6c13c68195fbcb55e3b9d6fb4ef756b"
)
EXPECTED_FROZEN_XML_SHA256 = (
    "2ca1ed48450dea16a5778cb5645dd4852d544490e4f47330dd938f743bc6f434"
)
GROUP_COUNT = 6
TARGET_VALID_SAMPLES = 5
PHASE_TIMEOUT_S = 10.0
EXPECTED_PROTOCOL = {
    "groups": GROUP_COUNT,
    "sequence": ["just_touch", "press_hard", "return_just_touch"],
    "target_valid_samples": TARGET_VALID_SAMPLES,
    "phase_timeout_s": PHASE_TIMEOUT_S,
    "pinch_center_policy": "diagnostic_only",
    "phase_completion": "accepted_sample_quota",
    "advance_key": "space",
    "start_trigger": "first_software_gate_accepted",
}
PHASE_LABELS = {
    "record_just_touch": "contact",
    "record_press_hard": "press",
    "record_return_touch": "contact",
}
TIP_LABELS = ("thumb_tip", "index_tip")
MIN_VALID_GROUPS = 5
MIN_IR_EBA = 0.75
MIN_IR_GAIN_OVER_GEOMETRY = 0.10
MIN_CONSISTENT_TIP_GROUPS = 4
MAX_PRESS_RECOVERY_RATIO = 0.50


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if isfinite(value) else None


def _finite_pair(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    values = tuple(_finite_number(item) for item in value)
    return values if all(item is not None for item in values) else None


def _median(values):
    return float(np.median(np.asarray(values, dtype=float)))


def _median_pair(values):
    return tuple(_median([value[index] for value in values]) for index in (0, 1))


def _centered_effect(phase_values):
    return (
        phase_values["record_press_hard"]
        - 0.5
        * (
            phase_values["record_just_touch"]
            + phase_values["record_return_touch"]
        )
    )


def _metadata_valid(rows):
    metadata_rows = [
        row for row in rows if row.get("row_type") == "metadata"
    ]
    summary_rows = [
        row for row in rows if row.get("row_type") == "summary"
    ]
    if len(metadata_rows) != 1 or len(summary_rows) != 1:
        return False
    metadata = metadata_rows[0]
    summary = summary_rows[0]
    valid_groups = summary.get("pinch_signal_valid_groups")
    invalid_groups = summary.get("pinch_signal_invalid_groups")
    if (
        isinstance(valid_groups, bool)
        or not isinstance(valid_groups, int)
        or isinstance(invalid_groups, bool)
        or not isinstance(invalid_groups, int)
        or valid_groups + invalid_groups != GROUP_COUNT
    ):
        return False
    expected_complete = valid_groups >= MIN_VALID_GROUPS
    return (
        metadata.get("status") == "ok"
        and metadata.get("schema_version") == 4
        and metadata.get("safety_mode") == "robot_free_hand_shadow_only"
        and metadata.get("stage0_runtime_sha256")
        == EXPECTED_STAGE0_SHA256
        and metadata.get("frozen_xml_sha256")
        == EXPECTED_FROZEN_XML_SHA256
        and metadata.get("pinch_signal_protocol") == EXPECTED_PROTOCOL
        and summary.get("pinch_signal_started") is True
        and summary.get("pinch_signal_protocol_completed")
        is expected_complete
        and summary.get("pinch_signal_acquisition_blocked")
        is (not expected_complete)
        and summary.get("status")
        == ("ok" if expected_complete else "blocked")
    )


def _empty_bucket():
    return {
        phase: {
            tip: {
                "raw": [],
                "corrected_exact": [],
                "patch": [],
                "corrected_patch": [],
                "uv": [],
                "thermal_pixel": [],
                "depth_m": [],
                "color_pixel": [],
                "depth_pixel": [],
            }
            for tip in TIP_LABELS
        }
        | {
            "pinch_2d": [],
            "center_uv": [],
            "center_shift_px": [],
            "patch_overlap": [],
        }
        for phase in PHASE_LABELS
    }


def _accepted_row_values(row):
    frame_median = _finite_number(row.get("thermal_frame_median_count"))
    center_uv = _finite_pair(row.get("thermal_pinch_center_uv"))
    geometry = row.get("pinch_geometry")
    pinch_2d = (
        None
        if not isinstance(geometry, dict) or geometry.get("valid") is not True
        else _finite_number(geometry.get("pinch_distance_2d_norm"))
    )
    patch_overlap = row.get("thermal_patches_overlap")
    if (
        frame_median is None
        or center_uv is None
        or pinch_2d is None
        or not isinstance(patch_overlap, bool)
    ):
        return None

    tips = {}
    for tip in row.get("fingertips", ()):
        label = tip.get("label")
        if label not in TIP_LABELS or label in tips:
            continue
        raw = _finite_number(tip.get("thermal_raw_count"))
        patch = _finite_number(tip.get("thermal_patch_3x3_mean_count"))
        uv = _finite_pair(tip.get("thermal_uv"))
        thermal_pixel = _finite_pair(tip.get("thermal_pixel"))
        depth_m = _finite_number(tip.get("depth_m"))
        color_pixel = _finite_pair(tip.get("color_pixel"))
        depth_pixel = _finite_pair(tip.get("depth_pixel"))
        if None in (
            raw,
            patch,
            uv,
            thermal_pixel,
            depth_m,
            color_pixel,
            depth_pixel,
        ):
            continue
        tips[label] = {
            "raw": raw,
            "corrected_exact": raw - frame_median,
            "patch": patch,
            "corrected_patch": patch - frame_median,
            "uv": uv,
            "thermal_pixel": thermal_pixel,
            "depth_m": depth_m,
            "color_pixel": color_pixel,
            "depth_pixel": depth_pixel,
        }
    if set(tips) != set(TIP_LABELS):
        return None
    return {
        "tips": tips,
        "pinch_2d": pinch_2d,
        "center_uv": center_uv,
        "patch_overlap": patch_overlap,
    }


def _extract_groups(rows):
    buckets = [_empty_bucket() for _index in range(GROUP_COUNT)]
    accepted_recording_rows = 0
    for row in rows:
        if (
            row.get("row_type") != "attempt"
            or row.get("status") != "software_gate_accepted"
        ):
            continue
        cue = row.get("pinch_signal")
        if (
            not isinstance(cue, dict)
            or cue.get("recording") is not True
            or cue.get("quota_accepted") is not True
            or cue.get("quota_reasons") != []
            or cue.get("target_valid_samples") != TARGET_VALID_SAMPLES
        ):
            continue
        group_index = cue.get("group_index")
        phase = cue.get("phase")
        if (
            isinstance(group_index, bool)
            or not isinstance(group_index, int)
            or not 0 <= group_index < GROUP_COUNT
            or phase not in PHASE_LABELS
            or cue.get("label") != PHASE_LABELS[phase]
        ):
            continue
        values = _accepted_row_values(row)
        if values is None:
            continue
        center_shift_px = _finite_number(row.get("pinch_center_shift_px"))
        if phase != "record_just_touch" and center_shift_px is None:
            continue

        bucket = buckets[group_index][phase]
        for tip in TIP_LABELS:
            for feature, value in values["tips"][tip].items():
                bucket[tip][feature].append(value)
        bucket["pinch_2d"].append(values["pinch_2d"])
        bucket["center_uv"].append(values["center_uv"])
        if center_shift_px is not None:
            bucket["center_shift_px"].append(center_shift_px)
        bucket["patch_overlap"].append(values["patch_overlap"])
        accepted_recording_rows += 1

    groups = []
    for group_index, bucket in enumerate(buckets):
        phase_rows = {
            phase: len(bucket[phase]["pinch_2d"]) for phase in PHASE_LABELS
        }
        valid = all(
            count == TARGET_VALID_SAMPLES for count in phase_rows.values()
        )
        group = {
            "group_index": group_index,
            "valid": valid,
            "phase_rows": phase_rows,
            "primary_effect": None,
            "patch_effect": None,
            "geometry_effect": None,
            "recovery_delta": None,
            "fingertips": {},
        }
        if not valid:
            groups.append(group)
            continue

        geometry_by_phase = {
            phase: _median(bucket[phase]["pinch_2d"])
            for phase in PHASE_LABELS
        }
        tip_effects = {}
        patch_effects = {}
        return_deltas = {}
        for tip in TIP_LABELS:
            feature_by_phase = {}
            for feature in (
                "raw",
                "corrected_exact",
                "patch",
                "corrected_patch",
            ):
                feature_by_phase[feature] = {
                    phase: _median(bucket[phase][tip][feature])
                    for phase in PHASE_LABELS
                }
            uv_by_phase = {
                phase: _median_pair(bucket[phase][tip]["uv"])
                for phase in PHASE_LABELS
            }
            thermal_pixel_by_phase = {
                phase: _median_pair(bucket[phase][tip]["thermal_pixel"])
                for phase in PHASE_LABELS
            }
            depth_by_phase = {
                phase: _median(bucket[phase][tip]["depth_m"])
                for phase in PHASE_LABELS
            }
            color_by_phase = {
                phase: _median_pair(bucket[phase][tip]["color_pixel"])
                for phase in PHASE_LABELS
            }
            depth_pixel_by_phase = {
                phase: _median_pair(bucket[phase][tip]["depth_pixel"])
                for phase in PHASE_LABELS
            }
            corrected = feature_by_phase["corrected_exact"]
            tip_effects[tip] = _centered_effect(corrected)
            patch_effects[tip] = _centered_effect(
                feature_by_phase["corrected_patch"]
            )
            return_deltas[tip] = (
                corrected["record_return_touch"]
                - corrected["record_just_touch"]
            )
            group["fingertips"][tip] = {
                "phase_medians": {
                    phase: {
                        "raw_count": feature_by_phase["raw"][phase],
                        "corrected_exact_count": (
                            feature_by_phase["corrected_exact"][phase]
                        ),
                        "patch_3x3_mean_count": (
                            feature_by_phase["patch"][phase]
                        ),
                        "corrected_patch_3x3_mean_count": (
                            feature_by_phase["corrected_patch"][phase]
                        ),
                        "thermal_uv": list(uv_by_phase[phase]),
                        "thermal_pixel": list(
                            thermal_pixel_by_phase[phase]
                        ),
                        "depth_m": depth_by_phase[phase],
                        "color_pixel": list(color_by_phase[phase]),
                        "depth_pixel": list(
                            depth_pixel_by_phase[phase]
                        ),
                    }
                    for phase in PHASE_LABELS
                },
                "raw_pre_touch_delta": (
                    feature_by_phase["raw"]["record_press_hard"]
                    - feature_by_phase["raw"]["record_just_touch"]
                ),
                "corrected_exact_effect": tip_effects[tip],
                "corrected_patch_effect": patch_effects[tip],
                "corrected_return_delta": return_deltas[tip],
                "press_uv_shift_px": hypot(
                    uv_by_phase["record_press_hard"][0]
                    - uv_by_phase["record_just_touch"][0],
                    uv_by_phase["record_press_hard"][1]
                    - uv_by_phase["record_just_touch"][1],
                ),
                "return_uv_shift_px": hypot(
                    uv_by_phase["record_return_touch"][0]
                    - uv_by_phase["record_just_touch"][0],
                    uv_by_phase["record_return_touch"][1]
                    - uv_by_phase["record_just_touch"][1],
                ),
                "press_thermal_pixel_shift_px": hypot(
                    thermal_pixel_by_phase["record_press_hard"][0]
                    - thermal_pixel_by_phase["record_just_touch"][0],
                    thermal_pixel_by_phase["record_press_hard"][1]
                    - thermal_pixel_by_phase["record_just_touch"][1],
                ),
                "return_thermal_pixel_shift_px": hypot(
                    thermal_pixel_by_phase["record_return_touch"][0]
                    - thermal_pixel_by_phase["record_just_touch"][0],
                    thermal_pixel_by_phase["record_return_touch"][1]
                    - thermal_pixel_by_phase["record_just_touch"][1],
                ),
                "press_depth_shift_m": (
                    depth_by_phase["record_press_hard"]
                    - depth_by_phase["record_just_touch"]
                ),
                "return_depth_shift_m": (
                    depth_by_phase["record_return_touch"]
                    - depth_by_phase["record_just_touch"]
                ),
                "press_color_shift_px": hypot(
                    color_by_phase["record_press_hard"][0]
                    - color_by_phase["record_just_touch"][0],
                    color_by_phase["record_press_hard"][1]
                    - color_by_phase["record_just_touch"][1],
                ),
                "return_color_shift_px": hypot(
                    color_by_phase["record_return_touch"][0]
                    - color_by_phase["record_just_touch"][0],
                    color_by_phase["record_return_touch"][1]
                    - color_by_phase["record_just_touch"][1],
                ),
                "press_depth_pixel_shift_px": hypot(
                    depth_pixel_by_phase["record_press_hard"][0]
                    - depth_pixel_by_phase["record_just_touch"][0],
                    depth_pixel_by_phase["record_press_hard"][1]
                    - depth_pixel_by_phase["record_just_touch"][1],
                ),
                "return_depth_pixel_shift_px": hypot(
                    depth_pixel_by_phase["record_return_touch"][0]
                    - depth_pixel_by_phase["record_just_touch"][0],
                    depth_pixel_by_phase["record_return_touch"][1]
                    - depth_pixel_by_phase["record_just_touch"][1],
                ),
            }

        group.update(
            primary_effect=_median(list(tip_effects.values())),
            patch_effect=_median(list(patch_effects.values())),
            geometry_effect=_centered_effect(geometry_by_phase),
            recovery_delta=_median(list(return_deltas.values())),
            press_center_shift_px=_median(
                bucket["record_press_hard"]["center_shift_px"]
            ),
            return_center_shift_px=_median(
                bucket["record_return_touch"]["center_shift_px"]
            ),
            patch_overlap_rate=(
                sum(
                    bool(value)
                    for phase in PHASE_LABELS
                    for value in bucket[phase]["patch_overlap"]
                )
                / (len(PHASE_LABELS) * TARGET_VALID_SAMPLES)
            ),
        )
        groups.append(group)
    return groups, accepted_recording_rows


def _paired_leave_one_group_out_eba(groups, feature):
    correct_contact = 0
    correct_press = 0
    for held_out in groups:
        value = held_out.get(feature)
        if not held_out["valid"] or value is None:
            continue
        training_press = [
            group[feature]
            for group in groups
            if group is not held_out
            and group["valid"]
            and group.get(feature) is not None
        ]
        if not training_press:
            continue
        press_median = _median(training_press)
        midpoint = press_median / 2.0

        def predict(observation):
            if press_median > 0.0:
                return "press" if observation >= midpoint else "contact"
            if press_median < 0.0:
                return "press" if observation <= midpoint else "contact"
            return "contact"

        if predict(0.0) == "contact":
            correct_contact += 1
        if predict(value) == "press":
            correct_press += 1
    return (
        correct_contact / GROUP_COUNT + correct_press / GROUP_COUNT
    ) / 2.0


def _tip_direction_summary(groups, tip):
    effects = [
        group["fingertips"][tip]["corrected_exact_effect"]
        for group in groups
        if group["valid"]
    ]
    if not effects:
        return {
            "median_effect": None,
            "consistent_groups": 0,
            "direction": None,
        }
    median_effect = _median(effects)
    direction = 1 if median_effect > 0.0 else -1 if median_effect < 0.0 else 0
    consistent = sum(effect * direction > 0.0 for effect in effects)
    return {
        "median_effect": median_effect,
        "consistent_groups": consistent,
        "direction": direction,
    }


def analyze_document(rows):
    if not isinstance(rows, list) or not all(
        isinstance(row, dict) for row in rows
    ):
        raise ValueError("rows must be a list of JSON objects")

    metadata_valid = _metadata_valid(rows)
    groups, accepted_rows = _extract_groups(rows)
    valid_groups = sum(group["valid"] for group in groups)
    ir_eba = _paired_leave_one_group_out_eba(groups, "primary_effect")
    geometry_eba = _paired_leave_one_group_out_eba(
        groups,
        "geometry_effect",
    )
    ir_gain = ir_eba - geometry_eba
    valid = [group for group in groups if group["valid"]]
    tip_summaries = {
        tip: _tip_direction_summary(groups, tip) for tip in TIP_LABELS
    }
    recovery_ratio_by_tip = {}
    for tip in TIP_LABELS:
        if not valid:
            recovery_ratio_by_tip[tip] = None
            continue
        recovery_ratio_by_tip[tip] = _median(
            [
                abs(
                    group["fingertips"][tip][
                        "corrected_return_delta"
                    ]
                )
                for group in valid
            ]
        ) / max(
            _median(
                [
                    abs(
                        group["fingertips"][tip][
                            "corrected_exact_effect"
                        ]
                    )
                    for group in valid
                ]
            ),
            1.0,
        )
    recovery_ratio = (
        max(recovery_ratio_by_tip.values()) if valid else None
    )
    exact_median = (
        _median([group["primary_effect"] for group in valid])
        if valid
        else None
    )
    patch_median = (
        _median([group["patch_effect"] for group in valid])
        if valid
        else None
    )

    reasons = []
    if not metadata_valid:
        reasons.append("metadata_mismatch")
    if valid_groups < MIN_VALID_GROUPS:
        reasons.append("insufficient_valid_groups")
    acquisition_blocked = bool(reasons)
    if not acquisition_blocked:
        if ir_eba < MIN_IR_EBA:
            reasons.append("ir_eba_below_threshold")
        if ir_gain < MIN_IR_GAIN_OVER_GEOMETRY:
            reasons.append("ir_gain_over_geometry_below_threshold")
        tip_directions = [
            tip_summaries[tip]["direction"] for tip in TIP_LABELS
        ]
        if (
            0 in tip_directions
            or len(set(tip_directions)) != 1
            or any(
                tip_summaries[tip]["consistent_groups"]
                < MIN_CONSISTENT_TIP_GROUPS
                for tip in TIP_LABELS
            )
        ):
            reasons.append("per_tip_press_direction_inconsistent")
        if (
            exact_median is None
            or patch_median is None
            or exact_median == 0.0
            or patch_median == 0.0
            or exact_median * patch_median < 0.0
        ):
            reasons.append("sampling_method_sign_disagreement")
        if (
            recovery_ratio is None
            or recovery_ratio > MAX_PRESS_RECOVERY_RATIO
        ):
            reasons.append("press_recovery_ratio_above_threshold")

    if acquisition_blocked:
        verdict = "BLOCKED_ACQUISITION"
    elif reasons:
        verdict = "STOP_BEFORE_STAGE1F"
    else:
        verdict = "PROCEED_TO_STAGE1F_SHADOW"

    return {
        "schema_version": 4,
        "verdict": verdict,
        "reasons": reasons,
        "accepted_recording_rows": accepted_rows,
        "valid_groups": valid_groups,
        "ir": {
            "feature": "centered_common_mode_corrected_exact_tip_count",
            "paired_eba": ir_eba,
            "median_effect": exact_median,
            "patch_median_effect": patch_median,
        },
        "fingertips": tip_summaries,
        "geometry": {
            "feature": "centered_pinch_distance_2d_norm",
            "paired_eba": geometry_eba,
        },
        "ir_gain_over_geometry": ir_gain,
        "press_recovery_ratio": recovery_ratio,
        "press_recovery_ratio_by_tip": recovery_ratio_by_tip,
        "thresholds": {
            "min_valid_groups": MIN_VALID_GROUPS,
            "target_valid_samples": TARGET_VALID_SAMPLES,
            "min_ir_eba": MIN_IR_EBA,
            "min_ir_gain_over_geometry": MIN_IR_GAIN_OVER_GEOMETRY,
            "min_consistent_tip_groups": MIN_CONSISTENT_TIP_GROUPS,
            "max_press_recovery_ratio": MAX_PRESS_RECOVERY_RATIO,
        },
        "groups": groups,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = analyze_document(rows)
    text = json.dumps(
        result,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.write("\n")
    print(text)


if __name__ == "__main__":
    raise SystemExit(main())
