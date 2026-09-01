"""Thermal-only descriptive salvage for sessions with failed D435 ROI tracking."""

from __future__ import annotations

from math import hypot
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


SEGMENT_OFFSET_COUNT = 100.0
SEGMENT_MIN_X = 50
SEGMENT_MIN_AREA_PX = 200
SEGMENT_MIN_CENTROID_X = 80.0
TIP_LEFT_QUANTILE = 3.0
DISTAL_INSET_PX = 5.0
REFERENCE_INSET_PX = 25.0


def _patch_mean(
    frame: np.ndarray,
    center: np.ndarray,
    size: int,
    *,
    label: str,
) -> float:
    x, y = np.rint(center).astype(int)
    radius = size // 2
    if (
        x - radius < 0
        or y - radius < 0
        or x + radius >= frame.shape[1]
        or y + radius >= frame.shape[0]
    ):
        raise ValueError(f"{label}_out_of_bounds")
    return float(
        np.mean(
            frame[
                y - radius : y + radius + 1,
                x - radius : x + radius + 1,
            ],
            dtype=np.float64,
        )
    )


def thermal_only_feature(frame) -> dict:
    frame = np.asarray(frame)
    if frame.shape != (120, 160) or frame.dtype != np.uint16:
        raise ValueError("thermal_frame_invalid")
    threshold = float(np.median(frame)) + SEGMENT_OFFSET_COUNT
    mask = (frame >= threshold).astype(np.uint8)
    mask[:, :SEGMENT_MIN_X] = 0
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
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
    y_pixels, x_pixels = np.nonzero(labels == component)
    left_cutoff = float(np.percentile(x_pixels, TIP_LEFT_QUANTILE))
    tip_band = x_pixels <= left_cutoff
    tip = np.asarray(
        [
            np.median(x_pixels[tip_band]),
            np.median(y_pixels[tip_band]),
        ],
        dtype=float,
    )
    component_center = np.asarray(
        [np.median(x_pixels), np.median(y_pixels)],
        dtype=float,
    )
    direction = component_center - tip
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("right_hot_hand_axis_invalid")
    direction /= norm
    distal = tip + DISTAL_INSET_PX * direction
    reference = tip + REFERENCE_INSET_PX * direction
    distal_mean = _patch_mean(
        frame,
        distal,
        3,
        label="distal_3x3",
    )
    reference_mean = _patch_mean(
        frame,
        reference,
        5,
        label="reference_5x5",
    )
    return {
        "threshold_count": threshold,
        "component_area_px": int(stats[component, cv2.CC_STAT_AREA]),
        "tip_uv": tip.tolist(),
        "distal_uv": distal.tolist(),
        "reference_uv": reference.tolist(),
        "distal_3x3_mean_count": distal_mean,
        "reference_5x5_mean_count": reference_mean,
        "primary_signal_count": distal_mean - reference_mean,
    }


def _bin_curve(
    samples: list[dict],
    field: str,
) -> tuple[list[float], list[int]]:
    values = []
    missing = []
    for bin_index in range(ANALYSIS_BINS):
        start = ANALYSIS_START_S + bin_index * BIN_DURATION_S
        end = start + BIN_DURATION_S
        in_bin = [
            float(sample[field])
            for sample in samples
            if start <= sample["time_s"] < end
        ]
        if in_bin:
            values.append(float(np.median(in_bin)))
        else:
            values.append(float("nan"))
            missing.append(bin_index)
    array = np.asarray(values, dtype=float)
    if missing:
        valid = np.flatnonzero(np.isfinite(array))
        if valid.size < 2:
            raise ValueError(f"insufficient_bins:{field}")
        absent = np.flatnonzero(~np.isfinite(array))
        array[absent] = np.interp(absent, valid, array[valid])
    return array.tolist(), missing


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
                "maximum_absolute_press_minus_null_count": float(
                    np.max(np.abs(values))
                ),
            }
        )
    return clusters


def analyze_thermal_only(rows, *, frame_loader) -> dict:
    grouped: dict[tuple[int, str], list[dict]] = {}
    segmentation_failures = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("row_type") != "frame"
            or row.get("condition") not in ("null", "press")
            or row.get("block_index") not in range(6)
            or not persisted_flag_is_true(row.get("artifact_write_ok"))
        ):
            continue
        try:
            feature = thermal_only_feature(frame_loader(row))
        except (OSError, ValueError) as exc:
            segmentation_failures.append(
                {
                    "frame_index": row.get("frame_index"),
                    "reason": str(exc),
                }
            )
            continue
        grouped.setdefault(
            (row["block_index"], row["condition"]),
            [],
        ).append(
            {
                "time_s": float(row["global_elapsed_s"]),
                **feature,
            }
        )
    expected = {
        (block, condition)
        for block in range(6)
        for condition in ("null", "press")
    }
    if set(grouped) != expected:
        raise ValueError("six complete primary block labels required")

    curves = []
    binned = {
        "primary_signal_count": {"null": [], "press": []},
        "distal_displacement_px": {"null": [], "press": []},
    }
    interpolated = []
    for block_index in range(6):
        for condition in ("null", "press"):
            samples = sorted(
                grouped[(block_index, condition)],
                key=lambda sample: sample["time_s"],
            )
            baseline_samples = [
                sample
                for sample in samples
                if 3.0 <= sample["time_s"] < 5.0
            ]
            if not baseline_samples:
                raise ValueError("A1 baseline missing")
            signal_baseline = float(
                np.median(
                    [
                        sample["primary_signal_count"]
                        for sample in baseline_samples
                    ]
                )
            )
            distal_baseline = np.median(
                [sample["distal_uv"] for sample in baseline_samples],
                axis=0,
            )
            normalized_samples = []
            for sample in samples:
                distal = np.asarray(sample["distal_uv"], dtype=float)
                normalized_samples.append(
                    {
                        **sample,
                        "normalized_count": (
                            sample["primary_signal_count"]
                            - signal_baseline
                        ),
                        "distal_displacement_px": hypot(
                            *(distal - distal_baseline)
                        ),
                    }
                )
            curve = {
                "block_index": block_index,
                "condition": condition,
                "baseline_count": signal_baseline,
                "time_s": [
                    sample["time_s"] for sample in normalized_samples
                ],
                "normalized_count": [
                    sample["normalized_count"]
                    for sample in normalized_samples
                ],
                "distal_u_px": [
                    sample["distal_uv"][0]
                    for sample in normalized_samples
                ],
                "distal_v_px": [
                    sample["distal_uv"][1]
                    for sample in normalized_samples
                ],
            }
            curves.append(curve)
            for output_field, sample_field in (
                ("primary_signal_count", "normalized_count"),
                ("distal_displacement_px", "distal_displacement_px"),
            ):
                values, missing = _bin_curve(
                    normalized_samples,
                    sample_field,
                )
                binned[output_field][condition].append(values)
                interpolated.extend(
                    {
                        "block_index": block_index,
                        "condition": condition,
                        "field": output_field,
                        "bin_index": bin_index,
                    }
                    for bin_index in missing
                )

    thermal_differences = (
        np.asarray(binned["primary_signal_count"]["press"], dtype=float)
        - np.asarray(binned["primary_signal_count"]["null"], dtype=float)
    )
    geometry_differences = (
        np.asarray(binned["distal_displacement_px"]["press"], dtype=float)
        - np.asarray(binned["distal_displacement_px"]["null"], dtype=float)
    )
    thermal_test = exact_cluster_test(thermal_differences)
    geometry_test = exact_cluster_test(geometry_differences)
    return {
        "analysis_role": "salvage_descriptive_not_preregistered",
        "formal_primary_verdict": "INCOMPLETE_FOR_PRIMARY_TEST",
        "formal_primary_reason": (
            "all captured rows failed the frozen D435-to-thermal ROI chain"
        ),
        "selected_pair_count": 6,
        "roi_method": {
            "segmentation_threshold": "frame_median_plus_100_count",
            "search_x_min": SEGMENT_MIN_X,
            "component": "largest_right_hot_component",
            "tip_definition": "leftmost_3_percent_band_median",
            "distal_inset_px": DISTAL_INSET_PX,
            "reference_inset_px": REFERENCE_INSET_PX,
            "distal_patch": "3x3_mean",
            "reference_patch": "5x5_mean",
        },
        "segmentation_failure_count": len(segmentation_failures),
        "segmentation_failures": segmentation_failures,
        "interpolated_bin_count": len(
            {
                (
                    item["block_index"],
                    item["condition"],
                    item["bin_index"],
                )
                for item in interpolated
            }
        ),
        "interpolated_bins": interpolated,
        "curves": curves,
        "binned": binned,
        "thermal_only_exploratory": {
            "test": thermal_test,
            "significant_clusters": _significant_clusters(
                thermal_test,
                thermal_differences,
            ),
            "median_press_minus_null_by_phase_count": {
                "X": float(np.median(thermal_differences[:, 0:10])),
                "A2": float(np.median(thermal_differences[:, 10:20])),
                "A3": float(np.median(thermal_differences[:, 20:30])),
            },
        },
        "thermal_geometry_diagnostic": {
            "test": geometry_test,
            "significant_clusters": _significant_clusters(
                geometry_test,
                geometry_differences,
            ),
        },
    }


def plot_salvage_curves(result: dict, path: str | Path) -> dict:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path = Path(path)
    figure, axis = plt.subplots(figsize=(11, 6))
    colors = {"null": "#2f6fb0", "press": "#e57b25"}
    condition_lines = []
    for curve in result["curves"]:
        (line,) = axis.plot(
            curve["time_s"],
            curve["normalized_count"],
            color=colors[curve["condition"]],
            linewidth=1.1,
            alpha=0.68,
        )
        condition_lines.append(line)
    phase_boundaries = [5.0, 10.0, 15.0]
    for boundary in phase_boundaries:
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
    axis.set(
        xlim=(0.0, 20.0),
        xlabel="Time from A1 start (s)",
        ylabel="Thermal-only distal - reference, baseline-subtracted (count)",
        title=(
            "POSTHOC THERMAL-ONLY SALVAGE — NOT PREREGISTERED PRIMARY\n"
            "Six paired Null (blue) and Press (orange) trials"
        ),
    )
    axis.grid(alpha=0.18)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return {
        "condition_curve_count": len(condition_lines),
        "phase_boundaries_s": phase_boundaries,
        "analysis_role": result["analysis_role"],
    }


def plot_feature_overlay(
    frame,
    feature: dict,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    frame = np.asarray(frame)
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.imshow(frame, cmap="inferno")
    for label, color in (
        ("tip", "#65d1ff"),
        ("distal", "#5df28b"),
        ("reference", "#ffffff"),
    ):
        u, v = feature[f"{label}_uv"]
        axis.scatter(
            [u],
            [v],
            s=80,
            facecolors="none",
            edgecolors=color,
            linewidths=1.8,
            label=label,
        )
    axis.set(
        title="POSTHOC thermal-only ROI definition — not primary",
        xlim=(0, frame.shape[1] - 1),
        ylim=(frame.shape[0] - 1, 0),
    )
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(Path(path), dpi=180)
    plt.close(figure)
