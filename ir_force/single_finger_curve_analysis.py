"""Paired inference and plotting for continuous single-finger thermal curves."""

from __future__ import annotations

import csv
from itertools import product
import json
from math import isfinite, sqrt
from pathlib import Path

import numpy as np

from ir_force.single_finger_curve_protocol import (
    BIN_DURATION_S,
    persisted_flag_is_true,
    trial_integrity,
)


ANALYSIS_START_S = 5.0
ANALYSIS_END_S = 20.0
ANALYSIS_BINS = 30
T_THRESHOLD_DF5 = 2.5706


def _frame_groups(rows) -> dict[tuple[int, str], list[dict]]:
    groups: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("row_type") != "frame":
            continue
        condition = row.get("condition")
        block = row.get("block_index")
        if condition not in ("null", "press") or not isinstance(block, int):
            continue
        groups.setdefault((block, condition), []).append(row)
    for trial_rows in groups.values():
        trial_rows.sort(key=lambda row: float(row.get("global_elapsed_s", -1.0)))
    return groups


def _finite_value(row: dict, field: str) -> float | None:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if isfinite(value) else None


def _median(values, *, label: str) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(label)
    return float(np.median(values))


def _trial_curve(block_index: int, condition: str, rows: list[dict]) -> dict:
    baseline_rows = [
        row
        for row in rows
        if persisted_flag_is_true(row.get("tracking_valid"))
        and row.get("phase") == "A1"
        and (elapsed := _finite_value(row, "phase_elapsed_s")) is not None
        and 3.0 <= elapsed < 5.0
    ]
    baseline_count = _median(
        [
            value
            for row in baseline_rows
            if (value := _finite_value(row, "primary_signal_count")) is not None
        ],
        label="primary_baseline_missing",
    )
    baseline_u = _median(
        [
            value
            for row in baseline_rows
            if (value := _finite_value(row, "distal_thermal_u_px")) is not None
        ],
        label="u_baseline_missing",
    )
    baseline_v = _median(
        [
            value
            for row in baseline_rows
            if (value := _finite_value(row, "distal_thermal_v_px")) is not None
        ],
        label="v_baseline_missing",
    )
    baseline_depth = _median(
        [
            value
            for row in baseline_rows
            if (value := _finite_value(row, "distal_depth_m")) is not None
        ],
        label="depth_baseline_missing",
    )

    time_s = []
    normalized_count = []
    uv_displacement_px = []
    depth_change_m = []
    for row in rows:
        if not persisted_flag_is_true(row.get("tracking_valid")):
            continue
        elapsed = _finite_value(row, "global_elapsed_s")
        count = _finite_value(row, "primary_signal_count")
        u = _finite_value(row, "distal_thermal_u_px")
        v = _finite_value(row, "distal_thermal_v_px")
        depth = _finite_value(row, "distal_depth_m")
        if None in (elapsed, count, u, v, depth):
            continue
        time_s.append(elapsed)
        normalized_count.append(count - baseline_count)
        uv_displacement_px.append(
            sqrt((u - baseline_u) ** 2 + (v - baseline_v) ** 2)
        )
        depth_change_m.append(depth - baseline_depth)
    return {
        "block_index": block_index,
        "condition": condition,
        "baseline_count": baseline_count,
        "baseline_distal_u_px": baseline_u,
        "baseline_distal_v_px": baseline_v,
        "baseline_depth_m": baseline_depth,
        "time_s": time_s,
        "normalized_count": normalized_count,
        "uv_displacement_px": uv_displacement_px,
        "depth_change_m": depth_change_m,
    }


def _bin_curve(curve: dict, field: str) -> list[float]:
    bins: list[list[float]] = [[] for _ in range(ANALYSIS_BINS)]
    for elapsed, value in zip(
        curve["time_s"],
        curve[field],
        strict=True,
    ):
        if ANALYSIS_START_S <= elapsed < ANALYSIS_END_S:
            bin_index = int(
                (elapsed - ANALYSIS_START_S) // BIN_DURATION_S
            )
            bins[bin_index].append(float(value))
    if any(not values for values in bins):
        raise ValueError(f"incomplete_bins:{field}")
    return [float(np.median(values)) for values in bins]


def bin_selected_pairs(rows) -> dict:
    rows = list(rows)
    groups = _frame_groups(rows)
    valid_blocks = []
    incomplete_blocks = {}
    for block_index in sorted({key[0] for key in groups}):
        by_condition = {
            condition: groups.get((block_index, condition), [])
            for condition in ("null", "press")
        }
        reasons = {}
        for condition, trial_rows in by_condition.items():
            integrity = trial_integrity(trial_rows)
            if not integrity["valid"]:
                reasons[condition] = integrity["reasons"]
        if not reasons and all(by_condition.values()):
            valid_blocks.append(block_index)
        else:
            incomplete_blocks[str(block_index)] = reasons or {
                "pair": ["condition_missing"]
            }
    selected_blocks = valid_blocks[:6]
    curves = []
    curve_failures = {}
    for block_index in selected_blocks:
        for condition in ("null", "press"):
            try:
                curves.append(
                    _trial_curve(
                        block_index,
                        condition,
                        groups[(block_index, condition)],
                    )
                )
            except ValueError as exc:
                curve_failures[f"{block_index}:{condition}"] = str(exc)
    if curve_failures:
        failed_blocks = {
            int(key.split(":", maxsplit=1)[0]) for key in curve_failures
        }
        selected_blocks = [
            block for block in selected_blocks if block not in failed_blocks
        ]
        curves = [
            curve
            for curve in curves
            if curve["block_index"] in selected_blocks
        ]

    binned = {
        field: {"null": [], "press": []}
        for field in (
            "primary_signal_count",
            "uv_displacement_px",
            "depth_change_m",
        )
    }
    field_map = {
        "primary_signal_count": "normalized_count",
        "uv_displacement_px": "uv_displacement_px",
        "depth_change_m": "depth_change_m",
    }
    bin_failures = {}
    for curve in curves:
        for output_field, curve_field in field_map.items():
            try:
                values = _bin_curve(curve, curve_field)
            except ValueError as exc:
                bin_failures[
                    f"{curve['block_index']}:{curve['condition']}:{output_field}"
                ] = str(exc)
                continue
            binned[output_field][curve["condition"]].append(values)

    complete = (
        len(selected_blocks) == 6
        and len(curves) == 12
        and not curve_failures
        and not bin_failures
        and all(
            len(by_condition[condition]) == 6
            for by_condition in binned.values()
            for condition in ("null", "press")
        )
    )
    selected_rows = [
        row
        for row in rows
        if row.get("block_index") in selected_blocks
        and row.get("condition") in ("null", "press")
        and row.get("row_type") == "frame"
    ]
    temperature_scale_k = (
        0.01
        if selected_rows
        and all(
            persisted_flag_is_true(row.get("tlinear_enabled"))
            and row.get("tlinear_resolution_k") == 0.01
            for row in selected_rows
        )
        else None
    )
    return {
        "complete": complete,
        "selected_blocks": selected_blocks,
        "selected_pair_count": len(selected_blocks),
        "invalid_blocks": incomplete_blocks,
        "curve_failures": curve_failures,
        "bin_failures": bin_failures,
        "temperature_scale_k": temperature_scale_k,
        "curves": curves,
        "binned": binned,
    }


def _t_statistics(differences: np.ndarray) -> np.ndarray:
    count = differences.shape[0]
    means = np.mean(differences, axis=0)
    standard_deviations = np.std(differences, axis=0, ddof=1)
    denominator = standard_deviations / np.sqrt(count)
    statistics = np.zeros(differences.shape[1], dtype=float)
    finite_denominator = denominator > 0.0
    statistics[finite_denominator] = (
        means[finite_denominator] / denominator[finite_denominator]
    )
    zero_variance_effect = (~finite_denominator) & (means != 0.0)
    statistics[zero_variance_effect] = np.copysign(
        np.inf,
        means[zero_variance_effect],
    )
    return statistics


def _clusters_from_t(statistics: np.ndarray) -> list[dict]:
    clusters = []
    start = None
    sign = None
    for bin_index, statistic in enumerate(statistics):
        active = abs(float(statistic)) > T_THRESHOLD_DF5
        current_sign = 1 if statistic > 0 else -1
        if active and start is not None and current_sign == sign:
            continue
        if start is not None:
            end = bin_index - 1
            clusters.append(
                {
                    "start_bin": start,
                    "end_bin": end,
                    "sign": sign,
                    "mass": float(
                        np.sum(np.abs(statistics[start : end + 1]))
                    ),
                }
            )
            start = None
            sign = None
        if active:
            start = bin_index
            sign = current_sign
    if start is not None:
        end = len(statistics) - 1
        clusters.append(
            {
                "start_bin": start,
                "end_bin": end,
                "sign": sign,
                "mass": float(np.sum(np.abs(statistics[start : end + 1]))),
            }
        )
    return clusters


def exact_cluster_test(differences) -> dict:
    differences = np.asarray(differences, dtype=float)
    if differences.shape != (6, ANALYSIS_BINS):
        raise ValueError("differences must have shape (6, 30)")
    if not np.all(np.isfinite(differences)):
        raise ValueError("differences must be finite")
    statistics = _t_statistics(differences)
    clusters = _clusters_from_t(statistics)
    maximum_masses = []
    for signs in product((-1.0, 1.0), repeat=6):
        permuted = differences * np.asarray(signs, dtype=float)[:, None]
        permuted_clusters = _clusters_from_t(_t_statistics(permuted))
        maximum_masses.append(
            max(
                (cluster["mass"] for cluster in permuted_clusters),
                default=0.0,
            )
        )
    for cluster in clusters:
        cluster["p_corrected"] = (
            sum(
                mass >= cluster["mass"]
                for mass in maximum_masses
            )
            / len(maximum_masses)
        )
    return {
        "pairs": 6,
        "bins": ANALYSIS_BINS,
        "t_threshold": T_THRESHOLD_DF5,
        "permutations": len(maximum_masses),
        "t_statistics": statistics.tolist(),
        "clusters": clusters,
        "permutation_max_cluster_masses": maximum_masses,
    }


def _significant_clusters(result: dict) -> list[dict]:
    return [
        cluster
        for cluster in result["clusters"]
        if cluster["p_corrected"] <= 0.05
    ]


def _paired_differences(paired: dict, field: str) -> np.ndarray:
    null = np.asarray(paired["binned"][field]["null"], dtype=float)
    press = np.asarray(paired["binned"][field]["press"], dtype=float)
    return press - null


def _window_summary(differences: np.ndarray, start: int, end: int) -> dict:
    values = differences[:, start:end]
    return {
        "start_s": ANALYSIS_START_S + start * BIN_DURATION_S,
        "end_s": ANALYSIS_START_S + end * BIN_DURATION_S,
        "median_count": float(np.median(values)),
        "maximum_absolute_count": float(np.max(np.abs(values))),
    }


def _thermal_summary(
    test_result: dict,
    differences: np.ndarray,
    temperature_scale_k: float | None,
) -> dict:
    significant = []
    for cluster in _significant_clusters(test_result):
        values = differences[
            :,
            cluster["start_bin"] : cluster["end_bin"] + 1,
        ]
        record = {
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
        if temperature_scale_k is not None:
            record["median_press_minus_null_k"] = (
                record["median_press_minus_null_count"]
                * temperature_scale_k
            )
            record["maximum_absolute_press_minus_null_k"] = (
                record["maximum_absolute_press_minus_null_count"]
                * temperature_scale_k
            )
        significant.append(record)
    x_late = differences[:, 7:10]
    slopes = [
        float(np.polyfit(np.arange(3) * BIN_DURATION_S, pair, 1)[0])
        for pair in x_late
    ]
    return {
        "test": test_result,
        "significant_clusters": significant,
        "descriptives": {
            "x_early": _window_summary(differences, 0, 3),
            "x_middle": _window_summary(differences, 3, 7),
            "x_late": _window_summary(differences, 7, 10),
            "a2_contact": _window_summary(differences, 10, 20),
            "a3_lift": _window_summary(differences, 20, 30),
            "median_x_late_slope_count_per_s": float(np.median(slopes)),
        },
    }


def _clusters_overlap(left: dict, right: dict) -> bool:
    return not (
        left["end_bin"] < right["start_bin"]
        or right["end_bin"] < left["start_bin"]
    )


def analyze_rows(rows) -> dict:
    paired = bin_selected_pairs(rows)
    result = {
        "verdict": "INCOMPLETE_FOR_PRIMARY_TEST",
        "selected_pair_count": paired["selected_pair_count"],
        "selected_blocks": paired["selected_blocks"],
        "invalid_blocks": paired["invalid_blocks"],
        "curve_failures": paired["curve_failures"],
        "bin_failures": paired["bin_failures"],
        "temperature_scale_k": paired["temperature_scale_k"],
    }
    if not paired["complete"]:
        return result

    thermal_differences = _paired_differences(
        paired,
        "primary_signal_count",
    )
    thermal_test = exact_cluster_test(thermal_differences)
    thermal = _thermal_summary(
        thermal_test,
        thermal_differences,
        paired["temperature_scale_k"],
    )
    geometry = {}
    for field in ("uv_displacement_px", "depth_change_m"):
        differences = _paired_differences(paired, field)
        test = exact_cluster_test(differences)
        geometry[field] = {
            "test": test,
            "significant_clusters": _significant_clusters(test),
        }

    thermal_clusters = thermal["significant_clusters"]
    geometry_clusters = [
        cluster
        for field in geometry.values()
        for cluster in field["significant_clusters"]
    ]
    confounded = any(
        _clusters_overlap(thermal_cluster, geometry_cluster)
        for thermal_cluster in thermal_clusters
        for geometry_cluster in geometry_clusters
    )
    if confounded:
        verdict = "GEOMETRY_CONFOUNDED"
    elif thermal_clusters:
        verdict = "SIGNIFICANT_CURVE_SEPARATION"
    else:
        verdict = "NO_DETECTED_SEPARATION_5S"
    result.update(
        verdict=verdict,
        thermal=thermal,
        geometry=geometry,
        paired=paired,
    )
    return result


def plot_all_curves(analysis_input, clusters, path: str | Path) -> dict:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path = Path(path)
    figure, axis = plt.subplots(figsize=(11, 6))
    condition_lines = []
    colors = {"null": "#2f6fb0", "press": "#e57b25"}
    for curve in analysis_input["curves"]:
        (line,) = axis.plot(
            curve["time_s"],
            curve["normalized_count"],
            color=colors[curve["condition"]],
            linewidth=1.2,
            alpha=0.72,
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
    spans = []
    for cluster in clusters:
        if cluster.get("p_corrected", 1.0) > 0.05:
            continue
        start = ANALYSIS_START_S + cluster["start_bin"] * BIN_DURATION_S
        end = (
            ANALYSIS_START_S
            + (cluster["end_bin"] + 1) * BIN_DURATION_S
        )
        axis.axvspan(start, end, color="#8a5cf6", alpha=0.16)
        spans.append((start, end))
    axis.set(
        xlim=(0.0, 20.0),
        xlabel="Time from A1 start (s)",
        ylabel="Distal 3x3 - reference 5x5, baseline-subtracted (count)",
        title="Six paired Null and Press trials",
    )
    axis.grid(alpha=0.18)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    axes_count = len(figure.axes)
    plt.close(figure)
    return {
        "condition_curve_count": len(condition_lines),
        "axes_count": axes_count,
        "phase_boundaries_s": phase_boundaries,
        "significant_cluster_spans_s": spans,
    }


def write_per_frame_csv(rows, path: str | Path) -> None:
    frame_rows = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("row_type") == "frame"
    ]
    fields = sorted({key for row in frame_rows for key in row})
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in frame_rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )
