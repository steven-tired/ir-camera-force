from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from ir_force.ir_report import FeatureRow, TelemetryRow, split_capture_windows


@dataclass(frozen=True)
class CapturePairSummary:
    pair_index: int
    baseline_start: int
    baseline_stop: int
    hold_start: int
    hold_stop: int
    baseline_area_px: float
    hold_area_px: float
    area_delta_px: float
    baseline_mean_delta: float
    hold_mean_delta: float
    hold_max_delta: float
    hold_load_peak: float
    hold_current_peak: float
    hold_load_final: float
    hold_current_final: float
    hold_gripper_mean: float
    flags: tuple[str, ...]


def _as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value.strip() == "":
        raise ValueError(f"missing numeric value for {key}")
    return float(value)


def _mean(rows: list[dict[str, str]], key: str) -> float:
    if not rows:
        return 0.0
    return float(mean(_as_float(row, key) for row in rows))


def _max(rows: list[dict[str, str]], key: str) -> float:
    if not rows:
        return 0.0
    return max(_as_float(row, key) for row in rows)


def _flags(
    *,
    baseline_area_px: float,
    hold_area_px: float,
    area_delta_px: float,
    hold_load_peak: float,
) -> tuple[str, ...]:
    flags: list[str] = []
    if area_delta_px < 0:
        flags.append("area_collapsed")
    if baseline_area_px >= 500:
        flags.append("large_baseline_mask")
    if hold_area_px < 100:
        flags.append("weak_hold_mask")
    if area_delta_px < 0 and hold_load_peak >= 300:
        flags.append("force_spike_without_contact_area")
    return tuple(flags)


def summarize_window_pairs(
    feature_rows: list[FeatureRow],
    telemetry_rows: list[TelemetryRow],
) -> list[CapturePairSummary]:
    if len(feature_rows) != len(telemetry_rows):
        raise ValueError("feature and telemetry row counts must match")
    windows = split_capture_windows(telemetry_rows)
    pairs: list[CapturePairSummary] = []
    for pair_index, window_index in enumerate(range(0, len(windows) - 1, 2)):
        baseline_start, baseline_stop = windows[window_index]
        hold_start, hold_stop = windows[window_index + 1]
        baseline_features = feature_rows[baseline_start:baseline_stop]
        hold_features = feature_rows[hold_start:hold_stop]
        hold_telemetry = telemetry_rows[hold_start:hold_stop]
        baseline_area_px = _mean(baseline_features, "area_px")
        hold_area_px = _mean(hold_features, "area_px")
        area_delta_px = hold_area_px - baseline_area_px
        hold_load_peak = _max(hold_telemetry, "present_load")
        pairs.append(
            CapturePairSummary(
                pair_index=pair_index,
                baseline_start=baseline_start,
                baseline_stop=baseline_stop,
                hold_start=hold_start,
                hold_stop=hold_stop,
                baseline_area_px=baseline_area_px,
                hold_area_px=hold_area_px,
                area_delta_px=area_delta_px,
                baseline_mean_delta=_mean(baseline_features, "mean_delta"),
                hold_mean_delta=_mean(hold_features, "mean_delta"),
                hold_max_delta=_max(hold_features, "max_delta"),
                hold_load_peak=hold_load_peak,
                hold_current_peak=_max(hold_telemetry, "present_current"),
                hold_load_final=_as_float(hold_telemetry[-1], "present_load") if hold_telemetry else 0.0,
                hold_current_final=_as_float(hold_telemetry[-1], "present_current") if hold_telemetry else 0.0,
                hold_gripper_mean=_mean(hold_telemetry, "gripper_pos"),
                flags=_flags(
                    baseline_area_px=baseline_area_px,
                    hold_area_px=hold_area_px,
                    area_delta_px=area_delta_px,
                    hold_load_peak=hold_load_peak,
                ),
            )
        )
    return pairs
