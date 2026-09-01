"""Raw-count analysis primitives for FLIR repeatability experiments."""

from __future__ import annotations

import json
import math
import statistics
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RawROI:
    x: int
    y: int
    width: int
    height: int

    def validate_for(self, *, frame_width: int, frame_height: int) -> None:
        if self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("raw ROI must have non-negative origin and positive size")
        if self.x + self.width > frame_width or self.y + self.height > frame_height:
            raise ValueError(f"raw ROI {self} is outside {frame_width}x{frame_height}")


@dataclass(frozen=True)
class RawFrame:
    frame_index: int
    timestamp_ns: int
    width: int
    height: int
    values: array
    ffc_state: str
    repeated_frame_flag: bool
    display_mapping_mode: str
    raw_low: int | None
    raw_high: int | None


def _median(values: Iterable[int]) -> int:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a median from no values")
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return int(ordered[midpoint])
    return int((ordered[midpoint - 1] + ordered[midpoint]) // 2)


def _roi_values(frame: RawFrame, roi: RawROI) -> list[int]:
    roi.validate_for(frame_width=frame.width, frame_height=frame.height)
    return [
        int(frame.values[y * frame.width + x])
        for y in range(roi.y, roi.y + roi.height)
        for x in range(roi.x, roi.x + roi.width)
    ]


def unique_normal_frames(frames: Iterable[RawFrame]) -> list[RawFrame]:
    return [frame for frame in frames if frame.ffc_state == "normal" and not frame.repeated_frame_flag]


def phase_metric(
    frames: Iterable[RawFrame],
    *,
    roi: RawROI,
    start_ns: int,
    end_ns: int,
) -> dict[str, int | float]:
    if end_ns < start_ns:
        raise ValueError("phase end must be after phase start")
    selected = [
        frame
        for frame in unique_normal_frames(frames)
        if start_ns <= frame.timestamp_ns <= end_ns
    ]
    if not selected:
        raise ValueError("phase contains no unique normal raw frames")
    per_frame_medians = [_median(_roi_values(frame, roi)) for frame in selected]
    return {
        "unique_frame_count": len(selected),
        "median": _median(per_frame_medians),
        "mean": statistics.fmean(per_frame_medians),
        "population_stdev": statistics.pstdev(per_frame_medians),
        "start_ns": selected[0].timestamp_ns,
        "end_ns": selected[-1].timestamp_ns,
    }


def analyze_ffc_recovery(
    frames: Iterable[RawFrame],
    *,
    roi: RawROI,
    window_s: float = 5.0,
    post_delay_s: float = 1.5,
) -> dict[str, int | float]:
    if window_s <= 0.0 or post_delay_s < 0.0:
        raise ValueError("window_s must be positive and post_delay_s must be non-negative")
    ordered = sorted(frames, key=lambda frame: frame.timestamp_ns)
    ffc_frames = [frame for frame in ordered if frame.ffc_state == "ffc"]
    if not ffc_frames:
        raise ValueError("capture contains no FFC frame")
    window_ns = round(window_s * 1_000_000_000)
    post_delay_ns = round(post_delay_s * 1_000_000_000)
    ffc_start_ns = ffc_frames[0].timestamp_ns
    ffc_end_ns = ffc_frames[-1].timestamp_ns
    pre = phase_metric(ordered, roi=roi, start_ns=ffc_start_ns - window_ns, end_ns=ffc_start_ns - 1)
    post = phase_metric(
        ordered,
        roi=roi,
        start_ns=ffc_end_ns + post_delay_ns,
        end_ns=ffc_end_ns + post_delay_ns + window_ns,
    )
    return {
        "ffc_frame_count": len(ffc_frames),
        "ffc_start_ns": ffc_start_ns,
        "ffc_end_ns": ffc_end_ns,
        "pre_median": pre["median"],
        "post_median": post["median"],
        "post_minus_pre_counts": int(post["median"]) - int(pre["median"]),
        "pre_unique_frame_count": pre["unique_frame_count"],
        "post_unique_frame_count": post["unique_frame_count"],
        "pre_population_stdev": pre["population_stdev"],
        "post_population_stdev": post["population_stdev"],
    }


def analyze_ffc_recovery_events(
    frames: Iterable[RawFrame],
    *,
    roi: RawROI,
    window_s: float = 5.0,
    post_delay_s: float = 1.5,
    cluster_gap_s: float = 1.0,
) -> list[dict[str, int | float]]:
    """Measure each distinct FFC interval without merging disjoint shutter events."""
    if window_s <= 0.0 or post_delay_s < 0.0 or cluster_gap_s <= 0.0:
        raise ValueError("window_s and cluster_gap_s must be positive and post_delay_s must be non-negative")
    ordered = sorted(frames, key=lambda frame: frame.timestamp_ns)
    ffc_frames = [frame for frame in ordered if frame.ffc_state == "ffc"]
    if not ffc_frames:
        raise ValueError("capture contains no FFC frame")
    cluster_gap_ns = round(cluster_gap_s * 1_000_000_000)
    ffc_clusters: list[list[RawFrame]] = []
    for frame in ffc_frames:
        if not ffc_clusters or frame.timestamp_ns - ffc_clusters[-1][-1].timestamp_ns > cluster_gap_ns:
            ffc_clusters.append([frame])
        else:
            ffc_clusters[-1].append(frame)

    window_ns = round(window_s * 1_000_000_000)
    post_delay_ns = round(post_delay_s * 1_000_000_000)
    events: list[dict[str, int | float]] = []
    for cluster in ffc_clusters:
        ffc_start_ns = cluster[0].timestamp_ns
        ffc_end_ns = cluster[-1].timestamp_ns
        pre = phase_metric(ordered, roi=roi, start_ns=ffc_start_ns - window_ns, end_ns=ffc_start_ns - 1)
        post = phase_metric(
            ordered,
            roi=roi,
            start_ns=ffc_end_ns + post_delay_ns,
            end_ns=ffc_end_ns + post_delay_ns + window_ns,
        )
        events.append(
            {
                "ffc_frame_count": len(cluster),
                "ffc_start_ns": ffc_start_ns,
                "ffc_end_ns": ffc_end_ns,
                "pre_median": pre["median"],
                "post_median": post["median"],
                "post_minus_pre_counts": int(post["median"]) - int(pre["median"]),
                "pre_unique_frame_count": pre["unique_frame_count"],
                "post_unique_frame_count": post["unique_frame_count"],
                "pre_population_stdev": pre["population_stdev"],
                "post_population_stdev": post["population_stdev"],
            }
        )
    return events


def _nearest_quantile(sorted_values: list[int], quantile: float) -> int:
    if not sorted_values:
        raise ValueError("cannot calculate a quantile from no raw values")
    return sorted_values[round((len(sorted_values) - 1) * quantile)]


def suggest_fixed_range(
    frames: Iterable[RawFrame],
    *,
    minimum_span: int = 256,
    alignment: int = 16,
    padding_fraction: float = 0.1,
) -> dict[str, int | float]:
    if minimum_span <= 0 or alignment <= 0 or padding_fraction < 0.0:
        raise ValueError("range settings must be positive and padding_fraction non-negative")
    values = sorted(
        int(value)
        for frame in unique_normal_frames(frames)
        for value in frame.values
    )
    lower_quantile = _nearest_quantile(values, 0.005)
    upper_quantile = _nearest_quantile(values, 0.995)
    span = max(upper_quantile - lower_quantile, minimum_span)
    padding = math.ceil(span * padding_fraction)
    raw_low = max(0, ((lower_quantile - padding) // alignment) * alignment)
    raw_high = min(65535, ((upper_quantile + padding + alignment - 1) // alignment) * alignment)
    if raw_high <= raw_low:
        raise ValueError("suggested fixed range is invalid")
    return {
        "raw_low": raw_low,
        "raw_high": raw_high,
        "source_p005": lower_quantile,
        "source_p995": upper_quantile,
        "source_unique_frame_count": len(unique_normal_frames(frames)),
        "minimum_span": minimum_span,
        "padding_fraction": padding_fraction,
    }


def restart_offsets(run_summaries: Iterable[dict[str, object]]) -> dict[str, object]:
    ordered = list(run_summaries)
    if not ordered:
        raise ValueError("at least one restart run summary is required")
    reference_id = str(ordered[0]["run_id"])
    reference_median = int(ordered[0]["stable_median"])
    by_run = {
        str(summary["run_id"]): int(summary["stable_median"]) - reference_median
        for summary in ordered
    }
    return {
        "reference_run_id": reference_id,
        "reference_stable_median": reference_median,
        "by_run": by_run,
        "max_absolute_offset_counts": max(abs(offset) for offset in by_run.values()),
    }


def analyze_dynamic_phases(
    frames: Iterable[RawFrame],
    *,
    target_roi: RawROI,
    control_roi: RawROI,
    phase_windows: dict[str, tuple[int, int]],
) -> dict[str, object]:
    """Measure target and control ROIs for the fixed, predeclared dynamic phases."""
    ordered_frames = list(frames)
    phases: dict[str, dict[str, dict[str, int | float]]] = {}
    for phase_name, (start_ns, end_ns) in phase_windows.items():
        phases[phase_name] = {
            "target": phase_metric(
                ordered_frames,
                roi=target_roi,
                start_ns=start_ns,
                end_ns=end_ns,
            ),
            "control": phase_metric(
                ordered_frames,
                roi=control_roi,
                start_ns=start_ns,
                end_ns=end_ns,
            ),
        }

    cycles: list[dict[str, int | str]] = []
    for baseline_name in phases:
        if not baseline_name.startswith("baseline_"):
            continue
        suffix = baseline_name.removeprefix("baseline_")
        hot_name = f"hot_hand_{suffix}"
        recovery_name = f"recovery_{suffix}"
        if hot_name not in phases or recovery_name not in phases:
            raise ValueError(f"dynamic phases for cycle {suffix} are incomplete")
        baseline = phases[baseline_name]
        hot = phases[hot_name]
        recovery = phases[recovery_name]
        cycles.append(
            {
                "cycle": suffix,
                "target_hot_minus_baseline_counts": int(hot["target"]["median"]) - int(baseline["target"]["median"]),
                "control_hot_minus_baseline_counts": int(hot["control"]["median"]) - int(baseline["control"]["median"]),
                "target_recovery_minus_baseline_counts": int(recovery["target"]["median"])
                - int(baseline["target"]["median"]),
                "control_recovery_minus_baseline_counts": int(recovery["control"]["median"])
                - int(baseline["control"]["median"]),
            }
        )
    if not cycles:
        raise ValueError("dynamic phase windows do not contain a baseline cycle")
    return {"phases": phases, "cycles": cycles}


def _read_u16le(path: Path, *, width: int, height: int) -> array:
    payload = path.read_bytes()
    expected_size = width * height * 2
    if len(payload) != expected_size:
        raise ValueError(f"{path}: expected {expected_size} bytes, found {len(payload)}")
    values = array("H")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def load_raw_frames(directory: Path) -> list[RawFrame]:
    frames: list[RawFrame] = []
    for metadata_path in sorted(directory.glob("raw_frame_*.json")):
        metadata = json.loads(metadata_path.read_text())
        width = int(metadata["width"])
        height = int(metadata["height"])
        raw_name = str(metadata["raw_file"])
        mapping = metadata["display_mapping"]
        frames.append(
            RawFrame(
                frame_index=int(metadata["frame_index"]),
                timestamp_ns=int(metadata["monotonic_timestamp_ns"]),
                width=width,
                height=height,
                values=_read_u16le(directory / raw_name, width=width, height=height),
                ffc_state=str(metadata["ffc_state"]),
                repeated_frame_flag=bool(metadata.get("repeated_frame_flag", False)),
                display_mapping_mode=str(mapping["mode"]),
                raw_low=mapping.get("raw_low") if mapping["mode"] == "fixed_raw_counts" else None,
                raw_high=mapping.get("raw_high") if mapping["mode"] == "fixed_raw_counts" else None,
            )
        )
    if not frames:
        raise ValueError(f"no raw frames found in {directory}")
    return frames
