#!/usr/bin/env python3
"""Verify raw thermal frames emitted by the reversible FLIR prototype."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from array import array
from collections import Counter
from pathlib import Path


def _read_u16_le(path: Path, *, width: int, height: int) -> array:
    expected_bytes = width * height * 2
    payload = path.read_bytes()
    if len(payload) != expected_bytes:
        raise ValueError(f"{path.name}: expected {expected_bytes} bytes, found {len(payload)}")
    values = array("H")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def _median_u16(values: array) -> int:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return int(ordered[midpoint])
    return int((ordered[midpoint - 1] + ordered[midpoint]) // 2)


def _require_int(metadata: dict[str, object], key: str) -> int:
    value = metadata.get(key)
    if not isinstance(value, int):
        raise ValueError(f"metadata field {key!r} must be an integer")
    return value


def _numeric_summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"minimum": None, "maximum": None, "mean": None, "population_stdev": None}
    return {
        "minimum": min(values),
        "maximum": max(values),
        "mean": statistics.fmean(values),
        "population_stdev": statistics.pstdev(values),
    }


def _nearest_quantile(sorted_values: list[int], quantile: float) -> int | None:
    if not sorted_values:
        return None
    return sorted_values[round((len(sorted_values) - 1) * quantile)]


def _linear_drift_per_second(timestamps_ns: list[int], values: list[int]) -> float | None:
    if len(timestamps_ns) != len(values) or len(values) < 2:
        return None
    elapsed_s = [(timestamp - timestamps_ns[0]) / 1_000_000_000 for timestamp in timestamps_ns]
    mean_time = statistics.fmean(elapsed_s)
    mean_value = statistics.fmean(values)
    denominator = sum((time_s - mean_time) ** 2 for time_s in elapsed_s)
    if denominator == 0.0:
        return None
    return sum(
        (time_s - mean_time) * (value - mean_value)
        for time_s, value in zip(elapsed_s, values)
    ) / denominator


def validate_capture_directory(
    directory: Path,
    *,
    expected_width: int | None = None,
    expected_height: int | None = None,
    require_ffc: bool = False,
    max_repeated_fraction: float | None = None,
) -> dict[str, object]:
    if max_repeated_fraction is not None and not 0.0 <= max_repeated_fraction <= 1.0:
        raise ValueError("max_repeated_fraction must be within [0, 1]")
    directory = directory.resolve()
    metadata_paths = sorted(directory.glob("raw_frame_*.json"))
    if not metadata_paths:
        raise ValueError(f"no raw metadata files found in {directory}")

    dimensions: set[tuple[int, int]] = set()
    ffc_states: Counter[str] = Counter()
    display_mapping_modes: Counter[str] = Counter()
    repeated_frames = 0
    values_above_255 = False
    timestamps_ns: list[int] = []
    normal_medians: list[int] = []
    deduplicated_normal_medians: list[int] = []
    deduplicated_normal_timestamps_ns: list[int] = []
    deduplicated_normal_pixels: list[int] = []
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("dtype") != "uint16" or metadata.get("byte_order") != "little":
            raise ValueError(f"{metadata_path.name}: expected uint16 little-endian metadata")
        width = _require_int(metadata, "width")
        height = _require_int(metadata, "height")
        if width <= 0 or height <= 0:
            raise ValueError(f"{metadata_path.name}: dimensions must be positive")
        if expected_width is not None and width != expected_width:
            raise ValueError(f"{metadata_path.name}: width {width} != expected {expected_width}")
        if expected_height is not None and height != expected_height:
            raise ValueError(f"{metadata_path.name}: height {height} != expected {expected_height}")
        raw_name = metadata.get("raw_file")
        if not isinstance(raw_name, str) or Path(raw_name).name != raw_name:
            raise ValueError(f"{metadata_path.name}: raw_file must be a plain file name")
        values = _read_u16_le(directory / raw_name, width=width, height=height)
        observed = {
            "raw_min": int(min(values)),
            "raw_median": _median_u16(values),
            "raw_max": int(max(values)),
        }
        timestamps_ns.append(_require_int(metadata, "monotonic_timestamp_ns"))
        for key, actual in observed.items():
            expected = _require_int(metadata, key)
            if actual != expected:
                raise ValueError(f"{metadata_path.name}: {key} {expected} does not match raw bytes {actual}")
        dimensions.add((width, height))
        values_above_255 |= observed["raw_max"] > 255
        ffc_state = metadata.get("ffc_state")
        if not isinstance(ffc_state, str):
            raise ValueError(f"{metadata_path.name}: ffc_state must be a string")
        ffc_states[ffc_state] += 1
        display_mapping = metadata.get("display_mapping")
        if not isinstance(display_mapping, dict):
            raise ValueError(f"{metadata_path.name}: display_mapping must be an object")
        mapping_mode = display_mapping.get("mode")
        if mapping_mode not in {"dynamic_per_frame_min_max", "fixed_raw_counts"}:
            raise ValueError(f"{metadata_path.name}: unsupported display_mapping mode {mapping_mode!r}")
        if mapping_mode == "fixed_raw_counts":
            raw_low = display_mapping.get("raw_low")
            raw_high = display_mapping.get("raw_high")
            if not isinstance(raw_low, int) or not isinstance(raw_high, int) or raw_low >= raw_high:
                raise ValueError(f"{metadata_path.name}: fixed display_mapping requires ordered integer bounds")
        display_mapping_modes[mapping_mode] += 1
        repeated = bool(metadata.get("repeated_frame_flag", False))
        repeated_frames += repeated
        if ffc_state == "normal":
            normal_medians.append(observed["raw_median"])
            if not repeated:
                deduplicated_normal_medians.append(observed["raw_median"])
                deduplicated_normal_timestamps_ns.append(timestamps_ns[-1])
                deduplicated_normal_pixels.extend(values)

    interval_ms = [
        (later - earlier) / 1_000_000
        for earlier, later in zip(timestamps_ns, timestamps_ns[1:])
    ]
    capture_duration_s = (timestamps_ns[-1] - timestamps_ns[0]) / 1_000_000_000 if len(timestamps_ns) > 1 else 0.0
    deduplicated_normal_pixels.sort()
    repeated_frame_fraction = repeated_frames / len(metadata_paths)
    if require_ffc and ffc_states["ffc"] == 0:
        raise ValueError("capture does not contain an FFC frame")
    if max_repeated_fraction is not None and repeated_frame_fraction > max_repeated_fraction:
        raise ValueError(
            f"repeated frame fraction {repeated_frame_fraction:.3f} exceeds {max_repeated_fraction:.3f}"
        )

    summary = {
        "schema_version": 1,
        "capture_directory": str(directory),
        "frame_count": len(metadata_paths),
        "dimensions": [list(shape) for shape in sorted(dimensions)],
        "all_uint16_little_endian": True,
        "has_values_above_255": values_above_255,
        "metadata_stat_mismatch_count": 0,
        "ffc_state_counts": dict(sorted(ffc_states.items())),
        "display_mapping_mode_counts": dict(sorted(display_mapping_modes.items())),
        "repeated_frame_count": repeated_frames,
        "repeated_frame_fraction": repeated_frame_fraction,
        "normal_frame_count": len(normal_medians),
        "deduplicated_normal_frame_count": len(deduplicated_normal_medians),
        "normal_raw_median_stats": _numeric_summary(normal_medians),
        "deduplicated_normal_raw_median_stats": _numeric_summary(deduplicated_normal_medians),
        "deduplicated_normal_raw_median_drift_counts_per_s": _linear_drift_per_second(
            deduplicated_normal_timestamps_ns,
            deduplicated_normal_medians,
        ),
        "deduplicated_normal_pixel_quantiles": {
            "p005": _nearest_quantile(deduplicated_normal_pixels, 0.005),
            "p01": _nearest_quantile(deduplicated_normal_pixels, 0.01),
            "p50": _nearest_quantile(deduplicated_normal_pixels, 0.5),
            "p99": _nearest_quantile(deduplicated_normal_pixels, 0.99),
            "p995": _nearest_quantile(deduplicated_normal_pixels, 0.995),
        },
        "timing": {
            "capture_duration_s": capture_duration_s,
            "frame_interval_ms": _numeric_summary(interval_ms),
            "frame_interval_median_ms": statistics.median(interval_ms) if interval_ms else None,
            "observed_fps": (len(metadata_paths) - 1) / capture_duration_s if capture_duration_s else None,
        },
        "raw_stream_accepted": bool(values_above_255 and len(dimensions) == 1),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--expected-width", type=int, default=80)
    parser.add_argument("--expected-height", type=int, default=60)
    parser.add_argument("--require-ffc", action="store_true")
    parser.add_argument("--max-repeated-fraction", type=float)
    parser.add_argument("--summary-path", type=Path)
    args = parser.parse_args()
    summary = validate_capture_directory(
        args.raw_dir,
        expected_width=args.expected_width,
        expected_height=args.expected_height,
        require_ffc=args.require_ffc,
        max_repeated_fraction=args.max_repeated_fraction,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.summary_path is not None:
        args.summary_path.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
