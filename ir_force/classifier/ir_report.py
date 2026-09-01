from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Iterable


FeatureRow = dict[str, str]
TelemetryRow = dict[str, str]


def _as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value.strip() == "":
        raise ValueError(f"missing numeric value for {key}")
    return float(value)


def _mean(rows: Iterable[dict[str, str]], key: str) -> float:
    values = [_as_float(row, key) for row in rows]
    if not values:
        return 0.0
    return float(mean(values))


def _max(rows: Iterable[dict[str, str]], key: str) -> float:
    values = [_as_float(row, key) for row in rows]
    if not values:
        return 0.0
    return float(max(values))


def split_capture_windows(telemetry_rows: list[TelemetryRow]) -> list[tuple[int, int]]:
    if not telemetry_rows:
        return []
    starts = [0]
    previous = _as_float(telemetry_rows[0], "t_capture")
    for index, row in enumerate(telemetry_rows[1:], start=1):
        current = _as_float(row, "t_capture")
        if current < previous:
            starts.append(index)
        previous = current
    starts.append(len(telemetry_rows))
    return list(zip(starts, starts[1:]))


def summarize_windows(
    feature_rows: list[FeatureRow],
    telemetry_rows: list[TelemetryRow],
) -> dict[str, dict[str, float] | list[tuple[int, int]]]:
    if len(feature_rows) != len(telemetry_rows):
        raise ValueError("feature and telemetry row counts must match")
    windows = split_capture_windows(telemetry_rows)
    if not windows:
        raise ValueError("cannot summarize empty trial")
    baseline_start, baseline_stop = windows[0]
    hold_start, hold_stop = windows[1] if len(windows) > 1 else windows[0]

    baseline_features = feature_rows[baseline_start:baseline_stop]
    hold_features = feature_rows[hold_start:hold_stop]
    baseline_telemetry = telemetry_rows[baseline_start:baseline_stop]
    hold_telemetry = telemetry_rows[hold_start:hold_stop]

    baseline = {
        "frame_count": float(len(baseline_features)),
        "area_px_mean": _mean(baseline_features, "area_px"),
        "mean_delta_mean": _mean(baseline_features, "mean_delta"),
        "max_delta_max": _max(baseline_features, "max_delta"),
        "present_load_max": _max(baseline_telemetry, "present_load"),
        "present_current_max": _max(baseline_telemetry, "present_current"),
    }
    hold = {
        "frame_count": float(len(hold_features)),
        "area_px_mean": _mean(hold_features, "area_px"),
        "mean_delta_mean": _mean(hold_features, "mean_delta"),
        "max_delta_max": _max(hold_features, "max_delta"),
        "present_load_max": _max(hold_telemetry, "present_load"),
        "present_current_max": _max(hold_telemetry, "present_current"),
    }
    change = {
        key: hold[key] - baseline[key]
        for key in ("area_px_mean", "mean_delta_mean", "max_delta_max", "present_load_max", "present_current_max")
    }
    return {"windows": windows, "baseline": baseline, "hold": hold, "change": change}


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_summary_json(trial_dir: Path, out_path: Path) -> dict[str, object]:
    feature_rows = load_csv_rows(trial_dir / "ir_features.csv")
    telemetry_rows = load_csv_rows(trial_dir / "telemetry.csv")
    summary = summarize_windows(feature_rows, telemetry_rows)
    metadata_path = trial_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    payload = {"metadata": metadata, "summary": summary}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
