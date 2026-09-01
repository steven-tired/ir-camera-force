from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ir-oak-squeeze")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_TRIAL = Path(
    "/home/zhuokai/hand-teleop/ir-camera-force/local/datasets/ir_hard_classifier/trials/"
    "oak-squeeze_s01_fixed-posture_foam_zk_rep02"
)


def _float(value: object) -> float | None:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _pearson(first: Iterable[float], second: Iterable[float]) -> float:
    x = np.asarray(list(first), dtype=float)
    y = np.asarray(list(second), dtype=float)
    if len(x) < 2 or len(x) != len(y):
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _zscore(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    deviation = float(array.std())
    if deviation == 0:
        return np.zeros_like(array)
    return (array - float(array.mean())) / deviation


def _lagged_pair(proxy: np.ndarray, ir: np.ndarray, lag_frames: int) -> tuple[np.ndarray, np.ndarray]:
    if lag_frames > 0:
        return proxy[:-lag_frames], ir[lag_frames:]
    if lag_frames < 0:
        lead = abs(lag_frames)
        return proxy[lead:], ir[:-lead]
    return proxy, ir


def _group_by_step(records: Iterable[dict[str, float | int]]) -> dict[tuple[int, int], list[dict[str, float | int]]]:
    groups: dict[tuple[int, int], list[dict[str, float | int]]] = defaultdict(list)
    for record in records:
        groups[(int(record["sequence_id"]), int(record["step_index"]))].append(record)
    for rows in groups.values():
        rows.sort(key=lambda row: float(row["step_elapsed_s"]))
    return groups


def lag_scan(
    records: Iterable[dict[str, float | int]],
    *,
    max_lag_frames: int = 10,
) -> list[dict[str, float | int]]:
    """Positive lag means the IR sequence is aligned later than the OAK proxy."""
    groups = _group_by_step(records)
    rows: list[dict[str, float | int]] = []
    for lag_frames in range(-max_lag_frames, max_lag_frames + 1):
        correlations: list[float] = []
        pair_count = 0
        for step_rows in groups.values():
            if len(step_rows) < abs(lag_frames) + 3:
                continue
            proxy, ir = _lagged_pair(
                _zscore(float(row["pinch_norm"]) for row in step_rows),
                _zscore(float(row["neg_area"]) for row in step_rows),
                lag_frames,
            )
            corr = _pearson(proxy, ir)
            if math.isfinite(corr):
                correlations.append(corr)
                pair_count += len(proxy)
        rows.append(
            {
                "lag_frames": lag_frames,
                "lag_seconds_at_10fps": lag_frames / 10.0,
                "median_pearson": float(median(correlations)) if correlations else float("nan"),
                "mean_pearson": float(fmean(correlations)) if correlations else float("nan"),
                "segment_count": len(correlations),
                "pair_count": pair_count,
            }
        )
    return rows


def window_change(
    records: Iterable[dict[str, float | int]],
    *,
    window_s: float = 0.75,
) -> list[dict[str, float | int]]:
    """Summarize first-versus-last window change for every recorded step."""
    summaries: list[dict[str, float | int]] = []
    for (sequence_id, step_index), step_rows in sorted(_group_by_step(records).items()):
        first_time = float(step_rows[0]["step_elapsed_s"])
        last_time = float(step_rows[-1]["step_elapsed_s"])
        first_rows = [row for row in step_rows if float(row["step_elapsed_s"]) <= first_time + window_s]
        last_rows = [row for row in step_rows if float(row["step_elapsed_s"]) >= last_time - window_s]
        if not first_rows or not last_rows:
            continue
        proxy_first = fmean(float(row["pinch_norm"]) for row in first_rows)
        proxy_last = fmean(float(row["pinch_norm"]) for row in last_rows)
        ir_first = fmean(float(row["neg_area"]) for row in first_rows)
        ir_last = fmean(float(row["neg_area"]) for row in last_rows)
        summaries.append(
            {
                "sequence_id": sequence_id,
                "step_index": step_index,
                "frame_count": len(step_rows),
                "proxy_first": proxy_first,
                "proxy_last": proxy_last,
                "proxy_change": proxy_last - proxy_first,
                "ir_first": ir_first,
                "ir_last": ir_last,
                "ir_change": ir_last - ir_first,
            }
        )
    return summaries


def _read_metrics(path: Path) -> list[dict[str, float | int | str]]:
    records: list[dict[str, float | int | str]] = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            pinch_norm = _float(raw.get("pinch_norm"))
            neg_area = _float(raw.get("neg_area"))
            step_elapsed_s = _float(raw.get("step_elapsed_s"))
            t_capture = _float(raw.get("t_capture"))
            if (
                raw.get("hand_detected", "").lower() != "true"
                or raw.get("frozen_frame_flag", "").lower() == "true"
                or raw.get("agc_jump_flag", "").lower() == "true"
                or pinch_norm is None
                or neg_area is None
                or step_elapsed_s is None
                or t_capture is None
            ):
                continue
            records.append(
                {
                    "phase": raw["phase"],
                    "sequence_id": int(raw["sequence_id"]),
                    "step_index": int(raw["step_index"]),
                    "target_squeeze_percent": float(raw["target_squeeze_percent"]),
                    "step_elapsed_s": step_elapsed_s,
                    "t_capture": t_capture,
                    "pinch_norm": pinch_norm,
                    "neg_area": neg_area,
                }
            )
    return records


def _last_window_records(records: Iterable[dict[str, float | int | str]], window_s: float = 0.75) -> list[dict[str, float | int | str]]:
    selected: list[dict[str, float | int | str]] = []
    for rows in _group_by_step(records).values():
        end = float(rows[-1]["step_elapsed_s"])
        selected.extend(row for row in rows if float(row["step_elapsed_s"]) >= end - window_s)
    return selected


def _write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    *,
    steady: list[dict[str, float | int | str]],
    lag_rows: list[dict[str, float | int]],
    release_changes: list[dict[str, float | int]],
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    prompt_groups: dict[float, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in steady:
        prompt_groups[float(row["target_squeeze_percent"])].append(row)
    levels = sorted(prompt_groups)
    aperture_means = [fmean(float(row["pinch_norm"]) for row in prompt_groups[level]) for level in levels]
    ir_means = [fmean(float(row["neg_area"]) for row in prompt_groups[level]) for level in levels]
    axes[0, 0].plot(levels, aperture_means, marker="o", color="tab:blue")
    axes[0, 0].set(title="OAK proxy follows subjective prompt", xlabel="Prompted squeeze (%)", ylabel="pinch aperture / palm scale")
    axes[0, 0].grid(alpha=0.25)
    scatter = axes[0, 1].scatter(
        [float(row["pinch_norm"]) for row in steady],
        [float(row["neg_area"]) for row in steady],
        c=[int(row["sequence_id"]) for row in steady],
        cmap="tab10",
    )
    figure.colorbar(scatter, ax=axes[0, 1], label="sequence")
    axes[0, 1].set(title="Steady steps: raw IR relation", xlabel="pinch aperture / palm scale", ylabel="IR negative area (px)")
    axes[0, 1].grid(alpha=0.25)
    valid_lag_rows = [row for row in lag_rows if math.isfinite(float(row["median_pearson"]))]
    axes[1, 0].plot(
        [float(row["lag_seconds_at_10fps"]) for row in valid_lag_rows],
        [float(row["median_pearson"]) for row in valid_lag_rows],
        marker="o",
        color="tab:purple",
    )
    axes[1, 0].axvline(0, color="0.2", linestyle="--", linewidth=1)
    axes[1, 0].set(title="Within-step lag scan (+ means IR later)", xlabel="IR lag (s)", ylabel="median Pearson r")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 1].scatter(
        [float(row["proxy_change"]) for row in release_changes],
        [float(row["ir_change"]) for row in release_changes],
        color="tab:orange",
    )
    axes[1, 1].axhline(0, color="0.2", linestyle="--", linewidth=1)
    axes[1, 1].axvline(0, color="0.2", linestyle="--", linewidth=1)
    axes[1, 1].set(title="Release: OAK versus IR change", xlabel="aperture proxy change", ylabel="IR negative-area change (px)")
    axes[1, 1].grid(alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def analyze(trial_dir: Path, *, max_lag_frames: int = 10) -> dict[str, object]:
    analysis_dir = trial_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    records = _read_metrics(analysis_dir / "oak_hand_frame_metrics.csv")
    holds = [record for record in records if record["phase"] == "target_hold"]
    releases = [record for record in records if record["phase"] == "release"]
    steady = _last_window_records(holds)
    lag_rows = lag_scan(holds, max_lag_frames=max_lag_frames)
    release_changes = window_change(releases)
    zero_steady = [row for row in steady if float(row["target_squeeze_percent"]) == 0.0]
    zero_drift = _pearson(
        (float(row["t_capture"]) for row in zero_steady),
        (float(row["neg_area"]) for row in zero_steady),
    )
    valid_lag_rows = [row for row in lag_rows if math.isfinite(float(row["median_pearson"]))]
    best_lag = max(valid_lag_rows, key=lambda row: float(row["median_pearson"])) if valid_lag_rows else None
    summary = {
        "trial_id": trial_dir.name,
        "analysis_scope": "single trial; OAK aperture proxy, not force or pressure",
        "gated_hand_and_thermal_frame_count": len(records),
        "gated_target_hold_frame_count": len(holds),
        "gated_release_frame_count": len(releases),
        "steady_step_frame_count": len(steady),
        "zero_prompt_steady_frame_count": len(zero_steady),
        "zero_prompt_time_vs_ir_negative_area_pearson": zero_drift,
        "best_within_hold_lag": best_lag,
        "lag_interpretation": "positive lag means the IR samples were shifted later than the OAK aperture samples",
        "hysteresis_interpretation": "not estimable from this randomized target order; release changes are only a persistence diagnostic",
        "release_window_s": 0.75,
    }
    _write_csv(analysis_dir / "oak_ir_lag_scan.csv", lag_rows)
    _write_csv(analysis_dir / "oak_ir_release_persistence.csv", release_changes)
    _write_csv(
        analysis_dir / "oak_ir_steady_frames.csv",
        [
            {
                "sequence_id": int(row["sequence_id"]),
                "step_index": int(row["step_index"]),
                "target_squeeze_percent": float(row["target_squeeze_percent"]),
                "t_capture": float(row["t_capture"]),
                "pinch_norm": float(row["pinch_norm"]),
                "neg_area": float(row["neg_area"]),
            }
            for row in steady
        ],
    )
    (analysis_dir / "oak_ir_dynamics_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _plot(
        steady=steady,
        lag_rows=lag_rows,
        release_changes=release_changes,
        path=analysis_dir / "oak_ir_dynamics.png",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze IR/OAK squeeze-proxy trial dynamics.")
    parser.add_argument("--trial", type=Path, default=DEFAULT_TRIAL)
    parser.add_argument("--max-lag-frames", type=int, default=10)
    args = parser.parse_args()
    summary = analyze(args.trial, max_lag_frames=args.max_lag_frames)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
