from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ir-foam-compression")

import matplotlib
import numpy as np
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_ROOT = Path("/home/zhuokai/hand-teleop/datasets/ir_foam_compression")


@dataclass(frozen=True)
class TrialAnalysis:
    trial_id: str
    primary_feature: str
    valid_frame_count: int
    invalid_event_count: int
    steps: list[dict[str, object]]
    actual_compression_spearman_rho: float
    actual_compression_spearman_p: float
    c30_minus_c0: float


def _float(value: object) -> float | None:
    try:
        result = float(str(value))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _truth(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _mean(values: list[float]) -> float:
    return float(fmean(values)) if values else float("nan")


def _event_key(row: dict[str, str]) -> tuple[str, int, int, str, int] | None:
    try:
        return (
            row["block"],
            int(row["sequence_id"]),
            int(row["step_index"]),
            row["state"],
            int(row["action_attempt"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def summarize_trial(trial_dir: Path, *, stable_window_s: float = 3.0) -> TrialAnalysis:
    if stable_window_s <= 0:
        raise ValueError("stable_window_s must be positive")
    metadata = json.loads((trial_dir / "metadata.json").read_text())
    primary_feature = str(metadata["primary_ir_feature"])
    telemetry_rows = _read_csv(trial_dir / "telemetry.csv")
    feature_by_frame = {row["frame"]: row for row in _read_csv(trial_dir / "frame_features.csv")}
    events_path = trial_dir / "events.csv"
    event_rows = _read_csv(events_path) if events_path.exists() else []
    invalid_keys = {
        key
        for row in event_rows
        if row.get("event_type") == "invalid"
        for key in [_event_key(row)]
        if key is not None
    }
    seen_hashes: set[str] = set()
    valid: list[dict[str, object]] = []
    for telemetry in telemetry_rows:
        features = feature_by_frame.get(telemetry["frame"])
        if features is None or telemetry.get("phase") != "stable_hold":
            continue
        telemetry_key = _event_key(telemetry)
        if telemetry_key in invalid_keys:
            continue
        if _truth(features.get("frozen_frame_flag")) or not _truth(telemetry.get("marker_detected")):
            continue
        frame_hash = features.get("thermal_frame_sha1", "")
        if not frame_hash or frame_hash in seen_hashes:
            continue
        compression = _float(telemetry.get("compression_pct"))
        primary_value = _float(features.get(primary_feature))
        step_elapsed_s = _float(telemetry.get("step_elapsed_s"))
        if compression is None or primary_value is None or step_elapsed_s is None:
            continue
        seen_hashes.add(frame_hash)
        valid.append(
            {
                "frame": int(telemetry["frame"]),
                "block": telemetry["block"],
                "state": telemetry["state"],
                "target_compression_pct": _float(telemetry.get("target_compression_pct")),
                "sequence_id": int(telemetry["sequence_id"]),
                "step_index": int(telemetry["step_index"]),
                "step_name": telemetry["step_name"],
                "action_attempt": int(telemetry["action_attempt"]),
                "step_elapsed_s": step_elapsed_s,
                "compression_pct": compression,
                primary_feature: primary_value,
                "background_norm": _float(features.get("background_norm")),
                "left_contact_norm": _float(features.get("left_contact_norm")),
                "right_contact_norm": _float(features.get("right_contact_norm")),
                "reference_span": _float(features.get("reference_span")),
            }
        )
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in valid:
        key = (row["block"], row["sequence_id"], row["step_index"], row["state"], row["action_attempt"])
        grouped[key].append(row)
    steps: list[dict[str, object]] = []
    for rows in grouped.values():
        rows.sort(key=lambda row: float(row["step_elapsed_s"]))
        final_time = float(rows[-1]["step_elapsed_s"])
        steady = [row for row in rows if float(row["step_elapsed_s"]) >= final_time - stable_window_s]
        first = rows[0]
        step = {
            "block": first["block"],
            "state": first["state"],
            "target_compression_pct": first["target_compression_pct"],
            "sequence_id": first["sequence_id"],
            "step_index": first["step_index"],
            "step_name": first["step_name"],
            "action_attempt": first["action_attempt"],
            "frame_count": len(steady),
            "compression_pct": _mean([float(row["compression_pct"]) for row in steady]),
            primary_feature: _mean([float(row[primary_feature]) for row in steady]),
            "background_norm": _mean([float(row["background_norm"]) for row in steady if row["background_norm"] is not None]),
            "left_contact_norm": _mean([float(row["left_contact_norm"]) for row in steady if row["left_contact_norm"] is not None]),
            "right_contact_norm": _mean([float(row["right_contact_norm"]) for row in steady if row["right_contact_norm"] is not None]),
            "reference_span": _mean([float(row["reference_span"]) for row in steady if row["reference_span"] is not None]),
        }
        steps.append(step)
    steps.sort(key=lambda row: (str(row["block"]), int(row["sequence_id"]), int(row["step_index"])))
    values = [(float(row["compression_pct"]), float(row[primary_feature])) for row in steps if math.isfinite(float(row[primary_feature]))]
    if len(values) >= 3:
        correlation = spearmanr(*zip(*values, strict=True))
        rho, p_value = float(correlation.statistic), float(correlation.pvalue)
    else:
        rho, p_value = float("nan"), float("nan")
    c0 = [float(row[primary_feature]) for row in steps if row["state"] == "C0" and math.isfinite(float(row[primary_feature]))]
    c30 = [float(row[primary_feature]) for row in steps if row["state"] == "C30" and math.isfinite(float(row[primary_feature]))]
    invalid_events = sum(row.get("event_type") == "invalid" for row in event_rows)
    return TrialAnalysis(
        trial_id=trial_dir.name,
        primary_feature=primary_feature,
        valid_frame_count=len(valid),
        invalid_event_count=invalid_events,
        steps=steps,
        actual_compression_spearman_rho=rho,
        actual_compression_spearman_p=p_value,
        c30_minus_c0=_mean(c30) - _mean(c0) if c0 and c30 else float("nan"),
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_analysis(trial_dir: Path, analysis: TrialAnalysis) -> Path:
    output = trial_dir / "analysis"
    output.mkdir(exist_ok=True)
    _write_csv(output / "foam_compression_step_summary.csv", analysis.steps)
    payload = asdict(analysis)
    payload.pop("steps")
    (output / "foam_compression_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if analysis.steps:
        x = [float(row["compression_pct"]) for row in analysis.steps]
        y = [float(row[analysis.primary_feature]) for row in analysis.steps]
        colors = [{"C0": "tab:blue", "C10": "tab:cyan", "C20": "tab:orange", "C30": "tab:red", "N": "0.45", "R": "0.2"}.get(str(row["state"]), "0.5") for row in analysis.steps]
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        axes[0].scatter(x, y, c=colors)
        axes[0].set(
            title="Pre-registered primary relation",
            xlabel="OAK marker compression (%)",
            ylabel=analysis.primary_feature,
        )
        axes[0].grid(alpha=0.25)
        states = ["N", "C0", "C10", "C20", "C30", "R"]
        grouped = {state: [float(row[analysis.primary_feature]) for row in analysis.steps if row["state"] == state] for state in states}
        axes[1].boxplot([grouped[state] for state in states if grouped[state]], tick_labels=[state for state in states if grouped[state]])
        axes[1].set(title="Frozen foam-center feature by state", xlabel="state", ylabel=analysis.primary_feature)
        axes[1].grid(axis="y", alpha=0.25)
        figure.savefig(output / "foam_compression_primary_summary.png", dpi=180)
        plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze fixed-geometry foam-compression recordings.")
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--stable-window-s", type=float, default=3.0)
    args = parser.parse_args()
    analysis = summarize_trial(args.trial, stable_window_s=args.stable_window_s)
    output = write_analysis(args.trial, analysis)
    print(f"wrote {output}")
    print(json.dumps({key: value for key, value in asdict(analysis).items() if key != "steps"}, indent=2))


if __name__ == "__main__":
    main()
