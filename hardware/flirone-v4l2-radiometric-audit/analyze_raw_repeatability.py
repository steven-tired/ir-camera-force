#!/usr/bin/env python3
"""Analyze raw FLIR count repeatability sessions recorded by the guided protocol."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections.abc import Iterable
from pathlib import Path

from capture_raw_validation import validate_capture_directory
from raw_repeatability import (
    RawFrame,
    RawROI,
    analyze_dynamic_phases,
    analyze_ffc_recovery_events,
    load_raw_frames,
    phase_metric,
    restart_offsets,
    suggest_fixed_range,
)


def phase_windows_from_events(events: Iterable[dict[str, str]]) -> dict[str, tuple[int, int]]:
    """Pair each predeclared phase_start event with its corresponding phase_end."""
    starts: dict[str, int] = {}
    windows: dict[str, tuple[int, int]] = {}
    for event in events:
        event_type = event["event_type"]
        phase = event["phase"]
        timestamp_ns = int(event["timestamp_ns"])
        if event_type == "phase_start":
            if not phase:
                raise ValueError("phase_start event is missing a phase name")
            if phase in starts or phase in windows:
                raise ValueError(f"phase {phase} starts more than once")
            starts[phase] = timestamp_ns
        elif event_type == "phase_end":
            if phase not in starts:
                raise ValueError(f"phase {phase} ends without a start")
            if timestamp_ns < starts[phase]:
                raise ValueError(f"phase {phase} ends before its start")
            windows[phase] = (starts.pop(phase), timestamp_ns)
    if starts:
        raise ValueError(f"phases are missing end events: {', '.join(sorted(starts))}")
    return windows


def _load_events(events_path: Path) -> list[dict[str, str]]:
    with events_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _roi_from_manifest(manifest: dict[str, object], key: str) -> RawROI:
    raw_roi = manifest.get(key)
    if not isinstance(raw_roi, dict):
        raise ValueError(f"run manifest is missing {key}")
    try:
        return RawROI(
            x=int(raw_roi["x"]),
            y=int(raw_roi["y"]),
            width=int(raw_roi["width"]),
            height=int(raw_roi["height"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"run manifest has an invalid {key}") from exc


def _check_display_contract(frames: Iterable[RawFrame], manifest: dict[str, object]) -> None:
    display = manifest.get("display")
    if not isinstance(display, dict) or display.get("mode") not in {"dynamic", "fixed"}:
        raise ValueError("run manifest has an invalid display contract")
    expected_mode = "dynamic_per_frame_min_max" if display["mode"] == "dynamic" else "fixed_raw_counts"
    for frame in frames:
        if frame.display_mapping_mode != expected_mode:
            raise ValueError(
                f"run display contract expects {expected_mode}, found {frame.display_mapping_mode} in raw frame {frame.frame_index}"
            )
        if expected_mode == "fixed_raw_counts" and (
            frame.raw_low != display.get("raw_low") or frame.raw_high != display.get("raw_high")
        ):
            raise ValueError("raw frame fixed display bounds do not match the run manifest")


def fixed_raw_count_to_palette_index(raw_count: int, *, raw_low: int, raw_high: int) -> int:
    """Mirror the bridge's fixed raw-count integer mapping to its 256 palette entries."""
    if raw_low >= raw_high:
        raise ValueError("raw_low must be less than raw_high")
    return min(255, max(0, (raw_count - raw_low) * 255 // (raw_high - raw_low)))


def _palette_rgb(palette_path: Path):
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is required to analyze palette-index RGB frames") from exc
    payload = palette_path.read_bytes()
    if len(payload) != 256 * 3:
        raise ValueError(f"{palette_path} must contain exactly 768 palette bytes")
    return np.frombuffer(payload, dtype=np.uint8).reshape(256, 3)


def _phase_for_timestamp(timestamp_ns: int, phase_windows: dict[str, tuple[int, int]]) -> str | None:
    for phase_name, (start_ns, end_ns) in phase_windows.items():
        if start_ns <= timestamp_ns <= end_ns:
            return phase_name
    return None


def rgb_palette_phase_metrics(
    run_dir: Path,
    *,
    phase_windows: dict[str, tuple[int, int]],
    target_roi: RawROI,
    palette_path: Path,
) -> dict[str, dict[str, int | float]]:
    """Summarize the display palette index at the target raw ROI by protocol phase."""
    try:
        import cv2
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV and NumPy are required to analyze saved RGB frames") from exc
    palette = _palette_rgb(palette_path).astype(np.int16)
    by_phase: dict[str, list[int]] = {}
    with (run_dir / "rgb_frames.csv").open(newline="") as handle:
        rgb_rows = list(csv.DictReader(handle))
    for row in rgb_rows:
        phase = _phase_for_timestamp(int(row["timestamp_ns"]), phase_windows)
        if phase is None:
            continue
        image_path = run_dir / row["file"]
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"could not read saved RGB frame {image_path}")
        x = target_roi.x * 2
        y = target_roi.y * 2
        width = target_roi.width * 2
        height = target_roi.height * 2
        crop_bgr = image_bgr[y : y + height, x : x + width]
        if crop_bgr.shape[:2] != (height, width):
            raise ValueError(f"target raw ROI {target_roi} is outside RGB frame {image_path}")
        rgb_pixels = crop_bgr[:, :, ::-1].reshape(-1, 1, 3).astype(np.int16)
        distances = ((rgb_pixels - palette.reshape(1, 256, 3)) ** 2).sum(axis=2)
        palette_indices = distances.argmin(axis=1)
        by_phase.setdefault(phase, []).append(int(statistics.median(palette_indices.tolist())))
    return {
        phase: {
            "frame_count": len(indices),
            "median_palette_index": int(statistics.median(indices)),
            "population_stdev_palette_index": statistics.pstdev(indices),
        }
        for phase, indices in by_phase.items()
    }


def fixed_palette_phase_comparison(
    *,
    dynamic_metrics: dict[str, object],
    rgb_metrics: dict[str, dict[str, int | float]],
    raw_low: int,
    raw_high: int,
) -> dict[str, dict[str, int | None]]:
    """Compare fixed-range RGB palette indices with the bridge's raw-count mapping."""
    phases = dynamic_metrics.get("phases")
    if not isinstance(phases, dict):
        raise ValueError("dynamic metrics are missing phase measurements")
    comparison: dict[str, dict[str, int | None]] = {}
    for phase_name, phase_metric_summary in phases.items():
        if not isinstance(phase_metric_summary, dict):
            raise ValueError(f"dynamic metric for {phase_name} is invalid")
        target = phase_metric_summary.get("target")
        if not isinstance(target, dict) or "median" not in target:
            raise ValueError(f"dynamic target metric for {phase_name} is invalid")
        target_raw_median = int(target["median"])
        expected = fixed_raw_count_to_palette_index(target_raw_median, raw_low=raw_low, raw_high=raw_high)
        observed = rgb_metrics.get(str(phase_name), {}).get("median_palette_index")
        observed_index = int(observed) if observed is not None else None
        comparison[str(phase_name)] = {
            "target_raw_median": target_raw_median,
            "expected_palette_index": expected,
            "observed_palette_index": observed_index,
            "observed_minus_expected_palette_index": None if observed_index is None else observed_index - expected,
        }
    return comparison


def analyze_run(
    run_dir: Path,
    *,
    ffc_window_s: float,
    ffc_post_delay_s: float,
) -> tuple[dict[str, object], list[RawFrame]]:
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path} must contain a JSON object")
    run_id = str(manifest.get("run_id", ""))
    mode = str(manifest.get("mode", ""))
    if not run_id or mode not in {"ffc", "restart", "dynamic"}:
        raise ValueError(f"{manifest_path} has an invalid run_id or mode")
    raw_dir = run_dir / "raw"
    frames = load_raw_frames(raw_dir)
    _check_display_contract(frames, manifest)
    target_roi = _roi_from_manifest(manifest, "target_raw_roi")
    control_roi = _roi_from_manifest(manifest, "control_raw_roi")
    events = _load_events(run_dir / "events.csv")
    summary: dict[str, object] = {
        "run_id": run_id,
        "mode": mode,
        "display": manifest["display"],
        "raw_capture": validate_capture_directory(raw_dir),
    }
    if mode == "ffc":
        target_events = analyze_ffc_recovery_events(
            frames,
            roi=target_roi,
            window_s=ffc_window_s,
            post_delay_s=ffc_post_delay_s,
        )
        control_events = analyze_ffc_recovery_events(
            frames,
            roi=control_roi,
            window_s=ffc_window_s,
            post_delay_s=ffc_post_delay_s,
        )
        summary["target"] = {"event_count": len(target_events), "events": target_events}
        summary["control"] = {"event_count": len(control_events), "events": control_events}
    else:
        windows = phase_windows_from_events(events)
        rgb_metrics = (
            rgb_palette_phase_metrics(
                run_dir,
                phase_windows=windows,
                target_roi=target_roi,
                palette_path=Path(__file__).resolve().parent / "palettes" / "Iron2.raw",
            )
            if (run_dir / "rgb_frames.csv").is_file()
            else {}
        )
        summary["rgb_target_palette_by_phase"] = rgb_metrics
        if mode == "restart":
            if "stable" not in windows:
                raise ValueError(f"{run_id}: restart run is missing the stable phase")
            start_ns, end_ns = windows["stable"]
            summary["target"] = phase_metric(frames, roi=target_roi, start_ns=start_ns, end_ns=end_ns)
            summary["control"] = phase_metric(frames, roi=control_roi, start_ns=start_ns, end_ns=end_ns)
        else:
            dynamic_metrics = analyze_dynamic_phases(
                frames,
                target_roi=target_roi,
                control_roi=control_roi,
                phase_windows=windows,
            )
            summary["metrics"] = dynamic_metrics
            display = manifest["display"]
            if isinstance(display, dict) and display.get("mode") == "fixed":
                raw_low = display.get("raw_low")
                raw_high = display.get("raw_high")
                if not isinstance(raw_low, int) or not isinstance(raw_high, int):
                    raise ValueError(f"{run_id}: fixed display contract lacks raw-count bounds")
                summary["fixed_palette_comparison"] = fixed_palette_phase_comparison(
                    dynamic_metrics=dynamic_metrics,
                    rgb_metrics=rgb_metrics,
                    raw_low=raw_low,
                    raw_high=raw_high,
                )
    return summary, frames


def analyze_session(
    session_root: Path,
    *,
    ffc_window_s: float = 10.0,
    ffc_post_delay_s: float = 3.0,
) -> dict[str, object]:
    run_dirs = sorted((session_root / "runs").glob("*/run_manifest.json"))
    if not run_dirs:
        raise ValueError(f"no run manifests found below {session_root / 'runs'}")
    run_summaries: list[dict[str, object]] = []
    dynamic_frames: list[RawFrame] = []
    for manifest_path in run_dirs:
        summary, frames = analyze_run(
            manifest_path.parent,
            ffc_window_s=ffc_window_s,
            ffc_post_delay_s=ffc_post_delay_s,
        )
        run_summaries.append(summary)
        if summary["mode"] == "dynamic" and isinstance(summary["display"], dict) and summary["display"].get("mode") == "dynamic":
            dynamic_frames.extend(frames)

    ffc = [summary for summary in run_summaries if summary["mode"] == "ffc"]
    restart = [summary for summary in run_summaries if summary["mode"] == "restart"]
    dynamic = [summary for summary in run_summaries if summary["mode"] == "dynamic"]
    restart_summary: dict[str, object] | None = None
    if restart:
        restart_summary = {
            "target": restart_offsets(
                {"run_id": summary["run_id"], "stable_median": summary["target"]["median"]}  # type: ignore[index]
                for summary in restart
            ),
            "control": restart_offsets(
                {"run_id": summary["run_id"], "stable_median": summary["control"]["median"]}  # type: ignore[index]
                for summary in restart
            ),
        }
    return {
        "schema_version": 1,
        "session_root": str(session_root.resolve()),
        "runs": run_summaries,
        "ffc": ffc,
        "restart_offsets": restart_summary,
        "dynamic": dynamic,
        "fixed_range_suggestion": suggest_fixed_range(dynamic_frames) if dynamic_frames else None,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path)
    parser.add_argument("--ffc-window-s", type=float, default=10.0)
    parser.add_argument("--ffc-post-delay-s", type=float, default=3.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = analyze_session(
        args.session_root,
        ffc_window_s=args.ffc_window_s,
        ffc_post_delay_s=args.ffc_post_delay_s,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.summary_path is not None:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
