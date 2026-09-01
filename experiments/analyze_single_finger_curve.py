#!/usr/bin/env python3
"""Offline analysis for continuous Null/Press single-finger thermal sessions."""

from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
import re
import sys

import numpy as np


_CHECKOUT_ROOT = Path(__file__).resolve().parents[1]
if str(_CHECKOUT_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHECKOUT_ROOT))

from ir_force.single_finger_curve_analysis import (  # noqa: E402
    ANALYSIS_START_S,
    BIN_DURATION_S,
    analyze_rows,
    plot_all_curves,
    write_per_frame_csv,
)


EXPERIMENT_IDENTITY = "single_finger_null_press_continuous_v1"
OUTPUT_TAG_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument("--output-tag")
    args = parser.parse_args(argv)
    if not args.session_dir.is_dir():
        parser.error("--session-dir must be an existing directory")
    if (
        args.output_tag is not None
        and not OUTPUT_TAG_PATTERN.fullmatch(args.output_tag)
    ):
        parser.error(
            "--output-tag must contain lowercase letters, digits, '_' or '-'"
        )
    return args


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"JSONL row must be an object at {path}:{line_number}"
                )
            rows.append(row)
    return rows


def _validate_inputs(session_dir: Path) -> tuple[Path, dict, list[dict]]:
    capture_path = session_dir / "capture.jsonl"
    manifest_path = session_dir / "manifest.json"
    if not capture_path.is_file():
        raise FileNotFoundError(capture_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("experiment_identity") != EXPERIMENT_IDENTITY:
        raise ValueError("manifest experiment identity mismatch")
    rows = _read_jsonl(capture_path)
    metadata = next(
        (row for row in rows if row.get("row_type") == "metadata"),
        None,
    )
    if (
        metadata is None
        or metadata.get("experiment_identity") != EXPERIMENT_IDENTITY
    ):
        raise ValueError("capture experiment identity mismatch")
    return capture_path, manifest, rows


def _validate_output_exclusivity(
    session_dir: Path,
    output_tag: str | None,
) -> dict[str, Path]:
    suffix = "" if output_tag is None else f"_{output_tag}"
    outputs = {
        "analysis": session_dir / f"analysis{suffix}.json",
        "csv": session_dir / f"per_frame{suffix}.csv",
        "figures": session_dir / f"figures{suffix}",
    }
    collisions = [path for path in outputs.values() if path.exists()]
    if collisions:
        raise FileExistsError(collisions[0])
    return outputs


def _json_safe(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if isfinite(float(value)) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    return value


def _plot_paired_differences(paired: dict, clusters: list[dict], path: Path):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    null = np.asarray(
        paired["binned"]["primary_signal_count"]["null"],
        dtype=float,
    )
    press = np.asarray(
        paired["binned"]["primary_signal_count"]["press"],
        dtype=float,
    )
    differences = press - null
    times = ANALYSIS_START_S + (
        np.arange(differences.shape[1]) + 0.5
    ) * BIN_DURATION_S
    figure, axis = plt.subplots(figsize=(10, 5))
    for pair in differences:
        axis.plot(times, pair, color="#7755aa", alpha=0.45, linewidth=1.0)
    axis.plot(
        times,
        np.median(differences, axis=0),
        color="black",
        linewidth=2.0,
        label="pair median",
    )
    for cluster in clusters:
        if cluster.get("p_corrected", 1.0) <= 0.05:
            axis.axvspan(
                ANALYSIS_START_S
                + cluster["start_bin"] * BIN_DURATION_S,
                ANALYSIS_START_S
                + (cluster["end_bin"] + 1) * BIN_DURATION_S,
                color="#8a5cf6",
                alpha=0.16,
            )
    for boundary in (10.0, 15.0):
        axis.axvline(boundary, color="0.4", linestyle="--", linewidth=0.8)
    axis.axhline(0.0, color="0.3", linewidth=0.8)
    axis.set(
        xlabel="Time from A1 start (s)",
        ylabel="Press - Null (count)",
        title="Paired thermal differences",
        xlim=(5.0, 20.0),
    )
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_geometry(paired: dict, path: Path):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    times = ANALYSIS_START_S + (
        np.arange(30) + 0.5
    ) * BIN_DURATION_S
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    specifications = (
        ("uv_displacement_px", "Press - Null UV displacement (px)"),
        ("depth_change_m", "Press - Null depth change (m)"),
    )
    for axis, (field, label) in zip(axes, specifications, strict=True):
        null = np.asarray(paired["binned"][field]["null"], dtype=float)
        press = np.asarray(paired["binned"][field]["press"], dtype=float)
        differences = press - null
        for pair in differences:
            axis.plot(times, pair, color="#397f75", alpha=0.4, linewidth=1.0)
        axis.plot(
            times,
            np.median(differences, axis=0),
            color="black",
            linewidth=2.0,
        )
        axis.axhline(0.0, color="0.3", linewidth=0.8)
        for boundary in (10.0, 15.0):
            axis.axvline(
                boundary,
                color="0.4",
                linestyle="--",
                linewidth=0.8,
            )
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Time from A1 start (s)")
    axes[0].set_title("ROI motion and depth diagnostics")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_analysis(
    session_dir: Path,
    *,
    output_tag: str | None = None,
) -> dict:
    session_dir = Path(session_dir)
    _capture_path, manifest, rows = _validate_inputs(session_dir)
    outputs = _validate_output_exclusivity(session_dir, output_tag)
    result = analyze_rows(rows)
    result["experiment_identity"] = EXPERIMENT_IDENTITY
    result["input_manifest_status"] = manifest.get("status")
    result["decision_metric"] = (
        "paired_exact_whole_curve_sign_flip_cluster_test"
    )

    write_per_frame_csv(rows, outputs["csv"])
    if result["verdict"] != "INCOMPLETE_FOR_PRIMARY_TEST":
        outputs["figures"].mkdir()
        paired = result["paired"]
        thermal_clusters = result["thermal"]["test"]["clusters"]
        plot_all_curves(
            paired,
            thermal_clusters,
            outputs["figures"] / "all_12_normalized_curves.png",
        )
        _plot_paired_differences(
            paired,
            thermal_clusters,
            outputs["figures"] / "paired_difference_diagnostics.png",
        )
        _plot_geometry(
            paired,
            outputs["figures"] / "roi_motion_and_depth.png",
        )
    with outputs["analysis"].open("x", encoding="utf-8") as stream:
        json.dump(
            _json_safe(result),
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
    return result


def main(argv=None) -> int:
    try:
        args = parse_args(argv)
        result = run_analysis(
            args.session_dir,
            output_tag=args.output_tag,
        )
    except (FileExistsError, FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"analysis blocked: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "selected_pair_count": result["selected_pair_count"],
                "session_dir": str(args.session_dir),
            },
            sort_keys=True,
        )
    )
    return int(result["verdict"] == "INCOMPLETE_FOR_PRIMARY_TEST")


if __name__ == "__main__":
    raise SystemExit(main())
