#!/usr/bin/env python3
"""Offline post-hoc analysis with A1-frozen, temporally tracked thermal ROIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2


_CHECKOUT_ROOT = Path(__file__).resolve().parents[1]
if str(_CHECKOUT_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHECKOUT_ROOT))

from analyze_single_finger_curve import (  # noqa: E402
    EXPERIMENT_IDENTITY,
    _json_safe,
    _validate_inputs,
)
from ir_force.single_finger_thermal_tracking import (  # noqa: E402
    analyze_tracked_thermal,
    plot_paired_median,
    plot_raw_and_rolling_curves,
    plot_tracking_overlay,
)


OUTPUT_JSON = "analysis_tracked_roi_v2.json"
OUTPUT_FIGURES = "figures_tracked_roi_v2"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.session_dir.is_dir():
        parser.error("--session-dir must be an existing directory")
    return args


def _load_frame(session_dir: Path, row: dict):
    relative = row.get("frame_artifacts", {}).get("thermal_uint16")
    if not isinstance(relative, str):
        raise ValueError("thermal_uint16 artifact path missing")
    path = (session_dir / relative).resolve()
    try:
        path.relative_to(session_dir.resolve())
    except ValueError as exc:
        raise ValueError("thermal_uint16 artifact escapes session") from exc
    frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise OSError(f"cannot read {path}")
    return frame


def run_analysis(session_dir: Path) -> dict:
    session_dir = Path(session_dir)
    output_json = session_dir / OUTPUT_JSON
    figures_dir = session_dir / OUTPUT_FIGURES
    if output_json.exists():
        raise FileExistsError(output_json)
    if figures_dir.exists():
        raise FileExistsError(figures_dir)
    _capture_path, manifest, rows = _validate_inputs(session_dir)
    result = analyze_tracked_thermal(
        rows,
        frame_loader=lambda row: _load_frame(session_dir, row),
    )
    result["experiment_identity"] = EXPERIMENT_IDENTITY
    result["input_manifest_status"] = manifest.get("status")
    result["source_session"] = str(session_dir)

    figures_dir.mkdir()
    result["rendered_raw_and_rolling"] = plot_raw_and_rolling_curves(
        result,
        figures_dir / "all_12_raw_and_rolling.png",
    )
    result["rendered_paired_median"] = plot_paired_median(
        result,
        figures_dir / "paired_press_minus_null.png",
    )
    anchors = result.pop("trial_anchors")
    if not anchors:
        raise ValueError("no valid A1 anchor available for ROI overlay")
    result["trial_anchor_summaries"] = [
        {
            "block_index": anchor["block_index"],
            "condition": anchor["condition"],
            "finger_width_px": anchor["finger_width_px"],
            "distal_pixel_count": int(
                anchor["distal_mask"].sum()
            ),
            "proximal_pixel_count": int(
                anchor["proximal_mask"].sum()
            ),
            "desk_pixel_count": int(anchor["desk_mask"].sum()),
        }
        for anchor in anchors
    ]
    result["rendered_roi_overlay"] = plot_tracking_overlay(
        anchors[0],
        figures_dir / "roi_definition.png",
    )
    result["representative_anchor"] = {
        "block_index": anchors[0]["block_index"],
        "condition": anchors[0]["condition"],
        "finger_width_px": anchors[0]["finger_width_px"],
        **result["rendered_roi_overlay"],
    }
    with output_json.open("x", encoding="utf-8") as stream:
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
        result = run_analysis(args.session_dir)
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"tracked ROI analysis blocked: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "analysis_role": result["analysis_role"],
                "complete_pair_count": result["complete_pair_count"],
                "invalid_frame_count": result["invalid_frame_count"],
                "missing_bin_count": result["missing_bin_count"],
                "session_dir": str(args.session_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
