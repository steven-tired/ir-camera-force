#!/usr/bin/env python3
"""Post-hoc thermal-only salvage for a failed formal ROI capture."""

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
from ir_force.single_finger_thermal_salvage import (  # noqa: E402
    analyze_thermal_only,
    plot_feature_overlay,
    plot_salvage_curves,
    thermal_only_feature,
)


OUTPUT_JSON = "salvage_thermal_only.json"
OUTPUT_FIGURES = "figures_salvage_thermal_only"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.session_dir.is_dir():
        parser.error("--session-dir must be an existing directory")
    return args


def _frame_path(session_dir: Path, row: dict) -> Path:
    relative = row.get("frame_artifacts", {}).get("thermal_uint16")
    if not isinstance(relative, str):
        raise ValueError("thermal_uint16 artifact path missing")
    path = (session_dir / relative).resolve()
    try:
        path.relative_to(session_dir.resolve())
    except ValueError as exc:
        raise ValueError("thermal_uint16 artifact escapes session") from exc
    return path


def _load_frame(session_dir: Path, row: dict):
    path = _frame_path(session_dir, row)
    frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise OSError(f"cannot read {path}")
    return frame


def _representative_row(rows: list[dict]) -> dict:
    candidates = [
        row
        for row in rows
        if row.get("row_type") == "frame"
        and row.get("block_index") == 0
        and row.get("condition") == "null"
        and row.get("phase") == "X"
    ]
    if not candidates:
        raise ValueError("representative Null X frame missing")
    return min(
        candidates,
        key=lambda row: abs(float(row["global_elapsed_s"]) - 7.5),
    )


def run_analysis(session_dir: Path) -> dict:
    session_dir = Path(session_dir)
    output_json = session_dir / OUTPUT_JSON
    figures_dir = session_dir / OUTPUT_FIGURES
    if output_json.exists():
        raise FileExistsError(output_json)
    if figures_dir.exists():
        raise FileExistsError(figures_dir)
    _capture_path, manifest, rows = _validate_inputs(session_dir)
    result = analyze_thermal_only(
        rows,
        frame_loader=lambda row: _load_frame(session_dir, row),
    )
    result["experiment_identity"] = EXPERIMENT_IDENTITY
    result["input_manifest_status"] = manifest.get("status")
    result["source_session"] = str(session_dir)

    figures_dir.mkdir()
    curve_render = plot_salvage_curves(
        result,
        figures_dir / "all_12_curves.png",
    )
    representative = _representative_row(rows)
    representative_frame = _load_frame(session_dir, representative)
    representative_feature = thermal_only_feature(representative_frame)
    plot_feature_overlay(
        representative_frame,
        representative_feature,
        figures_dir / "roi_definition.png",
    )
    result["rendered_curve_summary"] = curve_render
    result["representative_roi_frame"] = {
        "frame_index": representative.get("frame_index"),
        "block_index": representative.get("block_index"),
        "condition": representative.get("condition"),
        "phase": representative.get("phase"),
        "global_elapsed_s": representative.get("global_elapsed_s"),
        "feature": representative_feature,
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
        print(f"salvage analysis blocked: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "analysis_role": result["analysis_role"],
                "formal_primary_verdict": result["formal_primary_verdict"],
                "interpolated_bin_count": result["interpolated_bin_count"],
                "segmentation_failure_count": (
                    result["segmentation_failure_count"]
                ),
                "session_dir": str(args.session_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
