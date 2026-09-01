import json

import cv2
import numpy as np

import analyze_single_finger_thermal_tracking as runner
from ir_force.single_finger_curve_protocol import (
    PHASES,
)


def _frame(*, distal_effect=0):
    frame = np.full((120, 160), 29_000, dtype=np.uint16)
    frame[78:120, :] = 29_150
    frame[30:88, 100:160] = 29_500
    frame[48:65, 55:111] = 29_600
    frame[51:62, 62:71] += distal_effect
    return frame


def _session(tmp_path):
    session = tmp_path / "session"
    raw = session / "raw" / "thermal_uint16"
    raw.mkdir(parents=True)
    assert cv2.imwrite(str(raw / "baseline.png"), _frame())
    assert cv2.imwrite(
        str(raw / "effect.png"),
        _frame(distal_effect=40),
    )
    rows = [
        {
            "row_type": "metadata",
            "experiment_identity": runner.EXPERIMENT_IDENTITY,
        }
    ]
    frame_index = 0
    for block_index in range(6):
        for condition in ("null", "press"):
            for phase_index, phase in enumerate(PHASES):
                for phase_bin in range(10):
                    for offset in (0.1, 0.3):
                        active = condition == "press" and phase == "X"
                        rows.append(
                            {
                                "row_type": "frame",
                                "frame_index": frame_index,
                                "block_index": block_index,
                                "condition": condition,
                                "phase": phase,
                                "phase_elapsed_s": phase_bin * 0.5 + offset,
                                "global_elapsed_s": (
                                    phase_index * 5.0
                                    + phase_bin * 0.5
                                    + offset
                                ),
                                "artifact_write_ok": 1,
                                "frame_artifacts": {
                                    "thermal_uint16": (
                                        "raw/thermal_uint16/effect.png"
                                        if active
                                        else "raw/thermal_uint16/baseline.png"
                                    )
                                },
                            }
                        )
                        frame_index += 1
    with (session / "capture.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    (session / "manifest.json").write_text(
        json.dumps(
            {
                "experiment_identity": runner.EXPERIMENT_IDENTITY,
                "status": "incomplete",
            }
        )
    )
    return session


def test_cli_writes_non_overlapping_v2_json_and_three_figures(tmp_path):
    session = _session(tmp_path)
    (session / "analysis.json").write_text('{"primary": true}\n')
    (session / "salvage_thermal_only.json").write_text(
        '{"salvage": true}\n'
    )

    assert runner.main(["--session-dir", str(session)]) == 0

    result = json.loads(
        (session / "analysis_tracked_roi_v2.json").read_text()
    )
    assert result["method_version"] == "tracked_thermal_roi_v2"
    assert result["complete_pair_count"] == 6
    assert "trial_anchors" not in result
    assert len(result["trial_anchor_summaries"]) == 12
    assert all(
        summary["finger_width_px"] >= 10.0
        for summary in result["trial_anchor_summaries"]
    )
    assert (session / "analysis.json").read_text() == '{"primary": true}\n'
    assert (session / "salvage_thermal_only.json").read_text() == (
        '{"salvage": true}\n'
    )
    figures = session / "figures_tracked_roi_v2"
    assert all(
        (figures / name).is_file()
        for name in (
            "all_12_raw_and_rolling.png",
            "paired_press_minus_null.png",
            "roi_definition.png",
        )
    )


def test_cli_refuses_to_overwrite_existing_v2_output(tmp_path):
    session = _session(tmp_path)
    (session / "analysis_tracked_roi_v2.json").write_text("{}")

    assert runner.main(["--session-dir", str(session)]) == 1
