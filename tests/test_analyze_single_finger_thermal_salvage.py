import json

import cv2
import numpy as np

import analyze_single_finger_thermal_salvage as runner
from ir_force.single_finger_curve_protocol import (
    PHASES,
)


def _frame(effect=0):
    frame = np.full((120, 160), 29_000, dtype=np.uint16)
    frame[30:86, 100:160] = 29_500
    frame[50:65, 80:106] = 29_600
    frame[52:63, 84:92] += effect
    return frame


def _session(tmp_path):
    session = tmp_path / "session"
    raw = session / "raw" / "thermal_uint16"
    raw.mkdir(parents=True)
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
                    elapsed = phase_index * 5.0 + phase_bin * 0.5 + 0.25
                    relative = (
                        f"raw/thermal_uint16/frame_{frame_index:06d}.png"
                    )
                    effect = (
                        40
                        if condition == "press" and phase == "X"
                        else 0
                    )
                    assert cv2.imwrite(str(session / relative), _frame(effect))
                    rows.append(
                        {
                            "row_type": "frame",
                            "frame_index": frame_index,
                            "block_index": block_index,
                            "condition": condition,
                            "phase": phase,
                            "phase_elapsed_s": phase_bin * 0.5 + 0.25,
                            "global_elapsed_s": elapsed,
                            "artifact_write_ok": 1,
                            "frame_artifacts": {
                                "thermal_uint16": relative,
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


def test_cli_writes_explicit_salvage_outputs_without_touching_primary(tmp_path):
    session = _session(tmp_path)
    (session / "analysis.json").write_text('{"primary": true}\n')

    assert runner.main(["--session-dir", str(session)]) == 0

    result = json.loads(
        (session / "salvage_thermal_only.json").read_text()
    )
    assert result["analysis_role"] == (
        "salvage_descriptive_not_preregistered"
    )
    assert result["formal_primary_verdict"] == "INCOMPLETE_FOR_PRIMARY_TEST"
    assert (session / "analysis.json").read_text() == '{"primary": true}\n'
    assert (
        session
        / "figures_salvage_thermal_only"
        / "all_12_curves.png"
    ).is_file()
    assert (
        session / "figures_salvage_thermal_only" / "roi_definition.png"
    ).is_file()


def test_cli_refuses_to_overwrite_existing_salvage(tmp_path):
    session = _session(tmp_path)
    (session / "salvage_thermal_only.json").write_text("{}")

    assert runner.main(["--session-dir", str(session)]) == 1
