import csv
import json

import cv2
import pytest

import analyze_single_finger_curve as runner
from ir_force.single_finger_curve_protocol import (
    PHASES,
)


def test_analysis_json_writer_preserves_booleans():
    assert runner._json_safe(True) is True
    assert runner._json_safe(False) is False


def _rows(*, thermal_effect=False, geometry_effect=False, blocks=6):
    amplitudes = (2.0, 2.1, 1.9, 2.2, 1.8, 2.05)
    rows = []
    for block_index in range(blocks):
        for condition_index, condition in enumerate(("null", "press")):
            baseline = 100.0 + block_index * 7.0 + condition_index * 2.0
            thermal_clock = block_index * 100.0 + condition_index * 30.0
            for phase_index, phase in enumerate(PHASES):
                for phase_bin in range(10):
                    for offset in (0.1, 0.3):
                        phase_elapsed = phase_bin * 0.5 + offset
                        elapsed = phase_index * 5.0 + phase_elapsed
                        analysis_bin = int((elapsed - 5.0) // 0.5)
                        active = 0 <= analysis_bin < 16
                        press = condition == "press"
                        effect = (
                            amplitudes[block_index]
                            if thermal_effect and press and active
                            else 0.0
                        )
                        u_effect = (
                            amplitudes[block_index] / 2.0
                            if geometry_effect and press and 8 <= analysis_bin < 18
                            else 0.0
                        )
                        rows.append(
                            {
                                "row_type": "frame",
                                "block_index": block_index,
                                "condition": condition,
                                "phase": phase,
                                "phase_elapsed_s": phase_elapsed,
                                "global_elapsed_s": elapsed,
                                "thermal_host_s": thermal_clock + elapsed,
                                "tracking_valid": True,
                                "ffc_in_progress": False,
                                "ffc_state": "complete",
                                "artifact_write_ok": True,
                                "primary_signal_count": baseline + effect,
                                "distal_thermal_u_px": 30.0 + u_effect,
                                "distal_thermal_v_px": 40.0,
                                "distal_depth_m": 0.45,
                                "tlinear_enabled": True,
                                "tlinear_resolution_k": 0.01,
                            }
                        )
    return rows


def _session(tmp_path, rows):
    session = tmp_path / "single_finger_surface_press_curve_01"
    session.mkdir()
    with (session / "capture.jsonl").open("x") as stream:
        stream.write(
            json.dumps(
                {
                    "row_type": "metadata",
                    "experiment_identity": runner.EXPERIMENT_IDENTITY,
                }
            )
            + "\n"
        )
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    (session / "manifest.json").write_text(
        json.dumps(
            {
                "experiment_identity": runner.EXPERIMENT_IDENTITY,
                "status": "complete",
            }
        )
    )
    return session


def test_cli_requires_session_and_rejects_wrong_identity_or_collision(tmp_path):
    with pytest.raises(SystemExit, match="2"):
        runner.parse_args([])

    session = _session(tmp_path, _rows())
    manifest = session / "manifest.json"
    manifest.write_text(json.dumps({"experiment_identity": "wrong"}))
    assert runner.main(["--session-dir", str(session)]) == 1

    manifest.write_text(
        json.dumps({"experiment_identity": runner.EXPERIMENT_IDENTITY})
    )
    (session / "analysis.json").write_text("{}")
    assert runner.main(["--session-dir", str(session)]) == 1


def test_output_tag_allows_non_destructive_reanalysis(tmp_path):
    session = _session(tmp_path, _rows())
    (session / "analysis.json").write_text("{}")

    assert runner.main(
        [
            "--session-dir",
            str(session),
            "--output-tag",
            "boolfix",
        ]
    ) == 0

    assert (session / "analysis_boolfix.json").is_file()
    assert (session / "per_frame_boolfix.csv").is_file()
    assert (session / "figures_boolfix").is_dir()
    assert (session / "analysis.json").read_text() == "{}"


def test_end_to_end_positive_writes_all_outputs_and_exact_result(tmp_path):
    rows = _rows(thermal_effect=True)
    session = _session(tmp_path, rows)

    assert runner.main(["--session-dir", str(session)]) == 0

    analysis = json.loads((session / "analysis.json").read_text())
    assert analysis["verdict"] == "SIGNIFICANT_CURVE_SEPARATION"
    assert analysis["selected_pair_count"] == 6
    assert analysis["thermal"]["significant_clusters"][0][
        "p_corrected"
    ] == pytest.approx(0.03125)
    required = (
        session / "per_frame.csv",
        session / "figures/all_12_normalized_curves.png",
        session / "figures/paired_difference_diagnostics.png",
        session / "figures/roi_motion_and_depth.png",
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in required)
    with (session / "per_frame.csv").open(newline="") as stream:
        assert sum(1 for _row in csv.DictReader(stream)) == len(rows)
    assert cv2.imread(str(required[1]), cv2.IMREAD_COLOR) is not None


@pytest.mark.parametrize(
    ("thermal_effect", "geometry_effect", "expected"),
    [
        (False, False, "NO_DETECTED_SEPARATION_5S"),
        (True, True, "GEOMETRY_CONFOUNDED"),
    ],
)
def test_completed_negative_and_geometry_confound_verdicts(
    tmp_path,
    thermal_effect,
    geometry_effect,
    expected,
):
    session = _session(
        tmp_path,
        _rows(
            thermal_effect=thermal_effect,
            geometry_effect=geometry_effect,
        ),
    )

    assert runner.main(["--session-dir", str(session)]) == 0

    analysis = json.loads((session / "analysis.json").read_text())
    assert analysis["verdict"] == expected
    assert "eba" not in json.dumps(analysis).lower()


def test_incomplete_capture_writes_explicit_verdict_and_exits_one(tmp_path):
    session = _session(tmp_path, _rows(thermal_effect=True, blocks=5))

    assert runner.main(["--session-dir", str(session)]) == 1

    analysis = json.loads((session / "analysis.json").read_text())
    assert analysis["verdict"] == "INCOMPLETE_FOR_PRIMARY_TEST"
    assert analysis["selected_pair_count"] == 5
