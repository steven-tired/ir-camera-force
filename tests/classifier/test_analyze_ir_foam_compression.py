from __future__ import annotations

import csv
import importlib
import json


def test_summarize_trial_uses_only_unique_valid_gated_hold_frames(tmp_path):
    module = importlib.import_module("analyze_ir_foam_compression")
    trial = tmp_path / "foam-compression_s01_foam_zk_rep01"
    trial.mkdir()
    (trial / "metadata.json").write_text(json.dumps({"primary_ir_feature": "foam_center_norm"}))
    telemetry_fields = [
        "frame", "block", "phase", "state", "target_compression_pct", "sequence_id", "step_index",
        "step_name", "action_attempt", "step_elapsed_s", "compression_pct", "marker_detected",
    ]
    feature_fields = ["frame", "thermal_frame_sha1", "frozen_frame_flag", "foam_center_norm", "background_norm", "left_contact_norm", "right_contact_norm"]
    telemetry_rows = [
        [0, "steady_state", "stable_hold", "C0", 0, 1, 1, "c0", 1, 1.0, 0.5, True],
        [1, "steady_state", "stable_hold", "C0", 0, 1, 1, "c0", 1, 4.0, 0.2, True],
        [2, "steady_state", "stable_hold", "C30", 30, 1, 2, "c30", 1, 0.5, 29.0, True],
        [3, "steady_state", "stable_hold", "C30", 30, 1, 2, "c30", 1, 4.0, 30.2, True],
        [4, "steady_state", "stable_hold", "C30", 30, 1, 2, "c30", 1, 4.5, 30.1, True],
    ]
    feature_rows = [
        [0, "a", False, 0.10, 0.01, 0.2, 0.2],
        [1, "b", False, 0.12, 0.01, 0.2, 0.2],
        [2, "c", False, 0.30, 0.01, 0.2, 0.2],
        [3, "d", False, 0.42, 0.01, 0.2, 0.2],
        [4, "d", False, 0.99, 0.01, 0.2, 0.2],
    ]
    with (trial / "telemetry.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(telemetry_fields)
        writer.writerows(telemetry_rows)
    with (trial / "frame_features.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(feature_fields)
        writer.writerows(feature_rows)

    summary = module.summarize_trial(trial, stable_window_s=3.0)

    assert summary.primary_feature == "foam_center_norm"
    assert summary.valid_frame_count == 4
    assert len(summary.steps) == 2
    c30 = next(step for step in summary.steps if step["state"] == "C30")
    assert c30["foam_center_norm"] == 0.42
    assert c30["frame_count"] == 1


def test_summarize_trial_excludes_an_attempt_marked_invalid_in_events(tmp_path):
    module = importlib.import_module("analyze_ir_foam_compression")
    trial = tmp_path / "foam-compression_s01_foam_zk_rep01"
    trial.mkdir()
    (trial / "metadata.json").write_text(json.dumps({"primary_ir_feature": "foam_center_norm"}))
    telemetry_fields = [
        "frame", "block", "phase", "state", "target_compression_pct", "sequence_id", "step_index",
        "step_name", "action_attempt", "step_elapsed_s", "compression_pct", "marker_detected",
    ]
    feature_fields = ["frame", "thermal_frame_sha1", "frozen_frame_flag", "foam_center_norm", "background_norm", "left_contact_norm", "right_contact_norm"]
    with (trial / "telemetry.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(telemetry_fields)
        writer.writerow([0, "steady_state", "stable_hold", "C30", 30, 1, 2, "c30", 1, 4.0, 30.0, True])
        writer.writerow([1, "steady_state", "stable_hold", "C30", 30, 1, 2, "c30", 2, 4.0, 30.0, True])
    with (trial / "frame_features.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(feature_fields)
        writer.writerow([0, "a", False, 0.95, 0.01, 0.2, 0.2])
        writer.writerow([1, "b", False, 0.40, 0.01, 0.2, 0.2])
    with (trial / "events.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_type", "block", "sequence_id", "step_index", "state", "action_attempt"])
        writer.writeheader()
        writer.writerow({"event_type": "invalid", "block": "steady_state", "sequence_id": 1, "step_index": 2, "state": "C30", "action_attempt": 1})

    summary = module.summarize_trial(trial)

    assert len(summary.steps) == 1
    assert summary.steps[0]["action_attempt"] == 2
    assert summary.steps[0]["foam_center_norm"] == 0.40
