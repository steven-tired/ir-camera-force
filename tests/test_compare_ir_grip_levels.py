from __future__ import annotations

from pathlib import Path

import compare_ir_grip_levels


def test_visible_evidence_counts_continuous_and_preflight_visible(tmp_path: Path):
    trial_root = tmp_path / "trial"
    (trial_root / "flir_visible").mkdir(parents=True)
    (trial_root / "preflight").mkdir()
    (trial_root / "flir_visible" / "frame_000000.png").write_bytes(b"visible")
    (trial_root / "preflight" / "flir_visible.png").write_bytes(b"preflight")

    evidence = compare_ir_grip_levels._visible_evidence(trial_root)

    assert evidence.continuous_frames == 1
    assert evidence.preflight_frames == 1
    assert evidence.has_any


def test_trial_pairs_reads_requested_feature_file(tmp_path: Path, monkeypatch):
    spec = compare_ir_grip_levels.TrialSpec("foam", "soft", "low", 2)
    trial_root = tmp_path / "trials" / compare_ir_grip_levels.trial_id(spec)
    trial_root.mkdir(parents=True)
    (trial_root / "ir_features_roi.csv").write_text("frame,area_px,mean_delta,max_delta\n", encoding="utf-8")
    (trial_root / "telemetry.csv").write_text(
        "t_capture,present_load,present_current,gripper_pos\n",
        encoding="utf-8",
    )

    def fake_summarize_window_pairs(feature_rows, telemetry_rows):
        assert feature_rows == []
        assert telemetry_rows == []
        return ["roi-pair"]

    monkeypatch.setattr(compare_ir_grip_levels, "summarize_window_pairs", fake_summarize_window_pairs)

    found_root, pairs = compare_ir_grip_levels._trial_pairs(
        tmp_path,
        spec,
        feature_name="ir_features_roi.csv",
    )

    assert found_root == trial_root
    assert pairs == ["roi-pair"]
