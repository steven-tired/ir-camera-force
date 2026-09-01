from __future__ import annotations

import importlib

import pytest


def test_lag_scan_reports_positive_lag_when_ir_arrives_one_frame_later():
    module = importlib.import_module("analyze_ir_oak_squeeze_proxy")
    proxy = (0.0, 1.0, -1.0, 2.0, 0.0, -2.0, 1.0)
    ir = (-9.0, 0.0, 1.0, -1.0, 2.0, 0.0, -2.0)
    records = [
        {
            "sequence_id": 1,
            "step_index": 0,
            "step_elapsed_s": float(frame) / 10.0,
            "pinch_norm": proxy[frame],
            "neg_area": ir[frame],
        }
        for frame in range(len(proxy))
    ]

    rows = module.lag_scan(records, max_lag_frames=2)
    best = max(rows, key=lambda row: row["median_pearson"])

    assert best["lag_frames"] == 1
    assert best["segment_count"] == 1


def test_window_change_reports_persistence_when_proxy_relaxes_but_ir_does_not():
    module = importlib.import_module("analyze_ir_oak_squeeze_proxy")
    records = [
        {
            "sequence_id": 1,
            "step_index": 0,
            "step_elapsed_s": step_elapsed_s,
            "pinch_norm": pinch_norm,
            "neg_area": neg_area,
        }
        for step_elapsed_s, pinch_norm, neg_area in [
            (0.0, 0.10, 3000.0),
            (0.2, 0.12, 3100.0),
            (0.8, 0.40, 3050.0),
            (1.0, 0.42, 3000.0),
        ]
    ]

    summary = module.window_change(records, window_s=0.25)

    assert len(summary) == 1
    assert summary[0]["sequence_id"] == 1
    assert summary[0]["step_index"] == 0
    assert summary[0]["frame_count"] == 4
    assert summary[0]["proxy_first"] == pytest.approx(0.11)
    assert summary[0]["proxy_last"] == pytest.approx(0.41)
    assert summary[0]["proxy_change"] == pytest.approx(0.30)
    assert summary[0]["ir_first"] == pytest.approx(3050.0)
    assert summary[0]["ir_last"] == pytest.approx(3025.0)
    assert summary[0]["ir_change"] == pytest.approx(-25.0)
