from __future__ import annotations

from ir_force.classifier.ir_diagnostics import summarize_window_pairs


def _feature(area_px: int) -> dict[str, str]:
    return {"area_px": str(area_px), "mean_delta": "1", "max_delta": "2"}


def _telemetry(t_capture: float, load: int = 0, current: int = 0) -> dict[str, str]:
    return {
        "t_capture": str(t_capture),
        "present_load": str(load),
        "present_current": str(current),
        "gripper_pos": "10",
    }


def test_summarize_window_pairs_exposes_appended_xhigh_area_collapse():
    feature_rows = [
        _feature(300),
        _feature(300),
        _feature(20),
        _feature(20),
        _feature(3000),
        _feature(3000),
        _feature(200),
        _feature(200),
    ]
    telemetry_rows = [
        _telemetry(0),
        _telemetry(1),
        _telemetry(0, load=360, current=10),
        _telemetry(1, load=60, current=2),
        _telemetry(0),
        _telemetry(1),
        _telemetry(0, load=360, current=10),
        _telemetry(1, load=60, current=2),
    ]

    pairs = summarize_window_pairs(feature_rows, telemetry_rows)

    assert len(pairs) == 2
    assert pairs[0].area_delta_px == -280.0
    assert pairs[1].area_delta_px == -2800.0
    assert "area_collapsed" in pairs[1].flags
    assert pairs[1].hold_load_peak == 360.0
