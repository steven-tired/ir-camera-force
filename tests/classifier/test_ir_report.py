from ir_force.classifier.ir_report import (
    split_capture_windows,
    summarize_windows,
)


def test_split_capture_windows_uses_t_capture_reset_between_baseline_and_hold():
    rows = [
        {"t_capture": "0.0"},
        {"t_capture": "0.1"},
        {"t_capture": "0.2"},
        {"t_capture": "0.0"},
        {"t_capture": "0.1"},
    ]

    assert split_capture_windows(rows) == [(0, 3), (3, 5)]


def test_summarize_windows_reports_baseline_hold_delta_and_peak_load():
    feature_rows = [
        {"area_px": "10", "mean_delta": "1", "max_delta": "2"},
        {"area_px": "20", "mean_delta": "2", "max_delta": "3"},
        {"area_px": "100", "mean_delta": "4", "max_delta": "9"},
        {"area_px": "140", "mean_delta": "6", "max_delta": "11"},
    ]
    telemetry_rows = [
        {"t_capture": "0.0", "present_load": "-5", "present_current": "1"},
        {"t_capture": "0.1", "present_load": "-5", "present_current": "2"},
        {"t_capture": "0.0", "present_load": "30", "present_current": "4"},
        {"t_capture": "0.1", "present_load": "50", "present_current": "3"},
    ]

    summary = summarize_windows(feature_rows, telemetry_rows)

    assert summary["baseline"]["area_px_mean"] == 15.0
    assert summary["hold"]["area_px_mean"] == 120.0
    assert summary["change"]["area_px_mean"] == 105.0
    assert summary["hold"]["present_load_max"] == 50.0
    assert summary["hold"]["present_current_max"] == 4.0
