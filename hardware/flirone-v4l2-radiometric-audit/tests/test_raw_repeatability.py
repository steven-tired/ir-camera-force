from __future__ import annotations

from array import array

import pytest


def _frame(index: int, timestamp_s: float, value: int, *, ffc_state: str = "normal", repeated: bool = False):
    repeatability = __import__("raw_repeatability")
    return repeatability.RawFrame(
        frame_index=index,
        timestamp_ns=round(timestamp_s * 1_000_000_000),
        width=2,
        height=2,
        values=array("H", [value, value, value, value]),
        ffc_state=ffc_state,
        repeated_frame_flag=repeated,
        display_mapping_mode="dynamic_per_frame_min_max",
        raw_low=None,
        raw_high=None,
    )


def test_ffc_metric_compares_pre_and_post_unique_normal_windows():
    repeatability = __import__("raw_repeatability")
    roi = repeatability.RawROI(x=0, y=0, width=2, height=2)
    frames = [
        _frame(0, 0.0, 100),
        _frame(1, 1.0, 100),
        _frame(2, 2.0, 100, repeated=True),
        _frame(3, 3.0, 120, ffc_state="ffc"),
        _frame(4, 4.0, 121, ffc_state="ffc"),
        _frame(5, 5.0, 110, ffc_state="post_ffc_discarded"),
        _frame(6, 6.0, 104),
        _frame(7, 7.0, 104),
    ]

    metric = repeatability.analyze_ffc_recovery(
        frames,
        roi=roi,
        window_s=3.0,
        post_delay_s=1.0,
    )

    assert metric["pre_median"] == 100
    assert metric["post_median"] == 104
    assert metric["post_minus_pre_counts"] == 4
    assert metric["pre_unique_frame_count"] == 2
    assert metric["post_unique_frame_count"] == 2


def test_ffc_event_metrics_keep_disjoint_shutter_events_separate():
    repeatability = __import__("raw_repeatability")
    roi = repeatability.RawROI(x=0, y=0, width=2, height=2)
    frames = [
        _frame(0, 0.0, 100),
        _frame(1, 1.0, 100),
        _frame(2, 3.0, 120, ffc_state="ffc"),
        _frame(3, 4.0, 120, ffc_state="ffc"),
        _frame(4, 5.0, 120, ffc_state="post_ffc_discarded"),
        _frame(5, 6.0, 104),
        _frame(6, 7.0, 104),
        _frame(7, 10.0, 104),
        _frame(8, 11.0, 104),
        _frame(9, 13.0, 130, ffc_state="ffc"),
        _frame(10, 14.0, 130, ffc_state="ffc"),
        _frame(11, 15.0, 130, ffc_state="post_ffc_discarded"),
        _frame(12, 16.0, 108),
        _frame(13, 17.0, 108),
    ]

    events = repeatability.analyze_ffc_recovery_events(
        frames,
        roi=roi,
        window_s=2.0,
        post_delay_s=1.0,
        cluster_gap_s=1.0,
    )

    assert len(events) == 2
    assert [event["post_minus_pre_counts"] for event in events] == [4, 4]
    assert [event["ffc_frame_count"] for event in events] == [2, 2]


def test_suggest_fixed_range_pads_and_aligns_a_narrow_dynamic_distribution():
    repeatability = __import__("raw_repeatability")
    frames = [
        _frame(0, 0.0, 1000),
        _frame(1, 1.0, 1020),
        _frame(2, 2.0, 1100),
    ]

    suggestion = repeatability.suggest_fixed_range(frames, minimum_span=256, alignment=16)

    assert suggestion["raw_low"] == 960
    assert suggestion["raw_high"] == 1136
    assert suggestion["source_unique_frame_count"] == 3


def test_phase_metric_uses_only_unique_normal_frames_inside_the_event_window():
    repeatability = __import__("raw_repeatability")
    roi = repeatability.RawROI(x=0, y=0, width=2, height=2)
    frames = [
        _frame(0, 0.0, 100),
        _frame(1, 1.0, 100),
        _frame(2, 2.0, 150, repeated=True),
        _frame(3, 3.0, 120),
        _frame(4, 4.0, 120),
    ]

    baseline = repeatability.phase_metric(frames, roi=roi, start_ns=0, end_ns=2_000_000_000)
    intrusion = repeatability.phase_metric(frames, roi=roi, start_ns=2_000_000_000, end_ns=5_000_000_000)

    assert baseline["median"] == 100
    assert intrusion["median"] == 120
    assert intrusion["unique_frame_count"] == 2


def test_restart_offsets_are_relative_to_the_first_stable_run():
    repeatability = __import__("raw_repeatability")

    offsets = repeatability.restart_offsets(
        [
            {"run_id": "restart_01", "stable_median": 100},
            {"run_id": "restart_02", "stable_median": 104},
            {"run_id": "restart_03", "stable_median": 97},
        ]
    )

    assert offsets["reference_run_id"] == "restart_01"
    assert offsets["by_run"]["restart_02"] == 4
    assert offsets["by_run"]["restart_03"] == -3
    assert offsets["max_absolute_offset_counts"] == 4


def test_dynamic_metrics_compare_each_hot_hand_phase_with_its_predeclared_baseline():
    repeatability = __import__("raw_repeatability")
    roi = repeatability.RawROI(x=0, y=0, width=2, height=2)
    frames = [
        _frame(0, 0.0, 100),
        _frame(1, 1.0, 100),
        _frame(2, 2.0, 150),
        _frame(3, 3.0, 500, repeated=True),
        _frame(4, 4.0, 140),
        _frame(5, 5.0, 105),
    ]
    windows = {
        "baseline_01": (0, 1_000_000_000),
        "hot_hand_01": (2_000_000_000, 4_000_000_000),
        "recovery_01": (5_000_000_000, 5_000_000_000),
    }

    summary = repeatability.analyze_dynamic_phases(
        frames,
        target_roi=roi,
        control_roi=roi,
        phase_windows=windows,
    )

    assert summary["phases"]["hot_hand_01"]["target"]["median"] == 145
    assert summary["cycles"][0]["target_hot_minus_baseline_counts"] == 45
    assert summary["cycles"][0]["target_recovery_minus_baseline_counts"] == 5
