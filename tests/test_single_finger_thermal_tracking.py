import numpy as np
import warnings

from ir_force.single_finger_thermal_tracking import (
    TrialTracker,
    analyze_tracked_thermal,
    initialize_trial_anchor,
    plot_paired_median,
    plot_raw_and_rolling_curves,
    plot_tracking_overlay,
)
from ir_force.single_finger_curve_protocol import (
    PHASES,
)


def _synthetic_frame(*, shift_x=0, distal_effect=0):
    frame = np.full((120, 160), 29_000, dtype=np.uint16)
    frame[78:120, :] = 29_150
    frame[30:88, 100 + shift_x : 160] = 29_500
    frame[48:65, 55 + shift_x : 111 + shift_x] = 29_600
    frame[51:62, 62 + shift_x : 71 + shift_x] += distal_effect
    return frame


def test_a1_anchor_builds_disjoint_eroded_finger_rois_and_far_desk_roi():
    anchor = initialize_trial_anchor(
        [_synthetic_frame() for _ in range(5)]
    )

    distal = anchor["distal_mask"]
    proximal = anchor["proximal_mask"]
    desk = anchor["desk_mask"]
    eroded = anchor["eroded_hand_mask"]

    assert anchor["finger_width_px"] >= 10.0
    assert np.count_nonzero(distal) >= 35
    assert np.count_nonzero(proximal) >= 35
    assert np.count_nonzero(desk) == 15 * 15
    assert not np.any(distal & proximal)
    assert np.all(eroded[distal])
    assert np.all(eroded[proximal])
    assert not np.any(anchor["hand_mask"] & desk)


def test_distorted_distal_cross_section_is_not_misreported_as_wide_finger():
    frame = _synthetic_frame()
    frame[18:90, 76:84] = 29_600
    frame[64:105, 60:72] = 29_600

    anchor = initialize_trial_anchor([frame for _ in range(5)])

    assert 0.0 < anchor["finger_width_px"] < 10.0


def test_tracker_follows_one_pixel_but_rejects_a_three_pixel_jump():
    anchor = initialize_trial_anchor(
        [_synthetic_frame() for _ in range(5)]
    )
    tracker = TrialTracker(anchor)

    accepted = tracker.measure(
        _synthetic_frame(shift_x=1, distal_effect=40)
    )
    desk_center_before = accepted["desk_uv"]
    rejected = TrialTracker(anchor).measure(
        _synthetic_frame(shift_x=3)
    )

    assert accepted["tracking_valid"] is True
    assert accepted["shift_uv"] == [1, 0]
    assert accepted["primary_signal_count"] == 40.0
    assert accepted["desk_count"] == 29_150.0
    assert accepted["desk_uv"] == desk_center_before
    assert rejected["tracking_valid"] is False
    assert rejected["tracking_reasons"] == ["center_step_exceeded"]
    assert rejected["primary_signal_count"] is None


def test_finger_width_gate_has_small_numeric_tolerance_at_ten_pixels():
    anchor = initialize_trial_anchor(
        [_synthetic_frame() for _ in range(5)]
    )
    anchor["finger_width_px"] = 9.99

    measured = TrialTracker(anchor).measure(_synthetic_frame())

    assert measured["tracking_valid"] is True


def test_tracker_rejects_component_area_jump():
    anchor = initialize_trial_anchor(
        [_synthetic_frame() for _ in range(5)]
    )
    expanded = _synthetic_frame()
    expanded[18:102, 90:160] = 29_600

    measured = TrialTracker(anchor).measure(expanded)

    assert measured["tracking_valid"] is False
    assert measured["tracking_reasons"] == ["component_area_jump"]


def _rows_and_frames(*, distal_effect=40, desk_effect=0):
    rows = []
    frames = {}
    frame_index = 0
    for block_index in range(6):
        for condition in ("null", "press"):
            for phase_index, phase in enumerate(PHASES):
                for phase_bin in range(10):
                    for offset in (0.1, 0.3):
                        elapsed = (
                            phase_index * 5.0
                            + phase_bin * 0.5
                            + offset
                        )
                        active = condition == "press" and phase == "X"
                        frame = _synthetic_frame(
                            distal_effect=distal_effect if active else 0
                        )
                        if active and desk_effect:
                            frame[78:120, :50] += desk_effect
                        frames[frame_index] = frame
                        rows.append(
                            {
                                "row_type": "frame",
                                "frame_index": frame_index,
                                "block_index": block_index,
                                "condition": condition,
                                "phase": phase,
                                "phase_elapsed_s": phase_bin * 0.5 + offset,
                                "global_elapsed_s": elapsed,
                                "artifact_write_ok": 1,
                            }
                        )
                        frame_index += 1
    return rows, frames


def test_analysis_uses_raw_bins_without_interpolation_and_keeps_desk_diagnostic():
    rows, frames = _rows_and_frames(distal_effect=40, desk_effect=300)

    result = analyze_tracked_thermal(
        rows,
        frame_loader=lambda row: frames[row["frame_index"]],
    )

    assert result["analysis_role"] == (
        "posthoc_tracked_roi_v2_not_preregistered"
    )
    assert result["complete_pair_count"] == 6
    assert len(result["curves"]) == 12
    assert result["missing_bin_count"] == 0
    assert "interpolated_bins" not in result
    assert result["primary"]["significant_clusters"][0][
        "p_corrected"
    ] == 0.03125
    assert result["primary"]["median_press_minus_null_by_phase_count"][
        "X"
    ] == 40.0
    assert result["desk_drift_diagnostic"]["maximum_absolute_change_count"] >= 300
    assert result["invalid_frame_count"] == 0


def test_desk_change_does_not_create_primary_effect():
    rows, frames = _rows_and_frames(distal_effect=0, desk_effect=500)

    result = analyze_tracked_thermal(
        rows,
        frame_loader=lambda row: frames[row["frame_index"]],
    )

    assert result["primary"]["significant_clusters"] == []
    assert result["primary"]["median_press_minus_null_by_phase_count"][
        "X"
    ] == 0.0
    assert result["desk_drift_diagnostic"]["maximum_absolute_change_count"] >= 500


def test_v2_plots_show_raw_rolling_paired_and_fixed_masks(tmp_path):
    rows, frames = _rows_and_frames()
    result = analyze_tracked_thermal(
        rows,
        frame_loader=lambda row: frames[row["frame_index"]],
    )

    curves = plot_raw_and_rolling_curves(
        result,
        tmp_path / "raw_rolling.png",
    )
    paired = plot_paired_median(result, tmp_path / "paired.png")
    overlay = plot_tracking_overlay(
        result["trial_anchors"][0],
        tmp_path / "overlay.png",
    )

    assert curves["raw_curve_count"] == 12
    assert curves["rolling_curve_count"] == 12
    assert paired["paired_curve_count"] == 6
    assert paired["pair_count_by_bin"] == [6] * 30
    assert overlay["desk_pixel_count"] == 225
    assert all(
        (tmp_path / name).is_file()
        for name in ("raw_rolling.png", "paired.png", "overlay.png")
    )


def test_paired_plot_handles_missing_bins_without_runtime_warning(tmp_path):
    result = {
        "binned": {
            "primary_signal_count": {
                "null": [[None] * 30 for _ in range(6)],
                "press": [[None] * 30 for _ in range(6)],
            }
        }
    }

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        rendered = plot_paired_median(result, tmp_path / "missing.png")

    assert rendered["paired_curve_count"] == 6
    assert rendered["pair_count_by_bin"] == [0] * 30
    assert (tmp_path / "missing.png").is_file()


def test_invalid_frame_count_excludes_trial_level_baseline_failure():
    rows, frames = _rows_and_frames()
    for row in rows:
        if row["block_index"] == 0 and row["condition"] == "null":
            frame = _synthetic_frame()
            frame[18:90, 76:84] = 29_600
            frame[64:105, 60:72] = 29_600
            frames[row["frame_index"]] = frame

    result = analyze_tracked_thermal(
        rows,
        frame_loader=lambda row: frames[row["frame_index"]],
    )
    invalid_from_curves = sum(
        not valid
        for curve in result["curves"]
        for valid in curve["tracking_valid"]
    )

    assert result["invalid_frame_count"] == invalid_from_curves
    assert result["analysis_failure_counts"]["A1_baseline_missing"] == 1
