import numpy as np
import pytest

from ir_force.single_finger_thermal_salvage import (
    analyze_thermal_only,
    plot_feature_overlay,
    plot_salvage_curves,
    thermal_only_feature,
)
from ir_force.single_finger_curve_protocol import (
    PHASES,
)


def _synthetic_hand_frame(effect=0):
    frame = np.full((120, 160), 29_000, dtype=np.uint16)
    frame[30:86, 100:160] = 29_500
    frame[50:65, 80:106] = 29_600
    frame[52:63, 84:92] += effect
    return frame


def test_thermal_only_feature_uses_left_tip_and_two_inward_patches():
    feature = thermal_only_feature(_synthetic_hand_frame())

    assert feature["tip_uv"] == pytest.approx((82.0, 57.0), abs=2.0)
    assert feature["distal_uv"][0] > feature["tip_uv"][0]
    assert feature["reference_uv"][0] > feature["distal_uv"][0]
    assert feature["primary_signal_count"] > 0.0


def _complete_rows_and_frames():
    rows = []
    frames = {}
    frame_index = 0
    amplitudes = (40, 42, 38, 44, 36, 41)
    for block_index in range(6):
        for condition in ("null", "press"):
            for phase_index, phase in enumerate(PHASES):
                for phase_bin in range(10):
                    for offset in (0.1, 0.3):
                        phase_elapsed = phase_bin * 0.5 + offset
                        elapsed = phase_index * 5.0 + phase_elapsed
                        active = phase == "X" and condition == "press"
                        frames[frame_index] = _synthetic_hand_frame(
                            amplitudes[block_index] if active else 0
                        )
                        rows.append(
                            {
                                "row_type": "frame",
                                "frame_index": frame_index,
                                "block_index": block_index,
                                "condition": condition,
                                "phase": phase,
                                "phase_elapsed_s": phase_elapsed,
                                "global_elapsed_s": elapsed,
                                "artifact_write_ok": 1,
                            }
                        )
                        frame_index += 1
    return rows, frames


def test_salvage_analysis_is_explicitly_non_primary_and_keeps_twelve_curves():
    rows, frames = _complete_rows_and_frames()

    result = analyze_thermal_only(
        rows,
        frame_loader=lambda row: frames[row["frame_index"]],
    )

    assert result["analysis_role"] == (
        "salvage_descriptive_not_preregistered"
    )
    assert result["formal_primary_verdict"] == "INCOMPLETE_FOR_PRIMARY_TEST"
    assert result["selected_pair_count"] == 6
    assert len(result["curves"]) == 12
    assert result["interpolated_bin_count"] == 0
    assert result["thermal_only_exploratory"]["significant_clusters"][0][
        "p_corrected"
    ] == pytest.approx(0.03125)


def test_salvage_plots_are_labeled_and_contain_twelve_curves(tmp_path):
    rows, frames = _complete_rows_and_frames()
    result = analyze_thermal_only(
        rows,
        frame_loader=lambda row: frames[row["frame_index"]],
    )

    curves_path = tmp_path / "curves.png"
    rendered = plot_salvage_curves(result, curves_path)
    overlay_path = tmp_path / "roi.png"
    plot_feature_overlay(
        frames[0],
        thermal_only_feature(frames[0]),
        overlay_path,
    )

    assert curves_path.is_file()
    assert overlay_path.is_file()
    assert rendered["condition_curve_count"] == 12
    assert rendered["phase_boundaries_s"] == [5.0, 10.0, 15.0]
    assert rendered["analysis_role"] == (
        "salvage_descriptive_not_preregistered"
    )
