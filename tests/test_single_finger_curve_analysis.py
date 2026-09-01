import csv
from pathlib import Path

import numpy as np
import pytest

from ir_force.single_finger_curve_analysis import (
    analyze_rows,
    bin_selected_pairs,
    exact_cluster_test,
    plot_all_curves,
    write_per_frame_csv,
)
from ir_force.single_finger_curve_protocol import (
    PHASES,
)


def _complete_rows(*, press_effect_by_bin=None, press_u_effect_by_bin=None):
    press_effect_by_bin = press_effect_by_bin or {}
    press_u_effect_by_bin = press_u_effect_by_bin or {}
    rows = []
    for block_index in range(6):
        for condition_index, condition in enumerate(("null", "press")):
            baseline = 100.0 + block_index * 10.0 + condition_index * 3.0
            thermal_clock = block_index * 100.0 + condition_index * 30.0
            for phase_index, phase in enumerate(PHASES):
                for phase_bin in range(10):
                    for offset in (0.1, 0.3):
                        phase_elapsed = phase_bin * 0.5 + offset
                        global_elapsed = phase_index * 5.0 + phase_elapsed
                        analysis_bin = int((global_elapsed - 5.0) // 0.5)
                        effect = (
                            press_effect_by_bin.get(analysis_bin, 0.0)
                            if condition == "press" and analysis_bin >= 0
                            else 0.0
                        )
                        u_effect = (
                            press_u_effect_by_bin.get(analysis_bin, 0.0)
                            if condition == "press" and analysis_bin >= 0
                            else 0.0
                        )
                        rows.append(
                            {
                                "row_type": "frame",
                                "block_index": block_index,
                                "condition": condition,
                                "phase": phase,
                                "phase_elapsed_s": phase_elapsed,
                                "global_elapsed_s": global_elapsed,
                                "thermal_host_s": thermal_clock + global_elapsed,
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


def test_binning_subtracts_only_a1_seconds_three_to_five_baseline():
    rows = _complete_rows()
    target = [
        row
        for row in rows
        if row["block_index"] == 0 and row["condition"] == "null"
    ]
    for row in target:
        if row["phase"] == "A1" and row["phase_elapsed_s"] >= 3.0:
            row["primary_signal_count"] = (
                100.0 if row["phase_elapsed_s"] < 4.0 else 102.0
            )
        elif row["phase"] == "X" and row["phase_elapsed_s"] < 0.5:
            row["primary_signal_count"] = (
                110.0 if row["phase_elapsed_s"] < 0.2 else 112.0
            )

    result = bin_selected_pairs(rows)
    null_curve = next(
        curve
        for curve in result["curves"]
        if curve["block_index"] == 0 and curve["condition"] == "null"
    )

    assert result["complete"] is True
    assert result["selected_blocks"] == [0, 1, 2, 3, 4, 5]
    assert null_curve["baseline_count"] == pytest.approx(101.0)
    native_index = null_curve["time_s"].index(5.1)
    assert null_curve["normalized_count"][native_index] == pytest.approx(9.0)
    assert result["binned"]["primary_signal_count"]["null"][0][0] == pytest.approx(
        10.0
    )
    assert np.asarray(
        result["binned"]["primary_signal_count"]["null"]
    ).shape == (6, 30)


def test_binning_accepts_legacy_json_integer_flags():
    rows = _complete_rows()
    for row in rows:
        row["tracking_valid"] = 1
        row["ffc_in_progress"] = 0
        row["artifact_write_ok"] = 1

    result = bin_selected_pairs(rows)

    assert result["complete"] is True
    assert result["selected_pair_count"] == 6


def _synthetic_differences(active_bins, amplitudes):
    differences = np.zeros((6, 30), dtype=float)
    for pair, amplitude in enumerate(amplitudes):
        differences[pair, list(active_bins)] = amplitude
    return differences


def test_exact_cluster_test_finds_sustained_paired_effect():
    differences = _synthetic_differences(
        range(8, 16),
        (2.0, 2.1, 1.9, 2.2, 1.8, 2.05),
    )

    result = exact_cluster_test(differences)

    assert result["permutations"] == 64
    assert result["clusters"][0]["start_bin"] == 8
    assert result["clusters"][0]["end_bin"] == 15
    assert result["clusters"][0]["p_corrected"] == pytest.approx(0.03125)


def test_opposite_sign_neighbors_form_separate_clusters():
    differences = np.zeros((6, 30), dtype=float)
    differences[:, 5] = (2.0, 2.1, 1.9, 2.2, 1.8, 2.05)
    differences[:, 6] = (-2.0, -2.1, -1.9, -2.2, -1.8, -2.05)

    clusters = exact_cluster_test(differences)["clusters"]

    assert [cluster["sign"] for cluster in clusters] == [1, -1]


def test_plot_contains_exactly_twelve_native_curves_and_annotations(tmp_path):
    paired = bin_selected_pairs(_complete_rows())
    clusters = [
        {
            "start_bin": 2,
            "end_bin": 4,
            "sign": 1,
            "mass": 12.0,
            "p_corrected": 0.03125,
        }
    ]
    output = tmp_path / "curves.png"

    rendered = plot_all_curves(paired, clusters, output)

    assert output.is_file()
    assert rendered["condition_curve_count"] == 12
    assert rendered["axes_count"] == 1
    assert rendered["phase_boundaries_s"] == [5.0, 10.0, 15.0]
    assert rendered["significant_cluster_spans_s"] == [(6.0, 7.5)]


def test_analysis_marks_overlapping_geometry_effect_as_confound():
    thermal = {
        bin_index: amplitude
        for bin_index in range(8, 16)
        for amplitude in [2.0]
    }
    geometry = {
        bin_index: amplitude
        for bin_index in range(12, 18)
        for amplitude in [1.0]
    }
    rows = _complete_rows(
        press_effect_by_bin=thermal,
        press_u_effect_by_bin=geometry,
    )
    for row in rows:
        if row["condition"] == "press":
            row["primary_signal_count"] += row["block_index"] * 0.03
            row["distal_thermal_u_px"] += row["block_index"] * 0.01

    result = analyze_rows(rows)

    assert result["verdict"] == "GEOMETRY_CONFOUNDED"
    assert result["selected_pair_count"] == 6
    assert result["thermal"]["significant_clusters"]
    assert result["geometry"]["uv_displacement_px"]["significant_clusters"]


def test_incomplete_capture_never_runs_primary_inference():
    rows = [
        row
        for row in _complete_rows()
        if row["block_index"] != 5
    ]

    result = analyze_rows(rows)

    assert result["verdict"] == "INCOMPLETE_FOR_PRIMARY_TEST"
    assert result["selected_pair_count"] == 5
    assert "thermal" not in result


def test_write_per_frame_csv_keeps_frame_rows_and_serializes_nested_values(
    tmp_path,
):
    rows = [
        {
            "row_type": "metadata",
            "ignored": True,
        },
        {
            "row_type": "frame",
            "block_index": 0,
            "condition": "null",
            "nested": {"a": 1},
        },
    ]
    output = tmp_path / "frames.csv"

    write_per_frame_csv(rows, output)

    with output.open(newline="") as stream:
        saved = list(csv.DictReader(stream))
    assert len(saved) == 1
    assert saved[0]["condition"] == "null"
    assert saved[0]["nested"] == '{"a": 1}'
