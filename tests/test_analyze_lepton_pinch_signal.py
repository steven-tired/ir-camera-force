import json

import pytest

import analyze_lepton_pinch_signal as analyzer


PROTOCOL = {
    "groups": 6,
    "sequence": ["just_touch", "press_hard", "return_just_touch"],
    "target_valid_samples": 5,
    "phase_timeout_s": 10.0,
    "pinch_center_policy": "diagnostic_only",
    "phase_completion": "accepted_sample_quota",
    "advance_key": "space",
    "start_trigger": "first_software_gate_accepted",
}


def _metadata():
    return {
        "row_type": "metadata",
        "status": "ok",
        "schema_version": 4,
        "safety_mode": "robot_free_hand_shadow_only",
        "stage0_runtime_sha256": (
            "22d41109dcaefb29ad770fb5715c35dfd6c13c68195fbcb55e3b9d6fb4ef756b"
        ),
        "frozen_xml_sha256": (
            "2ca1ed48450dea16a5778cb5645dd4852d544490e4f47330dd938f743bc6f434"
        ),
        "pinch_signal_protocol": PROTOCOL,
    }


def _row(
    group_index,
    phase,
    sample_index,
    *,
    frame_median,
    thumb_residual,
    index_residual,
    thumb_patch_residual,
    index_patch_residual,
    pinch_2d,
    press_center_shift,
    return_center_shift,
):
    label = "press" if phase == "record_press_hard" else "contact"
    center_by_phase = {
        "record_just_touch": [50.0, 50.0],
        "record_press_hard": [50.0 + press_center_shift, 50.0],
        "record_return_touch": [50.0 + return_center_shift, 50.0],
    }
    center = center_by_phase[phase]
    shift = None if phase == "record_just_touch" else abs(center[0] - 50.0)
    return {
        "row_type": "attempt",
        "status": "software_gate_accepted",
        "thermal_frame_median_count": frame_median,
        "thermal_pinch_center_uv": center,
        "pinch_center_shift_px": shift,
        "thermal_patches_overlap": False,
        "pinch_signal": {
            "complete": False,
            "blocked": False,
            "group_index": group_index,
            "label": label,
            "phase": phase,
            "recording": True,
            "phase_elapsed_s": 0.1 + sample_index * 0.1,
            "phase_remaining_s": 9.9 - sample_index * 0.1,
            "phase_timeout_s": 10.0,
            "valid_samples": sample_index,
            "target_valid_samples": 5,
            "quota_accepted": True,
            "quota_reasons": [],
            "valid_samples_after": sample_index + 1,
            "pinch_center_shift_px": shift,
            "group_completed": (
                phase == "record_return_touch" and sample_index == 4
            ),
        },
        "fingertips": [
            {
                "label": "thumb_tip",
                "thermal_raw_count": frame_median + thumb_residual,
                "thermal_patch_3x3_mean_count": (
                    frame_median + thumb_patch_residual
                ),
                "thermal_patch_3x3_std_counts": 2.0,
                "thermal_uv": [49.0 + center[0] - 50.0, 50.0],
                "thermal_pixel": [49, 50],
                "depth_m": 0.45,
                "color_pixel": [600, 350],
                "depth_pixel": [590, 350],
            },
            {
                "label": "index_tip",
                "thermal_raw_count": frame_median + index_residual,
                "thermal_patch_3x3_mean_count": (
                    frame_median + index_patch_residual
                ),
                "thermal_patch_3x3_std_counts": 2.0,
                "thermal_uv": [51.0 + center[0] - 50.0, 50.0],
                "thermal_pixel": [51, 50],
                "depth_m": 0.45,
                "color_pixel": [620, 350],
                "depth_pixel": [610, 350],
            },
        ],
        "pinch_geometry": {
            "valid": True,
            "reason": "OK",
            "pinch_distance_2d_norm": pinch_2d,
            "pinch_depth_delta_m": 0.0,
        },
    }


def _document(
    *,
    thumb_press=-12.0,
    index_press=-12.0,
    thumb_patch_press=-11.0,
    index_patch_press=-11.0,
    return_delta=-2.0,
    thumb_return_delta=None,
    index_return_delta=None,
    geometry_press_delta=0.0,
    press_center_shift=0.5,
    return_center_shift=0.2,
    omit_groups=(),
):
    if thumb_return_delta is None:
        thumb_return_delta = return_delta
    if index_return_delta is None:
        index_return_delta = return_delta
    rows = [_metadata()]
    for group_index in range(6):
        if group_index in omit_groups:
            continue
        baseline_thumb = 100.0 + group_index * 0.2
        baseline_index = 102.0 + group_index * 0.2
        baseline_pinch = 0.20 + group_index * 0.001
        phase_values = {
            "record_just_touch": (
                baseline_thumb,
                baseline_index,
                baseline_thumb,
                baseline_index,
                baseline_pinch,
            ),
            "record_press_hard": (
                baseline_thumb + thumb_press,
                baseline_index + index_press,
                baseline_thumb + thumb_patch_press,
                baseline_index + index_patch_press,
                baseline_pinch + geometry_press_delta,
            ),
            "record_return_touch": (
                baseline_thumb + thumb_return_delta,
                baseline_index + index_return_delta,
                baseline_thumb + thumb_return_delta,
                baseline_index + index_return_delta,
                baseline_pinch,
            ),
        }
        for phase_index, (phase, values) in enumerate(phase_values.items()):
            for sample_index in range(5):
                # Large phase/group common-mode changes must cancel because
                # the primary feature subtracts each frame's median.
                frame_median = (
                    29000.0 + group_index * 100.0 + phase_index * 50.0
                )
                rows.append(
                    _row(
                        group_index,
                        phase,
                        sample_index,
                        frame_median=frame_median,
                        thumb_residual=values[0],
                        index_residual=values[1],
                        thumb_patch_residual=values[2],
                        index_patch_residual=values[3],
                        pinch_2d=values[4],
                        press_center_shift=press_center_shift,
                        return_center_shift=return_center_shift,
                    )
                )
    invalid_groups = len(omit_groups)
    rows.append(
        {
            "row_type": "summary",
            "status": "ok" if invalid_groups <= 1 else "blocked",
            "pinch_signal_started": True,
            "pinch_signal_protocol_completed": invalid_groups <= 1,
            "pinch_signal_acquisition_blocked": invalid_groups > 1,
            "pinch_signal_valid_groups": 6 - invalid_groups,
            "pinch_signal_invalid_groups": invalid_groups,
            "pinch_signal_phase_timeouts": invalid_groups,
        }
    )
    return rows


def test_centered_common_mode_corrected_signal_proceeds():
    result = analyzer.analyze_document(_document())

    assert result["schema_version"] == 4
    assert result["verdict"] == "PROCEED_TO_STAGE1F_SHADOW"
    assert result["valid_groups"] == 6
    assert result["ir"]["paired_eba"] == pytest.approx(1.0)
    assert result["geometry"]["paired_eba"] == pytest.approx(0.5)
    assert result["ir_gain_over_geometry"] == pytest.approx(0.5)
    assert result["fingertips"]["thumb_tip"]["median_effect"] == pytest.approx(
        -11.0
    )
    assert result["fingertips"]["index_tip"]["median_effect"] == pytest.approx(
        -11.0
    )
    assert result["press_recovery_ratio"] == pytest.approx(2.0 / 11.0)
    assert result["press_recovery_ratio_by_tip"] == {
        "thumb_tip": pytest.approx(2.0 / 11.0),
        "index_tip": pytest.approx(2.0 / 11.0),
    }
    thumb = result["groups"][0]["fingertips"]["thumb_tip"]
    assert thumb["phase_medians"]["record_press_hard"][
        "corrected_exact_count"
    ] == pytest.approx(88.0)
    assert thumb["press_thermal_pixel_shift_px"] == pytest.approx(0.0)
    assert thumb["return_depth_shift_m"] == pytest.approx(0.0)
    assert thumb["return_color_shift_px"] == pytest.approx(0.0)
    assert thumb["return_depth_pixel_shift_px"] == pytest.approx(0.0)


def test_large_pinch_center_motion_is_diagnostic_and_does_not_block():
    result = analyzer.analyze_document(
        _document(press_center_shift=6.0, return_center_shift=4.0)
    )

    assert result["verdict"] == "PROCEED_TO_STAGE1F_SHADOW"
    assert result["valid_groups"] == 6
    assert result["groups"][0]["press_center_shift_px"] == pytest.approx(6.0)
    assert result["groups"][0]["return_center_shift_px"] == pytest.approx(4.0)


def test_opposite_thumb_and_index_effects_stop_before_stage1f():
    result = analyzer.analyze_document(
        _document(thumb_press=-12.0, index_press=12.0)
    )

    assert result["verdict"] == "STOP_BEFORE_STAGE1F"
    assert "per_tip_press_direction_inconsistent" in result["reasons"]


def test_exact_and_patch_sign_disagreement_stops_before_stage1f():
    result = analyzer.analyze_document(
        _document(thumb_patch_press=12.0, index_patch_press=12.0)
    )

    assert result["verdict"] == "STOP_BEFORE_STAGE1F"
    assert "sampling_method_sign_disagreement" in result["reasons"]


def test_failed_return_to_touch_stops_before_stage1f():
    result = analyzer.analyze_document(_document(return_delta=-10.0))

    assert result["verdict"] == "STOP_BEFORE_STAGE1F"
    assert "press_recovery_ratio_above_threshold" in result["reasons"]


def test_opposite_return_errors_cannot_cancel_before_recovery_gate():
    result = analyzer.analyze_document(
        _document(thumb_return_delta=-10.0, index_return_delta=10.0)
    )

    assert result["verdict"] == "STOP_BEFORE_STAGE1F"
    assert result["press_recovery_ratio"] > 0.5
    assert "press_recovery_ratio_above_threshold" in result["reasons"]


def test_one_invalid_group_is_allowed_but_two_block_acquisition():
    allowed = analyzer.analyze_document(_document(omit_groups=(0,)))
    blocked = analyzer.analyze_document(_document(omit_groups=(0, 1)))

    assert allowed["valid_groups"] == 5
    assert allowed["verdict"] == "PROCEED_TO_STAGE1F_SHADOW"
    assert blocked["valid_groups"] == 4
    assert blocked["verdict"] == "BLOCKED_ACQUISITION"
    assert "insufficient_valid_groups" in blocked["reasons"]


def test_old_protocol_or_frozen_hash_mismatch_blocks_acquisition():
    old = _document()
    old[0]["pinch_signal_protocol"] = {
        "groups": 6,
        "record_duration_s": 1.0,
    }
    attempt04 = _document()
    attempt04[0]["schema_version"] = 3
    attempt04[0]["pinch_signal_protocol"] = {
        **PROTOCOL,
        "max_center_shift_px": 1.0,
    }
    del attempt04[0]["pinch_signal_protocol"]["pinch_center_policy"]
    wrong_hash = _document()
    wrong_hash[0]["frozen_xml_sha256"] = "wrong"

    for rows in (old, attempt04, wrong_hash):
        result = analyzer.analyze_document(rows)
        assert result["verdict"] == "BLOCKED_ACQUISITION"
        assert "metadata_mismatch" in result["reasons"]


def test_cli_never_overwrites_existing_output(tmp_path):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in _document())
    )
    output_path = tmp_path / "summary.json"
    output_path.write_text("keep")

    with pytest.raises(FileExistsError):
        analyzer.main(
            ["--input", str(input_path), "--output", str(output_path)]
        )

    assert output_path.read_text() == "keep"
