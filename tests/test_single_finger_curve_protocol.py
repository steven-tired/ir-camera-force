import pytest

from ir_force.single_finger_curve_protocol import (
    PHASES,
    PRIMARY_BLOCKS,
    RESERVE_BLOCKS,
    global_elapsed,
    phase_at,
    phase_elapsed,
    scheduled_trial_specs,
    trial_integrity,
)


def test_frozen_schedule_is_balanced_and_has_two_reserves():
    assert PRIMARY_BLOCKS == (
        ("null", "press"),
        ("press", "null"),
        ("press", "null"),
        ("null", "press"),
        ("null", "press"),
        ("press", "null"),
    )
    assert RESERVE_BLOCKS == (("null", "press"), ("press", "null"))

    specs = scheduled_trial_specs()

    assert len(specs) == 16
    assert [
        (spec.block_index, spec.condition, spec.order_in_block, spec.reserve)
        for spec in specs[:4]
    ] == [
        (0, "null", 0, False),
        (0, "press", 1, False),
        (1, "press", 0, False),
        (1, "null", 1, False),
    ]
    assert all(spec.reserve for spec in specs[-4:])


def test_four_phases_are_exactly_five_seconds():
    assert PHASES == ("A1", "X", "A2", "A3")
    assert phase_at(0.0) == "A1"
    assert phase_at(4.999) == "A1"
    assert phase_at(5.0) == "X"
    assert phase_at(10.0) == "A2"
    assert phase_at(15.0) == "A3"
    assert phase_at(19.999) == "A3"
    assert phase_at(20.0) is None
    assert phase_elapsed(16.25) == pytest.approx(1.25)
    assert global_elapsed("A3", 1.25) == pytest.approx(16.25)


@pytest.mark.parametrize("elapsed", [-0.1, float("nan"), float("inf")])
def test_phase_time_rejects_negative_or_nonfinite_values(elapsed):
    with pytest.raises(ValueError, match="finite and non-negative"):
        phase_at(elapsed)


def _complete_trial_rows():
    rows = []
    for phase_index, phase in enumerate(PHASES):
        for bin_index in range(10):
            for offset in (0.10, 0.30):
                elapsed = bin_index * 0.5 + offset
                rows.append(
                    {
                        "row_type": "frame",
                        "phase": phase,
                        "phase_elapsed_s": elapsed,
                        "thermal_host_s": (
                            phase_index * 5.0 + elapsed
                        ),
                        "tracking_valid": True,
                        "ffc_in_progress": False,
                        "artifact_write_ok": True,
                        "primary_signal_count": 123456.0,
                    }
                )
    return rows


def test_trial_integrity_uses_tracking_timing_ffc_and_writes_only():
    rows = _complete_trial_rows()

    result = trial_integrity(rows)

    assert result["valid"] is True
    assert result["reasons"] == []
    assert result["valid_frames_by_phase_bin"] == {
        phase: [2] * 10 for phase in PHASES
    }

    for row in rows:
        row["primary_signal_count"] = object()
    assert trial_integrity(rows)["valid"] is True


def test_trial_integrity_accepts_legacy_json_integer_flags():
    rows = _complete_trial_rows()
    for row in rows:
        row["tracking_valid"] = 1
        row["ffc_in_progress"] = 0
        row["artifact_write_ok"] = 1

    assert trial_integrity(rows)["valid"] is True


def test_trial_integrity_reports_closed_technical_reasons():
    rows = _complete_trial_rows()
    missing = rows.pop(2 * 10 + 2 * 4)
    assert missing["phase"] == "X"
    assert int(missing["phase_elapsed_s"] // 0.5) == 4
    rows[0]["ffc_in_progress"] = True
    rows[1]["artifact_write_ok"] = False
    rows[-1]["thermal_host_s"] = rows[-2]["thermal_host_s"] + 1.0

    result = trial_integrity(rows)

    assert result["valid"] is False
    assert result["reasons"] == [
        "ffc_active",
        "required_artifact_write_failed",
        "thermal_timestamp_gap",
        "insufficient_tracking:X:4",
    ]


def test_trial_integrity_rejects_unknown_phase_and_out_of_range_bin():
    rows = _complete_trial_rows()
    rows[0]["phase"] = "UNKNOWN"
    rows[1]["phase_elapsed_s"] = 5.0

    result = trial_integrity(rows)

    assert result["valid"] is False
    assert "invalid_phase" in result["reasons"]
    assert "invalid_phase_elapsed" in result["reasons"]
