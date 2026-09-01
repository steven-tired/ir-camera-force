import csv
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from ir_force.ir_hand_roi import PressureROI
from ir_force.ir_pressure import PressureReading


EXPECTED_HEADER = (
    "schema_version",
    "tick",
    "control_observed_at_s",
    "oak_observed_at_s",
    "thermal_observed_at_s",
    "sensor_skew_ms",
    "oak_age_ms",
    "thermal_age_ms",
    "loop_period_ms",
    "control_latency_ms",
    "state",
    "pinch",
    "roi_mode",
    "roi_x",
    "roi_y",
    "roi_width",
    "roi_height",
    "baseline_ready",
    "pressure",
    "quality",
    "pressure_available",
    "pressure_status",
    "base_gripper_pos",
    "proposed_gripper_pos",
    "actual_gripper_pos",
    "command_sent",
    "fault_latched",
    "fallback_used",
    "fallback_reason",
)


def _telemetry_module():
    return importlib.import_module("ir_force.ir_shadow_telemetry")


def _sample(module, *, control_observed_at_s=10.0, pressure=True):
    reading = None
    if pressure:
        reading = PressureReading(
            pressure_0_1=0.5,
            active=True,
            quality=0.75,
            available=True,
            status="active",
            roi=PressureROI(x=2, y=3, width=4, height=5),
            oak_observed_at_s=9.94,
            thermal_observed_at_s=9.96,
            sensor_skew_s=0.02,
            oak_age_s=0.06,
            thermal_age_s=0.04,
            pv_sequence=12,
            pv_sent_at_s=9.97,
            pv_received_at_s=9.98,
        )
    return module.IRShadowTelemetrySample(
        control_observed_at_s=control_observed_at_s,
        state="MOVING",
        pinch=0.04,
        roi_mode="tips" if pressure else None,
        pressure=reading,
        baseline_ready=True,
        base_gripper_pos=60.0,
        proposed_gripper_pos=51.0,
        actual_gripper_pos=42.0,
        fault_latched=False,
        fallback_used=False,
        fallback_reason=None,
    )


def test_sidecar_has_exact_versioned_header_and_nullable_diagnostics(tmp_path: Path):
    module = _telemetry_module()
    path = tmp_path / "shadow.csv"
    clock = iter((10.005, 10.025)).__next__
    logger = module.IRShadowTelemetryLogger(path, clock=clock)

    logger.finalize(_sample(module), command_sent=False)
    logger.finalize(
        _sample(module, control_observed_at_s=10.02, pressure=False),
        command_sent=True,
    )
    logger.close()

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert tuple(reader.fieldnames) == EXPECTED_HEADER
    assert rows[0]["schema_version"] == "1"
    assert [row["tick"] for row in rows] == ["0", "1"]
    assert float(rows[0]["sensor_skew_ms"]) == pytest.approx(20.0)
    assert float(rows[0]["oak_age_ms"]) == pytest.approx(60.0)
    assert float(rows[0]["thermal_age_ms"]) == pytest.approx(40.0)
    assert rows[0]["loop_period_ms"] == ""
    assert float(rows[1]["loop_period_ms"]) == pytest.approx(20.0)
    assert float(rows[0]["control_latency_ms"]) == pytest.approx(5.0)
    assert rows[0]["roi_mode"] == "tips"
    assert [rows[0][name] for name in ("roi_x", "roi_y", "roi_width", "roi_height")] == [
        "2",
        "3",
        "4",
        "5",
    ]
    assert rows[0]["command_sent"] == "false"
    assert rows[1]["command_sent"] == "true"
    assert rows[1]["oak_observed_at_s"] == ""
    assert rows[1]["thermal_observed_at_s"] == ""
    assert rows[1]["sensor_skew_ms"] == ""
    assert rows[1]["roi_x"] == ""
    assert rows[1]["pressure"] == ""


def test_sidecar_write_failure_prints_once_disables_and_never_raises(tmp_path: Path, capsys):
    module = _telemetry_module()
    logger = module.IRShadowTelemetryLogger(tmp_path / "shadow.csv", clock=lambda: 10.01)

    class BrokenWriter:
        def writerow(self, _row):
            raise OSError("disk unavailable")

    logger._writer = BrokenWriter()
    logger.finalize(_sample(module), command_sent=True)
    logger.finalize(_sample(module), command_sent=True)
    logger.close()

    assert not logger.enabled
    output = capsys.readouterr().out
    assert output.count("[ir-sidecar] disabled") == 1
    assert "disk unavailable" in output


def test_pv_sidecar_distinguishes_command_from_rate_limited_motor_readback(tmp_path: Path):
    module = _telemetry_module()
    path = tmp_path / "pv.csv"
    logger = module.IRShadowTelemetryLogger(
        path,
        clock=lambda: 10.01,
        extra_fields=module.PV_SHADOW_FIELDS,
    )
    sample = _sample(module)
    sample = module.IRShadowTelemetrySample(
        **{
            **sample.__dict__,
            "relative_reference_pos": 27.0,
            "relative_closure": 1.5,
            "relative_mapping_status": "active",
            "relative_track_hold_state": "HOLD",
            "relative_track_hold_residual": 0.04,
            "relative_track_hold_output": 0.75,
            "pv_adjustment_state": "temporary_hold",
            "pv_adjustment_event": "contact_lost",
            "pv_adjustment_anchor_target": 27.0,
            "pv_adjustment_release_since_s": 9.95,
            "pv_adjustment_release_elapsed_s": 0.05,
            "pv_adjustment_last_contact_at_s": 9.94,
            "pv_adjustment_recontact_since_s": None,
        }
    )
    motor = SimpleNamespace(
        observed_at_s=9.99,
        observed_gripper_pos=27.4,
        present_current=18,
        present_load=31,
        present_temperature=34,
    )

    logger.finalize(sample, command_sent=True, motor_telemetry=motor)
    logger.close()

    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    assert row["schema_version"] == "7"
    assert row["pv_sequence"] == "12"
    assert row["pv_source_observed_at_s"] == "9.96"
    assert row["pv_sent_at_s"] == "9.97"
    assert row["pv_received_at_s"] == "9.98"
    assert row["pv_frame_age_ms"] == "40.0"
    assert row["actual_gripper_pos"] == "42.0"
    assert row["commanded_gripper_pos"] == "42.0"
    assert float(row["motor_sample_age_ms"]) == pytest.approx(20.0)
    assert row["motor_sample_valid"] == "true"
    assert row["observed_gripper_pos"] == "27.4"
    assert row["observed_gripper_pos_valid"] == "true"
    assert row["present_current"] == "18"
    assert row["present_current_valid"] == "true"
    assert row["present_load"] == "31"
    assert row["present_load_valid"] == "true"
    assert row["present_temperature"] == "34"
    assert row["present_temperature_valid"] == "true"
    assert row["pv_adjustment_state"] == "temporary_hold"
    assert row["pv_adjustment_event"] == "contact_lost"
    assert row["pv_adjustment_anchor_target"] == "27.0"
    assert row["pv_adjustment_release_since_s"] == "9.95"
    assert row["pv_adjustment_release_elapsed_s"] == "0.05"
    assert row["pv_adjustment_last_contact_at_s"] == "9.94"
    assert row["pv_adjustment_recontact_since_s"] == ""
    assert row["relative_reference_pos"] == "27.0"
    assert row["relative_closure"] == "1.5"
    assert row["relative_mapping_status"] == "active"
    assert row["relative_track_hold_state"] == "HOLD"
    assert row["relative_track_hold_residual"] == "0.04"
    assert row["relative_track_hold_output"] == "0.75"


def test_pv_sidecar_marks_missing_motor_sample_without_inventing_values(tmp_path: Path):
    module = _telemetry_module()
    path = tmp_path / "pv-missing.csv"
    logger = module.IRShadowTelemetryLogger(
        path,
        clock=lambda: 10.01,
        extra_fields=module.PV_SHADOW_FIELDS,
    )

    logger.finalize(_sample(module), command_sent=True, motor_telemetry=None)
    logger.close()

    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    assert row["motor_observed_at_s"] == ""
    assert row["motor_sample_age_ms"] == ""
    assert row["motor_sample_valid"] == "false"
    assert row["observed_gripper_pos_valid"] == "false"
    assert row["present_current_valid"] == "false"
    assert row["present_load_valid"] == "false"
    assert row["present_temperature_valid"] == "false"
