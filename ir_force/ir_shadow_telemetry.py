"""Fail-soft CSV telemetry for IR pressure shadow and apply runs.

OAK and thermal timestamps are host read-completion observations. They do not
represent camera exposure synchronization.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable


IR_SHADOW_SCHEMA_VERSION = "1"
PV_SHADOW_SCHEMA_VERSION = "7"
IR_SHADOW_FIELDS = (
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
PV_SHADOW_FIELDS = (
    "pv_sequence",
    "pv_source_observed_at_s",
    "pv_sent_at_s",
    "pv_received_at_s",
    "pv_frame_age_ms",
    "pressure_level",
    "pressure_n_levels",
    "pressure_mode",
    "object_id",
    "object_profile_sha256",
    "trial_index",
    "phase_index",
    "expected_level",
    "trial_phase",
    "commanded_gripper_pos",
    "motor_observed_at_s",
    "motor_sample_age_ms",
    "motor_sample_valid",
    "observed_gripper_pos",
    "observed_gripper_pos_valid",
    "present_current",
    "present_current_valid",
    "present_load",
    "present_load_valid",
    "present_temperature",
    "present_temperature_valid",
    "pv_adjustment_state",
    "pv_adjustment_event",
    "pv_adjustment_anchor_target",
    "pv_adjustment_release_since_s",
    "pv_adjustment_release_elapsed_s",
    "pv_adjustment_last_contact_at_s",
    "pv_adjustment_recontact_since_s",
    "relative_reference_pos",
    "relative_closure",
    "relative_mapping_status",
    "relative_track_hold_state",
    "relative_track_hold_residual",
    "relative_track_hold_output",
)


@dataclass(frozen=True)
class IRShadowTelemetrySample:
    control_observed_at_s: float
    state: str
    pinch: float
    roi_mode: str | None
    pressure: object | None
    baseline_ready: bool
    base_gripper_pos: float
    proposed_gripper_pos: float
    actual_gripper_pos: float
    fault_latched: bool
    fallback_used: bool
    fallback_reason: str | None
    pressure_status: str | None = None
    pressure_level: int | None = None
    pressure_n_levels: int | None = None
    pressure_mode: str | None = None
    object_id: str | None = None
    object_profile_sha256: str | None = None
    trial_index: int | None = None
    phase_index: int | None = None
    expected_level: int | None = None
    trial_phase: str | None = None
    relative_reference_pos: float | None = None
    relative_closure: float | None = None
    relative_mapping_status: str | None = None
    relative_track_hold_state: str | None = None
    relative_track_hold_residual: float | None = None
    relative_track_hold_output: float | None = None
    pv_adjustment_state: str | None = None
    pv_adjustment_event: str | None = None
    pv_adjustment_anchor_target: float | None = None
    pv_adjustment_release_since_s: float | None = None
    pv_adjustment_release_elapsed_s: float | None = None
    pv_adjustment_last_contact_at_s: float | None = None
    pv_adjustment_recontact_since_s: float | None = None


def _milliseconds(value: float | None) -> float | None:
    return None if value is None else float(value) * 1000.0


def _csv_bool(value: bool) -> str:
    return "true" if value else "false"


class IRShadowTelemetryLogger:
    """Write one finalized row per caller action without affecting control."""

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], float] = time.perf_counter,
        extra_fields: tuple[str, ...] = (),
    ):
        self.path = Path(path)
        self._clock = clock
        self._file = None
        self._writer = None
        self._tick = 0
        self._previous_control_observed_at_s: float | None = None
        self._warned = False
        self.enabled = False
        unknown = set(extra_fields) - set(PV_SHADOW_FIELDS)
        if unknown:
            raise ValueError(f"unsupported telemetry fields: {sorted(unknown)}")
        self.extra_fields = tuple(extra_fields)
        self.fieldnames = IR_SHADOW_FIELDS + self.extra_fields
        try:
            self._file = self.path.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=self.fieldnames,
                lineterminator="\n",
            )
            self._writer.writeheader()
            self._file.flush()
            self.enabled = True
        except Exception as exc:
            self._disable(exc)

    def _disable(self, exc: Exception) -> None:
        if not self._warned:
            print(f"[ir-sidecar] disabled after logging error: {exc}")
            self._warned = True
        self.enabled = False
        file_obj = self._file
        self._file = None
        self._writer = None
        if file_obj is not None:
            try:
                file_obj.close()
            except Exception:
                pass

    def _row(
        self,
        sample: IRShadowTelemetrySample,
        *,
        command_sent: bool,
        finalized_at_s: float,
        motor_telemetry=None,
    ) -> dict:
        pressure = sample.pressure
        roi = getattr(pressure, "roi", None)
        control_t = float(sample.control_observed_at_s)
        loop_period_s = (
            None
            if self._previous_control_observed_at_s is None
            else control_t - self._previous_control_observed_at_s
        )
        pressure_status = (
            getattr(pressure, "status", None)
            if pressure is not None
            else sample.pressure_status
        )
        row = {
            "schema_version": PV_SHADOW_SCHEMA_VERSION if self.extra_fields else IR_SHADOW_SCHEMA_VERSION,
            "tick": self._tick,
            "control_observed_at_s": control_t,
            "oak_observed_at_s": getattr(pressure, "oak_observed_at_s", None),
            "thermal_observed_at_s": getattr(pressure, "thermal_observed_at_s", None),
            "sensor_skew_ms": _milliseconds(getattr(pressure, "sensor_skew_s", None)),
            "oak_age_ms": _milliseconds(getattr(pressure, "oak_age_s", None)),
            "thermal_age_ms": _milliseconds(getattr(pressure, "thermal_age_s", None)),
            "loop_period_ms": _milliseconds(loop_period_s),
            "control_latency_ms": _milliseconds(finalized_at_s - control_t),
            "state": sample.state,
            "pinch": sample.pinch,
            "roi_mode": sample.roi_mode,
            "roi_x": getattr(roi, "x", None),
            "roi_y": getattr(roi, "y", None),
            "roi_width": getattr(roi, "width", None),
            "roi_height": getattr(roi, "height", None),
            "baseline_ready": _csv_bool(sample.baseline_ready),
            "pressure": getattr(pressure, "pressure_0_1", None),
            "quality": getattr(pressure, "quality", None),
            "pressure_available": _csv_bool(bool(getattr(pressure, "available", False))),
            "pressure_status": pressure_status,
            "base_gripper_pos": sample.base_gripper_pos,
            "proposed_gripper_pos": sample.proposed_gripper_pos,
            "actual_gripper_pos": sample.actual_gripper_pos,
            "command_sent": _csv_bool(command_sent),
            "fault_latched": _csv_bool(sample.fault_latched),
            "fallback_used": _csv_bool(sample.fallback_used),
            "fallback_reason": sample.fallback_reason,
        }
        if self.extra_fields:
            motor_observed_at_s = getattr(motor_telemetry, "observed_at_s", None)
            observed_gripper_pos = getattr(
                motor_telemetry, "observed_gripper_pos", None
            )
            present_current = getattr(motor_telemetry, "present_current", None)
            present_load = getattr(motor_telemetry, "present_load", None)
            present_temperature = getattr(
                motor_telemetry, "present_temperature", None
            )
            row.update(
                {
                    "pv_sequence": getattr(pressure, "pv_sequence", None),
                    "pv_source_observed_at_s": getattr(
                        pressure, "thermal_observed_at_s", None
                    ),
                    "pv_sent_at_s": getattr(pressure, "pv_sent_at_s", None),
                    "pv_received_at_s": getattr(pressure, "pv_received_at_s", None),
                    "pv_frame_age_ms": _milliseconds(
                        getattr(pressure, "thermal_age_s", None)
                    ),
                    "pressure_level": sample.pressure_level,
                    "pressure_n_levels": sample.pressure_n_levels,
                    "pressure_mode": sample.pressure_mode,
                    "object_id": sample.object_id,
                    "object_profile_sha256": sample.object_profile_sha256,
                    "trial_index": sample.trial_index,
                    "phase_index": sample.phase_index,
                    "expected_level": sample.expected_level,
                    "trial_phase": sample.trial_phase,
                    # Keep actual_gripper_pos for backward compatibility. It is
                    # the command selected by the controller, not a bus read.
                    "commanded_gripper_pos": sample.actual_gripper_pos,
                    "motor_observed_at_s": motor_observed_at_s,
                    "motor_sample_age_ms": _milliseconds(
                        None
                        if motor_observed_at_s is None
                        else finalized_at_s - float(motor_observed_at_s)
                    ),
                    "motor_sample_valid": _csv_bool(motor_telemetry is not None),
                    "observed_gripper_pos": observed_gripper_pos,
                    "observed_gripper_pos_valid": _csv_bool(
                        observed_gripper_pos is not None
                    ),
                    "present_current": present_current,
                    "present_current_valid": _csv_bool(present_current is not None),
                    "present_load": present_load,
                    "present_load_valid": _csv_bool(present_load is not None),
                    "present_temperature": present_temperature,
                    "present_temperature_valid": _csv_bool(
                        present_temperature is not None
                    ),
                    "pv_adjustment_state": sample.pv_adjustment_state,
                    "pv_adjustment_event": sample.pv_adjustment_event,
                    "pv_adjustment_anchor_target": sample.pv_adjustment_anchor_target,
                    "pv_adjustment_release_since_s": sample.pv_adjustment_release_since_s,
                    "pv_adjustment_release_elapsed_s": sample.pv_adjustment_release_elapsed_s,
                    "pv_adjustment_last_contact_at_s": sample.pv_adjustment_last_contact_at_s,
                    "pv_adjustment_recontact_since_s": sample.pv_adjustment_recontact_since_s,
                    "relative_reference_pos": sample.relative_reference_pos,
                    "relative_closure": sample.relative_closure,
                    "relative_mapping_status": sample.relative_mapping_status,
                    "relative_track_hold_state": sample.relative_track_hold_state,
                    "relative_track_hold_residual": sample.relative_track_hold_residual,
                    "relative_track_hold_output": sample.relative_track_hold_output,
                }
            )
        return row

    def finalize(
        self,
        sample: IRShadowTelemetrySample | None,
        *,
        command_sent: bool,
        motor_telemetry=None,
    ) -> None:
        if not self.enabled or sample is None:
            return
        try:
            finalized_at_s = self._clock()
            self._writer.writerow(
                self._row(
                    sample,
                    command_sent=command_sent,
                    finalized_at_s=finalized_at_s,
                    motor_telemetry=motor_telemetry,
                )
            )
            self._file.flush()
            self._previous_control_observed_at_s = float(sample.control_observed_at_s)
            self._tick += 1
        except Exception as exc:
            self._disable(exc)

    def close(self) -> None:
        file_obj = self._file
        self._file = None
        self._writer = None
        self.enabled = False
        if file_obj is None:
            return
        try:
            file_obj.close()
        except Exception as exc:
            self._disable(exc)
