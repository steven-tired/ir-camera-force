from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


class LedgerSemanticError(ValueError):
    pass


def _session_schedule(schedule: dict[str, Any], session_index: int) -> dict[str, Any]:
    for session in schedule.get("sessions", []):
        if session.get("session_index") == session_index:
            return session
    raise LedgerSemanticError(f"schedule has no session_index {session_index}")


def validate_ledger_semantics(
    ledger: dict[str, Any], schedule: dict[str, Any]
) -> None:
    session = _session_schedule(schedule, ledger["session_index"])
    primary = session.get("primary")
    if not isinstance(primary, str) or len(primary) != 30:
        raise LedgerSemanticError("session schedule must contain exactly 30 primary labels")
    if set(primary) != {"L", "T"} or primary.count("L") != 15 or primary.count("T") != 15:
        raise LedgerSemanticError("session schedule must contain exactly 15 L/T labels each")
    computed_schedule_sha256 = hashlib.sha256(primary.encode()).hexdigest()
    if session.get("sha256") != computed_schedule_sha256:
        raise LedgerSemanticError("session schedule sha256 does not match primary labels")
    if ledger["schedule_sha256"] != computed_schedule_sha256:
        raise LedgerSemanticError("schedule_sha256 does not match the selected session")

    seen_keys: set[tuple[int, int]] = set()
    attempts_by_slot: dict[int, int] = {}
    aborts_by_label = {"loose": 0, "tight": 0}
    current_slot = 1

    for trial in ledger["trials"]:
        slot = trial["scheduled_slot"]
        attempt = trial["attempt_index"]
        key = (slot, attempt)
        if key in seen_keys:
            raise LedgerSemanticError(
                "duplicate scheduled_slot/attempt_index: "
                f"slot={slot}, attempt={attempt}"
            )
        seen_keys.add(key)

        if slot != current_slot:
            raise LedgerSemanticError(
                f"slot {slot} is out of order; expected slot {current_slot}"
            )
        expected_attempt = attempts_by_slot.get(slot, 0) + 1
        if attempt != expected_attempt:
            raise LedgerSemanticError(
                f"slot {slot} attempt_index must be {expected_attempt}, got {attempt}"
            )
        attempts_by_slot[slot] = attempt

        expected_label = "loose" if primary[slot - 1] == "L" else "tight"
        if trial["label"] != expected_label:
            raise LedgerSemanticError(
                f"slot {slot} label must be {expected_label}, got {trial['label']}"
            )

        if trial["outcome"] == "completed" and (
            trial["ffc"]["pretrial_complete"] is not True
            or trial["ffc"]["since_last_ffc_reset"] is not True
        ):
            raise LedgerSemanticError(
                f"slot {slot} completed without successful pretrial FFC/reset"
            )
        if (
            trial["outcome"] == "completed"
            and trial["ffc"]["analysis_window_active"] is True
            and trial["thermal_validity"]["valid"] is True
        ):
            raise LedgerSemanticError(
                f"slot {slot} marks thermal valid during active FFC"
            )
        if trial["outcome"] == "completed":
            artifacts = trial["artifacts"]
            if trial["realsense_validity"]["valid"] is True and (
                artifacts["rgb"] is None or artifacts["depth"] is None
            ):
                raise LedgerSemanticError(
                    f"slot {slot} valid RealSense modality requires RGB/depth artifacts"
                )
            if (
                trial["thermal_validity"]["valid"] is True
                and artifacts["thermal"] is None
            ):
                raise LedgerSemanticError(
                    f"slot {slot} valid thermal modality requires a thermal artifact"
                )

        if trial["outcome"] == "technical_abort":
            aborts_by_label[expected_label] += 1
            if aborts_by_label[expected_label] > 5:
                raise LedgerSemanticError(
                    f"{expected_label} technical-abort reserve budget exceeds five"
                )
        else:
            current_slot += 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a sealed IR hand-pinch session ledger."
    )
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--schedule", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
        schema = json.loads(args.schema.read_text(encoding="utf-8"))
        schedule = json.loads(args.schedule.read_text(encoding="utf-8"))
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        errors = sorted(validator.iter_errors(ledger), key=lambda item: list(item.path))
        if errors:
            raise LedgerSemanticError(errors[0].message)
        validate_ledger_semantics(ledger, schedule)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, LedgerSemanticError) as error:
        print(f"INVALID: {error}")
        return 1
    print("VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
