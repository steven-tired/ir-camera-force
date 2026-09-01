from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from validate_ir_hand_pinch_session_ledger import (
    LedgerSemanticError,
    validate_ledger_semantics,
)


ROOT = Path(__file__).resolve().parents[1]
STUDY_SCHEMA_PATH = ROOT / "ir_hand_pinch_study_contract.schema.json"
LEDGER_SCHEMA_PATH = ROOT / "ir_hand_pinch_session_ledger.schema.json"
SCHEDULE_PATH = ROOT / "ir_hand_pinch_trial_schedule.json"
PREREG_PATH = ROOT / "IR_HAND_PINCH_PREREGISTRATION.md"
CALIBRATION_RUNBOOK_PATH = ROOT / "REALSENSE_LEPTON_CALIBRATION.md"
FIXTURES = ROOT / "tests" / "fixtures"


def test_study_schema_freezes_preregistered_constants() -> None:
    schema = json.loads(STUDY_SCHEMA_PATH.read_text(encoding="utf-8"))
    properties = schema["properties"]

    assert properties["schema_version"]["const"] == "ir-hand-pinch-study-contract/v1"
    assert properties["participant_pseudonym"]["minLength"] == 1
    assert properties["session_indices"]["const"] == [1, 2, 3, 4, 5, 6]
    assert properties["distinct_calendar_days"]["const"] is True
    assert properties["labels"]["const"] == ["loose", "tight"]
    assert properties["completed_trials_per_class"]["const"] == 15
    assert properties["technical_abort_reserves_per_class"]["const"] == 5

    acquisition = properties["acquisition"]["properties"]
    assert acquisition["realsense_color_profile"]["const"] == "640x480 BGR8 @ 30 Hz"
    assert acquisition["realsense_depth_profile"]["const"] == "640x480 Z16 @ 30 Hz"
    assert acquisition["expected_aligned_realsense_fps"]["const"] == 30
    assert acquisition["minimum_realsense_frames"]["const"] == 30
    assert acquisition["realsense_minimum_yield_fraction"]["const"] == 0.5
    assert acquisition["analysis_window_seconds"]["const"] == 2.0
    assert acquisition["minimum_thermal_frames"]["const"] == 9
    assert acquisition["manual_ffc"]["const"] is True
    assert acquisition["footer_telemetry"]["const"] is True


def test_realized_schedules_are_exact_and_balanced() -> None:
    schedule = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
    expected = [
        "LLTLLTLTTTLTLTTTLTLLTLLLTTLLTT",
        "LTTLTTLTLTTLTLLTLTTTLLLTLTLLLT",
        "LLLLLLTTTLTTLTTTLLLTTTTTLTLLLT",
        "LLLTTTLLTLLLLLTTTLTTLTTLTTTTLL",
        "LLTTLLLLTLLLTLLTTTTTTTTLLLTLTT",
        "TLTTLLLLLTLLTTTLTLLLLLTTTTLTTT",
    ]
    hashes = [
        "85b9d32d60dedc71d14becd3db2f13db2d2bc70f9884079ad083a8cbab3ebf57",
        "69fb74a10652f36e3d971d5c9d4ec3d59dc81d5513ff4f700a7f52d2f7a9f316",
        "4b08c9e4a0732658f914fea1f8e0dfeca43a0d70b756fe0a49b846f29751c0b4",
        "3f70ed366db25b32a216413551220f06d98bde77432d4c2a21b247fbaeb4eade",
        "0d33866836ae227645d15d580b8802e2e8a8dfdec5a626a5fbdfcc796cbb60b9",
        "67559ed2f9d67d7b2cf25878dfaf39469cf09d549719d97a6367c2d101414faa",
    ]
    assert [item["primary"] for item in schedule["sessions"]] == expected
    assert [item["sha256"] for item in schedule["sessions"]] == hashes
    assert all(s.count("L") == s.count("T") == 15 for s in expected)
    assert [hashlib.sha256(s.encode()).hexdigest() for s in expected] == hashes
    assert schedule["reserve_tokens_per_class"] == {"L": 5, "T": 5}


def test_ledger_requires_trial_audit_fields() -> None:
    schema = json.loads(LEDGER_SCHEMA_PATH.read_text(encoding="utf-8"))
    trial = schema["properties"]["trials"]["items"]
    required = set(trial["required"])
    assert {"scheduled_slot", "attempt_index", "label", "outcome",
            "start_monotonic_ns", "end_monotonic_ns", "ffc",
            "realsense_validity", "thermal_validity", "artifacts"} <= required


def test_schemas_are_valid_draft_202012() -> None:
    for path in (STUDY_SCHEMA_PATH, LEDGER_SCHEMA_PATH):
        Draft202012Validator.check_schema(
            json.loads(path.read_text(encoding="utf-8"))
        )


def _validator_and_instance(schema_path: Path, fixture_name: str):
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    instance = json.loads((FIXTURES / fixture_name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema, format_checker=FormatChecker()), instance


def test_valid_contract_and_ledger_instances_validate() -> None:
    cases = [
        (STUDY_SCHEMA_PATH, "ir_hand_pinch_study_contract.valid.json"),
        (LEDGER_SCHEMA_PATH, "ir_hand_pinch_session_ledger.valid.json"),
    ]
    for schema_path, fixture_name in cases:
        validator, instance = _validator_and_instance(schema_path, fixture_name)
        assert list(validator.iter_errors(instance)) == []


def test_critical_invalid_instances_are_rejected() -> None:
    study_validator, study = _validator_and_instance(
        STUDY_SCHEMA_PATH, "ir_hand_pinch_study_contract.valid.json"
    )
    bad_fps = deepcopy(study)
    bad_fps["acquisition"]["expected_aligned_realsense_fps"] = 15
    bad_profile = deepcopy(study)
    bad_profile["acquisition"]["realsense_color_profile"] = "640x480 BGR8 @ 15 Hz"
    assert list(study_validator.iter_errors(bad_fps))
    assert list(study_validator.iter_errors(bad_profile))

    ledger_validator, ledger = _validator_and_instance(
        LEDGER_SCHEMA_PATH, "ir_hand_pinch_session_ledger.valid.json"
    )
    retry_three = deepcopy(ledger)
    retry_three["session_attempt"] = 3
    subjective_abort = deepcopy(ledger)
    subjective_abort["trials"][0]["outcome"] = "technical_abort"
    subjective_abort["trials"][0]["abort_reason"] = "subjective_bad_trial"
    completed_with_abort = deepcopy(ledger)
    completed_with_abort["trials"][0]["outcome"] = "completed"
    completed_with_abort["trials"][0]["abort_reason"] = "realsense_disconnect"
    abort_without_reason = deepcopy(ledger)
    abort_without_reason["trials"][0]["outcome"] = "technical_abort"
    abort_without_reason["trials"][0]["abort_reason"] = None
    exhausted_slot_reserves = deepcopy(ledger)
    exhausted_slot_reserves["trials"][0]["attempt_index"] = 7
    for invalid in (
        retry_three,
        subjective_abort,
        completed_with_abort,
        abort_without_reason,
        exhausted_slot_reserves,
    ):
        assert list(ledger_validator.iter_errors(invalid))


def test_preregistration_contains_nonnegotiable_decision_rules() -> None:
    text = PREREG_PATH.read_text(encoding="utf-8")
    assert "19,999" in text
    assert "missing predictions count as incorrect" in text.lower()
    assert "eBA_augmented <= eBA_baseline" in text
    assert "RETIRE" in text
    assert "validate_ir_hand_pinch_session_ledger.py" in text


def test_calibration_runbook_is_executable_from_a_clean_worktree() -> None:
    text = CALIBRATION_RUNBOOK_PATH.read_text(encoding="utf-8")
    assert "calibration_contract_test" in text
    assert "heldout_verifier_test" in text
    assert "/scripts/run_lepton_stream.sh start" in text
    assert "/scripts/run_lepton_stream.sh status" in text


def test_calibration_runbook_marks_pass_only_after_sealing() -> None:
    text = CALIBRATION_RUNBOOK_PATH.read_text(encoding="utf-8")
    assert "seal_run_pass" in text
    seal_definition = text.index("seal_run_pass()")
    manifest_install = text.index('mv "$final_manifest_tmp"', seal_definition)
    outer_seal_install = text.index('mv "$final_seal_tmp"', manifest_install)
    pass_write = text.index("PASS >", outer_seal_install)
    assert manifest_install < outer_seal_install < pass_write
    seal_call = text.rindex("seal_run_pass")
    disable_exit_trap = text.index("trap - EXIT", seal_call)
    assert seal_call < disable_exit_trap
    assert "RUN_TERMINAL_STATUS=PASS" not in text


def test_calibration_runbook_records_visibility_and_safety_provenance() -> None:
    text = CALIBRATION_RUNBOOK_PATH.read_text(encoding="utf-8")
    assert "heat_source_type=" in text
    assert "actual_target_distance_m=" in text
    assert "maximum_safe_surface_temperature_c=" in text
    assert "corners_visible_in_both_cameras=true" in text
    assert "thermal_saturation=false" in text


def _complete_session_one_ledger():
    ledger = json.loads(
        (FIXTURES / "ir_hand_pinch_session_ledger.valid.json").read_text(encoding="utf-8")
    )
    schedule = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
    primary = schedule["sessions"][0]["primary"]
    template = ledger["trials"][0]
    trials = []
    for slot, code in enumerate(primary, start=1):
        trial = deepcopy(template)
        trial["scheduled_slot"] = slot
        trial["attempt_index"] = 1
        trial["label"] = "loose" if code == "L" else "tight"
        trial["start_monotonic_ns"] = slot * 10_000_000_000
        trial["end_monotonic_ns"] = trial["start_monotonic_ns"] + 5_000_000_000
        trials.append(trial)
    ledger["schedule_sha256"] = schedule["sessions"][0]["sha256"]
    ledger["trials"] = trials
    return ledger, schedule


def _add_loose_aborts(ledger: dict, count: int) -> dict:
    modified = deepcopy(ledger)
    trials = []
    remaining = count
    for completed in modified["trials"]:
        if completed["label"] == "loose" and remaining:
            abort = deepcopy(completed)
            abort["outcome"] = "technical_abort"
            abort["abort_reason"] = "realsense_disconnect"
            completed["attempt_index"] = 2
            trials.extend((abort, completed))
            remaining -= 1
        else:
            trials.append(completed)
    assert remaining == 0
    modified["trials"] = trials
    return modified


def test_sealed_ledger_semantics_accept_the_frozen_schedule() -> None:
    ledger, schedule = _complete_session_one_ledger()
    validate_ledger_semantics(ledger, schedule)


def test_ledger_recomputes_schedule_hash_instead_of_trusting_schedule_file() -> None:
    ledger, schedule = _complete_session_one_ledger()
    tampered = deepcopy(schedule)
    primary = list(tampered["sessions"][0]["primary"])
    loose_index = primary.index("L")
    tight_index = primary.index("T")
    primary[loose_index], primary[tight_index] = primary[tight_index], primary[loose_index]
    tampered["sessions"][0]["primary"] = "".join(primary)
    for trial in ledger["trials"]:
        code = primary[trial["scheduled_slot"] - 1]
        trial["label"] = "loose" if code == "L" else "tight"
    with pytest.raises(LedgerSemanticError, match="sha256"):
        validate_ledger_semantics(ledger, tampered)


@pytest.mark.parametrize(
    "primary",
    (
        "X" + "L" * 14 + "T" * 15,
        "L" * 16 + "T" * 14,
    ),
)
def test_ledger_rejects_invalid_schedule_alphabet_or_balance(primary: str) -> None:
    ledger, schedule = _complete_session_one_ledger()
    schedule["sessions"][0]["primary"] = primary
    schedule["sessions"][0]["sha256"] = hashlib.sha256(primary.encode()).hexdigest()
    ledger["schedule_sha256"] = schedule["sessions"][0]["sha256"]
    with pytest.raises(LedgerSemanticError, match="L/T"):
        validate_ledger_semantics(ledger, schedule)


def test_completed_trial_requires_successful_pretrial_ffc() -> None:
    ledger, schedule = _complete_session_one_ledger()
    invalid = deepcopy(ledger)
    invalid["trials"][0]["ffc"]["pretrial_complete"] = False
    invalid["trials"][0]["ffc"]["since_last_ffc_reset"] = False
    schema = json.loads(LEDGER_SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    assert list(validator.iter_errors(invalid))
    with pytest.raises(LedgerSemanticError, match="pretrial FFC"):
        validate_ledger_semantics(invalid, schedule)


def test_active_analysis_ffc_forces_completed_thermal_modality_invalid() -> None:
    ledger, schedule = _complete_session_one_ledger()
    invalid = deepcopy(ledger)
    invalid["trials"][0]["ffc"]["analysis_window_active"] = True
    invalid["trials"][0]["thermal_validity"] = {"valid": True, "reason": None}
    invalid["trials"][0]["artifacts"]["thermal"] = deepcopy(
        invalid["trials"][0]["artifacts"]["rgb"]
    )
    schema = json.loads(LEDGER_SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    assert list(validator.iter_errors(invalid))
    with pytest.raises(LedgerSemanticError, match="active FFC"):
        validate_ledger_semantics(invalid, schedule)


def test_completed_valid_modalities_require_raw_artifact_references() -> None:
    ledger, schedule = _complete_session_one_ledger()
    invalid = deepcopy(ledger)
    trial = invalid["trials"][0]
    trial["realsense_validity"] = {"valid": True, "reason": None}
    trial["thermal_validity"] = {"valid": True, "reason": None}
    trial["artifacts"] = {"rgb": None, "depth": None, "thermal": None}
    schema = json.loads(LEDGER_SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    assert list(validator.iter_errors(invalid))
    with pytest.raises(LedgerSemanticError, match="artifact"):
        validate_ledger_semantics(invalid, schedule)


def test_ledger_rejects_more_than_five_reserves_for_one_label() -> None:
    ledger, schedule = _complete_session_one_ledger()
    too_many = _add_loose_aborts(ledger, 6)
    schema = json.loads(LEDGER_SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    assert list(validator.iter_errors(too_many))
    with pytest.raises(LedgerSemanticError, match="reserve budget"):
        validate_ledger_semantics(too_many, schedule)


def test_ledger_rejects_duplicate_slot_attempt_key_even_if_payload_differs() -> None:
    ledger, schedule = _complete_session_one_ledger()
    duplicate = deepcopy(ledger["trials"][0])
    duplicate["start_monotonic_ns"] += 1
    ledger["trials"].insert(1, duplicate)
    with pytest.raises(LedgerSemanticError, match="duplicate scheduled_slot/attempt_index"):
        validate_ledger_semantics(ledger, schedule)
