import json

import pytest

import analyze_lepton_pinch_signal as frozen_analyzer
import analyze_lepton_pinch_visualization as visualization_analyzer
# `_document` is a fixture builder owned by the signal test. This suite runs under
# pytest's importlib import mode -- required because the two IR lines contribute
# same-named test files -- and there a sibling test is not importable by bare name.
import importlib.util as _ilu
from pathlib import Path as _Path

_spec = _ilu.spec_from_file_location(
    "_pinch_signal_fixtures",
    _Path(__file__).with_name("test_analyze_lepton_pinch_signal.py"),
)
_fixtures = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)
_document = _fixtures._document


def _visualization_document(**kwargs):
    rows = _document(**kwargs)
    rows[0]["visualization_capture"] = {
        "experiment_id": "stage1e_tip_pinch_visualization_01",
        "experiment_role": "communication_only_descriptive_replication",
        "decision_authority": "none",
        "authoritative_reference": "stage1e_tip_pinch_signal_05",
        "can_update_thresholds": False,
        "can_authorize_stage1f": False,
        "artifact_paths": "relative_to_session_dir",
    }
    return rows


@pytest.mark.parametrize(
    "document_kwargs, expected_verdict",
    [
        ({}, "PROCEED_TO_STAGE1F_SHADOW"),
        (
            {"thumb_press": -12.0, "index_press": 12.0},
            "STOP_BEFORE_STAGE1F",
        ),
    ],
)
def test_visualization_analysis_preserves_frozen_result_but_has_no_authority(
    document_kwargs,
    expected_verdict,
):
    rows = _visualization_document(**document_kwargs)

    result = visualization_analyzer.analyze_visualization_document(rows)

    assert result == {
        "schema_version": 1,
        "experiment_id": "stage1e_tip_pinch_visualization_01",
        "analysis_role": "descriptive_replication",
        "decision_authority": "none",
        "authoritative_reference": "stage1e_tip_pinch_signal_05",
        "can_update_thresholds": False,
        "can_authorize_stage1f": False,
        "interpretation": "DESCRIPTIVE_ONLY_NO_STAGE1F_AUTHORITY",
        "frozen_analysis": frozen_analyzer.analyze_document(rows),
    }
    assert result["frozen_analysis"]["verdict"] == expected_verdict
    assert result["decision_authority"] == "none"
    assert result["can_authorize_stage1f"] is False


@pytest.mark.parametrize(
    "field, value",
    [
        ("experiment_role", "confirmatory"),
        ("decision_authority", "gate"),
        ("authoritative_reference", "stage1e_tip_pinch_signal_06"),
        ("can_update_thresholds", True),
        ("can_authorize_stage1f", True),
    ],
)
def test_visualization_analysis_rejects_mismatched_identity(field, value):
    rows = _visualization_document()
    rows[0]["visualization_capture"][field] = value

    with pytest.raises(ValueError, match="identity"):
        visualization_analyzer.analyze_visualization_document(rows)


def test_visualization_analysis_requires_exactly_one_metadata_row():
    missing = _document()
    duplicate = _visualization_document()
    duplicate.insert(1, duplicate[0].copy())

    for rows in (missing, duplicate):
        with pytest.raises(ValueError, match="metadata"):
            visualization_analyzer.analyze_visualization_document(rows)


def test_visualization_analysis_cli_writes_exclusively(tmp_path):
    input_path = tmp_path / "capture.jsonl"
    input_path.write_text(
        "".join(
            json.dumps(row, allow_nan=False) + "\n"
            for row in _visualization_document()
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "descriptive_analysis.json"

    assert (
        visualization_analyzer.main(
            ["--input", str(input_path), "--output", str(output_path)]
        )
        == 0
    )
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["decision_authority"] == "none"
    assert written["frozen_analysis"]["verdict"] == (
        "PROCEED_TO_STAGE1F_SHADOW"
    )
    before = output_path.read_bytes()

    with pytest.raises(FileExistsError):
        visualization_analyzer.main(
            ["--input", str(input_path), "--output", str(output_path)]
        )

    assert output_path.read_bytes() == before
