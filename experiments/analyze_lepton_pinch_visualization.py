#!/usr/bin/env python3
"""Replay frozen Stage 1E analysis as descriptive-only visualization data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_lepton_pinch_signal as frozen_analyzer


EXPERIMENT_ROLE = "communication_only_descriptive_replication"
AUTHORITATIVE_REFERENCE = "stage1e_tip_pinch_signal_05"


def _visualization_identity(rows: list[dict]) -> dict:
    metadata_rows = [
        row for row in rows if row.get("row_type") == "metadata"
    ]
    if len(metadata_rows) != 1:
        raise ValueError("expected exactly one metadata row")
    identity = metadata_rows[0].get("visualization_capture")
    if (
        not isinstance(identity, dict)
        or identity.get("experiment_role") != EXPERIMENT_ROLE
        or identity.get("decision_authority") != "none"
        or identity.get("authoritative_reference")
        != AUTHORITATIVE_REFERENCE
        or identity.get("can_update_thresholds") is not False
        or identity.get("can_authorize_stage1f") is not False
        or not isinstance(identity.get("experiment_id"), str)
    ):
        raise ValueError("visualization metadata identity mismatch")
    return identity


def analyze_visualization_document(rows: list[dict]) -> dict:
    if not isinstance(rows, list) or not all(
        isinstance(row, dict) for row in rows
    ):
        raise ValueError("rows must be a list of JSON objects")
    identity = _visualization_identity(rows)
    return {
        "schema_version": 1,
        "experiment_id": identity["experiment_id"],
        "analysis_role": "descriptive_replication",
        "decision_authority": "none",
        "authoritative_reference": AUTHORITATIVE_REFERENCE,
        "can_update_thresholds": False,
        "can_authorize_stage1f": False,
        "interpretation": "DESCRIPTIVE_ONLY_NO_STAGE1F_AUTHORITY",
        "frozen_analysis": frozen_analyzer.analyze_document(rows),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = analyze_visualization_document(rows)
    text = json.dumps(
        result,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.write("\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
