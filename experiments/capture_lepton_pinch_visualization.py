#!/usr/bin/env python3
"""Robot-free direct-pinch capture with descriptive Stage 1E calculations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import analyze_lepton_pinch_visualization as visualization_analyzer
import live_lepton_hand_shadow as runner
from ir_force.pinch_visualization import (
    FrameArchive,
    render_session,
)


SESSION_NAME = re.compile(r"stage1e_tip_pinch_visualization_\d{2}")
CONTACT_NULL_SESSION_NAME = re.compile(
    r"stage1e_tip_pinch_contact_null_\d{2}"
)
_DEFAULT_PINCH_SIGNAL_INSTRUCTION = runner._pinch_signal_instruction


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames",
        required=True,
        type=runner._attempt_count,
    )
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument(
        "--lepton-port",
        type=runner._udp_port,
        default=8080,
    )
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--manual-ffc", action="store_true")
    parser.add_argument("--contact-only", action="store_true")
    args = parser.parse_args(argv)
    expected_name = (
        CONTACT_NULL_SESSION_NAME if args.contact_only else SESSION_NAME
    )
    if not expected_name.fullmatch(args.session_dir.name):
        parser.error(
            "--session-dir basename must match "
            + (
                "stage1e_tip_pinch_contact_null_NN"
                if args.contact_only
                else "stage1e_tip_pinch_visualization_NN"
            )
        )
    if not args.preview:
        parser.error("visualization capture requires --preview")
    if not args.manual_ffc:
        parser.error("visualization capture requires --manual-ffc")
    return args


def _contact_only_instruction(cue):
    phase = cue["phase"]
    if phase == "prepare_press_hard":
        return "KEEP JUST TOUCH - SPACE WHEN READY"
    if phase == "record_press_hard":
        return "HOLD JUST TOUCH - NO PRESS"
    return _DEFAULT_PINCH_SIGNAL_INSTRUCTION(cue)


def _contact_only_display_label(cue):
    if cue["phase"] in ("prepare_press_hard", "record_press_hard"):
        return "contact"
    return cue["label"]


def create_session_dir(session_dir: Path) -> None:
    Path(session_dir).mkdir(parents=True, exist_ok=False)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.session_dir.exists():
        print(
            f"visualization capture blocked: session exists: "
            f"{args.session_dir}",
            file=sys.stderr,
        )
        return 1
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        print(f"raw D435i source blocked: {exc}", file=sys.stderr)
        return 1

    print(
        "Running approved Pi C++ manual FFC before capture...",
        file=sys.stderr,
        flush=True,
    )
    try:
        ffc_output = runner._run_manual_ffc()
    except Exception as exc:
        print(
            f"visualization manual FFC blocked: {exc!r}",
            file=sys.stderr,
        )
        return 1
    print(ffc_output.strip(), file=sys.stderr, flush=True)

    try:
        create_session_dir(args.session_dir)
        archive = FrameArchive(
            args.session_dir,
            **(
                {"physical_protocol": "contact_only"}
                if args.contact_only
                else {}
            ),
        )
    except Exception as exc:
        print(
            f"visualization session setup blocked: {exc!r}",
            file=sys.stderr,
        )
        return 1

    capture_jsonl = args.session_dir / "capture.jsonl"
    analysis_json = args.session_dir / "descriptive_analysis.json"
    print(
        (
            "Robot-free contact-null visualization; six B-mode groups, "
            "all three legacy analysis slots are light contact only, "
            "with no pressing."
            if args.contact_only
            else (
                "Robot-free descriptive visualization only; six B-mode "
                "groups."
            )
        )
        + " No threshold update, Stage 1F authority, controller, or actuation.",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"Collecting {args.frames} attempts -> {args.session_dir}",
        file=sys.stderr,
        flush=True,
    )
    try:
        summary = runner.run_shadow(
            attempts=args.frames,
            output_path=capture_jsonl,
            rs_module=rs,
            raw_source_factory=lambda: runner.RealSenseRawProjectorCamera(
                rs_module=rs
            ),
            thermal_source_factory=lambda: runner.LeptonUDPSource(
                port=args.lepton_port
            ),
            hands_factory=runner._default_hands_factory,
            preview=True,
            manual_ffc_before_start=True,
            diagnose_inward_samples=False,
            pinch_signal_trial=True,
            attempt_artifact_writer=archive,
            **(
                {
                    "pinch_signal_instruction": _contact_only_instruction,
                    "pinch_signal_display_label": (
                        _contact_only_display_label
                    ),
                }
                if args.contact_only
                else {}
            ),
        )
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(
            f"visualization capture blocked: {exc!r}",
            file=sys.stderr,
        )
        return 1

    try:
        render_session(capture_jsonl, args.session_dir)
        visualization_analyzer.main(
            [
                "--input",
                str(capture_jsonl),
                "--output",
                str(analysis_json),
            ]
        )
    except Exception as exc:
        print(
            f"visualization post-processing blocked: {exc!r}",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0 if summary["pinch_signal_protocol_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
