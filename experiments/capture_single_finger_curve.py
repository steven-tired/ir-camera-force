#!/usr/bin/env python3
"""Robot-free continuous Null/Press single-finger thermal capture."""

from __future__ import annotations

import argparse
from collections import deque
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
import sys
import threading
import time

import cv2
import numpy as np


_CHECKOUT_ROOT = Path(__file__).resolve().parents[1]
if str(_CHECKOUT_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHECKOUT_ROOT))

from live_lepton_hand_shadow import (  # noqa: E402
    _associate_color_to_raw_depth,
    _run_manual_ffc,
)
from live_lepton_projector_shadow import (  # noqa: E402
    FROZEN_XML,
    STAGE0_RUNTIME_JSON,
    _load_stage0_contract,
)
from ir_force.realsense_camera import (  # noqa: E402
    RealSenseRawProjectorCamera,
)
from webcam_input.webcam_source import WebcamSource  # noqa: E402
from ir_force.ir_capture import (  # noqa: E402
    LeptonUDPSource,
)
from ir_force.ir_thermal_projection import (  # noqa: E402
    FROZEN_EXTRINSIC_SHA256,
    load_frozen_thermal_geometry,
    project_raw_depth_pixel_to_thermal,
)
from ir_force.single_finger_curve_protocol import (  # noqa: E402
    PRIMARY_BLOCKS,
    RESERVE_BLOCKS,
    TrialSpec,
    phase_at,
    phase_elapsed,
    scheduled_trial_specs,
    trial_integrity,
)
from ir_force.single_finger_curve_runtime import (  # noqa: E402
    ContinuousFrameArchive,
    build_frame_row,
)
from ir_force.single_finger_thermal_tracking import (  # noqa: E402
    FINGER_WIDTH_TOLERANCE_PX,
    MIN_FINGER_WIDTH_PX,
    initialize_trial_anchor,
)


EXPERIMENT_IDENTITY = "single_finger_null_press_continuous_v1"
SESSION_NAME_PATTERN = re.compile(r"single_finger_surface_press_curve_\d{2}")
FFC_GUARD_S = 5.0
POST_A3_REST_S = 10.0


def _udp_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Lepton port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("Lepton port must be in [1, 65535]")
    return port


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument("--surface-material", required=True)
    parser.add_argument("--surface-photo", type=Path)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--manual-ffc", action="store_true")
    parser.add_argument("--lepton-port", type=_udp_port, default=8080)
    parser.add_argument(
        "--readiness-mode",
        choices=("thermal_only", "d435_and_thermal"),
        default="thermal_only",
        help=(
            "which modalities must be valid before SPACE starts a trial. "
            "The D435-to-thermal chain failed on 100%% of frames in session 01 "
            "at the working distance this ROI needs, and it does not feed the "
            "v2 primary value, so it is recorded but not gating by default."
        ),
    )
    args = parser.parse_args(argv)
    if not SESSION_NAME_PATTERN.fullmatch(args.session_dir.name):
        parser.error(
            "--session-dir basename must match "
            "single_finger_surface_press_curve_NN"
        )
    if args.session_dir.exists():
        parser.error("--session-dir must not already exist")
    if (
        args.surface_photo is not None
        and not args.surface_photo.is_file()
    ):
        print(
            "surface photo not found; continuing without setup photo",
            file=sys.stderr,
        )
        args.surface_photo = None
    args.surface_material = args.surface_material.strip()
    if not args.surface_material:
        parser.error("--surface-material must not be blank")
    if not args.preview:
        parser.error("--preview is required")
    if not args.manual_ffc:
        parser.error("--manual-ffc is required")
    return args


def phase_cue(phase: str, condition: str) -> str:
    if phase == "A1":
        return "A1: LIGHT CONTACT"
    if phase == "X" and condition == "null":
        return "X/null: KEEP LIGHT CONTACT"
    if phase == "X" and condition == "press":
        return "X/press: PRESS HARD"
    if phase == "A2":
        return "A2: LIGHT CONTACT"
    if phase == "A3":
        return "A3: LIFT - NO CONTACT"
    if phase == "REST":
        return "REST: NO CONTACT"
    raise ValueError("unknown phase or condition")


def realized_block_indices(validity_by_block) -> list[int]:
    realized = []
    valid_pairs = 0
    for block_index, valid in enumerate(validity_by_block):
        if block_index >= len(PRIMARY_BLOCKS) and valid_pairs >= 6:
            break
        if block_index >= len(PRIMARY_BLOCKS) + len(RESERVE_BLOCKS):
            break
        realized.append(block_index)
        valid_pairs += int(bool(valid))
    return realized


def _json_value(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if isfinite(float(value)) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_jsonl(stream, row) -> None:
    json.dump(_json_value(row), stream, sort_keys=True, allow_nan=False)
    stream.write("\n")
    stream.flush()


def _thermal_preview(frame: np.ndarray) -> np.ndarray:
    lower, upper = (float(value) for value in np.percentile(frame, (1.0, 99.0)))
    if upper <= lower:
        upper = lower + 1.0
    scaled = np.clip(
        (frame.astype(np.float32) - lower) / (upper - lower),
        0.0,
        1.0,
    )
    return cv2.applyColorMap(
        np.rint(scaled * 255.0).astype(np.uint8),
        cv2.COLORMAP_INFERNO,
    )


def default_key_source(color_rgb, thermal_counts, lines) -> int:
    color = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    for index, line in enumerate(lines):
        cv2.putText(
            color,
            str(line),
            (20, 35 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    thermal = cv2.resize(
        _thermal_preview(thermal_counts),
        (640, 480),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.imshow("single-finger D435i", color)
    cv2.imshow("single-finger Lepton (auto contrast, display only)", thermal)
    return cv2.waitKey(1) & 0xFF


class LatestRawProjectorSource:
    """Continuously read D435i so the Lepton loop does not drop native frames."""

    def __init__(self, source):
        self.source = source
        self._latest = None
        self._error = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._running = False
        self._thread = None

    def start(self) -> None:
        self.source.start()
        self._running = True
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("D435i did not produce a raw frame within 5 seconds")
        if self._error is not None:
            raise RuntimeError(f"D435i raw reader failed: {self._error!r}")

    def _produce(self) -> None:
        while self._running:
            try:
                sample = self.source.read()
            except Exception as exc:
                with self._lock:
                    self._error = exc
                self._ready.set()
                return
            with self._lock:
                self._latest = sample
            self._ready.set()

    def read_latest(self):
        with self._lock:
            sample = self._latest
            error = self._error
        if error is not None:
            raise RuntimeError(f"D435i raw reader failed: {error!r}")
        if sample is None:
            raise RuntimeError("D435i raw frame is not ready")
        return sample

    def close(self) -> None:
        self._running = False
        self.source.stop()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def capture_trial_frames(
    *,
    spec: TrialSpec,
    trial_index: int,
    start_frame_index: int,
    raw_reader,
    thermal_source,
    hands,
    projection_context,
    archive,
    stream,
    clock=time.perf_counter,
    key_source=default_key_source,
    frame_builder=build_frame_row,
) -> dict:
    started_s = float(clock())
    rows = []
    frame_index = start_frame_index
    aborted = False
    while True:
        thermal = thermal_source.read()
        now_s = float(clock())
        elapsed_s = now_s - started_s
        phase = phase_at(elapsed_s)
        if phase is None:
            break
        raw = raw_reader()
        artifact_paths = {}
        artifact_write_ok = True
        artifact_error = None
        try:
            artifact_paths = archive.capture(
                frame_index=frame_index,
                thermal_counts=thermal.frame,
                color_rgb=raw.color_rgb,
                depth_z16=raw.depth_z16,
                color_frame_number=raw.color_frame_number,
                depth_frame_number=raw.depth_frame_number,
            )
        except Exception as exc:
            artifact_write_ok = False
            artifact_error = repr(exc)
        context = {
            "trial_index": trial_index,
            "block_index": spec.block_index,
            "condition": spec.condition,
            "order_in_block": spec.order_in_block,
            "reserve": spec.reserve,
            "phase": phase,
            "phase_elapsed_s": phase_elapsed(elapsed_s),
            "global_elapsed_s": elapsed_s,
            "frame_index": frame_index,
            "now_s": now_s,
        }
        row = frame_builder(
            raw,
            thermal,
            hands,
            projection_context,
            context,
        )
        row.update(
            artifact_write_ok=artifact_write_ok,
            frame_artifacts=artifact_paths,
        )
        if artifact_error is not None:
            row["artifact_write_error"] = artifact_error
        _write_jsonl(stream, row)
        rows.append(row)
        frame_index += 1
        key = key_source(
            raw.color_rgb,
            thermal.frame,
            (
                f"block {spec.block_index + 1} "
                f"trial {spec.order_in_block + 1}: {spec.condition}",
                phase_cue(phase, spec.condition),
                f"{5.0 - phase_elapsed(elapsed_s):.1f} s",
                "q: abort",
            ),
        )
        if key in (ord("q"), ord("Q")):
            aborted = True
            break
    return {
        "rows": rows,
        "next_frame_index": frame_index,
        "aborted": aborted,
    }


def _wait_for_space(
    *,
    spec,
    raw_reader,
    thermal_source,
    hands,
    projection_context,
    key_source,
    readiness_mode="thermal_only",
    frame_builder=build_frame_row,
    thermal_anchor_builder=initialize_trial_anchor,
    thermal_window_size=5,
) -> bool:
    thermal_window = deque(maxlen=thermal_window_size)
    while True:
        raw = raw_reader()
        thermal = thermal_source.read()
        thermal_window.append(thermal.frame.copy())
        readiness = frame_builder(
            raw,
            thermal,
            hands,
            projection_context,
            {
                "trial_index": -1,
                "block_index": spec.block_index,
                "condition": spec.condition,
                "order_in_block": spec.order_in_block,
                "reserve": spec.reserve,
                "phase": "READY",
                "phase_elapsed_s": 0.0,
                "global_elapsed_s": 0.0,
                "frame_index": -1,
                "now_s": time.perf_counter(),
            },
        )
        d435_ready = readiness.get("tracking_valid") is True
        reasons = readiness.get("tracking_reasons") or ["unknown"]
        thermal_width = None
        thermal_reason = None
        if len(thermal_window) < thermal_window_size:
            thermal_reason = (
                f"thermal anchor {len(thermal_window)}/{thermal_window_size}"
            )
        else:
            try:
                thermal_anchor = thermal_anchor_builder(list(thermal_window))
                thermal_width = float(thermal_anchor["finger_width_px"])
            except ValueError as exc:
                thermal_reason = f"thermal anchor: {exc}"
        thermal_ready = (
            thermal_width is not None
            and thermal_width
            >= MIN_FINGER_WIDTH_PX - FINGER_WIDTH_TOLERANCE_PX
        )
        thermal_status = (
            f"thermal width {thermal_width:.1f}px"
            if thermal_width is not None
            else str(thermal_reason)
        )
        d435_gates = readiness_mode == "d435_and_thermal"
        ready = thermal_ready and (d435_ready or not d435_gates)
        d435_status = (
            "D435-to-thermal ROI valid"
            if d435_ready
            else "D435 " + ", ".join(str(reason) for reason in reasons)
        )
        if ready:
            readiness_line = (
                f"READY: {thermal_status}; {d435_status}"
                if d435_gates
                else f"READY: {thermal_status} [D435 recorded, not gating: "
                f"{d435_status}]"
            )
        else:
            blockers = []
            if d435_gates and not d435_ready:
                blockers.append(d435_status)
            if not thermal_ready:
                blockers.append(thermal_status)
            readiness_line = f"NOT READY: {'; '.join(blockers)}"
        key = key_source(
            raw.color_rgb,
            thermal.frame,
            (
                f"block {spec.block_index + 1} "
                f"trial {spec.order_in_block + 1}: {spec.condition}",
                readiness_line,
                "SPACE: start A1",
                "q: abort",
            ),
        )
        if key in (ord("q"), ord("Q")):
            return False
        if key == ord(" ") and ready:
            return True


def _run_rest(
    *,
    condition,
    raw_reader,
    thermal_source,
    key_source,
    clock,
) -> bool:
    started_s = float(clock())
    while float(clock()) - started_s < POST_A3_REST_S:
        raw = raw_reader()
        thermal = thermal_source.read()
        remaining = max(
            POST_A3_REST_S - (float(clock()) - started_s),
            0.0,
        )
        key = key_source(
            raw.color_rgb,
            thermal.frame,
            (
                phase_cue("REST", condition),
                f"{remaining:.1f} s",
                "q: abort",
            ),
        )
        if key in (ord("q"), ord("Q")):
            return False
    return True


def _projection_context(raw_source, rs_module, geometry) -> dict:
    R_tc, T_tc, thermal_K, thermal_D = geometry
    return {
        "split_results": WebcamSource.split_results,
        "associate_color_to_raw_depth": _associate_color_to_raw_depth,
        "project_depth_pixel_to_thermal": project_raw_depth_pixel_to_thermal,
        "association_kwargs": {
            "rs_module": rs_module,
            "depth_scale_m": raw_source.runtime_metadata["depth_scale_m"],
            "depth_intrinsics": raw_source.depth_intrinsics,
            "color_intrinsics": raw_source.color_intrinsics,
            "color_to_depth_extrinsics": raw_source.color_to_depth_extrinsics,
            "depth_to_color_extrinsics": raw_source.depth_to_color_extrinsics,
        },
        "projection_kwargs": {
            "rs_module": rs_module,
            "depth_scale_m": raw_source.runtime_metadata["depth_scale_m"],
            "depth_intrinsics": raw_source.depth_intrinsics,
            "depth_to_color_extrinsics": raw_source.depth_to_color_extrinsics,
            "R_tc": R_tc,
            "T_tc": T_tc,
            "thermal_K": thermal_K,
            "thermal_D": thermal_D,
        },
    }


def _default_hands_factory():
    import mediapipe as mp

    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
    )


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def run_session(
    args,
    *,
    rs_module,
    clock=time.perf_counter,
    key_source=default_key_source,
    raw_source_factory=None,
    thermal_source_factory=None,
    hands_factory=_default_hands_factory,
    manual_ffc=_run_manual_ffc,
    sleep=time.sleep,
) -> dict:
    raw_source_factory = raw_source_factory or (
        lambda: RealSenseRawProjectorCamera(rs_module=rs_module)
    )
    thermal_source_factory = thermal_source_factory or (
        lambda: LeptonUDPSource(port=args.lepton_port)
    )
    surface_photo_relative = (
        None
        if args.surface_photo is None
        else (Path("setup") / args.surface_photo.name).as_posix()
    )
    surface_photo_sha256 = (
        None
        if args.surface_photo is None
        else _file_sha256(args.surface_photo)
    )
    archive = ContinuousFrameArchive(args.session_dir, args.surface_photo)
    capture_path = args.session_dir / "capture.jsonl"
    manifest_path = args.session_dir / "manifest.json"
    stream = capture_path.open("x", encoding="utf-8")
    raw_latest = LatestRawProjectorSource(raw_source_factory())
    hands = None
    thermal_source = None
    realized_blocks = []
    valid_pairs = 0
    frame_index = 0
    aborted = False
    error = None
    stage0_hash = None
    try:
        stage0_contract, stage0_hash = _load_stage0_contract(
            STAGE0_RUNTIME_JSON
        )
        geometry = load_frozen_thermal_geometry(FROZEN_XML)
        raw_latest.start()
        if raw_latest.source.runtime_metadata != stage0_contract:
            raise ValueError("Stage 0 runtime metadata mismatch")
        hands = hands_factory()
        projection_context = _projection_context(
            raw_latest.source,
            rs_module,
            geometry,
        )
        _write_jsonl(
            stream,
            {
                "row_type": "metadata",
                "experiment_identity": EXPERIMENT_IDENTITY,
                "schema_version": 1,
                "safety_mode": "robot_free_no_actuation",
                "surface_material": args.surface_material,
                "surface_photo": surface_photo_relative,
                "surface_photo_sha256": surface_photo_sha256,
                "schedule": [list(pair) for pair in PRIMARY_BLOCKS + RESERVE_BLOCKS],
                "phase_duration_s": 5.0,
                "post_a3_rest_s": POST_A3_REST_S,
                "manual_ffc_guard_s": FFC_GUARD_S,
                "raw_thermal_authority": "uint16_counts",
                "rendered_thermal_role": "display_only_auto_contrast",
                "readiness_mode": args.readiness_mode,
                "d435_role": (
                    "gating_and_recorded"
                    if args.readiness_mode == "d435_and_thermal"
                    else "recorded_not_gating"
                ),
                "frozen_xml": str(FROZEN_XML),
                "frozen_xml_sha256": FROZEN_EXTRINSIC_SHA256,
                "stage0_runtime_json": str(STAGE0_RUNTIME_JSON),
                "stage0_runtime_sha256": stage0_hash,
                "d435_runtime": raw_latest.source.runtime_metadata,
            },
        )
        specs = scheduled_trial_specs()
        by_block = {
            block_index: [
                spec for spec in specs if spec.block_index == block_index
            ]
            for block_index in range(len(PRIMARY_BLOCKS) + len(RESERVE_BLOCKS))
        }
        trial_index = 0
        for block_index in range(len(by_block)):
            if block_index >= len(PRIMARY_BLOCKS) and valid_pairs >= 6:
                break
            if thermal_source is not None:
                thermal_source.close()
                thermal_source = None
            ffc_output = manual_ffc()
            sleep(FFC_GUARD_S)
            thermal_source = thermal_source_factory()
            trial_results = []
            for spec in by_block[block_index]:
                if not _wait_for_space(
                    spec=spec,
                    raw_reader=raw_latest.read_latest,
                    thermal_source=thermal_source,
                    hands=hands,
                    projection_context=projection_context,
                    key_source=key_source,
                    readiness_mode=args.readiness_mode,
                ):
                    aborted = True
                    break
                captured = capture_trial_frames(
                    spec=spec,
                    trial_index=trial_index,
                    start_frame_index=frame_index,
                    raw_reader=raw_latest.read_latest,
                    thermal_source=thermal_source,
                    hands=hands,
                    projection_context=projection_context,
                    archive=archive,
                    stream=stream,
                    clock=clock,
                    key_source=key_source,
                )
                frame_index = captured["next_frame_index"]
                integrity = trial_integrity(captured["rows"])
                trial_results.append(integrity)
                _write_jsonl(
                    stream,
                    {
                        "row_type": "trial_summary",
                        "trial_index": trial_index,
                        "block_index": block_index,
                        "condition": spec.condition,
                        "technical_integrity": integrity,
                    },
                )
                trial_index += 1
                if captured["aborted"]:
                    aborted = True
                    break
                if not _run_rest(
                    condition=spec.condition,
                    raw_reader=raw_latest.read_latest,
                    thermal_source=thermal_source,
                    key_source=key_source,
                    clock=clock,
                ):
                    aborted = True
                    break
            pair_valid = (
                len(trial_results) == 2
                and all(result["valid"] for result in trial_results)
            )
            valid_pairs += int(pair_valid)
            block_record = {
                "block_index": block_index,
                "reserve": block_index >= len(PRIMARY_BLOCKS),
                "pair_valid": pair_valid,
                "trial_integrity": trial_results,
                "manual_ffc_complete_marker": (
                    "Manual FFC complete" in str(ffc_output)
                ),
            }
            realized_blocks.append(block_record)
            _write_jsonl(stream, {"row_type": "block_summary", **block_record})
            if aborted:
                break
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        if thermal_source is not None:
            thermal_source.close()
        if hands is not None:
            hands.close()
        try:
            raw_latest.close()
        except Exception:
            pass
        stream.close()
        cv2.destroyAllWindows()
        manifest = {
            "experiment_identity": EXPERIMENT_IDENTITY,
            "status": (
                "error"
                if error is not None
                else "aborted"
                if aborted
                else "complete"
                if valid_pairs == 6
                else "incomplete"
            ),
            "error": error,
            "surface_material": args.surface_material,
            "surface_photo": surface_photo_relative,
            "surface_photo_sha256": surface_photo_sha256,
            "readiness_mode": args.readiness_mode,
            "capture_jsonl": "capture.jsonl",
            "capture_jsonl_sha256": (
                _file_sha256(capture_path) if capture_path.is_file() else None
            ),
            "stage0_runtime_sha256": stage0_hash,
            "frozen_xml_sha256": FROZEN_EXTRINSIC_SHA256,
            "realized_blocks": realized_blocks,
            "valid_pair_count": valid_pairs,
            "frame_row_count": frame_index,
            "raw_thermal_png_count": len(
                list((args.session_dir / "raw/thermal_uint16").glob("*.png"))
            ),
            "rendered_thermal_png_count": len(
                list(
                    (
                        args.session_dir
                        / "rendered/thermal_inferno_auto"
                    ).glob("*.png")
                )
            ),
            "software_ready": True,
            "physical_capture_complete": (
                error is None and not aborted and valid_pairs == 6
            ),
            "signal_verdict": "not_analyzed",
            "force_ground_truth": False,
            "controller_or_robot_actuation": False,
            "stage1f_authority": False,
            "attempt05_status_changed": False,
        }
        with manifest_path.open("x", encoding="utf-8") as output:
            json.dump(
                _json_value(manifest),
                output,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            output.write("\n")
    return manifest


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        print(f"raw D435i source blocked: {exc}", file=sys.stderr)
        return 1
    try:
        manifest = run_session(args, rs_module=rs)
    except Exception as exc:
        print(f"capture failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "valid_pair_count": manifest["valid_pair_count"],
                "session_dir": str(args.session_dir),
            },
            sort_keys=True,
        )
    )
    return 0 if manifest["physical_capture_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
