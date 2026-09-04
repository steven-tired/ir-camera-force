#!/usr/bin/env python3
"""Bounded robot-free live shadow for raw D435i to Lepton projection."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import sys
import time

import numpy as np


_CHECKOUT_ROOT = Path(__file__).resolve().parents[1]
if str(_CHECKOUT_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHECKOUT_ROOT))

from ir_force.data_paths import calibration_runs_root
from ir_force.realsense_camera import RealSenseRawProjectorCamera
from ir_force.ir_capture import LeptonUDPSource
from ir_force.ir_thermal_projection import (
    FROZEN_EXTRINSIC_SHA256,
    load_frozen_thermal_geometry,
)
from ir_force.ir_thermal_sparse_projection import (
    project_raw_depth_samples_to_sparse_thermal,
)


FROZEN_XML = (
    calibration_runs_root()
    / "worktrees/20260724T210232Z-attempt01/calibration/FINAL_flir_brown"
    / "extrinsic_refined.xml"
)
STAGE0_RUNTIME_JSON = _CHECKOUT_ROOT / "scratch_lepton/stage0b_runtime_datums.json"
STAGE0_RUNTIME_SHA256 = (
    "22d41109dcaefb29ad770fb5715c35dfd6c13c68195fbcb55e3b9d6fb4ef756b"
)
MAX_ATTEMPTS = 100
MAX_HOST_READ_COMPLETION_AGE_S = 0.35
MAX_HOST_READ_COMPLETION_SKEW_S = 0.15
SOURCE_TIME_LIMITATION = "color_thermal_source_time_not_comparable"
_CONTRACT_KEYS = (
    "requested",
    "sdk_version",
    "device",
    "resolved",
    "color_intrinsics",
    "depth_intrinsics",
    "depth_scale_m",
    "factory_extrinsics",
)


@dataclass(frozen=True)
class _PairState:
    color_frame_number: int
    depth_frame_number: int
    lepton_frame_counter: int | None
    lepton_packet_timestamp_ms: int | None


def _depth_pixel(value: str) -> tuple[int, int]:
    try:
        x_text, y_text = value.split(",", maxsplit=1)
        x, y = int(x_text), int(y_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("depth pixel must be X,Y integers") from exc
    if not (0 <= x < 1280 and 0 <= y < 720):
        raise argparse.ArgumentTypeError(
            "depth pixel must satisfy 0 <= X < 1280 and 0 <= Y < 720"
        )
    return x, y


def _attempt_count(value: str) -> int:
    try:
        attempts = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("frames must be an integer") from exc
    if not 1 <= attempts <= MAX_ATTEMPTS:
        raise argparse.ArgumentTypeError(
            f"frames must satisfy 1 <= N <= {MAX_ATTEMPTS}"
        )
    return attempts


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
    parser.add_argument(
        "--depth-pixel",
        action="append",
        type=_depth_pixel,
        required=True,
        dest="depth_pixels",
    )
    parser.add_argument("--frames", required=True, type=_attempt_count)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lepton-port", type=_udp_port, default=8080)
    return parser.parse_args(argv)


def _load_stage0_contract(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    actual_hash = sha256(raw).hexdigest()
    if actual_hash != STAGE0_RUNTIME_SHA256:
        raise ValueError(
            "Stage 0 runtime JSON SHA-256 mismatch: "
            f"expected {STAGE0_RUNTIME_SHA256}, got {actual_hash}"
        )
    document = json.loads(raw)
    if document.get("status") != "ok":
        raise ValueError("Stage 0 runtime JSON status is not ok")
    try:
        contract = {key: document[key] for key in _CONTRACT_KEYS}
    except KeyError as exc:
        raise ValueError(f"Stage 0 runtime JSON missing {exc.args[0]}") from exc
    return contract, actual_hash


def _pair_rejection_reasons(raw, thermal, *, now_s: float, prior):
    reasons = []
    times = (now_s, raw.observed_at_s, thermal.t)
    if not all(isfinite(value) for value in times):
        reasons.append("host_read_completion_time_invalid")
    else:
        d435_age = now_s - raw.observed_at_s
        lepton_age = now_s - thermal.t
        if d435_age < 0.0 or d435_age > MAX_HOST_READ_COMPLETION_AGE_S:
            reasons.append("d435_host_read_completion_stale")
        if lepton_age < 0.0 or lepton_age > MAX_HOST_READ_COMPLETION_AGE_S:
            reasons.append("lepton_host_read_completion_stale")
        if (
            abs(raw.observed_at_s - thermal.t)
            > MAX_HOST_READ_COMPLETION_SKEW_S
        ):
            reasons.append("host_read_completion_skew_exceeded")

    if raw.color_timestamp_domain != raw.depth_timestamp_domain:
        reasons.append("d435_timestamp_domain_mismatch")
    if prior is not None:
        if raw.color_frame_number <= prior.color_frame_number:
            reasons.append("d435_color_frame_non_increasing")
        if raw.depth_frame_number <= prior.depth_frame_number:
            reasons.append("d435_depth_frame_non_increasing")

    telemetry = thermal.lepton_telemetry
    if telemetry is None:
        reasons.append("lepton_telemetry_missing")
        return tuple(reasons)
    if telemetry.ffc_desired:
        reasons.append("lepton_ffc_desired")
    if telemetry.ffc_in_progress:
        reasons.append("lepton_ffc_in_progress")
    if telemetry.ffc_state != "complete":
        reasons.append("lepton_ffc_not_idle")
    if (
        not telemetry.tlinear_enabled
        or telemetry.tlinear_resolution_k != 0.01
    ):
        reasons.append("lepton_tlinear_invalid")
    if prior is not None and prior.lepton_frame_counter is not None:
        if telemetry.frame_counter <= prior.lepton_frame_counter:
            reasons.append("lepton_frame_counter_non_increasing")
    if prior is not None and prior.lepton_packet_timestamp_ms is not None:
        if telemetry.packet_timestamp_ms <= prior.lepton_packet_timestamp_ms:
            reasons.append("lepton_packet_timestamp_non_increasing")
    return tuple(reasons)


def _pair_state(raw, thermal):
    telemetry = thermal.lepton_telemetry
    return _PairState(
        color_frame_number=int(raw.color_frame_number),
        depth_frame_number=int(raw.depth_frame_number),
        lepton_frame_counter=(
            None if telemetry is None else int(telemetry.frame_counter)
        ),
        lepton_packet_timestamp_ms=(
            None if telemetry is None else int(telemetry.packet_timestamp_ms)
        ),
    )


def _advance_pair_state(prior, current):
    if prior is None:
        return current

    def maximum(left, right):
        if left is None:
            return right
        if right is None:
            return left
        return max(left, right)

    return _PairState(
        color_frame_number=max(
            prior.color_frame_number,
            current.color_frame_number,
        ),
        depth_frame_number=max(
            prior.depth_frame_number,
            current.depth_frame_number,
        ),
        lepton_frame_counter=maximum(
            prior.lepton_frame_counter,
            current.lepton_frame_counter,
        ),
        lepton_packet_timestamp_ms=maximum(
            prior.lepton_packet_timestamp_ms,
            current.lepton_packet_timestamp_ms,
        ),
    )


def _source_fields(raw, thermal, now_s):
    telemetry = thermal.lepton_telemetry
    d435_age = now_s - raw.observed_at_s
    lepton_age = now_s - thermal.t
    return {
        "d435": {
            "host_read_completion_s": raw.observed_at_s,
            "color_frame_number": raw.color_frame_number,
            "depth_frame_number": raw.depth_frame_number,
            "color_timestamp_ms": raw.color_timestamp_ms,
            "depth_timestamp_ms": raw.depth_timestamp_ms,
            "color_timestamp_domain": raw.color_timestamp_domain,
            "depth_timestamp_domain": raw.depth_timestamp_domain,
        },
        "lepton": {
            "host_read_completion_s": thermal.t,
            "frame_counter": None if telemetry is None else telemetry.frame_counter,
            "packet_timestamp_ms": (
                None if telemetry is None else telemetry.packet_timestamp_ms
            ),
            "ffc_desired": None if telemetry is None else telemetry.ffc_desired,
            "ffc_state": None if telemetry is None else telemetry.ffc_state,
            "ffc_in_progress": (
                None if telemetry is None else telemetry.ffc_in_progress
            ),
            "since_last_ffc_s": (
                None if telemetry is None else telemetry.since_last_ffc_s
            ),
            "tlinear_enabled": (
                None if telemetry is None else telemetry.tlinear_enabled
            ),
            "tlinear_resolution_k": (
                None if telemetry is None else telemetry.tlinear_resolution_k
            ),
        },
        "d435_host_read_completion_age_s": d435_age,
        "lepton_host_read_completion_age_s": lepton_age,
        "host_read_completion_age_s": max(d435_age, lepton_age),
        "host_read_completion_skew_s": abs(raw.observed_at_s - thermal.t),
    }


def _projection_fields(result, thermal_frame):
    winners = []
    for thermal_xy, projected in result.winners:
        thermal_x, thermal_y = thermal_xy
        winners.append(
            {
                "source_depth_pixel": list(projected.source_depth_xy),
                "depth_m": projected.depth_m,
                "color_xyz_m": list(projected.color_xyz_m),
                "thermal_xyz_m": list(projected.thermal_xyz_m),
                "thermal_uv": list(projected.thermal_uv),
                "thermal_pixel": [thermal_x, thermal_y],
                "thermal_raw_count": int(thermal_frame[thermal_y, thermal_x]),
            }
        )
    return {
        "winners": winners,
        "projection_rejections": [
            {
                "status": rejected.status,
                "source_depth_pixel": list(rejected.source_depth_xy),
            }
            for rejected in result.rejections
        ],
        "projection_counts": {
            "input": result.input_count,
            "accepted_before_collision": result.accepted_count,
            "rejected": result.rejected_count,
            "collisions": result.collision_count,
            "winners": len(result.winners),
        },
    }


def _write_jsonl(stream, row):
    json.dump(row, stream, sort_keys=True)
    stream.write("\n")
    stream.flush()


def _validated_direct_inputs(depth_pixels, attempts):
    pixels = tuple(depth_pixels)
    if not pixels:
        raise ValueError("at least one depth pixel is required")
    for pixel in pixels:
        if (
            not isinstance(pixel, tuple)
            or len(pixel) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in pixel)
            or not (0 <= pixel[0] < 1280 and 0 <= pixel[1] < 720)
        ):
            raise ValueError("depth pixels must be native 1280x720 integer coordinates")
    if isinstance(attempts, bool) or not isinstance(attempts, int):
        raise ValueError("attempts must be an integer")
    if not 1 <= attempts <= MAX_ATTEMPTS:
        raise ValueError(f"attempts must satisfy 1 <= N <= {MAX_ATTEMPTS}")
    return pixels


def run_shadow(
    *,
    depth_pixels,
    attempts,
    output_path,
    rs_module,
    raw_source_factory,
    thermal_source_factory,
    clock=time.perf_counter,
    frozen_xml=FROZEN_XML,
    stage0_json=STAGE0_RUNTIME_JSON,
):
    pixels = _validated_direct_inputs(depth_pixels, attempts)
    stage0_contract, stage0_hash = _load_stage0_contract(Path(stage0_json))
    R_tc, T_tc, thermal_K, thermal_D = load_frozen_thermal_geometry(frozen_xml)
    summary = {"attempted": 0, "software_gate_accepted": 0, "blocked": 0}

    with ExitStack() as cleanup:
        stream = cleanup.enter_context(
            Path(output_path).open("x", encoding="utf-8")
        )
        raw_source = raw_source_factory()
        cleanup.callback(raw_source.stop)
        try:
            raw_source.start()
        except Exception as exc:
            _write_jsonl(
                stream,
                {
                    "row_type": "metadata",
                    "status": "setup_blocked",
                    "reason": "raw_d435i_start_failed",
                    "error": repr(exc),
                },
            )
            raise
        if raw_source.runtime_metadata != stage0_contract:
            _write_jsonl(
                stream,
                {
                    "row_type": "metadata",
                    "status": "setup_blocked",
                    "reason": "Stage 0 runtime metadata mismatch",
                    "metadata_comparison": "exact_normalized_equality",
                    "expected": stage0_contract,
                    "observed": raw_source.runtime_metadata,
                },
            )
            raise ValueError("Stage 0 runtime metadata mismatch")

        _write_jsonl(
            stream,
            {
                "row_type": "metadata",
                "status": "ok",
                "schema_version": 1,
                "safety_mode": "robot_free_shadow_only",
                "depth_pixels": [list(pixel) for pixel in pixels],
                "max_attempts": attempts,
                "max_host_read_completion_age_s": (
                    MAX_HOST_READ_COMPLETION_AGE_S
                ),
                "max_host_read_completion_skew_s": (
                    MAX_HOST_READ_COMPLETION_SKEW_S
                ),
                "metadata_comparison": "exact_normalized_equality",
                "stage0_runtime_json": str(stage0_json),
                "stage0_runtime_sha256": stage0_hash,
                "frozen_xml": str(frozen_xml),
                "frozen_xml_sha256": FROZEN_EXTRINSIC_SHA256,
                "d435_runtime": raw_source.runtime_metadata,
                "thermal_shape": [120, 160],
                "limitations": [SOURCE_TIME_LIMITATION],
            },
        )

        thermal_source = thermal_source_factory()
        cleanup.callback(thermal_source.close)
        prior = None
        for attempt_index in range(attempts):
            raw = raw_source.read()
            thermal = thermal_source.read()
            now_s = clock()
            reasons = list(
                _pair_rejection_reasons(
                    raw,
                    thermal,
                    now_s=now_s,
                    prior=prior,
                )
            )
            if thermal.frame.shape != (120, 160) or thermal.frame.dtype != np.uint16:
                reasons.append("lepton_frame_invalid")
            row = {
                "row_type": "attempt",
                "attempt_index": attempt_index,
                "limitations": [SOURCE_TIME_LIMITATION],
                **_source_fields(raw, thermal, now_s),
            }
            summary["attempted"] += 1
            prior = _advance_pair_state(prior, _pair_state(raw, thermal))
            if reasons:
                row.update(status="blocked", reasons=reasons)
                summary["blocked"] += 1
                _write_jsonl(stream, row)
                continue

            samples = [
                (x, y, int(raw.depth_z16[y, x]))
                for x, y in pixels
            ]
            row["requested_depth"] = [
                {"depth_pixel": [x, y], "raw_depth": raw_depth}
                for x, y, raw_depth in samples
            ]
            sparse = project_raw_depth_samples_to_sparse_thermal(
                samples=samples,
                rs_module=rs_module,
                depth_scale_m=raw_source.runtime_metadata["depth_scale_m"],
                depth_intrinsics=raw_source.depth_intrinsics,
                depth_to_color_extrinsics=raw_source.depth_to_color_extrinsics,
                R_tc=R_tc,
                T_tc=T_tc,
                thermal_K=thermal_K,
                thermal_D=thermal_D,
            )
            row.update(_projection_fields(sparse, thermal.frame))
            if sparse.status != "ok" or not sparse.winners:
                reasons = [result.status for result in sparse.rejections]
                row.update(
                    status="blocked",
                    reasons=reasons or ["no_valid_projections"],
                )
                summary["blocked"] += 1
            else:
                row.update(status="software_gate_accepted", reasons=[])
                summary["software_gate_accepted"] += 1
            _write_jsonl(stream, row)
    return summary


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        print(f"raw D435i source blocked: {exc}", file=sys.stderr)
        return 1
    try:
        summary = run_shadow(
            depth_pixels=tuple(args.depth_pixels),
            attempts=args.frames,
            output_path=args.output,
            rs_module=rs,
            raw_source_factory=lambda: RealSenseRawProjectorCamera(
                rs_module=rs
            ),
            thermal_source_factory=lambda: LeptonUDPSource(
                port=args.lepton_port
            ),
        )
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Stage 1C live shadow blocked: {exc!r}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["software_gate_accepted"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
