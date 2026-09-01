"""Webcam END-EFFECTOR teleop of the SO-101 + live hand-cam panel, single process.

Shows the hand-camera feed with landmarks drawn, overlaying control state (MOVING/MIDDLE/HOLD),
the EE target delta, and the joint targets being sent. Right hand moves the gripper in space
(differential, reference latched when motion resumes); hold a RIGHT-hand V sign for 0.4 s = MIDDLE,
lost right hand = HOLD; pinch = gripper. The left hand does not affect arm motion.

Control lives in the SHARED `WebcamEEController` (same code the recorder uses), so live teleop and
recording can't diverge. Runs the robot with use_degrees=True. Keep the e-stop within reach.

Run (stop other camera apps first; default = laptop webcam index 0):
  cd /home/zhuokai/hand-teleop/webcam-input/lerobot_teleoperator_so101_webcam
  env -u PYTHONPATH QT_QPA_PLATFORM=xcb /home/zhuokai/hand-teleop/.venv-lerobot/bin/python teleop_viz_ee.py
Add --oak to use the OAK-D (clean stereo depth) instead of the monocular webcam.

To RECORD a dataset, use record_so101_ee.py (LeRobot record_loop) -- not this script.
"""

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import json
from pathlib import Path
import os
import sys
import time

_CHECKOUT_ROOT = Path(__file__).resolve().parents[1]
if (_CHECKOUT_ROOT / "webcam_input").is_dir():
    sys.path.insert(0, str(_CHECKOUT_ROOT))

import cv2
import mediapipe as mp
import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SOFollower
from webcam_input.depth import ScaleDepthStrategy
from webcam_input.webcam_source import WebcamSource
from webcam_input.wrist_estimator import WebcamWristEstimator

from lerobot_teleoperator_so101_webcam.config_so101_webcam_ee import SO101WebcamEEConfig
from lerobot_teleoperator_so101_webcam.ee_control import gripper_pos_from_pinch
from lerobot_teleoperator_so101_webcam.ee_controller import WebcamEEController
from lerobot_teleoperator_so101_webcam.hand_startup_gate import (
    HAND_STARTUP_DWELL_S,
    MAX_WRIST_ROLL_RANGE_DEG,
    ContinuousHandStartupGate,
)
from ir_force.ir_capture import (
    LatestFrameSource,
    LeptonUDPSource,
    OpenCVCameraSource,
)
from ir_force.ir_hand_calibration import (
    load_projection_calibration,
    validate_projection_calibration,
)
from ir_force.ir_pressure import (
    HandPressureEstimator,
    lepton_pressure_config,
)
from lerobot_teleoperator_so101_webcam.gripper_hardware import (
    GripperClosureLimits,
    GripperTelemetrySampler,
)
from ir_force.ir_shadow_telemetry import (
    IRShadowTelemetryLogger,
    PV_SHADOW_FIELDS,
)
from pressurevision_integration.pv_object_profile import (
    load_object_profile,
    object_profile_sha256,
)
from pressurevision_integration.pv_pressure import (
    PressureVisionSource,
    PressureVisionUDPSource,
)
from pressurevision_integration.pv_preview import (
    DEFAULT_PV_PREVIEW_SHARE,
    PressureVisionPreviewSource,
    draw_gripper_position_banner,
)
from pressurevision_integration.pv_trial_protocol import default_trial_protocol

ARM_PORT = "/dev/ttyACM0"
ARM_ID = "so101_follower_1"
CAMERA_INDEX = 0
URDF_PATH = os.environ.get(
    "SO101_URDF",
    str(Path(os.environ.get("SO_ARM100_DIR", "")) / "Simulation/SO101/so101_new_calib.urdf"),
)
DEFAULT_IR_CALIBRATION = str(
    Path(__file__).resolve().parents[1] / "calibration/flir_oak" / "oak_flir_hand_pressure_projection.json"
)
DEFAULT_THERMAL_PATH = "/dev/video21"
DEFAULT_PV_PORT = 8090
PV_PROFILE_MAPPINGS = ("absolute", "relative", "hard_profile")
PV_CONTROL_MAPPINGS = (
    "relative",
    "soft_direct",
    "soft_precise",
    "carton_span",
    "hard_profile",
)
MIN_IR_CALIBRATION_SAMPLES = 12
MAX_IR_CALIBRATION_RMS_PX = 8.0
MAX_IR_CALIBRATION_ERROR_PX = 16.0
# None on purpose: setting it makes send_action do a SECOND per-frame Present_Position read on a
# flaky bus. Per-step motion is capped by the controller's EMA + slew-limit instead.
MAX_RELATIVE_TARGET = None
MAX_COMM_FAILURES = 10      # consecutive serial write failures before giving up
PV_APPLY_EXTERNAL_VIDEOS = ("creative_side.ts",)
_THUMB_TIP = 4
_INDEX_TIP = 8


def compose_operator_view(
    hand_frame: np.ndarray,
    pv_frame: np.ndarray | None,
) -> np.ndarray:
    """Keep hand tracking and PV visible in one stable two-pane window."""
    height, width = hand_frame.shape[:2]
    pane = np.zeros((height, width, 3), dtype=np.uint8)
    if pv_frame is None:
        cv2.putText(
            pane,
            "PressureVision preview: waiting",
            (20, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        pv_height, pv_width = pv_frame.shape[:2]
        scale = min(width / pv_width, height / pv_height)
        resized = cv2.resize(
            pv_frame,
            (max(1, round(pv_width * scale)), max(1, round(pv_height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        y0 = (height - resized.shape[0]) // 2
        x0 = (width - resized.shape[1]) // 2
        pane[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    return np.hstack((hand_frame, pane))


@dataclass(frozen=True)
class LiveIRRuntime:
    pressure_source: object | None
    pressure_shadow: bool
    sidecar: IRShadowTelemetryLogger | None
    pressure_apply: bool = False
    object_profile: object | None = None
    object_profile_sha256: str | None = None
    trial_protocol: object | None = None
    pv_mapping: str = "absolute"
    gripper_closure_limits: GripperClosureLimits | None = None


def validate_pv_apply_evidence_gate(args) -> Path | None:
    """Require live external recordings before a PV apply can connect the arm."""
    if not bool(getattr(args, "pv_pressure", False)):
        return None
    evidence_dir = Path(args.pv_evidence_dir)
    missing = [
        name
        for name in PV_APPLY_EXTERNAL_VIDEOS
        if not (evidence_dir / name).is_file()
        or (evidence_dir / name).stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(
            "PV apply evidence gate missing live recording(s): " + ", ".join(missing)
        )
    oak_video = evidence_dir / "oak_hand.avi"
    if oak_video.exists():
        raise RuntimeError(f"refusing to overwrite OAK evidence: {oak_video}")
    return oak_video


def write_pv_control_contract(args, controller: WebcamEEController) -> None:
    """Persist the exact live mapping without changing historical mapping names."""
    if not bool(getattr(args, "pv_pressure", False)):
        return
    path = Path(args.pv_evidence_dir) / "control_contract.json"
    contract = {
        "schema_version": 1,
        "control_rate_hz": 30,
        "pv_mapping_contract": controller.mapping_contract,
        "wrist_roll_range_deg": float(args.wrist_roll_range_deg),
        "wrist_roll_gain": float(args.wrist_roll_gain),
    }
    with path.open("x", encoding="utf-8") as handle:
        json.dump(contract, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_lepton_pressure_source(*, port: int):
    """Blob-mode estimator fed by the Pi Lepton UDP streamer (no projection calibration)."""
    try:
        with ExitStack() as cleanup:
            thermal = LeptonUDPSource(port=port)
            close_thermal = getattr(thermal, "close", None)
            if callable(close_thermal):
                cleanup.callback(close_thermal)
            latest_thermal = LatestFrameSource(thermal)
            close_latest = getattr(latest_thermal, "close", None)
            if callable(close_latest):
                cleanup.callback(close_latest)
            estimator = HandPressureEstimator(
                calibration=None,
                thermal_source=latest_thermal,
                config=lepton_pressure_config(),
            )
            cleanup.pop_all()
    except Exception as exc:
        print(f"[ir-pressure] disabled: failed to open Lepton udp:{port}: {exc}")
        return None
    print(f"[ir-pressure] enabled: Lepton udp:{port}, blob ROI mode (no projection calibration)")
    return estimator


def build_pv_pressure_source(*, enabled: bool, port: int):
    """PressureVision source fed by the serve_pad_pressure.py sender over localhost UDP.

    No calibration path here: the sender owns the fitted level boundaries and ships
    a level, so there is one copy of the decision rule rather than a duplicate that
    can drift out of step with the session it was fitted on.
    """
    if not enabled:
        return None
    try:
        with ExitStack() as cleanup:
            udp = PressureVisionUDPSource(port=port)
            cleanup.callback(udp.close)
            latest = LatestFrameSource(udp)
            cleanup.callback(latest.close)
            source = PressureVisionSource(source=latest)
            cleanup.pop_all()
    except Exception as exc:
        print(f"[pv-pressure] disabled: failed to bind udp:{port}: {exc}")
        return None
    print(
        f"[pv-pressure] enabled: udp:{port} "
        "(start hand-pressure/scripts/serve_pad_pressure.py if not already running)"
    )
    return source


def build_ir_pressure_source(
    *, enabled: bool, calibration_path: str, thermal_path: str, lepton_port: int | None = None
):
    if not enabled:
        return None
    if lepton_port is not None:
        return build_lepton_pressure_source(port=lepton_port)
    path = Path(calibration_path)
    if not path.exists():
        print(f"[ir-pressure] disabled: calibration file not found: {path}")
        return None
    try:
        calibration = load_projection_calibration(path)
    except Exception as exc:
        print(f"[ir-pressure] disabled: failed to load calibration {path}: {exc}")
        return None
    try:
        validate_projection_calibration(
            calibration,
            min_samples=MIN_IR_CALIBRATION_SAMPLES,
            max_rms_error_px=MAX_IR_CALIBRATION_RMS_PX,
            max_error_px=MAX_IR_CALIBRATION_ERROR_PX,
            expected_image_size=(160, 128),
        )
    except ValueError as exc:
        print(f"[ir-pressure] disabled: calibration rejected: {exc}")
        return None
    try:
        with ExitStack() as cleanup:
            thermal = OpenCVCameraSource(thermal_path)
            close_thermal = getattr(thermal, "close", None)
            if callable(close_thermal):
                cleanup.callback(close_thermal)
            latest_thermal = LatestFrameSource(thermal)
            close_latest = getattr(latest_thermal, "close", None)
            if callable(close_latest):
                cleanup.callback(close_latest)
            estimator = HandPressureEstimator(
                calibration=calibration,
                thermal_source=latest_thermal,
            )
            cleanup.pop_all()
    except Exception as exc:
        print(f"[ir-pressure] disabled: failed to open thermal source {thermal_path}: {exc}")
        return None
    print(
        f"[ir-pressure] enabled: {thermal_path}, rms={calibration.rms_error_px:.2f}px, "
        f"samples={calibration.sample_count}"
    )
    return estimator


def parse_live_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--oak", action="store_true", help="Use OAK-D RGB/depth hand tracking.")
    parser.add_argument(
        "--wrist-roll-range-deg",
        type=float,
        default=0.0,
        help="Enable relative tool-axis wrist roll within +/- this many degrees. "
        "Zero keeps the validated fixed-down orientation.",
    )
    parser.add_argument(
        "--wrist-roll-gain",
        type=float,
        default=1.0,
        help="Multiply relative hand roll before applying the bounded tool-axis range.",
    )
    parser.add_argument(
        "--grip-mode",
        choices=("tracked", "latched"),
        default="tracked",
        help="Grip arm for the comparison. 'tracked' (default) is the validated "
        "path: the command follows pinch the whole grasp, with a slow-open EMA "
        "damping the loosening. 'latched' ratchets instead -- once committed, "
        "closing still tracks but loosening is blocked until a deliberate open. "
        "Leave this at 'tracked' unless running the comparison.",
    )
    parser.add_argument(
        "--grip-map",
        choices=("overdrive", "span"),
        default="overdrive",
        help="Pinch-to-command map. 'overdrive' (default) is the validated one; "
        "it clips the bottom of the pinch range, so a partial loosening near a "
        "firm grip cannot be expressed. 'span' narrows the range instead and "
        "stays monotone to full closure, at the cost of needing a tighter pinch "
        "for a full clamp. The recorded datasets and trained policies assume "
        "'overdrive' -- do not record with 'span' and deploy against them.",
    )
    pressure_mode = parser.add_mutually_exclusive_group()
    pressure_mode.add_argument(
        "--ir-pressure",
        action="store_true",
        help="Apply calibrated FLIR hand pressure overdrive.",
    )
    pressure_mode.add_argument(
        "--ir-pressure-shadow",
        action="store_true",
        help="Compute calibrated FLIR pressure proposals without changing robot commands.",
    )
    pressure_mode.add_argument(
        "--pv-pressure",
        action="store_true",
        help="Apply PressureVision gripper control using soft_direct, soft_precise, "
        "carton_span, hard_profile, or legacy relative mapping.",
    )
    pressure_mode.add_argument(
        "--pv-pressure-shadow",
        action="store_true",
        help="Compute PressureVision pressure proposals without changing robot commands. "
        "Needs hand-pressure/scripts/serve_pad_pressure.py running on .venv-pressurevision.",
    )
    parser.add_argument("--ir-sidecar", help="Write IR pressure telemetry to this CSV path.")
    parser.add_argument("--pv-sidecar", help="Write PressureVision telemetry to this CSV path.")
    parser.add_argument(
        "--pv-evidence-dir",
        type=Path,
        help="PV apply evidence directory; the Creative recording must already be live.",
    )
    parser.add_argument("--pv-max-load", type=float)
    parser.add_argument("--pv-max-current", type=float)
    parser.add_argument("--pv-max-position-lag", type=float)
    parser.add_argument("--pv-port", type=int, default=DEFAULT_PV_PORT)
    parser.add_argument(
        "--pv-preview-share",
        type=Path,
        default=DEFAULT_PV_PREVIEW_SHARE,
        help="Local mmap panel published by serve_pad_pressure.py.",
    )
    parser.add_argument(
        "--pv-object-profile",
        help="Validated labeled rigid-object profile (required by hard_profile/relative/absolute; "
        "omitted by soft_direct/soft_precise/carton_span).",
    )
    parser.add_argument(
        "--pv-mapping",
        choices=(
            "absolute",
            "relative",
            "soft_direct",
            "soft_precise",
            "carton_span",
            "hard_profile",
        ),
        default="absolute",
        help="PV proposal mapping. soft_direct maps 0..1 to gripper 100..0 with no "
        "object profile; soft_precise maps 0..1 to the exploratory carton range "
        "28..22; carton_span maps 0..1 to the 250 g pilot span 32..20; "
        "hard_profile maps 0..1 to the selected label's light..hard "
        "positions. relative is retained only for prior-trial compatibility.",
    )
    parser.add_argument(
        "--pv-trial-protocol",
        type=int,
        default=0,
        metavar="REPETITIONS",
        help="Overlay/log the open-light-open-hard guided protocol this many times (0 disables).",
    )
    parser.add_argument("--arm-port", default=ARM_PORT)
    parser.add_argument("--arm-id", default=ARM_ID)
    parser.add_argument("--ir-calibration", default=DEFAULT_IR_CALIBRATION)
    parser.add_argument("--thermal", default=DEFAULT_THERMAL_PATH)
    parser.add_argument(
        "--ir-lepton-port",
        type=int,
        nargs="?",
        const=8080,
        default=None,
        metavar="PORT",
        help="Use the Pi Lepton UDP streamer + blob ROI (no OAK depth or projection calibration).",
    )
    args = parser.parse_args(argv)
    if args.ir_pressure:
        parser.error("--ir-pressure is disabled until Stage 3 physical authorization")
    if args.pv_pressure and args.pv_mapping not in PV_CONTROL_MAPPINGS:
        parser.error(
            "--pv-pressure requires --pv-mapping soft_direct, soft_precise, "
            "carton_span, hard_profile, or relative"
        )
    requested = bool(args.ir_pressure or args.ir_pressure_shadow)
    if args.ir_sidecar and not requested:
        parser.error("--ir-sidecar requires --ir-pressure or --ir-pressure-shadow")
    # Projection ROI needs OAK depth; blob ROI (Lepton) reads the hottest patch directly,
    # so it drops the OAK requirement.
    if requested and not args.oak and args.ir_lepton_port is None:
        parser.error("--ir-pressure and --ir-pressure-shadow require --oak (or --ir-lepton-port)")
    # PressureVision watches the pad from its own camera in a separate process, so it
    # needs neither OAK depth nor the thermal rig.
    pv_requested = bool(args.pv_pressure or args.pv_pressure_shadow)
    if args.pv_sidecar and not pv_requested:
        parser.error("--pv-sidecar requires --pv-pressure or --pv-pressure-shadow")
    if pv_requested and args.pv_mapping in PV_PROFILE_MAPPINGS and not args.pv_object_profile:
        parser.error(f"--pv-mapping {args.pv_mapping} requires --pv-object-profile")
    if args.pv_mapping in ("soft_direct", "soft_precise", "carton_span") and args.pv_object_profile:
        parser.error(f"--pv-mapping {args.pv_mapping} does not use --pv-object-profile")
    if args.pv_mapping in PV_CONTROL_MAPPINGS and not pv_requested:
        parser.error(
            f"--pv-mapping {args.pv_mapping} requires --pv-pressure or --pv-pressure-shadow"
        )
    if args.pv_mapping in PV_CONTROL_MAPPINGS and not args.pv_sidecar:
        parser.error(f"--pv-mapping {args.pv_mapping} requires --pv-sidecar motor evidence")
    if args.pv_trial_protocol and args.pv_mapping in (
        "soft_direct",
        "soft_precise",
        "carton_span",
    ):
        parser.error(f"--pv-trial-protocol is not used by {args.pv_mapping}")
    if args.pv_trial_protocol < 0:
        parser.error("--pv-trial-protocol must not be negative")
    if args.pv_port <= 0:
        parser.error("--pv-port must be positive")
    if not 0.0 <= args.wrist_roll_range_deg <= MAX_WRIST_ROLL_RANGE_DEG:
        parser.error(
            f"--wrist-roll-range-deg must be within 0..{MAX_WRIST_ROLL_RANGE_DEG:g}"
        )
    if not 0.0 < args.wrist_roll_gain <= 4.0:
        parser.error("--wrist-roll-gain must be within (0, 4]")
    closure_limit_values = (
        args.pv_max_load,
        args.pv_max_current,
        args.pv_max_position_lag,
    )
    if args.pv_pressure:
        if not args.oak:
            parser.error("--pv-pressure requires --oak so the hand-control stream can be recorded")
        if args.pv_evidence_dir is None:
            parser.error("--pv-pressure requires --pv-evidence-dir")
    if any(value is not None for value in closure_limit_values) and any(
        value is None for value in closure_limit_values
    ):
        parser.error("PV closure limits must be provided together or omitted together")
    if any(value is not None and value <= 0.0 for value in closure_limit_values):
        parser.error("PV closure limits must be positive")
    return args


def build_live_ir_runtime(args) -> LiveIRRuntime:
    """Build whichever pressure source was asked for. IR and PV are mutually exclusive."""
    ir_requested = bool(args.ir_pressure or args.ir_pressure_shadow)
    pv_requested = bool(args.pv_pressure or args.pv_pressure_shadow)
    profile = None
    profile_hash = None
    trial_protocol = None
    if pv_requested and args.pv_mapping in PV_PROFILE_MAPPINGS:
        profile_path = getattr(args, "pv_object_profile", None)
        if not profile_path:
            raise RuntimeError(f"{args.pv_mapping} requires a validated object profile")
        try:
            profile = load_object_profile(profile_path)
            arm_id = getattr(args, "arm_id", ARM_ID)
            if profile.arm_id != arm_id:
                raise ValueError(
                    f"profile arm_id {profile.arm_id!r} does not match requested arm {arm_id!r}"
                )
            profile_hash = object_profile_sha256(profile_path)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"invalid PressureVision object profile: {exc}") from exc
    if pv_requested:
        repetitions = int(getattr(args, "pv_trial_protocol", 0))
        if repetitions:
            trial_protocol = default_trial_protocol(repetitions)
    with ExitStack() as cleanup:
        if pv_requested:
            pressure_source = build_pv_pressure_source(enabled=True, port=args.pv_port)
            label, sidecar_path = "PressureVision", args.pv_sidecar
        else:
            pressure_source = build_ir_pressure_source(
                enabled=ir_requested,
                calibration_path=args.ir_calibration,
                thermal_path=args.thermal,
                lepton_port=args.ir_lepton_port,
            )
            label, sidecar_path = "IR", args.ir_sidecar
        close_pressure = getattr(pressure_source, "close", None)
        if callable(close_pressure):
            cleanup.callback(close_pressure)
        if (ir_requested or pv_requested) and pressure_source is None:
            mode = "shadow" if (args.ir_pressure_shadow or args.pv_pressure_shadow) else "apply"
            raise RuntimeError(f"{label} pressure {mode} mode requested but could not be constructed")
        if sidecar_path and pv_requested:
            sidecar = IRShadowTelemetryLogger(sidecar_path, extra_fields=PV_SHADOW_FIELDS)
        else:
            sidecar = IRShadowTelemetryLogger(sidecar_path) if sidecar_path else None
        runtime = LiveIRRuntime(
            pressure_source=pressure_source,
            pressure_shadow=bool(args.ir_pressure_shadow or args.pv_pressure_shadow),
            sidecar=sidecar,
            pressure_apply=bool(args.pv_pressure),
            object_profile=profile,
            object_profile_sha256=profile_hash,
            trial_protocol=trial_protocol,
            pv_mapping=getattr(args, "pv_mapping", "absolute"),
            gripper_closure_limits=(
                GripperClosureLimits(
                    args.pv_max_load,
                    args.pv_max_current,
                    args.pv_max_position_lag,
                )
                if all(value is not None for value in (
                    args.pv_max_load,
                    args.pv_max_current,
                    args.pv_max_position_lag,
                ))
                else None
            ),
        )
        cleanup.pop_all()
        return runtime


def send_live_action(
    robot,
    joints,
    *,
    sidecar,
    telemetry_sample,
    motor_sampler=None,
) -> bool:
    command_sent = False
    motor_telemetry = None
    try:
        if joints is not None:
            robot.send_action(joints)
            command_sent = True
        return command_sent
    finally:
        if motor_sampler is not None:
            motor_telemetry = motor_sampler.poll(robot)
        if sidecar is not None:
            sidecar.finalize(
                telemetry_sample,
                command_sent=command_sent,
                motor_telemetry=motor_telemetry,
            )


def close_live_resources(release_cam, hands, controller, robot, sidecar=None) -> None:
    with ExitStack() as stack:
        if sidecar is not None:
            stack.callback(sidecar.close)
        stack.callback(robot.disconnect)
        stack.callback(cv2.destroyAllWindows)
        stack.callback(controller.close)
        stack.callback(hands.close)
        stack.callback(release_cam)


def disconnect_robot_safely(robot) -> None:
    """Close fully or partially connected LeRobot resources without masking failures."""
    try:
        if getattr(robot, "is_connected", False):
            robot.disconnect()
    except Exception as exc:
        print(f"[cleanup] robot disconnect failed: {exc}")

    for camera in getattr(robot, "cameras", {}).values():
        try:
            if getattr(camera, "is_connected", False):
                camera.disconnect()
        except Exception as exc:
            print(f"[cleanup] camera disconnect failed: {exc}")

    bus = getattr(robot, "bus", None)
    try:
        if bus is not None and getattr(bus, "is_connected", False):
            disable_torque = getattr(
                getattr(robot, "config", None),
                "disable_torque_on_disconnect",
                False,
            )
            bus.disconnect(disable_torque=disable_torque)
    except Exception as exc:
        print(f"[cleanup] robot bus disconnect failed: {exc}")


def read_positions(robot, tries=12):
    """Read joint positions, retrying through intermittent serial read drops."""
    for _ in range(tries):
        try:
            obs = robot.get_observation()
            return {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
        except ConnectionError:
            time.sleep(0.1)
    raise ConnectionError("Arm position read kept failing -- check the USB cable/port "
                          "(try a direct port instead of the shared hub).")


def _run_live(args, resources: ExitStack) -> None:
    use_oak = args.oak
    oak_video_path = validate_pv_apply_evidence_gate(args)
    oak_video_writer = None
    if oak_video_path is not None:
        oak_video_writer = cv2.VideoWriter(
            str(oak_video_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            30.0,
            (640, 480),
        )
        if not oak_video_writer.isOpened():
            raise RuntimeError(f"could not open OAK evidence writer: {oak_video_path}")
        resources.callback(oak_video_writer.release)
    cfg = SO101WebcamEEConfig(camera_index=CAMERA_INDEX)

    robot = SOFollower(SO101FollowerConfig(
        port=getattr(args, "arm_port", ARM_PORT),
        id=getattr(args, "arm_id", ARM_ID),
        use_degrees=True,
        max_relative_target=MAX_RELATIVE_TARGET, cameras={},
        disable_torque_on_disconnect=False,  # hold pose on exit instead of collapsing
    ))
    resources.callback(disconnect_robot_safely, robot)
    for attempt in range(3):
        try:
            robot.connect(calibrate=False)
            break
        except Exception as e:
            if attempt == 2:
                raise
            # connect() opens the bus first; a hiccup after that leaves the port open and the
            # guarded robot.disconnect() won't close it. Close the bus directly before retrying.
            print(f"[connect] attempt {attempt + 1} failed: {e!r} -- cleaning up, retrying")
            try:
                if robot.bus.is_connected:
                    robot.bus.disconnect(disable_torque=False)
            except Exception:
                pass
            time.sleep(1.0)
    # LeRobot default servo PID is good enough (verified) -- no tuned-PID re-apply.

    motors = list(robot.bus.motors.keys())
    kin = RobotKinematics(urdf_path=URDF_PATH, target_frame_name="gripper_frame_link", joint_names=motors)
    ir_runtime = build_live_ir_runtime(args)
    if ir_runtime.sidecar is not None:
        resources.callback(ir_runtime.sidecar.close)
    close_pressure = getattr(ir_runtime.pressure_source, "close", None)
    if callable(close_pressure):
        resources.callback(close_pressure)
    controller = WebcamEEController(
        robot,
        kin,
        cfg,
        use_oak=use_oak,
        pressure_source=ir_runtime.pressure_source,
        pressure_shadow=ir_runtime.pressure_shadow,
        pressure_apply=ir_runtime.pressure_apply,
        object_profile=ir_runtime.object_profile,
        object_profile_sha256=ir_runtime.object_profile_sha256,
        trial_protocol=ir_runtime.trial_protocol,
        pv_mapping=ir_runtime.pv_mapping,
        gripper_closure_limits=ir_runtime.gripper_closure_limits,
        grip_mode=args.grip_mode,
        grip_map=args.grip_map,
        wrist_roll_range_deg=getattr(args, "wrist_roll_range_deg", 0.0),
        wrist_roll_gain=getattr(args, "wrist_roll_gain", 1.0),
    )
    resources.callback(controller.close)
    write_pv_control_contract(args, controller)

    # Open the normal operator window before gating the first commanded motion.
    if use_oak:
        from webcam_input.depth import OAKDepthStrategy
        from webcam_input.oak_camera import OAKCamera
        oak_depth = OAKDepthStrategy(radius_px=6, ema_alpha=0.4)
        depth_strategy = oak_depth
        cam = OAKCamera(rgb_size=(640, 480), fps=30)
        resources.callback(cam.stop)
        cam.start()

        def read_frame():
            rgb, depth = cam.read()
            oak_depth.update_depth(depth)
            return rgb
    else:
        depth_strategy = ScaleDepthStrategy()
        cap = cv2.VideoCapture(CAMERA_INDEX)
        resources.callback(cap.release)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {CAMERA_INDEX} (another app using it?)")

        def read_frame():
            ok, frame = cap.read()
            return frame if ok else None

    src = WebcamSource(WebcamWristEstimator(depth_strategy, workspace_size_m=cfg.workspace_size_m))
    print(f"Camera source: {'OAK-D (stereo depth)' if use_oak else 'laptop webcam (monocular)'}")
    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.8, min_tracking_confidence=0.8,
    )
    resources.callback(hands.close)
    draw = mp.solutions.drawing_utils
    pv_preview = None
    if getattr(args, "pv_pressure", False) or getattr(args, "pv_pressure_shadow", False):
        pv_preview = PressureVisionPreviewSource(
            getattr(args, "pv_preview_share", DEFAULT_PV_PREVIEW_SHARE)
        )
        resources.callback(pv_preview.close)

    win = "so101_webcam_ee teleop (q to quit)"
    resources.callback(cv2.destroyAllWindows)
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1600 if pv_preview is not None else 1280, 600 if pv_preview is not None else 960)
    frame_id = 0
    startup_gate = ContinuousHandStartupGate()
    print("ARM LOCKED: keep the right hand continuously visible for 3.0 s to enable startup motion.")
    while True:
        frame = read_frame()
        observed_at_s = time.perf_counter()
        if frame is None:
            startup_gate.update(hand_valid=False, observed_at_s=observed_at_s)
            continue
        if oak_video_writer is not None:
            oak_video_writer.write(frame)
        src.image_shape = frame.shape[:2]
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        right, left = WebcamSource.split_results(results)
        wrist, landmarks = src.process_hands(
            right,
            left,
            observed_at_s=observed_at_s,
            frame_id=frame_id,
        )
        frame_id += 1
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                draw.draw_landmarks(frame, hand_lms, mp.solutions.hands.HAND_CONNECTIONS)
        elapsed_s = startup_gate.update(
            hand_valid=bool(wrist.valid and landmarks.valid),
            observed_at_s=observed_at_s,
        )
        cv2.putText(
            frame,
            f"ARM LOCKED: right hand {min(elapsed_s, HAND_STARTUP_DWELL_S):.1f}/3.0 s",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
        )
        pv_frame = None if pv_preview is None else pv_preview.read()
        cv2.imshow(win, compose_operator_view(frame, pv_frame))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Startup cancelled before the first robot motion.")
            return
        if elapsed_s >= HAND_STARTUP_DWELL_S:
            print("ARM ENABLED: continuous right-hand detection reached 3.0 s.")
            break

    # Ramp gently to the down ready pose (repeat each step so slow joints can follow), then build
    # the workspace box around the resulting EE pose and seed the controller's open-loop state.
    start = read_positions(robot)
    for a in np.linspace(0.0, 1.0, 30):
        cmd = {k: (1 - a) * start[k] + a * controller.middle_pose[k] for k in controller.middle_pose}
        for _ in range(3):
            robot.send_action(cmd)
            time.sleep(0.04)
    for _ in range(50):
        robot.send_action(dict(controller.middle_pose))
        time.sleep(0.04)

    obs0 = read_positions(robot)
    ee_centre = kin.forward_kinematics(np.array([obs0[f"{m}.pos"] for m in motors], float))[:3, 3]
    controller.build(ee_centre)
    controller.seed(obs0)
    motor_sampler = (
        GripperTelemetrySampler(interval_s=0.2)
        if ir_runtime.sidecar is not None
        and (
            ir_runtime.object_profile is not None
            or ir_runtime.pv_mapping in PV_CONTROL_MAPPINGS
        )
        else None
    )
    if motor_sampler is not None:
        controller.set_gripper_telemetry(motor_sampler.poll(robot, force=True))
    print(f"EE centre (ready FK): {np.round(ee_centre, 3)}  down rotvec: {np.round(controller.r_down, 3)}")

    comm_failures = 0
    joint_act = None
    print("Right hand moves the gripper; hold RIGHT V-sign 0.4s=MIDDLE; lost right hand=HOLD; "
          "left hand is ignored; pinch=gripper. q to quit.")
    while True:
        frame = read_frame()
        # Host read-completion observation; this is not the camera exposure timestamp.
        observed_at_s = time.perf_counter()
        if frame is None:
            continue
        if oak_video_writer is not None:
            oak_video_writer.write(frame)
        src.image_shape = frame.shape[:2]
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        right, left = WebcamSource.split_results(results)
        wrist, landmarks = src.process_hands(
            right,
            left,
            observed_at_s=observed_at_s,
            frame_id=frame_id,
        )
        frame_id += 1

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                draw.draw_landmarks(frame, hand_lms, mp.solutions.hands.HAND_CONNECTIONS)

        # All control (down orientation, slew, EMA, right-V/HOLD) lives in the shared controller.
        joints, state = controller.step(wrist, landmarks)
        try:
            if send_live_action(
                robot,
                joints,
                sidecar=ir_runtime.sidecar,
                telemetry_sample=controller.last_ir_shadow_telemetry,
                motor_sampler=motor_sampler,
            ):
                joint_act = joints
                comm_failures = 0
            if motor_sampler is not None:
                controller.set_gripper_telemetry(motor_sampler.latest)
            # else HOLD: send nothing; arm holds its last commanded pose.
        except ConnectionError as e:
            comm_failures += 1
            print(f"[serial] write {comm_failures}/{MAX_COMM_FAILURES}: {e}")
            if comm_failures >= MAX_COMM_FAILURES:
                print("[serial] too many consecutive write failures -- check arm/USB. Stopping.")
                break

        # --- overlay ---
        lm = np.asarray(landmarks.landmarks, dtype=float)
        pinch = float(np.linalg.norm(lm[_THUMB_TIP] - lm[_INDEX_TIP]))
        color = {"MOVING": (0, 200, 0), "MIDDLE": (0, 165, 255), "HOLD": (0, 0, 255)}[state]
        pressure = controller.last_pressure
        pressure_line = "pressure: off"
        if pressure is not None:
            sensor = "PV" if getattr(pressure, "roi_mode", None) == "pv" else "IR"
            pressure_line = (
                f"{sensor} pressure: {pressure.status} active={int(pressure.active)} "
                f"p={pressure.pressure_0_1:.2f} q={pressure.quality:.2f}"
            )
        middle_gesture = (
            "ACTIVE" if controller.middle_gesture_active
            else "PENDING" if controller.middle_gesture_seen
            else "off"
        )
        lines = [
            f"right_valid={wrist.valid}  middle(right_V_0.4s)={middle_gesture}",
            f"CONTROL: {state}  gripper={gripper_pos_from_pinch(pinch, cfg) if state == 'MOVING' else 0.0:5.1f}",
            pressure_line,
        ]
        relative = controller.last_relative_grip
        if relative is not None:
            track_state = (
                "WAIT" if relative.track_hold is None else relative.track_hold.state
            )
            reference = "--" if relative.reference_pos is None else f"{relative.reference_pos:.2f}"
            target = "--" if relative.target_pos is None else f"{relative.target_pos:.2f}"
            actual = (
                "--"
                if controller.last_pressure_control is None
                else f"{controller.last_pressure_control.actual_gripper:.2f}"
            )
            mode = "APPLY" if controller.pressure_apply else "SHADOW"
            lines.append(
                f"PV {controller.pv_mapping.upper()} {mode}: "
                f"{relative.status}/{track_state} ref={reference} "
                f"target={target} sent={actual}"
            )
        if controller.trial_protocol is not None:
            expected = controller.trial_protocol.expected(time.perf_counter())
            if expected is None:
                lines.append("PV protocol: complete")
            else:
                lines.append(
                    f"PV trial {int(expected['trial_index']) + 1}: {expected['trial_phase']} "
                    f"(level {int(expected['expected_level'])})"
                )
        if joint_act is not None:
            lines += [f"{m:>13}={joint_act.get(m + '.pos', float('nan')):+7.1f}" for m in motors]
        for i, line in enumerate(lines):
            line_color = (0, 255, 255) if line.startswith("PV ") else (
                color if i < 2 else (255, 255, 255)
            )
            cv2.putText(frame, line, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, line_color, 1, cv2.LINE_AA)

        operator_view = (
            frame
            if pv_preview is None
            else compose_operator_view(frame, pv_preview.read())
        )
        measured_gripper = (
            None
            if motor_sampler is None or motor_sampler.latest is None
            else motor_sampler.latest.observed_gripper_pos
        )
        draw_gripper_position_banner(
            operator_view,
            commanded=(
                None if joint_act is None else joint_act.get("gripper.pos")
            ),
            observed=measured_gripper,
        )
        cv2.imshow(win, operator_view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main():
    args = parse_live_args()
    with ExitStack() as resources:
        _run_live(args, resources)
    print("Disconnected.")


if __name__ == "__main__":
    main()
