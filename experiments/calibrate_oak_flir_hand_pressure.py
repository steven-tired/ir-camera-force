from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

_CHECKOUT_ROOT = Path(__file__).resolve().parents[1]
if (_CHECKOUT_ROOT / "webcam_input").is_dir():
    sys.path.insert(0, str(_CHECKOUT_ROOT))

import cv2
import numpy as np

from ir_force.ir_capture import LeptonUDPSource, OpenCVCameraSource
from ir_force.ir_hand_calibration import (
    ProjectionSample,
    fit_projection,
    load_projection_calibration,
    save_projection_calibration,
    validate_projection_calibration,
)
from ir_force.ir_pressure import timing_limit_exceeded
from webcam_input.depth import OAKDepthStrategy, ScaleDepthStrategy
from ir_force.realsense_camera import RealSenseCamera
from ir_force.types import WebcamSample
from webcam_input.webcam_source import WebcamSource
from webcam_input.wrist_estimator import WebcamWristEstimator

THUMB_TIP = 4
PALM_DEPTH_LANDMARKS = (0, 1, 2, 5, 9, 13, 17)
DEFAULT_CALIBRATION_DIR = _CHECKOUT_ROOT / "lerobot_teleoperator_so101_webcam" / "calibration"
COORDINATE_FIELDS = ("oak_x", "oak_y", "oak_z", "ir_x", "ir_y")
TIMING_FIELDS = ("oak_observed_at_s", "thermal_observed_at_s", "sensor_skew_s")
FIELDNAMES = COORDINATE_FIELDS + TIMING_FIELDS
# Match the runtime estimator's robust p95-p5 thermal intensity range.
MIN_WARM_BLOB_FRAME_RANGE = 2.0
LEPTON_IMAGE_SIZE = (160, 120)
FLIR_ONE_IMAGE_SIZE = (160, 128)
MAX_REPROJECTION_ERROR_PX = 3.0


@dataclass(frozen=True)
class TimedProjectionSample:
    sample: ProjectionSample
    oak_observed_at_s: float
    thermal_observed_at_s: float
    sensor_skew_s: float


@dataclass(frozen=True)
class CalibrationTarget:
    label: str
    hand_label: str
    image_size: tuple[int, int]
    samples_filename: str
    projection_filename: str
    error_report_filename: str


def calibration_target(args) -> CalibrationTarget:
    hand_camera = getattr(args, "hand_camera", "realsense")
    hand_label = "RealSense" if hand_camera == "realsense" else "OAK"
    hand_slug = hand_label.lower()
    if args.lepton_udp is not None:
        return CalibrationTarget(
            label="Lepton",
            hand_label=hand_label,
            image_size=LEPTON_IMAGE_SIZE,
            samples_filename=f"{hand_slug}_lepton_hand_pressure_samples.csv",
            projection_filename=f"{hand_slug}_lepton_hand_pressure_projection.json",
            error_report_filename=f"{hand_slug}_lepton_hand_pressure_error_report.json",
        )
    return CalibrationTarget(
        label="FLIR",
        hand_label=hand_label,
        image_size=FLIR_ONE_IMAGE_SIZE,
        samples_filename=f"{hand_slug}_flir_hand_pressure_samples.csv",
        projection_filename=f"{hand_slug}_flir_hand_pressure_projection.json",
        error_report_filename=f"{hand_slug}_flir_hand_pressure_error_report.json",
    )


def validate_gate_configuration(args, target: CalibrationTarget) -> None:
    if target.label != "Lepton":
        return
    thresholds = (float(args.max_rms_px), float(args.max_error_px))
    if any(value != MAX_REPROJECTION_ERROR_PX for value in thresholds):
        raise ValueError(
            "Lepton reprojection gate is locked at 3.00 px for both RMS and maximum error"
        )


def build_thermal_source(args):
    if args.lepton_udp is not None:
        return LeptonUDPSource(port=args.lepton_udp)
    return OpenCVCameraSource(args.thermal)


def select_calibration_hand(right, left):
    return right if right is not None else left


class RealSenseHandSource:
    """Synchronous aligned RealSense capture using the shared MediaPipe hand path."""

    def __init__(self, *, serial: str | None = None):
        self._camera = RealSenseCamera(serial=serial, width=640, height=480, fps=30)
        self._depth = OAKDepthStrategy(radius_px=6, ema_alpha=0.4)
        self._processor = WebcamSource(
            WebcamWristEstimator(self._depth),
            image_shape=(480, 640),
        )
        self._hands = None
        self._draw = None
        self._connections = None
        self._frame_id = 0

    def start(self) -> None:
        import mediapipe as mp

        self._camera.start()
        try:
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.8,
                min_tracking_confidence=0.8,
            )
        except Exception:
            self._camera.stop()
            raise
        self._draw = mp.solutions.drawing_utils
        self._connections = mp.solutions.hands.HAND_CONNECTIONS

    def latest_sample(self) -> WebcamSample:
        if self._hands is None:
            raise RuntimeError("RealSense hand source is not started")
        color_bgr, depth_mm, observed_at_s = self._camera.read()
        self._depth.update_depth(depth_mm)
        self._processor.image_shape = color_bgr.shape[:2]
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        right, left = self._processor.split_results(results)
        calibration_hand = select_calibration_hand(right, left)
        frame_id = self._frame_id
        self._frame_id += 1
        wrist, landmarks = self._processor.process_hands(
            right=calibration_hand,
            left=None,
            observed_at_s=observed_at_s,
            frame_id=frame_id,
        )
        annotated = color_bgr.copy()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self._draw.draw_landmarks(
                    annotated,
                    hand_landmarks,
                    self._connections,
                )
        return WebcamSample(
            preview_frame=annotated,
            wrist=wrist,
            landmarks=landmarks,
            observed_at_s=observed_at_s,
            frame_id=frame_id,
        )

    def stop(self) -> None:
        failures = []
        hands = self._hands
        self._hands = None
        if hands is not None:
            try:
                hands.close()
            except Exception as exc:
                failures.append(("MediaPipe hands", exc))
        try:
            self._camera.stop()
        except Exception as exc:
            failures.append(("RealSense camera", exc))
        if failures:
            detail = "; ".join(f"{label}: {exc}" for label, exc in failures)
            raise RuntimeError(f"RealSense hand source cleanup failed: {detail}") from failures[0][1]


def build_hand_source(args):
    if args.hand_camera == "realsense":
        return RealSenseHandSource(serial=args.realsense_serial)
    return WebcamSource(WebcamWristEstimator(ScaleDepthStrategy()))


def start_hand_source(source, args) -> None:
    if args.hand_camera == "realsense":
        source.start()
    else:
        source.start_oak()


def read_calibration_pair(hand_source, thermal_source):
    """Read the slower thermal stream first, then capture a fresh hand/depth frame."""
    thermal_sample = thermal_source.read()
    hand_sample = hand_source.latest_sample()
    return hand_sample, thermal_sample


def find_warm_blob_center(
    frame: np.ndarray,
    *,
    percentile: float = 97.0,
    min_area_px: int = 4,
    min_frame_range: float = MIN_WARM_BLOB_FRAME_RANGE,
) -> tuple[float, float] | None:
    thermal = np.asarray(frame)
    if thermal.size == 0:
        return None
    try:
        if not np.all(np.isfinite(thermal)):
            return None
    except TypeError:
        return None
    if thermal.ndim == 3:
        gray = cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY)
    elif thermal.ndim == 2:
        gray = thermal.astype(np.float32, copy=False)
    else:
        return None
    low, high = np.percentile(gray, (5.0, 95.0))
    if float(high - low) < min_frame_range:
        return None
    threshold = float(np.percentile(gray, percentile))
    mask = (gray > max(threshold, 1.0)).astype(np.uint8)
    if int(mask.sum()) < min_area_px:
        return None
    moments = cv2.moments(mask)
    if moments["m00"] <= 0.0:
        return None
    return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])


def preview_click_to_thermal_point(
    click_xy: tuple[int, int],
    *,
    hand_shape: tuple[int, ...],
    thermal_shape: tuple[int, ...],
) -> tuple[float, float] | None:
    hand_height, hand_width = hand_shape[:2]
    thermal_height, thermal_width = thermal_shape[:2]
    displayed_thermal_width = max(
        1,
        int(round(thermal_width * hand_height / thermal_height)),
    )
    click_x, click_y = click_xy
    thermal_display_x = click_x - hand_width
    if not (
        0 <= thermal_display_x < displayed_thermal_width
        and 0 <= click_y < hand_height
    ):
        return None
    return (
        thermal_display_x * thermal_width / displayed_thermal_width,
        click_y * thermal_height / hand_height,
    )


def build_calibration_preview(
    hand_frame,
    thermal_frame,
    landmarks,
    *,
    thermal_label: str = "FLIR",
    hand_label: str = "OAK",
    selected_ir_point: tuple[float, float] | None = None,
) -> np.ndarray:
    thermal_raw = np.asarray(thermal_frame)
    blob = find_warm_blob_center(thermal_raw)
    thermal_preview = thermal_raw.copy()
    if thermal_preview.ndim == 2:
        if thermal_preview.dtype != np.uint8:
            lo, hi = np.percentile(thermal_preview, (1.0, 99.0))
            if hi <= lo:
                thermal_preview = np.zeros(thermal_preview.shape, dtype=np.uint8)
            else:
                scaled = np.clip(
                    (thermal_preview.astype(np.float32) - lo) / (hi - lo),
                    0.0,
                    1.0,
                )
                thermal_preview = (scaled * 255.0).astype(np.uint8)
        thermal_preview = cv2.cvtColor(thermal_preview, cv2.COLOR_GRAY2BGR)
    if hand_frame is None:
        hand_preview = np.zeros_like(thermal_preview)
    else:
        hand_preview = np.asarray(hand_frame).copy()
        if hand_preview.ndim == 2:
            hand_preview = cv2.cvtColor(hand_preview, cv2.COLOR_GRAY2BGR)

    thumb_valid = (
        landmarks.valid
        and landmarks.image_xy is not None
        and landmarks.depth_m is not None
        and not np.isnan(float(landmarks.depth_m[THUMB_TIP]))
    )
    cv2.putText(
        hand_preview,
        f"{hand_label} thumb/depth: {'valid' if thumb_valid else 'invalid'}",
        (6, min(18, hand_preview.shape[0] - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if thumb_valid:
        thumb_x, thumb_y = landmarks.image_xy[THUMB_TIP]
        point = (
            int(np.clip(float(thumb_x), 0.0, 1.0) * (hand_preview.shape[1] - 1)),
            int(np.clip(float(thumb_y), 0.0, 1.0) * (hand_preview.shape[0] - 1)),
        )
        cv2.circle(hand_preview, point, 5, (0, 255, 0), -1)

    cv2.putText(
        thermal_preview,
        f"{thermal_label} warm blob: {'valid' if blob is not None else 'invalid'}",
        (6, min(18, thermal_preview.shape[0] - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if blob is not None:
        cv2.circle(
            thermal_preview,
            (int(round(blob[0])), int(round(blob[1]))),
            3,
            (0, 255, 255),
            -1,
        )
    if selected_ir_point is not None:
        cv2.drawMarker(
            thermal_preview,
            (int(round(selected_ir_point[0])), int(round(selected_ir_point[1]))),
            (255, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=9,
            thickness=1,
        )

    target_height = hand_preview.shape[0]
    thermal_width = max(
        1,
        int(round(thermal_preview.shape[1] * target_height / thermal_preview.shape[0])),
    )
    thermal_preview = cv2.resize(
        thermal_preview,
        (thermal_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    )
    return np.hstack((hand_preview, thermal_preview))


def _upgrade_coordinate_only_csv(csv_path: Path) -> None:
    if not csv_path.exists():
        return
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        existing_fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    if existing_fields == FIELDNAMES:
        return
    if existing_fields != COORDINATE_FIELDS:
        raise ValueError(f"unsupported calibration CSV fields: {existing_fields}")

    upgraded_path = csv_path.with_name(f"{csv_path.name}.upgrade")
    with upgraded_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in FIELDNAMES})
    upgraded_path.replace(csv_path)


def append_projection_sample(
    csv_path: Path,
    sample: ProjectionSample,
    *,
    oak_observed_at_s: float | None = None,
    thermal_observed_at_s: float | None = None,
    sensor_skew_s: float | None = None,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _upgrade_coordinate_only_csv(csv_path)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                **{name: getattr(sample, name) for name in COORDINATE_FIELDS},
                "oak_observed_at_s": oak_observed_at_s,
                "thermal_observed_at_s": thermal_observed_at_s,
                "sensor_skew_s": sensor_skew_s,
            }
        )


def load_projection_samples(csv_path: Path) -> list[ProjectionSample]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return [
            ProjectionSample(
                oak_x=float(row["oak_x"]),
                oak_y=float(row["oak_y"]),
                oak_z=float(row["oak_z"]),
                ir_x=float(row["ir_x"]),
                ir_y=float(row["ir_y"]),
            )
            for row in csv.DictReader(handle)
        ]


def save_projection_error_report(
    path: Path,
    *,
    calibration,
    samples: list[ProjectionSample],
    hand_label: str,
    thermal_label: str,
    gate_px: float,
    min_hand_depth_m: float | None,
    max_hand_depth_m: float | None,
) -> None:
    residuals = []
    coeff_x = np.asarray(calibration.coeff_x, dtype=float)
    coeff_y = np.asarray(calibration.coeff_y, dtype=float)
    for index, sample in enumerate(samples, start=1):
        hand_xyz = np.asarray([1.0, sample.oak_x, sample.oak_y, sample.oak_z])
        predicted = np.asarray([coeff_x @ hand_xyz, coeff_y @ hand_xyz])
        target = np.asarray([sample.ir_x, sample.ir_y])
        residuals.append(
            {
                "index": index,
                "hand_xyz": [sample.oak_x, sample.oak_y, sample.oak_z],
                "target_ir_xy": target.tolist(),
                "predicted_ir_xy": predicted.tolist(),
                "error_px": float(np.linalg.norm(predicted - target)),
            }
        )

    accepted = bool(
        calibration.rms_error_px <= gate_px and calibration.max_error_px <= gate_px
    )
    payload = {
        "version": 1,
        "source": {"hand": hand_label, "thermal": thermal_label},
        "method": "manual_thermal_point_depth_affine",
        "decision": "GO" if accepted else "ESCALATE",
        "accepted": accepted,
        "sample_count": calibration.sample_count,
        "image_size": list(calibration.image_size),
        "working_depth_m": {"min": min_hand_depth_m, "max": max_hand_depth_m},
        "gate_px": float(gate_px),
        "rms_error_px": calibration.rms_error_px,
        "max_error_px": calibration.max_error_px,
        "coeff_x": list(calibration.coeff_x),
        "coeff_y": list(calibration.coeff_y),
        "samples": residuals,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _finite_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def projection_pair_diagnostics(snapshot, thermal_sample, *, now_s: float) -> str:
    landmarks = snapshot.landmarks
    depth = None
    if landmarks.depth_m is not None and len(landmarks.depth_m) > THUMB_TIP:
        depth = _finite_float(landmarks.depth_m[THUMB_TIP])
    oak_t = _finite_float(getattr(snapshot, "observed_at_s", None))
    thermal_t = _finite_float(getattr(thermal_sample, "t", None))

    def milliseconds_since(timestamp):
        return None if timestamp is None else (now_s - timestamp) * 1000.0

    skew_ms = None
    if oak_t is not None and thermal_t is not None:
        skew_ms = abs(oak_t - thermal_t) * 1000.0
    return (
        f"hand_valid={landmarks.valid} thumb_depth_m={depth} "
        f"hand_age_ms={milliseconds_since(oak_t)} "
        f"thermal_age_ms={milliseconds_since(thermal_t)} skew_ms={skew_ms}"
    )


def calibration_hand_depth_m(
    landmark_depths,
    *,
    min_depth_m: float | None,
    max_depth_m: float | None,
) -> float | None:
    depths = np.asarray(landmark_depths, dtype=float)
    thumb_depth = _finite_float(depths[THUMB_TIP]) if len(depths) > THUMB_TIP else None
    if min_depth_m is None and max_depth_m is None:
        return thumb_depth

    palm_depths = depths[list(PALM_DEPTH_LANDMARKS)]
    valid = palm_depths[np.isfinite(palm_depths)]
    if min_depth_m is not None:
        valid = valid[valid >= min_depth_m]
    if max_depth_m is not None:
        valid = valid[valid <= max_depth_m]
    if valid.size:
        return float(np.median(valid))
    if thumb_depth is None:
        return None
    if min_depth_m is not None and thumb_depth < min_depth_m:
        return None
    if max_depth_m is not None and thumb_depth > max_depth_m:
        return None
    return thumb_depth


def pair_projection_sample(
    snapshot,
    thermal_sample,
    *,
    now_s: float | None = None,
    max_oak_age_s: float = 0.20,
    max_thermal_age_s: float = 0.20,
    max_pair_skew_s: float = 0.15,
    ir_point: tuple[float, float] | None = None,
    min_hand_depth_m: float | None = None,
    max_hand_depth_m: float | None = None,
) -> TimedProjectionSample | None:
    """Pair host read-completion observations; these are not exposure timestamps."""
    now_s = time.perf_counter() if now_s is None else now_s
    now_s = _finite_float(now_s)
    oak_t = _finite_float(getattr(snapshot, "observed_at_s", None))
    thermal_t = _finite_float(getattr(thermal_sample, "t", None))
    if now_s is None or oak_t is None or thermal_t is None:
        return None

    oak_age_s = now_s - oak_t
    thermal_age_s = now_s - thermal_t
    sensor_skew_s = abs(oak_t - thermal_t)
    if (
        oak_age_s < 0.0
        or thermal_age_s < 0.0
        or timing_limit_exceeded(now_s, oak_t, max_oak_age_s)
        or timing_limit_exceeded(now_s, thermal_t, max_thermal_age_s)
        or timing_limit_exceeded(
            max(oak_t, thermal_t),
            min(oak_t, thermal_t),
            max_pair_skew_s,
        )
    ):
        return None

    landmarks = snapshot.landmarks
    if not landmarks.valid or landmarks.image_xy is None or landmarks.depth_m is None:
        return None
    oak_x, oak_y = landmarks.image_xy[THUMB_TIP]
    oak_z = calibration_hand_depth_m(
        landmarks.depth_m,
        min_depth_m=min_hand_depth_m,
        max_depth_m=max_hand_depth_m,
    )
    if oak_z is None or not np.all(np.isfinite([oak_x, oak_y, oak_z])):
        return None
    thermal_height, thermal_width = np.asarray(thermal_sample.frame).shape[:2]
    thermal_point = ir_point
    if thermal_point is None:
        thermal_point = find_warm_blob_center(thermal_sample.frame)
    if thermal_point is None or not np.all(np.isfinite(thermal_point)):
        return None
    if not (
        0.0 <= thermal_point[0] < thermal_width
        and 0.0 <= thermal_point[1] < thermal_height
    ):
        return None
    return TimedProjectionSample(
        sample=ProjectionSample(
            oak_x=float(oak_x),
            oak_y=float(oak_y),
            oak_z=oak_z,
            ir_x=float(thermal_point[0]),
            ir_y=float(thermal_point[1]),
        ),
        oak_observed_at_s=oak_t,
        thermal_observed_at_s=thermal_t,
        sensor_skew_s=sensor_skew_s,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate OAK hand landmarks to thermal pixels.")
    parser.add_argument("--thermal", default="/dev/video21")
    parser.add_argument(
        "--hand-camera",
        choices=("realsense", "oak"),
        default="realsense",
        help="RGB/depth camera used for MediaPipe landmarks (default: realsense)",
    )
    parser.add_argument(
        "--realsense-serial",
        default=None,
        help="optional RealSense serial; the first attached device is used by default",
    )
    parser.add_argument(
        "--lepton-udp",
        type=int,
        nargs="?",
        const=8080,
        default=None,
        metavar="PORT",
        help="read raw Lepton frames from the Pi UDP streamer instead of --thermal",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--min-samples", type=int, default=12)
    parser.add_argument("--max-rms-px", type=float, default=MAX_REPROJECTION_ERROR_PX)
    parser.add_argument("--max-error-px", type=float, default=MAX_REPROJECTION_ERROR_PX)
    parser.add_argument("--max-oak-age-ms", type=float, default=200.0)
    parser.add_argument("--max-thermal-age-ms", type=float, default=200.0)
    parser.add_argument("--max-pair-skew-ms", type=float, default=150.0)
    parser.add_argument("--min-hand-depth-m", type=float, default=0.20)
    parser.add_argument("--max-hand-depth-m", type=float, default=0.90)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    target = calibration_target(args)
    validate_gate_configuration(args, target)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / target.samples_filename
    calibration_path = args.out_dir / target.projection_filename
    error_report_path = args.out_dir / target.error_report_filename

    with ExitStack() as cleanup:
        source = build_hand_source(args)
        cleanup.callback(source.stop)
        thermal = build_thermal_source(args)
        cleanup.callback(thermal.close)
        start_hand_source(source, args)
        cleanup.callback(cv2.destroyAllWindows)
        window = f"{target.hand_label.lower()}-{target.label.lower()} hand pressure calibration"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1280, 480)
        manual_point_required = target.hand_label == "RealSense" and target.label == "Lepton"
        click_state = {"hand_shape": None, "thermal_shape": None, "ir_point": None}

        def select_thermal_point(event, x, y, _flags, _userdata):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            if click_state["hand_shape"] is None or click_state["thermal_shape"] is None:
                return
            point = preview_click_to_thermal_point(
                (x, y),
                hand_shape=click_state["hand_shape"],
                thermal_shape=click_state["thermal_shape"],
            )
            if point is not None:
                click_state["ir_point"] = point
                print(f"[calibration] selected {target.label} point {point}")

        cv2.setMouseCallback(window, select_thermal_point)
        while True:
            snapshot, thermal_sample = read_calibration_pair(source, thermal)
            click_state["hand_shape"] = snapshot.preview_frame.shape
            click_state["thermal_shape"] = thermal_sample.frame.shape
            preview = build_calibration_preview(
                snapshot.preview_frame,
                thermal_sample.frame,
                snapshot.landmarks,
                thermal_label=target.label,
                hand_label=target.hand_label,
                selected_ir_point=click_state["ir_point"],
            )
            samples = load_projection_samples(csv_path)
            cv2.putText(
                preview,
                f"samples={len(samples)}  click IR tip, s=save  f=fit  q=quit",
                (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                if manual_point_required and click_state["ir_point"] is None:
                    print(f"[calibration] click the {target.label} thumb point before saving")
                    continue
                now_s = time.perf_counter()
                paired = pair_projection_sample(
                    snapshot,
                    thermal_sample,
                    now_s=now_s,
                    max_oak_age_s=args.max_oak_age_ms / 1000.0,
                    max_thermal_age_s=args.max_thermal_age_ms / 1000.0,
                    max_pair_skew_s=args.max_pair_skew_ms / 1000.0,
                    ir_point=click_state["ir_point"] if manual_point_required else None,
                    min_hand_depth_m=args.min_hand_depth_m,
                    max_hand_depth_m=args.max_hand_depth_m,
                )
                if paired is None:
                    print(
                        "[calibration] rejected invalid, stale, or skewed "
                        f"{target.hand_label}/{target.label} sample: "
                        f"{projection_pair_diagnostics(snapshot, thermal_sample, now_s=now_s)} "
                        f"ir_point={click_state['ir_point']}"
                    )
                    continue
                append_projection_sample(
                    csv_path,
                    paired.sample,
                    oak_observed_at_s=paired.oak_observed_at_s,
                    thermal_observed_at_s=paired.thermal_observed_at_s,
                    sensor_skew_s=paired.sensor_skew_s,
                )
                print(f"[calibration] saved sample #{len(samples) + 1}: {paired.sample}")
                click_state["ir_point"] = None
            if key == ord("f"):
                samples = load_projection_samples(csv_path)
                if len(samples) < args.min_samples:
                    print(f"[calibration] need {args.min_samples} samples, have {len(samples)}")
                    continue
                calibration = fit_projection(samples, image_size=target.image_size)
                save_projection_error_report(
                    error_report_path,
                    calibration=calibration,
                    samples=samples,
                    hand_label=target.hand_label,
                    thermal_label=target.label,
                    gate_px=MAX_REPROJECTION_ERROR_PX,
                    min_hand_depth_m=args.min_hand_depth_m,
                    max_hand_depth_m=args.max_hand_depth_m,
                )
                print(f"[calibration] saved error report {error_report_path}")
                try:
                    validate_projection_calibration(
                        calibration,
                        min_samples=args.min_samples,
                        max_rms_error_px=args.max_rms_px,
                        max_error_px=args.max_error_px,
                        expected_image_size=target.image_size,
                    )
                except ValueError as exc:
                    print(f"[calibration] rejected: {exc}")
                    continue
                save_projection_calibration(calibration_path, calibration)
                loaded = load_projection_calibration(calibration_path)
                print(f"[calibration] saved {calibration_path}: {loaded}")


if __name__ == "__main__":
    main()
