"""Runtime helpers for continuous single-finger thermal curve capture."""

from __future__ import annotations

from math import floor, isfinite
from pathlib import Path

import cv2
import numpy as np


LANDMARK_INDICES = {
    "TIP": 8,
    "DIP": 7,
    "PIP": 6,
    "MCP": 5,
}


def _finite_uv(value, *, label: str) -> tuple[float, float]:
    try:
        uv = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}_invalid") from exc
    if len(uv) != 2 or not all(isfinite(item) for item in uv):
        raise ValueError(f"{label}_invalid")
    return uv


def roi_centers(thermal_uv_by_landmark) -> dict[str, tuple[float, float]]:
    points = {
        label: _finite_uv(thermal_uv_by_landmark[label], label=label.lower())
        for label in LANDMARK_INDICES
    }
    tip = points["TIP"]
    dip = points["DIP"]
    pip = points["PIP"]
    mcp = points["MCP"]
    return {
        "tip_uv": tip,
        "distal_uv": tuple(
            tip[axis] + (dip[axis] - tip[axis]) / 3.0 for axis in (0, 1)
        ),
        "reference_uv": tuple(
            (pip[axis] + mcp[axis]) / 2.0 for axis in (0, 1)
        ),
    }


def _rounded_pixel(uv: tuple[float, float]) -> tuple[int, int]:
    return tuple(floor(value + 0.5) for value in uv)


def _patch_mean(
    frame: np.ndarray,
    center_uv: tuple[float, float],
    size: int,
    *,
    label: str,
) -> float:
    center_x, center_y = _rounded_pixel(center_uv)
    radius = size // 2
    x0, x1 = center_x - radius, center_x + radius + 1
    y0, y1 = center_y - radius, center_y + radius + 1
    if x0 < 0 or y0 < 0 or x1 > frame.shape[1] or y1 > frame.shape[0]:
        raise ValueError(f"{label}_out_of_bounds")
    return float(np.mean(frame[y0:y1, x0:x1], dtype=np.float64))


def thermal_features(frame, centers) -> dict[str, float | int]:
    frame = np.asarray(frame)
    if frame.shape != (120, 160) or frame.dtype != np.uint16:
        raise ValueError("thermal_frame_invalid")
    tip_uv = _finite_uv(centers["tip_uv"], label="tip")
    distal_uv = _finite_uv(centers["distal_uv"], label="distal")
    reference_uv = _finite_uv(centers["reference_uv"], label="reference")
    tip_x, tip_y = _rounded_pixel(tip_uv)
    if not (0 <= tip_x < frame.shape[1] and 0 <= tip_y < frame.shape[0]):
        raise ValueError("tip_out_of_bounds")

    distal_3x3 = _patch_mean(
        frame,
        distal_uv,
        3,
        label="distal_3x3",
    )
    distal_5x5 = _patch_mean(
        frame,
        distal_uv,
        5,
        label="distal_5x5",
    )
    reference_5x5 = _patch_mean(
        frame,
        reference_uv,
        5,
        label="reference_5x5",
    )
    return {
        "tip_raw_count": int(frame[tip_y, tip_x]),
        "distal_3x3_mean_count": distal_3x3,
        "distal_5x5_mean_count": distal_5x5,
        "reference_5x5_mean_count": reference_5x5,
        "primary_signal_count": distal_3x3 - reference_5x5,
    }


def _write_png_exclusive(path: Path, frame: np.ndarray) -> None:
    encoded, payload = cv2.imencode(".png", frame)
    if not encoded:
        raise RuntimeError(f"could not encode PNG {path}")
    with path.open("xb") as stream:
        stream.write(payload.tobytes())


def _thermal_inferno_auto(frame: np.ndarray) -> np.ndarray:
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


def _nonnegative_integer(value, *, label: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
    ):
        raise ValueError(f"{label} must be a non-negative integer")
    return int(value)


def _validate_capture_arrays(
    thermal_counts,
    color_rgb,
    depth_z16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    thermal_counts = np.asarray(thermal_counts)
    color_rgb = np.asarray(color_rgb)
    depth_z16 = np.asarray(depth_z16)
    if thermal_counts.shape != (120, 160) or thermal_counts.dtype != np.uint16:
        raise ValueError("thermal_counts must be uint16 with shape (120, 160)")
    if (
        color_rgb.ndim != 3
        or color_rgb.shape[2] != 3
        or color_rgb.dtype != np.uint8
    ):
        raise ValueError("color_rgb must be uint8 with shape (H, W, 3)")
    if depth_z16.ndim != 2 or depth_z16.dtype != np.uint16:
        raise ValueError("depth_z16 must be a two-dimensional uint16 array")
    if color_rgb.shape[:2] != depth_z16.shape:
        raise ValueError("color_rgb and depth_z16 dimensions must match")
    return thermal_counts, color_rgb, depth_z16


class ContinuousFrameArchive:
    def __init__(self, session_dir: Path, surface_photo: Path | None):
        self.session_dir = Path(session_dir)
        self.surface_photo = (
            None if surface_photo is None else Path(surface_photo)
        )
        if (
            self.surface_photo is not None
            and not self.surface_photo.is_file()
        ):
            raise FileNotFoundError(self.surface_photo)
        self.session_dir.mkdir(parents=True, exist_ok=False)
        for relative in (
            Path("raw/thermal_uint16"),
            Path("raw/d435_rgb"),
            Path("raw/d435_depth_z16"),
            Path("rendered/thermal_inferno_auto"),
            Path("setup"),
        ):
            (self.session_dir / relative).mkdir(parents=True, exist_ok=False)
        if self.surface_photo is not None:
            photo_target = (
                self.session_dir / "setup" / self.surface_photo.name
            )
            with (
                self.surface_photo.open("rb") as source,
                photo_target.open("xb") as target,
            ):
                target.write(source.read())
        self._color_paths: dict[int, str] = {}
        self._depth_paths: dict[int, str] = {}

    def capture(
        self,
        *,
        frame_index: int,
        thermal_counts: np.ndarray,
        color_rgb: np.ndarray,
        depth_z16: np.ndarray,
        color_frame_number: int,
        depth_frame_number: int,
    ) -> dict[str, str]:
        frame_index = _nonnegative_integer(frame_index, label="frame_index")
        color_frame_number = _nonnegative_integer(
            color_frame_number,
            label="color_frame_number",
        )
        depth_frame_number = _nonnegative_integer(
            depth_frame_number,
            label="depth_frame_number",
        )
        thermal_counts, color_rgb, depth_z16 = _validate_capture_arrays(
            thermal_counts,
            color_rgb,
            depth_z16,
        )

        thermal_relative = (
            Path("raw/thermal_uint16") / f"frame_{frame_index:06d}.png"
        ).as_posix()
        thermal_target = self.session_dir / thermal_relative
        if thermal_target.exists():
            raise FileExistsError(thermal_target)
        rendered_relative = (
            Path("rendered/thermal_inferno_auto")
            / f"frame_{frame_index:06d}.png"
        ).as_posix()
        rendered_target = self.session_dir / rendered_relative
        if rendered_target.exists():
            raise FileExistsError(rendered_target)

        color_relative = self._color_paths.get(color_frame_number)
        if color_relative is None:
            color_relative = (
                Path("raw/d435_rgb")
                / f"color_{color_frame_number:010d}.png"
            ).as_posix()
            if (self.session_dir / color_relative).exists():
                raise FileExistsError(self.session_dir / color_relative)

        depth_relative = self._depth_paths.get(depth_frame_number)
        if depth_relative is None:
            depth_relative = (
                Path("raw/d435_depth_z16")
                / f"depth_{depth_frame_number:010d}.png"
            ).as_posix()
            if (self.session_dir / depth_relative).exists():
                raise FileExistsError(self.session_dir / depth_relative)

        _write_png_exclusive(thermal_target, thermal_counts)
        _write_png_exclusive(
            rendered_target,
            _thermal_inferno_auto(thermal_counts),
        )
        if color_frame_number not in self._color_paths:
            _write_png_exclusive(
                self.session_dir / color_relative,
                cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR),
            )
            self._color_paths[color_frame_number] = color_relative
        if depth_frame_number not in self._depth_paths:
            _write_png_exclusive(self.session_dir / depth_relative, depth_z16)
            self._depth_paths[depth_frame_number] = depth_relative
        return {
            "thermal_uint16": thermal_relative,
            "thermal_inferno_auto": rendered_relative,
            "d435_rgb": self._color_paths[color_frame_number],
            "d435_depth_z16": self._depth_paths[depth_frame_number],
        }


def _source_fields(raw, thermal, now_s: float) -> dict:
    telemetry = thermal.lepton_telemetry
    d435_age = float(now_s) - float(raw.observed_at_s)
    lepton_age = float(now_s) - float(thermal.t)
    return {
        "thermal_host_s": float(thermal.t),
        "d435_host_s": float(raw.observed_at_s),
        "color_frame_number": int(raw.color_frame_number),
        "depth_frame_number": int(raw.depth_frame_number),
        "color_timestamp_ms": float(raw.color_timestamp_ms),
        "depth_timestamp_ms": float(raw.depth_timestamp_ms),
        "color_timestamp_domain": str(raw.color_timestamp_domain),
        "depth_timestamp_domain": str(raw.depth_timestamp_domain),
        "d435_host_read_completion_age_s": d435_age,
        "lepton_host_read_completion_age_s": lepton_age,
        "host_read_completion_age_s": max(d435_age, lepton_age),
        "host_read_completion_skew_s": abs(
            float(raw.observed_at_s) - float(thermal.t)
        ),
        "lepton_frame_counter": (
            None if telemetry is None else int(telemetry.frame_counter)
        ),
        "lepton_packet_timestamp_ms": (
            None if telemetry is None else int(telemetry.packet_timestamp_ms)
        ),
        "ffc_desired": (
            None if telemetry is None else bool(telemetry.ffc_desired)
        ),
        "ffc_state": (
            None if telemetry is None else str(telemetry.ffc_state)
        ),
        "ffc_in_progress": (
            None if telemetry is None else bool(telemetry.ffc_in_progress)
        ),
        "since_last_ffc_s": (
            None if telemetry is None else float(telemetry.since_last_ffc_s)
        ),
        "tlinear_enabled": (
            None if telemetry is None else bool(telemetry.tlinear_enabled)
        ),
        "tlinear_resolution_k": (
            None
            if telemetry is None or telemetry.tlinear_resolution_k is None
            else float(telemetry.tlinear_resolution_k)
        ),
    }


def _blocked_row(row: dict, reason: str) -> dict:
    row["tracking_valid"] = False
    row["tracking_reasons"] = [reason]
    return row


def build_frame_row(
    raw,
    thermal,
    hands,
    projection_context,
    trial_context,
) -> dict:
    now_s = float(trial_context["now_s"])
    row = {
        "row_type": "frame",
        **{
            key: value
            for key, value in trial_context.items()
            if key != "now_s"
        },
        **_source_fields(raw, thermal, now_s),
        "tracking_valid": False,
        "tracking_reasons": [],
    }
    if thermal.frame.shape != (120, 160) or thermal.frame.dtype != np.uint16:
        return _blocked_row(row, "lepton_frame_invalid")

    try:
        hand_results = hands.process(raw.color_rgb)
        right, _left = projection_context["split_results"](hand_results)
    except Exception:
        return _blocked_row(row, "mediapipe_processing_failed")
    if right is None:
        return _blocked_row(row, "physical_right_hand_missing")

    image_xy = np.asarray(right[1], dtype=float)
    if image_xy.shape != (21, 2) or not np.all(np.isfinite(image_xy)):
        return _blocked_row(row, "physical_right_landmarks_invalid")
    depth_sdk_buffer = (
        None
        if raw.depth_sdk_frame is None
        else raw.depth_sdk_frame.get_data()
    )
    association_fn = projection_context["associate_color_to_raw_depth"]
    projection_fn = projection_context["project_depth_pixel_to_thermal"]
    association_kwargs = projection_context.get("association_kwargs", {})
    projection_kwargs = projection_context.get("projection_kwargs", {})
    associations = {}
    projected = {}
    for label, landmark_index in LANDMARK_INDICES.items():
        association = association_fn(
            label=label,
            normalized_xy=image_xy[landmark_index],
            depth_z16=raw.depth_z16,
            depth_sdk_buffer=depth_sdk_buffer,
            **association_kwargs,
        )
        associations[label] = association
        if association.get("status") != "ok":
            reason = association.get("reason", "association_failed")
            row["landmark_associations"] = associations
            return _blocked_row(row, f"{label}:{reason}")
        result = projection_fn(
            source_depth_xy=tuple(association["depth_pixel"]),
            raw_depth=association["raw_depth"],
            **projection_kwargs,
        )
        if result.status != "ok" or result.thermal_uv is None:
            row["landmark_associations"] = associations
            return _blocked_row(row, f"{label}:{result.status}")
        projected[label] = result

    thermal_uv = {
        label: result.thermal_uv for label, result in projected.items()
    }
    centers = roi_centers(thermal_uv)
    try:
        features = thermal_features(thermal.frame, centers)
    except ValueError as exc:
        row["landmark_associations"] = associations
        row["thermal_uv_by_landmark"] = {
            label: list(uv) for label, uv in thermal_uv.items()
        }
        return _blocked_row(row, str(exc))

    tip_depth = float(associations["TIP"]["depth_m"])
    dip_depth = float(associations["DIP"]["depth_m"])
    distal_depth = tip_depth + (dip_depth - tip_depth) / 3.0
    row.update(
        tracking_valid=True,
        tracking_reasons=[],
        landmark_associations=associations,
        thermal_uv_by_landmark={
            label: list(uv) for label, uv in thermal_uv.items()
        },
        tip_thermal_uv=list(centers["tip_uv"]),
        distal_thermal_uv=list(centers["distal_uv"]),
        reference_thermal_uv=list(centers["reference_uv"]),
        distal_thermal_u_px=float(centers["distal_uv"][0]),
        distal_thermal_v_px=float(centers["distal_uv"][1]),
        index_tip_depth_m=tip_depth,
        distal_depth_m=distal_depth,
        **features,
    )
    if (
        row["tlinear_enabled"] is True
        and row["tlinear_resolution_k"] == 0.01
    ):
        row["primary_signal_delta_c"] = (
            float(row["primary_signal_count"]) * 0.01
        )
    else:
        row["primary_signal_delta_c"] = None
    return row
