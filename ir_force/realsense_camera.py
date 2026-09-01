"""Small aligned RGB/depth adapter for Intel RealSense calibration capture."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


class RealSenseCamera:
    def __init__(
        self,
        *,
        serial: str | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        rs_module=None,
    ):
        self.serial = serial
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self._rs = rs_module
        self._pipeline = None
        self._align = None
        self._depth_scale_m = None

    def start(self) -> None:
        if self._pipeline is not None:
            return
        if self._rs is None:
            try:
                import pyrealsense2 as rs
            except ImportError as exc:
                raise RuntimeError(
                    "pyrealsense2 is required for RealSense RGB/depth calibration"
                ) from exc
            self._rs = rs

        pipeline = self._rs.pipeline()
        config = self._rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(
            self._rs.stream.depth,
            self.width,
            self.height,
            self._rs.format.z16,
            self.fps,
        )
        config.enable_stream(
            self._rs.stream.color,
            self.width,
            self.height,
            self._rs.format.bgr8,
            self.fps,
        )
        try:
            profile = pipeline.start(config)
        except Exception:
            try:
                pipeline.stop()
            except Exception:
                pass
            raise

        self._pipeline = pipeline
        self._align = self._rs.align(self._rs.stream.color)
        self._depth_scale_m = float(
            profile.get_device().first_depth_sensor().get_depth_scale()
        )

    def read(self, *, timeout_ms: int = 1000) -> tuple[np.ndarray, np.ndarray, float]:
        if self._pipeline is None or self._align is None or self._depth_scale_m is None:
            raise RuntimeError("RealSense camera is not started")
        frames = self._pipeline.wait_for_frames(timeout_ms)
        aligned = self._align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("RealSense returned an incomplete aligned RGB/depth frameset")

        color_bgr = np.asanyarray(color_frame.get_data()).copy()
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_mm = depth_raw.astype(np.float32) * (self._depth_scale_m * 1000.0)
        return color_bgr, depth_mm, time.perf_counter()

    def stop(self) -> None:
        pipeline = self._pipeline
        self._pipeline = None
        self._align = None
        self._depth_scale_m = None
        if pipeline is not None:
            pipeline.stop()


RAW_PROJECTOR_SERIAL = "233522078685"
RAW_PROJECTOR_REQUESTED = {
    "serial": RAW_PROJECTOR_SERIAL,
    "color": {
        "width": 1280,
        "height": 720,
        "format": "rgb8",
        "fps": 15,
    },
    "depth": {
        "width": 1280,
        "height": 720,
        "format": "z16",
        "fps": 6,
    },
}


@dataclass(frozen=True)
class RealSenseRawSample:
    color_rgb: np.ndarray
    depth_z16: np.ndarray
    observed_at_s: float
    color_frame_number: int
    depth_frame_number: int
    color_timestamp_ms: float
    depth_timestamp_ms: float
    color_timestamp_domain: str
    depth_timestamp_domain: str
    depth_sdk_frame: object | None = None


def _profile_data(profile) -> dict:
    return {
        "width": profile.width(),
        "height": profile.height(),
        "format": str(profile.format()),
        "fps": profile.fps(),
        "stream": str(profile.stream_type()),
        "stream_index": profile.stream_index(),
        "unique_id": profile.unique_id(),
    }


def _intrinsics_data(intrinsics) -> dict:
    return {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "ppx": intrinsics.ppx,
        "ppy": intrinsics.ppy,
        "distortion_model": str(intrinsics.model),
        "coeffs": list(intrinsics.coeffs),
    }


def _extrinsics_data(extrinsics) -> dict:
    return {
        "rotation": list(extrinsics.rotation),
        "translation_m": list(extrinsics.translation),
    }


class RealSenseRawProjectorCamera:
    """Strict Stage 0 raw D435i source for the live projector."""

    def __init__(self, *, rs_module=None):
        self._rs = rs_module
        self._pipeline = None
        self._runtime_metadata = None
        self.color_intrinsics = None
        self.depth_intrinsics = None
        self.color_to_depth_extrinsics = None
        self.depth_to_color_extrinsics = None

    @property
    def runtime_metadata(self) -> dict:
        if self._runtime_metadata is None:
            raise RuntimeError("raw RealSense camera is not started")
        return self._runtime_metadata

    def start(self) -> None:
        if self._pipeline is not None:
            return
        if self._rs is None:
            try:
                import pyrealsense2 as rs
            except ImportError as exc:
                raise RuntimeError(
                    "pyrealsense2 is required for the raw D435i projector source"
                ) from exc
            self._rs = rs

        rs = self._rs
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(RAW_PROJECTOR_SERIAL)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 15)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
        started = False
        try:
            active = pipeline.start(config)
            started = True
            device = active.get_device()
            serial = device.get_info(rs.camera_info.serial_number)
            color = active.get_stream(rs.stream.color).as_video_stream_profile()
            depth = active.get_stream(rs.stream.depth).as_video_stream_profile()
            color_ok = (
                color.width() == 1280
                and color.height() == 720
                and color.format() == rs.format.rgb8
                and color.fps() == 15
            )
            depth_ok = (
                depth.width() == 1280
                and depth.height() == 720
                and depth.format() == rs.format.z16
                and depth.fps() == 6
            )
            if serial != RAW_PROJECTOR_SERIAL or not color_ok or not depth_ok:
                raise RuntimeError("resolved raw profile mismatch")

            color_intrinsics = color.get_intrinsics()
            depth_intrinsics = depth.get_intrinsics()
            depth_to_color = depth.get_extrinsics_to(color)
            color_to_depth = color.get_extrinsics_to(depth)
            runtime_metadata = {
                "requested": RAW_PROJECTOR_REQUESTED,
                "sdk_version": rs.__version__,
                "device": {
                    "serial": serial,
                    "name": device.get_info(rs.camera_info.name),
                    "firmware": device.get_info(rs.camera_info.firmware_version),
                    "product_line": device.get_info(rs.camera_info.product_line),
                },
                "resolved": {
                    "color": _profile_data(color),
                    "depth": _profile_data(depth),
                },
                "color_intrinsics": _intrinsics_data(color_intrinsics),
                "depth_intrinsics": _intrinsics_data(depth_intrinsics),
                "depth_scale_m": device.first_depth_sensor().get_depth_scale(),
                "factory_extrinsics": {
                    "depth_to_color": _extrinsics_data(depth_to_color),
                    "color_to_depth": _extrinsics_data(color_to_depth),
                },
            }
        except Exception:
            try:
                pipeline.stop()
            except Exception:
                pass
            raise

        if not started:
            raise RuntimeError("raw RealSense pipeline did not start")
        self._pipeline = pipeline
        self._runtime_metadata = runtime_metadata
        self.color_intrinsics = color_intrinsics
        self.depth_intrinsics = depth_intrinsics
        self.color_to_depth_extrinsics = color_to_depth
        self.depth_to_color_extrinsics = depth_to_color

    def read(self, *, timeout_ms: int = 1000) -> RealSenseRawSample:
        if self._pipeline is None:
            raise RuntimeError("raw RealSense camera is not started")
        frames = self._pipeline.wait_for_frames(timeout_ms)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("RealSense returned an incomplete raw RGB/depth frameset")

        color_rgb = np.asanyarray(color_frame.get_data()).copy()
        depth_z16 = np.asanyarray(depth_frame.get_data()).copy()
        if color_rgb.shape != (720, 1280, 3) or color_rgb.dtype != np.uint8:
            raise RuntimeError("RealSense returned an invalid raw RGB8 frame")
        if depth_z16.shape != (720, 1280) or depth_z16.dtype != np.uint16:
            raise RuntimeError("RealSense returned an invalid raw Z16 frame")
        observed_at_s = time.perf_counter()
        return RealSenseRawSample(
            color_rgb=color_rgb,
            depth_z16=depth_z16,
            observed_at_s=observed_at_s,
            color_frame_number=color_frame.get_frame_number(),
            depth_frame_number=depth_frame.get_frame_number(),
            color_timestamp_ms=color_frame.get_timestamp(),
            depth_timestamp_ms=depth_frame.get_timestamp(),
            color_timestamp_domain=str(color_frame.get_frame_timestamp_domain()),
            depth_timestamp_domain=str(depth_frame.get_frame_timestamp_domain()),
            depth_sdk_frame=depth_frame,
        )

    def stop(self) -> None:
        pipeline = self._pipeline
        self._pipeline = None
        self._runtime_metadata = None
        self.color_intrinsics = None
        self.depth_intrinsics = None
        self.color_to_depth_extrinsics = None
        self.depth_to_color_extrinsics = None
        if pipeline is not None:
            pipeline.stop()
