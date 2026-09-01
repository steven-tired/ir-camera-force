from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np


FROZEN_EXTRINSIC_SHA256 = "2ca1ed48450dea16a5778cb5645dd4852d544490e4f47330dd938f743bc6f434"
THERMAL_WIDTH = 160
THERMAL_HEIGHT = 120

@dataclass(frozen=True)
class ThermalProjectionResult:
    status: str
    source_depth_xy: tuple[int, int]
    depth_m: float | None = None
    color_xyz_m: tuple[float, float, float] | None = None
    thermal_xyz_m: tuple[float, float, float] | None = None
    thermal_uv: tuple[float, float] | None = None


def validate_thermal_geometry(
    R_tc, T_tc, thermal_K, thermal_D, *, direction, unit
) -> None:
    arrays = (
        ("rotation", np.asarray(R_tc, dtype=np.float64), (3, 3)),
        ("translation", np.asarray(T_tc, dtype=np.float64), (3,)),
        ("thermal intrinsics", np.asarray(thermal_K, dtype=np.float64), (3, 3)),
        ("thermal distortion", np.asarray(thermal_D, dtype=np.float64), (5,)),
    )
    for label, value, expected_shape in arrays:
        if value.shape != expected_shape:
            raise ValueError(f"{label} shape must be {expected_shape}; got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{label} must contain only finite values")

    if direction != "thermal_to_color":
        raise ValueError(f"extrinsic direction must be thermal_to_color; got {direction!r}")
    if unit != "meter":
        raise ValueError(f"extrinsic unit must be meter; got {unit!r}")

    rotation = arrays[0][1]
    if not np.allclose(rotation @ rotation.T, np.eye(3), rtol=0.0, atol=1e-6):
        raise ValueError("rotation must be orthonormal")
    if float(np.linalg.det(rotation)) <= 0.0:
        raise ValueError("rotation determinant must be positive")


def load_frozen_thermal_geometry(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xml_path = Path(path)
    actual_hash = sha256(xml_path.read_bytes()).hexdigest()
    if actual_hash != FROZEN_EXTRINSIC_SHA256:
        raise ValueError(
            "frozen extrinsic SHA-256 mismatch: "
            f"expected {FROZEN_EXTRINSIC_SHA256}, got {actual_hash}"
        )

    storage = cv2.FileStorage(str(xml_path), cv2.FILE_STORAGE_READ)
    try:
        if not storage.isOpened():
            raise ValueError(f"could not open frozen extrinsic XML: {xml_path}")
        R_tc = storage.getNode("R").mat()
        T_tc = storage.getNode("T").mat()
        thermal_K = storage.getNode("thermal_K").mat()
        thermal_D = storage.getNode("thermal_D").mat()
        direction = storage.getNode("direction").string()
        unit = storage.getNode("unit").string()
    finally:
        storage.release()

    if any(value is None for value in (R_tc, T_tc, thermal_K, thermal_D)):
        raise ValueError("frozen extrinsic XML is missing required geometry")

    R_tc = np.asarray(R_tc, dtype=np.float64)
    T_tc = np.asarray(T_tc, dtype=np.float64).reshape(-1)
    thermal_K = np.asarray(thermal_K, dtype=np.float64)
    thermal_D = np.asarray(thermal_D, dtype=np.float64).reshape(-1)
    validate_thermal_geometry(
        R_tc, T_tc, thermal_K, thermal_D, direction=direction, unit=unit
    )
    return R_tc.copy(), T_tc.copy(), thermal_K.copy(), thermal_D.copy()


def project_raw_depth_pixel_to_thermal(
    *,
    rs_module,
    source_depth_xy: tuple[int, int],
    raw_depth: int,
    depth_scale_m: float,
    depth_intrinsics,
    depth_to_color_extrinsics,
    R_tc: np.ndarray,
    T_tc: np.ndarray,
    thermal_K: np.ndarray,
    thermal_D: np.ndarray,
) -> ThermalProjectionResult:
    if (
        not isinstance(source_depth_xy, tuple)
        or len(source_depth_xy) != 2
        or any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            for value in source_depth_xy
        )
    ):
        return ThermalProjectionResult("source_depth_out_of_bounds", source_depth_xy)
    source_x, source_y = source_depth_xy
    if not (
        0 <= source_x < int(depth_intrinsics.width)
        and 0 <= source_y < int(depth_intrinsics.height)
    ):
        return ThermalProjectionResult("source_depth_out_of_bounds", source_depth_xy)

    if (
        isinstance(raw_depth, (bool, np.bool_))
        or not isinstance(raw_depth, (int, np.integer))
        or raw_depth <= 0
        or not np.isfinite(depth_scale_m)
        or depth_scale_m <= 0.0
    ):
        return ThermalProjectionResult("depth_invalid", source_depth_xy)
    if depth_intrinsics.model != rs_module.distortion.brown_conrady:
        return ThermalProjectionResult("depth_model_mismatch", source_depth_xy)

    depth_m = float(raw_depth) * float(depth_scale_m)
    depth_xyz = rs_module.rs2_deproject_pixel_to_point(
        depth_intrinsics, [float(source_x), float(source_y)], depth_m
    )
    color_xyz = np.asarray(
        rs_module.rs2_transform_point_to_point(depth_to_color_extrinsics, depth_xyz),
        dtype=np.float64,
    )
    thermal_xyz = np.asarray(R_tc, dtype=np.float64).T @ (
        color_xyz - np.asarray(T_tc, dtype=np.float64)
    )
    if (
        color_xyz.shape != (3,)
        or thermal_xyz.shape != (3,)
        or not np.all(np.isfinite(color_xyz))
        or not np.all(np.isfinite(thermal_xyz))
        or color_xyz[2] <= 0.0
        or thermal_xyz[2] <= 0.0
    ):
        return ThermalProjectionResult("nonpositive_or_nonfinite_geometry", source_depth_xy)

    thermal_intrinsics = rs_module.intrinsics()
    thermal_intrinsics.width = THERMAL_WIDTH
    thermal_intrinsics.height = THERMAL_HEIGHT
    thermal_intrinsics.fx = float(thermal_K[0, 0])
    thermal_intrinsics.fy = float(thermal_K[1, 1])
    thermal_intrinsics.ppx = float(thermal_K[0, 2])
    thermal_intrinsics.ppy = float(thermal_K[1, 2])
    thermal_intrinsics.model = rs_module.distortion.brown_conrady
    thermal_intrinsics.coeffs = np.asarray(thermal_D, dtype=np.float64).tolist()
    thermal_uv = np.asarray(
        rs_module.rs2_project_point_to_pixel(thermal_intrinsics, thermal_xyz.tolist()),
        dtype=np.float64,
    )
    if (
        thermal_uv.shape != (2,)
        or not np.all(np.isfinite(thermal_uv))
        or not (
            0.0 <= thermal_uv[0] < THERMAL_WIDTH
            and 0.0 <= thermal_uv[1] < THERMAL_HEIGHT
        )
    ):
        return ThermalProjectionResult("thermal_out_of_fov", source_depth_xy)

    return ThermalProjectionResult(
        status="ok",
        source_depth_xy=source_depth_xy,
        depth_m=depth_m,
        color_xyz_m=tuple(float(value) for value in color_xyz),
        thermal_xyz_m=tuple(float(value) for value in thermal_xyz),
        thermal_uv=tuple(float(value) for value in thermal_uv),
    )
