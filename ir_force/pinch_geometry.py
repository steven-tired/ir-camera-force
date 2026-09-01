"""Stateless D435i RGB-D pinch-geometry baseline features."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral

import numpy as np


_THUMB_TIP = 4
_INDEX_MCP = 5
_INDEX_TIP = 8
_PINKY_MCP = 17
_REQUIRED_XY = (_THUMB_TIP, _INDEX_TIP, _INDEX_MCP, _PINKY_MCP)


class PinchGeometryReason(str, Enum):
    OK = "OK"
    NONFINITE_REQUIRED_XY = "NONFINITE_REQUIRED_XY"
    REQUIRED_XY_OUT_OF_BOUNDS = "REQUIRED_XY_OUT_OF_BOUNDS"
    DEGENERATE_PALM_SCALE = "DEGENERATE_PALM_SCALE"
    INVALID_THUMB_TIP_DEPTH = "INVALID_THUMB_TIP_DEPTH"
    INVALID_INDEX_TIP_DEPTH = "INVALID_INDEX_TIP_DEPTH"
    INVALID_BOTH_TIP_DEPTHS = "INVALID_BOTH_TIP_DEPTHS"
    NONFINITE_DERIVED_FEATURE = "NONFINITE_DERIVED_FEATURE"


@dataclass(frozen=True)
class PinchGeometryFeatures:
    pinch_distance_2d_norm: float
    pinch_depth_delta_m: float
    valid: bool
    reason: PinchGeometryReason


def _pixel_distance(first: np.ndarray, second: np.ndarray, width_px: int, height_px: int) -> float:
    scale = np.array([width_px, height_px], dtype=float)
    return float(np.linalg.norm((first - second) * scale))


def compute_pinch_geometry(
    image_xy,
    depth_m,
    *,
    width_px: int,
    height_px: int,
) -> PinchGeometryFeatures:
    """Compute the preregisterable two-feature RGB-D pinch baseline.

    The 2D aperture is thumb-tip to index-tip pixel distance normalized by
    index-MCP to pinky-MCP palm width. Depth contributes only the absolute
    metric Z separation of the two fingertips; no pseudo-3D distance is formed.
    """
    try:
        xy = np.asarray(image_xy, dtype=float)
        depths = np.asarray(depth_m, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("image_xy and depth_m must be numeric arrays") from exc
    if xy.shape != (21, 2):
        raise ValueError(f"image_xy must have shape (21, 2), got {xy.shape}")
    if depths.shape != (21,):
        raise ValueError(f"depth_m must have shape (21,), got {depths.shape}")
    if (
        isinstance(width_px, (bool, np.bool_))
        or not isinstance(width_px, Integral)
        or int(width_px) <= 0
    ):
        raise ValueError("width_px must be a positive integer")
    if (
        isinstance(height_px, (bool, np.bool_))
        or not isinstance(height_px, Integral)
        or int(height_px) <= 0
    ):
        raise ValueError("height_px must be a positive integer")

    required_xy = xy[list(_REQUIRED_XY)]
    xy_reason = PinchGeometryReason.OK
    pinch_2d = float("nan")
    if not np.isfinite(required_xy).all():
        xy_reason = PinchGeometryReason.NONFINITE_REQUIRED_XY
    elif ((required_xy < 0.0) | (required_xy > 1.0)).any():
        xy_reason = PinchGeometryReason.REQUIRED_XY_OUT_OF_BOUNDS
    else:
        palm_scale = _pixel_distance(xy[_INDEX_MCP], xy[_PINKY_MCP], width_px, height_px)
        if not np.isfinite(palm_scale) or palm_scale <= 0.0:
            xy_reason = PinchGeometryReason.DEGENERATE_PALM_SCALE
        else:
            pinch_2d = _pixel_distance(xy[_THUMB_TIP], xy[_INDEX_TIP], width_px, height_px) / palm_scale
            if not np.isfinite(pinch_2d):
                xy_reason = PinchGeometryReason.NONFINITE_DERIVED_FEATURE
                pinch_2d = float("nan")

    thumb_depth_valid = bool(np.isfinite(depths[_THUMB_TIP]) and depths[_THUMB_TIP] > 0.0)
    index_depth_valid = bool(np.isfinite(depths[_INDEX_TIP]) and depths[_INDEX_TIP] > 0.0)
    depth_reason = PinchGeometryReason.OK
    pinch_depth = float("nan")
    if not thumb_depth_valid and not index_depth_valid:
        depth_reason = PinchGeometryReason.INVALID_BOTH_TIP_DEPTHS
    elif not thumb_depth_valid:
        depth_reason = PinchGeometryReason.INVALID_THUMB_TIP_DEPTH
    elif not index_depth_valid:
        depth_reason = PinchGeometryReason.INVALID_INDEX_TIP_DEPTH
    else:
        pinch_depth = abs(float(depths[_THUMB_TIP] - depths[_INDEX_TIP]))
        if not np.isfinite(pinch_depth):
            depth_reason = PinchGeometryReason.NONFINITE_DERIVED_FEATURE
            pinch_depth = float("nan")

    reason = xy_reason if xy_reason is not PinchGeometryReason.OK else depth_reason
    return PinchGeometryFeatures(
        pinch_distance_2d_norm=pinch_2d,
        pinch_depth_delta_m=pinch_depth,
        valid=reason is PinchGeometryReason.OK,
        reason=reason,
    )
