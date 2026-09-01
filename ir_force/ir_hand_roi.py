from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .ir_hand_calibration import ProjectionCalibration, ProjectionResult, project_oak_to_ir

THUMB_TIP = 4
INDEX_TIP = 8


@dataclass(frozen=True)
class PressureROI:
    x: int
    y: int
    width: int
    height: int

    @property
    def x_end(self) -> int:
        return self.x + self.width

    @property
    def y_end(self) -> int:
        return self.y + self.height

    def slices(self) -> tuple[slice, slice]:
        return slice(self.y, self.y_end), slice(self.x, self.x_end)


@dataclass(frozen=True)
class ROISelection:
    roi: PressureROI | None
    quality: float
    mode: str


def _clip_roi(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    width: int,
    height: int,
) -> tuple[PressureROI | None, float]:
    requested_x0 = int(np.floor(x0))
    requested_y0 = int(np.floor(y0))
    requested_x1 = int(np.ceil(x1))
    requested_y1 = int(np.ceil(y1))
    requested_area = max(0, requested_x1 - requested_x0) * max(
        0, requested_y1 - requested_y0
    )
    ix0 = max(0, requested_x0)
    iy0 = max(0, requested_y0)
    ix1 = min(width, requested_x1)
    iy1 = min(height, requested_y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return None, 0.0
    roi = PressureROI(ix0, iy0, ix1 - ix0, iy1 - iy0)
    retained_area = roi.width * roi.height
    return roi, retained_area / requested_area if requested_area else 0.0


def _project_tip(
    landmarks,
    calibration: ProjectionCalibration,
    index: int,
) -> ProjectionResult | None:
    if landmarks.image_xy is None or landmarks.depth_m is None:
        return None
    z = float(landmarks.depth_m[index])
    if not np.isfinite(z):
        return None
    oak_x, oak_y = landmarks.image_xy[index]
    return project_oak_to_ir(calibration, oak_x=float(oak_x), oak_y=float(oak_y), oak_z=z)


def select_pressure_roi(
    landmarks,
    calibration: ProjectionCalibration,
    frame_shape,
    *,
    tip_radius_px: int = 5,
    corridor_pad_px: int = 5,
) -> ROISelection:
    height, width = frame_shape[:2]
    if not getattr(landmarks, "valid", False) or landmarks.image_xy is None or landmarks.depth_m is None:
        return ROISelection(None, 0.0, "missing_oak_metadata")

    thumb = _project_tip(landmarks, calibration, THUMB_TIP)
    index = _project_tip(landmarks, calibration, INDEX_TIP)
    if (thumb is not None and not thumb.valid) or (index is not None and not index.valid):
        return ROISelection(None, 0.0, "projection_out_of_fov")
    if thumb is not None and index is not None:
        xs = [thumb.x, index.x]
        ys = [thumb.y, index.y]
        roi, retained_fraction = _clip_roi(
            min(xs) - corridor_pad_px,
            min(ys) - corridor_pad_px,
            max(xs) + corridor_pad_px + 1,
            max(ys) + corridor_pad_px + 1,
            width=width,
            height=height,
        )
        return ROISelection(roi, retained_fraction if roi is not None else 0.0, "tips")

    point = thumb if thumb is not None else index
    if point is None:
        return ROISelection(None, 0.0, "missing_tip_depth")
    roi, retained_fraction = _clip_roi(
        point.x - tip_radius_px,
        point.y - tip_radius_px,
        point.x + tip_radius_px + 1,
        point.y + tip_radius_px + 1,
        width=width,
        height=height,
    )
    quality = 0.5 * retained_fraction if roi is not None else 0.0
    return ROISelection(roi, quality, "single_tip")


def select_thermal_blob_roi(
    scalar_frame: np.ndarray,
    *,
    background_percentile: float = 25.0,
    min_delta_counts: float = 100.0,
    min_area_px: int = 4,
    max_area_px: int = 900,
    pad_px: int = 2,
) -> ROISelection:
    """Pick the hottest compact blob (skin-on-skin contact patch) as the pressure ROI.

    Needs no cross-camera calibration: during a pinch the merged fingertip patch is
    the hottest region in a Lepton FOV aimed at the hand workspace. A pixel is hot
    when it exceeds the background floor (low percentile of the frame) by
    min_delta_counts, so a flat frame yields no hotspot instead of thresholding
    noise, and the hot area stays a genuine area measurement rather than being
    clamped to a fixed top-percentile pixel budget.
    """
    scalar = np.asarray(scalar_frame, dtype=np.float32)
    height, width = scalar.shape[:2]
    threshold = float(np.percentile(scalar, background_percentile)) + min_delta_counts
    mask = scalar >= threshold
    if not mask.any():
        return ROISelection(None, 0.0, "blob_no_hotspot")

    labels, count = ndimage.label(mask)
    areas = ndimage.sum_labels(mask, labels, index=range(1, count + 1))
    largest = int(np.argmax(areas)) + 1
    area = float(areas[largest - 1])
    if area < min_area_px:
        return ROISelection(None, 0.0, "blob_too_small")
    if area > max_area_px:
        return ROISelection(None, 0.0, "blob_too_large")

    ys, xs = np.nonzero(labels == largest)
    roi, retained_fraction = _clip_roi(
        float(xs.min()) - pad_px,
        float(ys.min()) - pad_px,
        float(xs.max()) + pad_px + 1,
        float(ys.max()) + pad_px + 1,
        width=width,
        height=height,
    )
    if roi is None:
        return ROISelection(None, 0.0, "blob_no_hotspot")
    fill_fraction = area / (roi.width * roi.height)
    return ROISelection(roi, retained_fraction * fill_fraction, "blob")
