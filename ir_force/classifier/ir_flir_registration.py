"""Map FLIR-visible marker motion into the aligned thermal frame.

The soft-foam acquisition uses the FLIR visible and thermal streams as an
approximately co-located pair.  Visible-frame black dots are therefore used
only to update pre-registered thermal analysis regions; thermal pixels are
never expected to contain the dots themselves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ir_force.classifier.ir_foam_compression import (
    FrozenThermalRegions,
    MarkerObservation,
    PixelROI,
)


def project_marker_observation(
    marker: MarkerObservation,
    *,
    source_shape: tuple[int, int],
    destination_shape: tuple[int, int],
) -> MarkerObservation:
    """Project a marker observation by frame-size normalization only.

    Shapes use OpenCV order: ``(height, width)``.  This deliberately records a
    physical alignment assumption rather than claiming a calibrated stereo
    transform.
    """
    source_height, source_width = source_shape
    destination_height, destination_width = destination_shape
    if min(source_height, source_width, destination_height, destination_width) <= 0:
        raise ValueError("source and destination shapes must be positive")
    scale_x = destination_width / source_width
    scale_y = destination_height / source_height
    area_scale = scale_x * scale_y

    def project(point: tuple[float, float]) -> tuple[float, float]:
        return point[0] * scale_x, point[1] * scale_y

    return MarkerObservation(
        left_xy=project(marker.left_xy),
        right_xy=project(marker.right_xy),
        left_area_px=marker.left_area_px * area_scale,
        right_area_px=marker.right_area_px * area_scale,
    )


@dataclass(frozen=True)
class SimilarityTransform:
    """Two-point similarity transform in thermal-pixel coordinates."""

    scale: float
    angle_deg: float
    translation_xy: tuple[float, float]

    def apply_point(self, point: tuple[float, float]) -> tuple[float, float]:
        angle_rad = math.radians(self.angle_deg)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)
        x, y = point
        translation_x, translation_y = self.translation_xy
        return (
            self.scale * (cos_angle * x - sin_angle * y) + translation_x,
            self.scale * (sin_angle * x + cos_angle * y) + translation_y,
        )

    def transform_roi(self, roi: PixelROI, *, frame_shape: tuple[int, int]) -> PixelROI:
        """Track an axis-aligned ROI by its transformed centre and scale."""
        frame_height, frame_width = frame_shape
        if min(frame_height, frame_width) <= 0:
            raise ValueError("frame shape must be positive")
        width = max(1, round(roi.width * self.scale))
        height = max(1, round(roi.height * self.scale))
        width = min(width, frame_width)
        height = min(height, frame_height)
        x, y = self.apply_point((roi.x, roi.y))
        x = round(x)
        y = round(y)
        x = min(max(x, 0), frame_width - width)
        y = min(max(y, 0), frame_height - height)
        return PixelROI(x, y, width, height)


def track_foam_regions(
    regions: FrozenThermalRegions,
    transform: SimilarityTransform,
    *,
    frame_shape: tuple[int, int],
) -> FrozenThermalRegions:
    """Move only foam-attached regions; scene references remain camera-fixed."""
    return FrozenThermalRegions(
        foam_bbox=transform.transform_roi(regions.foam_bbox, frame_shape=frame_shape),
        foam_center=transform.transform_roi(regions.foam_center, frame_shape=frame_shape),
        left_contact=transform.transform_roi(regions.left_contact, frame_shape=frame_shape),
        right_contact=transform.transform_roi(regions.right_contact, frame_shape=frame_shape),
        background=regions.background,
        room_reference=regions.room_reference,
        warm_reference=regions.warm_reference,
        overlay_y_exclusive=regions.overlay_y_exclusive,
    )


def similarity_transform_from_markers(
    baseline: MarkerObservation,
    current: MarkerObservation,
) -> SimilarityTransform:
    """Return the thermal-frame motion from a baseline to a current marker pair."""
    baseline_dx = baseline.right_xy[0] - baseline.left_xy[0]
    baseline_dy = baseline.right_xy[1] - baseline.left_xy[1]
    current_dx = current.right_xy[0] - current.left_xy[0]
    current_dy = current.right_xy[1] - current.left_xy[1]
    baseline_distance = math.hypot(baseline_dx, baseline_dy)
    current_distance = math.hypot(current_dx, current_dy)
    if baseline_distance <= 1e-6 or current_distance <= 1e-6:
        raise ValueError("marker pairs must have non-zero distance")

    scale = current_distance / baseline_distance
    angle_deg = math.degrees(math.atan2(current_dy, current_dx) - math.atan2(baseline_dy, baseline_dx))
    angle_rad = math.radians(angle_deg)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    base_x, base_y = baseline.left_xy
    current_x, current_y = current.left_xy
    translation = (
        current_x - scale * (cos_angle * base_x - sin_angle * base_y),
        current_y - scale * (sin_angle * base_x + cos_angle * base_y),
    )
    return SimilarityTransform(scale=scale, angle_deg=angle_deg, translation_xy=translation)
