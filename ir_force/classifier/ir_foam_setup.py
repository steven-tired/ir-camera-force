"""Shared ROI layout helpers for the fixed-geometry foam experiment."""

from __future__ import annotations

from collections.abc import Mapping

from .ir_foam_compression import FrozenThermalRegions, PixelROI


THERMAL_ROI_KEYS = (
    "foam_bbox",
    "foam_center",
    "left_contact",
    "right_contact",
    "background",
    "room_reference",
    "warm_reference",
)
OAK_ROI_KEYS = ("oak_left_marker", "oak_right_marker")
ROI_KEYS = THERMAL_ROI_KEYS + OAK_ROI_KEYS

RECORDER_FLAGS = {
    "foam_bbox": "--thermal-foam-bbox",
    "foam_center": "--thermal-foam-roi",
    "left_contact": "--thermal-left-contact-roi",
    "right_contact": "--thermal-right-contact-roi",
    "background": "--thermal-background-roi",
    "room_reference": "--thermal-room-reference-roi",
    "warm_reference": "--thermal-warm-reference-roi",
    "oak_left_marker": "--oak-left-marker-roi",
    "oak_right_marker": "--oak-right-marker-roi",
}


def default_layout() -> dict[str, PixelROI]:
    """Return provisional regions that obey the thermal placement constraints."""
    return {
        "foam_bbox": PixelROI(68, 40, 28, 30),
        "foam_center": PixelROI(75, 48, 14, 18),
        "left_contact": PixelROI(68, 48, 6, 18),
        "right_contact": PixelROI(90, 48, 6, 18),
        "background": PixelROI(5, 5, 15, 15),
        "room_reference": PixelROI(15, 15, 12, 12),
        "warm_reference": PixelROI(130, 15, 12, 12),
        # Verified from the released and compressed OAK trajectory: the left
        # dot moves from about (235, 161) to (281, 155).
        "oak_left_marker": PixelROI(180, 90, 150, 140),
        "oak_right_marker": PixelROI(360, 100, 80, 100),
    }


def frozen_regions_from_layout(layout: Mapping[str, PixelROI]) -> FrozenThermalRegions:
    return FrozenThermalRegions(**{key: layout[key] for key in THERMAL_ROI_KEYS})


def roi_from_drag(
    start_xy: tuple[int, int],
    end_xy: tuple[int, int],
    *,
    scale: int,
    frame_shape: tuple[int, int],
) -> PixelROI:
    """Convert a display-space drag into a clamped inclusive pixel ROI."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    height, width = frame_shape
    if height <= 0 or width <= 0:
        raise ValueError("frame dimensions must be positive")
    start_x = min(max(start_xy[0] // scale, 0), width - 1)
    start_y = min(max(start_xy[1] // scale, 0), height - 1)
    end_x = min(max(end_xy[0] // scale, 0), width - 1)
    end_y = min(max(end_xy[1] // scale, 0), height - 1)
    left, right = sorted((start_x, end_x))
    top, bottom = sorted((start_y, end_y))
    return PixelROI(left, top, right - left + 1, bottom - top + 1)


def layout_metadata(layout: Mapping[str, PixelROI]) -> dict[str, list[int]]:
    return {key: layout[key].as_list() for key in ROI_KEYS}


def layout_from_metadata(payload: Mapping[str, object]) -> dict[str, PixelROI]:
    layout: dict[str, PixelROI] = {}
    for key in ROI_KEYS:
        values = payload[key]
        if not isinstance(values, list) or len(values) != 4:
            raise ValueError(f"layout field {key} must be a four-item list")
        layout[key] = PixelROI(*(int(value) for value in values))
    return layout


def recorder_roi_arguments(layout: Mapping[str, PixelROI]) -> list[str]:
    arguments: list[str] = []
    for key in ROI_KEYS:
        arguments.extend((RECORDER_FLAGS[key], ",".join(str(value) for value in layout[key].as_list())))
    return arguments
