"""Pre-registered geometry and thermal helpers for the desktop foam experiment."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import cv2
import numpy as np


COMPRESSION_TARGETS = {
    "R": 0.0,
    "N": 0.0,
    "C0": 0.0,
    "C10": 10.0,
    "C20": 20.0,
    "C30": 30.0,
}

STEADY_ORDERS = {
    1: ("C20", "N", "C0", "C30", "C10", "C0", "C20", "N", "C10", "C30"),
    2: ("N", "C10", "C30", "C0", "C20", "C30", "N", "C20", "C0", "C10"),
    3: ("C30", "C0", "N", "C20", "C10", "C20", "C10", "C30", "N", "C0"),
}


@dataclass(frozen=True)
class PixelROI:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError("ROI x and y must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ROI width and height must be positive")

    @property
    def x_end(self) -> int:
        return self.x + self.width

    @property
    def y_end(self) -> int:
        return self.y + self.height

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2.0, self.y + self.height / 2.0

    def slices(self) -> tuple[slice, slice]:
        return slice(self.y, self.y_end), slice(self.x, self.x_end)

    def is_inside(self, shape: tuple[int, int]) -> bool:
        height, width = shape
        return self.x_end <= width and self.y_end <= height

    def as_list(self) -> list[int]:
        return [self.x, self.y, self.width, self.height]


@dataclass(frozen=True)
class FoamCompressionStep:
    block: str
    phase: str
    state: str
    hold_s: float
    sequence_id: int
    step_index: int
    pulse_index: int = 0

    @property
    def target_compression_pct(self) -> float:
        return COMPRESSION_TARGETS[self.state]

    @property
    def name(self) -> str:
        return f"{self.block}_s{self.sequence_id:02d}_step{self.step_index:02d}_{self.state.lower()}"


@dataclass(frozen=True)
class MarkerObservation:
    left_xy: tuple[float, float]
    right_xy: tuple[float, float]
    left_area_px: float
    right_area_px: float

    @property
    def distance_px(self) -> float:
        return float(math.dist(self.left_xy, self.right_xy))

    @property
    def midpoint_xy(self) -> tuple[float, float]:
        return (
            (self.left_xy[0] + self.right_xy[0]) / 2.0,
            (self.left_xy[1] + self.right_xy[1]) / 2.0,
        )

    @property
    def angle_deg(self) -> float:
        return float(math.degrees(math.atan2(self.right_xy[1] - self.left_xy[1], self.right_xy[0] - self.left_xy[0])))


@dataclass
class StableCompressionGate:
    target_pct: float
    tolerance_pct: float
    required_s: float
    _stable_since: float | None = None
    _stable_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.tolerance_pct <= 0:
            raise ValueError("tolerance_pct must be positive")
        if self.required_s <= 0:
            raise ValueError("required_s must be positive")

    @property
    def stable_seconds(self) -> float:
        return self._stable_seconds

    def update(self, *, timestamp: float, compression_pct: float | None) -> bool:
        if compression_pct is None or abs(compression_pct - self.target_pct) > self.tolerance_pct:
            self._stable_since = None
            self._stable_seconds = 0.0
            return False
        if self._stable_since is None:
            self._stable_since = timestamp
            self._stable_seconds = 0.0
            return False
        self._stable_seconds = max(0.0, timestamp - self._stable_since)
        return self._stable_seconds >= self.required_s


@dataclass
class HoldToleranceGate:
    """Keep a completed hold valid through a short camera/marker outlier."""

    target_pct: float
    tolerance_pct: float
    max_gap_s: float
    _out_of_range_since: float | None = None

    def __post_init__(self) -> None:
        if self.tolerance_pct <= 0:
            raise ValueError("tolerance_pct must be positive")
        if self.max_gap_s < 0:
            raise ValueError("max_gap_s must be non-negative")

    def update(self, *, timestamp: float, compression_pct: float | None) -> bool:
        in_range = (
            compression_pct is not None
            and abs(compression_pct - self.target_pct) <= self.tolerance_pct
        )
        if in_range:
            self._out_of_range_since = None
            return True
        if self._out_of_range_since is None:
            self._out_of_range_since = timestamp
            return True
        return timestamp - self._out_of_range_since <= self.max_gap_s


@dataclass(frozen=True)
class FrozenThermalRegions:
    """Thermal regions selected before recording and kept fixed through analysis."""

    foam_bbox: PixelROI
    foam_center: PixelROI
    left_contact: PixelROI
    right_contact: PixelROI
    background: PixelROI
    room_reference: PixelROI
    warm_reference: PixelROI
    overlay_y_exclusive: int = 105

    def sampling_issues(self, shape: tuple[int, int]) -> list[str]:
        issues: list[str] = []
        named = {
            "foam_bbox": self.foam_bbox,
            "foam_center": self.foam_center,
            "left_contact": self.left_contact,
            "right_contact": self.right_contact,
            "background": self.background,
            "room_reference": self.room_reference,
            "warm_reference": self.warm_reference,
        }
        for name, roi in named.items():
            if not roi.is_inside(shape):
                issues.append(f"{name} lies outside thermal frame")
            if roi.y_end > self.overlay_y_exclusive:
                issues.append(f"{name} overlaps thermal overlay band")
        if not _contains(self.foam_bbox, self.foam_center):
            issues.append("foam_center must lie inside foam_bbox")
        return issues

    def preflight_issues(self, shape: tuple[int, int]) -> list[str]:
        issues = self.sampling_issues(shape)
        if self.foam_bbox.width < 24:
            issues.append("foam_bbox is narrower than 24 thermal pixels")
        foam_x, foam_y = self.foam_bbox.center
        if not 75.0 <= foam_x <= 85.0 or not 55.0 <= foam_y <= 65.0:
            issues.append("foam_bbox center is outside the required x=80+/-5, y=60+/-5 placement")
        if self.background.width < 15 or self.background.height < 15:
            issues.append("background must be at least 15x15 thermal pixels")
        for name, roi in (("room_reference", self.room_reference), ("warm_reference", self.warm_reference)):
            if roi.width < 12 or roi.height < 12:
                issues.append(f"{name} must be at least 12x12 thermal pixels")
        return issues

    def metadata(self) -> dict[str, list[int] | int]:
        return {
            "foam_bbox": self.foam_bbox.as_list(),
            "foam_center": self.foam_center.as_list(),
            "left_contact": self.left_contact.as_list(),
            "right_contact": self.right_contact.as_list(),
            "background": self.background.as_list(),
            "room_reference": self.room_reference.as_list(),
            "warm_reference": self.warm_reference.as_list(),
            "overlay_y_exclusive": self.overlay_y_exclusive,
        }


def _contains(outer: PixelROI, inner: PixelROI) -> bool:
    return (
        outer.x <= inner.x
        and outer.y <= inner.y
        and outer.x_end >= inner.x_end
        and outer.y_end >= inner.y_end
    )


def build_recording_plan(recording_index: int) -> list[FoamCompressionStep]:
    """Return the fixed before-data-collection protocol for recording 1, 2, or 3."""
    if recording_index not in STEADY_ORDERS:
        raise ValueError("recording_index must be 1, 2, or 3")
    steps: list[FoamCompressionStep] = [
        FoamCompressionStep("start_drift", "baseline", "R", 30.0, 0, 0),
    ]
    for index, state in enumerate(STEADY_ORDERS[recording_index]):
        steps.append(FoamCompressionStep("steady_state", "release", "R", 6.0, 1, index * 2))
        steps.append(FoamCompressionStep("steady_state", "target", state, 6.0, 1, index * 2 + 1))
    steps.append(FoamCompressionStep("middle_drift", "baseline", "R", 20.0, 2, 0))
    for index, state in enumerate(("C0", "C10", "C20", "C30", "C20", "C10", "C0", "R")):
        hold_s = 6.0 if state == "C30" else 12.0 if state == "R" else 4.0
        steps.append(FoamCompressionStep("hysteresis", "target", state, hold_s, 3, index))
    for pulse_index in range(1, 5):
        steps.append(FoamCompressionStep("release_pulses", "release", "R", 8.0, 4, (pulse_index - 1) * 3, pulse_index))
        steps.append(FoamCompressionStep("release_pulses", "target", "C30", 6.0, 4, (pulse_index - 1) * 3 + 1, pulse_index))
        steps.append(FoamCompressionStep("release_pulses", "release", "R", 12.0, 4, (pulse_index - 1) * 3 + 2, pulse_index))
    steps.append(FoamCompressionStep("end_drift", "baseline", "R", 30.0, 5, 0))
    return steps


def _marker_center(frame: np.ndarray, roi: PixelROI, *, max_gray: int, min_area_px: int) -> tuple[tuple[float, float], float] | None:
    if frame.ndim != 3 or not roi.is_inside(frame.shape[:2]):
        return None
    crop = frame[roi.slices()]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = (gray <= max_gray).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    candidates = [
        index
        for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= min_area_px
    ]
    if not candidates:
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    scored: list[tuple[float, int]] = []
    for index in candidates:
        component = (labels == index).astype(np.uint8)
        expanded = cv2.dilate(component, kernel)
        ring = (expanded.astype(bool) & ~component.astype(bool))
        if not np.any(ring):
            continue
        # The OAK image is not white-balanced to 255. On the current setup's
        # matte white tabs, the ring around a dot is typically 120--160.
        white_fraction = float(np.mean(gray[ring] >= 120))
        contours, _hierarchy = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = cv2.arcLength(contours[0], True) if contours else 0.0
        area = float(stats[index, cv2.CC_STAT_AREA])
        circularity = 4.0 * math.pi * area / (perimeter * perimeter) if perimeter else 0.0
        # White tabs make the intended dot both locally white-surrounded and compact.
        score = white_fraction * max(circularity, 0.0) / math.sqrt(area)
        if white_fraction >= 0.60:
            scored.append((score, index))
    if not scored:
        return None
    _score, index = max(scored)
    center_x, center_y = centers[index]
    return (roi.x + float(center_x), roi.y + float(center_y)), float(stats[index, cv2.CC_STAT_AREA])


def detect_marker_pair(
    frame: np.ndarray,
    *,
    left_roi: PixelROI,
    right_roi: PixelROI,
    max_gray: int = 110,
    min_area_px: int = 40,
) -> MarkerObservation | None:
    left = _marker_center(frame, left_roi, max_gray=max_gray, min_area_px=min_area_px)
    right = _marker_center(frame, right_roi, max_gray=max_gray, min_area_px=min_area_px)
    if left is None or right is None:
        return None
    return MarkerObservation(
        left_xy=left[0],
        right_xy=right[0],
        left_area_px=left[1],
        right_area_px=right[1],
    )


def _centered_dark_marker(
    frame: np.ndarray,
    roi: PixelROI,
    *,
    max_gray: int,
    min_area_px: int,
    max_area_px: int,
) -> tuple[tuple[float, float], float] | None:
    if frame.ndim != 3 or not roi.is_inside(frame.shape[:2]):
        return None
    if max_area_px < min_area_px:
        raise ValueError("max_area_px must be at least min_area_px")
    gray = cv2.cvtColor(frame[roi.slices()], cv2.COLOR_BGR2GRAY)
    mask = (gray <= max_gray).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    roi_center = np.asarray((roi.width / 2.0, roi.height / 2.0), dtype=np.float64)
    roi_diagonal = max(1.0, math.hypot(roi.width, roi.height))
    candidates: list[tuple[float, int]] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if not min_area_px <= area <= max_area_px:
            continue
        component = (labels == index).astype(np.uint8)
        contours, _hierarchy = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = cv2.arcLength(contours[0], True) if contours else 0.0
        circularity = 4.0 * math.pi * area / (perimeter * perimeter) if perimeter else 0.0
        distance = float(np.linalg.norm(centers[index] - roi_center)) / roi_diagonal
        candidates.append((circularity - distance, index))
    if not candidates:
        return None
    _score, index = max(candidates)
    center_x, center_y = centers[index]
    return (roi.x + float(center_x), roi.y + float(center_y)), float(stats[index, cv2.CC_STAT_AREA])


def detect_centered_dark_marker_pair(
    frame: np.ndarray,
    *,
    left_roi: PixelROI,
    right_roi: PixelROI,
    max_gray: int = 90,
    min_area_px: int = 20,
    max_area_px: int = 2_000,
) -> MarkerObservation | None:
    """Detect tight-ROI dark dots when FLIR RGB cannot provide a white surround."""
    left = _centered_dark_marker(
        frame,
        left_roi,
        max_gray=max_gray,
        min_area_px=min_area_px,
        max_area_px=max_area_px,
    )
    right = _centered_dark_marker(
        frame,
        right_roi,
        max_gray=max_gray,
        min_area_px=min_area_px,
        max_area_px=max_area_px,
    )
    if left is None or right is None:
        return None
    return MarkerObservation(
        left_xy=left[0],
        right_xy=right[0],
        left_area_px=left[1],
        right_area_px=right[1],
    )


def compression_percent(distance_px: float, *, d0_px: float) -> float:
    if d0_px <= 0:
        raise ValueError("d0_px must be positive")
    return 100.0 * (d0_px - distance_px) / d0_px


def reference_normalized_features(
    scalar: np.ndarray,
    regions: FrozenThermalRegions,
    *,
    strict_preflight: bool = True,
) -> dict[str, float]:
    if scalar.ndim != 2:
        raise ValueError("scalar thermal image must be two dimensional")
    issues = regions.preflight_issues(scalar.shape) if strict_preflight else regions.sampling_issues(scalar.shape)
    if issues:
        raise ValueError("invalid frozen thermal regions: " + "; ".join(issues))

    medians = {
        "foam_center": float(np.median(scalar[regions.foam_center.slices()])),
        "left_contact": float(np.median(scalar[regions.left_contact.slices()])),
        "right_contact": float(np.median(scalar[regions.right_contact.slices()])),
        "background": float(np.median(scalar[regions.background.slices()])),
        "room_reference": float(np.median(scalar[regions.room_reference.slices()])),
        "warm_reference": float(np.median(scalar[regions.warm_reference.slices()])),
    }
    denominator = medians["warm_reference"] - medians["room_reference"]
    if abs(denominator) < 1e-6:
        normalized = {f"{name}_norm": float("nan") for name in ("foam_center", "left_contact", "right_contact", "background")}
    else:
        normalized = {
            f"{name}_norm": (value - medians["room_reference"]) / denominator
            for name, value in medians.items()
            if name not in {"room_reference", "warm_reference"}
        }
    return {
        **{f"{name}_median": value for name, value in medians.items()},
        "reference_span": denominator,
        **normalized,
    }


def thermal_frame_hash(frame: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(frame).tobytes()).hexdigest()
