"""Operator-seeded thermal ROIs tracked by normalised cross-correlation.

The frozen v2 path derives the fingertip from a global segmentation
(`frame median + 100 counts`) and the leftmost 3% of the resulting blob. Both
assumptions broke in `single_finger_hold_check_01`: warm background merged into
the hand component, and the finger pointed down rather than left, so the ROIs
landed on the back of the fist and the measured effect was 0.25 counts.

This module removes both assumptions. The operator clicks the fingertip and a
point further up the same finger, which fixes the axis directly, and the ROIs
are then followed with template matching rather than by re-deriving the tip on
every frame. `TM_CCOEFF_NORMED` is invariant to affine intensity changes, so a
finger that cools by tens of counts does not drag the template with it.

Nothing here segments, actuates, or infers pressure.
"""

from __future__ import annotations

import cv2
import numpy as np


FRAME_SHAPE = (120, 160)
DISTAL_AXIAL_RANGE_PX = (5.0, 13.0)
PROXIMAL_AXIAL_RANGE_PX = (15.0, 23.0)
FINGER_HALF_WIDTH_PX = 3.5
TEMPLATE_PAD_PX = 8
# hold_check_05 round 2 lost 121 of 123 HARD frames to the search
# boundary: a hard press moves the fingertip further than 6 px.
SEARCH_RADIUS_PX = 12
MIN_TEMPLATE_SCORE = 0.5
MIN_ROI_PIXELS = 20
MIN_CLICK_SEPARATION_PX = 12.0
REFERENCE_PATCH_PX = 5


def _validated_frame(frame) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.shape != FRAME_SHAPE or frame.dtype != np.uint16:
        raise ValueError("thermal_frame_invalid")
    return frame


def _finite_uv(value, *, label: str) -> np.ndarray:
    uv = np.asarray(value, dtype=float)
    if uv.shape != (2,) or not np.all(np.isfinite(uv)):
        raise ValueError(f"{label}_invalid")
    if not (0 <= uv[0] < FRAME_SHAPE[1] and 0 <= uv[1] < FRAME_SHAPE[0]):
        raise ValueError(f"{label}_out_of_frame")
    return uv


def _local_finger_mask(frame: np.ndarray, tip: np.ndarray, along: np.ndarray):
    """Otsu-threshold a box around the clicked finger segment, keep its blob.

    Local rather than global, so warm background elsewhere in the scene cannot
    merge into the finger the way it does with a whole-frame threshold.
    """
    points = np.stack([tip, along])
    u0 = int(max(0, np.floor(points[:, 0].min()) - TEMPLATE_PAD_PX))
    u1 = int(min(FRAME_SHAPE[1], np.ceil(points[:, 0].max()) + TEMPLATE_PAD_PX + 1))
    v0 = int(max(0, np.floor(points[:, 1].min()) - TEMPLATE_PAD_PX))
    v1 = int(min(FRAME_SHAPE[0], np.ceil(points[:, 1].max()) + TEMPLATE_PAD_PX + 1))
    box = frame[v0:v1, u0:u1]
    if box.size == 0:
        raise ValueError("finger_box_empty")
    scaled = cv2.normalize(box, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _score, binary = cv2.threshold(
        scaled,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    count, labels = cv2.connectedComponents((binary > 0).astype(np.uint8))
    if count <= 1:
        raise ValueError("finger_component_missing")
    midpoint = np.rint((tip + along) / 2.0).astype(int)
    label = int(labels[midpoint[1] - v0, midpoint[0] - u0])
    if label == 0:
        # The midpoint fell on the cold side; take the blob holding the tip.
        tip_pixel = np.rint(tip).astype(int)
        label = int(labels[tip_pixel[1] - v0, tip_pixel[0] - u0])
    if label == 0:
        raise ValueError("finger_component_missing")
    mask = np.zeros(FRAME_SHAPE, dtype=bool)
    mask[v0:v1, u0:u1] = labels == label
    return mask


def rois_from_clicks(frame, tip_uv, along_uv, reference_uv=None) -> dict:
    """Build distal/proximal ROIs from two clicks on one finger.

    ``tip_uv`` is the fingertip, ``along_uv`` a point further up the same
    finger. ``reference_uv`` optionally marks a surface patch used only as a
    drift diagnostic; it is never subtracted from the primary value.
    """
    frame = _validated_frame(frame)
    tip = _finite_uv(tip_uv, label="tip_uv")
    along = _finite_uv(along_uv, label="along_uv")
    axis = along - tip
    length = float(np.linalg.norm(axis))
    if length < MIN_CLICK_SEPARATION_PX:
        raise ValueError("clicks_too_close")
    direction = axis / length
    finger_mask = _local_finger_mask(frame, tip, along)
    interior = cv2.erode(
        finger_mask.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ).astype(bool)

    yy, xx = np.indices(FRAME_SHAPE)
    delta_x = xx - tip[0]
    delta_y = yy - tip[1]
    axial = delta_x * direction[0] + delta_y * direction[1]
    perpendicular = np.abs(delta_x * (-direction[1]) + delta_y * direction[0])
    common = interior & (perpendicular <= FINGER_HALF_WIDTH_PX)
    distal = (
        common
        & (axial >= DISTAL_AXIAL_RANGE_PX[0])
        & (axial < DISTAL_AXIAL_RANGE_PX[1])
    )
    proximal = (
        common
        & (axial >= PROXIMAL_AXIAL_RANGE_PX[0])
        & (axial < PROXIMAL_AXIAL_RANGE_PX[1])
    )
    for label, roi in (("distal", distal), ("proximal", proximal)):
        if np.count_nonzero(roi) < MIN_ROI_PIXELS:
            raise ValueError(f"{label}_interior_pixels_insufficient")

    distance = cv2.distanceTransform(
        finger_mask.astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    radii = [
        float(distance[int(round(point[1])), int(round(point[0]))])
        for point in (
            tip + offset * direction
            for offset in np.linspace(
                DISTAL_AXIAL_RANGE_PX[0] + 1.0,
                DISTAL_AXIAL_RANGE_PX[1] - 1.0,
                7,
            )
        )
        if 0 <= int(round(point[0])) < FRAME_SHAPE[1]
        and 0 <= int(round(point[1])) < FRAME_SHAPE[0]
    ]
    positive = [radius for radius in radii if radius > 0.0]
    if not positive:
        raise ValueError("finger_interior_missing")

    reference = None
    if reference_uv is not None:
        centre = _finite_uv(reference_uv, label="reference_uv")
        radius = REFERENCE_PATCH_PX // 2
        u, v = (int(round(value)) for value in centre)
        reference = np.zeros(FRAME_SHAPE, dtype=bool)
        reference[
            max(0, v - radius) : v + radius + 1,
            max(0, u - radius) : u + radius + 1,
        ] = True
        if np.any(reference & finger_mask):
            raise ValueError("reference_patch_touches_finger")

    return {
        "tip_uv": [float(tip[0]), float(tip[1])],
        "along_uv": [float(along[0]), float(along[1])],
        "direction_uv": [float(direction[0]), float(direction[1])],
        "clicked_length_px": length,
        "finger_mask": finger_mask,
        "interior_mask": interior,
        "distal_mask": distal,
        "proximal_mask": proximal,
        "reference_mask": reference,
        "reference_uv": (
            None if reference_uv is None else [float(centre[0]), float(centre[1])]
        ),
        "finger_width_px": 2.0 * float(np.median(positive)),
        "distal_pixel_count": int(np.count_nonzero(distal)),
        "proximal_pixel_count": int(np.count_nonzero(proximal)),
    }


def _template_box(anchor: dict) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(anchor["distal_mask"] | anchor["proximal_mask"])
    u0 = int(max(0, xs.min() - TEMPLATE_PAD_PX))
    u1 = int(min(FRAME_SHAPE[1], xs.max() + TEMPLATE_PAD_PX + 1))
    v0 = int(max(0, ys.min() - TEMPLATE_PAD_PX))
    v1 = int(min(FRAME_SHAPE[0], ys.max() + TEMPLATE_PAD_PX + 1))
    return u0, v0, u1, v1


def _shift_mask(mask: np.ndarray, shift_u: int, shift_v: int) -> np.ndarray:
    shifted = np.zeros_like(mask)
    x0 = max(0, -shift_u)
    x1 = min(mask.shape[1], mask.shape[1] - shift_u)
    y0 = max(0, -shift_v)
    y1 = min(mask.shape[0], mask.shape[0] - shift_v)
    if x0 >= x1 or y0 >= y1:
        return shifted
    shifted[y0 + shift_v : y1 + shift_v, x0 + shift_u : x1 + shift_u] = mask[
        y0:y1, x0:x1
    ]
    return shifted


class TemplateTracker:
    """Follow the clicked ROIs by normalised cross-correlation.

    The template covers the finger edges around both ROIs, so matching is
    driven by finger-versus-background contrast rather than by the few-count
    signal being measured.
    """

    def __init__(
        self,
        anchor_frame,
        anchor: dict,
        *,
        search_radius_px: int = SEARCH_RADIUS_PX,
        min_score: float = MIN_TEMPLATE_SCORE,
    ):
        self.anchor = anchor
        self.search_radius_px = int(search_radius_px)
        self.min_score = float(min_score)
        frame = _validated_frame(anchor_frame)
        self._box = _template_box(anchor)
        u0, v0, u1, v1 = self._box
        self._template = frame[v0:v1, u0:u1].astype(np.float32)
        self._shift = (0, 0)

    def _match(self, frame: np.ndarray) -> tuple[tuple[int, int], float]:
        u0, v0, u1, v1 = self._box
        radius = self.search_radius_px
        su0 = max(0, u0 - radius)
        sv0 = max(0, v0 - radius)
        su1 = min(FRAME_SHAPE[1], u1 + radius)
        sv1 = min(FRAME_SHAPE[0], v1 + radius)
        search = frame[sv0:sv1, su0:su1].astype(np.float32)
        if (
            search.shape[0] < self._template.shape[0]
            or search.shape[1] < self._template.shape[1]
        ):
            return self._shift, -1.0
        response = cv2.matchTemplate(search, self._template, cv2.TM_CCOEFF_NORMED)
        _min_value, score, _min_loc, max_loc = cv2.minMaxLoc(response)
        return (su0 + max_loc[0] - u0, sv0 + max_loc[1] - v0), float(score)

    def measure(self, frame) -> dict:
        frame = _validated_frame(frame)
        shift, score = self._match(frame)
        at_boundary = max(abs(shift[0]), abs(shift[1])) >= self.search_radius_px
        reasons = []
        if score < self.min_score:
            reasons.append("template_score_below_threshold")
        if at_boundary:
            reasons.append("template_shift_at_search_boundary")
        distal = _shift_mask(self.anchor["distal_mask"], shift[0], shift[1])
        proximal = _shift_mask(self.anchor["proximal_mask"], shift[0], shift[1])
        for label, roi in (("distal", distal), ("proximal", proximal)):
            if np.count_nonzero(roi) < MIN_ROI_PIXELS:
                reasons.append(f"{label}_roi_left_the_frame")
        result = {
            "tracking_valid": not reasons,
            "tracking_reasons": reasons,
            "template_score": score,
            "shift_uv": [int(shift[0]), int(shift[1])],
            "shift_magnitude_px": float(np.hypot(*shift)),
            "distal_pixel_count": int(np.count_nonzero(distal)),
            "proximal_pixel_count": int(np.count_nonzero(proximal)),
            "distal_count": None,
            "proximal_count": None,
            "reference_count": None,
            "primary_signal_count": None,
        }
        if reasons:
            return result
        self._shift = shift
        distal_count = float(np.median(frame[distal]))
        proximal_count = float(np.median(frame[proximal]))
        reference = self.anchor.get("reference_mask")
        result.update(
            distal_count=distal_count,
            proximal_count=proximal_count,
            reference_count=(
                None if reference is None else float(np.median(frame[reference]))
            ),
            primary_signal_count=distal_count - proximal_count,
        )
        return result
