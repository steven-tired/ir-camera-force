import numpy as np
import pytest

from ir_force.single_finger_click_roi import (
    TemplateTracker,
    rois_from_clicks,
)


def _finger_frame(*, shift_x=0, shift_y=0, distal_effect=0, warm_background=True):
    """A finger pointing DOWN, with warm clutter on the far left.

    Both features broke the frozen v2 rule in hold_check_01: the leftmost 3%
    of a globally thresholded blob lands on the clutter, not the fingertip.
    """
    frame = np.full((120, 160), 29_000, dtype=np.int32)
    if warm_background:
        frame[20:100, 0:30] = 29_450
    frame[20:55, 60 + shift_x : 100 + shift_x] = 29_600  # fist
    frame[55 + shift_y : 95 + shift_y, 74 + shift_x : 86 + shift_x] = 29_620
    # distal ROI is axial 5-13 px up from the clicked tip at v=93
    frame[79 + shift_y : 90 + shift_y, 74 + shift_x : 86 + shift_x] += distal_effect
    return frame.astype(np.uint16)


TIP = (80.0, 93.0)
ALONG = (80.0, 60.0)
REFERENCE = (30.0, 110.0)


def test_clicked_rois_land_on_the_finger_not_the_warm_background():
    anchor = rois_from_clicks(_finger_frame(), TIP, ALONG, REFERENCE)

    distal_v = np.nonzero(anchor["distal_mask"])[0]
    proximal_v = np.nonzero(anchor["proximal_mask"])[0]
    distal_u = np.nonzero(anchor["distal_mask"])[1]

    assert anchor["distal_pixel_count"] >= 20
    assert anchor["proximal_pixel_count"] >= 20
    assert not np.any(anchor["distal_mask"] & anchor["proximal_mask"])
    # distal sits nearer the tip (larger v, the finger points down)
    assert distal_v.mean() > proximal_v.mean()
    # and on the finger column, not on the left-hand clutter
    assert distal_u.min() >= 70
    assert anchor["direction_uv"][1] < -0.9
    assert 8.0 <= anchor["finger_width_px"] <= 16.0


def test_reference_patch_is_rejected_when_it_overlaps_the_finger():
    with pytest.raises(ValueError, match="reference_patch_touches_finger"):
        rois_from_clicks(_finger_frame(), TIP, ALONG, (80.0, 80.0))


def test_clicks_too_close_together_cannot_define_an_axis():
    with pytest.raises(ValueError, match="clicks_too_close"):
        rois_from_clicks(_finger_frame(), TIP, (80.0, 88.0), REFERENCE)


def test_click_outside_the_frame_is_rejected():
    with pytest.raises(ValueError, match="tip_uv_out_of_frame"):
        rois_from_clicks(_finger_frame(), (200.0, 93.0), ALONG, REFERENCE)


def test_tracker_follows_a_translating_finger():
    frame = _finger_frame()
    anchor = rois_from_clicks(frame, TIP, ALONG, REFERENCE)
    tracker = TemplateTracker(frame, anchor)

    moved = tracker.measure(_finger_frame(shift_x=3, shift_y=2))

    assert moved["tracking_valid"] is True
    assert moved["shift_uv"] == [3, 2]
    assert moved["template_score"] > 0.9


def test_a_cooling_fingertip_does_not_drag_the_template():
    """TM_CCOEFF_NORMED is invariant to affine intensity change."""
    frame = _finger_frame()
    anchor = rois_from_clicks(frame, TIP, ALONG, REFERENCE)
    tracker = TemplateTracker(frame, anchor)

    cooled = tracker.measure(_finger_frame(distal_effect=-60))

    assert cooled["tracking_valid"] is True
    assert cooled["shift_uv"] == [0, 0]
    assert cooled["primary_signal_count"] < -40


def test_tracking_fails_closed_when_the_finger_leaves_the_search_window():
    frame = _finger_frame()
    anchor = rois_from_clicks(frame, TIP, ALONG, REFERENCE)
    tracker = TemplateTracker(frame, anchor, search_radius_px=3)

    lost = tracker.measure(_finger_frame(shift_x=25))

    assert lost["tracking_valid"] is False
    assert lost["primary_signal_count"] is None
    assert lost["tracking_reasons"]


def test_reference_count_is_reported_but_not_folded_into_the_primary():
    frame = _finger_frame()
    anchor = rois_from_clicks(frame, TIP, ALONG, REFERENCE)
    tracker = TemplateTracker(frame, anchor)

    measured = tracker.measure(frame)

    assert measured["reference_count"] is not None
    assert measured["primary_signal_count"] == (
        measured["distal_count"] - measured["proximal_count"]
    )
