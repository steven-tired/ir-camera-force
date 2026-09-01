import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ir_force.ir_hand_calibration import ProjectionCalibration
from ir_force.ir_hand_roi import (
    PressureROI,
    select_pressure_roi,
    select_thermal_blob_roi,
)
from ir_force.types import LandmarksData


def _calibration():
    return ProjectionCalibration(
        coeff_x=(0.0, 160.0, 0.0, 0.0),
        coeff_y=(0.0, 0.0, 128.0, 0.0),
        rms_error_px=2.0,
        max_error_px=4.0,
        sample_count=12,
        image_size=(160, 128),
    )


def _landmarks(thumb=(0.45, 0.50, 0.6), index=(0.50, 0.52, 0.6)):
    image_xy = np.zeros((21, 2), dtype=float)
    depth_m = np.full((21,), np.nan, dtype=float)
    image_xy[4] = thumb[:2]
    image_xy[8] = index[:2]
    depth_m[4] = thumb[2]
    depth_m[8] = index[2]
    return LandmarksData(np.zeros((21, 3)), True, image_xy=image_xy, depth_m=depth_m)


def test_select_pressure_roi_uses_fingertip_pair_when_both_depths_are_valid():
    selection = select_pressure_roi(_landmarks(), _calibration(), frame_shape=(128, 160, 3))

    assert selection.mode == "tips"
    assert selection.quality == 1.0
    assert isinstance(selection.roi, PressureROI)
    assert selection.roi.x <= 72 <= selection.roi.x_end
    assert selection.roi.x <= 80 <= selection.roi.x_end


def test_select_pressure_roi_returns_missing_for_no_image_metadata():
    landmarks = LandmarksData(np.zeros((21, 3)), True)

    selection = select_pressure_roi(landmarks, _calibration(), frame_shape=(128, 160, 3))

    assert selection.roi is None
    assert selection.quality == 0.0
    assert selection.mode == "missing_oak_metadata"


def test_select_pressure_roi_falls_back_to_single_tip_when_one_depth_is_missing():
    landmarks = _landmarks()
    landmarks.depth_m[8] = np.nan

    selection = select_pressure_roi(landmarks, _calibration(), frame_shape=(128, 160, 3))

    assert selection.mode == "single_tip"
    assert 0.0 < selection.quality < 1.0
    assert selection.roi is not None


def test_select_pressure_roi_rejects_out_of_fov_tip_projection():
    landmarks = _landmarks(thumb=(-0.01, 0.50, 0.6))

    selection = select_pressure_roi(landmarks, _calibration(), frame_shape=(128, 160, 3))

    assert selection.roi is None
    assert selection.quality == 0.0
    assert selection.mode == "projection_out_of_fov"


def test_select_pressure_roi_reduces_quality_when_corridor_is_clipped_at_border():
    landmarks = _landmarks(thumb=(0.01, 0.50, 0.6), index=(0.04, 0.52, 0.6))

    selection = select_pressure_roi(landmarks, _calibration(), frame_shape=(128, 160, 3))

    assert selection.mode == "tips"
    assert selection.roi is not None
    assert selection.roi.x == 0
    assert 0.0 < selection.quality < 1.0


def test_select_pressure_roi_reduces_single_tip_quality_when_clipped_at_border():
    landmarks = _landmarks(thumb=(0.01, 0.50, 0.6))
    landmarks.depth_m[8] = np.nan

    selection = select_pressure_roi(landmarks, _calibration(), frame_shape=(128, 160, 3))

    assert selection.mode == "single_tip"
    assert selection.roi is not None
    assert selection.roi.x == 0
    assert 0.0 < selection.quality < 0.5


# ---------------------------------------------------------------------------
# Thermal-blob ROI (Lepton path: no cross-camera calibration required)
# ---------------------------------------------------------------------------


def _blob_frame(background: int = 8000) -> np.ndarray:
    return np.full((120, 160), background, dtype=np.uint16)


def test_thermal_blob_roi_finds_hot_contact_patch():
    frame = _blob_frame()
    frame[50:60, 70:85] = 8500
    selection = select_thermal_blob_roi(frame)

    assert selection.mode == "blob"
    assert selection.roi is not None
    assert selection.quality > 0.0
    ys, xs = selection.roi.slices()
    assert ys.start <= 50 and ys.stop >= 60
    assert xs.start <= 70 and xs.stop >= 85


def test_thermal_blob_roi_reports_no_hotspot_on_flat_frame():
    selection = select_thermal_blob_roi(_blob_frame())
    assert selection.roi is None
    assert selection.mode == "blob_no_hotspot"
    assert selection.quality == 0.0


def test_thermal_blob_roi_rejects_tiny_and_giant_blobs():
    tiny = _blob_frame()
    tiny[10, 10] = 9000
    tiny_selection = select_thermal_blob_roi(tiny, min_area_px=4)
    assert tiny_selection.roi is None
    assert tiny_selection.mode == "blob_too_small"

    giant = _blob_frame()
    giant[:, :100] = 9000
    giant_selection = select_thermal_blob_roi(giant, max_area_px=900)
    assert giant_selection.roi is None
    assert giant_selection.mode == "blob_too_large"


def test_thermal_blob_roi_picks_largest_component():
    frame = _blob_frame()
    frame[10:14, 10:14] = 8500       # small distractor
    frame[80:95, 100:120] = 8500     # main contact patch
    selection = select_thermal_blob_roi(frame)

    assert selection.mode == "blob"
    ys, xs = selection.roi.slices()
    assert 80 - 3 <= ys.start <= 80 and 100 - 3 <= xs.start <= 100
