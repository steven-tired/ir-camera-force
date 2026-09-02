import math

import numpy as np
import pytest

from ir_force.pinch_geometry import (
    PinchGeometryReason,
    compute_pinch_geometry,
)


def _inputs():
    image_xy = np.full((21, 2), 0.5, dtype=float)
    image_xy[4] = [0.60, 0.50]   # thumb tip
    image_xy[8] = [0.70, 0.50]   # index tip
    image_xy[5] = [0.40, 0.50]   # index MCP
    image_xy[17] = [0.60, 0.50]  # pinky MCP
    depth_m = np.full(21, 0.5, dtype=float)
    depth_m[4] = 0.48
    depth_m[8] = 0.51
    return image_xy, depth_m


def test_valid_features_use_pixel_geometry_and_metric_tip_depths():
    image_xy, depth_m = _inputs()

    result = compute_pinch_geometry(image_xy, depth_m, width_px=640, height_px=480)

    assert result.valid is True
    assert result.reason is PinchGeometryReason.OK
    assert result.pinch_distance_2d_norm == pytest.approx(0.5)
    assert result.pinch_depth_delta_m == pytest.approx(0.03)


def test_pixel_scaling_respects_non_square_image_geometry():
    image_xy, depth_m = _inputs()
    image_xy[4] = [0.50, 0.50]
    image_xy[8] = [0.50, 0.60]
    image_xy[5] = [0.40, 0.50]
    image_xy[17] = [0.60, 0.50]

    result = compute_pinch_geometry(image_xy, depth_m, width_px=640, height_px=480)

    assert result.pinch_distance_2d_norm == pytest.approx(48.0 / 128.0)


def test_invalid_depth_keeps_computable_2d_feature_without_imputation():
    image_xy, depth_m = _inputs()
    depth_m[4] = np.nan

    result = compute_pinch_geometry(image_xy, depth_m, width_px=640, height_px=480)

    assert result.valid is False
    assert result.reason is PinchGeometryReason.INVALID_THUMB_TIP_DEPTH
    assert result.pinch_distance_2d_norm == pytest.approx(0.5)
    assert math.isnan(result.pinch_depth_delta_m)


def test_required_xy_checks_follow_fixed_precedence_without_clamping():
    image_xy, depth_m = _inputs()
    image_xy[4] = [np.nan, 1.2]
    depth_m[4] = np.nan
    depth_m[8] = np.nan

    result = compute_pinch_geometry(image_xy, depth_m, width_px=640, height_px=480)

    assert result.valid is False
    assert result.reason is PinchGeometryReason.NONFINITE_REQUIRED_XY
    assert math.isnan(result.pinch_distance_2d_norm)


def test_invalid_xy_keeps_computable_depth_feature():
    image_xy, depth_m = _inputs()
    image_xy[4, 0] = np.nan

    result = compute_pinch_geometry(image_xy, depth_m, width_px=640, height_px=480)

    assert result.reason is PinchGeometryReason.NONFINITE_REQUIRED_XY
    assert result.pinch_depth_delta_m == pytest.approx(0.03)


def test_xy_bounds_include_endpoints_and_reject_negative_values():
    image_xy, depth_m = _inputs()
    image_xy[4] = [0.0, 0.0]
    image_xy[8] = [1.0, 1.0]
    image_xy[5] = [0.0, 1.0]
    image_xy[17] = [1.0, 0.0]

    endpoint_result = compute_pinch_geometry(
        image_xy,
        depth_m,
        width_px=640,
        height_px=480,
    )
    assert endpoint_result.valid is True

    image_xy[4] = [-0.01, 0.0]
    negative_result = compute_pinch_geometry(
        image_xy,
        depth_m,
        width_px=640,
        height_px=480,
    )
    assert negative_result.reason is PinchGeometryReason.REQUIRED_XY_OUT_OF_BOUNDS


def test_infinite_required_values_are_invalid():
    image_xy, depth_m = _inputs()
    image_xy[4, 0] = np.inf

    xy_result = compute_pinch_geometry(image_xy, depth_m, width_px=640, height_px=480)
    assert xy_result.reason is PinchGeometryReason.NONFINITE_REQUIRED_XY

    image_xy, depth_m = _inputs()
    depth_m[8] = np.inf
    depth_result = compute_pinch_geometry(image_xy, depth_m, width_px=640, height_px=480)
    assert depth_result.reason is PinchGeometryReason.INVALID_INDEX_TIP_DEPTH


def test_zero_aperture_and_tiny_nonzero_palm_scale_are_not_thresholded():
    image_xy, depth_m = _inputs()
    image_xy[8] = image_xy[4]
    zero_aperture = compute_pinch_geometry(image_xy, depth_m, width_px=640, height_px=480)
    assert zero_aperture.valid is True
    assert zero_aperture.pinch_distance_2d_norm == 0.0

    image_xy, depth_m = _inputs()
    image_xy[5] = [0.5, 0.5]
    image_xy[17] = [0.500000001, 0.5]
    tiny_palm = compute_pinch_geometry(image_xy, depth_m, width_px=640, height_px=480)
    assert tiny_palm.valid is True
    assert math.isfinite(tiny_palm.pinch_distance_2d_norm)


def test_fractional_pixel_distances_are_not_rounded():
    image_xy, depth_m = _inputs()
    image_xy[4] = [0.500, 0.5]
    image_xy[8] = [0.501, 0.5]
    image_xy[5] = [0.400, 0.5]
    image_xy[17] = [0.600, 0.5]

    result = compute_pinch_geometry(image_xy, depth_m, width_px=641, height_px=479)

    assert result.pinch_distance_2d_norm == pytest.approx(0.005)


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda xy, _depth: xy.__setitem__(4, [1.01, 0.5]), PinchGeometryReason.REQUIRED_XY_OUT_OF_BOUNDS),
        (lambda xy, _depth: xy.__setitem__(17, xy[5]), PinchGeometryReason.DEGENERATE_PALM_SCALE),
        (lambda _xy, depth: depth.__setitem__(8, 0.0), PinchGeometryReason.INVALID_INDEX_TIP_DEPTH),
        (
            lambda _xy, depth: depth.__setitem__([4, 8], [np.nan, 0.0]),
            PinchGeometryReason.INVALID_BOTH_TIP_DEPTHS,
        ),
    ],
)
def test_frame_invalidity_reasons_are_stable(mutate, reason):
    image_xy, depth_m = _inputs()
    mutate(image_xy, depth_m)

    result = compute_pinch_geometry(image_xy, depth_m, width_px=640, height_px=480)

    assert result.valid is False
    assert result.reason is reason


@pytest.mark.parametrize(
    ("image_xy", "depth_m", "width_px", "height_px"),
    [
        (np.zeros((20, 2)), np.zeros(21), 640, 480),
        (np.zeros((21, 2)), np.zeros(20), 640, 480),
        (np.zeros((21, 2)), np.zeros(21), 0, 480),
        (np.zeros((21, 2)), np.zeros(21), 640, -1),
    ],
)
def test_contract_errors_raise_value_error(image_xy, depth_m, width_px, height_px):
    with pytest.raises(ValueError):
        compute_pinch_geometry(
            image_xy,
            depth_m,
            width_px=width_px,
            height_px=height_px,
        )


def test_nonconvertible_arrays_raise_value_error():
    with pytest.raises(ValueError):
        compute_pinch_geometry(object(), np.zeros(21), width_px=640, height_px=480)


@pytest.mark.parametrize(("width_px", "height_px"), [(True, 480), (640, False)])
def test_boolean_image_dimensions_are_rejected(width_px, height_px):
    with pytest.raises(ValueError):
        compute_pinch_geometry(
            np.zeros((21, 2)),
            np.ones(21),
            width_px=width_px,
            height_px=height_px,
        )
