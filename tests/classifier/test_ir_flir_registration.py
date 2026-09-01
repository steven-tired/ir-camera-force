from __future__ import annotations

import importlib

import pytest


def test_projects_flir_visible_marker_coordinates_into_thermal_frame():
    module = importlib.import_module("ir_force.classifier.ir_flir_registration")
    foam = importlib.import_module("ir_force.classifier.ir_foam_compression")
    visible_marker = foam.MarkerObservation(
        left_xy=(360.0, 270.0),
        right_xy=(1080.0, 270.0),
        left_area_px=100.0,
        right_area_px=100.0,
    )

    thermal_marker = module.project_marker_observation(
        visible_marker,
        source_shape=(1080, 1440),
        destination_shape=(128, 160),
    )

    assert thermal_marker.left_xy == pytest.approx((40.0, 32.0))
    assert thermal_marker.right_xy == pytest.approx((120.0, 32.0))
    assert thermal_marker.distance_px == pytest.approx(80.0)


def test_tracks_thermal_roi_with_two_visible_markers():
    module = importlib.import_module("ir_force.classifier.ir_flir_registration")
    foam = importlib.import_module("ir_force.classifier.ir_foam_compression")
    baseline = foam.MarkerObservation(
        left_xy=(40.0, 32.0),
        right_xy=(120.0, 32.0),
        left_area_px=1.0,
        right_area_px=1.0,
    )
    current = foam.MarkerObservation(
        left_xy=(48.0, 40.0),
        right_xy=(112.0, 40.0),
        left_area_px=1.0,
        right_area_px=1.0,
    )
    transform = module.similarity_transform_from_markers(baseline, current)

    roi = transform.transform_roi(foam.PixelROI(60, 44, 16, 12), frame_shape=(128, 160))

    assert transform.scale == pytest.approx(0.8)
    assert transform.translation_xy == pytest.approx((16.0, 14.4))
    assert roi.as_list() == [64, 50, 13, 10]


def test_tracks_only_foam_regions_and_keeps_references_fixed():
    module = importlib.import_module("ir_force.classifier.ir_flir_registration")
    foam = importlib.import_module("ir_force.classifier.ir_foam_compression")
    regions = foam.FrozenThermalRegions(
        foam_bbox=foam.PixelROI(60, 40, 24, 24),
        foam_center=foam.PixelROI(66, 46, 12, 12),
        left_contact=foam.PixelROI(60, 46, 5, 12),
        right_contact=foam.PixelROI(79, 46, 5, 12),
        background=foam.PixelROI(5, 5, 15, 15),
        room_reference=foam.PixelROI(15, 15, 12, 12),
        warm_reference=foam.PixelROI(130, 15, 12, 12),
    )
    transform = module.SimilarityTransform(scale=1.0, angle_deg=0.0, translation_xy=(3.0, 4.0))

    tracked = module.track_foam_regions(regions, transform, frame_shape=(128, 160))

    assert tracked.foam_bbox.as_list() == [63, 44, 24, 24]
    assert tracked.foam_center.as_list() == [69, 50, 12, 12]
    assert tracked.left_contact.as_list() == [63, 50, 5, 12]
    assert tracked.right_contact.as_list() == [82, 50, 5, 12]
    assert tracked.background == regions.background
    assert tracked.room_reference == regions.room_reference
    assert tracked.warm_reference == regions.warm_reference
