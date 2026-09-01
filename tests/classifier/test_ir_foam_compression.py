from __future__ import annotations

import importlib

import cv2
import numpy as np
import pytest


def test_recording_one_uses_the_prespecified_random_steady_order():
    module = importlib.import_module("ir_force.classifier.ir_foam_compression")

    steps = module.build_recording_plan(1)
    steady_targets = [
        step.state
        for step in steps
        if step.block == "steady_state" and step.phase == "target"
    ]

    assert steady_targets == ["C20", "N", "C0", "C30", "C10", "C0", "C20", "N", "C10", "C30"]
    assert sum(step.state == "C30" and step.block == "release_pulses" for step in steps) == 4
    assert [
        step.state
        for step in steps
        if step.block == "hysteresis"
    ] == ["C0", "C10", "C20", "C30", "C20", "C10", "C0", "R"]


def test_marker_tracker_reports_distance_and_compression_from_black_dots():
    module = importlib.import_module("ir_force.classifier.ir_foam_compression")
    frame = np.full((100, 160, 3), 255, dtype=np.uint8)
    cv2.circle(frame, (40, 30), 5, (0, 0, 0), -1)
    cv2.circle(frame, (100, 30), 5, (0, 0, 0), -1)

    observation = module.detect_marker_pair(
        frame,
        left_roi=module.PixelROI(20, 10, 40, 40),
        right_roi=module.PixelROI(80, 10, 40, 40),
        max_gray=60,
    )

    assert observation is not None
    assert observation.distance_px == pytest.approx(60.0)
    assert module.compression_percent(observation.distance_px, d0_px=75.0) == pytest.approx(20.0)


def test_marker_tracker_prefers_a_black_dot_surrounded_by_white_over_black_foam():
    module = importlib.import_module("ir_force.classifier.ir_foam_compression")
    frame = np.full((100, 180, 3), 255, dtype=np.uint8)
    cv2.circle(frame, (35, 35), 5, (0, 0, 0), -1)
    cv2.rectangle(frame, (50, 15), (75, 60), (0, 0, 0), -1)
    cv2.circle(frame, (130, 35), 5, (0, 0, 0), -1)

    observation = module.detect_marker_pair(
        frame,
        left_roi=module.PixelROI(10, 10, 70, 60),
        right_roi=module.PixelROI(100, 10, 60, 60),
        max_gray=60,
    )

    assert observation is not None
    assert observation.left_xy == pytest.approx((35.0, 35.0), abs=0.5)
    assert observation.right_xy == pytest.approx((130.0, 35.0), abs=0.5)


def test_marker_tracker_accepts_dim_white_tabs_as_seen_by_oak():
    module = importlib.import_module("ir_force.classifier.ir_foam_compression")
    frame = np.full((80, 160, 3), 145, dtype=np.uint8)
    cv2.circle(frame, (35, 35), 5, (75, 75, 75), -1)
    cv2.rectangle(frame, (52, 10), (78, 60), (45, 45, 45), -1)
    cv2.circle(frame, (120, 35), 5, (75, 75, 75), -1)

    observation = module.detect_marker_pair(
        frame,
        left_roi=module.PixelROI(10, 10, 70, 55),
        right_roi=module.PixelROI(95, 10, 50, 55),
        max_gray=120,
    )

    assert observation is not None
    assert observation.left_xy == pytest.approx((35.0, 35.0), abs=0.5)
    assert observation.right_xy == pytest.approx((120.0, 35.0), abs=0.5)


def test_marker_tracker_ignores_a_small_dark_foam_speck():
    module = importlib.import_module("ir_force.classifier.ir_foam_compression")
    frame = np.full((80, 160, 3), 145, dtype=np.uint8)
    cv2.circle(frame, (35, 35), 5, (75, 75, 75), -1)
    cv2.rectangle(frame, (42, 20), (45, 24), (75, 75, 75), -1)
    cv2.circle(frame, (120, 35), 5, (75, 75, 75), -1)

    observation = module.detect_marker_pair(
        frame,
        left_roi=module.PixelROI(10, 10, 50, 55),
        right_roi=module.PixelROI(95, 10, 50, 55),
        max_gray=120,
    )

    assert observation is not None
    assert observation.left_xy == pytest.approx((35.0, 35.0), abs=0.5)


def test_centered_dark_marker_tracker_selects_the_dot_over_an_edge_foam_blob():
    module = importlib.import_module("ir_force.classifier.ir_foam_compression")
    frame = np.full((90, 180, 3), 140, dtype=np.uint8)
    cv2.rectangle(frame, (10, 15), (30, 75), (40, 40, 40), -1)
    cv2.circle(frame, (50, 45), 8, (40, 40, 40), -1)
    cv2.rectangle(frame, (145, 15), (170, 75), (40, 40, 40), -1)
    cv2.circle(frame, (120, 45), 8, (40, 40, 40), -1)

    observation = module.detect_centered_dark_marker_pair(
        frame,
        left_roi=module.PixelROI(10, 15, 55, 60),
        right_roi=module.PixelROI(105, 15, 65, 60),
        max_gray=90,
        min_area_px=20,
        max_area_px=1000,
    )

    assert observation is not None
    assert observation.left_xy == pytest.approx((50.0, 45.0), abs=0.5)
    assert observation.right_xy == pytest.approx((120.0, 45.0), abs=0.5)


def test_stable_gate_only_opens_after_one_second_continuously_in_tolerance():
    module = importlib.import_module("ir_force.classifier.ir_foam_compression")
    gate = module.StableCompressionGate(target_pct=20.0, tolerance_pct=2.0, required_s=1.0)

    assert gate.update(timestamp=0.0, compression_pct=20.5) is False
    assert gate.update(timestamp=0.6, compression_pct=18.1) is False
    assert gate.update(timestamp=0.8, compression_pct=24.0) is False
    assert gate.update(timestamp=1.0, compression_pct=20.0) is False
    assert gate.update(timestamp=2.0, compression_pct=21.5) is True
    assert gate.stable_seconds == pytest.approx(1.0)


def test_hold_gate_allows_a_short_marker_outlier_but_rejects_a_persistent_deviation():
    module = importlib.import_module("ir_force.classifier.ir_foam_compression")
    gate = module.HoldToleranceGate(target_pct=0.0, tolerance_pct=5.0, max_gap_s=0.5)

    assert gate.update(timestamp=0.0, compression_pct=0.1)
    assert gate.update(timestamp=0.1, compression_pct=-34.0)
    assert gate.update(timestamp=0.2, compression_pct=0.0)
    assert gate.update(timestamp=0.3, compression_pct=6.0)
    assert not gate.update(timestamp=0.9, compression_pct=6.0)


def test_frozen_regions_reject_overlay_band_and_normalize_foam_against_references():
    module = importlib.import_module("ir_force.classifier.ir_foam_compression")
    regions = module.FrozenThermalRegions(
        foam_bbox=module.PixelROI(68, 40, 28, 30),
        foam_center=module.PixelROI(75, 48, 14, 18),
        left_contact=module.PixelROI(68, 48, 6, 18),
        right_contact=module.PixelROI(90, 48, 6, 18),
        background=module.PixelROI(5, 5, 15, 15),
        room_reference=module.PixelROI(15, 15, 12, 12),
        warm_reference=module.PixelROI(130, 15, 12, 12),
    )
    assert regions.preflight_issues((128, 160)) == []

    scalar = np.zeros((128, 160), dtype=np.float32)
    scalar[regions.room_reference.slices()] = 10.0
    scalar[regions.warm_reference.slices()] = 30.0
    scalar[regions.foam_center.slices()] = 15.0
    features = module.reference_normalized_features(scalar, regions)
    assert features["foam_center_norm"] == pytest.approx(0.25)

    bad = module.FrozenThermalRegions(
        foam_bbox=regions.foam_bbox,
        foam_center=module.PixelROI(75, 100, 14, 10),
        left_contact=regions.left_contact,
        right_contact=regions.right_contact,
        background=regions.background,
        room_reference=regions.room_reference,
        warm_reference=regions.warm_reference,
    )
    assert any("overlay" in issue for issue in bad.preflight_issues((128, 160)))


def test_reference_features_accept_a_registered_foam_roi_away_from_the_setup_position():
    module = importlib.import_module("ir_force.classifier.ir_foam_compression")
    regions = module.FrozenThermalRegions(
        foam_bbox=module.PixelROI(73, 40, 28, 30),
        foam_center=module.PixelROI(80, 48, 14, 18),
        left_contact=module.PixelROI(73, 48, 6, 18),
        right_contact=module.PixelROI(95, 48, 6, 18),
        background=module.PixelROI(5, 5, 15, 15),
        room_reference=module.PixelROI(15, 15, 12, 12),
        warm_reference=module.PixelROI(130, 15, 12, 12),
    )
    scalar = np.zeros((128, 160), dtype=np.float32)
    scalar[regions.room_reference.slices()] = 10.0
    scalar[regions.warm_reference.slices()] = 30.0
    scalar[regions.foam_center.slices()] = 22.0

    features = module.reference_normalized_features(scalar, regions, strict_preflight=False)

    assert features["foam_center_norm"] == pytest.approx(0.6)
