from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from ir_force.single_finger_curve_runtime import (
    ContinuousFrameArchive,
    build_frame_row,
    roi_centers,
    thermal_features,
)


def test_roi_centers_are_tip_to_dip_one_third_and_pip_mcp_midpoint():
    centers = roi_centers(
        {
            "TIP": (30.0, 30.0),
            "DIP": (36.0, 33.0),
            "PIP": (50.0, 40.0),
            "MCP": (60.0, 44.0),
        }
    )

    assert centers["tip_uv"] == pytest.approx((30.0, 30.0))
    assert centers["distal_uv"] == pytest.approx((32.0, 31.0))
    assert centers["reference_uv"] == pytest.approx((55.0, 42.0))


def test_primary_signal_is_distal_3x3_minus_reference_5x5():
    frame = np.full((120, 160), 1000, dtype=np.uint16)
    frame[30:33, 31:34] = 1100
    frame[40:45, 53:58] = 1020

    result = thermal_features(
        frame,
        {
            "tip_uv": (30.0, 30.0),
            "distal_uv": (32.0, 31.0),
            "reference_uv": (55.0, 42.0),
        },
    )

    assert result["tip_raw_count"] == 1000
    assert result["distal_3x3_mean_count"] == pytest.approx(1100.0)
    assert result["reference_5x5_mean_count"] == pytest.approx(1020.0)
    assert result["primary_signal_count"] == pytest.approx(80.0)


def test_thermal_features_rejects_any_patch_crossing_frame_boundary():
    frame = np.zeros((120, 160), dtype=np.uint16)

    with pytest.raises(ValueError, match="distal_5x5_out_of_bounds"):
        thermal_features(
            frame,
            {
                "tip_uv": (2.0, 2.0),
                "distal_uv": (1.0, 1.0),
                "reference_uv": (20.0, 20.0),
            },
        )


def _make_archive(tmp_path):
    photo = tmp_path / "surface.jpg"
    photo.write_bytes(b"setup-photo")
    session = tmp_path / "session"
    return ContinuousFrameArchive(session, photo), session


def test_archive_writes_every_thermal_frame_and_reuses_d435_paths(tmp_path):
    archive, session = _make_archive(tmp_path)
    thermal_0 = np.full((120, 160), 1010, dtype=np.uint16)
    thermal_1 = np.full((120, 160), 1020, dtype=np.uint16)
    color = np.full((8, 10, 3), 17, dtype=np.uint8)
    depth = np.full((8, 10), 900, dtype=np.uint16)

    first = archive.capture(
        frame_index=0,
        thermal_counts=thermal_0,
        color_rgb=color,
        depth_z16=depth,
        color_frame_number=20,
        depth_frame_number=10,
    )
    second = archive.capture(
        frame_index=1,
        thermal_counts=thermal_1,
        color_rgb=color,
        depth_z16=depth,
        color_frame_number=20,
        depth_frame_number=10,
    )

    assert first["thermal_uint16"] != second["thermal_uint16"]
    assert first["thermal_inferno_auto"] != second["thermal_inferno_auto"]
    assert first["d435_rgb"] == second["d435_rgb"]
    assert first["d435_depth_z16"] == second["d435_depth_z16"]
    assert len(list((session / "raw/thermal_uint16").glob("*.png"))) == 2
    assert len(list((session / "rendered/thermal_inferno_auto").glob("*.png"))) == 2
    assert len(list((session / "raw/d435_rgb").glob("*.png"))) == 1
    assert len(list((session / "raw/d435_depth_z16").glob("*.png"))) == 1
    restored = cv2.imread(
        str(session / second["thermal_uint16"]),
        cv2.IMREAD_UNCHANGED,
    )
    assert restored.dtype == np.uint16
    np.testing.assert_array_equal(restored, thermal_1)
    rendered = cv2.imread(
        str(session / second["thermal_inferno_auto"]),
        cv2.IMREAD_COLOR,
    )
    assert rendered.shape == (120, 160, 3)
    assert rendered.dtype == np.uint8
    assert (session / "setup/surface.jpg").read_bytes() == b"setup-photo"


def test_archive_is_exclusive_and_does_not_claim_duplicate_success(tmp_path):
    archive, session = _make_archive(tmp_path)
    thermal = np.zeros((120, 160), dtype=np.uint16)
    color = np.zeros((8, 10, 3), dtype=np.uint8)
    depth = np.zeros((8, 10), dtype=np.uint16)
    kwargs = {
        "frame_index": 0,
        "thermal_counts": thermal,
        "color_rgb": color,
        "depth_z16": depth,
        "color_frame_number": 20,
        "depth_frame_number": 10,
    }
    archive.capture(**kwargs)

    with pytest.raises(FileExistsError):
        archive.capture(**kwargs)
    with pytest.raises(FileExistsError):
        ContinuousFrameArchive(session, tmp_path / "surface.jpg")


def test_archive_allows_session_without_setup_photo(tmp_path):
    session = tmp_path / "session"

    archive = ContinuousFrameArchive(session, None)

    assert archive.surface_photo is None
    assert (session / "setup").is_dir()
    assert list((session / "setup").iterdir()) == []


def test_archive_validates_all_arrays_before_writing(tmp_path):
    archive, session = _make_archive(tmp_path)

    with pytest.raises(ValueError, match="depth_z16"):
        archive.capture(
            frame_index=0,
            thermal_counts=np.zeros((120, 160), dtype=np.uint16),
            color_rgb=np.zeros((8, 10, 3), dtype=np.uint8),
            depth_z16=np.zeros((8, 10), dtype=np.float32),
            color_frame_number=20,
            depth_frame_number=10,
        )

    assert list((session / "raw/thermal_uint16").iterdir()) == []


def _successful_projection_context(image_xy):
    def split_results(_results):
        return (np.zeros((21, 3)), image_xy, np.eye(3)), None

    def associate_color_to_raw_depth(*, label, normalized_xy, **_kwargs):
        landmark_index = {"TIP": 8, "DIP": 7, "PIP": 6, "MCP": 5}[label]
        return {
            "label": label,
            "status": "ok",
            "depth_pixel": [landmark_index, landmark_index + 1],
            "raw_depth": 1000 + landmark_index,
            "depth_m": 0.4 + landmark_index / 1000.0,
            "normalized_color_xy": list(normalized_xy),
        }

    def project_depth_pixel_to_thermal(*, source_depth_xy, **_kwargs):
        points = {
            (8, 9): (30.0, 30.0),
            (7, 8): (36.0, 33.0),
            (6, 7): (50.0, 40.0),
            (5, 6): (60.0, 44.0),
        }
        return SimpleNamespace(
            status="ok",
            thermal_uv=points[source_depth_xy],
            depth_m=0.5,
        )

    return {
        "split_results": split_results,
        "associate_color_to_raw_depth": associate_color_to_raw_depth,
        "project_depth_pixel_to_thermal": project_depth_pixel_to_thermal,
        "association_kwargs": {},
        "projection_kwargs": {},
    }


def _frame_inputs():
    image_xy = np.zeros((21, 2), dtype=float)
    image_xy[5:9] = (
        (0.50, 0.50),
        (0.51, 0.51),
        (0.52, 0.52),
        (0.53, 0.53),
    )
    raw = SimpleNamespace(
        color_rgb=np.zeros((720, 1280, 3), dtype=np.uint8),
        depth_z16=np.ones((720, 1280), dtype=np.uint16),
        depth_sdk_frame=None,
        observed_at_s=10.0,
        color_frame_number=4,
        depth_frame_number=3,
        color_timestamp_ms=100.0,
        depth_timestamp_ms=101.0,
        color_timestamp_domain="hardware_clock",
        depth_timestamp_domain="hardware_clock",
    )
    telemetry = SimpleNamespace(
        frame_counter=7,
        packet_timestamp_ms=300,
        ffc_desired=False,
        ffc_state="complete",
        ffc_in_progress=False,
        since_last_ffc_s=8.0,
        tlinear_enabled=True,
        tlinear_resolution_k=0.01,
    )
    thermal = SimpleNamespace(
        frame=np.full((120, 160), 1000, dtype=np.uint16),
        t=10.1,
        lepton_telemetry=telemetry,
    )
    hands = SimpleNamespace(process=lambda _frame: object())
    trial = {
        "block_index": 0,
        "condition": "press",
        "phase": "X",
        "phase_elapsed_s": 1.25,
        "global_elapsed_s": 6.25,
        "frame_index": 12,
        "now_s": 10.2,
    }
    return image_xy, raw, thermal, hands, trial


def test_build_frame_row_tracks_all_landmarks_and_logs_primary_signal():
    image_xy, raw, thermal, hands, trial = _frame_inputs()

    row = build_frame_row(
        raw,
        thermal,
        hands,
        _successful_projection_context(image_xy),
        trial,
    )

    assert row["tracking_valid"] is True
    assert row["tracking_reasons"] == []
    assert row["primary_signal_count"] == pytest.approx(0.0)
    assert row["distal_thermal_uv"] == pytest.approx([32.0, 31.0])
    assert row["reference_thermal_uv"] == pytest.approx([55.0, 42.0])
    assert row["index_tip_depth_m"] == pytest.approx(0.408)
    assert row["host_read_completion_skew_s"] == pytest.approx(0.1)
    assert row["ffc_state"] == "complete"
    assert row["tlinear_resolution_k"] == pytest.approx(0.01)


def test_build_frame_row_retains_closed_row_when_right_hand_is_missing():
    _image_xy, raw, thermal, hands, trial = _frame_inputs()
    context = {
        "split_results": lambda _results: (None, None),
        "associate_color_to_raw_depth": pytest.fail,
        "project_depth_pixel_to_thermal": pytest.fail,
        "association_kwargs": {},
        "projection_kwargs": {},
    }

    row = build_frame_row(raw, thermal, hands, context, trial)

    assert row["tracking_valid"] is False
    assert row["tracking_reasons"] == ["physical_right_hand_missing"]
    assert row["row_type"] == "frame"
    assert row["thermal_host_s"] == pytest.approx(10.1)
