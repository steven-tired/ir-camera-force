import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ir_force.pinch_visualization import (
    FrameArchive,
    render_session,
)


def _frames():
    thermal = np.arange(120 * 160, dtype=np.uint16).reshape(120, 160)
    color_rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    color_rgb[1, 2] = (10, 20, 30)
    depth_z16 = np.arange(4 * 5, dtype=np.uint16).reshape(4, 5)
    return thermal, color_rgb, depth_z16


def test_frame_archive_preserves_lossless_frames_and_returns_relative_paths(
    tmp_path,
):
    session_dir = tmp_path / "stage1e_tip_pinch_visualization_01"
    session_dir.mkdir()
    archive = FrameArchive(session_dir)
    thermal, color_rgb, depth_z16 = _frames()

    paths = archive.capture(
        attempt_index=7,
        thermal_counts=thermal,
        color_rgb=color_rgb,
        depth_z16=depth_z16,
    )

    assert paths == {
        "thermal_uint16": "raw/thermal_uint16/attempt_000007.png",
        "d435_rgb": "raw/d435_rgb/attempt_000007.png",
        "d435_depth_z16": "raw/d435_depth_z16/attempt_000007.png",
    }
    assert sorted(
        path.relative_to(session_dir).as_posix()
        for path in session_dir.glob("raw/*")
    ) == [
        "raw/d435_depth_z16",
        "raw/d435_rgb",
        "raw/thermal_uint16",
    ]
    saved_thermal = cv2.imread(
        str(session_dir / paths["thermal_uint16"]),
        cv2.IMREAD_UNCHANGED,
    )
    saved_color = cv2.imread(
        str(session_dir / paths["d435_rgb"]),
        cv2.IMREAD_COLOR,
    )
    saved_depth = cv2.imread(
        str(session_dir / paths["d435_depth_z16"]),
        cv2.IMREAD_UNCHANGED,
    )
    assert saved_thermal.dtype == np.uint16
    assert np.array_equal(saved_thermal, thermal)
    assert np.array_equal(
        saved_color,
        cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR),
    )
    assert saved_depth.dtype == np.uint16
    assert np.array_equal(saved_depth, depth_z16)


def test_frame_archive_metadata_freezes_descriptive_identity(tmp_path):
    session_dir = tmp_path / "stage1e_tip_pinch_visualization_01"
    session_dir.mkdir()

    metadata = FrameArchive(session_dir).metadata()

    assert metadata == {
        "experiment_id": "stage1e_tip_pinch_visualization_01",
        "experiment_role": "communication_only_descriptive_replication",
        "decision_authority": "none",
        "authoritative_reference": "stage1e_tip_pinch_signal_05",
        "can_update_thresholds": False,
        "can_authorize_stage1f": False,
        "artifact_paths": "relative_to_session_dir",
    }


def test_frame_archive_marks_contact_only_physical_protocol(tmp_path):
    session_dir = tmp_path / "stage1e_tip_pinch_contact_null_01"
    session_dir.mkdir()

    metadata = FrameArchive(
        session_dir,
        physical_protocol="contact_only",
    ).metadata()

    assert metadata["experiment_id"] == session_dir.name
    assert metadata["physical_protocol"] == "contact_only"
    assert metadata["phase_semantics"] == {
        "record_just_touch": "light_contact",
        "record_press_hard": "light_contact_legacy_analysis_slot",
        "record_return_touch": "light_contact",
    }
    assert metadata["decision_authority"] == "none"


def test_frame_archive_rejects_invalid_arrays_and_existing_target(tmp_path):
    session_dir = tmp_path / "stage1e_tip_pinch_visualization_01"
    session_dir.mkdir()
    archive = FrameArchive(session_dir)
    thermal, color_rgb, depth_z16 = _frames()

    with pytest.raises(ValueError, match="thermal"):
        archive.capture(
            attempt_index=0,
            thermal_counts=thermal.astype(np.float32),
            color_rgb=color_rgb,
            depth_z16=depth_z16,
        )
    with pytest.raises(ValueError, match="color"):
        archive.capture(
            attempt_index=0,
            thermal_counts=thermal,
            color_rgb=color_rgb.astype(np.uint16),
            depth_z16=depth_z16,
        )
    with pytest.raises(ValueError, match="depth"):
        archive.capture(
            attempt_index=0,
            thermal_counts=thermal,
            color_rgb=color_rgb,
            depth_z16=np.zeros((3, 5), dtype=np.uint16),
        )

    archive.capture(
        attempt_index=0,
        thermal_counts=thermal,
        color_rgb=color_rgb,
        depth_z16=depth_z16,
    )
    thermal_path = (
        session_dir / "raw/thermal_uint16/attempt_000000.png"
    )
    before = thermal_path.read_bytes()
    with pytest.raises(FileExistsError):
        archive.capture(
            attempt_index=0,
            thermal_counts=thermal + 1,
            color_rgb=color_rgb,
            depth_z16=depth_z16,
        )
    assert thermal_path.read_bytes() == before


def test_frame_archive_requires_existing_session_directory(tmp_path):
    missing = tmp_path / "stage1e_tip_pinch_visualization_01"

    with pytest.raises(FileNotFoundError):
        FrameArchive(missing)


def _artifact_row(attempt_index, artifacts, phase, *, accepted):
    row = {
        "row_type": "attempt",
        "attempt_index": attempt_index,
        "status": (
            "software_gate_accepted" if accepted else "blocked"
        ),
        "frame_artifacts": artifacts,
        "pinch_signal": {
            "phase": phase,
            "recording": phase.startswith("record_"),
            "quota_accepted": accepted and phase.startswith("record_"),
            "group_index": 0,
        },
        "fingertips": [],
    }
    if accepted:
        row["fingertips"] = [
            {
                "label": "thumb_tip",
                "thermal_pixel": [10, 10],
            },
            {
                "label": "index_tip",
                "thermal_pixel": [145, 105],
            },
        ]
    return row


def _write_render_fixture(session_dir):
    archive = FrameArchive(session_dir)
    color_rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    depth_z16 = np.zeros((4, 5), dtype=np.uint16)
    base = np.arange(120 * 160, dtype=np.uint16).reshape(120, 160)
    frames = [
        (base % 3000) + 500,
        (base % 1001) + 1000,
        (base % 1001) + 1100,
    ]
    for frame in frames:
        frame[0, 0] = 1500
    artifacts = [
        archive.capture(
            attempt_index=index,
            thermal_counts=frame,
            color_rgb=color_rgb,
            depth_z16=depth_z16,
        )
        for index, frame in enumerate(frames)
    ]
    rows = [
        {
            "row_type": "metadata",
            "visualization_capture": archive.metadata(),
        },
        _artifact_row(
            0,
            artifacts[0],
            "prepare_just_touch",
            accepted=False,
        ),
        _artifact_row(
            1,
            artifacts[1],
            "record_just_touch",
            accepted=True,
        ),
        _artifact_row(
            2,
            artifacts[2],
            "record_press_hard",
            accepted=True,
        ),
        {"row_type": "summary", "status": "ok"},
    ]
    capture_jsonl = session_dir / "capture.jsonl"
    capture_jsonl.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return capture_jsonl, frames


def test_render_session_uses_one_baseline_scale_and_marks_autoscale_display_only(
    tmp_path,
):
    session_dir = tmp_path / "stage1e_tip_pinch_visualization_01"
    session_dir.mkdir()
    capture_jsonl, frames = _write_render_fixture(session_dir)
    raw_before = [
        frame.copy()
        for frame in frames
    ]

    manifest = render_session(capture_jsonl, session_dir)

    baseline_lo, baseline_hi = np.percentile(frames[1], (1.0, 99.0))
    assert manifest["experiment_id"] == session_dir.name
    assert manifest["experiment_role"] == (
        "communication_only_descriptive_replication"
    )
    assert manifest["decision_authority"] == "none"
    assert manifest["rendering"]["fixed_scale"] == {
        "comparison_allowed": True,
        "source": "first_quota_accepted_record_just_touch",
        "source_attempt_index": 1,
        "lower_count": pytest.approx(float(baseline_lo)),
        "upper_count": pytest.approx(float(baseline_hi)),
        "percentiles": [1.0, 99.0],
    }
    assert manifest["rendering"]["autoscale_display_only"] == {
        "comparison_allowed": False,
        "scale": "per_frame_1st_99th_percentile",
    }
    assert manifest["rendering"]["colormap"] == "inferno"
    assert manifest["rendering"]["native_thermal_shape"] == [120, 160]
    assert manifest["rendering"]["full_frame_scale"] == 4
    assert manifest["artifact_counts"] == {
        "archived_attempts": 3,
        "fixed_scale_heatmaps": 3,
        "autoscale_display_only_heatmaps": 3,
        "fingertip_overlays": 2,
        "fingertip_crops": {
            "thumb_tip": 2,
            "index_tip": 2,
        },
    }
    fixed = [
        cv2.imread(
            str(
                session_dir
                / f"rendered/fixed_scale/attempt_{index:06d}.png"
            )
        )
        for index in range(3)
    ]
    assert all(image.shape == (480, 640, 3) for image in fixed)
    assert np.array_equal(fixed[0][0, 0], fixed[2][0, 0])
    assert (
        session_dir
        / "rendered/fingertip_overlays/attempt_000001.png"
    ).is_file()
    for tip in ("thumb_tip", "index_tip"):
        crop = cv2.imread(
            str(
                session_dir
                / f"rendered/fingertip_crops/{tip}/attempt_000001.png"
            )
        )
        assert crop.shape == (180, 180, 3)
    assert all(
        np.array_equal(frame, before)
        for frame, before in zip(frames, raw_before, strict=True)
    )
    assert json.loads(
        (session_dir / "manifest.json").read_text(encoding="utf-8")
    ) == manifest


def test_render_session_manifest_and_render_outputs_are_exclusive(tmp_path):
    session_dir = tmp_path / "stage1e_tip_pinch_visualization_01"
    session_dir.mkdir()
    capture_jsonl, _frames = _write_render_fixture(session_dir)

    render_session(capture_jsonl, session_dir)
    manifest_path = session_dir / "manifest.json"
    manifest_before = manifest_path.read_bytes()
    fixed_before = sorted(
        path.read_bytes()
        for path in (session_dir / "rendered/fixed_scale").glob("*.png")
    )

    with pytest.raises(FileExistsError):
        render_session(capture_jsonl, session_dir)

    assert manifest_path.read_bytes() == manifest_before
    assert sorted(
        path.read_bytes()
        for path in (session_dir / "rendered/fixed_scale").glob("*.png")
    ) == fixed_before
