from __future__ import annotations

import argparse
import csv
from pathlib import Path

import extract_ir_grip_features
import numpy as np


def _touch_frames(directory: Path, count: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (directory / f"frame_{index:06d}.png").write_bytes(b"png")


def _write_telemetry(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame"])
        writer.writeheader()
        for index in range(count):
            writer.writerow({"frame": index})


def test_skip_reason_rejects_too_few_thermal_frames(tmp_path):
    trial = tmp_path / "trial"
    _touch_frames(trial / "thermal", 5)
    _write_telemetry(trial / "telemetry.csv", 5)

    assert extract_ir_grip_features.skip_reason(trial, baseline_frames=20) == "too few thermal frames: 5 <= 20"


def test_skip_reason_rejects_frame_telemetry_mismatch(tmp_path):
    trial = tmp_path / "trial"
    _touch_frames(trial / "thermal", 30)
    _write_telemetry(trial / "telemetry.csv", 29)

    assert extract_ir_grip_features.skip_reason(trial, baseline_frames=20) == (
        "thermal frame count 30 does not match telemetry rows 29"
    )


def test_extract_trials_skips_incomplete_trials_and_continues(tmp_path, monkeypatch, capsys):
    root = tmp_path
    good = root / "trials" / "good"
    bad = root / "trials" / "bad"
    _touch_frames(good / "thermal", 30)
    _write_telemetry(good / "telemetry.csv", 30)
    _touch_frames(bad / "thermal", 0)
    _write_telemetry(bad / "telemetry.csv", 0)

    def fake_extract_trial(
        trial_dir,
        baseline_frames,
        noise_sigma,
        palette_path,
        invert_palette=False,
        roi=None,
        feature_name="ir_features.csv",
        overlay_dir_name="overlays",
    ):
        assert roi is None
        assert feature_name == "ir_features.csv"
        assert overlay_dir_name == "overlays"
        return trial_dir / "ir_features.csv"

    monkeypatch.setattr(extract_ir_grip_features, "extract_trial", fake_extract_trial)

    written = extract_ir_grip_features.extract_trials(
        root,
        baseline_frames=20,
        noise_sigma=3.0,
        palette_path=None,
        invert_palette=False,
        strict=False,
    )

    assert written == [good / "ir_features.csv"]
    assert "skipping" in capsys.readouterr().out


def test_extract_trials_can_target_matching_trial_folders(tmp_path, monkeypatch):
    root = tmp_path
    wanted = root / "trials" / "foam-block_soft_low_rep02"
    other = root / "trials" / "foam-block_soft_low_rep03"
    for trial in (wanted, other):
        _touch_frames(trial / "thermal", 30)
        _write_telemetry(trial / "telemetry.csv", 30)

    def fake_extract_trial(
        trial_dir,
        baseline_frames,
        noise_sigma,
        palette_path,
        invert_palette=False,
        roi=None,
        feature_name="ir_features.csv",
        overlay_dir_name="overlays",
    ):
        assert roi is None
        assert feature_name == "ir_features.csv"
        assert overlay_dir_name == "overlays"
        return trial_dir / "ir_features.csv"

    monkeypatch.setattr(extract_ir_grip_features, "extract_trial", fake_extract_trial)

    written = extract_ir_grip_features.extract_trials(
        root,
        baseline_frames=20,
        noise_sigma=3.0,
        palette_path=None,
        invert_palette=False,
        strict=False,
        trial_glob="*_rep02",
    )

    assert written == [wanted / "ir_features.csv"]


def test_parse_roi_accepts_xywh_contact_rectangle():
    roi = extract_ir_grip_features.parse_roi("10,20,30,40")

    assert (roi.x, roi.y, roi.width, roi.height) == (10, 20, 30, 40)


def test_parse_roi_rejects_empty_or_invalid_rectangles():
    assert extract_ir_grip_features.parse_roi("") is None

    for value in ("1,2,3", "1,2,0,4", "-1,2,3,4", "x,2,3,4"):
        try:
            extract_ir_grip_features.parse_roi(value)
        except argparse.ArgumentTypeError:
            continue
        raise AssertionError(f"expected invalid ROI to fail: {value}")


def test_extract_trials_passes_contact_roi_to_trial_extractor(tmp_path, monkeypatch):
    root = tmp_path
    trial = root / "trials" / "foam-block_soft_low_rep02"
    _touch_frames(trial / "thermal", 30)
    _write_telemetry(trial / "telemetry.csv", 30)
    roi = extract_ir_grip_features.parse_roi("2,3,4,5")

    def fake_extract_trial(
        trial_dir,
        baseline_frames,
        noise_sigma,
        palette_path,
        invert_palette=False,
        roi=None,
        feature_name="ir_features.csv",
        overlay_dir_name="overlays",
    ):
        assert roi == extract_ir_grip_features.parse_roi("2,3,4,5")
        assert feature_name == "ir_features_roi.csv"
        assert overlay_dir_name == "overlays_roi"
        return trial_dir / "ir_features.csv"

    monkeypatch.setattr(extract_ir_grip_features, "extract_trial", fake_extract_trial)

    written = extract_ir_grip_features.extract_trials(
        root,
        baseline_frames=20,
        noise_sigma=3.0,
        palette_path=None,
        invert_palette=False,
        strict=False,
        trial_glob="*_rep02",
        roi=roi,
        feature_name="ir_features_roi.csv",
        overlay_dir_name="overlays_roi",
    )

    assert written == [trial / "ir_features.csv"]


def test_extract_trials_passes_custom_feature_and_overlay_names(tmp_path, monkeypatch):
    root = tmp_path
    trial = root / "trials" / "foam-block_soft_low_rep02"
    _touch_frames(trial / "thermal", 30)
    _write_telemetry(trial / "telemetry.csv", 30)

    def fake_extract_trial(
        trial_dir,
        baseline_frames,
        noise_sigma,
        palette_path,
        invert_palette=False,
        roi=None,
        feature_name="ir_features.csv",
        overlay_dir_name="overlays",
    ):
        assert feature_name == "ir_features_roi.csv"
        assert overlay_dir_name == "overlays_roi"
        return trial_dir / feature_name

    monkeypatch.setattr(extract_ir_grip_features, "extract_trial", fake_extract_trial)

    written = extract_ir_grip_features.extract_trials(
        root,
        baseline_frames=20,
        noise_sigma=3.0,
        palette_path=None,
        invert_palette=False,
        strict=False,
        trial_glob="*_rep02",
        feature_name="ir_features_roi.csv",
        overlay_dir_name="overlays_roi",
    )

    assert written == [trial / "ir_features_roi.csv"]


def test_load_palette_for_opencv_converts_raw_rgb_palette_to_bgr(tmp_path):
    palette_path = tmp_path / "palette.raw"
    np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8).tofile(palette_path)

    palette = extract_ir_grip_features.load_palette_for_opencv(palette_path)

    assert palette.tolist() == [[3, 2, 1], [6, 5, 4]]
