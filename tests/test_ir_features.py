import csv

import numpy as np

from ir_force.ir_features import (
    FrameFeatures,
    ThermalROI,
    compute_baseline,
    extract_frame_features,
    overlay_mask,
    palette_index_image,
    write_features_csv,
)


def test_compute_baseline_uses_mean_and_noise_floor():
    frames = [
        np.full((4, 4), 10, dtype=np.uint8),
        np.full((4, 4), 12, dtype=np.uint8),
        np.full((4, 4), 14, dtype=np.uint8),
    ]
    stats = compute_baseline(frames)
    assert np.all(stats.mean == 12.0)
    assert round(stats.noise, 3) == 1.633


def test_extract_frame_features_thresholds_positive_delta():
    baseline = compute_baseline([np.zeros((4, 4), dtype=np.uint8), np.ones((4, 4), dtype=np.uint8)])
    frame = np.zeros((4, 4), dtype=np.uint8)
    frame[1:3, 1:3] = 20
    features, mask = extract_frame_features(frame, baseline, noise_sigma=3.0)
    assert features.area_px == 4
    assert features.max_delta == 19.5
    assert mask.sum() == 4


def test_palette_index_image_maps_colorized_thermal_to_relative_indices():
    palette = np.array(
        [
            [0, 0, 0],
            [10, 0, 0],
            [20, 0, 0],
            [30, 0, 0],
        ],
        dtype=np.uint8,
    )
    frame = np.array([[[0, 0, 0], [21, 0, 0], [30, 1, 0]]], dtype=np.uint8)
    assert palette_index_image(frame, palette).tolist() == [[0.0, 2.0, 3.0]]


def test_palette_index_image_uses_safe_dtype_for_full_range_deltas():
    palette = np.array(
        [
            [255, 255, 255],
            [0, 0, 1],
        ],
        dtype=np.uint8,
    )
    frame = np.array([[[0, 0, 0]]], dtype=np.uint8)

    assert palette_index_image(frame, palette).tolist() == [[1.0]]


def test_extract_frame_features_supports_inverted_palette_indices():
    palette = np.array(
        [
            [30, 0, 0],
            [20, 0, 0],
            [10, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    baseline_frames = [
        np.array([[[0, 0, 0], [0, 0, 0]]], dtype=np.uint8),
        np.array([[[0, 0, 0], [0, 0, 0]]], dtype=np.uint8),
    ]
    frame = np.array([[[30, 0, 0], [0, 0, 0]]], dtype=np.uint8)

    baseline = compute_baseline(baseline_frames, palette=palette, invert_palette=True)
    features, mask = extract_frame_features(frame, baseline, palette=palette, invert_palette=True)

    assert features.area_px == 1
    assert features.mean_delta == 3.0
    assert mask.tolist() == [[1, 0]]


def test_extract_frame_features_uses_grayscale_without_palette():
    baseline = compute_baseline(
        [
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.ones((2, 2, 3), dtype=np.uint8),
        ]
    )
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[0, 0] = [12, 12, 12]

    features, mask = extract_frame_features(frame, baseline, noise_sigma=3.0)

    assert features.area_px == 1
    assert features.max_delta == 11.5
    assert mask.tolist() == [[1, 0], [0, 0]]


def test_compute_baseline_crops_to_contact_roi_before_noise_estimate():
    frames = []
    for outside_value in (0, 40, 80):
        frame = np.full((4, 4), outside_value, dtype=np.uint8)
        frame[1:3, 1:3] = 20
        frames.append(frame)

    stats = compute_baseline(frames, roi=ThermalROI(x=1, y=1, width=2, height=2))

    assert stats.mean.shape == (2, 2)
    assert np.all(stats.mean == 20.0)
    assert stats.noise == 0.0


def test_extract_frame_features_ignores_hot_pixels_outside_contact_roi():
    baseline_frames = [np.full((4, 4), 10, dtype=np.uint8), np.full((4, 4), 10, dtype=np.uint8)]
    baseline = compute_baseline(baseline_frames, roi=ThermalROI(x=1, y=1, width=2, height=2))
    frame = np.full((4, 4), 10, dtype=np.uint8)
    frame[0, 0] = 80
    frame[1, 1] = 30

    features, mask = extract_frame_features(frame, baseline, noise_sigma=3.0)

    assert features.area_px == 1
    assert features.max_delta == 20.0
    assert mask.tolist() == [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]


def test_overlay_mask_does_not_mutate_input_frame():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    mask = np.array([[1, 0], [0, 0]], dtype=np.uint8)

    out = overlay_mask(frame, mask)

    assert frame.tolist() == np.zeros((2, 2, 3), dtype=np.uint8).tolist()
    assert out[0, 0].tolist() == [0, 0, 255]


def test_write_features_csv(tmp_path):
    out = tmp_path / "features.csv"
    write_features_csv([FrameFeatures("frame_000001.png", 4, 2.5, 9.0)], out)
    rows = list(csv.DictReader(out.open()))
    assert rows[0] == {
        "frame": "frame_000001.png",
        "area_px": "4",
        "mean_delta": "2.5",
        "max_delta": "9.0",
    }
