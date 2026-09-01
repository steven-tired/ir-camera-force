from __future__ import annotations

import csv
import importlib
import json

import cv2
import numpy as np


def _write_png(path, values):
    array = np.asarray(values, dtype=np.uint8)
    assert cv2.imwrite(str(path), array)


def _write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_trial(tmp_path):
    trial = tmp_path / "trials" / "hand-pressure_test_fingertip_sweep_rep02"
    (trial / "thermal").mkdir(parents=True)
    (trial / "bird").mkdir()
    metadata = {
        "trial_id": trial.name,
        "rep": 2,
        "thermal_roi": "0,0,3,2",
    }
    (trial / "metadata.json").write_text(json.dumps(metadata))

    phases = ["baseline", "baseline", "pressure_sweep", "pressure_sweep", "pressure_sweep"]
    progress = ["", "", "0.0", "0.5", "1.0"]
    rows = []
    thermal_frames = [
        [[10, 10, 10], [10, 10, 10]],
        [[10, 10, 10], [10, 10, 10]],
        [[10, 11, 12], [13, 14, 15]],
        [[10, 12, 14], [16, 18, 20]],
        [[10, 13, 16], [19, 22, 25]],
    ]
    for frame, (phase, sweep_progress, thermal) in enumerate(zip(phases, progress, thermal_frames, strict=True)):
        _write_png(trial / "thermal" / f"frame_{frame:06d}.png", thermal)
        _write_png(trial / "bird" / f"frame_{frame:06d}.png", np.full((4, 4), frame * 10, dtype=np.uint8))
        rows.append(
            {
                "frame": frame,
                "t_capture": frame / 10,
                "t_thermal": frame / 10,
                "t_bird": frame / 10,
                "t_flir_visible": "",
                "surface": "test",
                "contact": "fingertip",
                "phase": phase,
                "sweep_progress": sweep_progress,
            }
        )
    _write_csv(trial / "telemetry.csv", rows)
    return trial


def test_analyze_trial_saves_delta_field_and_rich_frame_features(tmp_path):
    module = importlib.import_module("analyze_ir_hand_pressure")
    trial = _make_trial(tmp_path)

    _summary, records = module.analyze_trial(
        trial,
        main_reps=(2,),
        touch_crop=(0, 0, 4, 4),
        shape_crop=(0, 0, 4, 4),
    )

    field_path = trial / "ir_delta_roi_fields.npz"
    assert field_path.exists()
    fields = np.load(field_path)
    assert fields["delta_roi"].shape == (5, 2, 3)
    assert fields["frames"].tolist() == [0, 1, 2, 3, 4]
    assert fields["roi_xywh"].tolist() == [0, 0, 3, 2]
    assert fields["phase"].astype(str).tolist() == [
        "baseline",
        "baseline",
        "pressure_sweep",
        "pressure_sweep",
        "pressure_sweep",
    ]

    first = records[0]
    assert first["frame"] == 2
    assert first["ir_roi_mean_delta"] == 2.5
    assert first["ir_roi_median_delta"] == 2.5
    assert round(first["ir_roi_std_delta"], 3) == 1.708
    assert first["ir_sum_positive_delta"] == 15.0
    assert first["ir_sum_negative_delta"] == 0.0
    assert round(first["ir_l2_delta"], 3) == 7.416
    assert first["ir_positive_area_1sigma_px"] == 5
    assert first["ir_positive_area_2sigma_px"] == 5
    assert first["ir_positive_area_3sigma_px"] == 5
    assert "ir_pca1_score" in first
    assert "ir_pca2_score" in first


def test_hand_foam_metrics_measure_distance_and_overlap():
    module = importlib.import_module("analyze_ir_hand_pressure")
    baseline = np.full((10, 12), 200, dtype=np.float32)
    baseline[4:7, 5:8] = 0
    foam_bbox = (5, 4, 3, 3)

    near = baseline.copy()
    near[4:7, 1:4] = 120
    near_metrics = module._hand_foam_metrics(
        near,
        baseline,
        foam_bbox,
        pixel_threshold=20,
    )

    assert near_metrics["visible_hand_foam_distance_px"] == 1.0
    assert near_metrics["visible_hand_foam_overlap_px"] == 0.0
    assert near_metrics["visible_hand_foam_pressure_proxy"] == -1.0

    overlapping = baseline.copy()
    overlapping[4:7, 4:7] = 120
    overlap_metrics = module._hand_foam_metrics(
        overlapping,
        baseline,
        foam_bbox,
        pixel_threshold=20,
    )

    assert overlap_metrics["visible_hand_foam_distance_px"] == 0.0
    assert overlap_metrics["visible_hand_foam_overlap_px"] == 6.0
    assert overlap_metrics["visible_hand_foam_pressure_proxy"] == 6.0


def test_proxy_comparison_includes_sum_and_pca_features():
    module = importlib.import_module("analyze_ir_hand_pressure")
    records = []
    for rep in (2, 3):
        for index in range(4):
            records.append(
                {
                    "trial_id": f"rep{rep}",
                    "rep": rep,
                    "used_in_main": True,
                    "post_touch_time_progress": index / 3,
                    "visible_motion_delta": float(index),
                    "visible_shape_height_delta_px": float(index),
                    "visible_hand_foam_distance_px": float(3 - index),
                    "visible_hand_foam_overlap_px": float(index),
                    "visible_hand_foam_pressure_proxy": float(index),
                    "ir_roi_mean_delta": float(index),
                    "ir_positive_area_px": float(index),
                    "ir_negative_area_px": float(3 - index),
                    "ir_sum_positive_delta": float(index * 2),
                    "ir_l1_delta": float(index * 3),
                    "ir_pca1_score": float(index),
                }
            )

    rows = module._correlation_rows(records)

    compared = {(row["proxy"], row["ir_signal"]) for row in rows}
    assert ("visible_motion_delta", "ir_sum_positive_delta") in compared
    assert ("visible_motion_delta", "ir_l1_delta") in compared
    assert ("visible_motion_delta", "ir_pca1_score") in compared
    assert ("visible_hand_foam_pressure_proxy", "ir_roi_mean_delta") in compared


def test_time_control_rows_report_incremental_ir_signal_after_time():
    module = importlib.import_module("analyze_ir_hand_pressure")
    records = []
    for rep in (2, 3):
        for index in range(8):
            time = index / 7
            ir = float(index % 2)
            records.append(
                {
                    "trial_id": f"rep{rep}",
                    "rep": rep,
                    "used_in_main": True,
                    "post_touch_time_progress": time,
                    "visible_hand_foam_pressure_proxy": 10 * time + 5 * ir,
                    "ir_negative_area_3sigma_px": ir,
                }
            )

    rows = module._time_control_rows(
        records,
        proxy="visible_hand_foam_pressure_proxy",
        ir_fields=("ir_negative_area_3sigma_px",),
    )

    row = rows[0]
    assert row["ir_signal"] == "ir_negative_area_3sigma_px"
    assert row["time_plus_ir_r2"] > row["time_only_r2"]
    assert row["delta_r2"] > 0.1
    assert row["partial_pearson_after_time"] > 0.9


def test_lagged_correlation_rows_find_positive_ir_lag():
    module = importlib.import_module("analyze_ir_hand_pressure")
    proxy_values = [0, 1, 0, 1, 0, 1, 0, 1]
    ir_values = [0, 0, 1, 0, 1, 0, 1, 0]
    records = [
        {
            "trial_id": "rep2",
            "rep": 2,
            "used_in_main": True,
            "frame": index,
            "visible_hand_foam_pressure_proxy": proxy,
            "ir_negative_area_3sigma_px": ir,
        }
        for index, (proxy, ir) in enumerate(zip(proxy_values, ir_values, strict=True))
    ]

    rows = module._lagged_correlation_rows(
        records,
        proxy="visible_hand_foam_pressure_proxy",
        ir_field="ir_negative_area_3sigma_px",
        max_lag_frames=2,
    )

    best = max(rows, key=lambda row: row["pooled_pearson"])
    assert best["lag_frames"] == 1
    assert best["pooled_pearson"] == 1.0
