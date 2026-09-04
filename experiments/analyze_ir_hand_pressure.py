from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import sys

_CHECKOUT_ROOT = Path(__file__).resolve().parents[1]
if str(_CHECKOUT_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHECKOUT_ROOT))

from ir_force.data_paths import dataset_root  # noqa: E402

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ir-hand-pressure")

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_ROOT = dataset_root("ir_hand_pressure_viability")
DEFAULT_MAIN_REPS = (2, 3, 4, 5)
DEFAULT_TOUCH_CROP = (330, 85, 230, 315)
DEFAULT_SHAPE_CROP = (355, 120, 165, 245)
SELECTED_PROXY = "visible_hand_foam_pressure_proxy"
SELECTED_PROXY_SLUG = "visible_hand_foam_pressure_proxy"
SELECTED_IR = "ir_negative_area_3sigma_px"
SELECTED_IR_SLUG = "ir_negative_area_3sigma_px"

PROXY_LABELS = {
    "post_touch_time_progress": "post-touch time progress",
    "visible_motion_delta": "visible motion / shape proxy",
    "visible_hand_foam_pressure_proxy": "hand-to-foam pressure proxy",
    "visible_hand_foam_distance_px": "hand-to-foam distance px",
    "visible_hand_foam_overlap_px": "hand-over-foam overlap px",
    "visible_shape_height_delta_px": "dark-shape height change",
}

IR_LABELS = {
    "ir_roi_mean_delta": "IR ROI mean delta",
    "ir_roi_median_delta": "IR ROI median delta",
    "ir_roi_std_delta": "IR ROI std delta",
    "ir_sum_positive_delta": "positive IR delta sum",
    "ir_sum_negative_delta": "negative IR delta sum",
    "ir_l1_delta": "IR L1 delta",
    "ir_l2_delta": "IR L2 delta",
    "ir_p90_delta": "IR p90 delta",
    "ir_p95_delta": "IR p95 delta",
    "ir_p99_delta": "IR p99 delta",
    "ir_top_1pct_mean_delta": "IR top 1% mean delta",
    "ir_robust_max_delta": "IR robust max delta",
    "ir_positive_area_px": "positive IR area px",
    "ir_negative_area_px": "negative IR area px",
    "ir_positive_area_1sigma_px": "positive IR area, 1 sigma",
    "ir_positive_area_2sigma_px": "positive IR area, 2 sigma",
    "ir_positive_area_3sigma_px": "positive IR area, 3 sigma",
    "ir_negative_area_1sigma_px": "negative IR area, 1 sigma",
    "ir_negative_area_2sigma_px": "negative IR area, 2 sigma",
    "ir_negative_area_3sigma_px": "negative IR area, 3 sigma",
    "ir_pca1_score": "IR PCA1 score",
    "ir_pca2_score": "IR PCA2 score",
    "ir_pca3_score": "IR PCA3 score",
}


@dataclass(frozen=True)
class TrialSummary:
    trial_id: str
    rep: int
    used_in_main: bool
    reason: str
    rows: int
    baseline_frames: int
    pressure_frames: int
    post_touch_frames: int
    touch_onset_frame: int | None
    touch_onset_progress: float | None
    touch_threshold: float
    baseline_motion_max: float
    ir_threshold: float


def _parse_crop(value: str) -> tuple[int, int, int, int]:
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("crop must be x,y,width,height")
    try:
        x, y, width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("crop x/y must be non-negative and width/height positive")
    return x, y, width, height


def _parse_main_reps(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    try:
        return tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"could not read {path}")
    return image.astype(np.float32)


def _crop(image: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = crop
    return image[y : y + height, x : x + width]


def _thermal_roi(trial_dir: Path, frame: int, roi: tuple[int, int, int, int]) -> np.ndarray:
    return _crop(_read_gray(trial_dir / "thermal" / f"frame_{frame:06d}.png"), roi)


def _bird_crop(trial_dir: Path, frame: int, crop: tuple[int, int, int, int]) -> np.ndarray:
    return _crop(_read_gray(trial_dir / "bird" / f"frame_{frame:06d}.png"), crop)


def _dark_shape_metrics(trial_dir: Path, frame: int, crop: tuple[int, int, int, int]) -> dict[str, float]:
    gray = _bird_crop(trial_dir, frame, crop)
    dark_mask = (gray < 55).astype(np.uint8)
    count, _labels, stats, _centers = cv2.connectedComponentsWithStats(dark_mask, 8)
    components = [
        (idx, int(stats[idx, cv2.CC_STAT_AREA]))
        for idx in range(1, count)
        if int(stats[idx, cv2.CC_STAT_AREA]) > 20
    ]
    if not components:
        return {
            "visible_shape_area_px": 0.0,
            "visible_shape_width_px": 0.0,
            "visible_shape_height_px": 0.0,
            "visible_shape_bbox_area_px": 0.0,
            "visible_shape_components": 0.0,
        }

    component_idx = max(components, key=lambda item: item[1])[0]
    width = float(stats[component_idx, cv2.CC_STAT_WIDTH])
    height = float(stats[component_idx, cv2.CC_STAT_HEIGHT])
    return {
        "visible_shape_area_px": float(stats[component_idx, cv2.CC_STAT_AREA]),
        "visible_shape_width_px": width,
        "visible_shape_height_px": height,
        "visible_shape_bbox_area_px": width * height,
        "visible_shape_components": float(len(components)),
    }


def _component_bboxes(mask: np.ndarray, min_area: int) -> list[tuple[int, int, int, int, int]]:
    count, _labels, stats, _centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [
        (
            int(stats[idx, cv2.CC_STAT_LEFT]),
            int(stats[idx, cv2.CC_STAT_TOP]),
            int(stats[idx, cv2.CC_STAT_WIDTH]),
            int(stats[idx, cv2.CC_STAT_HEIGHT]),
            int(stats[idx, cv2.CC_STAT_AREA]),
        )
        for idx in range(1, count)
        if int(stats[idx, cv2.CC_STAT_AREA]) >= min_area
    ]


def _largest_dark_bbox(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    min_area = max(3, int(gray.size * 0.002))
    components = _component_bboxes(gray < 55, min_area)
    if not components:
        return None
    x, y, width, height, _area = max(components, key=lambda item: item[4])
    return x, y, width, height


def _largest_motion_bbox(
    current: np.ndarray,
    baseline: np.ndarray,
    *,
    pixel_threshold: float,
) -> tuple[int, int, int, int] | None:
    min_area = max(3, int(current.size * 0.002))
    diff = np.abs(current.astype(float) - baseline.astype(float))
    components = _component_bboxes(diff > pixel_threshold, min_area)
    if not components:
        return None
    x, y, width, height, _area = max(components, key=lambda item: item[4])
    return x, y, width, height


def _rect_distance(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    first_x, first_y, first_w, first_h = first
    second_x, second_y, second_w, second_h = second
    dx = max(second_x - (first_x + first_w), first_x - (second_x + second_w), 0)
    dy = max(second_y - (first_y + first_h), first_y - (second_y + second_h), 0)
    return float(np.hypot(dx, dy))


def _hand_foam_metrics(
    current: np.ndarray,
    baseline: np.ndarray,
    foam_bbox: tuple[int, int, int, int] | None,
    *,
    pixel_threshold: float = 25.0,
) -> dict[str, float]:
    if foam_bbox is None:
        return {
            "visible_hand_foam_distance_px": 0.0,
            "visible_hand_foam_overlap_px": 0.0,
            "visible_hand_foam_pressure_proxy": 0.0,
        }

    hand_bbox = _largest_motion_bbox(current, baseline, pixel_threshold=pixel_threshold)
    if hand_bbox is None:
        distance = float(np.hypot(*current.shape))
        return {
            "visible_hand_foam_distance_px": distance,
            "visible_hand_foam_overlap_px": 0.0,
            "visible_hand_foam_pressure_proxy": -distance,
        }

    distance = _rect_distance(hand_bbox, foam_bbox)
    diff = np.abs(current.astype(float) - baseline.astype(float))
    motion_mask = diff > pixel_threshold
    x, y, width, height = foam_bbox
    foam_motion = motion_mask[y : y + height, x : x + width]
    overlap_px = float(foam_motion.sum())
    return {
        "visible_hand_foam_distance_px": distance,
        "visible_hand_foam_overlap_px": overlap_px,
        "visible_hand_foam_pressure_proxy": overlap_px - distance,
    }


def _delta_intensity_features(delta: np.ndarray, baseline_noise: float) -> dict[str, float]:
    positive = delta[delta > 0]
    negative = -delta[delta < 0]
    flat = delta.reshape(-1)
    top_count = max(1, int(np.ceil(flat.size * 0.01)))
    top_values = np.partition(flat, -top_count)[-top_count:]

    features: dict[str, float] = {
        "ir_roi_median_delta": float(np.median(delta)),
        "ir_roi_std_delta": float(delta.std()),
        "ir_sum_positive_delta": float(positive.sum()) if positive.size else 0.0,
        "ir_sum_negative_delta": float(negative.sum()) if negative.size else 0.0,
        "ir_l1_delta": float(np.abs(delta).sum()),
        "ir_l2_delta": float(np.sqrt(np.square(delta).sum())),
        "ir_p90_delta": float(np.quantile(flat, 0.90)),
        "ir_p95_delta": float(np.quantile(flat, 0.95)),
        "ir_p99_delta": float(np.quantile(flat, 0.99)),
        "ir_top_1pct_mean_delta": float(top_values.mean()),
        "ir_robust_max_delta": float(np.quantile(flat, 0.99)),
    }
    for sigma in (1, 2, 3):
        threshold = baseline_noise * sigma
        features[f"ir_positive_area_{sigma}sigma_px"] = int((delta > threshold).sum())
        features[f"ir_negative_area_{sigma}sigma_px"] = int((delta < -threshold).sum())
    return features


def _pca_scores(delta_fields: list[np.ndarray], components: int = 3) -> np.ndarray:
    if not delta_fields:
        return np.zeros((0, components), dtype=float)
    matrix = np.stack([field.reshape(-1) for field in delta_fields], axis=0).astype(float)
    matrix -= matrix.mean(axis=0, keepdims=True)
    scores = np.zeros((matrix.shape[0], components), dtype=float)
    if matrix.shape[0] < 2 or float(np.linalg.norm(matrix)) == 0.0:
        return scores
    u, singular_values, _vt = np.linalg.svd(matrix, full_matrices=False)
    available = min(components, len(singular_values))
    scores[:, :available] = u[:, :available] * singular_values[:available]
    return scores


def _write_delta_roi_fields(
    trial_dir: Path,
    rows: list[dict[str, str]],
    *,
    thermal_roi: tuple[int, int, int, int],
    baseline_image: np.ndarray,
    baseline_rows: list[dict[str, str]],
) -> None:
    frames = np.asarray([int(row["frame"]) for row in rows], dtype=np.int32)
    roi_fields = np.stack([_thermal_roi(trial_dir, int(frame), thermal_roi) for frame in frames], axis=0)
    delta_roi = roi_fields - baseline_image
    sweep_progress = np.asarray(
        [float(row["sweep_progress"]) if row.get("sweep_progress") else np.nan for row in rows],
        dtype=float,
    )
    np.savez_compressed(
        trial_dir / "ir_delta_roi_fields.npz",
        roi_scalar=roi_fields.astype(np.float32),
        delta_roi=delta_roi.astype(np.float32),
        baseline_mean_roi=baseline_image.astype(np.float32),
        frames=frames,
        phase=np.asarray([row["phase"] for row in rows]),
        sweep_progress=sweep_progress,
        roi_xywh=np.asarray(thermal_roi, dtype=np.int32),
        baseline_frames=np.asarray([int(row["frame"]) for row in baseline_rows], dtype=np.int32),
    )


def _pearson(values_a: Iterable[float], values_b: Iterable[float]) -> float:
    a = np.asarray(list(values_a), dtype=float)
    b = np.asarray(list(values_b), dtype=float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rankdata(values: Iterable[float]) -> np.ndarray:
    values_array = np.asarray(list(values), dtype=float)
    order = np.argsort(values_array)
    ranks = np.empty(len(values_array), dtype=float)
    i = 0
    while i < len(values_array):
        j = i + 1
        while j < len(values_array) and values_array[order[j]] == values_array[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def _spearman(values_a: Iterable[float], values_b: Iterable[float]) -> float:
    a = list(values_a)
    b = list(values_b)
    return _pearson(_rankdata(a), _rankdata(b))


def _rep_from_metadata(metadata: dict[str, object], trial_dir: Path) -> int:
    rep = metadata.get("rep")
    if isinstance(rep, int):
        return rep
    match = re.search(r"_rep(\d+)$", trial_dir.name)
    if not match:
        raise ValueError(f"could not infer rep from {trial_dir}")
    return int(match.group(1))


def _thermal_roi_from_metadata(metadata: dict[str, object]) -> tuple[int, int, int, int]:
    roi = metadata.get("thermal_roi")
    if not isinstance(roi, str) or not roi:
        raise ValueError("metadata must contain thermal_roi")
    return _parse_crop(roi)


def _validate_frame_counts(trial_dir: Path, rows: list[dict[str, str]]) -> None:
    expected = len(rows)
    thermal = len(list((trial_dir / "thermal").glob("frame_*.png")))
    bird = len(list((trial_dir / "bird").glob("frame_*.png")))
    if thermal != expected or bird != expected:
        raise RuntimeError(f"{trial_dir.name}: telemetry={expected}, thermal={thermal}, bird={bird}")


def _touch_motion_values(
    trial_dir: Path,
    baseline_rows: list[dict[str, str]],
    pressure_rows: list[dict[str, str]],
    touch_crop: tuple[int, int, int, int],
) -> tuple[int | None, list[float], float, float]:
    baseline_crops = np.stack(
        [_bird_crop(trial_dir, int(row["frame"]), touch_crop) for row in baseline_rows],
        axis=0,
    )
    baseline_template = np.median(baseline_crops, axis=0)
    baseline_diffs = np.asarray(
        [float(np.mean(np.abs(crop - baseline_template))) for crop in baseline_crops],
        dtype=float,
    )
    threshold = max(float(baseline_diffs.mean() + 6.0 * baseline_diffs.std()), 2.0)
    motion_values = [
        float(np.mean(np.abs(_bird_crop(trial_dir, int(row["frame"]), touch_crop) - baseline_template)))
        for row in pressure_rows
    ]
    onset_index = next((idx for idx, value in enumerate(motion_values) if value > threshold), None)
    return onset_index, motion_values, threshold, float(baseline_diffs.max())


def analyze_trial(
    trial_dir: Path,
    *,
    main_reps: tuple[int, ...],
    touch_crop: tuple[int, int, int, int],
    shape_crop: tuple[int, int, int, int],
) -> tuple[TrialSummary, list[dict[str, object]]]:
    metadata = json.loads((trial_dir / "metadata.json").read_text())
    rows = _read_csv(trial_dir / "telemetry.csv")
    _validate_frame_counts(trial_dir, rows)

    rep = _rep_from_metadata(metadata, trial_dir)
    trial_id = str(metadata.get("trial_id", trial_dir.name))
    thermal_roi = _thermal_roi_from_metadata(metadata)

    baseline_rows = [row for row in rows if row.get("phase") == "baseline"]
    pressure_rows = [row for row in rows if row.get("phase") == "pressure_sweep"]
    if not baseline_rows or not pressure_rows:
        raise RuntimeError(f"{trial_dir.name}: missing baseline or pressure_sweep rows")

    onset_index, motion_values, motion_threshold, baseline_motion_max = _touch_motion_values(
        trial_dir,
        baseline_rows,
        pressure_rows,
        touch_crop,
    )
    start_index = onset_index if onset_index is not None else 0
    onset_row = pressure_rows[start_index]
    onset_motion = motion_values[start_index]
    onset_progress = float(onset_row["sweep_progress"])

    baseline_stack = np.stack(
        [_thermal_roi(trial_dir, int(row["frame"]), thermal_roi) for row in baseline_rows],
        axis=0,
    )
    baseline_image = baseline_stack.mean(axis=0)
    baseline_noise = float(baseline_stack.std(axis=0).mean())
    ir_threshold = max(baseline_noise * 3.0, 1.0)
    _write_delta_roi_fields(
        trial_dir,
        rows,
        thermal_roi=thermal_roi,
        baseline_image=baseline_image,
        baseline_rows=baseline_rows,
    )

    baseline_touch_template = np.median(
        np.stack(
            [_bird_crop(trial_dir, int(row["frame"]), touch_crop) for row in baseline_rows],
            axis=0,
        ),
        axis=0,
    )
    baseline_foam_bbox = _largest_dark_bbox(baseline_touch_template)

    baseline_shape = {
        key: float(np.median([_dark_shape_metrics(trial_dir, int(row["frame"]), shape_crop)[key] for row in baseline_rows]))
        for key in (
            "visible_shape_area_px",
            "visible_shape_width_px",
            "visible_shape_height_px",
            "visible_shape_bbox_area_px",
        )
    }

    use_trial = rep in main_reps and onset_index is not None
    if rep not in main_reps:
        reason = "not in main reps"
    elif onset_index is None:
        reason = "no detected touch onset"
    else:
        reason = "main"

    records: list[dict[str, object]] = []
    post_touch_delta_fields: list[np.ndarray] = []
    for pressure_index, row in enumerate(pressure_rows[start_index:], start=start_index):
        frame = int(row["frame"])
        progress = float(row["sweep_progress"])
        thermal_delta = _thermal_roi(trial_dir, frame, thermal_roi) - baseline_image
        shape = _dark_shape_metrics(trial_dir, frame, shape_crop)
        hand_foam = _hand_foam_metrics(
            _bird_crop(trial_dir, frame, touch_crop),
            baseline_touch_template,
            baseline_foam_bbox,
        )
        post_touch_time = (progress - onset_progress) / max(1e-9, 1.0 - onset_progress)
        post_touch_delta_fields.append(thermal_delta)

        records.append(
            {
                "trial_id": trial_id,
                "rep": rep,
                "frame": frame,
                "phase": row["phase"],
                "sweep_progress": progress,
                "post_touch_time_progress": post_touch_time,
                "visible_motion_delta": motion_values[pressure_index] - onset_motion,
                "visible_shape_area_delta_px": shape["visible_shape_area_px"]
                - baseline_shape["visible_shape_area_px"],
                "visible_shape_width_delta_px": shape["visible_shape_width_px"]
                - baseline_shape["visible_shape_width_px"],
                "visible_shape_height_delta_px": shape["visible_shape_height_px"]
                - baseline_shape["visible_shape_height_px"],
                "visible_shape_bbox_area_delta_px": shape["visible_shape_bbox_area_px"]
                - baseline_shape["visible_shape_bbox_area_px"],
                "visible_shape_components": shape["visible_shape_components"],
                **hand_foam,
                "ir_roi_mean_delta": float(thermal_delta.mean()),
                **_delta_intensity_features(thermal_delta, baseline_noise),
                "ir_positive_area_px": int((thermal_delta > ir_threshold).sum()),
                "ir_negative_area_px": int((thermal_delta < -ir_threshold).sum()),
                "ir_positive_mean_delta": float(thermal_delta[thermal_delta > ir_threshold].mean())
                if np.any(thermal_delta > ir_threshold)
                else 0.0,
                "ir_negative_mean_delta": float((-thermal_delta[thermal_delta < -ir_threshold]).mean())
                if np.any(thermal_delta < -ir_threshold)
                else 0.0,
                "used_in_main": use_trial,
            }
        )
    scores = _pca_scores(post_touch_delta_fields, components=3)
    for record, row_scores in zip(records, scores, strict=True):
        record["ir_pca1_score"] = float(row_scores[0])
        record["ir_pca2_score"] = float(row_scores[1])
        record["ir_pca3_score"] = float(row_scores[2])

    summary = TrialSummary(
        trial_id=trial_id,
        rep=rep,
        used_in_main=use_trial,
        reason=reason,
        rows=len(rows),
        baseline_frames=len(baseline_rows),
        pressure_frames=len(pressure_rows),
        post_touch_frames=len(records),
        touch_onset_frame=int(onset_row["frame"]) if onset_index is not None else None,
        touch_onset_progress=onset_progress if onset_index is not None else None,
        touch_threshold=motion_threshold,
        baseline_motion_max=baseline_motion_max,
        ir_threshold=ir_threshold,
    )
    return summary, records


def _normalize_per_trial(records: list[dict[str, object]], field: str) -> list[float]:
    normalized: list[float] = []
    for trial_id in sorted({str(record["trial_id"]) for record in records}):
        trial_records = [record for record in records if record["trial_id"] == trial_id]
        values = np.asarray([float(record[field]) for record in trial_records], dtype=float)
        value_range = float(np.ptp(values))
        if value_range > 0:
            values = (values - float(values.min())) / value_range
        else:
            values = np.zeros_like(values)
        normalized.extend(float(value) for value in values)
    return normalized


def _correlation_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    proxies = (
        "post_touch_time_progress",
        "visible_motion_delta",
        "visible_hand_foam_pressure_proxy",
        "visible_hand_foam_distance_px",
        "visible_hand_foam_overlap_px",
        "visible_shape_height_delta_px",
    )
    ir_fields = (
        "ir_roi_mean_delta",
        "ir_roi_median_delta",
        "ir_roi_std_delta",
        "ir_sum_positive_delta",
        "ir_sum_negative_delta",
        "ir_l1_delta",
        "ir_l2_delta",
        "ir_p90_delta",
        "ir_p95_delta",
        "ir_p99_delta",
        "ir_top_1pct_mean_delta",
        "ir_robust_max_delta",
        "ir_positive_area_1sigma_px",
        "ir_positive_area_2sigma_px",
        "ir_positive_area_3sigma_px",
        "ir_negative_area_1sigma_px",
        "ir_negative_area_2sigma_px",
        "ir_negative_area_3sigma_px",
        "ir_positive_area_px",
        "ir_negative_area_px",
        "ir_pca1_score",
        "ir_pca2_score",
        "ir_pca3_score",
    )
    out: list[dict[str, object]] = []
    main_records = [record for record in records if record["used_in_main"]]
    for proxy in proxies:
        for ir_field in ir_fields:
            if any(ir_field not in record for record in main_records):
                continue
            per_rep = []
            for rep in sorted({int(record["rep"]) for record in main_records}):
                rep_records = [record for record in main_records if record["rep"] == rep]
                x = [float(record[proxy]) for record in rep_records]
                y = [float(record[ir_field]) for record in rep_records]
                per_rep.append(_pearson(x, y))
            x_norm = _normalize_per_trial(main_records, proxy)
            y_norm = _normalize_per_trial(main_records, ir_field)
            out.append(
                {
                    "proxy": proxy,
                    "ir_signal": ir_field,
                    "pooled_normalized_pearson": _pearson(x_norm, y_norm),
                    "pooled_normalized_spearman": _spearman(x_norm, y_norm),
                    "median_abs_per_rep_pearson": float(np.median([abs(corr) for corr in per_rep if np.isfinite(corr)]))
                    if any(np.isfinite(corr) for corr in per_rep)
                    else float("nan"),
                    "per_rep_pearson": ";".join(
                        f"rep{rep}:{corr:.3f}"
                        for rep, corr in zip(
                            sorted({int(record["rep"]) for record in main_records}),
                            per_rep,
                            strict=True,
                        )
                    ),
                }
            )
    return out


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_selected_relation(
    records: list[dict[str, object]],
    correlations: list[dict[str, object]],
    out_path: Path,
) -> None:
    main_records = [record for record in records if record["used_in_main"]]
    if not main_records:
        raise RuntimeError("no main records available for plotting")

    proxy_display = PROXY_LABELS.get(SELECTED_PROXY, SELECTED_PROXY)
    ir_display = IR_LABELS.get(SELECTED_IR, SELECTED_IR)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    scatter_ax, compare_ax = axes

    colors = {2: "tab:blue", 3: "tab:green", 4: "tab:purple", 5: "tab:orange"}
    for rep in sorted({int(record["rep"]) for record in main_records}):
        rep_records = [record for record in main_records if record["rep"] == rep]
        x = [float(record[SELECTED_PROXY]) for record in rep_records]
        y = [float(record[SELECTED_IR]) for record in rep_records]
        scatter_ax.scatter(x, y, s=24, alpha=0.72, color=colors.get(rep, "tab:gray"), label=f"rep{rep:02d}")

    x_all = np.asarray([float(record[SELECTED_PROXY]) for record in main_records], dtype=float)
    y_all = np.asarray([float(record[SELECTED_IR]) for record in main_records], dtype=float)
    if len(x_all) >= 2 and float(np.ptp(x_all)) > 0:
        slope, intercept = np.polyfit(x_all, y_all, deg=1)
        x_line = np.linspace(float(x_all.min()), float(x_all.max()), 100)
        scatter_ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=1.6, label="linear fit")

    selected = next(
        row
        for row in correlations
        if row["proxy"] == SELECTED_PROXY and row["ir_signal"] == SELECTED_IR
    )
    scatter_ax.set_title(
        (
            "Post-touch relation\n"
            f"pooled r={selected['pooled_normalized_pearson']:.2f}, "
            f"median |per-rep r|={selected['median_abs_per_rep_pearson']:.2f}"
        ),
        fontsize=10,
    )
    scatter_ax.set_xlabel(proxy_display)
    scatter_ax.set_ylabel(ir_display)
    scatter_ax.legend(fontsize=8)
    scatter_ax.grid(True, alpha=0.22)

    compare_rows = [row for row in correlations if row["ir_signal"] == SELECTED_IR]
    x_names = [PROXY_LABELS.get(str(row["proxy"]), str(row["proxy"])) for row in compare_rows]
    scores = [float(row["median_abs_per_rep_pearson"]) for row in compare_rows]
    colors = ["tab:blue" if row["proxy"] == SELECTED_PROXY else "0.65" for row in compare_rows]
    compare_ax.bar(x_names, scores, color=colors)
    compare_ax.set_ylim(0, 1)
    compare_ax.set_title("Proxy comparison for selected IR signal", fontsize=10)
    compare_ax.set_ylabel("median |per-rep Pearson r|")
    compare_ax.tick_params(axis="x", rotation=25)
    compare_ax.grid(axis="y", alpha=0.22)

    fig.suptitle(f"Hand pressure pilot: {proxy_display} vs {ir_display} after touch", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_ir_feature_ranking(correlations: list[dict[str, object]], out_path: Path) -> None:
    rows = [
        row
        for row in correlations
        if row["proxy"] == SELECTED_PROXY
        and np.isfinite(float(row["median_abs_per_rep_pearson"]))
    ]
    rows = sorted(rows, key=lambda row: float(row["median_abs_per_rep_pearson"]), reverse=True)[:12]
    if not rows:
        raise RuntimeError("no correlation rows available for IR feature ranking")

    labels = [IR_LABELS.get(str(row["ir_signal"]), str(row["ir_signal"])) for row in rows]
    scores = [float(row["median_abs_per_rep_pearson"]) for row in rows]
    colors = ["tab:blue" if row["ir_signal"] == SELECTED_IR else "0.65" for row in rows]
    y_positions = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    ax.barh(y_positions, scores, color=colors)
    ax.set_yticks(y_positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("median |per-rep Pearson r|")
    ax.set_title(f"IR feature ranking vs {PROXY_LABELS.get(SELECTED_PROXY, SELECTED_PROXY)}", fontsize=11)
    ax.grid(axis="x", alpha=0.22)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _normalize_values(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    value_range = float(np.ptp(array))
    if value_range <= 0:
        return np.zeros_like(array)
    return (array - float(array.min())) / value_range


def _linear_prediction(target: Iterable[float], predictors: list[Iterable[float]]) -> np.ndarray:
    y = np.asarray(list(target), dtype=float)
    columns = [np.ones(len(y), dtype=float)]
    columns.extend(np.asarray(list(predictor), dtype=float) for predictor in predictors)
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return design @ coefficients


def _r2_score(actual: Iterable[float], predicted: Iterable[float]) -> float:
    y = np.asarray(list(actual), dtype=float)
    y_hat = np.asarray(list(predicted), dtype=float)
    if len(y) < 2:
        return float("nan")
    total = float(np.square(y - float(y.mean())).sum())
    if total == 0:
        return float("nan")
    residual = float(np.square(y - y_hat).sum())
    return 1.0 - residual / total


def _residuals_after_controls(values: Iterable[float], controls: list[Iterable[float]]) -> np.ndarray:
    y = np.asarray(list(values), dtype=float)
    if not controls:
        return y - float(y.mean())
    return y - _linear_prediction(y, controls)


def _median_abs(values: Iterable[float]) -> float:
    finite_values = [abs(float(value)) for value in values if np.isfinite(float(value))]
    if not finite_values:
        return float("nan")
    return float(np.median(finite_values))


def _time_control_rows(
    records: list[dict[str, object]],
    *,
    proxy: str = SELECTED_PROXY,
    ir_fields: tuple[str, ...] = (SELECTED_IR,),
) -> list[dict[str, object]]:
    main_records = [record for record in records if record["used_in_main"]]
    if not main_records:
        return []

    rows: list[dict[str, object]] = []
    proxy_norm = _normalize_per_trial(main_records, proxy)
    time_norm = _normalize_per_trial(main_records, "post_touch_time_progress")
    for ir_field in ir_fields:
        if any(ir_field not in record for record in main_records):
            continue
        ir_norm = _normalize_per_trial(main_records, ir_field)
        time_prediction = _linear_prediction(proxy_norm, [time_norm])
        ir_prediction = _linear_prediction(proxy_norm, [ir_norm])
        time_plus_ir_prediction = _linear_prediction(proxy_norm, [time_norm, ir_norm])
        proxy_residual = _residuals_after_controls(proxy_norm, [time_norm])
        ir_residual = _residuals_after_controls(ir_norm, [time_norm])

        per_rep_partial: list[float] = []
        rep_labels: list[str] = []
        for rep in sorted({int(record["rep"]) for record in main_records}):
            rep_records = [record for record in main_records if int(record["rep"]) == rep]
            rep_proxy = _normalize_values(float(record[proxy]) for record in rep_records)
            rep_time = _normalize_values(float(record["post_touch_time_progress"]) for record in rep_records)
            rep_ir = _normalize_values(float(record[ir_field]) for record in rep_records)
            rep_proxy_residual = _residuals_after_controls(rep_proxy, [rep_time])
            rep_ir_residual = _residuals_after_controls(rep_ir, [rep_time])
            corr = _pearson(rep_proxy_residual, rep_ir_residual)
            per_rep_partial.append(corr)
            rep_labels.append(f"rep{rep}:{corr:.3f}")

        time_only_r2 = _r2_score(proxy_norm, time_prediction)
        ir_only_r2 = _r2_score(proxy_norm, ir_prediction)
        time_plus_ir_r2 = _r2_score(proxy_norm, time_plus_ir_prediction)
        rows.append(
            {
                "proxy": proxy,
                "ir_signal": ir_field,
                "time_only_r2": time_only_r2,
                "ir_only_r2": ir_only_r2,
                "time_plus_ir_r2": time_plus_ir_r2,
                "delta_r2": time_plus_ir_r2 - time_only_r2,
                "partial_pearson_after_time": _pearson(proxy_residual, ir_residual),
                "median_abs_per_rep_partial_pearson_after_time": _median_abs(per_rep_partial),
                "per_rep_partial_pearson_after_time": ";".join(rep_labels),
            }
        )
    return rows


def _lagged_pair(
    proxy_values: np.ndarray,
    ir_values: np.ndarray,
    lag_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    if lag_frames > 0:
        return proxy_values[:-lag_frames], ir_values[lag_frames:]
    if lag_frames < 0:
        lead = abs(lag_frames)
        return proxy_values[lead:], ir_values[:-lead]
    return proxy_values, ir_values


def _lagged_correlation_rows(
    records: list[dict[str, object]],
    *,
    proxy: str = SELECTED_PROXY,
    ir_field: str = SELECTED_IR,
    max_lag_frames: int = 10,
) -> list[dict[str, object]]:
    main_records = [record for record in records if record["used_in_main"]]
    if not main_records or any(ir_field not in record for record in main_records):
        return []

    rows: list[dict[str, object]] = []
    for lag_frames in range(-max_lag_frames, max_lag_frames + 1):
        pooled_proxy: list[float] = []
        pooled_ir: list[float] = []
        per_rep: list[float] = []
        rep_labels: list[str] = []
        for rep in sorted({int(record["rep"]) for record in main_records}):
            rep_records = sorted(
                (record for record in main_records if int(record["rep"]) == rep),
                key=lambda record: int(record["frame"]),
            )
            if len(rep_records) <= abs(lag_frames):
                continue
            proxy_values = _normalize_values(float(record[proxy]) for record in rep_records)
            ir_values = _normalize_values(float(record[ir_field]) for record in rep_records)
            proxy_aligned, ir_aligned = _lagged_pair(proxy_values, ir_values, lag_frames)
            if len(proxy_aligned) < 2:
                continue
            pooled_proxy.extend(float(value) for value in proxy_aligned)
            pooled_ir.extend(float(value) for value in ir_aligned)
            corr = _pearson(proxy_aligned, ir_aligned)
            per_rep.append(corr)
            rep_labels.append(f"rep{rep}:{corr:.3f}")

        rows.append(
            {
                "proxy": proxy,
                "ir_signal": ir_field,
                "lag_frames": lag_frames,
                "pooled_pearson": _pearson(pooled_proxy, pooled_ir),
                "median_abs_per_rep_pearson": _median_abs(per_rep),
                "per_rep_pearson": ";".join(rep_labels),
                "pair_count": len(pooled_proxy),
            }
        )
    return rows


def _normalized_selected_points(records: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for rep in sorted({int(record["rep"]) for record in records}):
        rep_records = [record for record in records if int(record["rep"]) == rep]
        x_norm = _normalize_values(float(record[SELECTED_PROXY]) for record in rep_records)
        y_norm = _normalize_values(float(record[SELECTED_IR]) for record in rep_records)
        for record, x_value, y_value in zip(rep_records, x_norm, y_norm, strict=True):
            normalized.append(
                {
                    "rep": rep,
                    "frame": int(record["frame"]),
                    "x_norm": float(x_value),
                    "y_norm": float(y_value),
                }
            )
    return normalized


def _representative_reps(records: list[dict[str, object]]) -> dict[int, float]:
    correlations_by_rep: dict[int, float] = {}
    for rep in sorted({int(record["rep"]) for record in records}):
        rep_records = [record for record in records if int(record["rep"]) == rep]
        correlations_by_rep[rep] = _pearson(
            (float(record[SELECTED_PROXY]) for record in rep_records),
            (float(record[SELECTED_IR]) for record in rep_records),
        )
    good_reps = {
        rep: corr
        for rep, corr in correlations_by_rep.items()
        if np.isfinite(corr) and corr >= 0.5
    }
    return good_reps if good_reps else correlations_by_rep


def _quantile_bins(
    points: list[dict[str, object]],
    *,
    bins: int = 6,
) -> list[dict[str, float]]:
    if not points:
        return []
    sorted_points = sorted(points, key=lambda item: float(item["x_norm"]))
    chunks = np.array_split(sorted_points, min(bins, len(sorted_points)))
    out: list[dict[str, float]] = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        x_values = np.asarray([float(item["x_norm"]) for item in chunk], dtype=float)
        y_values = np.asarray([float(item["y_norm"]) for item in chunk], dtype=float)
        out.append(
            {
                "x_median": float(np.median(x_values)),
                "y_median": float(np.median(y_values)),
                "y_q25": float(np.quantile(y_values, 0.25)),
                "y_q75": float(np.quantile(y_values, 0.75)),
                "count": float(len(chunk)),
            }
        )
    return out


def _plot_clear_relation(
    records: list[dict[str, object]],
    correlations: list[dict[str, object]],
    out_path: Path,
) -> None:
    main_records = [record for record in records if record["used_in_main"]]
    if not main_records:
        raise RuntimeError("no main records available for plotting")

    selected = next(
        row
        for row in correlations
        if row["proxy"] == SELECTED_PROXY and row["ir_signal"] == SELECTED_IR
    )
    normalized_points = _normalized_selected_points(main_records)
    representative = _representative_reps(main_records)
    representative_points = [point for point in normalized_points if int(point["rep"]) in representative]

    colors = {2: "tab:blue", 3: "tab:green", 4: "tab:purple", 5: "tab:orange"}
    proxy_display = PROXY_LABELS.get(SELECTED_PROXY, SELECTED_PROXY)
    ir_display = IR_LABELS.get(SELECTED_IR, SELECTED_IR)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), constrained_layout=True)
    all_ax, binned_ax, proxy_ax = axes

    for rep in sorted({int(point["rep"]) for point in normalized_points}):
        rep_points = [point for point in normalized_points if int(point["rep"]) == rep]
        x = np.asarray([float(point["x_norm"]) for point in rep_points], dtype=float)
        y = np.asarray([float(point["y_norm"]) for point in rep_points], dtype=float)
        corr = _pearson(x, y)
        representative_rep = rep in representative
        color = colors.get(rep, "tab:gray") if representative_rep else "0.6"
        alpha = 0.78 if representative_rep else 0.32
        label_suffix = "representative" if representative_rep else "inconsistent"
        all_ax.scatter(
            x,
            y,
            s=34 if representative_rep else 22,
            alpha=alpha,
            color=color,
            label=f"rep{rep:02d} r={corr:.2f} ({label_suffix})",
        )
        if len(x) >= 2 and float(np.ptp(x)) > 0:
            slope, intercept = np.polyfit(x, y, deg=1)
            x_line = np.linspace(0.0, 1.0, 100)
            all_ax.plot(
                x_line,
                slope * x_line + intercept,
                color=color,
                linewidth=2.0 if representative_rep else 1.2,
                alpha=0.9 if representative_rep else 0.45,
            )

    all_ax.set_title("Normalized post-touch points by rep", fontsize=10)
    all_ax.set_xlabel(f"{proxy_display}, normalized within rep")
    all_ax.set_ylabel(f"{ir_display}, normalized within rep")
    all_ax.set_xlim(-0.04, 1.04)
    all_ax.set_ylim(-0.08, 1.08)
    all_ax.grid(True, alpha=0.22)
    all_ax.legend(fontsize=7, loc="lower right")

    paired_values: list[tuple[float, float]] = []
    for rep in sorted(representative):
        rep_points = [point for point in representative_points if int(point["rep"]) == rep]
        if len(rep_points) < 4:
            continue
        x_values = np.asarray([float(point["x_norm"]) for point in rep_points], dtype=float)
        y_values = np.asarray([float(point["y_norm"]) for point in rep_points], dtype=float)
        low_cut = float(np.quantile(x_values, 0.25))
        high_cut = float(np.quantile(x_values, 0.75))
        low_y = y_values[x_values <= low_cut]
        high_y = y_values[x_values >= high_cut]
        if not len(low_y) or not len(high_y):
            continue
        pair = (float(np.median(low_y)), float(np.median(high_y)))
        paired_values.append(pair)
        binned_ax.plot(
            [0, 1],
            pair,
            marker="o",
            linewidth=2.0,
            color=colors.get(rep, "tab:gray"),
            alpha=0.86,
            label=f"rep{rep:02d}",
        )

    if paired_values:
        low_mean = float(np.mean([pair[0] for pair in paired_values]))
        high_mean = float(np.mean([pair[1] for pair in paired_values]))
        binned_ax.plot(
            [0, 1],
            [low_mean, high_mean],
            marker="o",
            linewidth=4.0,
            color="black",
            label="mean",
        )
    rep_text = ", ".join(f"rep{rep:02d}" for rep in sorted(representative))
    binned_ax.set_title(f"Representative reps: low vs high {proxy_display} ({rep_text})", fontsize=10)
    binned_ax.set_xlabel(f"{proxy_display} bin")
    binned_ax.set_ylabel(f"median normalized {ir_display}")
    binned_ax.set_xticks([0, 1], ["low", "high"])
    binned_ax.set_xlim(-0.18, 1.18)
    binned_ax.set_ylim(-0.08, 1.08)
    binned_ax.grid(True, alpha=0.22)
    binned_ax.legend(fontsize=8, loc="lower right")

    compare_rows = [row for row in correlations if row["ir_signal"] == SELECTED_IR]
    labels = [PROXY_LABELS.get(str(row["proxy"]), str(row["proxy"])) for row in compare_rows]
    scores = [float(row["median_abs_per_rep_pearson"]) for row in compare_rows]
    colors_for_bars = [
        "tab:blue" if str(row["proxy"]) == SELECTED_PROXY else "0.65"
        for row in compare_rows
    ]
    proxy_ax.bar(labels, scores, color=colors_for_bars)
    proxy_ax.set_ylim(0, 1)
    proxy_ax.set_title("Proxy comparison for selected IR signal", fontsize=10)
    proxy_ax.set_ylabel("median |per-rep Pearson r|")
    proxy_ax.tick_params(axis="x", rotation=20)
    proxy_ax.grid(axis="y", alpha=0.22)

    fig.suptitle(
        (
            "Clearer hand-pressure relation: normalize per rep and focus after touch\n"
            f"{proxy_display} vs {ir_display}: pooled normalized r={selected['pooled_normalized_pearson']:.2f}, "
            f"median |per-rep r|={selected['median_abs_per_rep_pearson']:.2f}"
        ),
        fontsize=12,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _parse_rep_values(value: object) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for part in str(value).split(";"):
        if not part or ":" not in part:
            continue
        label, raw_value = part.split(":", 1)
        try:
            out.append((label, float(raw_value)))
        except ValueError:
            continue
    return out


def _plot_time_control_summary(
    time_control: list[dict[str, object]],
    lagged: list[dict[str, object]],
    out_path: Path,
) -> None:
    if not time_control:
        raise RuntimeError("no time-control rows available for plotting")
    selected = time_control[0]
    finite_lagged = [row for row in lagged if np.isfinite(float(row["pooled_pearson"]))]
    best_lag = max(finite_lagged, key=lambda row: float(row["pooled_pearson"])) if finite_lagged else None

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    r2_ax, lag_ax, rep_ax = axes

    r2_labels = ["time only", "IR only", "time + IR"]
    r2_values = [
        float(selected["time_only_r2"]),
        float(selected["ir_only_r2"]),
        float(selected["time_plus_ir_r2"]),
    ]
    r2_ax.bar(r2_labels, r2_values, color=["0.55", "tab:blue", "tab:green"])
    r2_ax.set_ylim(0, max(1.0, max(r2_values) * 1.08))
    r2_ax.set_ylabel("R2 on per-rep normalized proxy")
    r2_ax.set_title(f"Increment over time-only: {float(selected['delta_r2']):.3f}", fontsize=10)
    r2_ax.grid(axis="y", alpha=0.22)

    if lagged:
        lag_x = [int(row["lag_frames"]) for row in lagged]
        lag_y = [float(row["pooled_pearson"]) for row in lagged]
        lag_ax.plot(lag_x, lag_y, marker="o", linewidth=1.6, color="tab:blue")
        lag_ax.axvline(0, color="0.25", linewidth=1.0, linestyle="--")
        if best_lag is not None:
            lag_ax.scatter(
                [int(best_lag["lag_frames"])],
                [float(best_lag["pooled_pearson"])],
                s=70,
                color="tab:green",
                zorder=3,
                label=f"best lag {int(best_lag['lag_frames'])}",
            )
            lag_ax.legend(fontsize=8)
    lag_ax.set_title("Lag scan: positive lag means IR later", fontsize=10)
    lag_ax.set_xlabel("IR lag, frames")
    lag_ax.set_ylabel("pooled Pearson r")
    lag_ax.grid(True, alpha=0.22)

    rep_values = _parse_rep_values(selected["per_rep_partial_pearson_after_time"])
    if rep_values:
        labels = [label for label, _value in rep_values]
        values = [value for _label, value in rep_values]
        colors = ["tab:green" if value >= 0 else "tab:red" for value in values]
        rep_ax.bar(labels, values, color=colors)
        rep_ax.axhline(0, color="0.25", linewidth=1.0)
    rep_ax.set_ylim(-1.0, 1.0)
    rep_ax.set_title(
        f"Partial r after time: {float(selected['partial_pearson_after_time']):.3f}",
        fontsize=10,
    )
    rep_ax.set_ylabel("partial Pearson r")
    rep_ax.grid(axis="y", alpha=0.22)

    proxy_display = PROXY_LABELS.get(str(selected["proxy"]), str(selected["proxy"]))
    ir_display = IR_LABELS.get(str(selected["ir_signal"]), str(selected["ir_signal"]))
    fig.suptitle(f"Time-control test: {proxy_display} vs {ir_display}", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _write_time_control_report(
    report_path: Path,
    *,
    selected_time_control: dict[str, object],
    selected_correlation: dict[str, object],
    best_lag: dict[str, object] | None,
    summary_plot: Path,
) -> None:
    delta_r2 = float(selected_time_control["delta_r2"])
    partial = float(selected_time_control["partial_pearson_after_time"])
    survived = np.isfinite(delta_r2) and np.isfinite(partial) and delta_r2 > 0 and partial > 0
    verdict = (
        "The selected IR feature keeps a positive relation after controlling for time."
        if survived
        else "The selected IR feature does not clearly survive the time-only control."
    )
    best_lag_text = (
        "not available"
        if best_lag is None
        else f"{int(best_lag['lag_frames'])} frames, pooled r={float(best_lag['pooled_pearson']):.3f}"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Hand-Foam IR Pressure Time-Control Result",
                "",
                f"Verdict: {verdict}",
                "",
                "Main data: reps 2-5, rep01 excluded from the primary claim.",
                "",
                "Selected proxy: "
                f"{PROXY_LABELS.get(str(selected_time_control['proxy']), str(selected_time_control['proxy']))}",
                "Selected IR signal: "
                f"{IR_LABELS.get(str(selected_time_control['ir_signal']), str(selected_time_control['ir_signal']))}",
                "",
                "Key numbers:",
                f"- Original pooled normalized Pearson r: {float(selected_correlation['pooled_normalized_pearson']):.3f}",
                f"- Original median abs per-rep Pearson r: {float(selected_correlation['median_abs_per_rep_pearson']):.3f}",
                f"- Time-only R2: {float(selected_time_control['time_only_r2']):.3f}",
                f"- IR-only R2: {float(selected_time_control['ir_only_r2']):.3f}",
                f"- Time-plus-IR R2: {float(selected_time_control['time_plus_ir_r2']):.3f}",
                f"- Incremental R2 from IR after time: {delta_r2:.3f}",
                f"- Partial Pearson after removing time: {partial:.3f}",
                f"- Per-rep partial Pearson after time: {selected_time_control['per_rep_partial_pearson_after_time']}",
                f"- Best lag scan result: {best_lag_text}",
                "",
                f"Summary plot: {summary_plot.name}",
                "",
                "Interpretation:",
                "This is the check that separates a useful IR-pressure signal from a simple sweep-time trend. "
                "Use the result as evidence for the hand/foam relation only if the incremental R2 and partial "
                "correlation stay positive and the per-rep values have the same sign.",
                "",
            ]
        )
    )


def analyze(
    root: Path,
    *,
    trial_glob: str,
    main_reps: tuple[int, ...],
    touch_crop: tuple[int, int, int, int],
    shape_crop: tuple[int, int, int, int],
) -> tuple[list[TrialSummary], list[dict[str, object]], list[dict[str, object]]]:
    summaries: list[TrialSummary] = []
    records: list[dict[str, object]] = []
    for trial_dir in sorted((root / "trials").glob(trial_glob)):
        if not trial_dir.is_dir():
            continue
        summary, trial_records = analyze_trial(
            trial_dir,
            main_reps=main_reps,
            touch_crop=touch_crop,
            shape_crop=shape_crop,
        )
        summaries.append(summary)
        records.extend(trial_records)
    correlations = _correlation_rows(records)
    return summaries, records, correlations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--trial-glob", default="hand-pressure_*_sweep_rep*")
    parser.add_argument("--main-reps", type=_parse_main_reps, default=DEFAULT_MAIN_REPS)
    parser.add_argument("--touch-crop", type=_parse_crop, default=DEFAULT_TOUCH_CROP)
    parser.add_argument("--shape-crop", type=_parse_crop, default=DEFAULT_SHAPE_CROP)
    parser.add_argument("--max-lag-frames", type=int, default=10)
    parser.add_argument("--out-prefix", default="hand_pressure_post_touch")
    args = parser.parse_args()

    summaries, records, correlations = analyze(
        args.root,
        trial_glob=args.trial_glob,
        main_reps=args.main_reps,
        touch_crop=args.touch_crop,
        shape_crop=args.shape_crop,
    )
    if not records:
        raise SystemExit("no records found")

    time_control = _time_control_rows(records, proxy=SELECTED_PROXY, ir_fields=(SELECTED_IR,))
    lagged = _lagged_correlation_rows(
        records,
        proxy=SELECTED_PROXY,
        ir_field=SELECTED_IR,
        max_lag_frames=args.max_lag_frames,
    )
    prefix = args.root / args.out_prefix
    _write_csv(
        prefix.with_name(prefix.name + "_frame_metrics.csv"),
        records,
    )
    _write_csv(
        prefix.with_name(prefix.name + "_trial_summary.csv"),
        [summary.__dict__ for summary in summaries],
    )
    _write_csv(
        prefix.with_name(prefix.name + "_proxy_comparison.csv"),
        correlations,
    )
    _write_csv(
        prefix.with_name(prefix.name + "_time_control.csv"),
        time_control,
    )
    _write_csv(
        prefix.with_name(prefix.name + "_lagged_correlation.csv"),
        lagged,
    )
    _plot_selected_relation(
        records,
        correlations,
        prefix.with_name(prefix.name + f"_{SELECTED_PROXY_SLUG}_vs_{SELECTED_IR_SLUG}.png"),
    )
    _plot_ir_feature_ranking(
        correlations,
        prefix.with_name(prefix.name + "_ir_feature_ranking.png"),
    )
    _plot_clear_relation(
        records,
        correlations,
        prefix.with_name(prefix.name + "_clear_normalized_relation.png"),
    )
    time_control_plot = prefix.with_name(prefix.name + "_time_control_summary.png")
    _plot_time_control_summary(time_control, lagged, time_control_plot)

    selected = next(
        row
        for row in correlations
        if row["proxy"] == SELECTED_PROXY and row["ir_signal"] == SELECTED_IR
    )
    selected_time_control = next(
        row
        for row in time_control
        if row["proxy"] == SELECTED_PROXY and row["ir_signal"] == SELECTED_IR
    )
    finite_lagged = [row for row in lagged if np.isfinite(float(row["pooled_pearson"]))]
    best_lag = max(finite_lagged, key=lambda row: float(row["pooled_pearson"])) if finite_lagged else None
    report_path = prefix.with_name(prefix.name + "_time_control_report.md")
    _write_time_control_report(
        report_path,
        selected_time_control=selected_time_control,
        selected_correlation=selected,
        best_lag=best_lag,
        summary_plot=time_control_plot,
    )
    print(f"wrote {prefix.with_name(prefix.name + '_frame_metrics.csv')}")
    print(f"wrote {prefix.with_name(prefix.name + '_trial_summary.csv')}")
    print(f"wrote {prefix.with_name(prefix.name + '_proxy_comparison.csv')}")
    print(f"wrote {prefix.with_name(prefix.name + '_time_control.csv')}")
    print(f"wrote {prefix.with_name(prefix.name + '_lagged_correlation.csv')}")
    print(f"wrote {prefix.with_name(prefix.name + f'_{SELECTED_PROXY_SLUG}_vs_{SELECTED_IR_SLUG}.png')}")
    print(f"wrote {prefix.with_name(prefix.name + '_ir_feature_ranking.png')}")
    print(f"wrote {prefix.with_name(prefix.name + '_clear_normalized_relation.png')}")
    print(f"wrote {time_control_plot}")
    print(f"wrote {report_path}")
    print(
        "selected relation: "
        f"{SELECTED_PROXY} vs {SELECTED_IR}; "
        f"pooled normalized Pearson={selected['pooled_normalized_pearson']:.3f}; "
        f"median abs per-rep Pearson={selected['median_abs_per_rep_pearson']:.3f}"
    )
    print(
        "time control: "
        f"time-only R2={selected_time_control['time_only_r2']:.3f}; "
        f"time+IR R2={selected_time_control['time_plus_ir_r2']:.3f}; "
        f"delta R2={selected_time_control['delta_r2']:.3f}; "
        f"partial Pearson after time={selected_time_control['partial_pearson_after_time']:.3f}"
    )
    if best_lag is not None:
        print(
            "best lag scan: "
            f"lag_frames={best_lag['lag_frames']}; "
            f"pooled Pearson={best_lag['pooled_pearson']:.3f}"
        )


if __name__ == "__main__":
    main()
