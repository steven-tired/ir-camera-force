from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ThermalROI:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError("thermal ROI x and y must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("thermal ROI width and height must be positive")

    @property
    def x_end(self) -> int:
        return self.x + self.width

    @property
    def y_end(self) -> int:
        return self.y + self.height

    def slices(self) -> tuple[slice, slice]:
        return slice(self.y, self.y_end), slice(self.x, self.x_end)


@dataclass(frozen=True)
class BaselineStats:
    mean: np.ndarray
    noise: float
    roi: ThermalROI | None = None
    source_shape: tuple[int, int] | None = None


@dataclass(frozen=True)
class ReferencePatch:
    name: str
    roi: ThermalROI

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise ValueError("reference patch name must be non-empty")
        if not name.replace("_", "").isalnum():
            raise ValueError("reference patch name must contain only letters, numbers, or underscores")
        object.__setattr__(self, "name", name)


@dataclass(frozen=True)
class FrameFeatures:
    frame: str
    area_px: int
    mean_delta: float
    max_delta: float


def load_palette(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size % 3 != 0:
        raise ValueError(f"palette size must be divisible by 3: {path}")
    return raw.reshape((-1, 3))


@lru_cache(maxsize=8)
def _palette_tree(palette_bytes: bytes, color_count: int) -> cKDTree:
    colors = np.frombuffer(palette_bytes, dtype=np.uint8).reshape((color_count, 3)).astype(np.float32)
    return cKDTree(colors)


def palette_index_image(frame: np.ndarray, palette: np.ndarray, invert: bool = False) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError("palette index conversion requires a 3-channel frame")
    palette = np.ascontiguousarray(palette, dtype=np.uint8)
    pixels = frame.reshape((-1, 3)).astype(np.float32)
    _, nearest = _palette_tree(palette.tobytes(), len(palette)).query(pixels)
    indices = nearest.reshape(frame.shape[:2]).astype(np.float32)
    if invert:
        return float(len(palette) - 1) - indices
    return indices


def _thermal_proxy(
    frame: np.ndarray,
    palette: np.ndarray | None = None,
    invert_palette: bool = False,
) -> np.ndarray:
    if palette is not None:
        return palette_index_image(frame, palette, invert=invert_palette)
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return frame.astype(np.float32)


def _validate_roi(roi: ThermalROI, shape: tuple[int, int]) -> None:
    height, width = shape
    if roi.x_end > width or roi.y_end > height:
        raise ValueError(f"thermal ROI {roi} is outside frame shape {width}x{height}")


def _crop_to_roi(scalar: np.ndarray, roi: ThermalROI) -> np.ndarray:
    source_shape = scalar.shape[:2]
    _validate_roi(roi, source_shape)
    return scalar[roi.slices()]


def compute_baseline(
    frames: list[np.ndarray],
    palette: np.ndarray | None = None,
    invert_palette: bool = False,
    roi: ThermalROI | None = None,
) -> BaselineStats:
    if not frames:
        raise ValueError("baseline requires at least one frame")
    source_shape = frames[0].shape[:2]
    scalars = []
    for frame in frames:
        scalar = _thermal_proxy(frame, palette=palette, invert_palette=invert_palette)
        if scalar.shape[:2] != source_shape:
            raise ValueError("baseline frames must have the same shape")
        if roi is not None:
            scalar = _crop_to_roi(scalar, roi)
        scalars.append(scalar)
    stack = np.stack(
        scalars,
        axis=0,
    )
    return BaselineStats(
        mean=stack.mean(axis=0),
        noise=float(stack.std(axis=0).mean()),
        roi=roi,
        source_shape=source_shape if roi is not None else None,
    )


def extract_frame_features(
    frame: np.ndarray,
    baseline: BaselineStats,
    noise_sigma: float = 3.0,
    frame_name: str = "",
    palette: np.ndarray | None = None,
    invert_palette: bool = False,
) -> tuple[FrameFeatures, np.ndarray]:
    scalar = _thermal_proxy(frame, palette=palette, invert_palette=invert_palette)
    if baseline.roi is not None:
        if baseline.source_shape is None:
            raise ValueError("ROI baseline is missing its source frame shape")
        if scalar.shape[:2] != baseline.source_shape:
            raise ValueError(
                f"frame shape {scalar.shape[1]}x{scalar.shape[0]} does not match ROI baseline "
                f"{baseline.source_shape[1]}x{baseline.source_shape[0]}"
            )
        delta = _crop_to_roi(scalar, baseline.roi) - baseline.mean
    else:
        delta = scalar - baseline.mean
    threshold = max(baseline.noise * noise_sigma, 1.0)
    local_mask = delta > threshold
    selected = delta[local_mask]
    features = FrameFeatures(
        frame=frame_name,
        area_px=int(local_mask.sum()),
        mean_delta=float(selected.mean()) if selected.size else 0.0,
        max_delta=float(selected.max()) if selected.size else 0.0,
    )
    if baseline.roi is None:
        return features, local_mask.astype(np.uint8)

    mask = np.zeros(baseline.source_shape, dtype=np.uint8)
    mask[baseline.roi.slices()] = local_mask.astype(np.uint8)
    return features, mask


def _round_feature(value: float) -> float:
    return round(float(value), 6)


def extract_classifier_frame_features(
    frame: np.ndarray,
    baseline: BaselineStats,
    *,
    frame_id: int,
    timestamp: float,
    reference_patches: tuple[ReferencePatch, ...] = (),
    previous_frame_p98: float | None = None,
    previous_scalar: np.ndarray | None = None,
    agc_jump_threshold: float = 8.0,
    noise_sigma: float = 3.0,
    palette: np.ndarray | None = None,
    invert_palette: bool = False,
) -> dict[str, object]:
    scalar = _thermal_proxy(frame, palette=palette, invert_palette=invert_palette)
    if baseline.roi is not None:
        if baseline.source_shape is None:
            raise ValueError("ROI baseline is missing its source frame shape")
        if scalar.shape[:2] != baseline.source_shape:
            raise ValueError(
                f"frame shape {scalar.shape[1]}x{scalar.shape[0]} does not match ROI baseline "
                f"{baseline.source_shape[1]}x{baseline.source_shape[0]}"
            )
        roi_scalar = _crop_to_roi(scalar, baseline.roi)
    else:
        roi_scalar = scalar

    delta = roi_scalar - baseline.mean
    threshold = max(baseline.noise * noise_sigma, 1.0)
    positive = delta >= threshold
    negative = delta <= -threshold
    frame_p2 = float(np.percentile(scalar, 2))
    frame_p98 = float(np.percentile(scalar, 98))

    result: dict[str, object] = {
        "timestamp": _round_feature(timestamp),
        "frame_id": frame_id,
        "roi_mean": _round_feature(roi_scalar.mean()),
        "roi_std": _round_feature(roi_scalar.std()),
        "delta_mean": _round_feature(delta.mean()),
        "delta_std": _round_feature(delta.std()),
        "pos_area": int(positive.sum()),
        "neg_area": int(negative.sum()),
        "l1_delta": _round_feature(np.abs(delta).sum()),
        "l2_delta": _round_feature(np.sqrt(np.square(delta).sum())),
        "delta_p90": _round_feature(np.percentile(delta, 90)),
        "delta_p95": _round_feature(np.percentile(delta, 95)),
        "delta_p99": _round_feature(np.percentile(delta, 99)),
        "frame_p2": _round_feature(frame_p2),
        "frame_p98": _round_feature(frame_p98),
        "frozen_frame_flag": bool(
            previous_scalar is not None
            and previous_scalar.shape == scalar.shape
            and np.array_equal(previous_scalar, scalar)
        ),
        "agc_jump_flag": bool(
            previous_frame_p98 is not None and abs(frame_p98 - previous_frame_p98) >= agc_jump_threshold
        ),
    }

    for patch in reference_patches:
        patch_scalar = _crop_to_roi(scalar, patch.roi)
        result[f"reference_{patch.name}_mean"] = _round_feature(patch_scalar.mean())
        result[f"reference_{patch.name}_std"] = _round_feature(patch_scalar.std())

    return result


def overlay_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        out = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        out = frame.copy()
    out[mask.astype(bool)] = (0, 0, 255)
    return out


def write_features_csv(features: list[FrameFeatures], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "area_px", "mean_delta", "max_delta"])
        writer.writeheader()
        for item in features:
            writer.writerow(
                {
                    "frame": item.frame,
                    "area_px": item.area_px,
                    "mean_delta": item.mean_delta,
                    "max_delta": item.max_delta,
                }
            )
