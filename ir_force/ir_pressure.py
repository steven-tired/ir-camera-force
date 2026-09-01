from __future__ import annotations

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np

from .ir_hand_calibration import ProjectionCalibration
from .ir_hand_roi import PressureROI, select_pressure_roi, select_thermal_blob_roi


@dataclass(frozen=True)
class PressureConfig:
    near_contact_pinch: float = 0.045
    exit_contact_pinch: float = 0.055
    baseline_alpha: float = 0.08
    full_scale_delta: float = 24.0
    min_quality: float = 0.5
    max_frame_age_s: float | None = None
    min_frame_range: float = 2.0
    max_repeated_frames: int = 5
    # Provisional host-observation limits; these do not imply exposure synchronization.
    max_oak_age_s: float = 0.20
    max_thermal_age_s: float = 0.20
    max_pair_skew_s: float = 0.15
    # ROI strategy: "projection" (landmarks via calibration) or "blob" (hottest patch).
    roi_mode: str = "projection"
    blob_background_percentile: float = 25.0
    blob_min_delta_counts: float = 100.0
    blob_min_area_px: int = 4
    blob_max_area_px: int = 900
    blob_pad_px: int = 2
    # Expected (width, height) when no calibration carries it (blob mode).
    thermal_image_size: tuple[int, int] | None = None


def lepton_pressure_config(**overrides) -> PressureConfig:
    """PressureConfig defaults for the raw-count Lepton 3.x UDP path.

    max_thermal_age_s must exceed the Lepton's ~115 ms inter-frame gap (~8.7 Hz)
    or every reading degrades to thermal_stale. full_scale_delta is a provisional
    raw-count scale pending Phase 2/3 refit.
    """
    defaults = dict(
        roi_mode="blob",
        thermal_image_size=(160, 120),
        max_thermal_age_s=0.35,
        full_scale_delta=200.0,
    )
    defaults.update(overrides)
    return PressureConfig(**defaults)


@dataclass(frozen=True)
class PressureReading:
    pressure_0_1: float
    active: bool
    quality: float
    available: bool
    status: str
    roi: PressureROI | None = None
    oak_observed_at_s: float | None = None
    thermal_observed_at_s: float | None = None
    sensor_skew_s: float | None = None
    oak_age_s: float | None = None
    thermal_age_s: float | None = None
    roi_mode: str | None = None
    fresh: bool = True
    level: int | None = None
    n_levels: int | None = None
    pv_sequence: int | None = None
    pv_sent_at_s: float | None = None
    pv_received_at_s: float | None = None


def inactive_pressure(
    status: str,
    *,
    available: bool = True,
    quality: float = 0.0,
    roi: PressureROI | None = None,
    oak_observed_at_s: float | None = None,
    thermal_observed_at_s: float | None = None,
    sensor_skew_s: float | None = None,
    oak_age_s: float | None = None,
    thermal_age_s: float | None = None,
    roi_mode: str | None = None,
    fresh: bool = True,
) -> PressureReading:
    return PressureReading(
        pressure_0_1=0.0,
        active=False,
        quality=quality,
        available=available,
        status=status,
        roi=roi,
        oak_observed_at_s=oak_observed_at_s,
        thermal_observed_at_s=thermal_observed_at_s,
        sensor_skew_s=sensor_skew_s,
        oak_age_s=oak_age_s,
        thermal_age_s=thermal_age_s,
        roi_mode=roi_mode,
        fresh=fresh,
    )


def _finite_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def timing_limit_exceeded(later_s: float, earlier_s: float, limit_s: float) -> bool:
    """Compare timestamp differences with one ULP of operand-scale tolerance."""
    elapsed_s = later_s - earlier_s
    tolerance_s = max(math.ulp(later_s), math.ulp(earlier_s))
    return elapsed_s > limit_s + tolerance_s


def _scalar_thermal(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return frame.astype(np.float32)


class HandPressureEstimator:
    def __init__(
        self,
        *,
        calibration: ProjectionCalibration | None,
        thermal_source,
        config: PressureConfig | None = None,
    ):
        self.calibration = calibration
        self.thermal_source = thermal_source
        self.config = config or PressureConfig()
        if self.config.roi_mode not in ("projection", "blob"):
            raise ValueError(f"unknown roi_mode {self.config.roi_mode!r}")
        if self.config.roi_mode == "projection" and calibration is None:
            raise ValueError("projection roi_mode requires a ProjectionCalibration")
        self._baseline: np.ndarray | None = None
        self._baseline_roi: PressureROI | None = None
        self._active = False
        self._last_frame: np.ndarray | None = None
        self._last_frame_t: float | None = None
        self._repeated_frames = 0
        self._closed = False

    def _reset_pressure_state(self) -> None:
        self._active = False
        self._baseline = None
        self._baseline_roi = None

    def reset(self) -> None:
        self._reset_pressure_state()
        self._last_frame = None
        self._repeated_frames = 0

    def _update_baseline(self, crop: np.ndarray, roi: PressureROI) -> None:
        if self._baseline is None or self._baseline_roi != roi or self._baseline.shape != crop.shape:
            self._baseline = crop.astype(np.float32)
            self._baseline_roi = roi
            return
        alpha = self.config.baseline_alpha
        self._baseline = alpha * crop.astype(np.float32) + (1.0 - alpha) * self._baseline

    def update(self, landmarks, pinch: float, enabled: bool) -> PressureReading:
        try:
            sample = self.thermal_source.read()
        except Exception:
            self.reset()
            return inactive_pressure("thermal_unavailable", available=False)

        now_s = time.perf_counter()
        oak_t = _finite_float(getattr(landmarks, "observed_at_s", None))
        frame_t = _finite_float(getattr(sample, "t", None))
        oak_age_s = None if oak_t is None else now_s - oak_t
        thermal_age_s = None if frame_t is None else now_s - frame_t
        sensor_skew_s = (
            None if oak_t is None or frame_t is None else abs(oak_t - frame_t)
        )
        timing = {
            "oak_observed_at_s": oak_t,
            "thermal_observed_at_s": frame_t,
            "sensor_skew_s": sensor_skew_s,
            "oak_age_s": oak_age_s,
            "thermal_age_s": thermal_age_s,
        }

        if (
            oak_t is None
            or oak_age_s is None
            or oak_age_s < 0.0
            or timing_limit_exceeded(now_s, oak_t, self.config.max_oak_age_s)
        ):
            self.reset()
            return inactive_pressure("oak_stale", available=False, **timing)

        max_thermal_age_s = (
            self.config.max_thermal_age_s
            if self.config.max_frame_age_s is None
            else self.config.max_frame_age_s
        )
        if (
            frame_t is None
            or thermal_age_s is None
            or thermal_age_s < 0.0
            or timing_limit_exceeded(now_s, frame_t, max_thermal_age_s)
        ):
            self.reset()
            return inactive_pressure("thermal_stale", available=False, **timing)

        if self._last_frame_t is not None:
            if frame_t < self._last_frame_t:
                self.reset()
                return inactive_pressure("thermal_stale", available=False, **timing)
            if frame_t == self._last_frame_t:
                return inactive_pressure(
                    "thermal_pending",
                    available=True,
                    fresh=False,
                    **timing,
                )

        if sensor_skew_s is None or timing_limit_exceeded(
            max(oak_t, frame_t),
            min(oak_t, frame_t),
            self.config.max_pair_skew_s,
        ):
            self.reset()
            return inactive_pressure("sensor_skew", available=False, **timing)

        thermal = np.asarray(sample.frame)
        if thermal.size == 0:
            self.reset()
            return inactive_pressure("thermal_unavailable", available=False, **timing)
        expected_size = (
            self.calibration.image_size
            if self.calibration is not None
            else self.config.thermal_image_size
        )
        if expected_size is not None:
            expected_width, expected_height = expected_size
            if thermal.shape[:2] != (expected_height, expected_width):
                self.reset()
                return inactive_pressure("thermal_shape_mismatch", available=False, **timing)
        try:
            scalar = _scalar_thermal(thermal)
        except Exception:
            self.reset()
            return inactive_pressure("thermal_unavailable", available=False, **timing)
        if not np.all(np.isfinite(scalar)):
            self.reset()
            return inactive_pressure("thermal_unavailable", available=False, **timing)
        frame_range = float(np.percentile(scalar, 95.0) - np.percentile(scalar, 5.0))
        if frame_range < self.config.min_frame_range:
            self.reset()
            return inactive_pressure("thermal_low_contrast", available=False, **timing)

        if self._last_frame is not None and np.array_equal(thermal, self._last_frame):
            self._repeated_frames += 1
        else:
            self._repeated_frames = 0
        self._last_frame = thermal.copy()
        self._last_frame_t = frame_t
        if self._repeated_frames > self.config.max_repeated_frames:
            self._reset_pressure_state()
            return inactive_pressure("thermal_stale", available=False, **timing)

        if self.config.roi_mode == "blob":
            selection = select_thermal_blob_roi(
                scalar,
                background_percentile=self.config.blob_background_percentile,
                min_delta_counts=self.config.blob_min_delta_counts,
                min_area_px=self.config.blob_min_area_px,
                max_area_px=self.config.blob_max_area_px,
                pad_px=self.config.blob_pad_px,
            )
        else:
            selection = select_pressure_roi(landmarks, self.calibration, thermal.shape)
        if selection.roi is None:
            self._reset_pressure_state()
            return inactive_pressure(
                selection.mode,
                available=False,
                roi_mode=selection.mode,
                **timing,
            )
        if selection.quality < self.config.min_quality:
            self._reset_pressure_state()
            return PressureReading(
                pressure_0_1=0.0,
                active=False,
                quality=selection.quality,
                available=False,
                status="low_quality",
                roi=selection.roi,
                roi_mode=selection.mode,
                **timing,
            )

        if not enabled:
            self.reset()
            return PressureReading(
                0.0,
                False,
                selection.quality,
                True,
                "disabled",
                selection.roi,
                roi_mode=selection.mode,
                **timing,
            )

        crop = scalar[selection.roi.slices()]
        if not self._active and pinch <= self.config.near_contact_pinch:
            self._active = True
        elif self._active and pinch >= self.config.exit_contact_pinch:
            self._active = False

        if not self._active:
            if self.config.roi_mode != "blob":
                self._update_baseline(crop, selection.roi)
            return PressureReading(
                0.0,
                False,
                selection.quality,
                True,
                "baseline",
                selection.roi,
                roi_mode=selection.mode,
                **timing,
            )

        if self.config.roi_mode == "blob":
            # Provisional pre-classifier scalar: the blob ROI tracks a moving contact
            # patch, so a per-ROI EMA baseline never matches; use the frame-internal
            # background floor instead. Refit against Phase 2 trial data.
            background = float(
                np.percentile(scalar, self.config.blob_background_percentile)
            )
            delta = max(float(crop.mean()) - background, 0.0)
            pressure = float(np.clip(delta / self.config.full_scale_delta, 0.0, 1.0))
            return PressureReading(
                pressure,
                True,
                selection.quality,
                True,
                "active",
                selection.roi,
                roi_mode=selection.mode,
                **timing,
            )

        if self._baseline is None or self._baseline_roi != selection.roi or self._baseline.shape != crop.shape:
            return PressureReading(
                0.0,
                True,
                selection.quality,
                False,
                "active_no_baseline",
                selection.roi,
                roi_mode=selection.mode,
                **timing,
            )

        delta = np.maximum(crop.astype(np.float32) - self._baseline, 0.0)
        pressure = float(np.clip(float(delta.mean()) / self.config.full_scale_delta, 0.0, 1.0))
        return PressureReading(
            pressure,
            True,
            selection.quality,
            True,
            "active",
            selection.roi,
            roi_mode=selection.mode,
            **timing,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self.thermal_source, "close", None)
        if callable(close):
            close()
