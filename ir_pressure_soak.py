"""Robot-free IR pressure shadow soak core."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np


_CHECKOUT_ROOT = Path(__file__).resolve().parent
_CHECKOUT_ROOT_TEXT = str(_CHECKOUT_ROOT)
if not sys.path or sys.path[0] != _CHECKOUT_ROOT_TEXT:
    sys.path.insert(0, _CHECKOUT_ROOT_TEXT)

import webcam_input
from webcam_input.depth import ScaleDepthStrategy
from webcam_input.webcam_source import WebcamSource
from webcam_input.wrist_estimator import WebcamWristEstimator


# Before the repository split, `webcam_input` lived in this checkout and this
# guard asserted exactly that. It now comes from the public mediapipe-so101
# package, so what still matters is that it is NOT a stale copy left behind in
# the old meta-workspace -- loading that would silently run pre-split code.
_LOADED_WEBCAM_PACKAGE = Path(webcam_input.__file__).resolve().parent
if "hand-teleop" in _LOADED_WEBCAM_PACKAGE.parts and "mediapipe-so101" not in _LOADED_WEBCAM_PACKAGE.parts:
    raise ImportError(
        "ir_pressure_soak loaded webcam_input from the pre-split workspace: "
        f"{_LOADED_WEBCAM_PACKAGE}. Install the public package "
        "(mediapipe-so101/packages/webcam_input) instead."
    )


_ROBOT_FREE_IMPORT_ENV = "LEROBOT_TELEOPERATOR_SO101_WEBCAM_ROBOT_FREE_IMPORT"
_ROBOT_FREE_IMPORT_WAS_SET = _ROBOT_FREE_IMPORT_ENV in os.environ
_ROBOT_FREE_IMPORT_PREVIOUS = os.environ.get(_ROBOT_FREE_IMPORT_ENV)
os.environ[_ROBOT_FREE_IMPORT_ENV] = "1"
try:
    from ir_force.ir_hand_calibration import (
        load_projection_calibration,
        validate_projection_calibration,
    )
    from ir_force.ir_pressure import (
        HandPressureEstimator,
        PressureConfig,
        inactive_pressure,
        timing_limit_exceeded,
    )
    from lerobot_teleoperator_so101_webcam.grip.proposal import (
        GRIP_CLOSE_ALPHA,
        GRIP_OPEN_ALPHA,
        GRIP_OVERDRIVE,
        PressureProposalStateMachine,
        apply_pressure_overdrive,
    )
    from ir_force.ir_shadow_telemetry import (
        IRShadowTelemetryLogger,
        IRShadowTelemetrySample,
    )
finally:
    if _ROBOT_FREE_IMPORT_WAS_SET:
        os.environ[_ROBOT_FREE_IMPORT_ENV] = _ROBOT_FREE_IMPORT_PREVIOUS
    else:
        os.environ.pop(_ROBOT_FREE_IMPORT_ENV, None)


DEFAULT_CALIBRATION = str(
    Path(__file__).resolve().parent
    / "calibration"
    / "oak_flir_hand_pressure_projection.json"
)
DEFAULT_THERMAL = "/dev/video21"
DEFAULT_DURATION_S = 1800.0
DEFAULT_MAX_OAK_STALL_MS = 500.0
DEFAULT_MIN_CYCLES = 100
DEFAULT_POLL_INTERVAL_S = 0.01

GRIP_PINCH_MIN = 0.02
GRIP_PINCH_MAX = 0.12
MIDDLE_GRIPPER = 50.0


@dataclass(frozen=True)
class FrameSample:
    t: float
    frame: np.ndarray


@dataclass(frozen=True)
class ThermalPublicationState:
    generation: int
    observed_at_s: float | None
    error: Exception | None
    source_started_at_s: float


@dataclass(frozen=True)
class ThermalPublicationClaim:
    generation: int
    observed_at_s: float | None
    error: Exception | None
    source_started_at_s: float
    sample: object | None


class OpenCVCameraSource:
    """Minimal FLIR source kept independent of robot telemetry modules."""

    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"could not open camera {path}")

    def read(self) -> FrameSample:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"could not read camera {self.path}")
        return FrameSample(t=time.perf_counter(), frame=frame)

    def close(self) -> None:
        self.cap.release()


class LatestFrameSource:
    """Publish the newest FLIR sample without blocking the soak loop."""

    _CLOSE_TIMEOUT_S = 1.0

    def __init__(self, source):
        self.source = source
        self._lock = threading.Lock()
        self._latest = None
        self._error: Exception | None = None
        self._generation = 0
        self._source_started_at_s = time.perf_counter()
        self._running = True
        self._closed = False
        self._source_close_called = False
        self._source_close_error: Exception | None = None
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()

    def _produce(self) -> None:
        while True:
            with self._lock:
                if not self._running:
                    return
            try:
                sample = self.source.read()
            except Exception as exc:
                with self._lock:
                    if self._running:
                        self._error = exc
                        self._generation += 1
                return
            with self._lock:
                if not self._running:
                    return
                self._latest = sample
                self._error = None
                self._generation += 1

    def publication_state(self) -> ThermalPublicationState:
        with self._lock:
            observed_at_s = (
                None if self._latest is None else getattr(self._latest, "t", None)
            )
            return ThermalPublicationState(
                generation=self._generation,
                observed_at_s=observed_at_s,
                error=self._error,
                source_started_at_s=self._source_started_at_s,
            )

    def claim_publication(self) -> ThermalPublicationClaim:
        with self._lock:
            sample = self._latest
            return ThermalPublicationClaim(
                generation=self._generation,
                observed_at_s=None if sample is None else getattr(sample, "t", None),
                error=self._error,
                source_started_at_s=self._source_started_at_s,
                sample=sample,
            )

    def read(self) -> FrameSample:
        with self._lock:
            if not self._running:
                raise RuntimeError("latest frame source is closed")
            if self._error is not None:
                raise self._error
            if self._latest is None:
                raise RuntimeError("latest frame unavailable before first producer sample")
            return self._latest

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._running = False
            close_source = not self._source_close_called
            self._source_close_called = True

        if close_source:
            try:
                self.source.close()
            except Exception as exc:
                self._source_close_error = exc

        self._thread.join(timeout=self._CLOSE_TIMEOUT_S)
        if self._thread.is_alive():
            raise RuntimeError(
                "latest frame producer thread did not terminate within "
                f"{self._CLOSE_TIMEOUT_S:.1f}s"
            ) from self._source_close_error
        if self._source_close_error is not None:
            raise self._source_close_error

        with self._lock:
            self._closed = True


class ClaimedFrameSource:
    """One-shot source that owns the latest-frame source it snapshots."""

    def __init__(self, source):
        self.source = source
        self._claim: ThermalPublicationClaim | None = None

    def set_claim(self, claim: ThermalPublicationClaim) -> None:
        self._claim = claim

    def read(self):
        claim = self._claim
        self._claim = None
        if claim is None:
            raise RuntimeError("no thermal publication has been claimed")
        if claim.error is not None:
            raise claim.error
        if claim.sample is None:
            raise RuntimeError("claimed thermal publication has no sample")
        return claim.sample

    def close(self) -> None:
        self.source.close()


def _nonnegative_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def _nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", required=True, help="Required shadow telemetry CSV path.")
    parser.add_argument("--duration-s", type=_nonnegative_float, default=DEFAULT_DURATION_S)
    parser.add_argument(
        "--max-oak-stall-ms",
        type=_positive_float,
        default=DEFAULT_MAX_OAK_STALL_MS,
    )
    parser.add_argument("--min-cycles", type=_nonnegative_int, default=DEFAULT_MIN_CYCLES)
    parser.add_argument("--thermal", default=DEFAULT_THERMAL)
    parser.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    return parser.parse_args(argv)


@dataclass
class PinchCycleCounter:
    """Count complete open -> closed -> open transitions with hysteresis."""

    config: PressureConfig = field(default_factory=PressureConfig)
    phase: str = "await_open"
    cycles: int = 0

    def observe(self, pinch: float) -> int:
        if self.phase == "await_open":
            if pinch >= self.config.exit_contact_pinch:
                self.phase = "await_close"
        elif self.phase == "await_close":
            if pinch <= self.config.near_contact_pinch:
                self.phase = "await_reopen"
        elif pinch >= self.config.exit_contact_pinch:
            self.cycles += 1
            self.phase = "await_close"
        return self.cycles


@dataclass
class LegacyActualGripper:
    """Software-only legacy fixed-overdrive and asymmetric actual EMA."""

    smoothed: float | None = None

    def current(self, default: float) -> float:
        return float(default if self.smoothed is None else self.smoothed)

    def reset(self) -> None:
        self.smoothed = None

    def update(self, base_gripper: float) -> float:
        raw = apply_pressure_overdrive(base_gripper, GRIP_OVERDRIVE, None)
        if self.smoothed is None:
            self.smoothed = float(raw)
        else:
            alpha = GRIP_CLOSE_ALPHA if raw < self.smoothed else GRIP_OPEN_ALPHA
            self.smoothed = alpha * raw + (1.0 - alpha) * self.smoothed
        return float(self.smoothed)


@dataclass(frozen=True)
class SoakSummary:
    exit_code: int
    reason: str
    ticks: int
    cycles: int
    state_counts: dict[str, int]
    status_counts: dict[str, int]
    rejection_counts: dict[str, int]
    fault_closure_violations: int
    sidecar: str
    metrics: dict[str, object] = field(default_factory=dict)
    cleanup_failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeFactories:
    load_calibration: Callable[..., object] = load_projection_calibration
    validate_calibration: Callable[..., object] = validate_projection_calibration
    logger_factory: Callable[..., object] = IRShadowTelemetryLogger
    scale_depth_factory: Callable[..., object] = ScaleDepthStrategy
    wrist_estimator_factory: Callable[..., object] = WebcamWristEstimator
    webcam_source_factory: Callable[..., object] = WebcamSource
    thermal_source_factory: Callable[..., object] = OpenCVCameraSource
    latest_frame_source_factory: Callable[..., object] = LatestFrameSource
    pressure_estimator_factory: Callable[..., object] = HandPressureEstimator


class RuntimeSetupError(RuntimeError):
    def __init__(self, stage: str, cause: Exception):
        super().__init__(f"{stage} failed: {cause}")
        self.stage = stage
        self.cause = cause

    @property
    def reason(self) -> str:
        return f"setup_error:{self.stage}:{type(self.cause).__name__}"


class SetupFailure(RuntimeError):
    def __init__(self, exit_code: int, reason: str):
        super().__init__(reason)
        self.exit_code = exit_code
        self.reason = reason


@dataclass
class SoakRuntime:
    source: object | None = None
    estimator: object | None = None
    logger: object | None = None
    _thermal_owner: object | None = None
    _thermal_owner_label: str | None = None
    _cleaned: bool = False
    _cleanup_failures: tuple[str, ...] = ()

    def set_thermal_owner(self, owner, label: str) -> None:
        self._thermal_owner = owner
        self._thermal_owner_label = label

    def cleanup(self) -> tuple[str, ...]:
        if self._cleaned:
            return self._cleanup_failures
        self._cleaned = True
        failures = []

        def attempt(label: str, cleanup: Callable[[], None]) -> None:
            try:
                cleanup()
            except Exception as exc:
                failures.append(f"{label}:{type(exc).__name__}:{exc}")

        if self._thermal_owner is not None:
            attempt(str(self._thermal_owner_label), self._thermal_owner.close)
        if self.source is not None:
            attempt("oak_source.stop", self.source.stop)
        if self.logger is not None:
            attempt("sidecar_logger.close", self.logger.close)

        self._cleanup_failures = tuple(failures)
        return self._cleanup_failures


_METRIC_NAMES = (
    "oak_age_ms",
    "thermal_age_ms",
    "pair_skew_ms",
    "loop_period_ms",
    "control_latency_ms",
)

_EXIT_OAK_FAILED = 10
_EXIT_OAK_WATCHDOG = 11
_EXIT_SOURCE_ERROR = 12
_EXIT_SIDECAR_DISABLED = 14
_EXIT_FAULT_CLOSURE = 15
_EXIT_INSUFFICIENT_CYCLES = 16
_EXIT_ESTIMATOR_ERROR = 17
_EXIT_PROPOSAL_ERROR = 18
_EXIT_SIDECAR_ERROR = 19
_EXIT_PRESSURE_FAULT_LATCHED = 20
_EXIT_SETUP_ERROR = 21
_EXIT_RUNTIME_ERROR = 22
_EXIT_CLEANUP_ERROR = 23


def _setup_call(stage: str, factory, *args, **kwargs):
    try:
        return factory(*args, **kwargs)
    except Exception as exc:
        raise RuntimeSetupError(stage, exc) from exc


def build_runtime(
    args,
    *,
    factories: RuntimeFactories | None = None,
    runtime: SoakRuntime | None = None,
) -> SoakRuntime:
    """Construct the robot-free Gate 1 sources in fail-closed order."""
    factories = factories or RuntimeFactories()
    runtime = runtime or SoakRuntime()

    calibration = _setup_call(
        "load_calibration",
        factories.load_calibration,
        Path(args.calibration),
    )
    calibration = _setup_call(
        "validate_calibration",
        factories.validate_calibration,
        calibration,
        min_samples=12,
        max_rms_error_px=8.0,
        max_error_px=16.0,
        expected_image_size=(160, 128),
    )

    runtime.logger = _setup_call(
        "sidecar_logger",
        factories.logger_factory,
        args.sidecar,
    )
    try:
        sidecar_enabled = bool(runtime.logger.enabled)
    except Exception as exc:
        raise RuntimeSetupError("sidecar_enabled", exc) from exc
    if not sidecar_enabled:
        raise SetupFailure(_EXIT_SIDECAR_DISABLED, "sidecar_disabled")

    depth = _setup_call("scale_depth", factories.scale_depth_factory)
    wrist_estimator = _setup_call(
        "wrist_estimator",
        factories.wrist_estimator_factory,
        depth,
    )
    runtime.source = _setup_call(
        "webcam_source",
        factories.webcam_source_factory,
        wrist_estimator,
    )
    _setup_call("start_oak", runtime.source.start_oak)

    thermal = _setup_call(
        "thermal_source",
        factories.thermal_source_factory,
        args.thermal,
    )
    runtime.set_thermal_owner(thermal, "thermal_source.close")
    latest = _setup_call(
        "latest_frame_source",
        factories.latest_frame_source_factory,
        thermal,
    )
    runtime.set_thermal_owner(latest, "latest_frame_source.close")
    inner_estimator = _setup_call(
        "pressure_estimator",
        factories.pressure_estimator_factory,
        calibration=calibration,
        thermal_source=latest,
    )
    runtime.estimator = inner_estimator
    runtime.set_thermal_owner(inner_estimator, "pressure_estimator.close")
    if callable(getattr(latest, "claim_publication", None)):
        runtime.estimator = _setup_call(
            "publication_pressure_estimator",
            PublicationGatedPressureEstimator,
            inner_estimator,
            latest,
        )
    runtime.set_thermal_owner(runtime.estimator, "pressure_estimator.close")
    return runtime


def _finite_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _metric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
    percentiles = np.percentile(np.asarray(values, dtype=float), [50, 95, 99])
    return {
        "count": len(values),
        "p50": float(percentiles[0]),
        "p95": float(percentiles[1]),
        "p99": float(percentiles[2]),
        "max": float(max(values)),
    }


def _summarize_metrics(values: dict[str, list[float]]) -> dict[str, object]:
    return {name: _metric_summary(values[name]) for name in _METRIC_NAMES}


def _append_milliseconds(values: list[float], seconds) -> None:
    finite_seconds = _finite_float(seconds)
    if finite_seconds is not None:
        values.append(finite_seconds * 1000.0)


def _error_reason(category: str, exc: Exception) -> str:
    return f"{category}:{type(exc).__name__}"


class PublicationGatedPressureEstimator:
    """Process each FLIR publication once and report producer stalls once."""

    def __init__(self, estimator, thermal_source, *, clock=time.perf_counter):
        self.estimator = estimator
        self.thermal_source = thermal_source
        self.clock = clock
        self.config = estimator.config
        self.claimed_source = ClaimedFrameSource(thermal_source)
        self.estimator.thermal_source = self.claimed_source
        self._last_generation = 0
        self._reported_fault_generation: int | None = None

    def _update_claim(self, claim, landmarks, *, pinch: float, enabled: bool):
        self.claimed_source.set_claim(claim)
        return self.estimator.update(
            landmarks,
            pinch=pinch,
            enabled=enabled,
        )

    def _health_pressure(self, claim, status: str, oak_observed_at_s):
        now_s = self.clock()
        oak_observed_at_s = _finite_float(oak_observed_at_s)
        thermal_observed_at_s = _finite_float(claim.observed_at_s)
        return inactive_pressure(
            status,
            available=False,
            oak_observed_at_s=oak_observed_at_s,
            thermal_observed_at_s=thermal_observed_at_s,
            sensor_skew_s=(
                None
                if oak_observed_at_s is None or thermal_observed_at_s is None
                else abs(oak_observed_at_s - thermal_observed_at_s)
            ),
            oak_age_s=(
                None
                if oak_observed_at_s is None
                else now_s - oak_observed_at_s
            ),
            thermal_age_s=(
                None
                if thermal_observed_at_s is None
                else now_s - thermal_observed_at_s
            ),
        )

    def monitor_health(self, *, discard_healthy: bool, oak_observed_at_s=None):
        """Acknowledge health independently from pressure/ROI estimation."""
        claim = self.thermal_source.claim_publication()
        now_s = self.clock()
        thermal_observed_at_s = _finite_float(claim.observed_at_s)
        reference_s = thermal_observed_at_s
        if reference_s is None:
            reference_s = _finite_float(claim.source_started_at_s)
        max_thermal_age_s = (
            self.config.max_thermal_age_s
            if self.config.max_frame_age_s is None
            else self.config.max_frame_age_s
        )
        stale = (
            reference_s is not None
            and timing_limit_exceeded(now_s, reference_s, max_thermal_age_s)
        )
        status = None
        if claim.error is not None:
            status = "thermal_unavailable"
        elif stale:
            status = (
                "thermal_stale"
                if thermal_observed_at_s is not None
                else "thermal_unavailable"
            )

        if status is not None:
            if self._reported_fault_generation == claim.generation:
                return None
            self._last_generation = claim.generation
            self._reported_fault_generation = claim.generation
            self.estimator.reset()
            return self._health_pressure(
                claim,
                status,
                oak_observed_at_s,
            )

        if discard_healthy and claim.generation != self._last_generation:
            self._last_generation = claim.generation
            self._reported_fault_generation = None
        return None

    def update_if_ready(self, landmarks, *, pinch: float, enabled: bool):
        claim = self.thermal_source.claim_publication()
        if claim.generation != self._last_generation:
            self._last_generation = claim.generation
            self._reported_fault_generation = None
            pressure = self._update_claim(
                claim,
                landmarks,
                pinch=pinch,
                enabled=enabled,
            )
            status = str(getattr(pressure, "status", ""))
            if claim.error is not None or status in {
                "thermal_stale",
                "thermal_unavailable",
            }:
                self._reported_fault_generation = claim.generation
            return True, pressure

        max_thermal_age_s = (
            self.config.max_thermal_age_s
            if self.config.max_frame_age_s is None
            else self.config.max_frame_age_s
        )
        thermal_observed_at_s = _finite_float(claim.observed_at_s)
        reference_s = thermal_observed_at_s
        if reference_s is None:
            reference_s = _finite_float(claim.source_started_at_s)
        now_s = self.clock()
        if (
            reference_s is None
            or not timing_limit_exceeded(now_s, reference_s, max_thermal_age_s)
            or self._reported_fault_generation == claim.generation
        ):
            return False, None

        self._reported_fault_generation = claim.generation
        pressure = self._update_claim(
            claim,
            landmarks,
            pinch=pinch,
            enabled=enabled,
        )
        return True, pressure

    def reset(self) -> None:
        self.estimator.reset()

    def close(self) -> None:
        self.estimator.close()


def _pinch_distance(landmarks) -> float:
    points = np.asarray(landmarks.landmarks, dtype=float)
    return float(np.linalg.norm(points[4] - points[8]))


def _base_gripper_from_pinch(pinch: float) -> float:
    span = GRIP_PINCH_MAX - GRIP_PINCH_MIN
    return float(np.clip((pinch - GRIP_PINCH_MIN) / span * 100.0, 0.0, 100.0))


def run_soak(
    *,
    source,
    estimator,
    logger,
    duration_s: float,
    min_cycles: int,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    max_oak_stall_s: float = DEFAULT_MAX_OAK_STALL_MS / 1000.0,
    pressure_config: PressureConfig | None = None,
    progress: Callable[[dict[str, object]], None] | None = None,
    progress_interval_s: float = 10.0,
) -> SoakSummary:
    """Run the hardware-independent polling core against injected dependencies."""
    config = pressure_config or PressureConfig()
    cycles = PinchCycleCounter(config=config)
    proposal = PressureProposalStateMachine(initial_gripper=MIDDLE_GRIPPER)
    legacy_actual = LegacyActualGripper()
    states: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    rejections: Counter[str] = Counter()
    fault_closure_violations = 0
    pressure_recovery_required = False
    previous_proposal: float | None = None
    previous_control_observed_at_s: float | None = None
    metric_values = {name: [] for name in _METRIC_NAMES}
    ticks = 0
    terminal_exit_code: int | None = None
    terminal_reason: str | None = None

    try:
        sidecar = str(getattr(logger, "path", ""))
    except Exception as exc:
        sidecar = ""
        terminal_exit_code = _EXIT_SIDECAR_ERROR
        terminal_reason = _error_reason("sidecar_error", exc)

    def summary(exit_code: int, reason: str) -> SoakSummary:
        return SoakSummary(
            exit_code=exit_code,
            reason=reason,
            ticks=ticks,
            cycles=cycles.cycles,
            state_counts=dict(sorted(states.items())),
            status_counts=dict(sorted(statuses.items())),
            rejection_counts=dict(sorted(rejections.items())),
            fault_closure_violations=fault_closure_violations,
            sidecar=sidecar,
            metrics=_summarize_metrics(metric_values),
        )

    if terminal_reason is not None:
        return summary(terminal_exit_code, terminal_reason)

    try:
        sidecar_enabled = bool(getattr(logger, "enabled", False))
    except Exception as exc:
        return summary(
            _EXIT_SIDECAR_ERROR,
            _error_reason("sidecar_error", exc),
        )
    if not sidecar_enabled:
        return summary(_EXIT_SIDECAR_DISABLED, "sidecar_disabled")

    started_at_s = clock()
    last_publication_id = None
    last_publication_observed_at_s: float | None = None
    next_progress_elapsed_s = float(progress_interval_s)

    while True:
        try:
            sidecar_enabled = bool(getattr(logger, "enabled", False))
        except Exception as exc:
            terminal_exit_code = _EXIT_SIDECAR_ERROR
            terminal_reason = _error_reason("sidecar_error", exc)
            break
        if not sidecar_enabled:
            terminal_exit_code = _EXIT_SIDECAR_DISABLED
            terminal_reason = "sidecar_disabled"
            break

        try:
            oak_failed = bool(getattr(source, "oak_failed", False))
        except Exception as exc:
            terminal_exit_code = _EXIT_SOURCE_ERROR
            terminal_reason = _error_reason("source_error", exc)
            break
        if oak_failed:
            terminal_exit_code = _EXIT_OAK_FAILED
            terminal_reason = "oak_failed"
            break

        control_observed_at_s = clock()
        duration_elapsed = control_observed_at_s - started_at_s >= duration_s

        try:
            sample = source.latest_sample()
            landmarks = sample.landmarks
            wrist = sample.wrist
            frame_id = getattr(
                sample,
                "frame_id",
                getattr(landmarks, "frame_id", None),
            )
            observed_at_s = _finite_float(
                getattr(
                    sample,
                    "observed_at_s",
                    getattr(landmarks, "observed_at_s", None),
                )
            )
        except Exception as exc:
            terminal_exit_code = _EXIT_SOURCE_ERROR
            terminal_reason = _error_reason("source_error", exc)
            break

        if frame_id is None:
            if timing_limit_exceeded(
                control_observed_at_s,
                started_at_s,
                max_oak_stall_s,
            ):
                terminal_exit_code = _EXIT_OAK_WATCHDOG
                terminal_reason = "oak_no_first_frame_stall"
                break
            if duration_elapsed:
                break
            sleep(poll_interval_s)
            continue
        if observed_at_s is None:
            terminal_exit_code = _EXIT_SOURCE_ERROR
            terminal_reason = "oak_observed_timestamp_invalid"
            break

        is_new_publication = last_publication_id is None
        if last_publication_id is not None:
            try:
                is_new_publication = frame_id > last_publication_id
            except Exception as exc:
                terminal_exit_code = _EXIT_SOURCE_ERROR
                terminal_reason = _error_reason("source_error", exc)
                break

        if is_new_publication:
            if last_publication_observed_at_s is not None:
                if observed_at_s < last_publication_observed_at_s:
                    terminal_exit_code = _EXIT_SOURCE_ERROR
                    terminal_reason = "oak_observed_timestamp_regression"
                    break
                if timing_limit_exceeded(
                    observed_at_s,
                    last_publication_observed_at_s,
                    max_oak_stall_s,
                ):
                    terminal_exit_code = _EXIT_OAK_WATCHDOG
                    terminal_reason = "oak_observed_timestamp_gap"
                    break
            last_publication_id = frame_id
            last_publication_observed_at_s = observed_at_s
        if (
            last_publication_observed_at_s is not None
            and timing_limit_exceeded(
                control_observed_at_s,
                last_publication_observed_at_s,
                max_oak_stall_s,
            )
        ):
            terminal_exit_code = _EXIT_OAK_WATCHDOG
            terminal_reason = "oak_publication_stall"
            break

        hand_valid = bool(
            getattr(wrist, "valid", False) and getattr(landmarks, "valid", False)
        )
        if hand_valid:
            try:
                pinch = _pinch_distance(landmarks)
            except Exception as exc:
                terminal_exit_code = _EXIT_SOURCE_ERROR
                terminal_reason = _error_reason("source_error", exc)
                break
        else:
            pinch = 0.0
        base_gripper = _base_gripper_from_pinch(pinch)
        pressure = None
        fist_state = getattr(wrist, "fist_state", None)
        confirmed_closed = isinstance(fist_state, str) and fist_state == "closed"
        confirmed_open = isinstance(fist_state, str) and fist_state == "open"
        unknown_clutch = hand_valid and not (confirmed_closed or confirmed_open)
        if not hand_valid or unknown_clutch:
            state = "HOLD"
        elif confirmed_closed:
            state = "MIDDLE"
        else:
            state = "MOVING"

        health_pressure = None
        monitor_health = getattr(estimator, "monitor_health", None)
        if callable(monitor_health):
            try:
                health_pressure = monitor_health(
                    discard_healthy=state != "MOVING",
                    oak_observed_at_s=observed_at_s,
                )
            except Exception as exc:
                terminal_exit_code = _EXIT_ESTIMATOR_ERROR
                terminal_reason = _error_reason("estimator_error", exc)
                break
        if duration_elapsed and health_pressure is None:
            break
        pressure = health_pressure

        if state == "HOLD":
            hold_reason = "clutch_unknown" if unknown_clutch else "hold"
            try:
                estimator.reset()
            except Exception as exc:
                terminal_exit_code = _EXIT_ESTIMATOR_ERROR
                terminal_reason = _error_reason("estimator_error", exc)
                break
            try:
                decision = proposal.reset(
                    base_gripper,
                    transition="hold",
                    middle_gripper=MIDDLE_GRIPPER,
                    reason=hold_reason,
                )
            except Exception as exc:
                terminal_exit_code = _EXIT_PROPOSAL_ERROR
                terminal_reason = _error_reason("proposal_error", exc)
                break
            if pressure is not None or pressure_recovery_required:
                try:
                    safe_proposal = decision.proposed_gripper
                    if previous_proposal is not None:
                        safe_proposal = max(safe_proposal, previous_proposal)
                    proposal.seed(safe_proposal)
                    decision = proposal.update(
                        safe_proposal,
                        pressure
                        if pressure is not None
                        else inactive_pressure("fault_latched", available=False),
                    )
                except Exception as exc:
                    terminal_exit_code = _EXIT_PROPOSAL_ERROR
                    terminal_reason = _error_reason("proposal_error", exc)
                    break
            actual_gripper = legacy_actual.current(MIDDLE_GRIPPER)
        elif state == "MIDDLE":
            try:
                estimator.reset()
            except Exception as exc:
                terminal_exit_code = _EXIT_ESTIMATOR_ERROR
                terminal_reason = _error_reason("estimator_error", exc)
                break
            try:
                decision = proposal.reset(
                    base_gripper,
                    transition="middle",
                    middle_gripper=MIDDLE_GRIPPER,
                )
            except Exception as exc:
                terminal_exit_code = _EXIT_PROPOSAL_ERROR
                terminal_reason = _error_reason("proposal_error", exc)
                break
            if pressure is not None or pressure_recovery_required:
                try:
                    safe_proposal = decision.proposed_gripper
                    if previous_proposal is not None:
                        safe_proposal = max(safe_proposal, previous_proposal)
                    proposal.seed(safe_proposal)
                    decision = proposal.update(
                        safe_proposal,
                        pressure
                        if pressure is not None
                        else inactive_pressure("fault_latched", available=False),
                    )
                except Exception as exc:
                    terminal_exit_code = _EXIT_PROPOSAL_ERROR
                    terminal_reason = _error_reason("proposal_error", exc)
                    break
            legacy_actual.reset()
            actual_gripper = MIDDLE_GRIPPER
        elif state == "MOVING":
            if is_new_publication:
                cycles.observe(pinch)
            if pressure is None:
                try:
                    update_if_ready = getattr(estimator, "update_if_ready", None)
                    if callable(update_if_ready):
                        pressure_ready, pressure = update_if_ready(
                            landmarks,
                            pinch=pinch,
                            enabled=True,
                        )
                    else:
                        pressure = estimator.update(
                            landmarks,
                            pinch=pinch,
                            enabled=True,
                        )
                        pressure_ready = True
                except Exception as exc:
                    terminal_exit_code = _EXIT_ESTIMATOR_ERROR
                    terminal_reason = _error_reason("estimator_error", exc)
                    break
                if not pressure_ready:
                    sleep(poll_interval_s)
                    continue
            if pressure is None:
                pressure = inactive_pressure("pressure_unavailable", available=False)
            try:
                decision = proposal.update(base_gripper, pressure)
            except Exception as exc:
                terminal_exit_code = _EXIT_PROPOSAL_ERROR
                terminal_reason = _error_reason("proposal_error", exc)
                break
            actual_gripper = legacy_actual.update(decision.base_gripper)

        if pressure is None:
            status = str(decision.reason)
        else:
            status = str(getattr(pressure, "status", "pressure_unavailable"))
        if decision.fault_latched:
            pressure_recovery_required = True
        elif (
            pressure is not None
            and decision.state == "armed"
            and bool(getattr(pressure, "available", False))
            and not bool(getattr(pressure, "active", False))
            and status == "baseline"
        ):
            pressure_recovery_required = False
        if (
            decision.fault_latched
            and previous_proposal is not None
            and decision.proposed_gripper < previous_proposal
        ):
            fault_closure_violations += 1
        previous_proposal = decision.proposed_gripper

        fallback_used = decision.reason not in {"active", "baseline"}
        telemetry = IRShadowTelemetrySample(
            control_observed_at_s=control_observed_at_s,
            state=state,
            pinch=pinch,
            roi_mode=getattr(pressure, "roi_mode", None),
            pressure=pressure,
            pressure_status=status,
            baseline_ready=decision.state == "armed" and not decision.fault_latched,
            base_gripper_pos=decision.base_gripper,
            proposed_gripper_pos=decision.proposed_gripper,
            actual_gripper_pos=actual_gripper,
            fault_latched=decision.fault_latched,
            fallback_used=fallback_used,
            fallback_reason=decision.reason if fallback_used else None,
        )
        try:
            logger.finalize(telemetry, command_sent=False)
        except Exception as exc:
            terminal_exit_code = _EXIT_SIDECAR_ERROR
            terminal_reason = _error_reason("sidecar_error", exc)
            break

        finalized_at_s = clock()
        if previous_control_observed_at_s is not None:
            _append_milliseconds(
                metric_values["loop_period_ms"],
                control_observed_at_s - previous_control_observed_at_s,
            )
        previous_control_observed_at_s = control_observed_at_s
        _append_milliseconds(
            metric_values["control_latency_ms"],
            finalized_at_s - control_observed_at_s,
        )
        if pressure is not None:
            _append_milliseconds(
                metric_values["oak_age_ms"],
                getattr(pressure, "oak_age_s", None),
            )
            _append_milliseconds(
                metric_values["thermal_age_ms"],
                getattr(pressure, "thermal_age_s", None),
            )
            _append_milliseconds(
                metric_values["pair_skew_ms"],
                getattr(pressure, "sensor_skew_s", None),
            )

        states[state] += 1
        statuses[status] += 1
        if pressure is not None and not bool(getattr(pressure, "available", False)):
            rejections[status] += 1
        ticks += 1

        try:
            sidecar_enabled = bool(getattr(logger, "enabled", False))
        except Exception as exc:
            terminal_exit_code = _EXIT_SIDECAR_ERROR
            terminal_reason = _error_reason("sidecar_error", exc)
            break
        if not sidecar_enabled:
            terminal_exit_code = _EXIT_SIDECAR_DISABLED
            terminal_reason = "sidecar_disabled"
            break

        elapsed_s = finalized_at_s - started_at_s
        if progress is not None and (
            progress_interval_s <= 0.0 or elapsed_s >= next_progress_elapsed_s
        ):
            report = {
                "elapsed_s": elapsed_s,
                "ticks": ticks,
                "cycles": cycles.cycles,
                "state": state,
                "status": status,
            }
            try:
                progress(report)
            except Exception:
                pass
            if progress_interval_s > 0.0:
                while next_progress_elapsed_s <= elapsed_s:
                    next_progress_elapsed_s += progress_interval_s

        if duration_elapsed:
            break
        sleep(poll_interval_s)

    if terminal_reason is not None:
        return summary(terminal_exit_code, terminal_reason)
    if fault_closure_violations:
        return summary(_EXIT_FAULT_CLOSURE, "fault_closure_violation")
    if pressure_recovery_required:
        return summary(_EXIT_PRESSURE_FAULT_LATCHED, "pressure_fault_latched")
    if cycles.cycles < min_cycles:
        return summary(_EXIT_INSUFFICIENT_CYCLES, "insufficient_cycles")
    return summary(0, "completed")


def _empty_summary(args, exit_code: int, reason: str) -> SoakSummary:
    return SoakSummary(
        exit_code=exit_code,
        reason=reason,
        ticks=0,
        cycles=0,
        state_counts={},
        status_counts={},
        rejection_counts={},
        fault_closure_violations=0,
        sidecar=str(args.sidecar),
        metrics=_summarize_metrics({name: [] for name in _METRIC_NAMES}),
    )


def _with_cleanup_failures(
    summary: SoakSummary,
    cleanup_failures: tuple[str, ...],
) -> SoakSummary:
    failures = tuple(summary.cleanup_failures) + tuple(cleanup_failures)
    if not failures:
        return summary
    exit_code = summary.exit_code
    reason = summary.reason
    if exit_code == 0:
        exit_code = _EXIT_CLEANUP_ERROR
        reason = "cleanup_failed"
    return SoakSummary(
        exit_code=exit_code,
        reason=reason,
        ticks=summary.ticks,
        cycles=summary.cycles,
        state_counts=summary.state_counts,
        status_counts=summary.status_counts,
        rejection_counts=summary.rejection_counts,
        fault_closure_violations=summary.fault_closure_violations,
        sidecar=summary.sidecar,
        metrics=summary.metrics,
        cleanup_failures=failures,
    )


def execute_soak(
    args,
    *,
    factories: RuntimeFactories | None = None,
    soak_runner=None,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> SoakSummary:
    """Build, run, and clean the Gate 1 runtime without swallowing BaseException."""
    runtime = SoakRuntime()
    runner = soak_runner or run_soak
    setup_complete = False
    summary = None
    try:
        build_runtime(args, factories=factories, runtime=runtime)
        setup_complete = True
        summary = runner(
            source=runtime.source,
            estimator=runtime.estimator,
            logger=runtime.logger,
            duration_s=args.duration_s,
            min_cycles=args.min_cycles,
            max_oak_stall_s=args.max_oak_stall_ms / 1000.0,
            progress=progress,
        )
    except SetupFailure as exc:
        summary = _empty_summary(args, exc.exit_code, exc.reason)
    except RuntimeSetupError as exc:
        summary = _empty_summary(args, _EXIT_SETUP_ERROR, exc.reason)
    except KeyboardInterrupt:
        summary = _empty_summary(args, 130, "keyboard_interrupt")
    except Exception as exc:
        category = "runtime_error" if setup_complete else "setup_error"
        summary = _empty_summary(
            args,
            _EXIT_RUNTIME_ERROR if setup_complete else _EXIT_SETUP_ERROR,
            _error_reason(category, exc),
        )
    finally:
        cleanup_failures = runtime.cleanup()

    return _with_cleanup_failures(summary, cleanup_failures)


def _summary_payload(summary: SoakSummary) -> dict[str, object]:
    return {
        "type": "summary",
        "exit_code": summary.exit_code,
        "reason": summary.reason,
        "ticks": summary.ticks,
        "cycles": summary.cycles,
        "state_counts": summary.state_counts,
        "status_counts": summary.status_counts,
        "rejection_counts": summary.rejection_counts,
        "fault_closure_violations": summary.fault_closure_violations,
        "sidecar": summary.sidecar,
        "metrics": summary.metrics,
        "cleanup_failures": summary.cleanup_failures,
    }


def _json_line(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def main(
    argv=None,
    *,
    factories: RuntimeFactories | None = None,
    soak_runner=None,
    output=print,
) -> int:
    args = parse_args(argv)

    def progress(report: dict[str, object]) -> None:
        output(_json_line({"type": "progress", **report}))

    summary = execute_soak(
        args,
        factories=factories,
        soak_runner=soak_runner,
        progress=progress,
    )
    output(_json_line(_summary_payload(summary)))
    return summary.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
