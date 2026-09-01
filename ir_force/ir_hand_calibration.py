from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

_MISSING = object()
_AFFINE_PARAMETER_COUNT = 4
_MAX_OAK_DESIGN_CONDITION = 1e6
_MIN_IR_TARGET_SPAN_PX = 2.0


@dataclass(frozen=True)
class ProjectionSample:
    oak_x: float
    oak_y: float
    oak_z: float
    ir_x: float
    ir_y: float


@dataclass(frozen=True)
class ProjectionCalibration:
    coeff_x: tuple[float, float, float, float]
    coeff_y: tuple[float, float, float, float]
    rms_error_px: float
    max_error_px: float
    sample_count: int
    image_size: tuple[int, int] = (160, 128)


class ProjectionResult(tuple):
    __slots__ = ()
    status = "ok"
    valid = True

    def __new__(cls, x, y=_MISSING):
        if y is _MISSING:
            coordinates = tuple(x)
            if len(coordinates) != 2:
                raise ValueError("projection result requires exactly two coordinates")
            x, y = coordinates
        return super().__new__(cls, (float(x), float(y)))

    @property
    def x(self) -> float:
        return self[0]

    @property
    def y(self) -> float:
        return self[1]


class InvalidProjectionResult(ProjectionResult):
    __slots__ = ()
    status = "projection_out_of_fov"
    valid = False

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(x={self.x!r}, y={self.y!r}, "
            f"status={self.status!r})"
        )

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and tuple.__eq__(self, other)

    def __ne__(self, other) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash((type(self), tuple(self)))


def validate_projection_calibration(
    calibration: ProjectionCalibration,
    *,
    min_samples: int = 6,
    max_rms_error_px: float | None = None,
    max_error_px: float | None = None,
    expected_image_size: tuple[int, int] | None = None,
) -> ProjectionCalibration:
    for name in ("coeff_x", "coeff_y"):
        values = np.asarray(getattr(calibration, name), dtype=float)
        if values.shape != (4,):
            raise ValueError(f"{name} must contain exactly 4 coefficients")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite coefficients")

    errors = np.asarray([calibration.rms_error_px, calibration.max_error_px], dtype=float)
    if not np.all(np.isfinite(errors)) or np.any(errors < 0.0):
        raise ValueError("projection residuals must be finite and non-negative")
    if calibration.max_error_px < calibration.rms_error_px:
        raise ValueError("max_error_px must be greater than or equal to rms_error_px")
    if calibration.sample_count < min_samples:
        raise ValueError(f"projection calibration requires at least {min_samples} samples")

    image_size = tuple(calibration.image_size)
    if len(image_size) != 2 or any(int(v) <= 0 for v in image_size):
        raise ValueError("image_size must contain two positive dimensions")
    if expected_image_size is not None and image_size != tuple(expected_image_size):
        raise ValueError(f"image_size must be {tuple(expected_image_size)}, got {image_size}")
    if max_rms_error_px is not None and calibration.rms_error_px > max_rms_error_px:
        raise ValueError(
            f"rms_error_px {calibration.rms_error_px:.2f} exceeds {max_rms_error_px:.2f}"
        )
    if max_error_px is not None and calibration.max_error_px > max_error_px:
        raise ValueError(f"max_error_px {calibration.max_error_px:.2f} exceeds {max_error_px:.2f}")
    return calibration


def _design_matrix(samples: Sequence[ProjectionSample]) -> np.ndarray:
    return np.array([[1.0, s.oak_x, s.oak_y, s.oak_z] for s in samples], dtype=float)


def fit_projection(
    samples: Sequence[ProjectionSample],
    *,
    image_size: tuple[int, int] = (160, 128),
) -> ProjectionCalibration:
    if len(samples) < 6:
        raise ValueError("projection calibration requires at least 6 samples")

    design = _design_matrix(samples)
    target_x = np.array([s.ir_x for s in samples], dtype=float)
    target_y = np.array([s.ir_y for s in samples], dtype=float)
    rank = int(np.linalg.matrix_rank(design))
    if rank < _AFFINE_PARAMETER_COUNT:
        raise ValueError(
            "OAK affine design matrix is rank-deficient "
            f"(rank {rank} < {_AFFINE_PARAMETER_COUNT})"
        )
    condition = float(np.linalg.cond(design))
    if not np.isfinite(condition) or condition > _MAX_OAK_DESIGN_CONDITION:
        raise ValueError(
            "OAK affine design matrix is ill-conditioned "
            f"(condition {condition:.3g} > {_MAX_OAK_DESIGN_CONDITION:.3g})"
        )
    span_x = float(np.ptp(target_x))
    span_y = float(np.ptp(target_y))
    if span_x < _MIN_IR_TARGET_SPAN_PX or span_y < _MIN_IR_TARGET_SPAN_PX:
        raise ValueError(
            "IR target spatial spread must be at least "
            f"{_MIN_IR_TARGET_SPAN_PX:.1f}px on both axes "
            f"(x={span_x:.3g}px, y={span_y:.3g}px)"
        )
    coeff_x, *_ = np.linalg.lstsq(design, target_x, rcond=None)
    coeff_y, *_ = np.linalg.lstsq(design, target_y, rcond=None)
    pred_x = design @ coeff_x
    pred_y = design @ coeff_y
    err = np.sqrt((pred_x - target_x) ** 2 + (pred_y - target_y) ** 2)

    return ProjectionCalibration(
        coeff_x=tuple(float(v) for v in coeff_x),
        coeff_y=tuple(float(v) for v in coeff_y),
        rms_error_px=float(np.sqrt(np.mean(err**2))),
        max_error_px=float(np.max(err)),
        sample_count=len(samples),
        image_size=tuple(int(v) for v in image_size),
    )


def _eval(coeff: tuple[float, float, float, float], oak_x: float, oak_y: float, oak_z: float) -> float:
    return float(coeff[0] + coeff[1] * oak_x + coeff[2] * oak_y + coeff[3] * oak_z)


def project_oak_to_ir(
    calibration: ProjectionCalibration,
    *,
    oak_x: float,
    oak_y: float,
    oak_z: float,
) -> ProjectionResult:
    width, height = calibration.image_size
    ir_x = _eval(calibration.coeff_x, oak_x, oak_y, oak_z)
    ir_y = _eval(calibration.coeff_y, oak_x, oak_y, oak_z)
    in_frame = (
        np.isfinite(ir_x)
        and np.isfinite(ir_y)
        and 0.0 <= ir_x < width
        and 0.0 <= ir_y < height
    )
    result_type = ProjectionResult if in_frame else InvalidProjectionResult
    return result_type(ir_x, ir_y)


def save_projection_calibration(path: Path, calibration: ProjectionCalibration) -> None:
    validate_projection_calibration(calibration)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(calibration)
    payload["version"] = 1
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_projection_calibration(path: Path) -> ProjectionCalibration:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("version", 0)) != 1:
        raise ValueError(f"unsupported projection calibration version in {path}")
    try:
        calibration = ProjectionCalibration(
            coeff_x=tuple(float(v) for v in payload["coeff_x"]),
            coeff_y=tuple(float(v) for v in payload["coeff_y"]),
            rms_error_px=float(payload["rms_error_px"]),
            max_error_px=float(payload["max_error_px"]),
            sample_count=int(payload["sample_count"]),
            image_size=tuple(int(v) for v in payload["image_size"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid projection calibration in {path}: {exc}") from exc
    return validate_projection_calibration(calibration)
