from __future__ import annotations

from dataclasses import dataclass, replace
from math import floor

import numpy as np

from .ir_thermal_projection import (
    THERMAL_HEIGHT,
    THERMAL_WIDTH,
    ThermalProjectionResult,
    project_raw_depth_pixel_to_thermal,
)


@dataclass(frozen=True)
class SparseThermalMapResult:
    status: str
    winners: tuple[
        tuple[tuple[int, int], ThermalProjectionResult], ...
    ]
    rejections: tuple[ThermalProjectionResult, ...]
    input_count: int
    accepted_count: int
    rejected_count: int
    collision_count: int


def _validated_samples(samples):
    if not isinstance(samples, (tuple, list)):
        raise ValueError("samples must be a finite tuple or list")
    normalized = []
    for sample in samples:
        if (
            not isinstance(sample, tuple)
            or len(sample) != 3
            or any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                for value in sample
            )
        ):
            raise ValueError(
                "samples must contain exactly (x, y, raw_depth) integers"
            )
        normalized.append(tuple(int(value) for value in sample))
    return tuple(normalized)


def _candidate(result, raw_depth):
    if result.thermal_uv is None or result.thermal_xyz_m is None:
        raise RuntimeError("scalar projector contract missing geometry")
    u, v = result.thermal_uv
    thermal_z = result.thermal_xyz_m[2]
    if (
        not np.all(np.isfinite((u, v, thermal_z)))
        or thermal_z <= 0.0
    ):
        raise RuntimeError("scalar projector contract has invalid geometry")
    thermal_xy = (floor(u + 0.5), floor(v + 0.5))
    winner_key = (
        thermal_z,
        result.source_depth_xy[1],
        result.source_depth_xy[0],
        raw_depth,
    )
    return thermal_xy, winner_key


def project_raw_depth_samples_to_sparse_thermal(
    *,
    samples,
    rs_module,
    depth_scale_m,
    depth_intrinsics,
    depth_to_color_extrinsics,
    R_tc,
    T_tc,
    thermal_K,
    thermal_D,
) -> SparseThermalMapResult:
    normalized = _validated_samples(samples)
    selected = {}
    rejections = []
    accepted_count = 0

    for source_x, source_y, raw_depth in normalized:
        projected = project_raw_depth_pixel_to_thermal(
            rs_module=rs_module,
            source_depth_xy=(source_x, source_y),
            raw_depth=raw_depth,
            depth_scale_m=depth_scale_m,
            depth_intrinsics=depth_intrinsics,
            depth_to_color_extrinsics=depth_to_color_extrinsics,
            R_tc=R_tc,
            T_tc=T_tc,
            thermal_K=thermal_K,
            thermal_D=thermal_D,
        )
        if projected.status != "ok":
            rejections.append(projected)
            continue

        thermal_xy, winner_key = _candidate(projected, raw_depth)
        if not (
            0 <= thermal_xy[0] < THERMAL_WIDTH
            and 0 <= thermal_xy[1] < THERMAL_HEIGHT
        ):
            rejections.append(
                replace(
                    projected,
                    status="thermal_quantized_out_of_bounds",
                    thermal_uv=None,
                )
            )
            continue

        accepted_count += 1
        current = selected.get(thermal_xy)
        if current is None or winner_key < current[0]:
            selected[thermal_xy] = (winner_key, projected)

    winners = tuple(
        (thermal_xy, selected[thermal_xy][1])
        for thermal_xy in sorted(
            selected,
            key=lambda xy: (xy[1], xy[0]),
        )
    )
    rejected_count = len(rejections)
    return SparseThermalMapResult(
        status="ok" if winners else "no_valid_projections",
        winners=winners,
        rejections=tuple(rejections),
        input_count=len(normalized),
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        collision_count=accepted_count - len(winners),
    )
