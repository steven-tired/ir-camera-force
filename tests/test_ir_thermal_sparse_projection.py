import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import pyrealsense2 as rs

import ir_force.ir_thermal_sparse_projection as sparse
from ir_force.data_paths import calibration_runs_root
from ir_force.ir_thermal_projection import (
    ThermalProjectionResult,
    load_frozen_thermal_geometry,
)


FROZEN_XML = (
    calibration_runs_root()
    / "worktrees/20260724T210232Z-attempt01/calibration/FINAL_flir_brown"
    / "extrinsic_refined.xml"
)

#: The frozen extrinsic is produced by the calibration rig and is not carried
#: in this repository (see docs/CALIBRATION_PROVENANCE.md). Only the tests that
#: actually read it are guarded — a module-level skip would take the pure
#: geometry tests with it, which need no rig at all.
requires_frozen_extrinsic = pytest.mark.skipif(
    not FROZEN_XML.exists(),
    reason=f"frozen calibration extrinsic not present at {FROZEN_XML}",
)
STAGE0_DATUMS = (
    Path(__file__).resolve().parents[1]
    / "local/scratch_lepton/stage0b_runtime_datums.json"
)


@pytest.fixture
def real_projection_inputs():
    runtime = json.loads(STAGE0_DATUMS.read_text())
    depth = runtime["depth_intrinsics"]
    intrinsics = rs.intrinsics()
    intrinsics.width = depth["width"]
    intrinsics.height = depth["height"]
    intrinsics.fx = depth["fx"]
    intrinsics.fy = depth["fy"]
    intrinsics.ppx = depth["ppx"]
    intrinsics.ppy = depth["ppy"]
    intrinsics.model = rs.distortion.brown_conrady
    intrinsics.coeffs = depth["coeffs"]

    factory = runtime["factory_extrinsics"]["depth_to_color"]
    extrinsics = rs.extrinsics()
    extrinsics.rotation = (
        np.asarray(factory["rotation"])
        .reshape(3, 3)
        .T.reshape(-1)
    )
    extrinsics.translation = factory["translation_m"]
    R_tc, T_tc, thermal_K, thermal_D = (
        load_frozen_thermal_geometry(FROZEN_XML)
    )
    return {
        "rs_module": rs,
        "depth_scale_m": runtime["depth_scale_m"],
        "depth_intrinsics": intrinsics,
        "depth_to_color_extrinsics": extrinsics,
        "R_tc": R_tc,
        "T_tc": T_tc,
        "thermal_K": thermal_K,
        "thermal_D": thermal_D,
    }


def ok_result(source_xy, uv, thermal_z):
    return ThermalProjectionResult(
        status="ok",
        source_depth_xy=source_xy,
        depth_m=0.5,
        color_xyz_m=(0.0, 0.0, 0.5),
        thermal_xyz_m=(0.0, 0.0, thermal_z),
        thermal_uv=uv,
    )


def install_scalar(monkeypatch, results):
    def fake_scalar(**kwargs):
        return results[(kwargs["source_depth_xy"], kwargs["raw_depth"])]

    monkeypatch.setattr(
        sparse,
        "project_raw_depth_pixel_to_thermal",
        fake_scalar,
    )


def call_sparse(samples):
    return sparse.project_raw_depth_samples_to_sparse_thermal(
        samples=samples,
        rs_module=object(),
        depth_scale_m=0.001,
        depth_intrinsics=object(),
        depth_to_color_extrinsics=object(),
        R_tc=object(),
        T_tc=object(),
        thermal_K=object(),
        thermal_D=object(),
    )


def test_quantizes_nearest_pixel_center_and_sorts(monkeypatch):
    results = {
        ((2, 1), 500): ok_result((2, 1), (10.49, 8.50), 0.5),
        ((1, 1), 500): ok_result((1, 1), (2.50, 3.49), 0.5),
    }
    install_scalar(monkeypatch, results)

    result = call_sparse([(2, 1, 500), (1, 1, 500)])

    assert result.status == "ok"
    assert tuple(item[0] for item in result.winners) == ((3, 3), (10, 9))
    assert result.input_count == 2
    assert result.accepted_count == 2
    assert result.rejected_count == 0
    assert result.collision_count == 0


def test_keeps_nearest_thermal_surface_independent_of_input_order(monkeypatch):
    far = ok_result((10, 20), (40.1, 30.1), 0.7)
    near = ok_result((11, 20), (40.2, 30.2), 0.4)
    install_scalar(
        monkeypatch,
        {((10, 20), 700): far, ((11, 20), 400): near},
    )

    forward = call_sparse([(10, 20, 700), (11, 20, 400)])
    reverse = call_sparse([(11, 20, 400), (10, 20, 700)])

    assert forward == reverse
    assert forward.winners == (((40, 30), near),)
    assert forward.accepted_count == 2
    assert forward.collision_count == 1


def test_equal_z_tie_uses_source_y_then_x(monkeypatch):
    later_key = ok_result((1, 2), (50.0, 50.0), 0.5)
    earlier_key = ok_result((9, 1), (50.0, 50.0), 0.5)
    install_scalar(
        monkeypatch,
        {((1, 2), 500): later_key, ((9, 1), 500): earlier_key},
    )

    result = call_sparse([(1, 2, 500), (9, 1, 500)])

    assert result.winners == (((50, 50), earlier_key),)


def test_rejects_coordinate_that_rounds_outside_native_frame(monkeypatch):
    projected = ok_result((1, 1), (159.6, 20.0), 0.5)
    install_scalar(monkeypatch, {((1, 1), 500): projected})

    result = call_sparse([(1, 1, 500)])

    assert result.status == "no_valid_projections"
    assert result.winners == ()
    assert result.rejections == (
        replace(
            projected,
            status="thermal_quantized_out_of_bounds",
            thermal_uv=None,
        ),
    )
    assert result.accepted_count == 0
    assert result.rejected_count == 1


def test_preserves_scalar_rejection_and_exact_counts(monkeypatch):
    rejected = ThermalProjectionResult(
        status="depth_invalid",
        source_depth_xy=(1, 1),
    )
    accepted = ok_result((2, 2), (5.0, 6.0), 0.5)
    install_scalar(
        monkeypatch,
        {((1, 1), 0): rejected, ((2, 2), 500): accepted},
    )

    result = call_sparse([(1, 1, 0), (2, 2, 500)])

    assert result.rejections == (rejected,)
    assert result.input_count == 2
    assert result.accepted_count == 1
    assert result.rejected_count == 1
    assert result.collision_count == 0


def test_empty_input_is_explicitly_unavailable():
    result = call_sparse([])

    assert result.status == "no_valid_projections"
    assert result.winners == ()
    assert result.rejections == ()
    assert result.input_count == 0


@pytest.mark.parametrize(
    "samples",
    [
        ((1, 2, 500) for _ in range(1)),
        [(1, 2)],
        [(True, 2, 500)],
        [(1.5, 2, 500)],
    ],
)
def test_malformed_collection_fails_before_geometry(monkeypatch, samples):
    monkeypatch.setattr(
        sparse,
        "project_raw_depth_pixel_to_thermal",
        lambda **_kwargs: pytest.fail("geometry must not run"),
    )

    with pytest.raises(ValueError, match="samples"):
        call_sparse(samples)


@pytest.mark.parametrize(
    "broken",
    [
        ok_result((1, 1), (5.0, 6.0), float("nan")),
        replace(ok_result((1, 1), (5.0, 6.0), 0.5), thermal_uv=None),
    ],
)
def test_broken_scalar_success_contract_fails_closed(
    monkeypatch,
    broken,
):
    install_scalar(monkeypatch, {((1, 1), 500): broken})

    with pytest.raises(RuntimeError, match="scalar projector contract"):
        call_sparse([(1, 1, 500)])


@requires_frozen_extrinsic
def test_real_stage1a_golden_sample_maps_to_native_pixel(
    real_projection_inputs,
):
    result = sparse.project_raw_depth_samples_to_sparse_thermal(
        samples=[(644, 352, 500)],
        **real_projection_inputs,
    )

    assert result.status == "ok"
    assert tuple(item[0] for item in result.winners) == ((85, 63),)
    projected = result.winners[0][1]
    assert projected.thermal_uv == pytest.approx(
        (85.33022454091251, 63.466041827895125),
        rel=0.0,
        abs=1e-3,
    )
    assert result.input_count == 1
    assert result.accepted_count == 1
    assert result.rejected_count == 0
    assert result.collision_count == 0
