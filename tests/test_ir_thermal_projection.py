import json
from pathlib import Path

import numpy as np
import pytest
import pyrealsense2 as rs

from ir_force.ir_thermal_projection import (
    load_frozen_thermal_geometry,
    project_raw_depth_pixel_to_thermal,
    validate_thermal_geometry,
)


FROZEN_XML = Path(
    "/home/zhuokai/hand-teleop/thermal-project-calibration-runs/"
    "worktrees/20260724T210232Z-attempt01/calibration/"
    "FINAL_flir_brown/extrinsic_refined.xml"
)
STAGE0_DATUMS = Path(__file__).resolve().parents[1] / "local/scratch_lepton/stage0b_runtime_datums.json"
EXPECTED_COLOR_XYZ_M = (
    0.016393667395103728,
    -0.001414177547977597,
    0.5008528157761936,
)
EXPECTED_THERMAL_XYZ_M = (
    0.029512874129902465,
    0.037040146356860565,
    0.4967085847151519,
)
EXPECTED_THERMAL_UV = (85.33022454091251, 63.466041827895125)


@pytest.fixture
def projection_inputs():
    datums = json.loads(STAGE0_DATUMS.read_text())
    depth = datums["depth_intrinsics"]
    depth_intrinsics = rs.intrinsics()
    depth_intrinsics.width = depth["width"]
    depth_intrinsics.height = depth["height"]
    depth_intrinsics.fx = depth["fx"]
    depth_intrinsics.fy = depth["fy"]
    depth_intrinsics.ppx = depth["ppx"]
    depth_intrinsics.ppy = depth["ppy"]
    depth_intrinsics.model = rs.distortion.brown_conrady
    depth_intrinsics.coeffs = depth["coeffs"]

    factory = datums["factory_extrinsics"]["depth_to_color"]
    depth_to_color = rs.extrinsics()
    # Stage 0 records the matrix in row-major order; the SDK setter is column-major.
    depth_to_color.rotation = np.asarray(factory["rotation"]).reshape(3, 3).T.reshape(-1)
    depth_to_color.translation = factory["translation_m"]
    R_tc, T_tc, thermal_K, thermal_D = load_frozen_thermal_geometry(FROZEN_XML)
    return {
        "rs_module": rs,
        "source_depth_xy": (644, 352),
        "raw_depth": 500,
        "depth_scale_m": datums["depth_scale_m"],
        "depth_intrinsics": depth_intrinsics,
        "depth_to_color_extrinsics": depth_to_color,
        "R_tc": R_tc,
        "T_tc": T_tc,
        "thermal_K": thermal_K,
        "thermal_D": thermal_D,
    }


def test_loads_only_the_frozen_thermal_to_color_geometry():
    R_tc, T_tc, thermal_K, thermal_D = load_frozen_thermal_geometry(FROZEN_XML)

    assert R_tc.shape == (3, 3)
    assert T_tc.shape == (3,)
    assert thermal_K.shape == (3, 3)
    assert thermal_D.shape == (5,)
    np.testing.assert_allclose(R_tc @ R_tc.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(R_tc) == pytest.approx(1.0, abs=1e-9)


def test_rejects_a_byte_modified_xml_before_parsing(tmp_path):
    modified = tmp_path / "extrinsic_refined.xml"
    modified.write_bytes(FROZEN_XML.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="SHA-256"):
        load_frozen_thermal_geometry(modified)


@pytest.mark.parametrize(
    ("direction", "unit", "R_edit", "message"),
    [
        ("color_to_thermal", "meter", None, "direction"),
        ("thermal_to_color", "millimeter", None, "unit"),
        ("thermal_to_color", "meter", (0, 0, 2.0), "rotation"),
    ],
)
def test_rejects_invalid_transform_contract(direction, unit, R_edit, message):
    R_tc = np.eye(3)
    if R_edit is not None:
        row, col, value = R_edit
        R_tc[row, col] = value

    with pytest.raises(ValueError, match=message):
        validate_thermal_geometry(
            R_tc,
            np.zeros(3),
            np.eye(3),
            np.zeros(5),
            direction=direction,
            unit=unit,
        )


def test_projects_independent_golden_datum_without_software_flip(projection_inputs):
    result = project_raw_depth_pixel_to_thermal(**projection_inputs)

    assert result.status == "ok"
    assert result.depth_m == pytest.approx(0.5000000237487257, rel=0.0, abs=1e-12)
    np.testing.assert_allclose(result.color_xyz_m, EXPECTED_COLOR_XYZ_M, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(result.thermal_xyz_m, EXPECTED_THERMAL_XYZ_M, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(result.thermal_uv, EXPECTED_THERMAL_UV, rtol=0.0, atol=1e-3)
    assert result.thermal_uv[0] != pytest.approx(159.0 - EXPECTED_THERMAL_UV[0], abs=1e-3)


@pytest.mark.parametrize(
    ("raw_depth", "depth_scale_m"),
    [
        (0, 0.001),
        (-1, 0.001),
        (500.0, 0.001),
        (500, 0.0),
        (500, -0.001),
        (500, float("nan")),
    ],
)
def test_rejects_invalid_raw_depth_or_scale(projection_inputs, raw_depth, depth_scale_m):
    inputs = {**projection_inputs, "raw_depth": raw_depth, "depth_scale_m": depth_scale_m}
    result = project_raw_depth_pixel_to_thermal(**inputs)

    assert result.status == "depth_invalid"
    assert result.thermal_uv is None


@pytest.mark.parametrize("source_depth_xy", [(-1, 352), (1280, 352)])
def test_rejects_integer_source_coordinates_outside_native_depth_bounds(
    projection_inputs, source_depth_xy
):
    result = project_raw_depth_pixel_to_thermal(
        **{**projection_inputs, "source_depth_xy": source_depth_xy}
    )

    assert result.status == "source_depth_out_of_bounds"
    assert result.thermal_uv is None


@pytest.mark.parametrize("source_depth_xy", [(644.5, 352), (644, 352.5)])
def test_rejects_noninteger_source_coordinates_before_sdk_geometry(
    projection_inputs, source_depth_xy
):
    class NoGeometryRS:
        distortion = rs.distortion

        @staticmethod
        def rs2_deproject_pixel_to_point(*_args):
            raise AssertionError("SDK geometry must not run for noninteger coordinates")

    inputs = {
        **projection_inputs,
        "source_depth_xy": source_depth_xy,
        "rs_module": NoGeometryRS,
    }
    result = project_raw_depth_pixel_to_thermal(**inputs)

    assert result.status == "source_depth_out_of_bounds"
    assert result.thermal_uv is None


def test_rejects_non_brown_depth_model(projection_inputs):
    projection_inputs["depth_intrinsics"].model = rs.distortion.inverse_brown_conrady
    result = project_raw_depth_pixel_to_thermal(**projection_inputs)

    assert result.status == "depth_model_mismatch"
    assert result.thermal_uv is None


def test_rejects_point_behind_thermal_camera(projection_inputs):
    inputs = {
        **projection_inputs,
        "R_tc": np.eye(3),
        "T_tc": np.array([0.0, 0.0, 2.0]),
    }
    result = project_raw_depth_pixel_to_thermal(**inputs)

    assert result.status == "nonpositive_or_nonfinite_geometry"
    assert result.thermal_uv is None


def test_rejects_projection_outside_native_thermal_frame(projection_inputs):
    thermal_K = projection_inputs["thermal_K"].copy()
    thermal_K[0, 2] = 10_000.0
    result = project_raw_depth_pixel_to_thermal(
        **{**projection_inputs, "thermal_K": thermal_K}
    )

    assert result.status == "thermal_out_of_fov"
    assert result.thermal_uv is None


def test_projection_is_deterministic(projection_inputs):
    first = project_raw_depth_pixel_to_thermal(**projection_inputs)
    second = project_raw_depth_pixel_to_thermal(**projection_inputs)

    assert first == second
