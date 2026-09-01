import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from ir_force.ir_hand_calibration import (
    InvalidProjectionResult,
    ProjectionCalibration,
    ProjectionResult,
    ProjectionSample,
    fit_projection,
    load_projection_calibration,
    project_oak_to_ir,
    save_projection_calibration,
)


def _samples():
    samples = []
    for oak_x in (0.2, 0.5, 0.8):
        for oak_y in (0.25, 0.55, 0.85):
            for oak_z in (0.35, 0.65):
                ir_x = 12.0 + 140.0 * oak_x + 6.0 * oak_z
                ir_y = 5.0 + 110.0 * oak_y - 4.0 * oak_z
                samples.append(ProjectionSample(oak_x, oak_y, oak_z, ir_x, ir_y))
    return samples


def _direct_calibration():
    return ProjectionCalibration(
        coeff_x=(0.0, 1.0, 0.0, 0.0),
        coeff_y=(0.0, 0.0, 1.0, 0.0),
        rms_error_px=0.0,
        max_error_px=0.0,
        sample_count=12,
        image_size=(160, 128),
    )


def test_fit_projection_recovers_depth_affine_mapping():
    calibration = fit_projection(_samples())

    projection = project_oak_to_ir(calibration, oak_x=0.4, oak_y=0.6, oak_z=0.5)
    ir_x, ir_y = projection

    assert projection.valid is True
    assert projection.status == "ok"
    assert abs(ir_x - (12.0 + 140.0 * 0.4 + 6.0 * 0.5)) < 1e-6
    assert abs(ir_y - (5.0 + 110.0 * 0.6 - 4.0 * 0.5)) < 1e-6
    assert calibration.rms_error_px < 1e-9
    assert calibration.sample_count == len(_samples())


def test_projection_returns_explicit_invalid_result_outside_thermal_image_bounds():
    calibration = fit_projection(_samples(), image_size=(160, 128))

    projection = project_oak_to_ir(calibration, oak_x=-2.0, oak_y=3.0, oak_z=0.5)

    assert projection.valid is False
    assert projection.status == "projection_out_of_fov"
    assert projection.x < 0.0
    assert projection.y >= 128.0


def test_valid_projection_preserves_full_two_tuple_api():
    projection = project_oak_to_ir(
        _direct_calibration(),
        oak_x=12.0,
        oak_y=34.0,
        oak_z=0.5,
    )

    assert isinstance(projection, tuple)
    assert len(projection) == 2
    assert projection[0] == 12.0
    assert projection[1] == 34.0
    assert projection == (12.0, 34.0)
    x, y = projection
    assert (x, y) == (12.0, 34.0)
    assert projection.x == 12.0
    assert projection.y == 34.0
    assert projection.status == "ok"
    assert projection.valid is True


@pytest.mark.parametrize("result_type", [ProjectionResult, InvalidProjectionResult])
def test_projection_result_supports_tuple_style_iterable_reconstruction(result_type):
    original = result_type(12.0, 34.0)

    restored = result_type(iter(original))

    assert type(restored) is result_type
    assert tuple(restored) == (12.0, 34.0)
    assert restored.status == original.status
    assert restored.valid is original.valid


def _pickle_round_trip(value):
    return pickle.loads(pickle.dumps(value))


@pytest.mark.parametrize(
    "round_trip",
    [_pickle_round_trip, copy.copy, copy.deepcopy],
    ids=["pickle", "copy", "deepcopy"],
)
@pytest.mark.parametrize("result_type", [ProjectionResult, InvalidProjectionResult])
def test_projection_result_round_trips_preserve_type_and_status(result_type, round_trip):
    original = result_type(12.0, 34.0)

    restored = round_trip(original)

    assert type(restored) is result_type
    assert tuple(restored) == (12.0, 34.0)
    assert restored.status == original.status
    assert restored.valid is original.valid


def test_valid_and_invalid_projection_equality_repr_and_hash_contracts():
    raw = (12.0, 34.0)
    valid = ProjectionResult(raw)
    invalid = InvalidProjectionResult(raw)
    equivalent_invalid = InvalidProjectionResult(12.0, 34.0)

    assert valid == raw
    assert raw == valid
    assert hash(valid) == hash(raw)
    assert invalid == equivalent_invalid
    assert hash(invalid) == hash(equivalent_invalid)
    assert invalid != raw
    assert raw != invalid
    assert invalid != valid
    assert valid != invalid
    assert len({invalid, equivalent_invalid, valid}) == 2
    assert repr(invalid) == (
        "InvalidProjectionResult(x=12.0, y=34.0, status='projection_out_of_fov')"
    )


def test_projection_accepts_exact_zero_coordinates():
    projection = project_oak_to_ir(
        _direct_calibration(),
        oak_x=0.0,
        oak_y=0.0,
        oak_z=0.5,
    )

    assert projection.valid is True
    assert projection == (0.0, 0.0)


def test_projection_accepts_fractional_coordinates_immediately_below_image_bounds():
    oak_x = np.nextafter(160.0, -np.inf)
    oak_y = np.nextafter(128.0, -np.inf)

    projection = project_oak_to_ir(
        _direct_calibration(),
        oak_x=oak_x,
        oak_y=oak_y,
        oak_z=0.5,
    )

    assert projection.valid is True
    assert projection == (oak_x, oak_y)


@pytest.mark.parametrize(
    ("oak_x", "oak_y"),
    [
        (-0.01, 64.0),
        (160.0, 64.0),
        (80.0, -0.01),
        (80.0, 128.0),
    ],
)
def test_projection_rejects_each_axis_outside_half_open_image_bounds(oak_x, oak_y):
    projection = project_oak_to_ir(
        _direct_calibration(),
        oak_x=oak_x,
        oak_y=oak_y,
        oak_z=0.5,
    )

    assert projection.valid is False
    assert projection.status == "projection_out_of_fov"


@pytest.mark.parametrize(
    ("oak_x", "oak_y"),
    [
        (np.nan, 64.0),
        (np.inf, 64.0),
        (80.0, np.nan),
        (80.0, -np.inf),
    ],
)
def test_projection_rejects_nonfinite_coordinates(oak_x, oak_y):
    projection = project_oak_to_ir(
        _direct_calibration(),
        oak_x=oak_x,
        oak_y=oak_y,
        oak_z=0.5,
    )

    assert projection.valid is False
    assert projection.status == "projection_out_of_fov"


@pytest.mark.parametrize("oak_z", [np.nan, np.inf, -np.inf])
def test_projection_rejects_nonfinite_oak_depth(oak_z):
    projection = project_oak_to_ir(
        _direct_calibration(),
        oak_x=80.0,
        oak_y=64.0,
        oak_z=oak_z,
    )

    assert projection.valid is False
    assert projection.status == "projection_out_of_fov"


def test_fit_projection_rejects_too_few_samples():
    with pytest.raises(ValueError, match="at least 6"):
        fit_projection(_samples()[:5])


def test_fit_projection_rejects_constant_ir_targets_despite_varied_oak_samples():
    samples = [
        ProjectionSample(sample.oak_x, sample.oak_y, sample.oak_z, 79.5, 63.5)
        for sample in _samples()
    ]

    with pytest.raises(ValueError, match="IR target spatial spread"):
        fit_projection(samples)


def test_fit_projection_rejects_rank_deficient_oak_affine_design():
    samples = [
        ProjectionSample(
            oak_x=value,
            oak_y=2.0 * value,
            oak_z=0.5,
            ir_x=20.0 + 80.0 * value,
            ir_y=15.0 + 60.0 * value,
        )
        for value in np.linspace(0.1, 0.9, 12)
    ]

    with pytest.raises(ValueError, match="OAK affine design matrix is rank-deficient"):
        fit_projection(samples)


def test_fit_projection_rejects_full_rank_but_near_degenerate_oak_design():
    samples = [
        ProjectionSample(
            oak_x=value,
            oak_y=value**2,
            oak_z=0.5 + (1e-9 if index % 2 else -1e-9),
            ir_x=10.0 + 100.0 * value,
            ir_y=8.0 + 90.0 * value**2,
        )
        for index, value in enumerate(np.linspace(0.1, 0.9, 12))
    ]

    design = np.array(
        [[1.0, sample.oak_x, sample.oak_y, sample.oak_z] for sample in samples]
    )
    assert np.linalg.matrix_rank(design) == 4

    with pytest.raises(ValueError, match="OAK affine design matrix is ill-conditioned"):
        fit_projection(samples)


def test_projection_save_load_round_trip(tmp_path: Path):
    path = tmp_path / "projection.json"
    calibration = fit_projection(_samples())

    save_projection_calibration(path, calibration)
    loaded = load_projection_calibration(path)

    assert loaded == calibration


def _calibration_payload(**overrides):
    payload = {
        "version": 1,
        "coeff_x": [0.0, 160.0, 0.0, 0.0],
        "coeff_y": [0.0, 0.0, 128.0, 0.0],
        "rms_error_px": 2.0,
        "max_error_px": 4.0,
        "sample_count": 12,
        "image_size": [160, 128],
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"coeff_x": [0.0, 160.0, 0.0]}, "coeff_x"),
        ({"coeff_y": [0.0, 0.0, float("nan"), 0.0]}, "coeff_y"),
        ({"sample_count": 5}, "at least 6"),
        ({"image_size": [0, 128]}, "image_size"),
    ],
)
def test_load_projection_calibration_rejects_invalid_metadata(tmp_path: Path, overrides, message):
    path = tmp_path / "projection.json"
    path.write_text(json.dumps(_calibration_payload(**overrides)), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_projection_calibration(path)
