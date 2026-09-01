import pytest
import numpy as np
from dataclasses import replace
from types import SimpleNamespace
import io
import json
import sys

import live_lepton_hand_shadow as runner
from ir_force.ir_capture import (
    FrameSample,
    LeptonTelemetry,
)
from ir_force.ir_thermal_projection import (
    ThermalProjectionResult,
)
from ir_force.ir_thermal_sparse_projection import (
    SparseThermalMapResult,
)
from ir_force.realsense_camera import RealSenseRawSample


def test_cli_accepts_900_attempts_but_rejects_901():
    args = runner.parse_args(
        ["--frames", "900", "--output", "hand-shadow.jsonl"]
    )

    assert args.frames == 900
    with pytest.raises(SystemExit, match="2"):
        runner.parse_args(
            ["--frames", "901", "--output", "hand-shadow.jsonl"]
        )


def test_cli_defaults_lepton_port_and_rejects_invalid_port():
    args = runner.parse_args(
        ["--frames", "1", "--output", "hand-shadow.jsonl"]
    )

    assert args.lepton_port == 8080
    with pytest.raises(SystemExit, match="2"):
        runner.parse_args(
            [
                "--frames",
                "1",
                "--output",
                "hand-shadow.jsonl",
                "--lepton-port",
                "0",
            ]
        )


def test_cli_preview_and_manual_ffc_are_explicitly_opt_in():
    headless = runner.parse_args(
        ["--frames", "1", "--output", "headless.jsonl"]
    )
    preview = runner.parse_args(
        [
            "--frames",
            "1",
            "--output",
            "preview.jsonl",
            "--preview",
            "--manual-ffc",
            "--diagnose-inward-samples",
        ]
    )

    assert headless.preview is False
    assert headless.manual_ffc is False
    assert headless.diagnose_inward_samples is False
    assert preview.preview is True
    assert preview.manual_ffc is True
    assert preview.diagnose_inward_samples is True


def test_pinch_signal_cli_requires_preview_manual_ffc_and_no_diagnostics():
    args = runner.parse_args(
        [
            "--frames",
            "900",
            "--output",
            "pinch-signal.jsonl",
            "--preview",
            "--manual-ffc",
            "--pinch-signal-trial",
        ]
    )

    assert args.pinch_signal_trial is True
    for missing in ("--preview", "--manual-ffc"):
        argv = [
            "--frames",
            "900",
            "--output",
            "pinch-signal.jsonl",
            "--preview",
            "--manual-ffc",
            "--pinch-signal-trial",
        ]
        argv.remove(missing)
        with pytest.raises(SystemExit, match="2"):
            runner.parse_args(argv)
    with pytest.raises(SystemExit, match="2"):
        runner.parse_args(
            [
                "--frames",
                "900",
                "--output",
                "pinch-signal.jsonl",
                "--preview",
                "--manual-ffc",
                "--pinch-signal-trial",
                "--diagnose-inward-samples",
            ]
        )


def _accept_pinch_samples(state, centers):
    results = []
    for center in centers:
        state, result = runner._pinch_signal_accept_sample(state, center)
        results.append(result)
    return state, results


def test_pinch_signal_phase_requires_five_valid_samples_not_elapsed_time():
    assert runner.PINCH_SIGNAL_GROUPS == 6
    assert runner.PINCH_SIGNAL_TARGET_VALID_SAMPLES == 5
    assert runner.PINCH_SIGNAL_PHASE_TIMEOUT_S == 10.0
    state = runner._pinch_signal_initial_state()
    assert runner._pinch_signal_cue(state, 0.0)["phase"] == "warmup"

    state = runner._pinch_signal_unlock(state)
    assert runner._pinch_signal_cue(state, 10.0)["phase"] == (
        "prepare_just_touch"
    )
    state = runner._pinch_signal_press_space(state, 10.0)
    cue = runner._pinch_signal_cue(state, 10.25)
    assert cue["phase"] == "record_just_touch"
    assert cue["label"] == "contact"
    assert cue["recording"] is True
    assert cue["valid_samples"] == 0
    assert cue["target_valid_samples"] == 5
    assert cue["phase_remaining_s"] == pytest.approx(9.75)

    # A read arriving after the old 1 s window cannot complete an empty phase.
    assert runner._pinch_signal_cue(state, 11.5)["phase"] == (
        "record_just_touch"
    )
    assert runner._pinch_signal_timed_out(state, 11.5) is False

    state, results = _accept_pinch_samples(
        state,
        [(10.0 + index * 0.1, 20.0) for index in range(5)],
    )
    assert [result["valid_samples_after"] for result in results] == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert all(result["quota_accepted"] for result in results)
    assert state["phase"] == "prepare_press_hard"
    assert state["baseline_center_uv"] == pytest.approx((10.2, 20.0))

    state = runner._pinch_signal_press_space(state, 12.0)
    assert runner._pinch_signal_cue(state, 12.5)["label"] == "press"
    state, result = runner._pinch_signal_accept_sample(
        state,
        (17.0, 20.0),
    )
    assert result["quota_accepted"] is True
    assert result["quota_reasons"] == []
    assert result["pinch_center_shift_px"] == pytest.approx(6.8)
    assert state["valid_samples"] == 1
    state, _results = _accept_pinch_samples(
        state,
        [(17.0, 20.0)] * 4,
    )
    assert state["phase"] == "prepare_return_touch"

    state = runner._pinch_signal_press_space(state, 14.0)
    assert runner._pinch_signal_cue(state, 14.5)["label"] == "contact"
    state, results = _accept_pinch_samples(
        state,
        [(10.2, 20.0)] * 5,
    )
    assert results[-1]["group_completed"] is True
    assert state["phase"] == "rest"

    state = runner._pinch_signal_press_space(state, 16.0)
    assert state == {
        "group_index": 1,
        "phase": "prepare_just_touch",
        "phase_started_s": None,
        "valid_samples": 0,
        "baseline_center_uv": None,
        "just_touch_centers": (),
    }


def test_pinch_signal_timeout_explicitly_invalidates_group():
    state = runner._pinch_signal_press_space(
        runner._pinch_signal_unlock(runner._pinch_signal_initial_state()),
        10.0,
    )

    assert runner._pinch_signal_timed_out(state, 19.999) is False
    assert runner._pinch_signal_timed_out(state, 20.0) is True

    state = runner._pinch_signal_invalidate_group(state, stop=False)
    assert state["phase"] == "rest"
    state = runner._pinch_signal_press_space(state, 21.0)
    assert state["group_index"] == 1
    assert state["phase"] == "prepare_just_touch"

    state = runner._pinch_signal_press_space(state, 22.0)
    state = runner._pinch_signal_invalidate_group(state, stop=True)
    assert state["phase"] == "blocked"


def test_pinch_signal_cue_rejects_invalid_or_backward_time():
    state = runner._pinch_signal_press_space(
        runner._pinch_signal_unlock(runner._pinch_signal_initial_state()),
        10.0,
    )
    for now_s in (9.999, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="time"):
            runner._pinch_signal_cue(state, now_s)


@pytest.mark.parametrize(
    ("cue", "instruction"),
    [
        ({"phase": "warmup", "label": None}, "SHOW RIGHT HAND"),
        (
            {"phase": "prepare_just_touch", "label": "contact"},
            "JUST TOUCH - SPACE WHEN READY",
        ),
        (
            {"phase": "record_just_touch", "label": "contact"},
            "HOLD JUST TOUCH",
        ),
        (
            {"phase": "prepare_press_hard", "label": "press"},
            "PRESS HARD - SPACE WHEN READY",
        ),
        (
            {"phase": "record_press_hard", "label": "press"},
            "HOLD PRESS HARD",
        ),
        (
            {"phase": "prepare_return_touch", "label": "contact"},
            "RETURN TO JUST TOUCH - SPACE WHEN READY",
        ),
        (
            {"phase": "record_return_touch", "label": "contact"},
            "HOLD JUST TOUCH",
        ),
        (
            {"phase": "rest", "label": None},
            "SEPARATE AND REST - SPACE FOR NEXT GROUP",
        ),
        ({"phase": "complete", "label": None}, "PROTOCOL COMPLETE"),
        ({"phase": "blocked", "label": None}, "PROTOCOL BLOCKED"),
    ],
)
def test_pinch_signal_instruction_is_fixed(cue, instruction):
    assert runner._pinch_signal_instruction(cue) == instruction


def test_thermal_patch_and_pinch_center_diagnostics_are_bounded():
    frame = np.arange(25, dtype=np.uint16).reshape(5, 5)

    corner = runner._thermal_patch_statistics(frame, (0, 0))
    assert corner["thermal_patch_3x3_mean_count"] == pytest.approx(3.0)
    assert corner["thermal_patch_3x3_std_counts"] == pytest.approx(
        np.std([[0, 1], [5, 6]])
    )
    center = runner._thermal_patch_statistics(frame, (2, 2))
    assert center["thermal_patch_3x3_mean_count"] == pytest.approx(12.0)
    assert runner._thermal_patches_overlap(
        [(0, 0), (2, 2)],
        frame.shape,
    ) is True
    assert runner._thermal_patches_overlap(
        [(0, 0), (4, 4)],
        frame.shape,
    ) is False

    assert runner._thermal_pinch_center_uv(
        [
            {"label": "index_tip", "thermal_uv": [14.0, 22.0]},
            {"label": "thumb_tip", "thermal_uv": [10.0, 20.0]},
        ]
    ) == pytest.approx((12.0, 21.0))


def test_manual_ffc_restarts_approved_cpp_streamer_and_requires_marker():
    calls = []

    def success(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout="Lepton verified\nManual FFC complete\n",
            stderr="",
        )

    assert "Manual FFC complete" in runner._run_manual_ffc(success)
    assert calls == [
        (
            [
                "/home/zhuokai/hand-teleop/scripts/run_lepton_stream.sh",
                "start",
            ],
            {
                "capture_output": True,
                "text": True,
                "timeout": 30,
            },
        )
    ]

    with pytest.raises(RuntimeError, match="completion marker"):
        runner._run_manual_ffc(
            lambda *_args, **_kwargs: SimpleNamespace(
                returncode=0,
                stdout="Lepton verified\n",
                stderr="",
            )
        )


def test_normalized_color_pixel_uses_floor_and_accepts_exact_edges():
    assert runner._normalized_color_pixel((0.0, 0.0)) == (0, 0)
    assert runner._normalized_color_pixel((0.5, 0.5)) == (640, 360)
    assert runner._normalized_color_pixel((1.0, 1.0)) == (1279, 719)


def test_inward_diagnostic_samples_move_toward_previous_finger_joint():
    image_xy = np.zeros((21, 2), dtype=float)
    image_xy[4] = (0.50, 0.60)
    image_xy[3] = (0.40, 0.40)
    image_xy[8] = (0.70, 0.50)
    image_xy[7] = (0.50, 0.30)

    samples = runner._inward_diagnostic_samples(image_xy)

    assert [(label, fraction) for label, fraction, _xy in samples] == [
        ("thumb_tip", 0.25),
        ("thumb_tip", 0.5),
        ("index_tip", 0.25),
        ("index_tip", 0.5),
    ]
    np.testing.assert_allclose(
        [xy for _label, _fraction, xy in samples],
        [(0.475, 0.55), (0.45, 0.5), (0.65, 0.45), (0.6, 0.4)],
    )


def test_pinch_geometry_record_reuses_tip_depth_and_full_hand_landmarks():
    image_xy = np.full((21, 2), 0.5, dtype=float)
    image_xy[4] = (0.60, 0.50)
    image_xy[8] = (0.70, 0.50)
    image_xy[5] = (0.40, 0.50)
    image_xy[17] = (0.60, 0.50)

    result = runner._pinch_geometry_record(
        image_xy,
        [
            {"label": "thumb_tip", "depth_m": 0.48},
            {"label": "index_tip", "depth_m": 0.51},
        ],
    )

    assert result["valid"] is True
    assert result["reason"] == "OK"
    assert result["pinch_distance_2d_norm"] == pytest.approx(0.5)
    assert result["pinch_depth_delta_m"] == pytest.approx(0.03)


@pytest.mark.parametrize(
    ("normalized_xy", "reason"),
    [
        ((-0.0001, 0.5), "normalized_color_out_of_bounds"),
        ((1.0001, 0.5), "normalized_color_out_of_bounds"),
        ((0.5, float("nan")), "normalized_color_nonfinite"),
    ],
)
def test_normalized_color_pixel_blocks_genuinely_invalid_values(
    normalized_xy,
    reason,
):
    with pytest.raises(ValueError, match=reason):
        runner._normalized_color_pixel(normalized_xy)


def test_color_to_raw_depth_uses_sdk_truncated_source_cell_and_logs_half_up():
    depth = np.zeros((720, 1280), dtype=np.uint16)
    depth[352, 644] = 500
    depth[353, 645] = 800
    sdk_buffer = object()
    calls = {}

    def color_to_depth(
        data,
        scale,
        depth_min,
        depth_max,
        depth_intrinsics,
        color_intrinsics,
        color_to_depth_extrinsics,
        depth_to_color_extrinsics,
        color_pixel,
    ):
        calls["sdk"] = (
            data,
            scale,
            depth_min,
            depth_max,
            depth_intrinsics,
            color_intrinsics,
            color_to_depth_extrinsics,
            depth_to_color_extrinsics,
            color_pixel,
        )
        return [644.9, 352.9]

    def deproject(_intrinsics, pixel, value):
        calls.setdefault("deproject", []).append((list(pixel), value))
        return [float(pixel[0]), float(pixel[1]), value]

    def project_to_color(_intrinsics, point):
        if point[0] == pytest.approx(644.9):
            return [640.6, 360.0]
        return [641.1, 360.0]

    rs = SimpleNamespace(
        rs2_project_color_pixel_to_depth_pixel=color_to_depth,
        rs2_deproject_pixel_to_point=deproject,
        rs2_transform_point_to_point=lambda _extrinsics, point: point,
        rs2_project_point_to_pixel=project_to_color,
    )
    geometry = {
        "depth_intrinsics": object(),
        "color_intrinsics": object(),
        "color_to_depth_extrinsics": object(),
        "depth_to_color_extrinsics": object(),
    }

    result = runner._associate_color_to_raw_depth(
        label="thumb_tip",
        normalized_xy=(0.5, 0.5),
        depth_z16=depth,
        depth_sdk_buffer=sdk_buffer,
        rs_module=rs,
        depth_scale_m=0.001,
        **geometry,
    )

    assert result["status"] == "ok"
    assert result["color_pixel"] == [640, 360]
    assert result["sdk_depth_uv"] == [644.9, 352.9]
    assert result["depth_pixel"] == [644, 352]
    assert result["legacy_half_up_depth_pixel"] == [645, 353]
    assert result["raw_depth"] == 500
    assert result["depth_m"] == 0.5
    assert result["sdk_reprojected_color_uv"] == [640.6, 360.0]
    assert result["sdk_match_error_px"] == pytest.approx(0.6)
    assert result["source_cell_reprojected_color_uv"] == [641.1, 360.0]
    assert calls["sdk"] == (
        sdk_buffer,
        0.001,
        0.2,
        0.9,
        geometry["depth_intrinsics"],
        geometry["color_intrinsics"],
        geometry["color_to_depth_extrinsics"],
        geometry["depth_to_color_extrinsics"],
        [640.0, 360.0],
    )
    assert calls["deproject"] == [
        ([644.9, 352.9], 0.5),
        ([644.0, 352.0], 0.5),
    ]


def _association_result(
    *,
    sdk_depth_uv=(644.2, 352.2),
    raw_depth=500,
    reprojected_color_uv=(640.1, 360.1),
    depth_xyz=(0.0, 0.0, 0.5),
    color_xyz=(0.0, 0.0, 0.5),
):
    depth = np.zeros((720, 1280), dtype=np.uint16)
    if (
        all(np.isfinite(sdk_depth_uv))
        and 0 <= sdk_depth_uv[0] < 1280
        and 0 <= sdk_depth_uv[1] < 720
    ):
        depth[int(sdk_depth_uv[1]), int(sdk_depth_uv[0])] = raw_depth
    rs = SimpleNamespace(
        rs2_project_color_pixel_to_depth_pixel=lambda *_args: list(
            sdk_depth_uv
        ),
        rs2_deproject_pixel_to_point=lambda _intrinsics, _pixel, _value: list(
            depth_xyz
        ),
        rs2_transform_point_to_point=lambda _extrinsics, _point: list(
            color_xyz
        ),
        rs2_project_point_to_pixel=lambda _intrinsics, _point: list(
            reprojected_color_uv
        ),
    )
    return runner._associate_color_to_raw_depth(
        label="thumb_tip",
        normalized_xy=(0.5, 0.5),
        depth_z16=depth,
        depth_sdk_buffer=depth,
        rs_module=rs,
        depth_scale_m=0.001,
        depth_intrinsics=object(),
        color_intrinsics=object(),
        color_to_depth_extrinsics=object(),
        depth_to_color_extrinsics=object(),
    )


def _fractional_sdk_match_result(sdk_reprojected_x):
    depth = np.zeros((720, 1280), dtype=np.uint16)
    depth[352, 644] = 500
    rs = SimpleNamespace(
        rs2_project_color_pixel_to_depth_pixel=lambda *_args: [644.2, 352.2],
        rs2_deproject_pixel_to_point=lambda _intrinsics, pixel, value: [
            float(pixel[0]),
            float(pixel[1]),
            value,
        ],
        rs2_transform_point_to_point=lambda _extrinsics, point: point,
        rs2_project_point_to_pixel=lambda _intrinsics, _point: [
            sdk_reprojected_x,
            360.0,
        ],
    )
    return runner._associate_color_to_raw_depth(
        label="thumb_tip",
        normalized_xy=(0.5, 0.5),
        depth_z16=depth,
        depth_sdk_buffer=depth,
        rs_module=rs,
        depth_scale_m=0.001,
        depth_intrinsics=object(),
        color_intrinsics=object(),
        color_to_depth_extrinsics=object(),
        depth_to_color_extrinsics=object(),
    )


@pytest.mark.parametrize(
    ("sdk_reprojected_x", "status", "reason"),
    [
        (640.75, "ok", None),
        (
            640.7501,
            "blocked",
            "color_to_depth_sdk_match_error_exceeded",
        ),
    ],
)
def test_color_to_raw_depth_applies_inclusive_fractional_sdk_error_gate(
    sdk_reprojected_x,
    status,
    reason,
):
    result = _fractional_sdk_match_result(sdk_reprojected_x)

    assert result["status"] == status
    assert result.get("reason") == reason


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"sdk_depth_uv": (-1.0, -1.0)}, "color_to_depth_sdk_no_match"),
        (
            {"sdk_depth_uv": (float("nan"), 352.0)},
            "color_to_depth_sdk_nonfinite",
        ),
        (
            {"sdk_depth_uv": (1280.0, 352.0)},
            "color_to_depth_sdk_out_of_bounds",
        ),
        ({"raw_depth": 0}, "color_to_depth_zero_depth"),
        ({"raw_depth": 950}, "color_to_depth_depth_out_of_range"),
        (
            {"depth_xyz": (0.0, 0.0, 0.0)},
            "color_to_depth_forward_geometry_invalid",
        ),
        (
            {"color_xyz": (0.0, 0.0, 0.0)},
            "color_to_depth_forward_geometry_invalid",
        ),
    ],
)
def test_color_to_raw_depth_blocks_invalid_or_inexact_candidates(
    kwargs,
    reason,
):
    result = _association_result(**kwargs)

    assert result["status"] == "blocked"
    assert result["reason"] == reason


@pytest.mark.parametrize(
    ("normalized_xy", "sdk_raises", "reason"),
    [
        (
            (float("nan"), 0.5),
            False,
            "normalized_color_nonfinite",
        ),
        (
            (1.01, 0.5),
            False,
            "normalized_color_out_of_bounds",
        ),
        (
            (0.5, 0.5),
            True,
            "color_to_depth_sdk_failed",
        ),
    ],
)
def test_color_to_raw_depth_converts_input_or_sdk_errors_to_blocked_records(
    normalized_xy,
    sdk_raises,
    reason,
):
    def sdk(*_args):
        if sdk_raises:
            raise RuntimeError("SDK failure")
        return [644.2, 352.2]

    result = runner._associate_color_to_raw_depth(
        label="thumb_tip",
        normalized_xy=normalized_xy,
        depth_z16=np.zeros((720, 1280), dtype=np.uint16),
        depth_sdk_buffer=object(),
        rs_module=SimpleNamespace(
            rs2_project_color_pixel_to_depth_pixel=sdk,
        ),
        depth_scale_m=0.001,
        depth_intrinsics=object(),
        color_intrinsics=object(),
        color_to_depth_extrinsics=object(),
        depth_to_color_extrinsics=object(),
    )

    assert result["status"] == "blocked"
    assert result["reason"] == reason


@pytest.mark.parametrize("sdk_result", [None, [644.0]])
def test_color_to_raw_depth_blocks_malformed_sdk_results(sdk_result):
    result = runner._associate_color_to_raw_depth(
        label="thumb_tip",
        normalized_xy=(0.5, 0.5),
        depth_z16=np.zeros((720, 1280), dtype=np.uint16),
        depth_sdk_buffer=object(),
        rs_module=SimpleNamespace(
            rs2_project_color_pixel_to_depth_pixel=lambda *_args: sdk_result,
        ),
        depth_scale_m=0.001,
        depth_intrinsics=object(),
        color_intrinsics=object(),
        color_to_depth_extrinsics=object(),
        depth_to_color_extrinsics=object(),
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "color_to_depth_sdk_invalid"


def test_color_to_raw_depth_blocks_forward_geometry_sdk_failure():
    depth = np.zeros((720, 1280), dtype=np.uint16)
    depth[352, 644] = 500

    def deproject(*_args):
        raise RuntimeError("geometry failure")

    result = runner._associate_color_to_raw_depth(
        label="thumb_tip",
        normalized_xy=(0.5, 0.5),
        depth_z16=depth,
        depth_sdk_buffer=depth,
        rs_module=SimpleNamespace(
            rs2_project_color_pixel_to_depth_pixel=lambda *_args: [
                644.2,
                352.2,
            ],
            rs2_deproject_pixel_to_point=deproject,
        ),
        depth_scale_m=0.001,
        depth_intrinsics=object(),
        color_intrinsics=object(),
        color_to_depth_extrinsics=object(),
        depth_to_color_extrinsics=object(),
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "color_to_depth_forward_geometry_failed"


def test_jsonl_writer_converts_nonfinite_observations_to_null():
    stream = io.StringIO()

    runner._write_jsonl(
        stream,
        {
            "status": "blocked",
            "reasons": ["host_read_completion_time_invalid"],
            "observation": [float("nan"), float("inf"), -float("inf")],
        },
    )

    text = stream.getvalue()
    assert "NaN" not in text
    assert "Infinity" not in text
    assert json.loads(text)["observation"] == [None, None, None]


def _runtime_metadata():
    return {
        "requested": {},
        "sdk_version": "2.58.3",
        "device": {"serial": "233522078685"},
        "resolved": {},
        "color_intrinsics": {},
        "depth_intrinsics": {},
        "depth_scale_m": 0.001,
        "factory_extrinsics": {},
    }


class _RawSource:
    def __init__(self, sample):
        self._sample = sample
        self.runtime_metadata = _runtime_metadata()
        self.color_intrinsics = object()
        self.depth_intrinsics = object()
        self.color_to_depth_extrinsics = object()
        self.depth_to_color_extrinsics = object()
        self.stopped = False

    def start(self):
        pass

    def read(self):
        return self._sample

    def stop(self):
        self.stopped = True


class _SequenceRawSource(_RawSource):
    def __init__(self, samples):
        super().__init__(samples[0])
        self._samples = iter(samples)

    def read(self):
        return next(self._samples)


class _ThermalSource:
    def __init__(self, sample):
        self._sample = sample
        self.closed = False

    def read(self):
        return self._sample

    def close(self):
        self.closed = True


class _SequenceThermalSource(_ThermalSource):
    def __init__(self, samples):
        super().__init__(samples[0])
        self._samples = iter(samples)

    def read(self):
        return next(self._samples)


class _Hands:
    def __init__(self):
        self.closed = False
        self.seen = []

    def process(self, image):
        self.seen.append(image)
        return object()

    def close(self):
        self.closed = True


def test_run_shadow_rejects_unsafe_direct_pinch_signal_mode(tmp_path):
    kwargs = {
        "attempts": 1,
        "output_path": tmp_path / "unused.jsonl",
        "rs_module": SimpleNamespace(),
        "raw_source_factory": lambda: pytest.fail("must reject before I/O"),
        "thermal_source_factory": lambda: pytest.fail("must reject before I/O"),
        "hands_factory": lambda: pytest.fail("must reject before I/O"),
        "pinch_signal_trial": True,
    }

    with pytest.raises(ValueError, match="preview"):
        runner.run_shadow(**kwargs)
    with pytest.raises(ValueError, match="manual FFC"):
        runner.run_shadow(**kwargs, preview=True)


def test_run_shadow_rejects_attempt_archive_outside_pinch_signal(tmp_path):
    writer = SimpleNamespace(
        metadata=lambda: {},
        capture=lambda **_kwargs: {},
    )

    with pytest.raises(ValueError, match="pinch signal"):
        runner.run_shadow(
            attempts=1,
            output_path=tmp_path / "unused.jsonl",
            rs_module=SimpleNamespace(),
            raw_source_factory=lambda: pytest.fail("must reject before I/O"),
            thermal_source_factory=lambda: pytest.fail("must reject before I/O"),
            hands_factory=lambda: pytest.fail("must reject before I/O"),
            attempt_artifact_writer=writer,
        )


def _raw_sample():
    depth = np.zeros((720, 1280), dtype=np.uint16)
    depth[352, 644] = 500
    depth[352, 700] = 500
    return RealSenseRawSample(
        color_rgb=np.zeros((720, 1280, 3), dtype=np.uint8),
        depth_z16=depth,
        observed_at_s=10.0,
        color_frame_number=10,
        depth_frame_number=8,
        color_timestamp_ms=100.0,
        depth_timestamp_ms=98.0,
        color_timestamp_domain="timestamp_domain.global_time",
        depth_timestamp_domain="timestamp_domain.global_time",
        depth_sdk_frame=SimpleNamespace(get_data=lambda: depth),
    )


def _thermal_sample():
    frame = np.zeros((120, 160), dtype=np.uint16)
    frame[63, 85] = 29234
    frame[63, 90] = 29345
    return FrameSample(
        t=10.05,
        frame=frame,
        lepton_telemetry=LeptonTelemetry(
            frame_counter=20,
            packet_timestamp_ms=5000,
            ffc_desired=False,
            ffc_state="complete",
            ffc_in_progress=False,
            since_last_ffc_s=30.0,
            tlinear_enabled=True,
            tlinear_resolution_k=0.01,
        ),
    )


def _projection(source_xy, thermal_uv):
    return ThermalProjectionResult(
        status="ok",
        source_depth_xy=source_xy,
        depth_m=0.5,
        color_xyz_m=(0.0, 0.0, 0.5),
        thermal_xyz_m=(0.0, 0.0, 0.5),
        thermal_uv=thermal_uv,
    )


def test_run_shadow_accepts_only_two_labeled_distinct_exact_winners(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        runner,
        "_load_stage0_contract",
        lambda _path: (_runtime_metadata(), "stage0-hash"),
    )
    monkeypatch.setattr(
        runner,
        "load_frozen_thermal_geometry",
        lambda _path: (
            np.eye(3),
            np.zeros(3),
            np.eye(3),
            np.zeros(5),
        ),
    )
    image_xy = np.zeros((21, 2), dtype=float)
    image_xy[3] = (0.48, 0.5)
    image_xy[4] = (0.5, 0.5)
    image_xy[7] = (0.53, 0.5)
    image_xy[8] = (0.55, 0.5)
    monkeypatch.setattr(
        runner.WebcamSource,
        "split_results",
        staticmethod(
            lambda _results: (
                (np.zeros((21, 3)), image_xy, np.eye(3)),
                None,
            )
        ),
    )

    def color_to_depth(*args):
        return [644.2, 352.2] if args[-1][0] == 640.0 else [700.2, 352.2]

    def project_to_color(_intrinsics, point):
        if point[0] == pytest.approx(644.2):
            return [640.1, 360.1]
        if point[0] == pytest.approx(700.2):
            return [704.1, 360.1]
        if point[0] == pytest.approx(644.0):
            return [640.9, 360.1]
        return [704.9, 360.1]

    rs = SimpleNamespace(
        rs2_project_color_pixel_to_depth_pixel=color_to_depth,
        rs2_deproject_pixel_to_point=lambda _intrinsics, pixel, depth: [
            pixel[0],
            pixel[1],
            depth,
        ],
        rs2_transform_point_to_point=lambda _extrinsics, point: point,
        rs2_project_point_to_pixel=project_to_color,
    )
    thumb = _projection((644, 352), (85.0, 63.0))
    index = _projection((700, 352), (90.0, 63.0))

    def sparse(**kwargs):
        assert kwargs["samples"] == [
            (644, 352, 500),
            (700, 352, 500),
        ]
        return SparseThermalMapResult(
            status="ok",
            winners=(((85, 63), thumb), ((90, 63), index)),
            rejections=(),
            input_count=2,
            accepted_count=2,
            rejected_count=0,
            collision_count=0,
        )

    monkeypatch.setattr(
        runner,
        "project_raw_depth_samples_to_sparse_thermal",
        sparse,
    )
    raw = _RawSource(_raw_sample())
    thermal = _ThermalSource(_thermal_sample())
    hands = _Hands()
    output = tmp_path / "hand-shadow.jsonl"

    summary = runner.run_shadow(
        attempts=1,
        output_path=output,
        rs_module=rs,
        raw_source_factory=lambda: raw,
        thermal_source_factory=lambda: thermal,
        hands_factory=lambda: hands,
        clock=lambda: 10.06,
        diagnose_inward_samples=True,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert summary == {
        "frame_events": 1,
        "depth_reused_events": 0,
        "attempted": 1,
        "fresh_depth_attempts": 1,
        "association_eligible_attempts": 1,
        "software_gate_accepted": 1,
        "blocked": 0,
    }
    assert rows[0]["safety_mode"] == "robot_free_hand_shadow_only"
    assert rows[0]["schema_version"] == 2
    assert rows[0]["sdk_match_error_metric"] == "euclidean_color_pixels"
    assert rows[0]["max_sdk_match_error_px"] == 0.75
    assert rows[0]["diagnose_inward_samples"] is True
    assert "pinch_signal_protocol" not in rows[0]
    assert "visualization_capture" not in rows[0]
    assert rows[1]["status"] == "software_gate_accepted"
    assert "frame_artifacts" not in rows[1]
    assert [tip["label"] for tip in rows[1]["fingertips"]] == [
        "thumb_tip",
        "index_tip",
    ]
    assert rows[1]["fingertips"][0]["thermal_pixel"] == [85, 63]
    assert rows[1]["fingertips"][0]["thermal_raw_count"] == 29234
    assert rows[1]["fingertips"][1]["thermal_pixel"] == [90, 63]
    assert rows[1]["fingertips"][1]["thermal_raw_count"] == 29345
    assert "thermal_frame_median_count" not in rows[1]
    assert "thermal_pinch_center_uv" not in rows[1]
    assert "thermal_patches_overlap" not in rows[1]
    assert (
        "thermal_patch_3x3_mean_count"
        not in rows[1]["fingertips"][0]
    )
    assert (
        rows[1]["fingertips"][0]["sdk_match_error_px"]
        <= rows[0]["max_sdk_match_error_px"]
    )
    assert "sdk_reprojected_color_uv" in rows[1]["fingertips"][0]
    assert "source_cell_reprojected_color_uv" in rows[1]["fingertips"][0]
    diagnostics = rows[1]["inward_association_diagnostics"]
    assert [
        (sample["label"], sample["inward_fraction"])
        for sample in diagnostics
    ] == [
        ("thumb_tip", 0.25),
        ("thumb_tip", 0.5),
        ("index_tip", 0.25),
        ("index_tip", 0.5),
    ]
    assert rows[1]["fingertips"][0]["normalized_color_xy"] == [0.5, 0.5]
    assert hands.seen == [raw._sample.color_rgb]
    assert hands.closed is True
    assert raw.stopped is True
    assert thermal.closed is True


def test_pinch_signal_run_starts_on_first_accepted_row_and_seals_completion(
    monkeypatch,
    tmp_path,
):
    _install_run_contract(monkeypatch)
    monkeypatch.setattr(runner, "_pair_rejection_reasons", lambda *_a, **_k: ())
    image_xy = np.full((21, 2), 0.5, dtype=float)
    image_xy[4] = (0.50, 0.50)
    image_xy[8] = (0.55, 0.50)
    image_xy[5] = (0.40, 0.50)
    image_xy[17] = (0.60, 0.50)
    monkeypatch.setattr(
        runner.WebcamSource,
        "split_results",
        staticmethod(
            lambda _results: (
                (np.zeros((21, 3)), image_xy, np.eye(3)),
                None,
            )
        ),
    )

    def associate(*, label, normalized_xy, **_kwargs):
        depth_pixel = [644, 352] if label == "thumb_tip" else [700, 352]
        return {
            "label": label,
            "status": "ok",
            "normalized_color_xy": list(normalized_xy),
            "depth_pixel": depth_pixel,
            "raw_depth": 500,
            "depth_m": 0.5,
        }

    monkeypatch.setattr(runner, "_associate_color_to_raw_depth", associate)
    thumb = _projection((644, 352), (85.0, 63.0))
    index = _projection((700, 352), (90.0, 63.0))
    monkeypatch.setattr(
        runner,
        "project_raw_depth_samples_to_sparse_thermal",
        lambda **_kwargs: SparseThermalMapResult(
            status="ok",
            winners=(((85, 63), thumb), ((90, 63), index)),
            rejections=(),
            input_count=2,
            accepted_count=2,
            rejected_count=0,
            collision_count=0,
        ),
    )
    depth_numbers = (8, 9, 9, 10, 11, 12, 13, 14, 15)
    raw_samples = [
        replace(
            _raw_sample(),
            color_frame_number=10 + index,
            depth_frame_number=depth_number,
        )
        for index, depth_number in enumerate(depth_numbers)
    ]
    thermal_samples = [
        replace(
            _thermal_sample(),
            lepton_telemetry=replace(
                _thermal_sample().lepton_telemetry,
                frame_counter=20 + index,
                packet_timestamp_ms=5000 + index,
            ),
        )
        for index in range(len(raw_samples))
    ]
    fake_cv2 = SimpleNamespace(destroyAllWindows=lambda: None)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setattr(runner, "PINCH_SIGNAL_GROUPS", 1)
    monkeypatch.setattr(runner, "PINCH_SIGNAL_TARGET_VALID_SAMPLES", 2)
    preview_keys = iter((32, -1, 32, -1, 32, -1, -1, -1))
    monkeypatch.setattr(
        runner,
        "_show_preview",
        lambda *_args, **_kwargs: next(preview_keys),
    )
    clock_values = iter(
        (10.0, 10.2, 10.3, 10.4, 10.6, 10.8, 11.0, 11.2, 11.4)
    )
    output = tmp_path / "pinch-signal.jsonl"
    artifact_calls = []

    class ArtifactWriter:
        @staticmethod
        def metadata():
            return {
                "experiment_id": "stage1e_tip_pinch_visualization_01",
                "experiment_role": (
                    "communication_only_descriptive_replication"
                ),
                "decision_authority": "none",
            }

        @staticmethod
        def capture(
            *,
            attempt_index,
            thermal_counts,
            color_rgb,
            depth_z16,
        ):
            artifact_calls.append(
                {
                    "attempt_index": attempt_index,
                    "thermal_shape": thermal_counts.shape,
                    "thermal_dtype": thermal_counts.dtype,
                    "color_is_current": any(
                        color_rgb is sample.color_rgb
                        for sample in raw_samples
                    ),
                    "depth_is_current": any(
                        depth_z16 is sample.depth_z16
                        for sample in raw_samples
                    ),
                }
            )
            return {
                "thermal_uint16": (
                    f"raw/thermal_uint16/attempt_{attempt_index:06d}.png"
                )
            }

    instruction_phases = []
    display_label_phases = []

    def instruction(cue):
        instruction_phases.append(cue["phase"])
        return runner._pinch_signal_instruction(cue)

    def display_label(cue):
        display_label_phases.append(cue["phase"])
        return cue["label"]

    summary = runner.run_shadow(
        attempts=10,
        output_path=output,
        rs_module=SimpleNamespace(),
        raw_source_factory=lambda: _SequenceRawSource(raw_samples),
        thermal_source_factory=lambda: _SequenceThermalSource(thermal_samples),
        hands_factory=_Hands,
        preview=True,
        manual_ffc_before_start=True,
        pinch_signal_trial=True,
        attempt_artifact_writer=ArtifactWriter(),
        pinch_signal_instruction=instruction,
        pinch_signal_display_label=display_label,
        clock=lambda: next(clock_values),
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[0]["schema_version"] == 4
    assert rows[0]["visualization_capture"] == {
        "experiment_id": "stage1e_tip_pinch_visualization_01",
        "experiment_role": "communication_only_descriptive_replication",
        "decision_authority": "none",
    }
    assert rows[0]["pinch_signal_protocol"]["groups"] == 1
    assert rows[0]["pinch_signal_protocol"]["target_valid_samples"] == 2
    assert rows[0]["pinch_signal_protocol"]["phase_timeout_s"] == 10.0
    assert rows[0]["pinch_signal_protocol"]["pinch_center_policy"] == (
        "diagnostic_only"
    )
    assert "max_center_shift_px" not in rows[0]["pinch_signal_protocol"]
    assert rows[1]["status"] == "software_gate_accepted"
    assert rows[1]["pinch_signal"]["phase"] == "prepare_just_touch"
    assert rows[1]["pinch_geometry"]["valid"] is True
    assert rows[2]["pinch_signal"]["phase"] == "record_just_touch"
    assert rows[2]["pinch_signal"]["valid_samples_after"] == 1
    assert rows[3]["event"] == "depth_reused"
    assert "frame_artifacts" not in rows[3]
    assert rows[3]["pinch_signal"]["quota_accepted"] is False
    assert rows[3]["pinch_signal"]["valid_samples"] == 1
    assert rows[4]["pinch_signal"]["phase"] == "record_just_touch"
    assert rows[4]["pinch_signal"]["valid_samples_after"] == 2
    assert rows[5]["pinch_signal"]["phase"] == "record_press_hard"
    assert rows[6]["pinch_signal"]["phase"] == "record_press_hard"
    assert rows[7]["pinch_signal"]["phase"] == "record_return_touch"
    assert rows[8]["pinch_signal"]["phase"] == "record_return_touch"
    assert rows[8]["pinch_signal"]["group_completed"] is True
    assert rows[8]["thermal_frame_median_count"] == 0.0
    assert rows[8]["thermal_pinch_center_uv"] == [87.5, 63.0]
    assert rows[8]["pinch_center_shift_px"] == pytest.approx(0.0)
    assert rows[8]["thermal_patches_overlap"] is False
    assert rows[8]["fingertips"][0][
        "thermal_patch_3x3_mean_count"
    ] == pytest.approx(29234.0 / 9.0)
    assert rows[9]["event"] == "pinch_signal_complete"
    assert rows[9]["pinch_signal"]["complete"] is True
    assert rows[10]["row_type"] == "summary"
    assert rows[10]["pinch_signal_started"] is True
    assert rows[10]["pinch_signal_protocol_completed"] is True
    assert rows[10]["pinch_signal_valid_groups"] == 1
    assert rows[10]["pinch_signal_invalid_groups"] == 0
    assert summary["attempted"] == 7
    assert summary["depth_reused_events"] == 1
    assert summary["pinch_signal_protocol_completed"] is True
    attempt_rows = [row for row in rows if row["row_type"] == "attempt"]
    assert [
        row["frame_artifacts"]["thermal_uint16"] for row in attempt_rows
    ] == [
        f"raw/thermal_uint16/attempt_{index:06d}.png"
        for index in range(7)
    ]
    assert artifact_calls == [
        {
            "attempt_index": index,
            "thermal_shape": (120, 160),
            "thermal_dtype": np.dtype(np.uint16),
            "color_is_current": True,
            "depth_is_current": True,
        }
        for index in range(7)
    ]
    assert "prepare_press_hard" in instruction_phases
    assert "record_press_hard" in instruction_phases
    assert "prepare_press_hard" in display_label_phases
    assert "record_press_hard" in display_label_phases


def _install_run_contract(monkeypatch):
    monkeypatch.setattr(
        runner,
        "_load_stage0_contract",
        lambda _path: (_runtime_metadata(), "stage0-hash"),
    )
    monkeypatch.setattr(
        runner,
        "load_frozen_thermal_geometry",
        lambda _path: (
            np.eye(3),
            np.zeros(3),
            np.eye(3),
            np.zeros(5),
        ),
    )


def test_pair_gate_blocks_before_mediapipe(monkeypatch, tmp_path):
    _install_run_contract(monkeypatch)
    monkeypatch.setattr(
        runner,
        "_pair_rejection_reasons",
        lambda *_args, **_kwargs: ("lepton_ffc_in_progress",),
    )
    raw = _RawSource(_raw_sample())
    thermal = _ThermalSource(_thermal_sample())
    hands = _Hands()
    output = tmp_path / "blocked.jsonl"

    summary = runner.run_shadow(
        attempts=1,
        output_path=output,
        rs_module=SimpleNamespace(),
        raw_source_factory=lambda: raw,
        thermal_source_factory=lambda: thermal,
        hands_factory=lambda: hands,
        clock=lambda: 10.06,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert summary["blocked"] == 1
    assert rows[1]["reasons"] == ["lepton_ffc_in_progress"]
    assert hands.seen == []
    assert hands.closed is True
    assert raw.stopped is True
    assert thermal.closed is True


def test_duplicate_depth_is_event_not_association_attempt(
    monkeypatch,
    tmp_path,
):
    _install_run_contract(monkeypatch)
    monkeypatch.setattr(
        runner.WebcamSource,
        "split_results",
        staticmethod(lambda _results: (None, None)),
    )
    raw_samples = [
        replace(
            _raw_sample(),
            color_frame_number=color_number,
            depth_frame_number=depth_number,
        )
        for color_number, depth_number in ((10, 8), (11, 8), (12, 9))
    ]
    thermal_samples = [
        replace(
            _thermal_sample(),
            lepton_telemetry=replace(
                _thermal_sample().lepton_telemetry,
                frame_counter=counter,
                packet_timestamp_ms=timestamp,
            ),
        )
        for counter, timestamp in ((20, 5000), (21, 5001), (22, 5002))
    ]
    hands = _Hands()
    output = tmp_path / "fresh-depth.jsonl"

    summary = runner.run_shadow(
        attempts=2,
        output_path=output,
        rs_module=SimpleNamespace(),
        raw_source_factory=lambda: _SequenceRawSource(raw_samples),
        thermal_source_factory=lambda: _SequenceThermalSource(thermal_samples),
        hands_factory=lambda: hands,
        clock=lambda: 10.06,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert summary == {
        "frame_events": 3,
        "depth_reused_events": 1,
        "attempted": 2,
        "fresh_depth_attempts": 2,
        "association_eligible_attempts": 2,
        "software_gate_accepted": 0,
        "blocked": 2,
    }
    assert [row["row_type"] for row in rows] == [
        "metadata",
        "attempt",
        "frame_event",
        "attempt",
    ]
    assert [rows[1]["attempt_index"], rows[3]["attempt_index"]] == [0, 1]
    assert rows[2]["event"] == "depth_reused"
    assert rows[2]["status"] == "not_attempted"
    assert rows[2]["reasons"] == ["d435_depth_frame_reused"]
    assert rows[2]["d435"]["depth_frame_number"] == 8
    assert len(hands.seen) == 2


def test_preview_shows_blocked_raw_frame_and_closes(monkeypatch, tmp_path):
    _install_run_contract(monkeypatch)
    monkeypatch.setattr(
        runner,
        "_pair_rejection_reasons",
        lambda *_args, **_kwargs: ("lepton_ffc_in_progress",),
    )
    calls = {"text": [], "imshow": [], "destroyed": 0}
    fake_cv2 = SimpleNamespace(
        COLOR_RGB2BGR=1,
        FONT_HERSHEY_SIMPLEX=2,
        LINE_AA=3,
        cvtColor=lambda image, _code: image.copy(),
        putText=lambda _image, text, *_args: calls["text"].append(text),
        circle=lambda *_args: None,
        imshow=lambda name, image: calls["imshow"].append(
            (name, image.shape)
        ),
        waitKey=lambda _delay: -1,
        destroyAllWindows=lambda: calls.__setitem__(
            "destroyed", calls["destroyed"] + 1
        ),
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    output = tmp_path / "preview.jsonl"

    runner.run_shadow(
        attempts=1,
        output_path=output,
        rs_module=SimpleNamespace(),
        raw_source_factory=lambda: _RawSource(_raw_sample()),
        thermal_source_factory=lambda: _ThermalSource(_thermal_sample()),
        hands_factory=_Hands,
        clock=lambda: 10.06,
        preview=True,
    )

    assert calls["imshow"] == [
        ("Stage 1D hand shadow preview", (720, 1280, 3))
    ]
    assert any("1/1 blocked" in text for text in calls["text"])
    assert any("lepton_ffc_in_progress" in text for text in calls["text"])
    assert "tip depth: unavailable" in calls["text"]
    assert calls["destroyed"] == 1


def test_preview_shows_actual_fingertip_depths():
    texts = []
    windows = []
    fake_cv2 = SimpleNamespace(
        COLOR_RGB2BGR=1,
        FONT_HERSHEY_SIMPLEX=2,
        LINE_AA=3,
        cvtColor=lambda image, _code: image.copy(),
        putText=lambda _image, text, *_args: texts.append(text),
        circle=lambda *_args: None,
        imshow=lambda name, _image: windows.append(name),
        waitKey=lambda _delay: -1,
    )

    runner._show_preview(
        fake_cv2,
        np.zeros((720, 1280, 3), dtype=np.uint8),
        {
            "attempt_index": 0,
            "status": "software_gate_accepted",
            "reasons": [],
            "pinch_center_shift_px": 1.2,
            "pinch_signal": {
                "complete": False,
                "blocked": False,
                "group_index": 0,
                "label": "press",
                "phase": "record_press_hard",
                "recording": True,
                "phase_remaining_s": 8.0,
                "valid_samples": 2,
                "target_valid_samples": 5,
                "quota_accepted": True,
                "quota_reasons": [],
                "valid_samples_after": 3,
            },
            "fingertips": [
                {
                    "label": "thumb_tip",
                    "status": "ok",
                    "normalized_color_xy": [0.25, 0.50],
                    "depth_m": 0.3234,
                },
                {
                    "label": "index_tip",
                    "status": "ok",
                    "normalized_color_xy": [0.50, 0.50],
                    "depth_m": 0.2914,
                },
            ],
        },
        attempts=1,
        pinch_signal_instruction=lambda _cue: (
            "HOLD JUST TOUCH - NO PRESS"
        ),
        pinch_signal_display_label=lambda _cue: "contact",
    )

    assert "tip depth: thumb=0.323 m index=0.291 m" in texts
    assert "none" in texts
    assert "HOLD JUST TOUCH - NO PRESS" in texts
    assert any(
        "contact valid 3/5 timeout 8.0s" in text for text in texts
    )
    assert windows == ["Stage 1E pinch signal preview"]


def test_missing_physical_right_hand_blocks_without_projection(
    monkeypatch,
    tmp_path,
):
    _install_run_contract(monkeypatch)
    monkeypatch.setattr(
        runner.WebcamSource,
        "split_results",
        staticmethod(lambda _results: (None, None)),
    )
    monkeypatch.setattr(
        runner,
        "project_raw_depth_samples_to_sparse_thermal",
        lambda **_kwargs: pytest.fail("projection must not run"),
    )
    raw = _RawSource(_raw_sample())
    thermal = _ThermalSource(_thermal_sample())
    hands = _Hands()
    output = tmp_path / "missing-hand.jsonl"

    summary = runner.run_shadow(
        attempts=1,
        output_path=output,
        rs_module=SimpleNamespace(),
        raw_source_factory=lambda: raw,
        thermal_source_factory=lambda: thermal,
        hands_factory=lambda: hands,
        clock=lambda: 10.06,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert summary["blocked"] == 1
    assert rows[1]["reasons"] == ["physical_right_hand_missing"]


def test_missing_right_hand_records_raw_mediapipe_detections(
    monkeypatch,
    tmp_path,
):
    _install_run_contract(monkeypatch)
    results = SimpleNamespace(
        multi_hand_landmarks=[object()],
        multi_hand_world_landmarks=[object()],
        multi_handedness=[
            SimpleNamespace(
                classification=[
                    SimpleNamespace(label="Right", score=0.91)
                ]
            )
        ],
    )
    monkeypatch.setattr(
        runner.WebcamSource,
        "split_results",
        staticmethod(lambda received: (None, None) if received is results else None),
    )
    hands = _Hands()
    hands.process = lambda _image: results
    output = tmp_path / "raw-mediapipe.jsonl"

    runner.run_shadow(
        attempts=1,
        output_path=output,
        rs_module=SimpleNamespace(),
        raw_source_factory=lambda: _RawSource(_raw_sample()),
        thermal_source_factory=lambda: _ThermalSource(_thermal_sample()),
        hands_factory=lambda: hands,
        clock=lambda: 10.06,
    )

    row = json.loads(output.read_text().splitlines()[1])
    assert row["mediapipe"] == {
        "image_hand_count": 1,
        "world_hand_count": 1,
        "handedness": [{"label": "Right", "score": 0.91}],
    }


def test_run_shadow_reports_periodic_terminal_progress(
    monkeypatch,
    tmp_path,
    capsys,
):
    _install_run_contract(monkeypatch)
    monkeypatch.setattr(
        runner.WebcamSource,
        "split_results",
        staticmethod(lambda _results: (None, None)),
    )
    monkeypatch.setattr(
        runner,
        "project_raw_depth_samples_to_sparse_thermal",
        lambda **_kwargs: pytest.fail("projection must not run"),
    )
    output = tmp_path / "progress.jsonl"
    raw_samples = [
        replace(
            _raw_sample(),
            color_frame_number=10 + index,
            depth_frame_number=8 + index,
        )
        for index in range(31)
    ]
    thermal_samples = [
        replace(
            _thermal_sample(),
            lepton_telemetry=replace(
                _thermal_sample().lepton_telemetry,
                frame_counter=20 + index,
                packet_timestamp_ms=5000 + index,
            ),
        )
        for index in range(31)
    ]

    summary = runner.run_shadow(
        attempts=31,
        output_path=output,
        rs_module=SimpleNamespace(),
        raw_source_factory=lambda: _SequenceRawSource(raw_samples),
        thermal_source_factory=lambda: _SequenceThermalSource(thermal_samples),
        hands_factory=_Hands,
        clock=lambda: 10.06,
    )

    assert summary["attempted"] == 31
    assert (
        "Stage 1D progress: 30/31 attempts; accepted=0 blocked=30"
        in capsys.readouterr().err
    )


@pytest.mark.parametrize(
    ("sparse", "reason"),
    [
        (
            SparseThermalMapResult(
                status="ok",
                winners=(((85, 63), _projection((644, 352), (85.0, 63.0))),),
                rejections=(),
                input_count=2,
                accepted_count=1,
                rejected_count=1,
                collision_count=0,
            ),
            "fingertip_sparse_pair_incomplete",
        ),
        (
            SparseThermalMapResult(
                status="ok",
                winners=(((85, 63), _projection((644, 352), (85.0, 63.0))),),
                rejections=(),
                input_count=2,
                accepted_count=2,
                rejected_count=0,
                collision_count=1,
            ),
            "fingertip_thermal_collision",
        ),
        (
            SparseThermalMapResult(
                status="ok",
                winners=(
                    ((85, 63), _projection((644, 352), (85.0, 63.0))),
                    ((90, 63), _projection((701, 352), (90.0, 63.0))),
                ),
                rejections=(),
                input_count=2,
                accepted_count=2,
                rejected_count=0,
                collision_count=0,
            ),
            "fingertip_sparse_identity_mismatch",
        ),
    ],
)
def test_sparse_pair_gate_blocks_partial_collision_or_identity_loss(
    sparse,
    reason,
):
    assert runner._sparse_pair_rejection_reason(
        sparse,
        ((644, 352), (700, 352)),
    ) == reason


def test_main_wires_only_robot_free_sources(monkeypatch, tmp_path, capsys):
    rs = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "pyrealsense2", rs)
    raw = object()
    thermal = object()
    hands = object()
    monkeypatch.setattr(
        runner,
        "RealSenseRawProjectorCamera",
        lambda *, rs_module: raw if rs_module is rs else None,
    )
    monkeypatch.setattr(
        runner,
        "LeptonUDPSource",
        lambda *, port: thermal if port == 9090 else None,
    )
    monkeypatch.setattr(runner, "_default_hands_factory", lambda: hands)
    captured = {}

    def run_shadow(**kwargs):
        captured.update(kwargs)
        return {
            "attempted": 1,
            "software_gate_accepted": 1,
            "blocked": 0,
        }

    monkeypatch.setattr(runner, "run_shadow", run_shadow)
    output = tmp_path / "hand-shadow.jsonl"

    assert runner.main(
        [
            "--frames",
            "1",
            "--output",
            str(output),
            "--lepton-port",
            "9090",
        ]
    ) == 0
    assert captured["attempts"] == 1
    assert captured["output_path"] == output
    assert captured["rs_module"] is rs
    assert captured["raw_source_factory"]() is raw
    assert captured["thermal_source_factory"]() is thermal
    assert captured["hands_factory"]() is hands
    assert captured["preview"] is False
    assert captured["manual_ffc_before_start"] is False
    stderr = capsys.readouterr().err
    assert "headless JSONL-only; no preview window" in stderr
    assert "Show the physical RIGHT hand" in stderr
    assert f"Collecting 1 attempts -> {output}" in stderr


def test_main_runs_requested_manual_ffc_before_capture(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setitem(sys.modules, "pyrealsense2", SimpleNamespace())
    events = []
    monkeypatch.setattr(
        runner,
        "_run_manual_ffc",
        lambda: events.append("ffc") or "Manual FFC complete\n",
    )

    def run_shadow(**kwargs):
        events.append("capture")
        assert kwargs["manual_ffc_before_start"] is True
        return {
            "attempted": 1,
            "software_gate_accepted": 1,
            "blocked": 0,
        }

    monkeypatch.setattr(runner, "run_shadow", run_shadow)

    assert runner.main(
        [
            "--frames",
            "1",
            "--output",
            str(tmp_path / "manual-ffc.jsonl"),
            "--manual-ffc",
        ]
    ) == 0
    assert events == ["ffc", "capture"]
