import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import live_lepton_projector_shadow as runner
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


def _runtime_metadata():
    return {
        "requested": {
            "serial": "233522078685",
            "color": {
                "width": 1280,
                "height": 720,
                "format": "rgb8",
                "fps": 15,
            },
            "depth": {
                "width": 1280,
                "height": 720,
                "format": "z16",
                "fps": 6,
            },
        },
        "sdk_version": "2.58.3",
        "device": {
            "serial": "233522078685",
            "name": "RealSense D435I",
            "firmware": "5.16.0.1",
            "product_line": "D400",
        },
        "resolved": {
            "color": {"format": "format.rgb8", "fps": 15},
            "depth": {"format": "format.z16", "fps": 6},
        },
        "color_intrinsics": {"width": 1280, "model": "inverse_brown"},
        "depth_intrinsics": {"width": 1280, "model": "brown"},
        "depth_scale_m": 0.001,
        "factory_extrinsics": {
            "depth_to_color": {"rotation": [1.0], "translation_m": [0.0]},
            "color_to_depth": {"rotation": [1.0], "translation_m": [0.0]},
        },
    }


def _raw_sample(
    *,
    observed_at_s=10.0,
    color_frame_number=10,
    depth_frame_number=8,
):
    depth = np.zeros((720, 1280), dtype=np.uint16)
    depth[352, 644] = 500
    return RealSenseRawSample(
        color_rgb=np.zeros((720, 1280, 3), dtype=np.uint8),
        depth_z16=depth,
        observed_at_s=observed_at_s,
        color_frame_number=color_frame_number,
        depth_frame_number=depth_frame_number,
        color_timestamp_ms=100.0,
        depth_timestamp_ms=98.0,
        color_timestamp_domain="timestamp_domain.global_time",
        depth_timestamp_domain="timestamp_domain.global_time",
    )


def _telemetry(**changes):
    values = {
        "frame_counter": 20,
        "packet_timestamp_ms": 5000,
        "ffc_desired": False,
        "ffc_state": "complete",
        "ffc_in_progress": False,
        "since_last_ffc_s": 30.0,
        "tlinear_enabled": True,
        "tlinear_resolution_k": 0.01,
    }
    values.update(changes)
    return LeptonTelemetry(**values)


_DEFAULT_TELEMETRY = object()


def _thermal_sample(*, observed_at_s=10.05, telemetry=_DEFAULT_TELEMETRY):
    frame = np.zeros((120, 160), dtype=np.uint16)
    frame[63, 85] = 29234
    return FrameSample(
        t=observed_at_s,
        frame=frame,
        lepton_telemetry=(
            _telemetry() if telemetry is _DEFAULT_TELEMETRY else telemetry
        ),
    )


class _RawSource:
    def __init__(self, samples, *, metadata=None):
        self.samples = iter(samples)
        self.runtime_metadata = metadata or _runtime_metadata()
        self.depth_intrinsics = SimpleNamespace()
        self.depth_to_color_extrinsics = SimpleNamespace()
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def read(self):
        return next(self.samples)

    def stop(self):
        self.stopped = True


class _ThermalSource:
    def __init__(self, samples):
        self.samples = iter(samples)
        self.closed = False

    def read(self):
        return next(self.samples)

    def close(self):
        self.closed = True


def _install_contract(monkeypatch):
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


def _install_projector(monkeypatch):
    projected = ThermalProjectionResult(
        status="ok",
        source_depth_xy=(644, 352),
        depth_m=0.5,
        color_xyz_m=(0.0, 0.0, 0.5),
        thermal_xyz_m=(0.0, 0.0, 0.5),
        thermal_uv=(85.0, 63.0),
    )

    def fake_sparse(**kwargs):
        assert kwargs["samples"] == [(644, 352, 500)]
        return SparseThermalMapResult(
            status="ok",
            winners=(((85, 63), projected),),
            rejections=(),
            input_count=1,
            accepted_count=1,
            rejected_count=0,
            collision_count=0,
        )

    monkeypatch.setattr(
        runner,
        "project_raw_depth_samples_to_sparse_thermal",
        fake_sparse,
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["--frames", "1", "--output", "out.jsonl"],
        ["--depth-pixel", "-1,0", "--frames", "1", "--output", "out.jsonl"],
        ["--depth-pixel", "1280,0", "--frames", "1", "--output", "out.jsonl"],
        ["--depth-pixel", "0,720", "--frames", "1", "--output", "out.jsonl"],
        ["--depth-pixel", "0,0", "--frames", "0", "--output", "out.jsonl"],
        ["--depth-pixel", "0,0", "--frames", "101", "--output", "out.jsonl"],
    ],
)
def test_cli_rejects_unbounded_or_out_of_domain_inputs(argv):
    with pytest.raises(SystemExit, match="2"):
        runner.parse_args(argv)


def test_successful_attempt_writes_metadata_and_native_thermal_count(
    monkeypatch,
    tmp_path,
):
    _install_contract(monkeypatch)
    _install_projector(monkeypatch)
    raw = _RawSource([_raw_sample()])
    thermal = _ThermalSource([_thermal_sample()])
    output = tmp_path / "shadow.jsonl"

    summary = runner.run_shadow(
        depth_pixels=((644, 352),),
        attempts=1,
        output_path=output,
        rs_module=SimpleNamespace(),
        raw_source_factory=lambda: raw,
        thermal_source_factory=lambda: thermal,
        clock=lambda: 10.06,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert summary == {"attempted": 1, "software_gate_accepted": 1, "blocked": 0}
    assert rows[0]["row_type"] == "metadata"
    assert rows[0]["safety_mode"] == "robot_free_shadow_only"
    assert rows[0]["metadata_comparison"] == "exact_normalized_equality"
    assert rows[0]["max_attempts"] == 1
    assert rows[1]["status"] == "software_gate_accepted"
    assert rows[1]["requested_depth"][0]["raw_depth"] == 500
    assert rows[1]["winners"][0]["thermal_pixel"] == [85, 63]
    assert rows[1]["winners"][0]["thermal_raw_count"] == 29234
    assert rows[1]["limitations"] == [
        "color_thermal_source_time_not_comparable"
    ]
    assert raw.stopped is True
    assert thermal.closed is True


def test_blocked_repeat_consumes_attempt_budget_without_retry(monkeypatch, tmp_path):
    _install_contract(monkeypatch)
    _install_projector(monkeypatch)
    raw_sample = _raw_sample()
    thermal_sample = _thermal_sample()
    raw = _RawSource([raw_sample, raw_sample])
    thermal = _ThermalSource([thermal_sample, thermal_sample])
    output = tmp_path / "shadow.jsonl"

    summary = runner.run_shadow(
        depth_pixels=((644, 352),),
        attempts=2,
        output_path=output,
        rs_module=SimpleNamespace(),
        raw_source_factory=lambda: raw,
        thermal_source_factory=lambda: thermal,
        clock=lambda: 10.06,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert summary == {"attempted": 2, "software_gate_accepted": 1, "blocked": 1}
    assert len(rows) == 3
    assert rows[2]["status"] == "blocked"
    assert "d435_color_frame_non_increasing" in rows[2]["reasons"]
    assert "d435_depth_frame_non_increasing" in rows[2]["reasons"]
    assert "lepton_frame_counter_non_increasing" in rows[2]["reasons"]
    assert "lepton_packet_timestamp_non_increasing" in rows[2]["reasons"]


def test_raw_identity_is_not_reused_after_missing_lepton_telemetry():
    raw = _raw_sample()
    prior = runner._pair_state(
        raw,
        _thermal_sample(telemetry=None),
    )

    reasons = runner._pair_rejection_reasons(
        raw,
        _thermal_sample(),
        now_s=10.06,
        prior=prior,
    )

    assert "d435_color_frame_non_increasing" in reasons
    assert "d435_depth_frame_non_increasing" in reasons


def test_regressed_identity_cannot_lower_the_future_repeat_bar():
    prior = runner._PairState(
        color_frame_number=10,
        depth_frame_number=8,
        lepton_frame_counter=20,
        lepton_packet_timestamp_ms=5000,
    )
    regressed = runner._PairState(
        color_frame_number=9,
        depth_frame_number=7,
        lepton_frame_counter=19,
        lepton_packet_timestamp_ms=4999,
    )

    retained = runner._advance_pair_state(prior, regressed)

    assert retained == prior


@pytest.mark.parametrize(
    ("raw", "thermal", "now_s", "prior", "reason"),
    [
        (
            _raw_sample(observed_at_s=9.0),
            _thermal_sample(),
            10.06,
            None,
            "d435_host_read_completion_stale",
        ),
        (
            _raw_sample(observed_at_s=9.8),
            _thermal_sample(observed_at_s=10.05),
            10.06,
            None,
            "host_read_completion_skew_exceeded",
        ),
        (
            _raw_sample(),
            _thermal_sample(telemetry=None),
            10.06,
            None,
            "lepton_telemetry_missing",
        ),
        (
            _raw_sample(),
            _thermal_sample(telemetry=_telemetry(ffc_state="in_progress")),
            10.06,
            None,
            "lepton_ffc_not_idle",
        ),
        (
            _raw_sample(),
            _thermal_sample(telemetry=_telemetry(ffc_state="unknown")),
            10.06,
            None,
            "lepton_ffc_not_idle",
        ),
        (
            _raw_sample(),
            _thermal_sample(telemetry=_telemetry(ffc_desired=True)),
            10.06,
            None,
            "lepton_ffc_desired",
        ),
        (
            _raw_sample(),
            _thermal_sample(telemetry=_telemetry(ffc_in_progress=True)),
            10.06,
            None,
            "lepton_ffc_in_progress",
        ),
        (
            _raw_sample(),
            _thermal_sample(telemetry=_telemetry(tlinear_enabled=False)),
            10.06,
            None,
            "lepton_tlinear_invalid",
        ),
        (
            _raw_sample(),
            _thermal_sample(
                telemetry=_telemetry(tlinear_resolution_k=0.1)
            ),
            10.06,
            None,
            "lepton_tlinear_invalid",
        ),
    ],
)
def test_pair_gate_rejects_each_provisional_or_telemetry_violation(
    raw,
    thermal,
    now_s,
    prior,
    reason,
):
    reasons = runner._pair_rejection_reasons(
        raw,
        thermal,
        now_s=now_s,
        prior=prior,
    )
    assert reason in reasons


def test_pair_gate_does_not_compare_cross_device_source_timestamps():
    raw = replace(
        _raw_sample(),
        color_timestamp_ms=1.0e12,
        depth_timestamp_ms=1.0e12,
    )
    thermal = _thermal_sample(
        telemetry=_telemetry(packet_timestamp_ms=1)
    )

    assert runner._pair_rejection_reasons(
        raw,
        thermal,
        now_s=10.06,
        prior=None,
    ) == ()


def test_metadata_mismatch_fails_before_projection_and_closes_sources(
    monkeypatch,
    tmp_path,
):
    _install_contract(monkeypatch)
    monkeypatch.setattr(
        runner,
        "project_raw_depth_samples_to_sparse_thermal",
        lambda **_kwargs: pytest.fail("projection must not run"),
    )
    bad_metadata = _runtime_metadata()
    bad_metadata["device"] = {
        **bad_metadata["device"],
        "serial": "wrong",
    }
    raw = _RawSource([_raw_sample()], metadata=bad_metadata)
    thermal = _ThermalSource([_thermal_sample()])
    output = tmp_path / "blocked.jsonl"

    with pytest.raises(ValueError, match="Stage 0 runtime metadata mismatch"):
        runner.run_shadow(
            depth_pixels=((644, 352),),
            attempts=1,
            output_path=output,
            rs_module=SimpleNamespace(),
            raw_source_factory=lambda: raw,
            thermal_source_factory=lambda: thermal,
            clock=lambda: 10.06,
        )

    row = json.loads(output.read_text())
    assert row["status"] == "setup_blocked"
    assert raw.stopped is True
    assert thermal.closed is False


def test_raw_start_failure_is_recorded_and_source_is_closed(monkeypatch, tmp_path):
    _install_contract(monkeypatch)
    raw = _RawSource([])
    raw.start = lambda: (_ for _ in ()).throw(
        RuntimeError("device permission denied")
    )
    output = tmp_path / "setup-blocked.jsonl"

    with pytest.raises(RuntimeError, match="device permission denied"):
        runner.run_shadow(
            depth_pixels=((644, 352),),
            attempts=1,
            output_path=output,
            rs_module=SimpleNamespace(),
            raw_source_factory=lambda: raw,
            thermal_source_factory=lambda: pytest.fail(
                "thermal source must not open"
            ),
            clock=lambda: 10.06,
        )

    row = json.loads(output.read_text())
    assert row["status"] == "setup_blocked"
    assert row["reason"] == "raw_d435i_start_failed"
    assert "device permission denied" in row["error"]
    assert raw.stopped is True


def test_existing_output_fails_before_opening_sources(monkeypatch, tmp_path):
    _install_contract(monkeypatch)
    output = tmp_path / "existing.jsonl"
    output.write_text("preserve\n")

    with pytest.raises(FileExistsError):
        runner.run_shadow(
            depth_pixels=((644, 352),),
            attempts=1,
            output_path=output,
            rs_module=SimpleNamespace(),
            raw_source_factory=lambda: pytest.fail("raw source must not open"),
            thermal_source_factory=lambda: pytest.fail(
                "thermal source must not open"
            ),
            clock=lambda: 10.06,
        )

    assert output.read_text() == "preserve\n"


def test_interruption_closes_both_sources(monkeypatch, tmp_path):
    _install_contract(monkeypatch)
    raw = _RawSource([_raw_sample()])
    thermal = _ThermalSource([_thermal_sample()])
    raw.read = lambda: (_ for _ in ()).throw(KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        runner.run_shadow(
            depth_pixels=((644, 352),),
            attempts=1,
            output_path=tmp_path / "interrupted.jsonl",
            rs_module=SimpleNamespace(),
            raw_source_factory=lambda: raw,
            thermal_source_factory=lambda: thermal,
            clock=lambda: 10.06,
        )

    assert raw.stopped is True
    assert thermal.closed is True
