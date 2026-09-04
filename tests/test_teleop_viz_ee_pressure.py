from pathlib import Path
import inspect
import json
import mmap
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.modules.setdefault("mediapipe", SimpleNamespace(solutions=SimpleNamespace()))

from experiments import teleop_viz_ee
from ir_force.data_paths import CHECKOUT_ROOT
from ir_force.ir_hand_calibration import (
    ProjectionCalibration,
    save_projection_calibration,
)
from pressurevision_integration.pv_object_profile import (
    PressureVisionObjectProfile,
    save_object_profile,
)
from pressurevision_integration.pv_preview import (
    PREVIEW_HEADER,
    PREVIEW_HEADER_SIZE,
    PREVIEW_MAGIC,
    PressureVisionPreviewSource,
    draw_gripper_position_banner,
    format_gripper_positions,
)


def _write_preview(path, frame, *, sequence=2, observed_at_s=10.0):
    with path.open("w+b") as handle:
        handle.truncate(PREVIEW_HEADER_SIZE + frame.nbytes)
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_WRITE)
        mapped[:PREVIEW_HEADER.size] = PREVIEW_HEADER.pack(
            PREVIEW_MAGIC,
            sequence,
            observed_at_s,
            frame.shape[0],
            frame.shape[1],
            frame.shape[2],
            frame.nbytes,
        )
        mapped[PREVIEW_HEADER_SIZE:] = frame.tobytes()
        mapped.close()


def _pv_apply_gate_args(evidence_dir: Path) -> list[str]:
    return [
        "--oak",
        "--pv-evidence-dir", str(evidence_dir),
        "--pv-max-load", "240",
        "--pv-max-current", "35",
        "--pv-max-position-lag", "5.0",
    ]


def test_wrist_roll_range_is_explicit_and_bounded():
    args = teleop_viz_ee.parse_live_args([
        "--wrist-roll-range-deg", "30",
        "--wrist-roll-gain", "2",
    ])

    assert args.wrist_roll_range_deg == 30.0
    assert args.wrist_roll_gain == 2.0
    with pytest.raises(SystemExit, match="2"):
        teleop_viz_ee.parse_live_args(["--wrist-roll-range-deg", "46"])
    with pytest.raises(SystemExit, match="2"):
        teleop_viz_ee.parse_live_args(["--wrist-roll-gain", "5"])


def test_startup_hand_gate_requires_three_continuous_seconds():
    gate = teleop_viz_ee.ContinuousHandStartupGate()

    assert gate.update(hand_valid=True, observed_at_s=10.0) == 0.0
    assert gate.update(hand_valid=True, observed_at_s=12.9) == pytest.approx(2.9)
    assert gate.update(hand_valid=False, observed_at_s=12.95) == 0.0
    assert gate.update(hand_valid=True, observed_at_s=20.0) == 0.0
    assert gate.update(hand_valid=True, observed_at_s=23.0) == pytest.approx(3.0)


def test_live_hand_gate_follows_connection_and_precedes_first_motion():
    source = inspect.getsource(teleop_viz_ee._run_live)

    gate_passed = source.index("elapsed_s >= HAND_STARTUP_DWELL_S")
    robot_created = source.index("robot = SOFollower")
    first_motion = source.index("robot.send_action")

    assert robot_created < gate_passed < first_motion


def test_pv_preview_source_reads_fresh_complete_frame_and_rejects_stale(tmp_path):
    path = tmp_path / "pv-preview.mmap"
    frame = np.arange(4 * 6 * 3, dtype=np.uint8).reshape((4, 6, 3))
    _write_preview(path, frame)
    source = PressureVisionPreviewSource(path, stale_after_s=0.75)
    try:
        assert np.array_equal(source.read(now_s=10.5), frame)
        assert source.read(now_s=11.0) is None
    finally:
        source.close()


def test_pv_preview_source_rejects_frame_being_written(tmp_path):
    path = tmp_path / "pv-preview.mmap"
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    _write_preview(path, frame, sequence=3)
    source = PressureVisionPreviewSource(path)
    try:
        assert source.read(now_s=10.1) is None
    finally:
        source.close()


def test_operator_view_has_stable_equal_width_panes():
    hand = np.full((120, 160, 3), 7, dtype=np.uint8)
    pv = np.full((60, 180, 3), 19, dtype=np.uint8)

    combined = teleop_viz_ee.compose_operator_view(hand, pv)
    waiting = teleop_viz_ee.compose_operator_view(hand, None)

    assert combined.shape == (120, 320, 3)
    assert waiting.shape == combined.shape
    assert np.array_equal(combined[:, :160], hand)


def test_gripper_position_banner_shows_command_readback_and_missing_values():
    assert format_gripper_positions(20.04, 22.66) == "CMD q=20.0    OBS q=22.7"
    assert format_gripper_positions(32.0, None) == "CMD q=32.0    OBS q=--"

    frame = np.zeros((160, 900, 3), dtype=np.uint8)
    rendered = draw_gripper_position_banner(
        frame,
        commanded=20.04,
        observed=22.66,
    )
    assert rendered is frame
    assert np.any(frame[52:126, 10:760])


def test_build_ir_pressure_source_returns_none_when_disabled(tmp_path: Path):
    assert teleop_viz_ee.build_ir_pressure_source(
        enabled=False,
        calibration_path=str(tmp_path / "missing.json"),
        thermal_path="/dev/video21",
    ) is None


def test_default_ir_calibration_resolves_inside_this_repo():
    """The calibration moved to calibration/flir_oak/ when the repo split; what
    still matters is that the default never points outside this checkout."""
    repo = Path(__file__).resolve().parents[1]
    default = Path(teleop_viz_ee.DEFAULT_IR_CALIBRATION)

    assert default == repo / "calibration" / "flir_oak" / "oak_flir_hand_pressure_projection.json"
    assert repo in default.parents


def test_build_ir_pressure_source_returns_none_when_calibration_missing(tmp_path: Path):
    assert teleop_viz_ee.build_ir_pressure_source(
        enabled=True,
        calibration_path=str(tmp_path / "missing.json"),
        thermal_path="/dev/video21",
    ) is None


def test_build_ir_pressure_source_loads_calibration_when_present(tmp_path: Path, monkeypatch):
    path = tmp_path / "projection.json"
    save_projection_calibration(
        path,
        ProjectionCalibration((0.0, 160.0, 0.0, 0.0), (0.0, 0.0, 128.0, 0.0), 1.0, 2.0, 12),
    )

    class FakeThermal:
        def __init__(self, thermal_path):
            self.thermal_path = thermal_path

    monkeypatch.setattr(teleop_viz_ee, "OpenCVCameraSource", FakeThermal)

    source = teleop_viz_ee.build_ir_pressure_source(
        enabled=True,
        calibration_path=str(path),
        thermal_path="/dev/video21",
    )

    assert source is not None
    assert source.calibration.sample_count == 12


def test_build_ir_pressure_source_returns_none_for_malformed_calibration(tmp_path: Path, capsys):
    path = tmp_path / "projection.json"
    path.write_text("{not-json\n", encoding="utf-8")

    source = teleop_viz_ee.build_ir_pressure_source(
        enabled=True,
        calibration_path=str(path),
        thermal_path="/dev/video21",
    )

    assert source is None
    assert "failed to load calibration" in capsys.readouterr().out


def test_build_ir_pressure_source_returns_none_for_unsupported_calibration_version(
    tmp_path: Path,
    capsys,
):
    path = tmp_path / "projection.json"
    path.write_text(
        '{"coeff_x":[0,160,0,0],"coeff_y":[0,0,128,0],"rms_error_px":1.0,'
        '"max_error_px":2.0,"sample_count":12,"image_size":[160,128],"version":2}\n',
        encoding="utf-8",
    )

    source = teleop_viz_ee.build_ir_pressure_source(
        enabled=True,
        calibration_path=str(path),
        thermal_path="/dev/video21",
    )

    assert source is None
    assert "failed to load calibration" in capsys.readouterr().out


def test_build_ir_pressure_source_returns_none_when_thermal_source_init_fails(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    path = tmp_path / "projection.json"
    save_projection_calibration(
        path,
        ProjectionCalibration((0.0, 160.0, 0.0, 0.0), (0.0, 0.0, 128.0, 0.0), 1.0, 2.0, 12),
    )

    def fail(_thermal_path):
        raise RuntimeError("device unavailable")

    monkeypatch.setattr(teleop_viz_ee, "OpenCVCameraSource", fail)

    source = teleop_viz_ee.build_ir_pressure_source(
        enabled=True,
        calibration_path=str(path),
        thermal_path="/dev/video21",
    )

    assert source is None
    assert "failed to open thermal source" in capsys.readouterr().out


def test_build_ir_pressure_source_closes_latest_thermal_when_estimator_init_fails(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    path = tmp_path / "projection.json"
    save_projection_calibration(
        path,
        ProjectionCalibration((0.0, 160.0, 0.0, 0.0), (0.0, 0.0, 128.0, 0.0), 1.0, 2.0, 12),
    )
    closed = []
    latest = SimpleNamespace(close=lambda: closed.append("latest"))
    monkeypatch.setattr(teleop_viz_ee, "OpenCVCameraSource", lambda _path: object())
    monkeypatch.setattr(teleop_viz_ee, "LatestFrameSource", lambda _source: latest)
    monkeypatch.setattr(
        teleop_viz_ee,
        "HandPressureEstimator",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("estimator failed")),
    )

    source = teleop_viz_ee.build_ir_pressure_source(
        enabled=True,
        calibration_path=str(path),
        thermal_path="/dev/video21",
    )

    assert source is None
    assert closed == ["latest"]
    assert "failed to open thermal source" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("rms_error_px", "max_error_px"),
    [(9.0, 10.0), (2.0, 17.0)],
)
def test_build_ir_pressure_source_rejects_poor_projection_quality(
    tmp_path: Path,
    monkeypatch,
    capsys,
    rms_error_px,
    max_error_px,
):
    path = tmp_path / "projection.json"
    save_projection_calibration(
        path,
        ProjectionCalibration(
            (0.0, 160.0, 0.0, 0.0),
            (0.0, 0.0, 128.0, 0.0),
            rms_error_px,
            max_error_px,
            12,
        ),
    )

    class FakeThermal:
        def __init__(self, thermal_path):
            self.thermal_path = thermal_path

    monkeypatch.setattr(teleop_viz_ee, "OpenCVCameraSource", FakeThermal)

    source = teleop_viz_ee.build_ir_pressure_source(
        enabled=True,
        calibration_path=str(path),
        thermal_path="/dev/video21",
    )

    assert source is None
    assert "calibration rejected" in capsys.readouterr().out


def test_build_ir_pressure_source_rejects_too_few_runtime_samples(tmp_path: Path, monkeypatch, capsys):
    path = tmp_path / "projection.json"
    save_projection_calibration(
        path,
        ProjectionCalibration((0.0, 160.0, 0.0, 0.0), (0.0, 0.0, 128.0, 0.0), 1.0, 2.0, 6),
    )

    monkeypatch.setattr(teleop_viz_ee, "OpenCVCameraSource", lambda _path: object())

    source = teleop_viz_ee.build_ir_pressure_source(
        enabled=True,
        calibration_path=str(path),
        thermal_path="/dev/video21",
    )

    assert source is None
    assert "calibration rejected" in capsys.readouterr().out


def test_close_live_resources_releases_everything_when_camera_close_raises(monkeypatch):
    calls = []

    def release_cam():
        calls.append("camera")
        raise RuntimeError("camera close failed")

    hands = SimpleNamespace(close=lambda: calls.append("hands"))
    controller = SimpleNamespace(close=lambda: calls.append("controller"))
    robot = SimpleNamespace(disconnect=lambda: calls.append("robot"))
    monkeypatch.setattr(teleop_viz_ee.cv2, "destroyAllWindows", lambda: calls.append("windows"))

    with pytest.raises(RuntimeError, match="camera close failed"):
        teleop_viz_ee.close_live_resources(release_cam, hands, controller, robot)

    assert calls == ["camera", "hands", "controller", "windows", "robot"]


def test_live_main_setup_failure_closes_all_constructed_resources(monkeypatch):
    calls = []

    class FakeRobot:
        def __init__(self, _config):
            self.bus = SimpleNamespace(
                # norm_mode is read to resolve the gripper's calibrated centre,
                # which seeds the pressure runtime.
                motors={
                    "gripper": SimpleNamespace(
                        norm_mode=SimpleNamespace(value="range_0_100")
                    )
                },
                is_connected=True,
            )
            self.cameras = {}
            self.is_connected = True

        def connect(self, calibrate=False):
            calls.append("robot-connect")

        def send_action(self, _action):
            return None

        def disconnect(self):
            if self.is_connected:
                calls.append("robot-close")
                self.is_connected = False
                self.bus.is_connected = False

    class FakeKinematics:
        def __init__(self, **_kwargs):
            pass

        def forward_kinematics(self, _joints):
            import numpy as np

            return np.eye(4)

    pressure_source = SimpleNamespace(close=lambda: calls.append("pressure-close"))
    sidecar = SimpleNamespace(close=lambda: calls.append("sidecar-close"))

    class FakeController:
        def __init__(self, *_args, **_kwargs):
            self.middle_pose = {"gripper.pos": 50.0}
            self.r_down = [0.0, 0.0, 0.0]

        def build(self, _centre):
            pass

        def seed(self, _positions):
            pass

        def close(self):
            calls.append("controller-close")

    class FakeOAKDepth:
        def __init__(self, **_kwargs):
            pass

        def update_depth(self, _depth):
            pass

    class FakeOAKCamera:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            calls.append("camera-start")

        def stop(self):
            calls.append("camera-close")

    hands = SimpleNamespace(close=lambda: calls.append("hands-close"))
    args = SimpleNamespace(
        oak=True,
        ir_pressure=False,
        ir_pressure_shadow=True,
        ir_sidecar="unused.csv",
        ir_calibration="unused.json",
        thermal="unused",
        grip_mode="tracked",
        grip_map="overdrive",
    )
    monkeypatch.setattr(teleop_viz_ee, "parse_live_args", lambda: args)
    monkeypatch.setattr(teleop_viz_ee, "SOFollower", FakeRobot)
    monkeypatch.setattr(teleop_viz_ee, "RobotKinematics", FakeKinematics)
    monkeypatch.setattr(
        teleop_viz_ee,
        "build_live_ir_runtime",
        lambda _args: teleop_viz_ee.LiveIRRuntime(pressure_source, True, sidecar),
    )
    monkeypatch.setattr(teleop_viz_ee, "WebcamEEController", FakeController)
    monkeypatch.setattr(
        teleop_viz_ee,
        "read_positions",
        lambda _robot: {"gripper.pos": 50.0},
    )
    monkeypatch.setattr("webcam_input.depth.OAKDepthStrategy", FakeOAKDepth)
    monkeypatch.setattr("webcam_input.oak_camera.OAKCamera", FakeOAKCamera)
    monkeypatch.setattr(
        teleop_viz_ee.mp.solutions,
        "hands",
        SimpleNamespace(Hands=lambda **_kwargs: hands, HAND_CONNECTIONS=object()),
        raising=False,
    )
    monkeypatch.setattr(
        teleop_viz_ee.mp.solutions,
        "drawing_utils",
        SimpleNamespace(),
        raising=False,
    )
    monkeypatch.setattr(
        teleop_viz_ee.cv2,
        "destroyAllWindows",
        lambda: calls.append("windows-close"),
    )
    monkeypatch.setattr(
        teleop_viz_ee.cv2,
        "namedWindow",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("window failed")),
    )
    monkeypatch.setattr(teleop_viz_ee.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="window failed"):
        teleop_viz_ee.main()

    for expected in (
        "robot-close",
        "pressure-close",
        "sidecar-close",
        "controller-close",
        "camera-close",
        "hands-close",
        "windows-close",
    ):
        assert expected in calls


def test_live_flags_reject_apply_and_shadow_together():
    with pytest.raises(SystemExit):
        teleop_viz_ee.parse_live_args(["--ir-pressure", "--ir-pressure-shadow"])


def test_live_flags_reject_sidecar_without_shadow(tmp_path: Path):
    with pytest.raises(SystemExit):
        teleop_viz_ee.parse_live_args(["--oak", "--ir-sidecar", str(tmp_path / "shadow.csv")])


def test_live_shadow_mode_requires_oak_projection_input():
    with pytest.raises(SystemExit, match="2"):
        teleop_viz_ee.parse_live_args(["--ir-pressure-shadow"])


def test_live_apply_mode_is_disabled_until_stage3(capsys):
    with pytest.raises(SystemExit, match="2"):
        teleop_viz_ee.parse_live_args(["--oak", "--ir-pressure"])

    assert "disabled until Stage 3" in capsys.readouterr().err


def test_live_shadow_runtime_constructs_pressure_source_and_sidecar(tmp_path: Path, monkeypatch):
    seen = {}
    sidecar_path = tmp_path / "shadow.csv"

    def fake_builder(*, enabled, calibration_path, thermal_path, lepton_port=None):
        seen.update(
            enabled=enabled,
            calibration_path=calibration_path,
            thermal_path=thermal_path,
            lepton_port=lepton_port,
        )
        return "pressure-source"

    monkeypatch.setattr(teleop_viz_ee, "build_ir_pressure_source", fake_builder)
    args = teleop_viz_ee.parse_live_args([
        "--oak",
        "--ir-pressure-shadow",
        "--ir-sidecar",
        str(sidecar_path),
    ])

    runtime = teleop_viz_ee.build_live_ir_runtime(args)

    assert runtime.pressure_source == "pressure-source"
    assert runtime.pressure_shadow is True
    assert runtime.sidecar is not None
    assert seen["enabled"] is True
    runtime.sidecar.close()


def test_live_shadow_runtime_rejects_source_construction_failure(monkeypatch):
    monkeypatch.setattr(teleop_viz_ee, "build_ir_pressure_source", lambda **_kwargs: None)
    args = teleop_viz_ee.parse_live_args(["--oak", "--ir-pressure-shadow"])

    with pytest.raises(RuntimeError, match="requested but could not be constructed"):
        teleop_viz_ee.build_live_ir_runtime(args)


def test_live_runtime_closes_pressure_source_when_sidecar_setup_fails(monkeypatch):
    closed = []
    source = SimpleNamespace(close=lambda: closed.append("pressure"))
    monkeypatch.setattr(teleop_viz_ee, "build_ir_pressure_source", lambda **_kwargs: source)
    monkeypatch.setattr(
        teleop_viz_ee,
        "IRShadowTelemetryLogger",
        lambda _path: (_ for _ in ()).throw(RuntimeError("sidecar failed")),
    )
    args = teleop_viz_ee.parse_live_args([
        "--oak",
        "--ir-pressure-shadow",
        "--ir-sidecar",
        "/tmp/unused.csv",
    ])

    with pytest.raises(RuntimeError, match="sidecar failed"):
        teleop_viz_ee.build_live_ir_runtime(args)

    assert closed == ["pressure"]


def test_pv_shadow_runtime_loads_profile_and_extended_sidecar(tmp_path: Path, monkeypatch):
    profile_path = tmp_path / "profile.json"
    sidecar_path = tmp_path / "pv.csv"
    save_object_profile(
        profile_path,
        PressureVisionObjectProfile("rigid_block", "so101_follower_1", 95.0, 30.0, 20.0),
    )
    source = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(teleop_viz_ee, "build_pv_pressure_source", lambda **_kwargs: source)

    args = teleop_viz_ee.parse_live_args([
        "--pv-pressure-shadow",
        "--pv-object-profile", str(profile_path),
        "--pv-sidecar", str(sidecar_path),
        "--pv-mapping", "relative",
        "--pv-trial-protocol", "2",
    ])
    runtime = teleop_viz_ee.build_live_ir_runtime(args)

    assert runtime.object_profile.object_id == "rigid_block"
    assert runtime.pressure_shadow is True
    assert runtime.pressure_apply is False
    assert runtime.pv_mapping == "relative"
    assert runtime.trial_protocol.repetitions == 2
    assert "pressure_level" in runtime.sidecar.fieldnames
    runtime.sidecar.close()


def test_pv_relative_apply_runtime_requires_sidecar_and_is_constructible(
    tmp_path: Path, monkeypatch
):
    profile_path = tmp_path / "profile.json"
    sidecar_path = tmp_path / "pv_apply.csv"
    save_object_profile(
        profile_path,
        PressureVisionObjectProfile("rigid_block", "so101_follower_1", 95.0, 25.0, 23.0),
    )
    source = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(teleop_viz_ee, "build_pv_pressure_source", lambda **_kwargs: source)

    with pytest.raises(SystemExit, match="2"):
        teleop_viz_ee.parse_live_args([
            "--pv-pressure",
            "--pv-object-profile", str(profile_path),
            "--pv-mapping", "relative",
        ])

    args = teleop_viz_ee.parse_live_args([
        "--pv-pressure",
        "--pv-object-profile", str(profile_path),
        "--pv-sidecar", str(sidecar_path),
        "--pv-mapping", "relative",
        *_pv_apply_gate_args(tmp_path),
    ])
    runtime = teleop_viz_ee.build_live_ir_runtime(args)

    assert runtime.pressure_shadow is False
    assert runtime.pressure_apply is True
    assert runtime.pv_mapping == "relative"
    assert runtime.sidecar is not None
    assert runtime.gripper_closure_limits.max_load == 240
    runtime.sidecar.close()


def test_pv_absolute_apply_remains_disabled(tmp_path: Path, capsys):
    profile_path = tmp_path / "profile.json"
    save_object_profile(
        profile_path,
        PressureVisionObjectProfile("rigid_block", "so101_follower_1", 95.0, 25.0, 23.0),
    )

    with pytest.raises(SystemExit, match="2"):
        teleop_viz_ee.parse_live_args([
            "--pv-pressure",
            "--pv-object-profile", str(profile_path),
        ])

    assert "requires --pv-mapping soft_direct, soft_precise" in capsys.readouterr().err


def test_pv_soft_direct_apply_requires_no_object_profile(tmp_path: Path, monkeypatch):
    sidecar_path = tmp_path / "soft.csv"
    source = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(teleop_viz_ee, "build_pv_pressure_source", lambda **_kwargs: source)

    args = teleop_viz_ee.parse_live_args([
        "--pv-pressure",
        "--pv-mapping", "soft_direct",
        "--pv-sidecar", str(sidecar_path),
        *_pv_apply_gate_args(tmp_path),
    ])
    runtime = teleop_viz_ee.build_live_ir_runtime(args)

    assert runtime.pv_mapping == "soft_direct"
    assert runtime.object_profile is None
    assert runtime.pressure_apply is True
    assert "observed_gripper_pos" in runtime.sidecar.fieldnames
    runtime.sidecar.close()


def test_pv_soft_precise_apply_requires_no_object_profile(tmp_path: Path, monkeypatch):
    sidecar_path = tmp_path / "precise.csv"
    source = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(teleop_viz_ee, "build_pv_pressure_source", lambda **_kwargs: source)

    args = teleop_viz_ee.parse_live_args([
        "--pv-pressure",
        "--pv-mapping", "soft_precise",
        "--pv-sidecar", str(sidecar_path),
        *_pv_apply_gate_args(tmp_path),
    ])
    runtime = teleop_viz_ee.build_live_ir_runtime(args)

    assert runtime.pv_mapping == "soft_precise"
    assert runtime.object_profile is None
    assert runtime.pressure_apply is True
    runtime.sidecar.close()


def test_pv_carton_span_apply_is_distinct_from_soft_precise(tmp_path: Path, monkeypatch):
    source = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(teleop_viz_ee, "build_pv_pressure_source", lambda **_kwargs: source)
    args = teleop_viz_ee.parse_live_args([
        "--pv-pressure",
        "--pv-mapping", "carton_span",
        "--pv-sidecar", str(tmp_path / "span.csv"),
        *_pv_apply_gate_args(tmp_path),
    ])

    runtime = teleop_viz_ee.build_live_ir_runtime(args)

    assert runtime.pv_mapping == "carton_span"
    assert runtime.object_profile is None
    runtime.sidecar.close()


def test_live_control_contract_records_exact_mapping_and_roll(tmp_path: Path):
    args = SimpleNamespace(
        pv_pressure=True,
        pv_evidence_dir=tmp_path,
        wrist_roll_range_deg=0.0,
        wrist_roll_gain=1.0,
    )
    controller = SimpleNamespace(
        mapping_contract={
            "mapping": "carton_span",
            "release_pos": 100.0,
            "pressure_zero_pos": 32.0,
            "pressure_one_pos": 20.0,
            "cutoff_hz": 1.0,
            "stabilize": False,
            "max_grip_step_per_control_frame": 2.0,
        }
    )

    teleop_viz_ee.write_pv_control_contract(args, controller)

    contract = json.loads(
        (tmp_path / "control_contract.json").read_text(encoding="utf-8")
    )
    assert contract["pv_mapping_contract"]["mapping"] == "carton_span"
    assert contract["pv_mapping_contract"]["pressure_zero_pos"] == 32.0
    assert contract["wrist_roll_range_deg"] == 0.0


def test_pv_hard_profile_apply_loads_selected_label(tmp_path: Path, monkeypatch):
    profile_path = tmp_path / "rigid_block_01.json"
    sidecar_path = tmp_path / "hard.csv"
    save_object_profile(
        profile_path,
        PressureVisionObjectProfile("rigid_block_01", "so101_follower_1", 95, 25, 24.5),
    )
    source = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(teleop_viz_ee, "build_pv_pressure_source", lambda **_kwargs: source)

    args = teleop_viz_ee.parse_live_args([
        "--pv-pressure",
        "--pv-mapping", "hard_profile",
        "--pv-object-profile", str(profile_path),
        "--pv-sidecar", str(sidecar_path),
        *_pv_apply_gate_args(tmp_path),
    ])
    runtime = teleop_viz_ee.build_live_ir_runtime(args)

    assert runtime.pv_mapping == "hard_profile"
    assert runtime.object_profile.object_id == "rigid_block_01"
    assert runtime.object_profile.hard_pos == 24.5
    runtime.sidecar.close()


def test_pv_apply_requires_evidence_dir(tmp_path: Path):
    with pytest.raises(SystemExit, match="2"):
        teleop_viz_ee.parse_live_args([
            "--oak",
            "--pv-pressure",
            "--pv-mapping", "soft_direct",
            "--pv-sidecar", str(tmp_path / "sidecar.csv"),
        ])


def test_pv_apply_allows_motor_telemetry_without_closure_limits(
    tmp_path: Path, monkeypatch
):
    source = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(teleop_viz_ee, "build_pv_pressure_source", lambda **_kwargs: source)
    args = teleop_viz_ee.parse_live_args([
        "--oak",
        "--pv-pressure",
        "--pv-mapping", "soft_direct",
        "--pv-sidecar", str(tmp_path / "sidecar.csv"),
        "--pv-evidence-dir", str(tmp_path),
    ])

    runtime = teleop_viz_ee.build_live_ir_runtime(args)

    assert runtime.gripper_closure_limits is None
    assert runtime.sidecar is not None
    runtime.sidecar.close()


def test_pv_apply_rejects_partial_closure_limits(tmp_path: Path):
    with pytest.raises(SystemExit, match="2"):
        teleop_viz_ee.parse_live_args([
            "--oak",
            "--pv-pressure",
            "--pv-mapping", "soft_direct",
            "--pv-sidecar", str(tmp_path / "sidecar.csv"),
            "--pv-evidence-dir", str(tmp_path),
            "--pv-max-current", "35",
        ])


def _repo_script(name: str) -> Path:
    """A carton launcher, read from this repository's own `scripts/`.

    Until 2026-09-04 these two assertions read the copies still sitting in the
    meta-workspace's `scripts/`, which the split left behind: those are the
    pre-split launchers, pointing into the retired `webcam-input/` tree. The
    test passed while checking a file no longer used by anything.
    """
    return CHECKOUT_ROOT / "scripts" / name


def test_carton_launcher_keeps_motor_telemetry_observational():
    script = _repo_script("run_pv_carton_soft_direct_apply.sh").read_text(
        encoding="utf-8"
    )

    assert "--pv-max-load" not in script
    assert "--pv-max-current" not in script
    assert "--pv-max-position-lag" not in script
    assert "PV_CARTON_FIDUCIALS_CONFIRMED" not in script
    assert 'MAPPING="${PV_MAPPING:-soft_precise}"' in script
    assert '--pv-mapping "${MAPPING}"' in script
    assert "pv_carton_soft_precise_apply" in script

    span = _repo_script("run_pv_carton_span_apply.sh").read_text(
        encoding="utf-8"
    )
    assert "PV_MAPPING=carton_span" in span
    assert "--wrist-roll-range-deg 0" in span


def test_pv_apply_evidence_gate_requires_only_live_creative_file(tmp_path: Path):
    args = teleop_viz_ee.parse_live_args([
        "--pv-pressure",
        "--pv-mapping", "soft_direct",
        "--pv-sidecar", str(tmp_path / "sidecar.csv"),
        *_pv_apply_gate_args(tmp_path),
    ])
    with pytest.raises(RuntimeError, match="creative_side.ts"):
        teleop_viz_ee.validate_pv_apply_evidence_gate(args)

    (tmp_path / "creative_side.ts").write_bytes(b"creative")

    assert teleop_viz_ee.validate_pv_apply_evidence_gate(args) == tmp_path / "oak_hand.avi"


def test_live_hold_finalizes_sidecar_as_no_command():
    finalized = []
    logger = SimpleNamespace(
        finalize=lambda sample, *, command_sent, motor_telemetry=None: finalized.append(
            (sample, command_sent, motor_telemetry)
        )
    )
    robot = SimpleNamespace(send_action=lambda _joints: pytest.fail("HOLD must not send"))

    sent = teleop_viz_ee.send_live_action(
        robot,
        None,
        sidecar=logger,
        telemetry_sample="hold-sample",
    )

    assert sent is False
    assert finalized == [("hold-sample", False, None)]


def test_live_action_attaches_rate_limited_motor_telemetry():
    finalized = []
    motor = SimpleNamespace(observed_gripper_pos=27.0)
    sampler = SimpleNamespace(poll=lambda _robot: motor)
    logger = SimpleNamespace(
        finalize=lambda sample, *, command_sent, motor_telemetry=None: finalized.append(
            (sample, command_sent, motor_telemetry)
        )
    )
    robot = SimpleNamespace(send_action=lambda _joints: None)

    sent = teleop_viz_ee.send_live_action(
        robot,
        {"gripper.pos": 26.0},
        sidecar=logger,
        telemetry_sample="sample",
        motor_sampler=sampler,
    )

    assert sent is True
    assert finalized == [("sample", True, motor)]


def test_relative_pv_mapping_requires_shadow_sidecar(tmp_path: Path):
    profile = tmp_path / "profile.json"
    save_object_profile(
        profile,
        PressureVisionObjectProfile("rigid_block", "so101_follower_1", 95, 25, 23),
    )

    with pytest.raises(SystemExit, match="2"):
        teleop_viz_ee.parse_live_args([
            "--pv-pressure-shadow",
            "--pv-object-profile", str(profile),
            "--pv-mapping", "relative",
        ])


# ---------------------------------------------------------------------------
# Lepton blob-mode wiring (no OAK depth, no projection calibration required)
# ---------------------------------------------------------------------------


def test_live_shadow_lepton_mode_does_not_require_oak():
    args = teleop_viz_ee.parse_live_args(
        ["--ir-pressure-shadow", "--ir-lepton-port", "8080"]
    )
    assert args.ir_lepton_port == 8080


def test_live_lepton_port_defaults_to_none():
    args = teleop_viz_ee.parse_live_args(["--oak", "--ir-pressure-shadow"])
    assert args.ir_lepton_port is None


def test_build_ir_pressure_source_uses_lepton_blob_mode_without_calibration(monkeypatch):
    built = {}

    class FakeLepton:
        def __init__(self, *, port):
            built["port"] = port

        def close(self):
            pass

    captured = {}

    class FakeLatest:
        def __init__(self, source):
            self.source = source

        def close(self):
            pass

    def fake_estimator(*, calibration, thermal_source, config=None):
        captured.update(calibration=calibration, thermal_source=thermal_source, config=config)
        return SimpleNamespace(calibration=calibration, thermal_source=thermal_source)

    monkeypatch.setattr(teleop_viz_ee, "LeptonUDPSource", FakeLepton)
    monkeypatch.setattr(teleop_viz_ee, "LatestFrameSource", FakeLatest)
    monkeypatch.setattr(teleop_viz_ee, "HandPressureEstimator", fake_estimator)

    source = teleop_viz_ee.build_ir_pressure_source(
        enabled=True,
        calibration_path="/nonexistent/ignored.json",
        thermal_path="/dev/video21",
        lepton_port=8080,
    )

    assert source is not None
    assert built["port"] == 8080
    assert captured["calibration"] is None
    assert captured["config"].roi_mode == "blob"
    assert isinstance(captured["thermal_source"], FakeLatest)


def test_live_runtime_passes_lepton_port_to_builder(monkeypatch):
    seen = {}

    def fake_builder(*, enabled, calibration_path, thermal_path, lepton_port=None):
        seen.update(enabled=enabled, lepton_port=lepton_port)
        return "pressure-source"

    monkeypatch.setattr(teleop_viz_ee, "build_ir_pressure_source", fake_builder)
    args = teleop_viz_ee.parse_live_args(["--ir-pressure-shadow", "--ir-lepton-port", "9000"])

    runtime = teleop_viz_ee.build_live_ir_runtime(args)

    assert runtime.pressure_source == "pressure-source"
    assert seen["lepton_port"] == 9000
