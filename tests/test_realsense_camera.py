import numpy as np
import pytest

from ir_force.realsense_camera import (
    RealSenseCamera,
    RealSenseRawProjectorCamera,
)


class _Frame:
    def __init__(self, data):
        self._data = data

    def get_data(self):
        return self._data

    def __bool__(self):
        return True


class _Frames:
    def __init__(self, color, depth):
        self._color = _Frame(color)
        self._depth = _Frame(depth)

    def get_color_frame(self):
        return self._color

    def get_depth_frame(self):
        return self._depth


class _Config:
    def __init__(self):
        self.device = None
        self.streams = []

    def enable_device(self, serial):
        self.device = serial

    def enable_stream(self, *args):
        self.streams.append(args)


class _Pipeline:
    def __init__(self, frames):
        self.frames = frames
        self.config = None
        self.stop_calls = 0

    def start(self, config):
        self.config = config
        sensor = type("Sensor", (), {"get_depth_scale": lambda _self: 0.0005})()
        device = type("Device", (), {"first_depth_sensor": lambda _self: sensor})()
        return type("Profile", (), {"get_device": lambda _self: device})()

    def wait_for_frames(self, timeout_ms):
        assert timeout_ms == 1000
        return self.frames

    def stop(self):
        self.stop_calls += 1


class _Align:
    def __init__(self):
        self.inputs = []

    def process(self, frames):
        self.inputs.append(frames)
        return frames


class _FakeRS:
    class stream:
        color = "color"
        depth = "depth"

    class format:
        bgr8 = "bgr8"
        z16 = "z16"

    def __init__(self):
        color = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        depth = np.array([[0, 2, 4], [6, 8, 10]], dtype=np.uint16)
        self.frames = _Frames(color, depth)
        self.pipeline_instance = _Pipeline(self.frames)
        self.config_instance = _Config()
        self.align_instance = _Align()

    def pipeline(self):
        return self.pipeline_instance

    def config(self):
        return self.config_instance

    def align(self, target):
        assert target == self.stream.color
        return self.align_instance


def test_realsense_camera_reads_color_aligned_metric_depth(monkeypatch):
    rs = _FakeRS()
    observed = iter((12.5,))
    monkeypatch.setattr("ir_force.realsense_camera.time.perf_counter", lambda: next(observed))
    camera = RealSenseCamera(
        serial="233522078685",
        width=640,
        height=480,
        fps=30,
        rs_module=rs,
    )

    camera.start()
    color, depth_mm, observed_at_s = camera.read(timeout_ms=1000)
    camera.stop()

    assert rs.config_instance.device == "233522078685"
    assert rs.config_instance.streams == [
        ("depth", 640, 480, "z16", 30),
        ("color", 640, 480, "bgr8", 30),
    ]
    assert rs.align_instance.inputs == [rs.frames]
    np.testing.assert_array_equal(color, np.arange(18, dtype=np.uint8).reshape(2, 3, 3))
    np.testing.assert_allclose(
        depth_mm,
        np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32),
    )
    assert observed_at_s == 12.5
    assert rs.pipeline_instance.stop_calls == 1


def test_realsense_camera_stop_is_idempotent():
    rs = _FakeRS()
    camera = RealSenseCamera(rs_module=rs)

    camera.start()
    camera.stop()
    camera.stop()

    assert rs.pipeline_instance.stop_calls == 1


def test_realsense_start_preserves_original_start_failure_when_cleanup_also_fails():
    rs = _FakeRS()

    def fail_start(_config):
        raise RuntimeError("start failed")

    def fail_stop():
        raise RuntimeError("stop before start failed")

    rs.pipeline_instance.start = fail_start
    rs.pipeline_instance.stop = fail_stop
    camera = RealSenseCamera(rs_module=rs)

    with pytest.raises(RuntimeError, match="start failed"):
        camera.start()


class _RawFrame(_Frame):
    def __init__(self, data, *, number, timestamp_ms, domain):
        super().__init__(data)
        self._number = number
        self._timestamp_ms = timestamp_ms
        self._domain = domain

    def get_frame_number(self):
        return self._number

    def get_timestamp(self):
        return self._timestamp_ms

    def get_frame_timestamp_domain(self):
        return self._domain


class _RawFrames(_Frames):
    def __init__(self):
        self._color = _RawFrame(
            np.full((720, 1280, 3), 7, dtype=np.uint8),
            number=11,
            timestamp_ms=101.5,
            domain="global_time",
        )
        self._depth = _RawFrame(
            np.full((720, 1280), 500, dtype=np.uint16),
            number=8,
            timestamp_ms=99.0,
            domain="global_time",
        )


class _Intrinsics:
    def __init__(self, *, model):
        self.width = 1280
        self.height = 720
        self.fx = 10.0
        self.fy = 11.0
        self.ppx = 12.0
        self.ppy = 13.0
        self.model = model
        self.coeffs = [0.0] * 5


class _Extrinsics:
    def __init__(self, offset):
        self.rotation = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.translation = [offset, 0.0, 0.0]


class _VideoProfile:
    def __init__(self, *, stream, fmt, fps, unique_id, model):
        self._stream = stream
        self._format = fmt
        self._fps = fps
        self._unique_id = unique_id
        self._intrinsics = _Intrinsics(model=model)
        self._extrinsics = {}

    def width(self):
        return 1280

    def height(self):
        return 720

    def format(self):
        return self._format

    def fps(self):
        return self._fps

    def stream_type(self):
        return self._stream

    def stream_index(self):
        return 0

    def unique_id(self):
        return self._unique_id

    def as_video_stream_profile(self):
        return self

    def get_intrinsics(self):
        return self._intrinsics

    def get_extrinsics_to(self, other):
        return self._extrinsics[id(other)]


class _RawDevice:
    def __init__(self, rs):
        self._rs = rs

    def get_info(self, key):
        return {
            self._rs.camera_info.serial_number: "233522078685",
            self._rs.camera_info.name: "RealSense D435I",
            self._rs.camera_info.firmware_version: "5.16.0.1",
            self._rs.camera_info.product_line: "D400",
        }[key]

    def first_depth_sensor(self):
        return type("Sensor", (), {"get_depth_scale": lambda _self: 0.001})()


class _RawActiveProfile:
    def __init__(self, rs):
        self._rs = rs
        self.color = _VideoProfile(
            stream=rs.stream.color,
            fmt=rs.format.rgb8,
            fps=15,
            unique_id=3,
            model="inverse_brown_conrady",
        )
        self.depth = _VideoProfile(
            stream=rs.stream.depth,
            fmt=rs.format.z16,
            fps=6,
            unique_id=0,
            model="brown_conrady",
        )
        self.depth._extrinsics[id(self.color)] = _Extrinsics(0.015)
        self.color._extrinsics[id(self.depth)] = _Extrinsics(-0.015)
        self.device = _RawDevice(rs)

    def get_device(self):
        return self.device

    def get_stream(self, stream):
        return self.color if stream == self._rs.stream.color else self.depth


class _RawPipeline:
    def __init__(self, rs):
        self.rs = rs
        self.frames = _RawFrames()
        self.stop_calls = 0

    def start(self, config):
        self.config = config
        return self.rs.active_profile

    def wait_for_frames(self, timeout_ms):
        assert timeout_ms == 1000
        return self.frames

    def stop(self):
        self.stop_calls += 1


class _RawRS:
    __version__ = "2.58.3"

    class stream:
        color = "color"
        depth = "depth"

    class format:
        rgb8 = "rgb8"
        z16 = "z16"

    class camera_info:
        serial_number = "serial_number"
        name = "name"
        firmware_version = "firmware_version"
        product_line = "product_line"

    def __init__(self):
        self.config_instance = _Config()
        self.active_profile = _RawActiveProfile(self)
        self.pipeline_instance = _RawPipeline(self)

    def config(self):
        return self.config_instance

    def pipeline(self):
        return self.pipeline_instance


def test_raw_projector_camera_reads_unaligned_rgb8_z16_and_metadata(monkeypatch):
    rs = _RawRS()
    monkeypatch.setattr(
        "ir_force.realsense_camera.time.perf_counter",
        lambda: 12.5,
    )
    camera = RealSenseRawProjectorCamera(rs_module=rs)

    camera.start()
    sample = camera.read()

    assert rs.config_instance.device == "233522078685"
    assert rs.config_instance.streams == [
        ("color", 1280, 720, "rgb8", 15),
        ("depth", 1280, 720, "z16", 6),
    ]
    assert sample.color_rgb.shape == (720, 1280, 3)
    assert sample.color_rgb.dtype == np.uint8
    assert sample.depth_z16.shape == (720, 1280)
    assert sample.depth_z16.dtype == np.uint16
    assert sample.depth_sdk_frame is rs.pipeline_instance.frames._depth
    assert sample.observed_at_s == 12.5
    assert sample.color_frame_number == 11
    assert sample.depth_frame_number == 8
    assert sample.color_timestamp_ms == 101.5
    assert sample.depth_timestamp_ms == 99.0
    assert sample.color_timestamp_domain == "global_time"
    assert sample.depth_timestamp_domain == "global_time"
    assert camera.runtime_metadata["sdk_version"] == "2.58.3"
    assert camera.runtime_metadata["device"]["serial"] == "233522078685"
    assert camera.runtime_metadata["resolved"]["color"]["format"] == "rgb8"
    assert camera.runtime_metadata["resolved"]["depth"]["fps"] == 6
    assert camera.runtime_metadata["depth_scale_m"] == 0.001
    assert camera.depth_intrinsics is rs.active_profile.depth._intrinsics
    assert (
        camera.depth_to_color_extrinsics
        is rs.active_profile.depth._extrinsics[id(rs.active_profile.color)]
    )
    camera.stop()
    assert rs.pipeline_instance.stop_calls == 1


def test_raw_projector_camera_rejects_resolved_profile_mismatch_and_stops():
    rs = _RawRS()
    rs.active_profile.depth._fps = 15
    camera = RealSenseRawProjectorCamera(rs_module=rs)

    with pytest.raises(RuntimeError, match="resolved raw profile mismatch"):
        camera.start()

    assert rs.pipeline_instance.stop_calls == 1


def test_raw_projector_camera_rejects_incomplete_frameset():
    rs = _RawRS()
    rs.pipeline_instance.frames._depth = None
    camera = RealSenseRawProjectorCamera(rs_module=rs)
    camera.start()

    with pytest.raises(RuntimeError, match="incomplete raw RGB/depth frameset"):
        camera.read()

    camera.stop()
