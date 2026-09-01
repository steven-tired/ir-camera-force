import pytest

from ir_force.ir_devices import (
    VideoFormat,
    assert_distinct_video_devices,
    assert_expected_thermal_device,
    assert_stable_bird_device_path,
    parse_v4l2_formats,
    select_device_path,
)


def test_assert_distinct_video_devices_accepts_unique_paths():
    assert_distinct_video_devices(
        {
            "bird": "/dev/v4l/by-id/usb-bird-index0",
            "thermal": "/dev/video21",
            "flir_visible": "/dev/video20",
        }
    )


def test_assert_distinct_video_devices_rejects_collision():
    with pytest.raises(ValueError, match="video device collision"):
        assert_distinct_video_devices(
            {
                "bird": "/dev/video2",
                "thermal": "/dev/video2",
            }
        )


def test_assert_distinct_video_devices_rejects_symlink_collision(tmp_path):
    target = tmp_path / "video20"
    target.touch()
    by_id_path = tmp_path / "v4l" / "by-id" / "usb-flir-visible-index0"
    by_id_path.parent.mkdir(parents=True)
    by_id_path.symlink_to(target)

    with pytest.raises(ValueError, match="video device collision"):
        assert_distinct_video_devices(
            {
                "flir_visible": str(by_id_path),
                "thermal": str(target),
            }
        )


def test_select_device_path_prefers_explicit_then_by_id_then_fallback():
    assert select_device_path("/dev/video9", "/dev/video8", "/dev/video7") == "/dev/video9"
    assert select_device_path(None, "/dev/video8", "/dev/video7") == "/dev/video8"
    assert select_device_path(None, None, "/dev/video7") == "/dev/video7"


def test_assert_stable_bird_device_path_accepts_v4l_by_id_video_index0():
    assert_stable_bird_device_path("/dev/v4l/by-id/usb-0825_Camera-video-index0")


def test_assert_stable_bird_device_path_rejects_unstable_dev_video_path():
    with pytest.raises(ValueError, match="stable /dev/v4l/by-id/.+-video-index0"):
        assert_stable_bird_device_path("/dev/video2")


def test_assert_expected_thermal_device_rejects_default_path_mismatch():
    with pytest.raises(ValueError, match="/dev/video21"):
        assert_expected_thermal_device(
            "/dev/video22",
            (VideoFormat(width=160, height=128, pixelformat="RGB3", fps=30.0),),
        )


def test_assert_expected_thermal_device_rejects_missing_rgb3_160x128():
    with pytest.raises(ValueError, match="RGB3 160x128"):
        assert_expected_thermal_device(
            "/dev/video21",
            (VideoFormat(width=320, height=256, pixelformat="YUYV", fps=9.0),),
        )


def test_parse_v4l2_formats_extracts_resolution_pixelformat_and_fps():
    output = """
    [0]: 'YUYV' (YUYV 4:2:2)
        Size: Discrete 640x480
            Interval: Discrete 0.100s (10.000 fps)
    [1]: 'GRAY' (8-bit Greyscale)
        Size: Discrete 160x120
            Interval: Discrete 0.111s (9.000 fps)
    """
    assert parse_v4l2_formats(output) == (
        VideoFormat(width=640, height=480, pixelformat="YUYV", fps=10.0),
        VideoFormat(width=160, height=120, pixelformat="GRAY", fps=9.0),
    )
