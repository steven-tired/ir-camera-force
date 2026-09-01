import pytest

from verify_ir_grip_setup import validate_setup


def test_validate_setup_rejects_unstable_bird_path_before_v4l2_query():
    seen_paths: list[str] = []

    def fake_v4l2_reader(path: str) -> str:
        seen_paths.append(path)
        return ""

    with pytest.raises(ValueError, match="stable /dev/v4l/by-id/.+-video-index0"):
        validate_setup(
            bird="/dev/video2",
            thermal="/dev/video21",
            flir_visible="/dev/video20",
            v4l2_reader=fake_v4l2_reader,
        )

    assert seen_paths == []


def test_validate_setup_rejects_thermal_v4l2_failure():
    def fake_v4l2_reader(path: str) -> str:
        if path == "/dev/video21":
            raise RuntimeError("v4l2-ctl failed for /dev/video21: boom")
        return """
        [0]: 'RGB3' (24-bit RGB 8-8-8)
            Size: Discrete 160x128
                Interval: Discrete 0.033s (30.000 fps)
        """

    with pytest.raises(RuntimeError, match="v4l2-ctl failed for /dev/video21"):
        validate_setup(
            bird="/dev/v4l/by-id/usb-0825_Camera-video-index0",
            thermal="/dev/video21",
            flir_visible="/dev/video20",
            v4l2_reader=fake_v4l2_reader,
        )


def test_validate_setup_rejects_thermal_path_mismatch_before_v4l2_query():
    seen_paths: list[str] = []

    def fake_v4l2_reader(path: str) -> str:
        seen_paths.append(path)
        return ""

    with pytest.raises(ValueError, match="/dev/video21"):
        validate_setup(
            bird="/dev/v4l/by-id/usb-0825_Camera-video-index0",
            thermal="/dev/video22",
            flir_visible="/dev/video20",
            v4l2_reader=fake_v4l2_reader,
        )

    assert seen_paths == []


def test_validate_setup_rejects_unparsed_thermal_formats():
    def fake_v4l2_reader(path: str) -> str:
        if path == "/dev/video21":
            return "plain text without V4L2 format blocks"
        return """
        [0]: 'RGB3' (24-bit RGB 8-8-8)
            Size: Discrete 160x128
                Interval: Discrete 0.033s (30.000 fps)
        """

    with pytest.raises(ValueError, match="could not parse thermal formats"):
        validate_setup(
            bird="/dev/v4l/by-id/usb-0825_Camera-video-index0",
            thermal="/dev/video21",
            flir_visible="/dev/video20",
            v4l2_reader=fake_v4l2_reader,
        )
