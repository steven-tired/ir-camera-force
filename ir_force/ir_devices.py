from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_THERMAL_PATH = "/dev/video21"
DEFAULT_FLIR_VISIBLE_PATH = "/dev/video20"
EXPECTED_THERMAL_PIXELFORMAT = "RGB3"
EXPECTED_THERMAL_SIZE = (160, 128)


@dataclass(frozen=True)
class VideoFormat:
    width: int
    height: int
    pixelformat: str
    fps: float | None = None


@dataclass(frozen=True)
class VideoDevice:
    path: str
    label: str = ""
    formats: tuple[VideoFormat, ...] = ()


def assert_distinct_video_devices(named_paths: dict[str, str]) -> None:
    seen_paths: dict[Path, tuple[str, str]] = {}
    for name, path in named_paths.items():
        if not path:
            continue
        canonical_path = Path(path).resolve(strict=False)
        if canonical_path in seen_paths:
            first_name, first_path = seen_paths[canonical_path]
            raise ValueError(
                f"video device collision: {first_name} ({first_path}) and {name} ({path}) both resolve to {canonical_path}"
            )
        seen_paths[canonical_path] = (name, path)


def parse_v4l2_formats(output: str) -> tuple[VideoFormat, ...]:
    formats: list[VideoFormat] = []
    current_pixelformat: str | None = None
    current_size: tuple[int, int] | None = None

    for raw_line in output.splitlines():
        line = raw_line.strip()

        if line.startswith("["):
            match = re.search(r"'([^']+)'", line)
            current_pixelformat = match.group(1) if match else None
            current_size = None
            continue

        size_match = re.search(r"Size:\s+Discrete\s+(\d+)x(\d+)", line)
        if size_match:
            current_size = (int(size_match.group(1)), int(size_match.group(2)))
            continue

        fps_match = re.search(r"\((\d+(?:\.\d+)?)\s+fps\)", line)
        if current_pixelformat and current_size and fps_match:
            formats.append(
                VideoFormat(
                    width=current_size[0],
                    height=current_size[1],
                    pixelformat=current_pixelformat,
                    fps=float(fps_match.group(1)),
                )
            )

    return tuple(formats)


def select_device_path(explicit_path: str | None, by_id_path: str | None, fallback_path: str) -> str:
    if explicit_path:
        return explicit_path
    if by_id_path:
        return by_id_path
    return fallback_path


def assert_stable_bird_device_path(path: str) -> None:
    if path.startswith("/dev/v4l/by-id/") and path.endswith("-video-index0"):
        return

    raise ValueError(
        f"bird path must be a stable /dev/v4l/by-id/...-video-index0 path by default; got {path!r}. "
        "Pass --allow-unstable-bird-path to override."
    )


def has_video_format(
    formats: tuple[VideoFormat, ...],
    *,
    pixelformat: str,
    width: int,
    height: int,
) -> bool:
    return any(
        video_format.pixelformat == pixelformat
        and video_format.width == width
        and video_format.height == height
        for video_format in formats
    )


def assert_expected_thermal_device(
    path: str,
    formats: tuple[VideoFormat, ...],
    *,
    expected_path: str = DEFAULT_THERMAL_PATH,
    expected_pixelformat: str = EXPECTED_THERMAL_PIXELFORMAT,
    expected_size: tuple[int, int] = EXPECTED_THERMAL_SIZE,
) -> None:
    assert_expected_thermal_device_path(path, expected_path=expected_path)
    assert_expected_thermal_formats(
        path,
        formats,
        expected_pixelformat=expected_pixelformat,
        expected_size=expected_size,
    )


def assert_expected_thermal_device_path(
    path: str,
    *,
    expected_path: str = DEFAULT_THERMAL_PATH,
) -> None:
    if path != expected_path:
        raise ValueError(
            f"thermal path must be {expected_path!r} by default; got {path!r}. "
            "Pass --allow-thermal-mismatch to override."
        )


def assert_expected_thermal_formats(
    path: str,
    formats: tuple[VideoFormat, ...],
    *,
    expected_pixelformat: str = EXPECTED_THERMAL_PIXELFORMAT,
    expected_size: tuple[int, int] = EXPECTED_THERMAL_SIZE,
) -> None:
    if not formats:
        raise ValueError(
            f"could not parse thermal formats for {path!r}; expected {expected_pixelformat} "
            f"{expected_size[0]}x{expected_size[1]}. Pass --allow-thermal-mismatch to override."
        )

    if has_video_format(
        formats,
        pixelformat=expected_pixelformat,
        width=expected_size[0],
        height=expected_size[1],
    ):
        return

    raise ValueError(
        f"thermal formats for {path!r} must include {expected_pixelformat} "
        f"{expected_size[0]}x{expected_size[1]} by default. "
        "Pass --allow-thermal-mismatch to override."
    )
