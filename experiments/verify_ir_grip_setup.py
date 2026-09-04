from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from ir_force.ir_devices import (
    DEFAULT_FLIR_VISIBLE_PATH,
    DEFAULT_THERMAL_PATH,
    assert_distinct_video_devices,
    assert_expected_thermal_device,
    assert_expected_thermal_device_path,
    assert_stable_bird_device_path,
    parse_v4l2_formats,
    VideoFormat,
)
from ir_force.data_paths import dataset_root


def _v4l2_formats(path: str) -> str:
    try:
        return subprocess.check_output(
            ["v4l2-ctl", "--device", path, "--list-formats-ext"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "v4l2-ctl is required to verify camera formats before capture, but it was not found"
        ) from exc
    except subprocess.CalledProcessError as exc:
        output = exc.output.strip() if exc.output else str(exc)
        raise RuntimeError(f"v4l2-ctl failed for {path}: {output}") from exc


def _capture_sample(path: str, out_path: Path) -> None:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {path}")

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"could not read a frame from {path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)


def validate_setup(
    *,
    bird: str,
    thermal: str,
    flir_visible: str,
    allow_unstable_bird_path: bool = False,
    allow_thermal_mismatch: bool = False,
    v4l2_reader=_v4l2_formats,
) -> dict[str, tuple[str, tuple[VideoFormat, ...]]]:
    named_paths = {
        "bird": bird,
        "thermal": thermal,
        "flir_visible": flir_visible,
    }
    assert_distinct_video_devices(named_paths)

    if not allow_unstable_bird_path:
        assert_stable_bird_device_path(bird)
    if not allow_thermal_mismatch:
        assert_expected_thermal_device_path(thermal)

    inspected_devices: dict[str, tuple[str, tuple[VideoFormat, ...]]] = {}
    for name, path in named_paths.items():
        formats_text = v4l2_reader(path)
        formats = parse_v4l2_formats(formats_text)
        inspected_devices[name] = (path, formats)

        if name != "thermal" or allow_thermal_mismatch:
            continue

        assert_expected_thermal_device(path, formats)

    return inspected_devices


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify bird-view and FLIR IR-grip camera setup.")
    parser.add_argument("--bird", required=True, help="SO-101 bird-view RGB path")
    parser.add_argument("--thermal", default=DEFAULT_THERMAL_PATH, help="FLIR thermal loopback path")
    parser.add_argument("--flir-visible", default=DEFAULT_FLIR_VISIBLE_PATH, help="FLIR visible RGB path")
    parser.add_argument(
        "--allow-unstable-bird-path",
        action="store_true",
        help="Allow bird camera paths outside /dev/v4l/by-id/...-video-index0.",
    )
    parser.add_argument(
        "--allow-thermal-mismatch",
        action="store_true",
        help="Allow thermal device path/format mismatches and unparsed thermal formats.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(dataset_root("ir_grip_force_viability") / "setup_check"),
        help="Directory for captured verification frames",
    )
    args = parser.parse_args()

    try:
        inspected_devices = validate_setup(
            bird=args.bird,
            thermal=args.thermal,
            flir_visible=args.flir_visible,
            allow_unstable_bird_path=args.allow_unstable_bird_path,
            allow_thermal_mismatch=args.allow_thermal_mismatch,
        )
    except (RuntimeError, ValueError) as exc:
        parser.exit(2, f"setup verification failed: {exc}\n")

    out_dir = Path(args.out_dir)
    for name, (path, formats) in inspected_devices.items():
        print(f"[{name}] {path}")
        print(f"  formats: {formats if formats else 'not parsed'}")
        _capture_sample(path, out_dir / f"{name}.png")
        print(f"  sample: {out_dir / f'{name}.png'}")


if __name__ == "__main__":
    main()
