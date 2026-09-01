"""Guided event logging for raw-count repeatability recordings."""

from __future__ import annotations

import argparse
import csv
import math
import shlex
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from raw_repeatability_protocol import ProtocolPhase, build_protocol, prepare_run


def append_event(
    events_path: Path,
    *,
    timestamp_ns: int,
    event_type: str,
    phase: str,
    run_id: str,
) -> None:
    """Append an event with a CLOCK_MONOTONIC-compatible timestamp."""
    with events_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("timestamp_ns", "event_type", "phase", "run_id"))
        writer.writerow(
            {
                "timestamp_ns": timestamp_ns,
                "event_type": event_type,
                "phase": phase,
                "run_id": run_id,
            }
        )


def build_bridge_command(
    *,
    audit_root: Path,
    run_dir: Path,
    duration_s: float,
    display_mode: str,
    fixed_raw_low: int | None,
    fixed_raw_high: int | None,
) -> list[str]:
    """Build the explicit command run in the separate privileged terminal."""
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    command = [
        "sudo",
        "timeout",
        "--signal=INT",
        "--kill-after=5s",
        f"{math.ceil(duration_s) + 30}s",
        str(audit_root / "flirone"),
        str(audit_root / "palettes" / "Iron2.raw"),
        "--raw-dir",
        str(run_dir / "raw"),
        "--raw-frame-limit",
        "0",
    ]
    if display_mode == "fixed":
        if fixed_raw_low is None or fixed_raw_high is None or fixed_raw_low >= fixed_raw_high:
            raise ValueError("fixed display mode requires ordered raw-count bounds")
        command.extend(
            [
                "--fixed-raw-low",
                str(fixed_raw_low),
                "--fixed-raw-high",
                str(fixed_raw_high),
            ]
        )
    elif display_mode != "dynamic":
        raise ValueError("display_mode must be dynamic or fixed")
    return command


def require_ready(read_input: Callable[[str], str] = input) -> None:
    if read_input("Type READY after the bridge is running and /dev/video21 is available: ").strip() != "READY":
        raise RuntimeError("recording was not started because READY was not confirmed")


def wait_for_raw_export(
    raw_dir: Path,
    *,
    timeout_s: float,
    poll_interval_s: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Confirm that the newly started bridge is exporting raw metadata."""
    if timeout_s <= 0.0 or poll_interval_s <= 0.0:
        raise ValueError("raw export timeout and poll interval must be positive")
    deadline_s = monotonic() + timeout_s
    while True:
        raw_metadata = sorted(raw_dir.glob("raw_frame_*.json"))
        if raw_metadata:
            return raw_metadata[0]
        remaining_s = deadline_s - monotonic()
        if remaining_s <= 0.0:
            break
        sleep(min(poll_interval_s, remaining_s))
    raise RuntimeError(
        f"no raw metadata appeared in {raw_dir} within {timeout_s:g}s; "
        "the raw bridge is not running with --raw-dir for this run"
    )


def _append_rgb_frame(
    frames_path: Path,
    *,
    frame_index: int,
    timestamp_ns: int,
    relative_path: str,
) -> None:
    with frames_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("frame_index", "timestamp_ns", "file"))
        writer.writerow(
            {
                "frame_index": frame_index,
                "timestamp_ns": timestamp_ns,
                "file": relative_path,
            }
        )


def capture_protocol(
    *,
    capture: Any,
    run_dir: Path,
    run_id: str,
    phases: Iterable[ProtocolPhase],
    fps: float,
    max_rgb_gap_s: float = 3.0,
    monotonic: Callable[[], float],
    monotonic_ns: Callable[[], int],
    sleep: Callable[[float], None],
    write_image: Callable[[Path, Any], bool],
) -> int:
    """Capture RGB frames and event boundaries on the raw bridge's monotonic clock."""
    if fps <= 0.0 or max_rgb_gap_s <= 0.0:
        raise ValueError("fps and max_rgb_gap_s must be positive")
    events_path = run_dir / "events.csv"
    frames_path = run_dir / "rgb_frames.csv"
    if not events_path.is_file() or not frames_path.is_file():
        raise FileNotFoundError("run directory must be prepared before capture")

    period_s = 1.0 / fps
    next_capture_s = monotonic()
    last_rgb_success_s = next_capture_s
    frame_index = 0
    append_event(
        events_path,
        timestamp_ns=monotonic_ns(),
        event_type="run_start",
        phase="",
        run_id=run_id,
    )
    for phase in phases:
        print(f"{phase.name}: {phase.instruction} ({phase.duration_s:g}s)", flush=True)
        append_event(
            events_path,
            timestamp_ns=monotonic_ns(),
            event_type="phase_start",
            phase=phase.name,
            run_id=run_id,
        )
        phase_end_s = monotonic() + phase.duration_s
        while monotonic() < phase_end_s:
            ok, image = capture.read()
            timestamp_ns = monotonic_ns()
            if not ok or image is None:
                append_event(
                    events_path,
                    timestamp_ns=timestamp_ns,
                    event_type="rgb_read_failure",
                    phase=phase.name,
                    run_id=run_id,
                )
                if monotonic() - last_rgb_success_s > max_rgb_gap_s:
                    append_event(
                        events_path,
                        timestamp_ns=monotonic_ns(),
                        event_type="rgb_capture_abort",
                        phase=phase.name,
                        run_id=run_id,
                    )
                    raise RuntimeError(f"RGB stream was unavailable for more than {max_rgb_gap_s:g}s")
                next_capture_s += period_s
                sleep(max(0.0, next_capture_s - monotonic()))
                continue
            last_rgb_success_s = monotonic()
            relative_path = f"rgb/frame_{frame_index:06d}.png"
            output_path = run_dir / relative_path
            if not write_image(output_path, image):
                raise RuntimeError(f"could not write RGB frame {output_path}")
            _append_rgb_frame(
                frames_path,
                frame_index=frame_index,
                timestamp_ns=timestamp_ns,
                relative_path=relative_path,
            )
            frame_index += 1
            next_capture_s += period_s
            sleep(max(0.0, next_capture_s - monotonic()))
        append_event(
            events_path,
            timestamp_ns=monotonic_ns(),
            event_type="phase_end",
            phase=phase.name,
            run_id=run_id,
        )
    append_event(
        events_path,
        timestamp_ns=monotonic_ns(),
        event_type="run_end",
        phase="",
        run_id=run_id,
    )
    return frame_index


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("ffc", "restart", "dynamic"), required=True)
    parser.add_argument("--target-raw-roi", required=True, help="x,y,width,height in the 80x60 raw frame")
    parser.add_argument("--control-raw-roi", required=True, help="x,y,width,height in the 80x60 raw frame")
    parser.add_argument("--thermal", default="/dev/video21", help="FLIR colour V4L2 output")
    parser.add_argument("--fps", type=float, default=10.0, help="RGB capture rate")
    parser.add_argument("--display-mode", choices=("dynamic", "fixed"), default="dynamic")
    parser.add_argument("--fixed-raw-low", type=int)
    parser.add_argument("--fixed-raw-high", type=int)
    parser.add_argument("--raw-ready-timeout-s", type=float, default=15.0)
    parser.add_argument("--max-rgb-gap-s", type=float, default=3.0)
    parser.add_argument("--audit-root", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args(argv)


def _open_rgb_capture(path: str) -> Any:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV is required to capture the FLIR RGB stream") from exc
    capture = cv2.VideoCapture(path, cv2.CAP_V4L2)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"could not open FLIR RGB stream {path}")
    return capture


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.fps <= 0.0 or args.max_rgb_gap_s <= 0.0:
        raise ValueError("fps and max_rgb_gap_s must be positive")
    run_dir = prepare_run(
        args.session_root,
        run_id=args.run_id,
        mode=args.mode,
        target_raw_roi=args.target_raw_roi,
        control_raw_roi=args.control_raw_roi,
        display_mode=args.display_mode,
        fixed_raw_low=args.fixed_raw_low,
        fixed_raw_high=args.fixed_raw_high,
    )
    phases = build_protocol(args.mode)
    bridge_command = build_bridge_command(
        audit_root=args.audit_root,
        run_dir=run_dir,
        duration_s=sum(phase.duration_s for phase in phases),
        display_mode=args.display_mode,
        fixed_raw_low=args.fixed_raw_low,
        fixed_raw_high=args.fixed_raw_high,
    )
    print(f"prepared run directory: {run_dir}")
    print("Start the raw bridge in a separate terminal, then leave it running:")
    print(f"  cd {shlex.quote(str(args.audit_root))}")
    print(f"  {shlex.join(bridge_command)}")
    require_ready()
    first_raw_metadata = wait_for_raw_export(
        run_dir / "raw",
        timeout_s=args.raw_ready_timeout_s,
        poll_interval_s=0.5,
    )
    print(f"raw export confirmed: {first_raw_metadata.name}")

    capture = _open_rgb_capture(args.thermal)
    try:
        frame_count = capture_protocol(
            capture=capture,
            run_dir=run_dir,
            run_id=args.run_id,
            phases=phases,
            fps=args.fps,
            max_rgb_gap_s=args.max_rgb_gap_s,
            monotonic=time.monotonic,
            monotonic_ns=time.monotonic_ns,
            sleep=time.sleep,
            write_image=lambda path, image: __import__("cv2").imwrite(str(path), image),
        )
    finally:
        capture.release()
    print(f"completed {args.mode} run {args.run_id}: saved {frame_count} RGB frames to {run_dir}")


if __name__ == "__main__":
    main()
