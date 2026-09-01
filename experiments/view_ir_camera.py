from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def lepton_frame_to_u8(frame: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    """Percentile-autoscale a raw uint16 Lepton frame to displayable uint8."""
    lo, hi = np.percentile(frame, (p_lo, p_hi))
    if hi <= lo:
        return np.zeros(frame.shape, dtype=np.uint8)
    scaled = np.clip((frame.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick FLIR thermal camera viewer.")
    parser.add_argument("--thermal", default="/dev/video21", help="FLIR thermal loopback path")
    parser.add_argument(
        "--lepton-udp",
        type=int,
        nargs="?",
        const=8080,
        default=None,
        metavar="PORT",
        help="read raw Lepton frames from the Pi UDP streamer instead of a V4L2 device",
    )
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--save-dir", type=Path, default=Path("/tmp/ir_view_snapshots"))
    args = parser.parse_args()

    lepton = None
    cap = None
    if args.lepton_udp is not None:
        from lerobot_teleoperator_so101_webcam.ir_capture import (
            FrameUnavailableError,
            LeptonUDPSource,
        )

        lepton = LeptonUDPSource(port=args.lepton_udp)
        window = f"Lepton udp:{lepton.port}  |  q quit, s save"
    else:
        cap = cv2.VideoCapture(args.thermal, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not cap.isOpened():
            raise SystemExit(f"could not open thermal camera: {args.thermal}")
        window = f"IR thermal {args.thermal}  |  q quit, s save"

    args.save_dir.mkdir(parents=True, exist_ok=True)
    last_frame_t = None
    fps = 0.0
    reported_ffc_state = None

    while True:
        raw = None
        if lepton is not None:
            try:
                sample = lepton.read()
            except FrameUnavailableError as exc:
                print(f"warning: {exc}")
                key = cv2.waitKey(50) & 0xFF
                if key == ord("q"):
                    break
                continue
            raw = sample.frame
            frame = cv2.cvtColor(lepton_frame_to_u8(raw), cv2.COLOR_GRAY2BGR)
            if (
                sample.lepton_telemetry is not None
                and sample.lepton_telemetry.ffc_state != reported_ffc_state
            ):
                telemetry = sample.lepton_telemetry
                print(
                    "Lepton telemetry: "
                    f"frame={telemetry.frame_counter} "
                    f"packet_timestamp_ms={telemetry.packet_timestamp_ms} "
                    f"host_read_complete_monotonic_s={sample.t:.6f} "
                    f"ffc={telemetry.ffc_state} "
                    f"since_last_ffc={telemetry.since_last_ffc_s:.3f}s "
                    f"tlinear={telemetry.tlinear_enabled} "
                    f"resolution={telemetry.tlinear_resolution_k}K",
                    flush=True,
                )
                reported_ffc_state = telemetry.ffc_state
            if last_frame_t is not None and sample.t > last_frame_t:
                fps = 0.9 * fps + 0.1 / (sample.t - last_frame_t)
            last_frame_t = sample.t
        else:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("warning: failed to read frame")
                key = cv2.waitKey(50) & 0xFF
                if key == ord("q"):
                    break
                continue

        if args.scale > 1:
            frame_view = cv2.resize(
                frame,
                (frame.shape[1] * args.scale, frame.shape[0] * args.scale),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            frame_view = frame

        if raw is not None:
            if sample.temperature_c is None:
                stats = (
                    f"min {int(raw.min())}  med {int(np.median(raw))}  "
                    f"max {int(raw.max())}  {fps:.1f} fps"
                )
            else:
                stats = (
                    f"{float(sample.temperature_c.min()):.2f}/"
                    f"{float(np.median(sample.temperature_c)):.2f}/"
                    f"{float(sample.temperature_c.max()):.2f} C  {fps:.1f} fps"
                )
            cv2.putText(
                frame_view,
                stats,
                (8, frame_view.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow(window, frame_view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_path = args.save_dir / f"ir_snapshot_{stamp}.png"
            cv2.imwrite(str(out_path), raw if raw is not None else frame)
            print(f"saved {out_path}")

    if cap is not None:
        cap.release()
    if lepton is not None:
        lepton.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
