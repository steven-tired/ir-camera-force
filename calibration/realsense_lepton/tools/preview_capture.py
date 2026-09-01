"""Live dual-detection capture for Lepton↔D435i checkerboard pairs.

Shows the D435i color and the Lepton thermal side by side with findChessboardCornersSB
overlays (the SAME detector the calibration uses). Press 'c' to save a pair ONLY when
BOTH cameras detect the full 4x3 board — so every saved pair is guaranteed both-detected
(no more blind capture). Saves color_image_N.png + thermal_grayimage_N.png into the given
build dir (same layout/naming as depth_saver), continuing after any existing N.

Run (needs .venv-lerobot; Wayland needs xcb):
  cd /home/zhuokai/hand-teleop
  QT_QPA_PLATFORM=xcb env -u PYTHONPATH \
    PYTHONPATH=webcam-input/.worktrees/ir-hand-pressure-so101-teleop/lerobot_teleoperator_so101_webcam \
    .venv-lerobot/bin/python scripts/preview_capture.py \
    --out <run>/stream/build [--port 8080]

Requires the Pi Lepton streamer running (scripts/run_lepton_stream.sh start) and the D435i
connected. This tool OWNS the D435i + UDP port while open — stop depth_saver first.
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs

SERIAL = "233522078685"
PATTERN = (4, 3)
FLAGS = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
MIN_T, MAX_T = 27300.0, 33500.0  # depth_saver -mintemp/-maxtemp scaling


def detect(gray):
    ok, corners = cv2.findChessboardCornersSB(gray, PATTERN, flags=FLAGS)
    return (ok and corners is not None and len(corners) == 12), corners


def next_index(out):
    n = 0
    for f in glob.glob(os.path.join(out, "images", "color_image_*.png")):
        try:
            n = max(n, int("".join(c for c in os.path.basename(f) if c.isdigit())))
        except ValueError:
            pass
    return n + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="build dir with images/ + thermal_images/")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--cooldown", type=float, default=1.2,
                    help="min seconds between auto-saves")
    ap.add_argument("--move-thresh", type=float, default=30.0,
                    help="min mean color-corner move (px) vs last save, for diversity")
    ap.add_argument("--no-auto", action="store_true", help="disable auto-capture (manual c only)")
    ap.add_argument("--thermal-only", action="store_true",
                    help="INTRINSIC pass: capture thermal only (no D435i); auto-save when the"
                         " thermal board detects. Move the board over the WHOLE thermal FOV"
                         " (corners+edges) with tilts/distances. Saves to <out>/intrinsic_thermal.")
    args = ap.parse_args()
    thermal_only = args.thermal_only
    therm_dir = os.path.join(args.out, "intrinsic_thermal") if thermal_only \
        else os.path.join(args.out, "thermal_images")
    os.makedirs(therm_dir, exist_ok=True)
    if not thermal_only:
        os.makedirs(os.path.join(args.out, "images"), exist_ok=True)

    from lerobot_teleoperator_so101_webcam.ir_capture import (
        FrameUnavailableError, LeptonUDPSource)
    lepton = LeptonUDPSource(port=args.port)

    pipe = None
    if not thermal_only:
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(SERIAL)
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 15)
        pipe.start(cfg)
        for _ in range(5):
            pipe.wait_for_frames()

    # next index: thermal_grayimage_* in the target thermal dir
    idx = 0
    for f in glob.glob(os.path.join(therm_dir, "thermal_grayimage_*.png")):
        try:
            idx = max(idx, int("".join(c for c in os.path.basename(f) if c.isdigit())))
        except ValueError:
            pass
    idx += 1
    saved = 0
    auto = not args.no_auto
    last_save_t = 0.0
    last_corners = None
    ok_streak = 0

    def save(rgb, tgray, cur_cor):
        nonlocal idx, saved, last_save_t, last_corners
        cv2.imwrite(os.path.join(therm_dir, f"thermal_grayimage_{idx}.png"), tgray)
        if not thermal_only:
            cv2.imwrite(os.path.join(args.out, "images", f"color_image_{idx}.png"),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"  saved {'thermal' if thermal_only else 'pair'} {idx}")
        idx += 1
        saved += 1
        last_save_t = time.time()
        last_corners = cur_cor.reshape(-1, 2).copy()

    mode = "THERMAL-ONLY (intrinsics)" if thermal_only else "PAIR (extrinsics)"
    print(f"ready. mode={mode}. next index={idx}. AUTO={'on' if auto else 'off'}. a=toggle c=save q=quit.")
    try:
        while True:
            rgb = None; c_ok = False; c_cor = None
            if not thermal_only:
                fs = pipe.wait_for_frames()
                cframe = fs.get_color_frame()
                if not cframe:
                    continue
                rgb = np.asanyarray(cframe.get_data())
                c_ok, c_cor = detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
            try:
                s = lepton.read()
                traw = np.asarray(s.frame)
            except FrameUnavailableError:
                continue
            tgray = np.clip((traw.astype(np.float32) - MIN_T) / (MAX_T - MIN_T) * 255.0,
                            0, 255).astype(np.uint8)
            t_ok, t_cor = detect(tgray)

            # gate + the corners used for the move-diversity check
            ok = t_ok if thermal_only else (c_ok and t_ok)
            gate_cor = t_cor if thermal_only else c_cor
            gate_scale = (160.0 if thermal_only else 1280.0)  # normalize move-thresh across cams
            ok_streak = ok_streak + 1 if ok else 0

            now = time.time()
            moved = True
            if last_corners is not None and ok:
                # scale move-thresh to the gate camera (color 1280 vs thermal 160)
                mt = args.move_thresh * (gate_scale / 1280.0)
                moved = float(np.linalg.norm(gate_cor.reshape(-1, 2) - last_corners, axis=1).mean()) >= mt
            if auto and ok and ok_streak >= 2 and (now - last_save_t) >= args.cooldown and moved:
                save(rgb, tgray, gate_cor)

            # display
            tdisp = cv2.cvtColor(cv2.resize(tgray, (360, 270), interpolation=cv2.INTER_NEAREST),
                                 cv2.COLOR_GRAY2BGR)
            if t_ok:
                cv2.drawChessboardCorners(tdisp, PATTERN, (t_cor * (360.0 / 160.0)).astype(np.float32), True)
            cv2.putText(tdisp, f"THERMAL {'12/12' if t_ok else 'no'}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0) if t_ok else (0, 0, 255), 2)
            if thermal_only:
                canvas = tdisp
            else:
                cdisp = cv2.cvtColor(cv2.resize(rgb, (480, 270)), cv2.COLOR_RGB2BGR)
                if c_ok:
                    cv2.drawChessboardCorners(cdisp, PATTERN, (c_cor * (480.0 / 1280.0)).astype(np.float32), True)
                cv2.putText(cdisp, f"COLOR {'12/12' if c_ok else 'no'}", (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0) if c_ok else (0, 0, 255), 2)
                canvas = np.hstack([cdisp, tdisp])
            if ok and auto and not moved:
                status, col = "move the board (too close to last)", (0, 165, 255)
            elif ok:
                status, col = ("AUTO saving..." if auto else "OK - press c"), (0, 220, 0)
            else:
                status, col = ("wait for THERMAL green" if thermal_only else "wait for BOTH green"), (0, 165, 255)
            cv2.putText(canvas, f"{status}  saved={saved} next={idx} auto={'on' if auto else 'off'} [{mode}]",
                        (8, 262), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            cv2.imshow("preview_capture  (a=auto c=save q=quit)", canvas)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("a"):
                auto = not auto; print(f"  auto = {'on' if auto else 'off'}")
            if k == ord("c"):
                if not ok:
                    print("  not saved: required camera(s) must show 12/12"); continue
                save(rgb, tgray, gate_cor)
    finally:
        if pipe is not None:
            pipe.stop()
        lepton.close()
        cv2.destroyAllWindows()
        print(f"done. saved {saved} both-detected pairs.")


if __name__ == "__main__":
    main()
