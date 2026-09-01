"""RGB-half calibration rehearsal — needs only the D435i + the checkerboard.

No heat source required. It mirrors the C++ extrinsic detector EXACTLY:
  patternSize = (cols-1, rows-1) = (4,3) for the 5x4 / 30mm board,
  cv2.findChessboardCorners(gray, (4,3), ADAPTIVE_THRESH|NORMALIZE_IMAGE).

Purpose BEFORE the heat source arrives:
  1. Prove the physical board really resolves as 4x3 = 12 inner corners in RGB.
  2. Rehearse all 30 intrinsic + 36 paired poses at the real 0.25-0.72 m distances,
     so capture is fast once thermal contrast is available.
  3. Confirm the D435i RGB sees all 12 corners even at the hard far range.

It SAVES NOTHING into any calibration directory and fits no model. It is a
rehearsal/QA aid only; the immutable run still uses the sanctioned C++ tools.

  # live window (default), rehearse poses:
  env -u PYTHONPATH .venv-lerobot/bin/python scripts/calib_rgb_rehearsal.py
  # headless detection-rate check over N frames (no display):
  env -u PYTHONPATH .venv-lerobot/bin/python scripts/calib_rgb_rehearsal.py --headless 60
"""
import argparse
import time

import cv2
import numpy as np
import pyrealsense2 as rs

SERIAL = "233522078685"
PATTERN = (4, 3)          # (cols-1, rows-1) == inner_corner_size, matches C++
FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
COLOR_W, COLOR_H, COLOR_FPS = 1280, 720, 15
DEPTH_W, DEPTH_H, DEPTH_FPS = 1280, 720, 6


def start_pipeline():
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(SERIAL)
    cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.rgb8, COLOR_FPS)
    cfg.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, DEPTH_FPS)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    return pipe, align, depth_scale


def center_distance_m(depth_frame, depth_scale):
    d = np.asanyarray(depth_frame.get_data())
    h, w = d.shape
    patch = d[h // 2 - 10:h // 2 + 10, w // 2 - 10:w // 2 + 10]
    valid = patch[patch > 0]
    return float(np.median(valid) * depth_scale) if valid.size else float("nan")


def detect(color_rgb):
    gray = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2GRAY)
    found, corners = cv2.findChessboardCorners(gray, PATTERN, flags=FLAGS)
    if found:
        cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TermCriteria_EPS + cv2.TermCriteria_COUNT, 30, 0.1),
        )
    return found, corners


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", type=int, default=0,
                    help="grab N frames, print detection rate, no window")
    args = ap.parse_args()

    pipe, align, depth_scale = start_pipeline()
    print(f"D435i {SERIAL} up. pattern=(4,3)=12 corners, board 5x4 @30mm.")
    for _ in range(5):
        pipe.wait_for_frames()

    hits = total = 0
    try:
        if args.headless:
            for _ in range(args.headless):
                fs = align.process(pipe.wait_for_frames())
                c = fs.get_color_frame()
                z = fs.get_depth_frame()
                if not c:
                    continue
                total += 1
                rgb = np.asanyarray(c.get_data())
                found, _ = detect(rgb)
                hits += 1 if found else 0
                dist = center_distance_m(z, depth_scale) if z else float("nan")
                print(f"frame {total:3d}: {'12/12 FOUND' if found else 'not found':11s}  center~{dist:.3f} m")
            print(f"\ndetection rate: {hits}/{total}"
                  + ("  (board not in view — that's fine, pipeline verified)" if hits == 0 else ""))
            return

        last = time.time()
        fps = 0.0
        while True:
            fs = align.process(pipe.wait_for_frames())
            c = fs.get_color_frame()
            z = fs.get_depth_frame()
            if not c:
                continue
            rgb = np.asanyarray(c.get_data())
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            found, corners = detect(rgb)
            if found:
                cv2.drawChessboardCorners(bgr, PATTERN, corners, found)
            dist = center_distance_m(z, depth_scale) if z else float("nan")
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-3))
            last = now
            color = (0, 255, 0) if found else (0, 0, 255)
            cv2.putText(bgr, f"{'12/12 corners' if found else 'NO PATTERN'}  d~{dist:.3f}m  {fps:4.1f}fps",
                        (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(bgr, "rehearse 0.25-0.72m poses | q quit  (saves nothing)",
                        (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow("D435i RGB checkerboard rehearsal (4x3)", bgr)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
