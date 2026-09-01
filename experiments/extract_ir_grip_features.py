from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from ir_force.ir_features import (
    ThermalROI,
    compute_baseline,
    extract_frame_features,
    load_palette,
    overlay_mask,
    write_features_csv,
)


def _read_frames(paths: list[Path]) -> list:
    frames = []
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise RuntimeError(f"could not read {path}")
        frames.append(frame)
    return frames


def _telemetry_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def skip_reason(trial_dir: Path, baseline_frames: int) -> str | None:
    frame_count = len(list((trial_dir / "thermal").glob("*.png")))
    if frame_count <= baseline_frames:
        return f"too few thermal frames: {frame_count} <= {baseline_frames}"
    telemetry_count = _telemetry_rows(trial_dir / "telemetry.csv")
    if telemetry_count != frame_count:
        return f"thermal frame count {frame_count} does not match telemetry rows {telemetry_count}"
    return None


def parse_roi(value: str) -> ThermalROI | None:
    if not value:
        return None
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("thermal ROI must be x,y,width,height")
    try:
        x, y, width, height = [int(part) for part in parts]
        return ThermalROI(x=x, y=y, width=width, height=height)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def load_palette_for_opencv(path: Path):
    # FLIR palette files are RGB, while frames loaded through OpenCV are BGR.
    return load_palette(path)[:, ::-1].copy()


def extract_trial(
    trial_dir: Path,
    baseline_frames: int,
    noise_sigma: float,
    palette_path: Path | None,
    invert_palette: bool = False,
    roi: ThermalROI | None = None,
    feature_name: str = "ir_features.csv",
    overlay_dir_name: str = "overlays",
) -> Path:
    thermal_dir = trial_dir / "thermal"
    frame_paths = sorted(thermal_dir.glob("*.png"))
    if len(frame_paths) <= baseline_frames:
        raise RuntimeError(f"{trial_dir} has too few thermal frames")

    palette = load_palette_for_opencv(palette_path) if palette_path is not None else None
    baseline = compute_baseline(
        _read_frames(frame_paths[:baseline_frames]),
        palette=palette,
        invert_palette=invert_palette,
        roi=roi,
    )

    features = []
    overlays = trial_dir / overlay_dir_name
    overlays.mkdir(parents=True, exist_ok=True)
    for path in frame_paths:
        frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise RuntimeError(f"could not read {path}")
        item, mask = extract_frame_features(
            frame,
            baseline,
            noise_sigma=noise_sigma,
            frame_name=path.name,
            palette=palette,
            invert_palette=invert_palette,
        )
        features.append(item)
        cv2.imwrite(str(overlays / path.name), overlay_mask(frame, mask))

    out_csv = trial_dir / feature_name
    write_features_csv(features, out_csv)
    return out_csv


def extract_trials(
    root: Path,
    *,
    baseline_frames: int,
    noise_sigma: float,
    palette_path: Path | None,
    invert_palette: bool,
    strict: bool,
    trial_glob: str = "*",
    roi: ThermalROI | None = None,
    feature_name: str = "ir_features.csv",
    overlay_dir_name: str = "overlays",
) -> list[Path]:
    written: list[Path] = []
    for trial_dir in sorted((root / "trials").glob(trial_glob)):
        if not trial_dir.is_dir():
            continue
        reason = skip_reason(trial_dir, baseline_frames)
        if reason is not None:
            message = f"skipping {trial_dir}: {reason}"
            if strict:
                raise RuntimeError(message)
            print(message)
            continue
        out_csv = extract_trial(
            trial_dir,
            baseline_frames,
            noise_sigma,
            palette_path,
            invert_palette=invert_palette,
            roi=roi,
            feature_name=feature_name,
            overlay_dir_name=overlay_dir_name,
        )
        print(f"wrote {out_csv}")
        written.append(out_csv)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/zhuokai/hand-teleop/datasets/ir_grip_force_viability")
    parser.add_argument("--baseline-frames", type=int, default=20)
    parser.add_argument("--noise-sigma", type=float, default=3.0)
    parser.add_argument(
        "--palette",
        default="",
        help="optional flirone-v4l2 palette .raw file, e.g. /tmp/flirone-v4l2/palettes/Iron2.raw",
    )
    parser.add_argument(
        "--invert-palette",
        action="store_true",
        help="invert palette indices so hotter colors map to larger relative values when needed",
    )
    parser.add_argument(
        "--trial-glob",
        default="*",
        help="only extract trial folders matching this glob under root/trials, e.g. '*_rep02'",
    )
    parser.add_argument(
        "--thermal-roi",
        type=parse_roi,
        default=None,
        help="optional contact ROI in thermal-frame pixels as x,y,width,height; pixels outside it are ignored",
    )
    parser.add_argument(
        "--feature-name",
        default="ir_features.csv",
        help="feature CSV filename written inside each trial folder",
    )
    parser.add_argument(
        "--overlay-dir",
        default="overlays",
        help="overlay frame directory name written inside each trial folder",
    )
    parser.add_argument("--strict", action="store_true", help="fail instead of skipping incomplete trials")
    args = parser.parse_args()

    root = Path(args.root)
    palette_path = Path(args.palette) if args.palette else None
    extract_trials(
        root,
        baseline_frames=args.baseline_frames,
        noise_sigma=args.noise_sigma,
        palette_path=palette_path,
        invert_palette=args.invert_palette,
        strict=args.strict,
        trial_glob=args.trial_glob,
        roi=args.thermal_roi,
        feature_name=args.feature_name,
        overlay_dir_name=args.overlay_dir,
    )


if __name__ == "__main__":
    main()
