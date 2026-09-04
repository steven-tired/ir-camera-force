from __future__ import annotations

import argparse
from pathlib import Path

from ir_force.ir_capture import (
    FrameSource,
    LeptonUDPSource,
    OpenCVCameraSource,
    capture_setup_snapshot,
    record_labeled_camera_window,
)
from ir_force.ir_dataset import (
    HandPressureTrialSpec,
    LEPTON_THERMAL_STREAM_KIND,
    THERMAL_STREAM_KIND,
    create_hand_pressure_trial_paths,
    ensure_fresh_trial,
    write_hand_pressure_metadata,
)
from ir_force.data_paths import dataset_root


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", default="foam", help="contact surface, e.g. foam, brick, wood")
    parser.add_argument("--contact", default="whole hand", help="hand contact, e.g. whole hand, fingertip, palm")
    parser.add_argument("--rep", required=True, type=int)
    parser.add_argument("--thermal", default="/dev/video21")
    parser.add_argument(
        "--lepton-udp",
        type=int,
        nargs="?",
        const=8080,
        default=None,
        metavar="PORT",
        help="read thermal frames from the Pi Lepton UDP streamer instead of --thermal",
    )
    parser.add_argument("--bird", required=True)
    parser.add_argument("--flir-visible", default=None)
    parser.add_argument("--record-flir-visible", action="store_true")
    parser.add_argument("--root", default=str(dataset_root("ir_hand_pressure_viability")))
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--baseline-s", type=float, default=2.0)
    parser.add_argument("--press-s", type=float, default=5.0)
    parser.add_argument("--hold-s", type=float, default=1.0)
    parser.add_argument("--release-s", type=float, default=5.0)
    parser.add_argument("--thermal-roi", default="", help="optional ROI note as x,y,width,height")
    parser.add_argument("--notes", default="", help="free-form experiment note written to metadata")
    parser.add_argument("--append", action="store_true")
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)
    if args.flir_visible is None:
        # The FLIR One visible channel does not exist on the Lepton path.
        args.flir_visible = "" if args.lepton_udp is not None else "/dev/video20"
    return args


def _thermal_source_label(args: argparse.Namespace) -> str:
    if args.lepton_udp is not None:
        return f"lepton-udp:{args.lepton_udp}"
    return args.thermal


def _thermal_stream_kind(args: argparse.Namespace) -> str:
    if args.lepton_udp is not None:
        return LEPTON_THERMAL_STREAM_KIND
    return THERMAL_STREAM_KIND


def _build_thermal_source(args: argparse.Namespace) -> FrameSource:
    if args.lepton_udp is not None:
        return LeptonUDPSource(port=args.lepton_udp)
    return OpenCVCameraSource(args.thermal)


def _prepare_trial(args: argparse.Namespace):
    spec = HandPressureTrialSpec(
        surface=args.surface,
        contact=args.contact,
        rep=args.rep,
    )
    paths = create_hand_pressure_trial_paths(Path(args.root), spec)
    if not args.append:
        ensure_fresh_trial(paths)
    write_hand_pressure_metadata(
        paths,
        spec,
        {
            "thermal_path": _thermal_source_label(args),
            "thermal_stream_kind": _thermal_stream_kind(args),
            "bird_path": args.bird,
            "flir_visible_path": args.flir_visible,
            "record_flir_visible": args.record_flir_visible,
            "recording_mode": "continuous_pressure_sweep",
            "fps": args.fps,
            "baseline_s": args.baseline_s,
            "press_s": args.press_s,
            "hold_s": args.hold_s,
            "release_s": args.release_s,
            "thermal_roi": args.thermal_roi,
            "notes": args.notes,
        },
    )
    return spec, paths


def _record_trial(
    *,
    args: argparse.Namespace,
    spec: HandPressureTrialSpec,
    paths,
    thermal: FrameSource,
    bird: OpenCVCameraSource,
    visible: OpenCVCameraSource | None,
) -> None:
    print("capturing camera preflight")
    capture_setup_snapshot(paths, thermal=thermal, bird=bird, flir_visible=visible)
    continuous_visible = visible if args.record_flir_visible else None

    common_labels = {
        "surface": spec.surface,
        "contact": spec.contact,
    }

    print("capturing no-contact baseline")
    record_labeled_camera_window(
        paths,
        thermal=thermal,
        bird=bird,
        flir_visible=continuous_visible,
        duration_s=args.baseline_s,
        fps=args.fps,
        labels={**common_labels, "phase": "baseline"},
    )

    print("capturing continuous pressure sweep")
    record_labeled_camera_window(
        paths,
        thermal=thermal,
        bird=bird,
        flir_visible=continuous_visible,
        duration_s=args.press_s,
        fps=args.fps,
        labels={**common_labels, "phase": "pressure_sweep"},
        progress_duration_s=args.press_s,
    )
    if args.hold_s > 0:
        print("capturing high-pressure hold")
        record_labeled_camera_window(
            paths,
            thermal=thermal,
            bird=bird,
            flir_visible=continuous_visible,
            duration_s=args.hold_s,
            fps=args.fps,
            labels={**common_labels, "phase": "hold"},
            progress_duration_s=args.hold_s,
            progress_start=1.0,
            progress_end=1.0,
        )
    if args.release_s > 0:
        print("capturing continuous release")
        record_labeled_camera_window(
            paths,
            thermal=thermal,
            bird=bird,
            flir_visible=continuous_visible,
            duration_s=args.release_s,
            fps=args.fps,
            labels={**common_labels, "phase": "release"},
            progress_duration_s=args.release_s,
            progress_start=1.0,
            progress_end=0.0,
        )
    print(f"saved hand-pressure trial to {paths.root}")


def main() -> None:
    args = _parse_args()
    spec, paths = _prepare_trial(args)

    print("This script records a camera-only hand-pressure IR trial.")
    print("It does not connect to or move the SO-101 robot.")
    print(f"Trial: {paths.trial_id}")
    print(f"Surface/contact: {spec.surface} / {spec.contact}")
    print(f"Baseline: no hand contact for {args.baseline_s:g}s")
    print(f"Continuous pressure sweep: low to high over {args.press_s:g}s")
    if args.hold_s > 0:
        print(f"Hold: keep high pressure for {args.hold_s:g}s")
    if args.release_s > 0:
        print(f"Release: high to no-contact over {args.release_s:g}s")
    if input("Type YES to continue: ").strip() != "YES":
        raise SystemExit("aborted")

    thermal = _build_thermal_source(args)
    bird = OpenCVCameraSource(args.bird)
    visible = OpenCVCameraSource(args.flir_visible) if args.flir_visible else None
    try:
        _record_trial(
            args=args,
            spec=spec,
            paths=paths,
            thermal=thermal,
            bird=bird,
            visible=visible,
        )
    finally:
        thermal.close()
        bird.close()
        if visible is not None:
            visible.close()


if __name__ == "__main__":
    main()
