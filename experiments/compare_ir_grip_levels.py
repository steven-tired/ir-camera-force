from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from ir_force.ir_dataset import GRIP_LEVELS, TrialSpec, trial_id
from ir_force.ir_diagnostics import CapturePairSummary, summarize_window_pairs


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_levels(value: str) -> tuple[str, ...]:
    levels = tuple(level.strip() for level in value.split(",") if level.strip())
    invalid = [level for level in levels if level not in GRIP_LEVELS]
    if invalid:
        raise argparse.ArgumentTypeError(f"invalid level(s): {', '.join(invalid)}")
    return levels


def _latest_pair(pairs: list[CapturePairSummary]) -> CapturePairSummary | None:
    return pairs[-1] if pairs else None


@dataclass(frozen=True)
class VisibleEvidence:
    continuous_frames: int
    preflight_frames: int

    @property
    def has_any(self) -> bool:
        return self.continuous_frames > 0 or self.preflight_frames > 0


def _visible_evidence(trial_root: Path) -> VisibleEvidence:
    return VisibleEvidence(
        continuous_frames=len(list((trial_root / "flir_visible").glob("frame_*.png"))),
        preflight_frames=len(list((trial_root / "preflight").glob("flir_visible.png"))),
    )


def _trial_pairs(
    root: Path,
    spec: TrialSpec,
    feature_name: str = "ir_features.csv",
) -> tuple[Path, list[CapturePairSummary]]:
    trial_root = root / "trials" / trial_id(spec)
    telemetry_csv = trial_root / "telemetry.csv"
    features_csv = trial_root / feature_name
    if not telemetry_csv.exists() or not features_csv.exists():
        return trial_root, []
    return trial_root, summarize_window_pairs(_load_csv(features_csv), _load_csv(telemetry_csv))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/zhuokai/hand-teleop/datasets/ir_grip_force_viability")
    parser.add_argument("--object", required=True, dest="object_name")
    parser.add_argument("--hardness", required=True, choices=["soft", "solid"])
    parser.add_argument("--rep", required=True, type=int)
    parser.add_argument("--levels", type=_parse_levels, default="low,med,high,xhigh")
    parser.add_argument("--features", default="ir_features.csv", help="feature CSV filename to read in each trial")
    args = parser.parse_args()

    root = Path(args.root)
    latest_by_level: dict[str, CapturePairSummary] = {}
    print(
        "level,pair,area_base,area_hold,area_delta,mean_delta_base,mean_delta_hold,"
        "max_delta_hold,load_peak,current_peak,load_final,current_final,gripper_hold_mean,"
        "flags,visible_frames,preflight_visible"
    )
    visible_by_level: dict[str, VisibleEvidence] = {}
    for level in args.levels:
        spec = TrialSpec(args.object_name, args.hardness, level, args.rep)
        trial_root, pairs = _trial_pairs(root, spec, feature_name=args.features)
        visible = _visible_evidence(trial_root)
        visible_by_level[level] = visible
        if len(pairs) > 1:
            print(f"# warning: {trial_root.name} has {len(pairs)} capture pairs; latest pair is used for comparison")
        if not pairs:
            print(f"# warning: missing features/telemetry for {trial_root.name}")
            continue
        for pair in pairs:
            print(
                f"{level},{pair.pair_index},{pair.baseline_area_px:.1f},{pair.hold_area_px:.1f},"
                f"{pair.area_delta_px:.1f},{pair.baseline_mean_delta:.1f},{pair.hold_mean_delta:.1f},"
                f"{pair.hold_max_delta:.1f},{pair.hold_load_peak:.1f},{pair.hold_current_peak:.1f},"
                f"{pair.hold_load_final:.1f},{pair.hold_current_final:.1f},{pair.hold_gripper_mean:.2f},"
                f"{'|'.join(pair.flags) or 'ok'},{visible.continuous_frames},{visible.preflight_frames}"
            )
        latest_by_level[level] = _latest_pair(pairs)

    xhigh = latest_by_level.get("xhigh")
    high = latest_by_level.get("high")
    if xhigh is not None and high is not None:
        print()
        if xhigh.area_delta_px < 0 <= high.area_delta_px:
            print(
                "diagnosis: xhigh force increases but contact-area signal collapses while high increases; "
                "treat xhigh as mechanical/contact failure, not a stronger valid IR sample."
            )
        if xhigh.baseline_area_px > max(high.baseline_area_px * 3, 500):
            print(
                "diagnosis: xhigh baseline mask is much larger than high; check thermal background, object pose, "
                "and camera framing before closure."
            )
    if latest_by_level:
        if not any(visible.has_any for visible in visible_by_level.values()):
            print("diagnosis: no FLIR visible RGB frames were recorded, so tilt/alignment cannot be diagnosed from these trials.")


if __name__ == "__main__":
    main()
