from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_ROOT = Path("/home/zhuokai/hand-teleop/ir-camera-force/local/datasets/ir_hand_pressure_viability")


@dataclass(frozen=True)
class ResultGroup:
    name: str
    prefixes: tuple[str, ...]
    output_parts: tuple[str, ...]


GROUPS = (
    ResultGroup(
        name="final_hand_foam_time_control",
        prefixes=("foam_rep02_rep03_rep04_rep05_hand_distance_time_control",),
        output_parts=("final_hand_foam_time_control",),
    ),
    ResultGroup(
        name="quality_filtered_hand_foam",
        prefixes=("foam_rep02_rep03_rep04_rep05_hand_distance_quality_filtered",),
        output_parts=("older_runs", "quality_filtered_hand_foam"),
    ),
    ResultGroup(
        name="all_foam_reps_hand_distance",
        prefixes=("foam_rep01_rep02_rep03_rep04_rep05_hand_distance",),
        output_parts=("older_runs", "all_foam_reps_hand_distance"),
    ),
    ResultGroup(
        name="first_post_touch_run",
        prefixes=("hand_pressure_post_touch",),
        output_parts=("scratch_checks", "first_post_touch_run"),
    ),
    ResultGroup(
        name="foam_rep01_check",
        prefixes=("foam_rep01_check",),
        output_parts=("scratch_checks", "foam_rep01_check"),
    ),
    ResultGroup(
        name="foam_rep02_check",
        prefixes=("foam_rep02_check",),
        output_parts=("scratch_checks", "foam_rep02_check"),
    ),
    ResultGroup(
        name="foam_rep02_hand_distance_check",
        prefixes=("foam_rep02_hand_distance_check",),
        output_parts=("scratch_checks", "foam_rep02_hand_distance_check"),
    ),
)


def _artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".png":
        return "figures"
    if suffix in {".md", ".txt"}:
        return "reports"
    if suffix == ".json":
        return "metadata"
    return "other"


def _matching_group(path: Path) -> ResultGroup | None:
    matches = [
        group
        for group in GROUPS
        if any(path.name.startswith(prefix + "_") or path.name == prefix for prefix in group.prefixes)
    ]
    if not matches:
        return None
    return max(matches, key=lambda group: max(len(prefix) for prefix in group.prefixes))


def organize_results(root: Path, *, dry_run: bool = False) -> dict[str, object]:
    root = root.expanduser().resolve()
    organized_root = root / "organized_results"
    manifest: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        "organized_root": str(organized_root),
        "mode": "copy",
        "dry_run": dry_run,
        "groups": {},
        "unmatched": [],
    }

    files = sorted(path for path in root.iterdir() if path.is_file())
    for source in files:
        group = _matching_group(source)
        if group is None:
            manifest["unmatched"].append(source.name)
            continue

        kind = _artifact_kind(source)
        destination_dir = organized_root.joinpath(*group.output_parts, kind)
        destination = destination_dir / source.name
        group_manifest = manifest["groups"].setdefault(
            group.name,
            {
                "destination": str(organized_root.joinpath(*group.output_parts)),
                "files": [],
            },
        )
        group_manifest["files"].append(
            {
                "source": source.name,
                "kind": kind,
                "destination": str(destination.relative_to(root)),
                "bytes": source.stat().st_size,
            }
        )
        if not dry_run:
            destination_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    if not dry_run:
        for group_name, group_manifest in manifest["groups"].items():
            destination = Path(str(group_manifest["destination"]))
            group_manifest_path = destination / "manifest.json"
            group_manifest_path.write_text(
                json.dumps(
                    {
                        "generated_at": manifest["generated_at"],
                        "source_root": manifest["source_root"],
                        "group": group_name,
                        "mode": manifest["mode"],
                        "files": group_manifest["files"],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        organized_root.mkdir(parents=True, exist_ok=True)
        (organized_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = organize_results(args.root, dry_run=args.dry_run)
    group_count = len(manifest["groups"])
    file_count = sum(len(group["files"]) for group in manifest["groups"].values())
    print(f"organized groups: {group_count}")
    print(f"matched files: {file_count}")
    print(f"unmatched files: {len(manifest['unmatched'])}")
    print(f"organized root: {manifest['organized_root']}")


if __name__ == "__main__":
    main()
