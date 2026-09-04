from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


_CHECKOUT_ROOT = Path(__file__).resolve().parents[1]
if str(_CHECKOUT_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHECKOUT_ROOT))

from ir_force.data_paths import dataset_root  # noqa: E402

DEFAULT_ROOT = dataset_root("ir_grip_force_viability")


@dataclass(frozen=True)
class ResultGroup:
    name: str
    prefixes: tuple[str, ...]
    output_parts: tuple[str, ...]


GROUPS = (
    ResultGroup(
        name="final_hard_sweep_goal30_to25",
        prefixes=(
            "hard_sweep_rep02_rep03_rep04_goal30_to25_moving_mean_delta_vs_load",
            "hard_sweep_rep02_rep03_rep04_goal30_to25_mean_delta_vs_load",
            "hard_sweep_rep02_rep03_rep04_goal30_to25_summary",
            "hard_sweep_rep02_rep03_rep04_load_area_mean_delta_compare",
        ),
        output_parts=("final_hard_sweep_goal30_to25",),
    ),
    ResultGroup(
        name="hard_sweep_rep02_checks",
        prefixes=(
            "hard_sweep_rep02_goal30_to25_load_ir_area",
            "hard_sweep_rep02_goal30_to25_moving_load_ir_area",
            "hard_sweep_rep02_goal30_to25_with_hold_load_ir_area",
        ),
        output_parts=("per_rep_checks", "rep02"),
    ),
    ResultGroup(
        name="hard_sweep_rep03_checks",
        prefixes=(
            "hard_sweep_rep03_contact_window_montage",
            "hard_sweep_rep03_goal30_to25_load_ir_area",
            "hard_sweep_rep03_goal30_to25_moving_load_ir_area",
            "hard_sweep_rep03_goal30_to25_with_plateau_load_ir_area",
        ),
        output_parts=("per_rep_checks", "rep03"),
    ),
    ResultGroup(
        name="hard_sweep_rep04_checks",
        prefixes=(
            "hard_sweep_rep04_area_spike_high_load_montage",
            "hard_sweep_rep04_goal30_to25_area_and_mean_delta",
            "hard_sweep_rep04_goal30_to25_ir_load",
            "hard_sweep_rep04_trace_load_area_mean_delta",
        ),
        output_parts=("per_rep_checks", "rep04"),
    ),
    ResultGroup(
        name="two_rep_hard_sweep_checks",
        prefixes=(
            "hard_sweep_rep02_rep03_goal30_to25_compare",
            "hard_sweep_rep02_rep03_goal30_to25_load_mean_delta_compare",
            "hard_sweep_rep02_rep03_goal30_to25_summary",
        ),
        output_parts=("older_checks", "two_rep_hard_sweep"),
    ),
    ResultGroup(
        name="early_hard_sweep_checks",
        prefixes=(
            "hard_sweep_rep01_analysis",
            "hard_sweep_rep01_rep02_analysis",
        ),
        output_parts=("older_checks", "early_hard_sweep"),
    ),
    ResultGroup(
        name="soft_sweep_negative",
        prefixes=("soft_sweep_ir_load_summary",),
        output_parts=("soft_sweep_negative",),
    ),
    ResultGroup(
        name="sweep_overview",
        prefixes=("sweep_ir_load_analysis",),
        output_parts=("overview", "sweep_ir_load_analysis"),
    ),
    ResultGroup(
        name="dataset_metadata",
        prefixes=("grip_targets",),
        output_parts=("dataset_metadata",),
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


def _matches_prefix(path: Path, prefix: str) -> bool:
    return path.name == prefix or path.name.startswith(prefix + ".") or path.name.startswith(prefix + "_")


def _matching_group(path: Path) -> ResultGroup | None:
    matches = [group for group in GROUPS if any(_matches_prefix(path, prefix) for prefix in group.prefixes)]
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
