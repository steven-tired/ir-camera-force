from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from ir_force.data_paths import workspace_root


WORKSPACE = workspace_root()
DEFAULT_TRIAL = (
    WORKSPACE
    / "datasets/ir_foam_compression/trials"
    / "foam-compression_foam-20260715retry2_foam_zhuokai_rep08"
)
DEFAULT_OUTPUT = WORKSPACE / "exports/gpt_pro_ir_foam_rep08_review_20260715"


@dataclass(frozen=True)
class ReviewCase:
    case_id: str
    step_name: str
    purpose: str


@dataclass(frozen=True)
class SelectedFrame:
    case_id: str
    frame: int
    step_name: str
    purpose: str


REVIEW_CASES = (
    ReviewCase("released_baseline", "steady_state_s01_step00_r", "released baseline before contact"),
    ReviewCase("near_no_contact", "steady_state_s01_step03_n", "near foam but no contact control"),
    ReviewCase("just_contact", "steady_state_s01_step05_c0", "just contact, no material compression"),
    ReviewCase("c10_steady", "steady_state_s01_step09_c10", "10 percent compression steady hold"),
    ReviewCase("c20_steady", "steady_state_s01_step13_c20", "20 percent compression steady hold"),
    ReviewCase("c30_steady", "steady_state_s01_step19_c30", "30 percent compression steady hold"),
    ReviewCase("c10_loading", "hysteresis_s03_step01_c10", "10 percent while loading"),
    ReviewCase("c20_loading", "hysteresis_s03_step02_c20", "20 percent while loading"),
    ReviewCase("c30_loading", "hysteresis_s03_step03_c30", "30 percent while loading"),
    ReviewCase("c20_unloading", "hysteresis_s03_step04_c20", "20 percent while unloading"),
    ReviewCase("c10_unloading", "hysteresis_s03_step05_c10", "10 percent while unloading"),
    ReviewCase("released_after_hysteresis", "hysteresis_s03_step07_r", "released after the hysteresis cycle"),
)


def _truth(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _frame_id(row: dict[str, str]) -> int:
    return int(row["frame"])


def _elapsed(row: dict[str, str]) -> float:
    return float(row["step_elapsed_s"])


def select_representative_frames(rows: Iterable[dict[str, str]]) -> list[SelectedFrame]:
    """Choose one late, valid stable-hold frame for each predeclared review case."""
    row_list = list(rows)
    selected: list[SelectedFrame] = []
    for case in REVIEW_CASES:
        candidates = [
            row
            for row in row_list
            if row.get("step_name") == case.step_name
            and row.get("phase") == "stable_hold"
            and _truth(row.get("marker_detected"))
            and not _truth(row.get("frozen_frame_flag"))
        ]
        if not candidates:
            raise ValueError(f"no valid stable frame for {case.case_id}: {case.step_name}")
        final = max(candidates, key=_elapsed)
        selected.append(
            SelectedFrame(
                case_id=case.case_id,
                frame=_frame_id(final),
                step_name=case.step_name,
                purpose=case.purpose,
            )
        )
    return selected


def _copy(path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_rows(trial: Path) -> tuple[list[SelectedFrame], dict[int, dict[str, str]]]:
    feature_by_frame = {row["frame"]: row for row in _read_csv(trial / "frame_features.csv")}
    rows: list[dict[str, str]] = []
    for telemetry in _read_csv(trial / "telemetry.csv"):
        features = feature_by_frame.get(telemetry["frame"])
        if features is None:
            continue
        rows.append({**telemetry, "frozen_frame_flag": features.get("frozen_frame_flag", "")})
    selected = select_representative_frames(rows)
    return selected, {int(row["frame"]): row for row in rows}


def _load_for_sheet(path: Path, stream: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if stream == "thermal":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if stream == "oak_rgb":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        normalized = np.zeros(image.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(finite, [5, 95])
        if high <= low:
            normalized = np.zeros(image.shape, dtype=np.uint8)
        else:
            normalized = np.clip((image.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def _write_contact_sheets(materials: Path, trial: Path, selected: list[SelectedFrame]) -> list[Path]:
    output = materials / "visuals"
    output.mkdir(parents=True, exist_ok=True)
    streams = (("thermal", "FLIR thermal"), ("oak_rgb", "OAK RGB"), ("oak_depth", "OAK depth"))
    sheets: list[Path] = []
    for part, cases in enumerate((selected[:6], selected[6:]), start=1):
        width, height, header = 320, 240, 44
        canvas = np.full((header + len(cases) * (height + header), len(streams) * width, 3), 250, dtype=np.uint8)
        for column, (_, stream_label) in enumerate(streams):
            cv2.putText(canvas, stream_label, (column * width + 8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        for row_index, item in enumerate(cases):
            top = header + row_index * (height + header)
            label = f"{item.case_id} | frame {item.frame:06d}"
            cv2.putText(canvas, label, (8, top + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            for column, (stream, _) in enumerate(streams):
                source = trial / stream / f"frame_{item.frame:06d}.png"
                image = _load_for_sheet(source, stream)
                resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST if stream == "thermal" else cv2.INTER_AREA)
                canvas[top + header : top + header + height, column * width : (column + 1) * width] = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        path = output / f"selected_raw_contact_sheet_{part:02d}.png"
        cv2.imwrite(str(path), canvas)
        sheets.append(path)
    return sheets


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n")


def _write_docs(materials: Path, trial: Path, selected: list[SelectedFrame]) -> None:
    _write_text(
        materials / "README_FOR_GPT_PRO.md",
        f"""
# Rep08 IR Foam Compression Review Pack

## Scope

This package contains only `{trial.name}`, the one complete fixed-geometry
foam-compression recording. It deliberately excludes earlier prompt-labelled
trials and incomplete repetitions.

The scientific question is whether this setup contains evidence that a
colorized FLIR ONE signal can support a slow binary `hard press` versus
`not hard press` decision. This is not a force-estimation claim.

## Reference and Labels

The reference is geometric foam compression measured from the OAK RGB distance
between two fixed black markers:

`compression_pct = 100 * (d0_px - d_px) / d0_px`.

For the auxiliary binary screen only:

- `hard`: actual marker compression >= 25 percent.
- `not hard`: actual marker compression <= 10 percent.
- 10 to 25 percent: excluded as ambiguous.

There is no force sensor and no Newton-force ground truth.

## Fixed Analysis

The primary thermal feature is the foam-center palette median normalized by
fixed room and warm reference patches. Stable-hold frames are deduplicated by
thermal image hash; frozen frames, missing markers, and attempts marked invalid
are excluded. Step summaries use the final three seconds of each valid hold.

## Observed Single-Trial Result

- 3,138 valid deduplicated stable-hold frames.
- 36 usable step summaries for the binary screen: 7 hard and 29 not hard.
- Spearman compression versus primary IR feature: rho = -0.415, p = 0.0056.
- Rank AUC for the single trial, where lower feature values mean hard: 0.724.
- Five attempted actions were marked invalid and excluded.

These are descriptive single-recording results, not a trained or externally
validated classifier. They do not establish generalization across days,
repositioning, people, foam pieces, or camera gain states.

## Contents

- `tables/`: complete rep08 metadata, events, telemetry, and frame features.
- `analysis/`: primary relation figure, step summary, and binary screen.
- `preflight/`: camera/marker placement snapshots and automatic preflight report.
- `selected_raw/`: 12 selected, synchronized thermal/OAK RGB/OAK depth frame triplets.
- `visuals/`: two contact sheets of those triplets for direct visual review.
- `source_code/`: capture and analysis code relevant to rep08.

The 12 raw examples cover released baseline, near/no contact, just contact,
C10/C20/C30 steady states, and C10/C20/C30 loading and unloading states.
Their exact frame IDs and metadata are in `selected_raw/selected_frame_index.csv`.
""",
    )
    _write_text(
        materials / "GPT_PRO_REVIEW_REQUEST.md",
        """
You are reviewing one complete fixed-geometry foam-compression experiment.

Goal: decide whether its colorized FLIR ONE signal has enough real evidence to
justify collecting the next dataset for a hard-press / not-hard-press
classifier. Do not treat this as force estimation and do not claim that this
single recording validates a deployable classifier.

Please inspect the contact sheets and selected synchronized thermal/OAK frames,
then use the full telemetry and feature tables to answer:

1. Is the reported inverse compression-to-foam-center-IR relation plausibly
   related to compression, or more likely AGC, hand visibility, marker tracking,
   ROI placement, reference normalization, or time drift?
2. Does the N (near but no contact) control distinguish hand-presence effects
   from compression effects in this recording?
3. Does the loading/unloading subset show meaningful hysteresis that would make
   a memoryless hard/not-hard classifier unsafe?
4. Is the current binary split (>=25% versus <=10%) defensible as a next-round
   label, and what minimal number of independent recordings is needed before
   fitting any classifier?
5. Recommend the smallest next experiment and a blocked validation design.

Use the raw images and the tables, not the p-value alone. State clearly what
can and cannot be concluded from a single recording.
""",
    )
    _write_text(
        materials / "UPLOAD_INSTRUCTIONS.md",
        """
# Upload

Run `./send_rep08_to_gpt_pro.sh --dry-run` from this export directory to list
the exact attachments. Run `./send_rep08_to_gpt_pro.sh --send` to start the
local WebAI Chrome profile and attach the bundle plus the two visual contact
sheets to GPT Pro. The script requires a typed confirmation before any data is
sent externally.
""",
    )


def _copy_selected_raw(materials: Path, trial: Path, selected: list[SelectedFrame], rows: dict[int, dict[str, str]]) -> None:
    output = materials / "selected_raw"
    output.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, object]] = []
    for item in selected:
        for stream in ("thermal", "oak_rgb", "oak_depth"):
            source = trial / stream / f"frame_{item.frame:06d}.png"
            destination = output / f"{item.case_id}_{stream}_frame_{item.frame:06d}.png"
            _copy(source, destination)
        row = rows[item.frame]
        index_rows.append(
            {
                **asdict(item),
                "state": row.get("state", ""),
                "block": row.get("block", ""),
                "step_index": row.get("step_index", ""),
                "action_attempt": row.get("action_attempt", ""),
                "step_elapsed_s": row.get("step_elapsed_s", ""),
                "compression_pct": row.get("compression_pct", ""),
                "marker_distance_px": row.get("marker_distance_px", ""),
            }
        )
    with (output / "selected_frame_index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)


def _copy_materials(materials: Path, trial: Path) -> None:
    for filename in ("metadata.json", "events.csv", "telemetry.csv", "frame_features.csv"):
        _copy(trial / filename, materials / "tables" / filename)
    shutil.copytree(trial / "analysis", materials / "analysis")
    metadata = json.loads((trial / "metadata.json").read_text())
    preflight = Path(metadata.get("preflight_report_path", ""))
    if preflight.is_dir():
        shutil.copytree(preflight, materials / "preflight")
    project = Path(__file__).resolve().parent
    for source in (
        project / "analyze_ir_foam_compression.py",
        project / "record_ir_foam_compression_experiment.py",
        project / "lerobot_teleoperator_so101_webcam/ir_foam_compression.py",
        project / "lerobot_teleoperator_so101_webcam/ir_features.py",
        Path(__file__).resolve(),
    ):
        _copy(source, materials / "source_code" / source.name)


def _write_manifest(output: Path, materials: Path, selected: list[SelectedFrame]) -> None:
    files = [path for path in sorted(materials.rglob("*")) if path.is_file()]
    payload = {
        "scope": "rep08 only",
        "trial_id": DEFAULT_TRIAL.name,
        "selected_frames": [asdict(item) for item in selected],
        "files": [
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        ],
    }
    (output / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_review_bundle(trial: Path, output: Path, *, overwrite: bool = False) -> Path:
    if not trial.is_dir():
        raise FileNotFoundError(trial)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to rebuild it")
        shutil.rmtree(output)
    materials = output / "materials"
    materials.mkdir(parents=True)
    selected, rows = _selected_rows(trial)
    _copy_materials(materials, trial)
    _copy_selected_raw(materials, trial, selected, rows)
    _write_contact_sheets(materials, trial, selected)
    _write_docs(materials, trial, selected)
    _write_manifest(output, materials, selected)
    archive = shutil.make_archive(str(output / "rep08_gpt_pro_review_bundle"), "zip", root_dir=output, base_dir="materials")
    return Path(archive)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the selected rep08-only GPT Pro review package.")
    parser.add_argument("--trial", type=Path, default=DEFAULT_TRIAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    archive = build_review_bundle(args.trial, args.output, overwrite=args.overwrite)
    print(f"wrote {archive}")


if __name__ == "__main__":
    main()
