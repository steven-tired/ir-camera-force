from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_THERMAL_PATH = "/dev/video21"
DEFAULT_FLIR_VISIBLE_PATH = "/dev/video20"
THERMAL_STREAM_KIND = "colorized_relative_intensity"
HAND_PRESSURE_STREAM_KIND = "hand_pressure"
VALID_HARDNESS = frozenset({"soft", "solid"})
GRIP_LEVELS = ("low", "med", "high", "xhigh")
SWEEP_GRIP_LEVEL = "sweep"
VALID_GRIP_LEVELS = frozenset((*GRIP_LEVELS, SWEEP_GRIP_LEVEL))


@dataclass(frozen=True)
class TrialSpec:
    object_name: str
    hardness: str
    grip_level: str
    rep: int
    warmed: bool = False

    def __post_init__(self) -> None:
        object_name = self.object_name.strip()
        if not object_name:
            raise ValueError("object_name must be non-empty")
        if self.hardness not in VALID_HARDNESS:
            raise ValueError(f"hardness must be one of {sorted(VALID_HARDNESS)}")
        if self.grip_level not in VALID_GRIP_LEVELS:
            raise ValueError(f"grip_level must be one of {sorted(VALID_GRIP_LEVELS)}")
        if not 1 <= self.rep <= 99:
            raise ValueError("rep must be in the range 1..99")
        object.__setattr__(self, "object_name", object_name)


@dataclass(frozen=True)
class HandPressureTrialSpec:
    surface: str
    contact: str
    rep: int

    def __post_init__(self) -> None:
        surface = self.surface.strip()
        contact = self.contact.strip()
        if not surface:
            raise ValueError("surface must be non-empty")
        if not contact:
            raise ValueError("contact must be non-empty")
        if not 1 <= self.rep <= 99:
            raise ValueError("rep must be in the range 1..99")
        object.__setattr__(self, "surface", surface)
        object.__setattr__(self, "contact", contact)


@dataclass(frozen=True)
class TrialPaths:
    root: Path
    trial_id: str
    metadata_path: Path
    telemetry_csv: Path
    thermal_dir: Path
    bird_dir: Path
    flir_visible_dir: Path
    preflight_dir: Path
    overlays_dir: Path
    plots_dir: Path


def _slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def trial_id(spec: TrialSpec) -> str:
    warmed_suffix = "_warmed" if spec.warmed else ""
    return (
        f"{_slug(spec.object_name)}_{_slug(spec.hardness)}_"
        f"{_slug(spec.grip_level)}_rep{spec.rep:02d}{warmed_suffix}"
    )


def hand_pressure_trial_id(spec: HandPressureTrialSpec) -> str:
    return (
        f"hand-pressure_{_slug(spec.surface)}_{_slug(spec.contact)}_"
        f"sweep_rep{spec.rep:02d}"
    )


def _create_trial_paths_for_id(root: Path, tid: str) -> TrialPaths:
    trial_root = root / "trials" / tid
    paths = TrialPaths(
        root=trial_root,
        trial_id=tid,
        metadata_path=trial_root / "metadata.json",
        telemetry_csv=trial_root / "telemetry.csv",
        thermal_dir=trial_root / "thermal",
        bird_dir=trial_root / "bird",
        flir_visible_dir=trial_root / "flir_visible",
        preflight_dir=trial_root / "preflight",
        overlays_dir=trial_root / "overlays",
        plots_dir=trial_root / "plots",
    )
    for directory in (
        paths.root,
        paths.thermal_dir,
        paths.bird_dir,
        paths.flir_visible_dir,
        paths.preflight_dir,
        paths.overlays_dir,
        paths.plots_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def create_trial_paths(root: Path, spec: TrialSpec) -> TrialPaths:
    return _create_trial_paths_for_id(root, trial_id(spec))


def create_hand_pressure_trial_paths(root: Path, spec: HandPressureTrialSpec) -> TrialPaths:
    return _create_trial_paths_for_id(root, hand_pressure_trial_id(spec))


def write_metadata(paths: TrialPaths, spec: TrialSpec, extra: dict[str, object]) -> None:
    metadata: dict[str, Any] = {
        "trial_id": paths.trial_id,
        **asdict(spec),
        "thermal_path": DEFAULT_THERMAL_PATH,
        "flir_visible_path": DEFAULT_FLIR_VISIBLE_PATH,
        "thermal_stream_kind": THERMAL_STREAM_KIND,
        **extra,
    }
    paths.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def write_hand_pressure_metadata(
    paths: TrialPaths,
    spec: HandPressureTrialSpec,
    extra: dict[str, object],
) -> None:
    metadata: dict[str, Any] = {
        "trial_id": paths.trial_id,
        "experiment_kind": HAND_PRESSURE_STREAM_KIND,
        **asdict(spec),
        "thermal_path": DEFAULT_THERMAL_PATH,
        "flir_visible_path": DEFAULT_FLIR_VISIBLE_PATH,
        "thermal_stream_kind": THERMAL_STREAM_KIND,
        **extra,
    }
    paths.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def append_telemetry_row(csv_path: Path, row: dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    already_exists = csv_path.exists()
    fieldnames = list(row.keys())
    if already_exists:
        with csv_path.open(newline="") as handle:
            reader = csv.reader(handle)
            existing_header = next(reader, None)
        if existing_header is not None and existing_header != fieldnames:
            raise ValueError(
                f"telemetry row keys must match existing CSV header {existing_header}, got {fieldnames}"
            )
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not already_exists:
            writer.writeheader()
        writer.writerow(row)


def _has_telemetry_rows(csv_path: Path) -> bool:
    if not csv_path.exists():
        return False
    with csv_path.open(newline="") as handle:
        return any(True for _ in csv.DictReader(handle))


def trial_has_capture_data(paths: TrialPaths) -> bool:
    if _has_telemetry_rows(paths.telemetry_csv):
        return True
    for directory in (paths.thermal_dir, paths.bird_dir, paths.flir_visible_dir, paths.preflight_dir):
        if any(directory.glob("*.png")):
            return True
    return False


def ensure_fresh_trial(paths: TrialPaths) -> None:
    if trial_has_capture_data(paths):
        raise FileExistsError(
            f"{paths.root} already contains capture data; use a new --rep or pass --append intentionally"
        )
