#!/usr/bin/env python3
"""Build professor-facing figures from preserved IR evidence.

The Stage 1E attempt 05 recorder did not save full thermal phase frames. This
script therefore keeps two evidence types visibly separate:

1. real 16-bit thermal orientation snapshots from the same Lepton rig; and
2. plots derived from the immutable attempt 05 summary measurements.

It also copies selected older whole-hand/foam thermal montages as explicitly
labelled context. Those older images are not Stage 1E direct-pinch evidence.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
ASSET_DIR = PACKAGE_DIR / "assets"
RAW_DIR = ASSET_DIR / "raw_same_rig_orientation"

SUMMARY_PATH = REPO_ROOT / "scratch_lepton/stage1e_tip_pinch_signal_05_summary.json"
# The preserved evidence moved under local/ when this repository split out of
# the meta-workspace; nothing here assumes a checkout location any more.
ORIENTATION_DIR = (
    REPO_ROOT / "local/scratch_lepton/stage0b_thermal_orientation_20260727"
)

ARCHIVE_ROOT = REPO_ROOT / "local/exports"
LEGACY_ROOT = (
    ARCHIVE_ROOT
    / "ir_archive/hand_pressure_20260713"
    / "ir_hand_pressure_hysteresis/webai_review_20260713/processed"
)
LEGACY_SEQUENCE = (
    ARCHIVE_ROOT
    / "ir_archive/hand_pressure_20260713"
    / "ir_hand_pressure_combined_review_20260713/time_control_final/figures"
    / "hand_foam_rep04_thermal_before_to_after_montage_2x2.png"
)

PHASES = ("record_just_touch", "record_press_hard", "record_return_touch")
PHASE_LABELS = ("just touch", "press hard", "return touch")
TIP_COLORS = {"thumb_tip": "#d62728", "index_tip": "#1f77b4"}


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _load_orientation_images() -> tuple[list[Path], list[np.ndarray]]:
    paths = sorted(ORIENTATION_DIR.glob("ir_snapshot_*.png"))
    if len(paths) != 3:
        raise RuntimeError(f"expected three orientation snapshots, found {len(paths)}")
    arrays = [np.asarray(Image.open(path), dtype=np.float64) for path in paths]
    return paths, arrays


def _copy_source_assets(orientation_paths: list[Path]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for source in orientation_paths:
        shutil.copy2(source, RAW_DIR / source.name)

    copies = {
        LEGACY_ROOT / "ir_hysteresis_rep03_montage.png":
            ASSET_DIR / "legacy_whole_hand_foam_rep03_actual_frames.png",
        LEGACY_ROOT / "ir_hysteresis_rep04_montage.png":
            ASSET_DIR / "legacy_whole_hand_foam_rep04_actual_frames.png",
        LEGACY_SEQUENCE:
            ASSET_DIR / "legacy_passive_foam_contact_actual_sequence.png",
    }
    for source, target in copies.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, target)


def _orientation_figure(paths: list[Path], arrays: list[np.ndarray]) -> None:
    stacked = np.concatenate([array.ravel() for array in arrays])
    vmin, vmax = np.percentile(stacked, (1.0, 99.0))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
    last_image = None
    for index, (axis, path, array) in enumerate(zip(axes, paths, arrays), start=1):
        last_image = axis.imshow(
            array,
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        axis.set_title(f"Same-rig orientation snapshot {index}\n{path.name}")
        axis.set_axis_off()

    assert last_image is not None
    colorbar = fig.colorbar(last_image, ax=axes, fraction=0.022, pad=0.02)
    colorbar.set_label("16-bit raw count (shared display scale)")
    fig.suptitle(
        "Real Lepton frames from the Stage 0B orientation check",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.01,
        "These are real frames from the same Lepton rig, but they are NOT "
        "touch/hard/return frames from attempt 05. The recorder did not save "
        "full attempt-05 phase images.",
        ha="center",
        va="top",
        fontsize=10,
        color="#7f0000",
    )
    _save_figure(fig, ASSET_DIR / "same_rig_actual_thermal_snapshots.png")


def _group_arrays(summary: dict) -> dict[str, np.ndarray]:
    groups = summary["groups"]
    return {
        "x": np.asarray([group["group_index"] + 1 for group in groups]),
        "primary": np.asarray([group["primary_effect"] for group in groups]),
        "thumb": np.asarray(
            [
                group["fingertips"]["thumb_tip"]["corrected_exact_effect"]
                for group in groups
            ]
        ),
        "index": np.asarray(
            [
                group["fingertips"]["index_tip"]["corrected_exact_effect"]
                for group in groups
            ]
        ),
        "center_shift": np.asarray(
            [group["press_center_shift_px"] for group in groups]
        ),
    }


def _draw_primary_effects(axis: plt.Axes, values: dict[str, np.ndarray]) -> None:
    colors = np.where(values["primary"] >= 0, "#4c78a8", "#e45756")
    axis.bar(values["x"], values["primary"], color=colors)
    axis.axhline(0.0, color="black", linewidth=1)
    for x_value, effect in zip(values["x"], values["primary"]):
        axis.text(
            x_value,
            effect + (5 if effect >= 0 else -5),
            f"{effect:+.2f}",
            ha="center",
            va="bottom" if effect >= 0 else "top",
            fontsize=8,
        )
    axis.set_title("A. Group primary effect changes sign")
    axis.set_xlabel("Independent group")
    axis.set_ylabel("Common-mode-corrected IR effect (count)")
    axis.set_xticks(values["x"])
    axis.grid(axis="y", alpha=0.25)


def _draw_fingertip_effects(axis: plt.Axes, values: dict[str, np.ndarray]) -> None:
    width = 0.36
    axis.bar(
        values["x"] - width / 2,
        values["thumb"],
        width,
        label="thumb tip",
        color=TIP_COLORS["thumb_tip"],
    )
    axis.bar(
        values["x"] + width / 2,
        values["index"],
        width,
        label="index tip",
        color=TIP_COLORS["index_tip"],
    )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_title("B. Thumb and index often disagree")
    axis.set_xlabel("Independent group")
    axis.set_ylabel("Corrected hard-vs-touch effect (count)")
    axis.set_xticks(values["x"])
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)


def _draw_phase_trajectories(axis: plt.Axes, summary: dict) -> None:
    for group in summary["groups"]:
        for tip_name in ("thumb_tip", "index_tip"):
            medians = group["fingertips"][tip_name]["phase_medians"]
            phase_values = np.asarray(
                [medians[phase]["corrected_exact_count"] for phase in PHASES]
            )
            phase_values -= phase_values[0]
            axis.plot(
                range(3),
                phase_values,
                marker="o",
                linewidth=1.2,
                alpha=0.48,
                color=TIP_COLORS[tip_name],
            )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_title("C. Per-group phase response is not repeatable")
    axis.set_ylabel("Change from just-touch baseline (count)")
    axis.set_xticks(range(3), PHASE_LABELS)
    axis.grid(alpha=0.25)
    axis.text(
        0.02,
        0.97,
        "red = thumb (6 groups)\nblue = index (6 groups)",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )


def _draw_center_shift(axis: plt.Axes, values: dict[str, np.ndarray]) -> None:
    axis.bar(values["x"], values["center_shift"], color="#72b7b2")
    axis.axhline(1.0, color="#b279a2", linestyle="--", linewidth=1.4)
    axis.set_title("D. Pinch center still moves during hard press")
    axis.set_xlabel("Independent group")
    axis.set_ylabel("Median press-center shift (thermal px)")
    axis.set_xticks(values["x"])
    axis.grid(axis="y", alpha=0.25)
    axis.text(
        0.98,
        0.93,
        "range: "
        f"{values['center_shift'].min():.2f}–{values['center_shift'].max():.2f} px",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )


def _attempt_detail_figure(summary: dict) -> None:
    values = _group_arrays(summary)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    _draw_primary_effects(axes[0, 0], values)
    _draw_fingertip_effects(axes[0, 1], values)
    _draw_phase_trajectories(axes[1, 0], summary)
    _draw_center_shift(axes[1, 1], values)

    fig.suptitle(
        "Stage 1E attempt 05: direct fingertip-pinch IR is not direction-stable",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.945,
        "90/90 valid rows; IR EBA 0.50 < 0.75; geometry EBA 0.50; "
        "IR gain 0.00; recovery ratio 0.6928 > 0.50",
        ha="center",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.012,
        "Derived directly from the immutable attempt-05 summary. Counts are "
        "relative IR measurements, not calibrated temperature or force.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.91))
    _save_figure(fig, ASSET_DIR / "attempt05_signal_instability_detailed.png")


def _overview_figure(
    summary: dict,
    orientation_paths: list[Path],
    orientation_arrays: list[np.ndarray],
) -> None:
    values = _group_arrays(summary)
    stacked = np.concatenate([array.ravel() for array in orientation_arrays])
    vmin, vmax = np.percentile(stacked, (1.0, 99.0))

    fig = plt.figure(figsize=(16, 10.5))
    grid = fig.add_gridspec(2, 6, height_ratios=(0.78, 1.22))
    for index, (path, array) in enumerate(
        zip(orientation_paths, orientation_arrays)
    ):
        axis = fig.add_subplot(grid[0, index * 2 : (index + 1) * 2])
        axis.imshow(
            array,
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        axis.set_title(f"Real same-rig frame {index + 1}\n{path.name}", fontsize=9)
        axis.set_axis_off()

    primary_axis = fig.add_subplot(grid[1, 0:2])
    fingertip_axis = fig.add_subplot(grid[1, 2:4])
    phase_axis = fig.add_subplot(grid[1, 4:6])
    _draw_primary_effects(primary_axis, values)
    _draw_fingertip_effects(fingertip_axis, values)
    _draw_phase_trajectories(phase_axis, summary)

    fig.suptitle(
        "Direct pinch: visible hand, but unstable IR hard-vs-touch signal",
        fontsize=19,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.925,
        "Top: actual 16-bit Lepton orientation frames from the same rig "
        "(not attempt-05 phase frames). Bottom: actual attempt-05 measurements.",
        ha="center",
        fontsize=11,
        color="#7f0000",
    )
    fig.text(
        0.5,
        0.012,
        "Frozen verdict: STOP_BEFORE_STAGE1F — IR EBA 0.50 vs 0.75 gate, "
        "IR gain 0.00, opposite thumb/index median directions, failed recovery.",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.89))
    _save_figure(fig, ASSET_DIR / "professor_overview_attempt05.png")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    summary = _read_json(SUMMARY_PATH)
    orientation_paths, orientation_arrays = _load_orientation_images()
    _copy_source_assets(orientation_paths)
    _orientation_figure(orientation_paths, orientation_arrays)
    _attempt_detail_figure(summary)
    _overview_figure(summary, orientation_paths, orientation_arrays)
    print(f"wrote professor evidence assets to {ASSET_DIR}")


if __name__ == "__main__":
    main()
