from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ir-grip")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ir_force.ir_report import load_csv_rows, summarize_windows, write_summary_json


def _read_rgb(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _frame_paths(trial_dir: Path, stream: str) -> list[Path]:
    paths = sorted((trial_dir / stream).glob("frame_*.png"))
    if not paths:
        raise RuntimeError(f"no {stream} frames found in {trial_dir}")
    return paths


def _series(rows: list[dict[str, str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def write_trial_report(trial_dir: Path, out_png: Path, out_json: Path) -> None:
    feature_rows = load_csv_rows(trial_dir / "ir_features.csv")
    telemetry_rows = load_csv_rows(trial_dir / "telemetry.csv")
    summary = summarize_windows(feature_rows, telemetry_rows)
    write_summary_json(trial_dir, out_json)

    windows = summary["windows"]
    baseline_start, baseline_stop = windows[0]
    hold_start, hold_stop = windows[1] if len(windows) > 1 else windows[0]
    thermal_paths = _frame_paths(trial_dir, "thermal")
    overlay_paths = _frame_paths(trial_dir, "overlays")
    bird_paths = _frame_paths(trial_dir, "bird")

    baseline_index = min(max(baseline_stop - 1, baseline_start), len(thermal_paths) - 1)
    hold_index = min(max(hold_stop - 1, hold_start), len(thermal_paths) - 1)
    x = list(range(len(feature_rows)))

    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    grid = fig.add_gridspec(3, 4, height_ratios=[1.2, 1.0, 1.0])

    image_specs = [
        ("baseline thermal", thermal_paths[baseline_index]),
        ("hold thermal", thermal_paths[hold_index]),
        ("hold overlay", overlay_paths[hold_index]),
        ("bird view", bird_paths[hold_index]),
    ]
    for col, (title, path) in enumerate(image_specs):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(_read_rgb(path))
        ax.set_title(f"{title}\n{path.name}", fontsize=9)
        ax.axis("off")

    area_ax = fig.add_subplot(grid[1, :2])
    area_ax.plot(x, _series(feature_rows, "area_px"), color="tab:red", linewidth=1.8)
    area_ax.axvline(hold_start, color="black", linestyle="--", linewidth=1)
    area_ax.set_title("IR mask area")
    area_ax.set_xlabel("frame")
    area_ax.set_ylabel("pixels")

    delta_ax = fig.add_subplot(grid[1, 2:])
    delta_ax.plot(x, _series(feature_rows, "mean_delta"), label="mean_delta", color="tab:blue")
    delta_ax.plot(x, _series(feature_rows, "max_delta"), label="max_delta", color="tab:orange")
    delta_ax.axvline(hold_start, color="black", linestyle="--", linewidth=1)
    delta_ax.set_title("Relative thermal delta")
    delta_ax.set_xlabel("frame")
    delta_ax.legend(fontsize=8)

    pos_ax = fig.add_subplot(grid[2, :2])
    pos_ax.plot(x, _series(telemetry_rows, "gripper_pos"), label="gripper_pos", color="tab:green")
    pos_ax.plot(x, _series(telemetry_rows, "goal_gripper_pos"), label="goal", color="tab:gray", linestyle=":")
    pos_ax.axvline(hold_start, color="black", linestyle="--", linewidth=1)
    pos_ax.set_title("Gripper position")
    pos_ax.set_xlabel("frame")
    pos_ax.legend(fontsize=8)

    load_ax = fig.add_subplot(grid[2, 2:])
    load_ax.plot(x, _series(telemetry_rows, "present_load"), label="present_load", color="tab:purple")
    load_ax.plot(x, _series(telemetry_rows, "present_current"), label="present_current", color="tab:brown")
    load_ax.axvline(hold_start, color="black", linestyle="--", linewidth=1)
    load_ax.set_title("Gripper telemetry")
    load_ax.set_xlabel("frame")
    load_ax.legend(fontsize=8)

    baseline = summary["baseline"]
    hold = summary["hold"]
    change = summary["change"]
    fig.suptitle(
        (
            f"{trial_dir.name}: area {baseline['area_px_mean']:.1f} -> {hold['area_px_mean']:.1f} px "
            f"(delta {change['area_px_mean']:+.1f}); "
            f"load peak {hold['present_load_max']:.1f}"
        ),
        fontsize=12,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trial_dir", type=Path)
    parser.add_argument("--out-png", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    trial_dir = args.trial_dir
    out_png = args.out_png or trial_dir / "trial_report.png"
    out_json = args.out_json or trial_dir / "trial_report.json"
    write_trial_report(trial_dir, out_png, out_json)
    print(f"wrote {out_png}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
