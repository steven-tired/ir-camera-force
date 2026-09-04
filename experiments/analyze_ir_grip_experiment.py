from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ir-grip")

import matplotlib.pyplot as plt

from ir_force.ir_analysis import (
    decision_from_summaries,
    summarize_trial,
    write_summary_json,
)
from ir_force.data_paths import dataset_root


def _plot_area_vs_current(summaries, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {"soft": "tab:blue", "solid": "tab:orange"}
    plt.figure(figsize=(7, 5))
    for item in summaries:
        if item.warmed:
            continue
        plt.scatter(item.peak_current, item.hold_mean_area_px, color=colors.get(item.hardness, "tab:gray"))
        plt.text(item.peak_current, item.hold_mean_area_px, f"{item.object_name}:{item.grip_level}", fontsize=7)
    plt.xlabel("Peak gripper Present_Current raw")
    plt.ylabel("Hold mean IR contact area px")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(dataset_root("ir_grip_force_viability")))
    args = parser.parse_args()
    root = Path(args.root)
    summaries = [summarize_trial(path) for path in sorted((root / "trials").glob("*")) if path.is_dir()]
    decision = decision_from_summaries(summaries)
    write_summary_json(summaries, decision, root / "analysis" / "summary.json")
    _plot_area_vs_current(summaries, root / "analysis" / "ir_area_vs_current.png")
    print(decision)
    print(f"wrote {root / 'analysis' / 'summary.json'}")
    print(f"wrote {root / 'analysis' / 'ir_area_vs_current.png'}")


if __name__ == "__main__":
    main()
