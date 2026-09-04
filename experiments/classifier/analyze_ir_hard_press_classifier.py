from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
import sys

_CHECKOUT_ROOT = Path(__file__).resolve().parents[2]
if str(_CHECKOUT_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHECKOUT_ROOT))

from ir_force.data_paths import dataset_root  # noqa: E402

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ir-hard-press")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_ROOT = dataset_root("ir_hard_classifier")
DEFAULT_TRIALS = (
    DEFAULT_ROOT / "trials/hard-classifier_s01_fixed-posture_foam_zk_rep01",
    DEFAULT_ROOT / "trials/oak-squeeze_s01_fixed-posture_foam_zk_rep02",
)
FEATURE_NAMES = (
    "roi_mean",
    "roi_std",
    "delta_mean",
    "delta_std",
    "pos_area",
    "neg_area",
    "l1_delta",
    "l2_delta",
    "delta_p90",
    "delta_p95",
    "delta_p99",
)


@dataclass(frozen=True)
class ThresholdRule:
    sign: int
    threshold: float
    train_balanced_accuracy: float


def _float(value: object) -> float | None:
    try:
        result = float(str(value))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _target_column(rows: list[dict[str, str]]) -> str:
    if not rows:
        raise ValueError("frame_features.csv contains no rows")
    if "target_squeeze_percent" in rows[0]:
        return "target_squeeze_percent"
    if "target_force_percent" in rows[0]:
        return "target_force_percent"
    raise ValueError("frame_features.csv has no prompt target percent column")


def _label_target(target_percent: float, *, hard_min_percent: float, not_hard_max_percent: float) -> int | None:
    if target_percent >= hard_min_percent:
        return 1
    if target_percent <= not_hard_max_percent:
        return 0
    return None


def load_trial(
    trial_dir: Path,
    *,
    stable_window_s: float = 1.0,
    hard_min_percent: float = 70.0,
    not_hard_max_percent: float = 50.0,
) -> list[dict[str, object]]:
    """Return one leakage-resistant final-window sample per prompted target hold."""
    if stable_window_s <= 0:
        raise ValueError("stable_window_s must be positive")
    if not 0 <= not_hard_max_percent < hard_min_percent <= 100:
        raise ValueError("require 0 <= not_hard_max_percent < hard_min_percent <= 100")
    raw_rows = _read_csv(trial_dir / "frame_features.csv")
    target_column = _target_column(raw_rows)
    grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        if row.get("phase") != "target_hold":
            continue
        if row.get("frozen_frame_flag", "").lower() == "true":
            continue
        if row.get("agc_jump_flag", "").lower() == "true":
            continue
        if any(_float(row.get(feature)) is None for feature in FEATURE_NAMES):
            continue
        sequence_id = _float(row.get("sequence_id"))
        step_index = _float(row.get("step_index"))
        timestamp = _float(row.get("timestamp"))
        target_percent = _float(row.get(target_column))
        if sequence_id is None or step_index is None or timestamp is None or target_percent is None:
            continue
        grouped[(int(sequence_id), int(step_index))].append(row)

    samples: list[dict[str, object]] = []
    for (sequence_id, step_index), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: float(row["timestamp"]))
        target_percent = float(rows[0][target_column])
        hard_label = _label_target(
            target_percent,
            hard_min_percent=hard_min_percent,
            not_hard_max_percent=not_hard_max_percent,
        )
        if hard_label is None:
            continue
        final_time = float(rows[-1]["timestamp"])
        final_rows = [row for row in rows if float(row["timestamp"]) >= final_time - stable_window_s]
        if not final_rows:
            continue
        sample: dict[str, object] = {
            "trial_id": trial_dir.name,
            "sequence_id": sequence_id,
            "step_index": step_index,
            "target_percent": target_percent,
            "hard_label": hard_label,
            "source_target_column": target_column,
            "final_window_frame_count": len(final_rows),
        }
        for feature in FEATURE_NAMES:
            sample[feature] = float(fmean(float(row[feature]) for row in final_rows))
        samples.append(sample)
    return samples


def _balanced_metrics(labels: list[int], predictions: list[int], scores: list[float]) -> dict[str, float]:
    y = np.asarray(labels, dtype=int)
    prediction = np.asarray(predictions, dtype=int)
    score = np.asarray(scores, dtype=float)
    positives = y == 1
    negatives = y == 0
    sensitivity = float(np.mean(prediction[positives] == 1)) if np.any(positives) else float("nan")
    specificity = float(np.mean(prediction[negatives] == 0)) if np.any(negatives) else float("nan")
    balanced_accuracy = float(np.nanmean((sensitivity, specificity)))
    if not np.any(positives) or not np.any(negatives):
        rank_auc = float("nan")
    else:
        positive_scores = score[positives]
        negative_scores = score[negatives]
        wins = sum((value > negative_scores).sum() + 0.5 * (value == negative_scores).sum() for value in positive_scores)
        rank_auc = float(wins / (len(positive_scores) * len(negative_scores)))
    return {
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "rank_auc": rank_auc,
    }


def _threshold_candidates(values: list[float]) -> np.ndarray:
    unique = np.sort(np.unique(np.asarray(values, dtype=float)))
    if len(unique) == 1:
        return unique
    scale = max(1.0, float(np.ptp(unique)) * 0.01)
    return np.concatenate(((unique[0] - scale,), (unique[:-1] + unique[1:]) / 2.0, (unique[-1] + scale,)))


def fit_threshold(values: list[float], labels: list[int]) -> ThresholdRule:
    if len(values) != len(labels) or not values:
        raise ValueError("values and labels must be non-empty and equally sized")
    best: ThresholdRule | None = None
    for sign in (1, -1):
        for threshold in _threshold_candidates(values):
            predictions = apply_threshold(ThresholdRule(sign, float(threshold), float("nan")), values)
            metrics = _balanced_metrics(labels, predictions, [sign * (value - threshold) for value in values])
            candidate = ThresholdRule(sign, float(threshold), metrics["balanced_accuracy"])
            if best is None or (candidate.train_balanced_accuracy, candidate.sign, -candidate.threshold) > (
                best.train_balanced_accuracy,
                best.sign,
                -best.threshold,
            ):
                best = candidate
    assert best is not None
    return best


def apply_threshold(rule: ThresholdRule, values: list[float]) -> list[int]:
    return [int(rule.sign * (value - rule.threshold) >= 0.0) for value in values]


def _within_trial_univariate(samples: list[dict[str, object]]) -> list[dict[str, object]]:
    labels = [int(sample["hard_label"]) for sample in samples]
    groups = [int(sample["sequence_id"]) for sample in samples]
    rows: list[dict[str, object]] = []
    for feature in FEATURE_NAMES:
        values = [float(sample[feature]) for sample in samples]
        predictions = [0] * len(samples)
        scores = [0.0] * len(samples)
        for held_group in sorted(set(groups)):
            train_indices = [index for index, group in enumerate(groups) if group != held_group]
            test_indices = [index for index, group in enumerate(groups) if group == held_group]
            rule = fit_threshold([values[index] for index in train_indices], [labels[index] for index in train_indices])
            for index in test_indices:
                predictions[index] = apply_threshold(rule, [values[index]])[0]
                scores[index] = rule.sign * (values[index] - rule.threshold)
        rows.append(
            {
                "trial_id": str(samples[0]["trial_id"]),
                "validation": "leave_one_sequence_out",
                "model": "univariate_threshold",
                "feature": feature,
                "step_sample_count": len(samples),
                "hard_step_count": sum(labels),
                **_balanced_metrics(labels, predictions, scores),
            }
        )
    return rows


def _centroid_prediction(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    deviation = train_x.std(axis=0)
    deviation[deviation == 0.0] = 1.0
    train_z = (train_x - mean) / deviation
    test_z = (test_x - mean) / deviation
    not_hard_centroid = train_z[train_y == 0].mean(axis=0)
    hard_centroid = train_z[train_y == 1].mean(axis=0)
    not_hard_distance = ((test_z - not_hard_centroid) ** 2).sum(axis=1)
    hard_distance = ((test_z - hard_centroid) ** 2).sum(axis=1)
    return (hard_distance < not_hard_distance).astype(int), not_hard_distance - hard_distance


def _within_trial_centroid(samples: list[dict[str, object]]) -> dict[str, object]:
    labels = np.asarray([int(sample["hard_label"]) for sample in samples], dtype=int)
    groups = np.asarray([int(sample["sequence_id"]) for sample in samples], dtype=int)
    values = np.asarray([[float(sample[feature]) for feature in FEATURE_NAMES] for sample in samples], dtype=float)
    predictions = np.zeros(len(samples), dtype=int)
    scores = np.zeros(len(samples), dtype=float)
    for held_group in sorted(set(groups)):
        train_mask = groups != held_group
        test_mask = ~train_mask
        predictions[test_mask], scores[test_mask] = _centroid_prediction(values[train_mask], labels[train_mask], values[test_mask])
    return {
        "trial_id": str(samples[0]["trial_id"]),
        "validation": "leave_one_sequence_out",
        "model": "all_feature_nearest_centroid",
        "feature": "all_features",
        "step_sample_count": len(samples),
        "hard_step_count": int(labels.sum()),
        **_balanced_metrics(labels.tolist(), predictions.tolist(), scores.tolist()),
    }


def _cross_trial_rows(by_trial: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for train_id, train_samples in by_trial.items():
        for test_id, test_samples in by_trial.items():
            if train_id == test_id:
                continue
            train_y = [int(sample["hard_label"]) for sample in train_samples]
            test_y = [int(sample["hard_label"]) for sample in test_samples]
            for feature in FEATURE_NAMES:
                rule = fit_threshold([float(sample[feature]) for sample in train_samples], train_y)
                test_values = [float(sample[feature]) for sample in test_samples]
                predictions = apply_threshold(rule, test_values)
                scores = [rule.sign * (value - rule.threshold) for value in test_values]
                rows.append(
                    {
                        "train_trial_id": train_id,
                        "test_trial_id": test_id,
                        "validation": "leave_one_trial_out",
                        "model": "univariate_threshold",
                        "feature": feature,
                        "train_step_sample_count": len(train_samples),
                        "test_step_sample_count": len(test_samples),
                        "train_balanced_accuracy": rule.train_balanced_accuracy,
                        "rule_sign": rule.sign,
                        "rule_threshold": rule.threshold,
                        **_balanced_metrics(test_y, predictions, scores),
                    }
                )
            train_x = np.asarray([[float(sample[feature]) for feature in FEATURE_NAMES] for sample in train_samples], dtype=float)
            test_x = np.asarray([[float(sample[feature]) for feature in FEATURE_NAMES] for sample in test_samples], dtype=float)
            predictions, scores = _centroid_prediction(train_x, np.asarray(train_y, dtype=int), test_x)
            rows.append(
                {
                    "train_trial_id": train_id,
                    "test_trial_id": test_id,
                    "validation": "leave_one_trial_out",
                    "model": "all_feature_nearest_centroid",
                    "feature": "all_features",
                    "train_step_sample_count": len(train_samples),
                    "test_step_sample_count": len(test_samples),
                    "train_balanced_accuracy": float("nan"),
                    "rule_sign": float("nan"),
                    "rule_threshold": float("nan"),
                    **_balanced_metrics(test_y, predictions.tolist(), scores.tolist()),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(within_rows: list[dict[str, object]], cross_rows: list[dict[str, object]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    trial_ids = sorted({str(row["trial_id"]) for row in within_rows})
    for trial_id in trial_ids:
        by_feature = {
            str(row["feature"]): row
            for row in within_rows
            if row["trial_id"] == trial_id and row["model"] == "univariate_threshold"
        }
        axes[0].plot(
            FEATURE_NAMES,
            [float(by_feature[feature]["balanced_accuracy"]) for feature in FEATURE_NAMES],
            marker="o",
            label=trial_id.rsplit("_", 1)[-1],
        )
    axes[0].axhline(0.5, color="0.2", linestyle="--", linewidth=1)
    axes[0].set(title="Within-trial blocked validation", ylabel="Balanced accuracy", xlabel="IR feature")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(fontsize=7)
    axes[0].grid(axis="y", alpha=0.25)

    directions = sorted({f"{row['train_trial_id']} -> {row['test_trial_id']}" for row in cross_rows})
    for direction in directions:
        train_id, test_id = direction.split(" -> ")
        by_feature = {
            str(row["feature"]): row
            for row in cross_rows
            if row["train_trial_id"] == train_id and row["test_trial_id"] == test_id and row["model"] == "univariate_threshold"
        }
        axes[1].plot(
            FEATURE_NAMES,
            [float(by_feature[feature]["balanced_accuracy"]) for feature in FEATURE_NAMES],
            marker="o",
            label=f"{train_id.rsplit('_', 1)[-1]} -> {test_id.rsplit('_', 1)[-1]}",
        )
    axes[1].axhline(0.5, color="0.2", linestyle="--", linewidth=1)
    axes[1].set(title="Cross-trial validation", ylabel="Balanced accuracy", xlabel="IR feature")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(fontsize=7)
    axes[1].grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def analyze(
    trial_dirs: list[Path],
    *,
    output_dir: Path,
    stable_window_s: float = 1.0,
    hard_min_percent: float = 70.0,
    not_hard_max_percent: float = 50.0,
) -> dict[str, object]:
    by_trial = {
        trial_dir.name: load_trial(
            trial_dir,
            stable_window_s=stable_window_s,
            hard_min_percent=hard_min_percent,
            not_hard_max_percent=not_hard_max_percent,
        )
        for trial_dir in trial_dirs
    }
    if any(not samples for samples in by_trial.values()):
        missing = [trial_id for trial_id, samples in by_trial.items() if not samples]
        raise ValueError(f"no valid target-hold samples for {missing}")
    all_samples = [sample for samples in by_trial.values() for sample in samples]
    within_rows = [row for samples in by_trial.values() for row in _within_trial_univariate(samples)]
    within_rows.extend(_within_trial_centroid(samples) for samples in by_trial.values())
    cross_rows = _cross_trial_rows(by_trial)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "hard_press_step_level_dataset.csv", all_samples)
    _write_csv(output_dir / "hard_press_within_trial_blocked_validation.csv", within_rows)
    _write_csv(output_dir / "hard_press_cross_trial_validation.csv", cross_rows)
    _plot(within_rows, cross_rows, output_dir / "hard_press_classifier_validation.png")
    summary = {
        "objective": "IR-only hard-press versus not-hard prompted-squeeze classifier",
        "label_definition": {
            "hard": f"prompt target >= {hard_min_percent:g}%",
            "not_hard": f"prompt target <= {not_hard_max_percent:g}%",
            "interpretation": "prompt label only; not measured force, pressure, or OAK geometry",
        },
        "aggregation": {
            "source": "target_hold frames with frozen and AGC-jump frames excluded",
            "one_sample_per_prompted_hold": True,
            "final_window_s": stable_window_s,
        },
        "validation": {
            "within_trial": "leave-one-sequence-out",
            "cross_trial": "train one complete trial, test the other",
            "frame_random_split_used": False,
        },
        "trial_sample_counts": {
            trial_id: {"step_samples": len(samples), "hard_steps": sum(int(sample["hard_label"]) for sample in samples)}
            for trial_id, samples in by_trial.items()
        },
        "best_within_trial_balanced_accuracy": {
            trial_id: max(
                float(row["balanced_accuracy"])
                for row in within_rows
                if row["trial_id"] == trial_id and row["model"] == "univariate_threshold"
            )
            for trial_id in by_trial
        },
        "best_cross_trial_balanced_accuracy": max(float(row["balanced_accuracy"]) for row in cross_rows),
        "deployment_readiness": "not supported by this two-trial pilot; inspect cross-trial results before any controller use",
    }
    (output_dir / "hard_press_classifier_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run blocked IR hard-press/not-hard classifier analysis.")
    parser.add_argument("--trial", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "analysis/hard_press_classifier")
    parser.add_argument("--stable-window-s", type=float, default=1.0)
    parser.add_argument("--hard-min-percent", type=float, default=70.0)
    parser.add_argument("--not-hard-max-percent", type=float, default=50.0)
    args = parser.parse_args()
    trial_dirs = args.trial or list(DEFAULT_TRIALS)
    summary = analyze(
        trial_dirs,
        output_dir=args.output_dir,
        stable_window_s=args.stable_window_s,
        hard_min_percent=args.hard_min_percent,
        not_hard_max_percent=args.not_hard_max_percent,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
