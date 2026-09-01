from __future__ import annotations

import csv
import importlib


def test_load_trial_uses_final_valid_hold_window_and_prompt_binary_labels(tmp_path):
    module = importlib.import_module("analyze_ir_hard_press_classifier")
    trial = tmp_path / "oak-squeeze_s01_fixed-posture_foam_zk_rep02"
    trial.mkdir()
    fields = [
        "phase", "sequence_id", "step_index", "target_squeeze_percent", "timestamp",
        *module.FEATURE_NAMES, "frozen_frame_flag", "agc_jump_flag",
    ]
    rows = [
        ["target_hold", 1, 0, 0, 0.0, *([1.0] * len(module.FEATURE_NAMES)), "False", "False"],
        ["target_hold", 1, 0, 0, 2.5, *([2.0] * len(module.FEATURE_NAMES)), "False", "False"],
        ["target_hold", 1, 1, 75, 2.5, *([9.0] * len(module.FEATURE_NAMES)), "False", "False"],
        ["target_hold", 1, 1, 75, 2.8, *([99.0] * len(module.FEATURE_NAMES)), "True", "False"],
    ]
    with (trial / "frame_features.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)

    samples = module.load_trial(trial, stable_window_s=1.0)

    assert len(samples) == 2
    assert [(sample["target_percent"], sample["hard_label"]) for sample in samples] == [(0.0, 0), (75.0, 1)]
    assert samples[0]["roi_mean"] == 2.0
    assert samples[1]["roi_mean"] == 9.0


def test_fit_threshold_selects_a_direction_and_classifies_separable_values():
    module = importlib.import_module("analyze_ir_hard_press_classifier")

    rule = module.fit_threshold([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])

    assert rule.sign == 1
    assert rule.threshold > 0.2
    assert rule.threshold < 0.8
    assert module.apply_threshold(rule, [0.15, 0.85]) == [0, 1]
