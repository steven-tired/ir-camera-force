from __future__ import annotations

import importlib


def test_select_representative_frames_covers_release_contact_levels_and_hysteresis():
    module = importlib.import_module("prepare_gpt_pro_rep08_review")
    rows = [
        {
            "frame": str(frame),
            "step_name": step_name,
            "phase": "stable_hold",
            "marker_detected": "True",
            "frozen_frame_flag": "False",
            "step_elapsed_s": str(elapsed),
        }
        for frame, step_name, elapsed in [
            (10, "steady_state_s01_step00_r", 3.0),
            (11, "steady_state_s01_step00_r", 5.0),
            (20, "steady_state_s01_step03_n", 4.0),
            (30, "steady_state_s01_step05_c0", 4.0),
            (40, "steady_state_s01_step09_c10", 4.0),
            (50, "steady_state_s01_step13_c20", 4.0),
            (60, "steady_state_s01_step19_c30", 4.0),
            (70, "hysteresis_s03_step01_c10", 3.0),
            (80, "hysteresis_s03_step02_c20", 3.0),
            (90, "hysteresis_s03_step03_c30", 5.0),
            (100, "hysteresis_s03_step04_c20", 3.0),
            (110, "hysteresis_s03_step05_c10", 3.0),
            (120, "hysteresis_s03_step07_r", 8.0),
            (121, "hysteresis_s03_step07_r", 9.0),
            (999, "steady_state_s01_step19_c30", 6.0),
        ]
    ]
    rows[-1]["frozen_frame_flag"] = "True"

    selected = module.select_representative_frames(rows)

    assert [item.case_id for item in selected] == [
        "released_baseline",
        "near_no_contact",
        "just_contact",
        "c10_steady",
        "c20_steady",
        "c30_steady",
        "c10_loading",
        "c20_loading",
        "c30_loading",
        "c20_unloading",
        "c10_unloading",
        "released_after_hysteresis",
    ]
    assert [item.frame for item in selected] == [11, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 121]
