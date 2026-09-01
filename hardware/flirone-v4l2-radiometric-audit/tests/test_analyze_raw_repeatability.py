from __future__ import annotations

import csv
import json
import struct

import cv2
import numpy as np


def _write_raw_frame(
    directory,
    *,
    index: int,
    timestamp_ns: int,
    value: int,
    ffc_state: str = "normal",
    repeated: bool = False,
    display_mode: str = "dynamic",
    raw_low: int | None = None,
    raw_high: int | None = None,
):
    values = [value] * 4
    raw_path = directory / f"raw_frame_{index:06d}.u16le"
    raw_path.write_bytes(struct.pack("<4H", *values))
    (directory / f"raw_frame_{index:06d}.json").write_text(
        json.dumps(
            {
                "frame_index": index,
                "monotonic_timestamp_ns": timestamp_ns,
                "width": 2,
                "height": 2,
                "dtype": "uint16",
                "byte_order": "little",
                "raw_file": raw_path.name,
                "ffc_state": ffc_state,
                "raw_min": value,
                "raw_median": value,
                "raw_max": value,
                "repeated_frame_flag": repeated,
                "display_mapping": {
                    "mode": "dynamic_per_frame_min_max" if display_mode == "dynamic" else "fixed_raw_counts",
                    "raw_low": -1 if raw_low is None else raw_low,
                    "raw_high": -1 if raw_high is None else raw_high,
                },
            }
        )
    )


def _write_run(
    session_root,
    *,
    run_id: str,
    mode: str,
    events: list[dict[str, str]],
    frames: list[tuple[int, int, int, str, bool]],
    display_mode: str = "dynamic",
    raw_low: int | None = None,
    raw_high: int | None = None,
):
    run_dir = session_root / "runs" / run_id
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "mode": mode,
                "target_raw_roi": {"x": 0, "y": 0, "width": 2, "height": 2},
                "control_raw_roi": {"x": 0, "y": 0, "width": 2, "height": 2},
                "display": {"mode": display_mode, "raw_low": raw_low, "raw_high": raw_high},
            }
        )
    )
    with (run_dir / "events.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("timestamp_ns", "event_type", "phase", "run_id"))
        writer.writeheader()
        writer.writerows(events)
    for index, timestamp_ns, value, ffc_state, repeated in frames:
        _write_raw_frame(
            raw_dir,
            index=index,
            timestamp_ns=timestamp_ns,
            value=value,
            ffc_state=ffc_state,
            repeated=repeated,
            display_mode=display_mode,
            raw_low=raw_low,
            raw_high=raw_high,
        )
    return run_dir


def test_phase_window_parser_uses_monotonic_start_and_end_events():
    analyzer = __import__("analyze_raw_repeatability")

    windows = analyzer.phase_windows_from_events(
        [
            {"timestamp_ns": "100", "event_type": "run_start", "phase": "", "run_id": "dynamic"},
            {"timestamp_ns": "200", "event_type": "phase_start", "phase": "baseline_01", "run_id": "dynamic"},
            {"timestamp_ns": "300", "event_type": "phase_end", "phase": "baseline_01", "run_id": "dynamic"},
            {"timestamp_ns": "400", "event_type": "phase_start", "phase": "hot_hand_01", "run_id": "dynamic"},
            {"timestamp_ns": "500", "event_type": "phase_end", "phase": "hot_hand_01", "run_id": "dynamic"},
        ]
    )

    assert windows == {"baseline_01": (200, 300), "hot_hand_01": (400, 500)}


def test_phase_window_parser_rejects_incomplete_or_reordered_events():
    analyzer = __import__("analyze_raw_repeatability")

    try:
        analyzer.phase_windows_from_events(
            [{"timestamp_ns": "100", "event_type": "phase_end", "phase": "baseline_01", "run_id": "dynamic"}]
        )
    except ValueError as error:
        assert "without a start" in str(error)
    else:
        raise AssertionError("expected incomplete event sequence to fail")


def test_session_analysis_reports_ffc_restart_dynamic_and_fixed_range_metrics(tmp_path):
    analyzer = __import__("analyze_raw_repeatability")
    session_root = tmp_path / "session"
    _write_run(
        session_root,
        run_id="ffc_01",
        mode="ffc",
        events=[],
        frames=[
            (0, 0, 100, "normal", False),
            (1, 1_000_000_000, 100, "normal", False),
            (2, 3_000_000_000, 120, "ffc", False),
            (3, 4_000_000_000, 120, "ffc", False),
            (4, 5_000_000_000, 120, "post_ffc_discarded", False),
            (5, 6_000_000_000, 105, "normal", False),
            (6, 7_000_000_000, 105, "normal", False),
        ],
    )
    for index, value in enumerate((100, 104)):
        _write_run(
            session_root,
            run_id=f"restart_{index + 1:02d}",
            mode="restart",
            events=[
                {"timestamp_ns": "0", "event_type": "phase_start", "phase": "stable", "run_id": "restart"},
                {"timestamp_ns": "1000000000", "event_type": "phase_end", "phase": "stable", "run_id": "restart"},
            ],
            frames=[(0, 0, value, "normal", False), (1, 1_000_000_000, value, "normal", False)],
        )
    _write_run(
        session_root,
        run_id="dynamic_01",
        mode="dynamic",
        events=[
            {"timestamp_ns": "0", "event_type": "phase_start", "phase": "baseline_01", "run_id": "dynamic"},
            {"timestamp_ns": "1000000000", "event_type": "phase_end", "phase": "baseline_01", "run_id": "dynamic"},
            {"timestamp_ns": "2000000000", "event_type": "phase_start", "phase": "hot_hand_01", "run_id": "dynamic"},
            {"timestamp_ns": "3000000000", "event_type": "phase_end", "phase": "hot_hand_01", "run_id": "dynamic"},
            {"timestamp_ns": "4000000000", "event_type": "phase_start", "phase": "recovery_01", "run_id": "dynamic"},
            {"timestamp_ns": "5000000000", "event_type": "phase_end", "phase": "recovery_01", "run_id": "dynamic"},
        ],
        frames=[
            (0, 0, 100, "normal", False),
            (1, 1_000_000_000, 100, "normal", False),
            (2, 2_000_000_000, 150, "normal", False),
            (3, 3_000_000_000, 150, "normal", False),
            (4, 4_000_000_000, 105, "normal", False),
            (5, 5_000_000_000, 105, "normal", False),
        ],
    )

    summary = analyzer.analyze_session(session_root, ffc_window_s=2.0, ffc_post_delay_s=1.0)

    assert summary["ffc"][0]["target"]["event_count"] == 1
    assert summary["ffc"][0]["target"]["events"][0]["post_minus_pre_counts"] == 5
    assert summary["restart_offsets"]["target"]["by_run"]["restart_02"] == 4
    assert summary["dynamic"][0]["metrics"]["cycles"][0]["target_hot_minus_baseline_counts"] == 50
    assert summary["fixed_range_suggestion"]["raw_low"] < 100
    assert summary["fixed_range_suggestion"]["raw_high"] > 150


def test_analyzer_cli_accepts_session_and_explicit_summary_path(tmp_path):
    analyzer = __import__("analyze_raw_repeatability")

    args = analyzer.parse_args(
        [
            "--session-root",
            str(tmp_path / "session"),
            "--summary-path",
            str(tmp_path / "summary.json"),
            "--ffc-window-s",
            "8",
        ]
    )

    assert args.session_root.name == "session"
    assert args.summary_path.name == "summary.json"
    assert args.ffc_window_s == 8.0


def test_rgb_phase_analysis_reads_the_saved_palette_index_from_a_raw_roi(tmp_path):
    analyzer = __import__("analyze_raw_repeatability")
    run_dir = tmp_path / "run"
    (run_dir / "rgb").mkdir(parents=True)
    (run_dir / "rgb_frames.csv").write_text("frame_index,timestamp_ns,file\n0,100,rgb/frame_000000.png\n")
    palette_path = tmp_path / "palette.raw"
    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[:, 0] = np.arange(256, dtype=np.uint8)
    palette_path.write_bytes(palette.tobytes())
    image = np.zeros((128, 160, 3), dtype=np.uint8)
    image[4:6, 6:8] = (0, 0, 128)  # OpenCV BGR encoding of RGB palette entry 128.
    assert cv2.imwrite(str(run_dir / "rgb" / "frame_000000.png"), image)

    metrics = analyzer.rgb_palette_phase_metrics(
        run_dir,
        phase_windows={"baseline_01": (0, 200)},
        target_roi=analyzer.RawROI(x=3, y=2, width=1, height=1),
        palette_path=palette_path,
    )

    assert metrics["baseline_01"]["frame_count"] == 1
    assert metrics["baseline_01"]["median_palette_index"] == 128
    assert analyzer.fixed_raw_count_to_palette_index(150, raw_low=100, raw_high=200) == 127


def test_fixed_palette_comparison_reports_expected_and_observed_phase_indices():
    analyzer = __import__("analyze_raw_repeatability")

    comparison = analyzer.fixed_palette_phase_comparison(
        dynamic_metrics={"phases": {"baseline_01": {"target": {"median": 150}}}},
        rgb_metrics={"baseline_01": {"median_palette_index": 127}},
        raw_low=100,
        raw_high=200,
    )

    assert comparison["baseline_01"] == {
        "target_raw_median": 150,
        "expected_palette_index": 127,
        "observed_palette_index": 127,
        "observed_minus_expected_palette_index": 0,
    }


def test_fixed_range_suggestion_uses_only_the_dynamic_display_probe_frames(tmp_path):
    analyzer = __import__("analyze_raw_repeatability")
    session_root = tmp_path / "session"
    events = [
        {"timestamp_ns": "0", "event_type": "phase_start", "phase": "baseline_01", "run_id": "dynamic"},
        {"timestamp_ns": "1000000000", "event_type": "phase_end", "phase": "baseline_01", "run_id": "dynamic"},
        {"timestamp_ns": "2000000000", "event_type": "phase_start", "phase": "hot_hand_01", "run_id": "dynamic"},
        {"timestamp_ns": "3000000000", "event_type": "phase_end", "phase": "hot_hand_01", "run_id": "dynamic"},
        {"timestamp_ns": "4000000000", "event_type": "phase_start", "phase": "recovery_01", "run_id": "dynamic"},
        {"timestamp_ns": "5000000000", "event_type": "phase_end", "phase": "recovery_01", "run_id": "dynamic"},
    ]
    _write_run(
        session_root,
        run_id="dynamic_agc_01",
        mode="dynamic",
        events=events,
        frames=[(0, 0, 100, "normal", False), (1, 1_000_000_000, 100, "normal", False), (2, 2_000_000_000, 150, "normal", False), (3, 3_000_000_000, 150, "normal", False), (4, 4_000_000_000, 105, "normal", False), (5, 5_000_000_000, 105, "normal", False)],
    )
    _write_run(
        session_root,
        run_id="dynamic_fixed_01",
        mode="dynamic",
        events=events,
        frames=[(0, 0, 10_000, "normal", False), (1, 1_000_000_000, 10_000, "normal", False), (2, 2_000_000_000, 10_000, "normal", False), (3, 3_000_000_000, 10_000, "normal", False), (4, 4_000_000_000, 10_000, "normal", False), (5, 5_000_000_000, 10_000, "normal", False)],
        display_mode="fixed",
        raw_low=9000,
        raw_high=11000,
    )

    summary = analyzer.analyze_session(session_root)

    assert summary["fixed_range_suggestion"]["source_p995"] == 150
