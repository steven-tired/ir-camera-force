from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest


def _write_raw_frame(
    directory,
    *,
    index: int,
    values: list[int],
    width: int = 2,
    height: int = 2,
    timestamp_ns: int | None = None,
    ffc_state: str = "normal",
    repeated: bool = False,
) -> None:
    raw_path = directory / f"raw_frame_{index:06d}.u16le"
    raw_path.write_bytes(struct.pack(f"<{len(values)}H", *values))
    sorted_values = sorted(values)
    metadata = {
        "schema_version": 1,
        "frame_index": index,
        "monotonic_timestamp_ns": 1000 + index if timestamp_ns is None else timestamp_ns,
        "camera_timestamp_available": False,
        "camera_timestamp": None,
        "width": width,
        "height": height,
        "dtype": "uint16",
        "byte_order": "little",
        "raw_file": raw_path.name,
        "ffc_state": ffc_state,
        "raw_min": sorted_values[0],
        "raw_median": (sorted_values[len(values) // 2 - 1] + sorted_values[len(values) // 2]) // 2,
        "raw_max": sorted_values[-1],
        "dropped_frame_flag": False,
        "dropped_frame_observable": False,
        "repeated_frame_flag": repeated,
        "display_mapping": {"mode": "dynamic_per_frame_min_max", "raw_low": -1, "raw_high": -1},
        "calibration": {"used": False, "source": "not_used"},
    }
    (directory / f"raw_frame_{index:06d}.json").write_text(json.dumps(metadata))


def test_raw_reader_verifies_little_endian_uint16_metadata_and_statistics(tmp_path):
    capture = __import__("capture_raw_validation")
    _write_raw_frame(tmp_path, index=3, values=[100, 300, 400, 500])

    summary = capture.validate_capture_directory(tmp_path)

    assert summary["frame_count"] == 1
    assert summary["all_uint16_little_endian"] is True
    assert summary["has_values_above_255"] is True
    assert summary["metadata_stat_mismatch_count"] == 0
    assert summary["raw_stream_accepted"] is True
    assert summary["normal_frame_count"] == 1
    assert summary["deduplicated_normal_frame_count"] == 1
    assert summary["normal_raw_median_stats"]["minimum"] == 350
    assert summary["normal_raw_median_stats"]["population_stdev"] == 0.0
    assert summary["deduplicated_normal_raw_median_drift_counts_per_s"] is None


def test_raw_reader_rejects_metadata_that_does_not_match_raw_bytes(tmp_path):
    capture = __import__("capture_raw_validation")
    _write_raw_frame(tmp_path, index=3, values=[100, 300, 400, 500])
    metadata_path = tmp_path / "raw_frame_000003.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["raw_max"] = 42
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="raw_max"):
        capture.validate_capture_directory(tmp_path)


def test_raw_reader_requires_an_explicit_display_mapping_mode(tmp_path):
    capture = __import__("capture_raw_validation")
    _write_raw_frame(tmp_path, index=3, values=[100, 300, 400, 500])
    metadata_path = tmp_path / "raw_frame_000003.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("display_mapping")
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="display_mapping"):
        capture.validate_capture_directory(tmp_path)


def test_raw_reader_can_require_an_ffc_frame(tmp_path):
    capture = __import__("capture_raw_validation")
    _write_raw_frame(tmp_path, index=3, values=[100, 300, 400, 500])

    with pytest.raises(ValueError, match="FFC"):
        capture.validate_capture_directory(tmp_path, require_ffc=True)


def test_raw_reader_rejects_a_capture_above_the_repeated_frame_limit(tmp_path):
    capture = __import__("capture_raw_validation")
    _write_raw_frame(tmp_path, index=3, values=[100, 300, 400, 500])
    _write_raw_frame(tmp_path, index=4, values=[100, 300, 400, 500], repeated=True)

    with pytest.raises(ValueError, match="repeated frame fraction"):
        capture.validate_capture_directory(tmp_path, max_repeated_fraction=0.25)


def test_reference_evaluation_requires_strict_count_order_and_reports_linear_fit():
    radiometry = __import__("validate_radiometry")
    observations = [
        {"reference_c": 20.0, "raw_median": 1000.0},
        {"reference_c": 25.0, "raw_median": 1200.0},
        {"reference_c": 30.0, "raw_median": 1400.0},
        {"reference_c": 35.0, "raw_median": 1600.0},
        {"reference_c": 40.0, "raw_median": 1800.0},
    ]

    summary = radiometry.evaluate_reference_medians(observations)

    assert summary["strictly_ordered_raw_medians"] is True
    assert summary["r_squared"] == pytest.approx(1.0)
    assert summary["calibration_model"] == "linear_raw_count_to_celsius"


def test_lepton2_decoder_reads_little_endian_sensor_words():
    source = Path("src/flirone.c").read_text()

    assert "sensor_pix[y * 80 + x] = read_le16(&row[x * 2]);" in source


def test_thermal_loopback_writer_reports_write_failures_and_recovery():
    source = Path("src/flirone.c").read_text()

    assert "thermal RGB loopback write failed" in source
    assert "thermal RGB loopback write recovered" in source
