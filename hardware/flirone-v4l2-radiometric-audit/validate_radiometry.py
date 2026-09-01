#!/usr/bin/env python3
"""Evaluate independently measured reference temperatures against raw counts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def evaluate_reference_medians(observations: list[dict[str, float]]) -> dict[str, object]:
    """Assess only raw-count ordering and a linear empirical fit.

    This cannot establish factory radiometry. The caller must provide
    independently measured reference temperatures and later include FFC and
    restart repeatability data before accepting a temperature conversion.
    """
    grouped: dict[float, list[float]] = defaultdict(list)
    for observation in observations:
        grouped[float(observation["reference_c"])].append(float(observation["raw_median"]))
    if len(grouped) < 2:
        raise ValueError("at least two reference temperatures are required")
    points = [(temperature, _mean(grouped[temperature])) for temperature in sorted(grouped)]
    counts = [raw_median for _temperature, raw_median in points]
    temperatures = [temperature for temperature, _raw_median in points]
    strictly_ordered = all(right > left for left, right in zip(counts, counts[1:]))

    mean_count = _mean(counts)
    mean_temperature = _mean(temperatures)
    denominator = sum((count - mean_count) ** 2 for count in counts)
    if denominator == 0.0:
        raise ValueError("raw medians are constant; cannot fit a calibration line")
    slope = sum((count - mean_count) * (temperature - mean_temperature) for count, temperature in zip(counts, temperatures)) / denominator
    intercept = mean_temperature - slope * mean_count
    predictions = [slope * count + intercept for count in counts]
    residual_sum = sum((actual - predicted) ** 2 for actual, predicted in zip(temperatures, predictions))
    total_sum = sum((temperature - mean_temperature) ** 2 for temperature in temperatures)
    r_squared = 1.0 if total_sum == 0.0 else 1.0 - residual_sum / total_sum
    mae = _mean([abs(actual - predicted) for actual, predicted in zip(temperatures, predictions)])

    return {
        "schema_version": 1,
        "reference_level_count": len(points),
        "reference_points": [
            {"reference_c": temperature, "raw_median": raw_median}
            for temperature, raw_median in points
        ],
        "strictly_ordered_raw_medians": strictly_ordered,
        "calibration_model": "linear_raw_count_to_celsius",
        "slope_celsius_per_count": slope,
        "intercept_celsius": intercept,
        "r_squared": r_squared,
        "in_sample_mae_celsius": mae,
        "factory_radiometry_validated": False,
        "requires_independent_ffc_restart_and_emissivity_validation": True,
    }


def _load_observations(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"reference_c", "raw_median"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain CSV columns {sorted(required)}")
        return [
            {"reference_c": float(row["reference_c"]), "raw_median": float(row["raw_median"])}
            for row in reader
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references-csv", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path)
    args = parser.parse_args()
    summary = evaluate_reference_medians(_load_observations(args.references_csv))
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.summary_path is not None:
        args.summary_path.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
