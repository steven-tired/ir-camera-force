from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean


LEVELS = ("low", "med", "high")


@dataclass(frozen=True)
class TrialSummary:
    trial_id: str
    object_name: str
    hardness: str
    grip_level: str
    warmed: bool
    peak_current: float
    hold_mean_area_px: float
    hold_mean_delta: float
    hold_max_delta: float


def monotonic_fraction(values_by_level: dict[str, float]) -> float:
    ordered = [values_by_level[level] for level in LEVELS if level in values_by_level]
    if len(ordered) < 2:
        return 0.0
    comparisons = [ordered[index + 1] > ordered[index] for index in range(len(ordered) - 1)]
    return sum(comparisons) / len(comparisons)


def hardness_effect_size(soft_values: list[float], solid_values: list[float]) -> float:
    if not soft_values or not solid_values:
        return 0.0
    pooled = soft_values + solid_values
    mu = mean(pooled)
    var = mean((value - mu) ** 2 for value in pooled)
    if var == 0:
        return 0.0
    return abs(mean(soft_values) - mean(solid_values)) / (var**0.5)


def summarize_trial(trial_dir: Path, hold_fraction: float = 0.5) -> TrialSummary:
    meta = json.loads((trial_dir / "metadata.json").read_text())
    trial_id = str(meta["trial_id"])
    telemetry_path = trial_dir / "telemetry.csv"
    with (trial_dir / "telemetry.csv").open(newline="") as telemetry_handle:
        telemetry_rows = list(csv.DictReader(telemetry_handle))
    with (trial_dir / "ir_features.csv").open(newline="") as features_handle:
        feature_rows = list(csv.DictReader(features_handle))
    if not telemetry_rows or not feature_rows:
        raise RuntimeError(f"missing rows in {trial_dir}")
    hold_start = int(len(feature_rows) * hold_fraction)
    hold_features = feature_rows[hold_start:]
    currents: list[float] = []
    for row_number, row in enumerate(telemetry_rows, start=2):
        raw_current = row.get("present_current")
        if raw_current is None:
            raise RuntimeError(
                f"missing present_current in trial {trial_id} row {row_number} of {telemetry_path}"
            )
        stripped_current = raw_current.strip()
        if not stripped_current:
            raise RuntimeError(
                f"blank present_current in trial {trial_id} row {row_number} of {telemetry_path}"
            )
        try:
            currents.append(float(stripped_current))
        except ValueError as exc:
            raise RuntimeError(
                f"invalid present_current value {raw_current!r} in trial {trial_id} "
                f"row {row_number} of {telemetry_path}"
            ) from exc
    if not currents:
        raise RuntimeError(f"missing present_current values in trial {trial_id} at {telemetry_path}")
    return TrialSummary(
        trial_id=str(meta["trial_id"]),
        object_name=str(meta["object_name"]),
        hardness=str(meta["hardness"]),
        grip_level=str(meta["grip_level"]),
        warmed=bool(meta["warmed"]),
        peak_current=max(currents),
        hold_mean_area_px=mean(float(row["area_px"]) for row in hold_features),
        hold_mean_delta=mean(float(row["mean_delta"]) for row in hold_features),
        hold_max_delta=max(float(row["max_delta"]) for row in hold_features),
    )


def decision_from_summaries(summaries: list[TrialSummary]) -> dict[str, object]:
    warmed = [summary for summary in summaries if summary.warmed]
    passive = [summary for summary in summaries if not summary.warmed]
    warmed_sanity_passed = any(
        summary.hold_mean_area_px >= 25 and summary.hold_max_delta >= 10 for summary in warmed
    )

    by_object: dict[str, dict[str, list[float]]] = {}
    for summary in passive:
        by_object.setdefault(summary.object_name, {}).setdefault(summary.grip_level, []).append(
            summary.hold_mean_area_px
        )

    monotonic_objects = 0
    for levels in by_object.values():
        if any(level not in levels for level in LEVELS):
            continue
        averaged = {level: mean(values) for level, values in levels.items()}
        if monotonic_fraction(averaged) >= 1.0:
            monotonic_objects += 1

    soft = [summary.hold_mean_area_px for summary in passive if summary.hardness == "soft"]
    solid = [summary.hold_mean_area_px for summary in passive if summary.hardness == "solid"]
    effect = hardness_effect_size(soft, solid)
    go = warmed_sanity_passed and (monotonic_objects >= 3 or effect >= 0.8)

    return {
        "decision": "GO" if go else "NO-GO",
        "warmed_sanity_passed": warmed_sanity_passed,
        "monotonic_objects": monotonic_objects,
        "hardness_effect_size": effect,
        "n_passive_trials": len(passive),
        "n_warmed_trials": len(warmed),
    }


def write_summary_json(summaries: list[TrialSummary], decision: dict[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"decision": decision, "trials": [asdict(summary) for summary in summaries]}
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
