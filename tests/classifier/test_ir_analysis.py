import pytest

from ir_force.classifier.ir_analysis import (
    TrialSummary,
    decision_from_summaries,
    hardness_effect_size,
    monotonic_fraction,
    summarize_trial,
)


def _write_trial_fixture_files(trial_dir, telemetry_csv: str) -> None:
    (trial_dir / "metadata.json").write_text(
        (
            '{"trial_id":"trial_001","object_name":"foam","hardness":"soft",'
            '"grip_level":"low","warmed":false}'
        )
    )
    (trial_dir / "telemetry.csv").write_text(telemetry_csv)
    (trial_dir / "ir_features.csv").write_text(
        "area_px,mean_delta,max_delta\n10,1,2\n20,2,3\n"
    )


def test_monotonic_fraction_scores_ordered_levels():
    assert monotonic_fraction({"low": 1.0, "med": 2.0, "high": 3.0}) == 1.0
    assert monotonic_fraction({"low": 1.0, "med": 3.0, "high": 2.0}) == 0.5


def test_hardness_effect_size_is_zero_when_groups_match():
    assert hardness_effect_size([1.0, 2.0], [1.0, 2.0]) == 0.0


def test_decision_go_when_warmed_passes_and_three_objects_are_monotonic():
    summaries = []
    for object_name, hardness in [
        ("foam", "soft"),
        ("sponge", "soft"),
        ("wood", "solid"),
        ("plastic", "solid"),
    ]:
        for level, value, current in [
            ("low", 10.0, 20.0),
            ("med", 20.0, 35.0),
            ("high", 30.0, 50.0),
        ]:
            summaries.append(
                TrialSummary(
                    trial_id=f"{object_name}_{level}",
                    object_name=object_name,
                    hardness=hardness,
                    grip_level=level,
                    warmed=False,
                    peak_current=current,
                    hold_mean_area_px=value,
                    hold_mean_delta=value,
                    hold_max_delta=value,
                )
            )
    summaries.append(
        TrialSummary(
            trial_id="sanity_warmed",
            object_name="wood",
            hardness="solid",
            grip_level="high",
            warmed=True,
            peak_current=50.0,
            hold_mean_area_px=400.0,
            hold_mean_delta=80.0,
            hold_max_delta=90.0,
        )
    )
    decision = decision_from_summaries(summaries)
    assert decision["decision"] == "GO"
    assert decision["warmed_sanity_passed"] is True
    assert decision["monotonic_objects"] == 4


def test_decision_no_go_when_warmed_sanity_fails():
    summaries = [
        TrialSummary("warmed", "wood", "solid", "high", True, 50.0, 1.0, 1.0, 1.0),
        TrialSummary("foam_low", "foam", "soft", "low", False, 20.0, 1.0, 1.0, 1.0),
    ]
    assert decision_from_summaries(summaries)["decision"] == "NO-GO"


def test_decision_does_not_count_partial_grip_coverage_as_monotonic():
    summaries = [
        TrialSummary("warmed", "wood", "solid", "high", True, 50.0, 40.0, 20.0, 20.0),
        TrialSummary("foam_low", "foam", "soft", "low", False, 20.0, 10.0, 1.0, 1.0),
        TrialSummary("foam_med", "foam", "soft", "med", False, 35.0, 20.0, 2.0, 2.0),
        TrialSummary("sponge_low", "sponge", "soft", "low", False, 20.0, 11.0, 1.0, 1.0),
        TrialSummary("sponge_med", "sponge", "soft", "med", False, 35.0, 21.0, 2.0, 2.0),
        TrialSummary("wood_low", "wood", "solid", "low", False, 20.0, 12.0, 1.0, 1.0),
        TrialSummary("wood_med", "wood", "solid", "med", False, 35.0, 22.0, 2.0, 2.0),
    ]

    decision = decision_from_summaries(summaries)

    assert decision["decision"] == "NO-GO"
    assert decision["monotonic_objects"] == 0


def test_summarize_trial_raises_clear_error_when_present_current_is_missing(tmp_path):
    trial_dir = tmp_path / "trial_001"
    trial_dir.mkdir()
    _write_trial_fixture_files(trial_dir, "present_current\n\n \n")

    with pytest.raises((RuntimeError, ValueError), match="present_current"):
        summarize_trial(trial_dir)


@pytest.mark.parametrize(
    ("telemetry_csv", "expected_message"),
    [
        ("present_current\n12\n \n15\n", "row 3"),
        ("present_current\n12\nnot-a-number\n15\n", "not-a-number"),
        ("other_column\n12\n15\n", "present_current"),
    ],
)
def test_summarize_trial_rejects_any_invalid_present_current_row(
    tmp_path, telemetry_csv, expected_message
):
    trial_dir = tmp_path / "trial_001"
    trial_dir.mkdir()
    _write_trial_fixture_files(trial_dir, telemetry_csv)

    with pytest.raises(RuntimeError, match="present_current") as exc_info:
        summarize_trial(trial_dir)

    assert "trial_001" in str(exc_info.value)
    assert expected_message in str(exc_info.value)
