import pytest

from ir_force.classifier.ir_hard_classifier import (
    ForceLabelThresholds,
    HardClassifierTrialSpec,
    build_force_target_steps,
    force_label,
    hard_classifier_trial_id,
    target_force_newton,
)


def test_force_label_uses_objective_session_fmax_thresholds():
    thresholds = ForceLabelThresholds(hard_fraction=0.70, not_hard_fraction=0.50)

    assert force_label(4.9, fmax_n=10.0, thresholds=thresholds) == "not_hard"
    assert force_label(5.0, fmax_n=10.0, thresholds=thresholds) == "not_hard"
    assert force_label(6.0, fmax_n=10.0, thresholds=thresholds) == "ambiguous"
    assert force_label(7.0, fmax_n=10.0, thresholds=thresholds) == "hard"


def test_force_label_rejects_bad_thresholds_and_fmax():
    with pytest.raises(ValueError, match="fmax"):
        force_label(1.0, fmax_n=0.0)

    with pytest.raises(ValueError, match="not_hard_fraction"):
        ForceLabelThresholds(hard_fraction=0.5, not_hard_fraction=0.7)


def test_target_force_newton_converts_percent_of_session_fmax():
    assert target_force_newton(75.0, fmax_n=12.0) == 9.0


def test_hard_classifier_trial_id_is_stable_and_descriptive():
    spec = HardClassifierTrialSpec(
        session_id="S01",
        block_type="fixed_posture",
        rep=2,
        object_id="foam block",
        participant_id="ZK",
    )

    assert hard_classifier_trial_id(spec) == "hard-classifier_s01_fixed-posture_foam-block_zk_rep02"


def test_build_force_target_steps_randomizes_repeated_target_sequences():
    steps = build_force_target_steps(
        target_percents=(0, 25, 50, 75),
        sequences=3,
        hold_s=3.0,
        release_s=0.75,
        fmax_n=20.0,
        seed=7,
        block_type="fixed_posture",
        posture_condition="neutral",
    )

    assert len(steps) == 12
    assert [step.step_index for step in steps[:4]] == [0, 1, 2, 3]
    assert {step.target_force_percent for step in steps[:4]} == {0.0, 25.0, 50.0, 75.0}
    assert [step.target_force_percent for step in steps[:4]] != [0.0, 25.0, 50.0, 75.0]
    assert steps[0].sequence_id == 1
    assert steps[4].sequence_id == 2
    assert steps[0].target_force_newton == steps[0].target_force_percent / 100.0 * 20.0
    assert steps[0].hold_s == 3.0
    assert steps[0].release_s == 0.75
    assert steps[0].block_type == "fixed_posture"
    assert steps[0].posture_condition == "neutral"
