"""The private IR gripper adapter, and the direction of the dependency.

`ir-camera-force` may consume the public grip contract; `mediapipe-so101` may
never import anything from here. The adapter therefore lives on this side of the
line, implements the *public* protocol, and reuses this repo's own IR proposal
policy internally.

Default behaviour is shadow: the adapter computes what IR would command and
records it, but returns the held command, so wiring it into a live teleop run
cannot move the motor until someone explicitly arms it.
"""

from dataclasses import replace

import pytest

from lerobot_teleoperator_so101_webcam.grip.contract import GripInput, GripperController
from lerobot_teleoperator_so101_webcam.grip.mediapipe import RELEASE_POS

from ir_force.gripper_adapter import IRShadowGripperController
from ir_force.ir_pressure import PressureReading
from lerobot_teleoperator_so101_webcam.grip.proposal import PressureProposalDecision


def grip(**overrides) -> GripInput:
    base = dict(grasp_active=True, explicit_release=False, severity=0.5,
                valid=True, observed_at_s=1.0)
    base.update(overrides)
    return GripInput(**base)


def reading(**overrides) -> PressureReading:
    base = dict(pressure_0_1=0.0, active=False, quality=1.0, available=True,
                status="baseline")
    base.update(overrides)
    return PressureReading(**base)


def test_adapter_satisfies_the_public_protocol():
    assert isinstance(IRShadowGripperController(), GripperController)


def test_invalid_ir_holds_current_command():
    controller = IRShadowGripperController()
    controller.current_command = 28.0
    held = grip(valid=False)

    assert controller.step(held, 28.5) == 28.0


def test_invalid_ir_holds_even_though_the_arm_has_drifted():
    """The hold is on the last *command*, not on where the arm happens to be.

    Returning `actual_pos` would let servo droop ratchet the command open frame
    after frame, which is how a held object gets dropped.
    """
    controller = IRShadowGripperController()
    controller.current_command = 20.0

    for actual in (24.0, 31.0, 38.0):
        assert controller.step(grip(valid=False), actual) == 20.0


def test_explicit_release_is_the_only_thing_that_opens_the_gripper():
    controller = IRShadowGripperController()
    controller.current_command = 12.0

    assert controller.step(grip(explicit_release=True), 12.0) == RELEASE_POS


def test_release_wins_over_an_invalid_reading():
    """Release authority belongs to MediaPipe, so a dead IR sender cannot veto it."""
    controller = IRShadowGripperController()
    controller.current_command = 12.0

    assert controller.step(grip(explicit_release=True, valid=False), 12.0) == RELEASE_POS


def arm(controller, command):
    """Walk the proposal machine through the baseline it requires before closing."""
    controller.seed(command)
    controller.observe(reading())
    controller.step(grip(severity=0.0), command)
    return controller


def test_an_active_reading_before_any_baseline_does_not_close():
    """The proposal machine arms on a baseline. Skipping it must not squeeze."""
    controller = IRShadowGripperController(shadow_only=False)
    controller.current_command = 40.0
    controller.observe(reading(pressure_0_1=0.9, active=True, status="active"))

    assert controller.step(grip(severity=0.9), 40.0) == 40.0
    assert controller.last_proposal.reason == "pressure_disarmed"


def test_shadow_mode_records_a_proposal_without_moving_the_command():
    controller = arm(IRShadowGripperController(), 40.0)
    controller.observe(reading(pressure_0_1=0.8, active=True, status="active"))

    command = controller.step(grip(severity=0.8), 40.0)

    assert command == 40.0, "shadow mode must not actuate"
    assert controller.last_proposal is not None
    assert controller.last_proposal.proposed_gripper < 40.0, "IR wanted to close further"


def test_armed_mode_applies_the_proposal():
    controller = arm(IRShadowGripperController(shadow_only=False), 40.0)
    controller.observe(reading(pressure_0_1=0.8, active=True, status="active"))

    command = controller.step(grip(severity=0.8), 40.0)

    assert command < 40.0
    assert command == pytest.approx(controller.last_proposal.proposed_gripper)


class OpeningMachine:
    """A proposal machine that asks for a *larger* (more open) gripper value."""

    def update(self, base_gripper, pressure):
        return PressureProposalDecision(
            base_gripper=base_gripper, raw_gripper=90.0, proposed_gripper=90.0,
            state="armed", fault_latched=False, reason="test_opening")


def test_armed_mode_still_never_opens_without_an_explicit_release():
    """A proposal that would open the gripper is clamped, not obeyed.

    The IR path is allowed to squeeze harder. Letting it relax is how it would
    drop an object while MediaPipe still believes the grasp is held.
    """
    controller = IRShadowGripperController(shadow_only=False, machine=OpeningMachine())
    controller.current_command = 20.0
    controller.observe(reading(pressure_0_1=0.0, active=True, status="active"))

    assert controller.step(grip(severity=0.0), 20.0) == 20.0
    assert controller.last_proposal.proposed_gripper == 90.0, "the proposal is still recorded"


def test_no_grasp_holds_rather_than_closing():
    controller = arm(IRShadowGripperController(shadow_only=False), 45.0)
    controller.observe(reading(pressure_0_1=0.9, active=True, status="active"))

    assert controller.step(grip(grasp_active=False, severity=0.9), 45.0) == 45.0


def test_reset_returns_to_release_and_drops_the_proposal():
    controller = arm(IRShadowGripperController(shadow_only=False), 8.0)
    controller.observe(reading(pressure_0_1=0.9, active=True, status="active"))
    controller.step(grip(severity=0.9), 8.0)

    controller.reset()

    assert controller.current_command == RELEASE_POS
    assert controller.last_proposal is None


def test_grip_input_stays_frozen_so_the_adapter_cannot_edit_intent():
    held = grip()
    with pytest.raises(Exception):
        held.severity = 0.1                                    # type: ignore[misc]
    assert replace(held, severity=0.1).severity == 0.1


def test_public_repo_does_not_import_the_private_package():
    """The one-way dependency, asserted from the side that would be violated."""
    import subprocess
    from pathlib import Path

    public = Path(__file__).resolve().parents[2] / "mediapipe-so101"
    if not (public / ".git").exists():
        pytest.skip("public checkout not present")
    # Not a bare name search: the public repo's own boundary test names `ir_force`
    # as a forbidden module, which is the guard working, not a violation.
    found = subprocess.run(
        ["git", "-C", str(public), "grep", "-nE", r"^\s*(from|import)\s+ir_force\b"],
        capture_output=True, text=True)
    assert found.stdout == "", f"public repo imports the private package:\n{found.stdout}"
