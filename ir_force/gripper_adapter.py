"""Private IR gripper adapter over the public grip contract.

The dependency runs one way: this repo consumes `mediapipe-so101`'s
`GripperController` protocol, and the public repo never imports `ir_force`. So
the adapter lives here, on the private side, and reuses this repo's own
`PressureProposalStateMachine` for the actual policy.

Two safety properties are structural rather than tuned:

* **Shadow by default.** `shadow_only=True` computes and records the proposal
  but returns the held command, so wiring this into a live run cannot move the
  motor until someone passes `shadow_only=False` on purpose.
* **It can only ever squeeze.** MediaPipe owns grasp and release authority (see
  the public contract's module docstring); the IR path may propose closing
  further, never relaxing. An invalid or stale reading holds the last *command*
  -- not `actual_pos`, since servo droop would otherwise ratchet the gripper
  open one frame at a time and drop the object.
"""

from __future__ import annotations

from lerobot_teleoperator_so101_webcam.grip.contract import GripInput
from lerobot_teleoperator_so101_webcam.grip.mediapipe import RELEASE_POS

from ir_force.ir_pressure_proposal import (
    PressureProposalDecision,
    PressureProposalStateMachine,
)

__all__ = ["IRShadowGripperController"]


class IRShadowGripperController:
    """A `GripperController` that proposes IR-driven closure, in shadow by default."""

    def __init__(
        self,
        *,
        shadow_only: bool = True,
        initial_command: float = RELEASE_POS,
        machine: PressureProposalStateMachine | None = None,
    ):
        self.shadow_only = bool(shadow_only)
        self.current_command = float(initial_command)
        self.last_proposal: PressureProposalDecision | None = None
        self._pressure = None
        self._machine = machine or self._new_machine(self.current_command)

    @staticmethod
    def _new_machine(initial_gripper: float) -> PressureProposalStateMachine:
        return PressureProposalStateMachine(initial_gripper=initial_gripper)

    def seed(self, command: float) -> None:
        """Set the held command and tell the proposal machine where it starts.

        The machine rate-limits each step to `MAX_PRESSURE_GRIP_STEP`, so a
        command set behind its back would take several frames to catch up --
        during which the proposal reads as "barely closing" when it is really
        just walking. Callers that jump the gripper (a middle-pose reset, say)
        should seed rather than assign `current_command`.
        """
        self.current_command = float(command)
        self._machine.seed(command, reset_smoothed=True)

    def observe(self, pressure) -> None:
        """Hand in the latest `PressureReading`. `None` means no measurement."""
        self._pressure = pressure

    def reset(self) -> None:
        self.current_command = RELEASE_POS
        self.last_proposal = None
        self._pressure = None
        self._machine = self._new_machine(RELEASE_POS)

    def step(self, grip: GripInput, actual_pos: float) -> float:
        if grip.explicit_release:
            # MediaPipe's release is unconditional: a dead IR sender must not be
            # able to veto it, so this is checked before validity.
            self.reset()
            return self.current_command

        if not grip.valid or not grip.grasp_active:
            return self.current_command

        decision = self._machine.update(self.current_command, self._pressure)
        self.last_proposal = decision
        if not self.shadow_only:
            self.current_command = min(self.current_command, decision.proposed_gripper)
        return self.current_command
