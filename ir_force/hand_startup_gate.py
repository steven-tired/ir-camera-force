"""Right-hand startup gate for the IR teleop.

The IR branch keeps the arm locked until the operator's right hand has been
continuously visible for a dwell period, so a half-tracked hand cannot jog the
arm while the thermal camera is still settling. The public controller has no
such gate — it clutches on a left fist instead — so this lives on the IR side
rather than widening the public module.
"""

from __future__ import annotations

#: Wrist-roll span, in degrees, that still counts as the same continuous hold.
MAX_WRIST_ROLL_RANGE_DEG = 45.0

#: Seconds the right hand must stay visible before the arm unlocks.
HAND_STARTUP_DWELL_S = 3.0

__all__ = ["MAX_WRIST_ROLL_RANGE_DEG", "HAND_STARTUP_DWELL_S", "ContinuousHandStartupGate"]


class ContinuousHandStartupGate:
    required_s: float = HAND_STARTUP_DWELL_S
    detected_since_s: float | None = None

    def update(self, *, hand_valid: bool, observed_at_s: float) -> float:
        if not hand_valid:
            self.detected_since_s = None
            return 0.0
        if self.detected_since_s is None:
            self.detected_since_s = float(observed_at_s)
        return max(0.0, float(observed_at_s) - self.detected_since_s)
