"""IR-side payload types.

The `ir-hand-pressure-so101-teleop` branch extended `webcam_input.types` with
fields only the thermal path needs: per-landmark image coordinates and depth for
projecting a hand into a thermal frame, plus capture timestamps for aligning a
webcam frame with a thermal one. Nothing in the public repository constructs or
reads them — its six `LandmarksData` call sites all pass just `landmarks` and
`valid` — so they are defined here instead of widening the public API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from webcam_input.types import WristData

__all__ = ["WristData", "LandmarksData", "WebcamSample"]


@dataclass
class LandmarksData:
    """Hand landmarks, with the extra channels the thermal path needs."""

    landmarks: np.ndarray                 # (21, 3) MANO joint_pos
    valid: bool
    image_xy: np.ndarray | None = None    # (21, 2) normalized OAK/RGB coordinates
    depth_m: np.ndarray | None = None     # (21,) aligned depth in metres
    # Host-monotonic read-completion time, NOT a sensor exposure timestamp.
    # Do not treat it as a hardware-synchronised capture instant.
    observed_at_s: float | None = None
    frame_id: int | None = None


@dataclass
class WebcamSample:
    """One atomic webcam publication from a single captured frame.

    Atomic matters: the preview frame, wrist pose, and landmarks must come from
    the *same* frame, or a consumer correlating them with a thermal frame is
    silently comparing different instants.
    """

    preview_frame: np.ndarray | None
    wrist: WristData
    landmarks: LandmarksData
    observed_at_s: float | None
    frame_id: int | None
