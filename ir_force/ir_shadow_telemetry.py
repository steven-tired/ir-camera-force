"""The IR half of the shadow-telemetry CSV.

The generic logger and the shared columns now live in the public core package;
the PressureVision columns live in the PV integration. What is left here is what
is genuinely IR's: its own schema version, and a logger that keeps the existing
constructor so the soak and the live viz do not have to know about the split.

OAK and thermal timestamps are host read-completion observations. They do not
represent camera exposure synchronization.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from lerobot_teleoperator_so101_webcam.shadow_telemetry import (
    CONTROL_SHADOW_FIELDS,
    ShadowTelemetryLogger,
    ShadowTelemetrySample,
)

__all__ = [
    "IR_SHADOW_FIELDS",
    "IR_SHADOW_SCHEMA_VERSION",
    "IRShadowTelemetryLogger",
    "IRShadowTelemetrySample",
]

IR_SHADOW_SCHEMA_VERSION = "1"
#: The IR row is exactly the shared row. The alias is kept because readers of
#: existing v1 CSVs refer to the columns by this name.
IR_SHADOW_FIELDS = CONTROL_SHADOW_FIELDS
IRShadowTelemetrySample = ShadowTelemetrySample


class IRShadowTelemetryLogger(ShadowTelemetryLogger):
    """The shared logger at IR's schema version, with the PV columns opt-in.

    Passing `extra_fields=PV_SHADOW_FIELDS` switches the row to PV schema v7.
    The PV extension is imported only then: the soak logs IR rows and must stay
    free of every import it does not actually need.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], float] = time.perf_counter,
        extra_fields: tuple[str, ...] = (),
    ):
        extra_row = None
        schema_version = IR_SHADOW_SCHEMA_VERSION
        if extra_fields:
            from pressurevision_integration.pv_shadow_telemetry import (
                PV_SHADOW_FIELDS,
                PV_SHADOW_SCHEMA_VERSION,
                pv_shadow_row,
            )

            unknown = set(extra_fields) - set(PV_SHADOW_FIELDS)
            if unknown:
                raise ValueError(f"unsupported telemetry fields: {sorted(unknown)}")
            extra_row = pv_shadow_row
            schema_version = PV_SHADOW_SCHEMA_VERSION
        super().__init__(
            path,
            schema_version=schema_version,
            clock=clock,
            extra_fields=tuple(extra_fields),
            extra_row=extra_row,
            log_prefix="[ir-sidecar]",
        )
