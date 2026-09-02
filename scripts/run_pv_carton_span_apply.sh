#!/usr/bin/env bash
# The 250 g carton pilot mapping, PV in APPLY mode. THE ARM MOVES.
#
# carton_span is a separate named control contract rather than an edit to
# soft_precise's 28..22, so the historical launcher and the evidence recorded
# under it stay reproducible.
#
# Roll is frozen for the observability and formal-dataset phases; a dedicated
# roll experiment can still override it on the command line.
set -euo pipefail
export PV_MAPPING=carton_span
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PV_EVIDENCE_ROOT="${PV_EVIDENCE_ROOT:-${HERE}/../local/evidence/pv_carton_span_apply}"
exec "${HERE}/run_pv_carton_soft_direct_apply.sh" \
    --wrist-roll-range-deg 0 \
    --wrist-roll-gain 1 \
    "$@"
