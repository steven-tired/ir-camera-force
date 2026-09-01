# flirone-v4l2 radiometric audit — upstream and local provenance

**Upstream:** https://github.com/fnoop/flirone-v4l2.git (same upstream as
`../flirone-v4l2`)
**Branch:** `codex/flir-radiometric-feasibility`
**Commit:** `e0b603c` ("feat: raw-count radiometric feasibility audit for FLIR ONE"),
on top of the same `b07e3a1` base
**Migrated from:** `tools/flirone-v4l2-radiometric-audit` (path relative to the
pre-split `hand-teleop` workspace root)
**Migrated on:** 2026-09-01
**Working tree at migration:** clean.

## What this branch adds

The raw-count radiometric feasibility audit: whether the FLIR ONE's raw counts
are repeatable enough to serve as a force proxy.

* `src/flirone.c` — the patched Gen 2 decoder **plus** raw-frame capture
  instrumentation (`raw_output_dir`, `raw_frame_limit`, frame hashing to detect
  repeated frames).
* `capture_raw_validation.py`, `raw_repeatability*.py`,
  `record_raw_repeatability_events.py`, `analyze_raw_repeatability.py`,
  `validate_radiometry.py`, `report_radiometric_feasibility.py`
* `FLIR_RADIOMETRIC_FEASIBILITY_REPORT.md`, `RAW_COUNT_REPEATABILITY_PLAN.md`,
  `radiometric_capability.json`
* `tests/`

## Why this is a sibling tree and not a merged subtree

Its `src/flirone.c` and `../flirone-v4l2/src/flirone.c` are divergent variants of
the same locally-patched decoder. Merging them into one tree would have meant
choosing one and silently discarding the other's changes — and the discarded one
would be the uncommitted patch that exists in no repository. Both are kept whole.

## Deliberately excluded

* `.git/`, `__pycache__/`, `.pytest_cache/`, `src/flirone.o`.
* `validation_capture/` (5.7 MB of captured device frames) — moved to the
  ignored `local/evidence/flir_radiometric_validation_capture/`. It is
  measurement evidence, not source.
