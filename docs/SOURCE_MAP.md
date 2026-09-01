# Source Map

Provenance for everything migrated from the `hand-teleop` meta-workspace on
2026-09-01.

**All source paths are relative to the old workspace root**, never absolute — an
absolute path would fail the migration audit and record a developer's home
directory.

Shorthand for the two source checkouts:

| Tag | Path | Branch | HEAD |
| --- | --- | --- | --- |
| `base` | `webcam-input/` | `so101-webcam-diffusion` | 4e1f2fb |
| `wt` | `webcam-input/.worktrees/ir-hand-pressure-so101-teleop/` | `ir-hand-pressure-so101-teleop` | af97c9f |

Disposition: `verbatim` (unchanged) · `path-rewrite` (imports/paths only) ·
`rewritten` (reconstructed) · `new`.

## The branch fork

`base` and `wt` are **not** successive versions of one line. They diverged and
were never merged, and four modules exist in both with different content:

| Module | base | wt | Divergence |
| --- | --- | --- | --- |
| `ir_capture.py` | 319 | 533 | 274 lines; wt adds socket/threading streaming |
| `ir_dataset.py` | 198 | 199 | 1 line |
| `ir_features.py` | 277 | 191 | 86 lines; **base has `ReferencePatch` and `extract_classifier_frame_features`, wt has neither, and they appear nowhere else in wt** |
| `ir_robot.py` | 107 | 311 | 204 lines; wt adds timing/callback machinery |

Both lines are kept, in separate directories, and nothing is merged:

- `ir_force/` — the **wt** line: pressure, shadow telemetry, thermal projection.
  This is the line that reached live integration.
- `ir_force/classifier/` — the **base** line: the hard-press classifier
  experiments, which own `ReferencePatch` and `extract_classifier_frame_features`
  along with `record_ir_hard_classifier_experiment.py` and `test_ir_features.py`.

Merging them requires deciding, per function, which behaviour is intended. That
decision has not been made and is not made by this migration.

## Scaffold

| Destination | Source | Disposition |
| --- | --- | --- |
| `README.md` | — | new |
| `pyproject.toml`, `.gitignore` | — | new |
| `docs/PUBLIC_INTERFACE_LOCK.md` | — | new (pins the public commit) |
| `docs/CLAIMS_AND_GATES.md` | `wt` IR_GRIP_FORCE_EXPERIMENT.md, IR_HAND_PINCH_PREREGISTRATION.md, HARDWARE.md | rewritten |

_Module-level rows are appended as each group migrates._

## Hardware (Task 4)

Source paths are relative to the pre-split `hand-teleop` workspace root. Per-tree
provenance — upstream URL, exact commit, and the uncommitted work carried across —
is in each tree's own `UPSTREAM.md`.

| Destination | Source | Disposition |
| --- | --- | --- |
| `hardware/flirone-v4l2/` | `tools/flirone-v4l2` @ `b07e3a1` + dirty tree | copied, no `.git`, no binaries |
| `hardware/flirone-v4l2-radiometric-audit/` | `tools/flirone-v4l2-radiometric-audit` @ `e0b603c` (branch `codex/flir-radiometric-feasibility`) | copied, no `.git`, no `validation_capture/` |
| `hardware/lepton/` | `lepton-module-ir-hand-pinch-phase1` @ `d5b6879` (branch `ir-hand-pinch-phase1`) + dirty tree | copied, no `.git`, no build output |
| `local/evidence/flir_radiometric_validation_capture/` | `tools/flirone-v4l2-radiometric-audit/validation_capture/` | moved to ignored evidence (5.7 MB of captured frames) |

`lepton-module` (master) is **not** separately migrated: it sits at the same
commit `d5b6879` as the worktree above, so the worktree copy is a strict superset.

### Recorded conflict: two divergent `src/flirone.c`

The plan required that the radiometric-audit content not silently overwrite the
working decoder patch. It would have:

| Tree | `src/flirone.c` | Contains |
| --- | --- | --- |
| `hardware/flirone-v4l2/` | 884 lines | Gen 2 `decode_g2_thermal()` patch, `/dev/video20`+`/dev/video21` remap — **uncommitted upstream, exists in no repository** |
| `hardware/flirone-v4l2-radiometric-audit/` | 1198 lines | the same decoder patch **plus** raw-frame capture instrumentation and frame hashing |

They are divergent variants of the same locally-patched file. Merging into one
subtree would have meant discarding one — and the discarded side would be the
patch with no other home. Both are kept whole as sibling trees.

### Hardware trees carry their own test suites

`testpaths = ["tests"]` means the repo suite does not reach them. They are run
directly, and were run at migration:

| Tree | Check | Result |
| --- | --- | --- |
| `hardware/flirone-v4l2/` | `make` | compiles clean |
| `hardware/flirone-v4l2-radiometric-audit/` | `make`; `pytest tests` | compiles clean; 33 passed |
| `hardware/lepton/software/` | `cmake` + `make` + `ctest` | configures, builds (incl. the phase-1 `test_lepton_vospi` binary), 1/1 passed |

These are **software/locked** checks. No camera was attached, and the Lepton
itself has been dead since 2026-07-17 — none of this is live-sensor evidence.
