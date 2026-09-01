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
