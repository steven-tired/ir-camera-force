# CLAUDE.md — ir-camera-force

FLIR / Lepton / thermal work split out of the `hand-teleop` meta-workspace.
Robot-free by design: nothing here drives an arm.

## The one rule

**Dependency runs one way:** `ir-camera-force → mediapipe-so101 → LeRobot`.
The public repo must never import `ir_force`. Asserted in CI here by
`tests/test_gripper_adapter.py::test_public_repo_does_not_import_the_private_package`,
and the public commit consumed is pinned in `docs/PUBLIC_INTERFACE_LOCK.md`.

Corollary: if a module is not actually about IR/thermal sensing, it belongs in
the public repo, not here. Four already moved out for that reason
(`ir_robot.py`, `hand_startup_gate.py`, `ir_pressure_proposal.py`, and the
duplicate classifier copy of the first). `tests/test_private_namespace_boundary.py`
fails if any of them comes back.

## Layout

- `ir_force/` — the **wt** line: pressure, shadow telemetry, thermal projection.
  The line that reached live integration.
- `ir_force/classifier/` — the **base** line: the hard-press classifier, which
  owns `ReferencePatch` and `extract_classifier_frame_features`.
  **The two lines are forked, not sequential. Nothing is merged.** Four modules
  exist in both with different content; see `docs/SOURCE_MAP.md` before touching.
- `experiments/`, `experiments/classifier/` — programs, imported by bare name
  (pyproject puts both on `pythonpath`).
- `ir_pressure_soak.py` — at the repo root, because its test resolves it there.
- `hardware/` — FLIR and Lepton source, each with `UPSTREAM.md`. Preserved, not
  actively developed; the Lepton has been dead since 2026-07-17.
- `calibration/` — thermal calibration code and artifacts.
- `local/` — git-ignored, ~20 GB of datasets/evidence/exports. Inventory in
  `docs/LOCAL_ARTIFACT_MIGRATION_MANIFEST.tsv`.

## Running things

```bash
env -u PYTHONPATH ../.venv-lerobot/bin/python -m pytest -q     # 1009 tests
```

`conftest.py` prepends the sibling `mediapipe-so101` src dirs: this machine
still carries an editable install pointing at the pre-split `webcam-input/`
tree, which `ir_pressure_soak` refuses to load.

The soak must stay **robot-free**. It imports public helpers with
`LEROBOT_TELEOPERATOR_SO101_WEBCAM_ROBOT_FREE_IMPORT=1` so the public package's
`__init__` skips the plugin classes; without it, `lerobot.motors` and a serial
stack come along. `tests/test_ir_pressure_soak.py` asserts both the allowed and
the prohibited module sets — update them when imports change, rather than
widening what the test tolerates.

## Paths

`ir_force/data_paths.py` owns every location this repo reads from. Never write
an absolute path in a tracked file: `dataset_root("name")` for recordings,
`CHECKOUT_ROOT` for files in this repo, `workspace_root()` /
`calibration_runs_root()` for the rig material that lives outside it. Each
honours an env var (`IR_FORCE_DATA_ROOT`, `IR_FORCE_WORKSPACE_ROOT`,
`IR_FORCE_CALIBRATION_RUNS`) and otherwise resolves to what the old absolute
paths named, so behaviour on this machine is unchanged.

`tests/test_no_developer_paths_in_published_files.py` enforces it across every
tracked file except `hardware/`, which is verbatim upstream code. A test that
needs a file only the rig has must `skip`, not fail — a bare clone runs 1127
of the 1160 tests and skips 33.

The module is `data_paths.py`, not `paths.py`: the public repo owns a
`paths.py`, and `tests/test_private_namespace_boundary.py` refuses the name.
