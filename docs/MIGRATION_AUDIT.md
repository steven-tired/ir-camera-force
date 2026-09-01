# Private repository migration audit

Run 2026-09-01, at `4c0beea`, after Tasks 1–6.

## Tests

`env -u PYTHONPATH ../.venv-lerobot/bin/python -m pytest -q` → **844 passed, 0 failed**.
Robot-free: no arm, no camera, no Lepton (the Lepton has been dead since
2026-07-17 regardless).

Against the pre-split source baseline of 1018 passed / 1 failed, the difference
is the modules that belong to the public repo — the SO-101 recorder, teleop
visualiser, and PressureVision integration — which is the split working, not
tests lost. The three hardware trees carry a further 33 + 1 tests that
`testpaths` does not reach; they were run directly (see `SOURCE_MAP.md`).

## Dependency direction

`ir-camera-force → mediapipe-so101 → LeRobot`, one way.

* `git -C ../mediapipe-so101 grep -nE '^\s*(from|import)\s+ir_force\b'` → empty.
  Asserted in CI here too, by `tests/test_gripper_adapter.py`.
  The pattern is deliberately not a bare name search: the public repo's own
  boundary test names `ir_force` as forbidden, which is the guard working.
* `tests/test_private_namespace_boundary.py` asserts this side does not own a
  copy of any public module, and that `ir_force/` is non-empty so those
  assertions cannot pass vacuously.
* `docs/PUBLIC_INTERFACE_LOCK.md` pins the public commit consumed.

## Artifact tracking

`git ls-files` contains no `.mp4`, `.avi`, `.npz`, `.pt`, or `.zip`. All 20 GB of
datasets, exports, evidence, and calibration runs sit under the ignored `local/`,
inventoried in `docs/LOCAL_ARTIFACT_MIGRATION_MANIFEST.tsv`.

## Repository state

Clean working tree, one worktree (`main`), no nested `.git` from any migrated
upstream.

## What is preserved but not verified

Stated plainly so nobody reads this audit as a green light on the hardware line:

* the C++ thermal calibration project was **not built** (needs OpenCV and the
  RealSense SDK) — see `docs/CALIBRATION_PROVENANCE.md`;
* `thermal-project`'s uncommitted "realsense2 optional" patch is preserved as a
  patch file and **not applied**;
* the two divergent `src/flirone.c` variants are kept as sibling trees and **not
  merged**;
* the Lepton hardware itself is dead; nothing here is live-sensor evidence.

## Not yet done

Stage 4: the old workspace still holds every source tree this repo was migrated
from. Nothing has been retired, and retirement needs explicit per-path approval.
