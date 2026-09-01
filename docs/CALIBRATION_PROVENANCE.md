# Thermal calibration provenance

Source paths are relative to the pre-split `hand-teleop` workspace root.
Migrated 2026-09-01.

## Code

| Destination | Source | Notes |
| --- | --- | --- |
| `calibration/thermal_project/` | `thermal-project-calibration` @ `933c8bc`, branch `hand-teleop-heldout-verifier` | The branch **contains all of `main`** (`merge-base --is-ancestor` confirmed) plus ~2600 lines: `HeldOutVerifier`, `CalibrationContracts`, `resolve_extrinsic`, `ThermalFrameAssembler`, and their tests. So only this worktree is copied. |
| `calibration/realsense_lepton/tools/` | `scripts/{calib_rgb_rehearsal.py, refine_extrinsic.py, preview_capture.py, lepton_spi_matrix.py, lepton_i2c_diag.cpp, calib_run.sh, compute_calib_from_captures.sh, run_lepton_stream.sh}` | the D435i↔Lepton line |
| `calibration/realsense_lepton/{LEPTON_HANDOFF,LEPTON_PI_HOWTO}.md` | `docs/` | |
| `calibration/flir_oak/` | already migrated in Task 2 | Only the OAK↔FLIR projection artifacts remain here. Two RealSense/Lepton files had been misfiled under it and were moved to `calibration/realsense_lepton/`. |

The OAK↔FLIR **registration code** is not in `calibration/`: it is
`experiments/calibrate_oak_flir_hand_pressure.py` and `ir_force/ir_hand_calibration.py`,
where their tests expect them. `calibration/flir_oak/` holds that pair's
artifacts only.

## Uncommitted patch not applied

`thermal-project` (main, `e0dbe7d`) had one uncommitted edit:
`stream/CMakeLists.txt`, making `realsense2` optional so the thermal-only
`lepton` target still builds without the RealSense SDK. The branch worktree's
version of that file has diverged (it still says `REQUIRED`), so applying the
patch is a merge decision, not a copy. It is preserved verbatim at
`calibration/thermal_project/patches-uncommitted/realsense2-optional.patch`
and **is not applied**.

## Worktrees

`thermal-project` and `thermal-project-calibration` are worktrees of one repo;
`thermal-project-calibration-runs/worktrees/` holds four detached run checkouts.

| Path | HEAD | State | Dirty | Destination |
| --- | --- | --- | --- | --- |
| `thermal-project` | `e0dbe7d` | branch `main` | `stream/CMakeLists.txt` | patch preserved (above); tree superseded by the branch |
| `thermal-project-calibration` | `933c8bc` | branch `hand-teleop-heldout-verifier` | clean | `calibration/thermal_project/` |
| `…-runs/worktrees/20260723T195904Z-attempt01` | `53f7f63` | detached | clean | run outputs only |
| `…-runs/worktrees/20260723T204122Z-attempt01` | `53f7f63` | detached | clean | run outputs only |
| `…-runs/worktrees/20260723T211932Z-attempt01` | `933c8bc` | detached | `calibration/calibration.xml`, `calibration/extrinsic.xml` | `local/calibration_runs/20260723T211932Z-attempt01/` |
| `…-runs/worktrees/20260724T210232Z-attempt01` | `933c8bc` | detached | same two XML + untracked `FINAL_fisheye/`, `FINAL_flir_brown/` | `local/calibration_runs/20260724T210232Z-attempt01/` |

`…-runs/runs/` and `…-runs/provenance-archive/` were copied to
`local/calibration_runs/` alongside them. All of `local/` is git-ignored: these
are calibration *artifacts*, not source. The reusable verifier code is tracked,
in `calibration/thermal_project/`.

The four run worktrees total 136 MB, almost all captured frames and build
output. Nothing under them is deleted by this migration; they stay in place
until the stage-4 retirement proposal is approved.

## Verification status

The C++ calibration project was **not built** as part of this migration —
it needs OpenCV and (for `depth_saver`) the RealSense SDK, and the code is
migrated for preservation rather than active use. The Python tools under
`calibration/realsense_lepton/tools/` are likewise copied as-is. Treat all of
this as preserved-not-verified, distinct from the software/locked checks in
`docs/SOURCE_MAP.md` that were actually run.

## Local artifacts (Task 6)

`docs/LOCAL_ARTIFACT_MIGRATION_MANIFEST.tsv` lists every tree, with source paths
relative to the pre-split workspace root. ~19.5 GB total, all under the ignored
`local/`.

These were **moved, not copied** — same filesystem, so the rename is instant and
costs no disk, and nothing is deleted. Copying would have duplicated 19.5 GB for
no benefit.

### Classification is by content, not by name

`scratch_lepton/` is mostly PressureVision carton work despite the name: only
`single_finger_surface_press_curve_01` and the extrinsic/fisheye/SB-detector
planning documents are IR, and only those moved. `evidence/` (20 GB) is entirely
PV, carton, and policy-deploy work and stayed where it is. `datasets/` split
cleanly: every non-`hand_tracking_*` tree is IR or Lepton.

### Path rewrites

The migrated experiment programs had hard-coded `/home/zhuokai/hand-teleop/datasets/…`
and `…/tools/flirone-v4l2/palettes/Iron2.raw` defaults, which the move would have
broken. They now point at this repo's `local/datasets/` and `hardware/flirone-v4l2/`.
They are still absolute paths — fine while this repo sits inside the old
workspace, and something to revisit if it ever moves.
