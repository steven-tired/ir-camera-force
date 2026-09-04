# ir-camera-force

Estimating grip force from thermal imaging — the heat a fingertip leaves on an
object it presses. The work spans camera calibration (FLIR↔OAK,
RealSense↔Lepton), a hardware bridge for the FLIR ONE, a Lepton capture path,
and a series of grasp experiments.

**This is a research repository, published as-is.** It is a record of what was
tried, including what did not work: the negative results and the open gates are
kept rather than pruned, because for this question they are most of the finding.
Read `docs/CLAIMS_AND_GATES.md` before believing any number here — it separates
what was measured from what was assumed.

## Dependency direction

```
ir-camera-force  ->  mediapipe-so101  ->  LeRobot
```

One way, always. This repository consumes the public gripper contract
(`GripInput` / `GripperController`) at a pinned commit recorded in
`docs/PUBLIC_INTERFACE_LOCK.md`. The public repository knows nothing about this
one.

## Layout

```
ir_force/              the library: everything importable
ir_force/classifier/     the second, parallel line (see "Two parallel lines")
experiments/           the programs: capture, record, analyze, report
experiments/classifier/  same, for the classifier line
calibration/           camera-pair calibration: data, tooling, the C++ rig
hardware/              third-party camera bridges and firmware
tests/                 1162 of them; no camera, no robot
docs/                  provenance, and what is actually claimed
scripts/               the two live-rig launchers
local/                 git-ignored: datasets, evidence, exports, runs
```

### `ir_force/` — the library

Nothing here is a program. Four modules reach into the public repo —
`gripper_adapter.py` for the gripper contract itself, `ir_capture.py` (both
copies) and `ir_shadow_telemetry.py` for shared telemetry types — and the rest
is self-contained. Grouped by what they do:

| Group | Modules |
| --- | --- |
| Capture and devices | `ir_capture.py`, `ir_devices.py`, `realsense_camera.py`, `ir_diagnostics.py` |
| Signal extraction | `ir_features.py`, `ir_pressure.py`, `ir_analysis.py`, `ir_hand_roi.py` |
| Geometry | `ir_thermal_projection.py`, `ir_thermal_sparse_projection.py`, `ir_hand_calibration.py`, `pinch_geometry.py` |
| The single-finger study | `single_finger_*.py` — click ROI, curve protocol/runtime/analysis, thermal tracking and salvage |
| Datasets and reporting | `ir_dataset.py`, `ir_report.py`, `ir_shadow_telemetry.py`, `pinch_visualization.py` |
| Robot seam | `gripper_adapter.py` — the only file that touches `mediapipe-so101` |
| Paths | `data_paths.py` — every location this repo reads from |
| Types | `types.py` |

`ir_force/classifier/` repeats seven of those names — `ir_analysis`,
`ir_capture`, `ir_dataset`, `ir_devices`, `ir_diagnostics`, `ir_features`,
`ir_report` — with different implementations, and adds four of its own
(`ir_flir_registration`, `ir_foam_compression`, `ir_foam_setup`,
`ir_hard_classifier`). That is deliberate; see below.

### `experiments/` — the programs

Named by verb, and the verb is the whole taxonomy:

- **`record_*`** and **`capture_*`** — run a trial against real hardware and
  write a dataset. These are the only programs that need a camera.
- **`analyze_*`** and **`compare_*`** — read a recorded dataset, compute the
  figures and CSVs. Offline; these are what you can actually run from a clone
  if you have data.
- **`report_*`**, **`organize_*`**, **`extract_*`** — turn analysis output into
  the artifacts a writeup cites.
- **`live_*`** — real-time visualizations (`live_lepton_hand_shadow.py`,
  `live_lepton_projector_shadow.py`). Rig only.
- **`verify_*`**, **`characterize_*`**, **`validate_*`** — preflight and
  contract checks run before a session, not experiments themselves.
- `teleop_viz_ee.py` is the odd one out: the live teleop program with the IR
  panel attached, and the only program that moves the arm.

### `calibration/` — camera pairs

- **`flir_oak/`** — the FLIR ONE ↔ OAK projection, as fitted data
  (`.json` + samples `.csv`). No code; the fitting lives in `experiments/`.
- **`realsense_lepton/`** — the D435i ↔ Lepton 3.1R line. `tools/` holds the
  capture and refinement programs (`preview_capture.py`, `refine_extrinsic.py`)
  and the operator boilerplate that reproduces the runbook
  (`calib_run.sh`); `LEPTON_HANDOFF.md` and `LEPTON_PI_HOWTO.md` are the
  handover notes. The runbook itself is `REALSENSE_LEPTON_CALIBRATION.md` at
  the root.
- **`thermal_project/`** — the C++ calibration and thermal-point-cloud rig,
  third-party. **Its license is unresolved — see `THIRD_PARTY_NOTICES.md`.**

### `hardware/` — third-party, carried verbatim

Three trees, each at a recorded upstream commit with its own `LICENSE` and
`UPSTREAM.md`. Two of them carry local patches that exist in no other
repository — most notably the FLIR ONE Gen 2 thermal decoder
(`decode_g2_thermal()` in `flirone-v4l2/src/flirone.c`), without which a Gen 2
produces unusable frames on Linux.

### `docs/` — what is claimed, and where it came from

- **`CLAIMS_AND_GATES.md`** — read this first. Separates measured from
  assumed, and names the gates still open.
- **`SOURCE_MAP.md`**, **`CALIBRATION_PROVENANCE.md`**, **`MIGRATION_AUDIT.md`**
  — where every migrated file came from, at which commit.
- **`PUBLIC_INTERFACE_LOCK.md`** — the pinned `mediapipe-so101` commit.

### Root files

The experiment writeups (`IR_GRIP_FORCE_EXPERIMENT.md`,
`FOAM_COMPRESSION_EXPERIMENT.md`, `IR_HAND_PRESSURE_TELEOP.md`,
`IR_ASSISTED_TELEOP_PROGRESS.md`, …), the rig inventory (`HARDWARE.md`), the
preregistration (`IR_HAND_PINCH_PREREGISTRATION.md`) with its machine-checked
`*.schema.json` contracts and `ir_hand_pinch_trial_schedule.json`, and
`ir_pressure_soak.py`, the robot-free soak that exercises the shadow path
without hardware.

## Running it

```bash
python -m pytest -q          # 1162 tests, no camera or robot required
```

Recordings default to `local/datasets/` inside this checkout, and the live-rig
programs look for their sibling repositories one directory up. Both are
resolved in `ir_force/data_paths.py` and can be pointed elsewhere:

```bash
export IR_FORCE_DATA_ROOT=/mnt/big/ir-datasets       # where datasets live
export IR_FORCE_WORKSPACE_ROOT=~/robots/hand-teleop  # sibling repos + scripts/
export IR_FORCE_CALIBRATION_RUNS=/mnt/big/calib-runs # dated calibration runs
```

The datasets, the calibration runs and the FLIR/Lepton hardware itself are not
in this repository. Tests that need them skip; everything else runs on a bare
clone.

## License

Apache-2.0 for the work original to this repository — the same license as
`mediapipe-so101`.

`hardware/` and `calibration/thermal_project/` are third-party, carried at
recorded upstream commits, and keep their own terms: two GPL-2.0 trees, one
BSD-3-Clause, and one that states no license at all. `LICENSE` gives the exact
scope and `THIRD_PARTY_NOTICES.md` the per-tree detail.

## Two parallel lines

The IR work forked into two lines that were never merged, and both are kept:

- **`ir_force/`** — pressure / shadow / thermal-projection, from the
  `ir-hand-pressure-so101-teleop` branch. The line that reached live integration.
- **`ir_force/classifier/`** — the hard-press classifier experiments, from the
  `so101-webcam-diffusion` branch. It owns `ReferencePatch` and
  `extract_classifier_frame_features`, which the other line never had.

They share module names but not implementations. Do not merge them without
deciding, per function, which behaviour is intended; `docs/SOURCE_MAP.md` records
where every file came from.

## Status

See `docs/CLAIMS_AND_GATES.md`. In short: this is **research**, the Lepton
hardware has been dead since 2026-07-17, and nothing here drives a robot
autonomously.
