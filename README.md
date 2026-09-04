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
calibration/           camera-pair calibration: data, tooling, additions
hardware/              third-party camera bridges and firmware
protocols/             the preregistration and its machine-checked schemas
docs/                  writeups, runbooks, provenance, what is claimed
docs/experiments/        the five experiment writeups
tests/                 1165 of them; no camera, no robot
scripts/               the two live-rig launchers
local/                 git-ignored: datasets, evidence, exports, runs
```

The repository root holds only `README.md`, `LICENSE`,
`THIRD_PARTY_NOTICES.md`, `pyproject.toml` and `conftest.py`. Everything else
lives in one of the directories above.

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
  handover notes. The runbook itself is `docs/REALSENSE_LEPTON_CALIBRATION.md`.
- **`thermal_project/`** — **not** a copy of the C++ calibration rig this line
  was built on. That rig is Anuj Natraj's
  [ThermalProject](https://github.com/AnujN9/ThermalProject), which states no
  license, so it is not redistributed here. What is kept is this project's own
  ~2600 lines of additions (`HeldOutVerifier`, `CalibrationContracts`,
  `resolve_extrinsic`, `ThermalFrameAssembler` and their tests) plus the
  patches, and a recipe that reconstructs the exact commit those calibration
  runs used. See `calibration/thermal_project/README.md`.

### `hardware/` — third-party, carried verbatim

Three trees, each at a recorded upstream commit with its own `LICENSE` and
`UPSTREAM.md`. Two of them carry local patches that exist in no other
repository — most notably the FLIR ONE Gen 2 thermal decoder
(`decode_g2_thermal()` in `flirone-v4l2/src/flirone.c`), without which a Gen 2
produces unusable frames on Linux.

### `docs/` — what is claimed, and where it came from

- **`CLAIMS_AND_GATES.md`** — read this first. Separates measured from
  assumed, and names the gates still open.
- **`experiments/`** — the five writeups, one per line of work:
  `IR_GRIP_FORCE_EXPERIMENT.md`, `FOAM_COMPRESSION_EXPERIMENT.md`,
  `IR_HAND_PRESSURE_TELEOP.md`, `IR_FOAM_CLASSIFIER_LOCAL_ARCHITECTURE.md`,
  `IR_ASSISTED_TELEOP_PROGRESS.md`.
- **`HARDWARE.md`** — the rig inventory, enumerated 2026-06-17.
- **`REALSENSE_LEPTON_CALIBRATION.md`** — the calibration runbook that
  `calibration/realsense_lepton/tools/calib_run.sh` reproduces.
- **`SOURCE_MAP.md`**, **`CALIBRATION_PROVENANCE.md`**, **`MIGRATION_AUDIT.md`**
  — where every migrated file came from, at which commit.
- **`PUBLIC_INTERFACE_LOCK.md`** — the pinned `mediapipe-so101` commit.

### `protocols/` — the preregistration

`IR_HAND_PINCH_PREREGISTRATION.md` and the three JSON files that make it
machine-checkable: two `*.schema.json` contracts and the frozen
`ir_hand_pinch_trial_schedule.json`.
`tests/test_ir_hand_pinch_preregistration_contract.py` holds the document and
the schemas to each other, so the preregistration cannot drift from what the
recording programs actually enforce.

## Running it

### Install

Python 3.12. The one hard dependency is a checkout of
[mediapipe-so101](https://github.com/steven-tired/mediapipe-so101) beside this
one — the gripper contract is imported from there, never vendored.

```bash
git clone https://github.com/steven-tired/ir-camera-force.git
git clone https://github.com/steven-tired/mediapipe-so101.git   # sibling
cd ir-camera-force
pip install -e .            # numpy, scipy, opencv-python
python -m pytest -q         # 1165 tests, no camera or robot required
```

`conftest.py` puts the sibling checkout on the path for the tests. If it lives
somewhere else, `export MEDIAPIPE_SO101_DIR=/path/to/mediapipe-so101`.

Do the editable install before running any program directly — it is what puts
`ir_force` on the path. The one exception is `experiments/teleop_viz_ee.py`,
which also needs the sibling checkout and so is launched through
`scripts/`, never by hand.

### Where things are written

Recordings go to `local/datasets/` inside this checkout, and the live-rig
programs look for their sibling repositories one directory up. Everything is
resolved in `ir_force/data_paths.py`; nothing is hardcoded:

```bash
export IR_FORCE_DATA_ROOT=/mnt/big/ir-datasets       # where datasets live
export IR_FORCE_WORKSPACE_ROOT=~/robots/hand-teleop  # sibling repos + scripts/
export IR_FORCE_CALIBRATION_RUNS=/mnt/big/calib-runs # dated calibration runs
```

The datasets, the calibration runs and the FLIR/Lepton hardware itself are not
in this repository. Tests that need them skip; a bare clone runs 1147 of 1165.

### Without any hardware

The soak exercises the whole pressure-shadow path against a recorded
calibration, so it is the fastest way to see the pipeline do something:

```bash
python experiments/ir_pressure_soak.py \
    --sidecar /tmp/shadow.csv \
    --calibration <a calibration json> \
    --duration-s 30
```

It is robot-free on purpose and stays that way: it imports the public helpers
with `LEROBOT_TELEOPERATOR_SO101_WEBCAM_ROBOT_FREE_IMPORT=1` so the plugin
classes — and with them `lerobot.motors` and a serial stack — never load.
`tests/test_ir_pressure_soak.py` asserts both the allowed and the prohibited
module sets.

With a dataset in hand, the `analyze_*` programs are the other half that needs
no camera:

```bash
python experiments/analyze_ir_grip_experiment.py --root <dataset>
python experiments/analyze_ir_hand_pressure.py --root <dataset> --out-prefix run01
```

Every program takes `--help`, and `--root` defaults to the right place under
`IR_FORCE_DATA_ROOT`, so it can usually be left off.

### With a camera

```bash
python experiments/view_ir_camera.py                     # is the thermal device alive?
python experiments/verify_ir_grip_setup.py --bird /dev/video2   # preflight, writes frames
python experiments/record_ir_grip_trial.py --bird /dev/video2 \
    --object carton --hardness medium --grip-level 1,2,3 --rep 1
```

`--bird` is the overhead camera and is required — there is no sensible default
for which `/dev/video*` it landed on. `--thermal` defaults to `/dev/video21`
and `--flir-visible` to `/dev/video20`, which is where the V4L2 bridge puts
them.

`verify_*` before `record_*` is not optional advice — it is what catches a dead
thermal device before a session's worth of trials records zeros. The FLIR ONE
needs the V4L2 bridge from `hardware/flirone-v4l2/` running first; that tree's
README covers building it.

### With the arm

```bash
./scripts/run_pv_carton_soft_direct_apply.sh    # PressureVision drives the grip
./scripts/run_pv_carton_span_apply.sh           # the 250 g carton mapping
```

**These move the arm. Keep the e-stop within reach.** They resolve their own
paths through `scripts/_common.sh` and refuse to start until every evidence
stream is recording — a run whose evidence never started is not evidence.

## Conventions

Three rules, each with a test that enforces it:

1. **The dependency runs one way**: `ir-camera-force → mediapipe-so101 →
   LeRobot`. The public repo must never import `ir_force`, and a module that is
   not actually about IR/thermal sensing belongs there rather than here — four
   have already moved out for that reason.
   (`test_gripper_adapter.py`, `test_private_namespace_boundary.py`)
2. **No absolute paths in tracked files.** Use `dataset_root("name")`,
   `CHECKOUT_ROOT`, `workspace_root()` or `calibration_runs_root()` from
   `ir_force/data_paths.py`. A test that needs a file only the rig has must
   `skip`, not fail. (`test_no_developer_paths_in_published_files.py`)
3. **Nothing new at the root.** Writeups go in `docs/experiments/`, study
   contracts in `protocols/`, programs in `experiments/`.

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
