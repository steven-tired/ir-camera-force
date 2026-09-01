# ir-camera-force

**Private research repository.** FLIR / Lepton / thermal force sensing for robot
grasping. Not for publication: it contains unpublished experiments, negative
results, and raw recordings.

## What this is

An attempt to estimate grip force from thermal imaging — the heat a fingertip
leaves on an object it presses. The work spans camera calibration
(FLIR↔OAK, RealSense↔Lepton), a hardware bridge for the FLIR ONE, a Lepton
capture path, and a series of grasp experiments.

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
ir_force/            IR estimation, capture, features, robot glue
ir_force/classifier/ the hard-classifier experiment line (see below)
experiments/         analyze / capture / record / report programs
calibration/         FLIR-OAK, RealSense-Lepton, thermal-project tooling
hardware/            FLIR ONE V4L2 bridge, Lepton firmware/host source
protocols/           study contracts, trial schedules, validators
local/               git-ignored: datasets, evidence, exports, calibration runs
```

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
