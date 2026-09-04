# IR Squeeze/Contact Proxy Teleop

The code and CLI retain `ir-pressure` names for compatibility. The signal is an
IR-derived squeeze/contact proxy from the operator's hand. It is not physical
force or pressure, and it is not calibrated in Newtons or pressure units.

Apply mode and formal recording remain blocked while Gates 0-4 are incomplete.
The Gate 1 executable is now available, but Gate 1 is not accepted until the
hardware soak and manual fault injections below pass.

## Calibration Fit

The Gate 1 runtime expects the persistent patched FLIR bridge documented in
`../../docs/experiments/IR_GRIP_FORCE_EXPERIMENT.md`. Load both loopback devices and start that bridge
before calibration or a soak:

```bash
sudo modprobe v4l2loopback video_nr=20,21 card_label=FLIR_ONE_VISIBLE,FLIR_ONE_THERMAL exclusive_caps=1,1
cd hardware/flirone-v4l2
sudo ./flirone palettes/Iron2.raw
```

`/dev/video21` must report colorized `RGB3` at `160x128`. It is relative
intensity, not radiometric temperature.

Run guided OAK-to-FLIR calibration:

```bash
cd $WORKSPACE_ROOT/webcam-input/.worktrees/ir-hand-pressure-so101-teleop/lerobot_teleoperator_so101_webcam
env -u PYTHONPATH $WORKSPACE_ROOT/.venv-lerobot/bin/python \
  calibrate_oak_flir_hand_pressure.py \
  --thermal /dev/video21 \
  --out-dir $WORKSPACE_ROOT/webcam-input/.worktrees/ir-hand-pressure-so101-teleop/lerobot_teleoperator_so101_webcam/calibration
```

Move a warm fingertip across the shared OAK/FLIR view at multiple image
positions and depths. Press `s` to save a sample. After at least 12 samples,
press `f` to save `calibration/oak_flir_hand_pressure_projection.json`.

Runtime accepts a calibration only when all four checks pass:

- at least 12 fit samples;
- RMS residual at most 8 pixels;
- maximum residual at most 16 pixels;
- calibration image size exactly `160x128`.

The RMS and maximum residuals are in-sample fit diagnostics only. Passing these
runtime checks is insufficient for Gate 2 and must not be reported as held-out
accuracy.

## Independent Holdout

After fitting, collect at least 20 new correspondences that were not used for
the fit. Cover left, center, right, top, center, bottom, image edges, and at
least two depths. Report all of the following on those independent points:

- RMS projection error in pixels;
- p95 projection error in pixels;
- maximum projection error in pixels;
- out-of-FOV count;
- runtime-ROI hit rate.

An affine projection outside `[0, width) x [0, height)` is invalid. Never count
a point clipped to the image border as a hit. Gate 2 requires either a holdout
maximum error within the runtime ROI margin or an adaptive margin with at least
99% holdout ROI hit rate.

## Gate 1 Status

`../../experiments/ir_pressure_soak.py` is the standalone robot-free Gate 1 executable. It opens
only OAK and FLIR sources plus the required CSV sidecar. It does not enumerate,
construct, connect, read, torque, or command a robot, and every CSV row has
`command_sent=false`.

Do not replace this executable with `teleop_viz_ee.py`, `record_so101_ee.py`, or
their wrappers. Those live/recording paths connect to the robot, enable torque,
ramp to a start pose, and send commands even when IR proxy computation is in
shadow mode.

Run this short robot-free smoke test first:

```bash
cd $WORKSPACE_ROOT/webcam-input/.worktrees/ir-hand-pressure-so101-teleop/lerobot_teleoperator_so101_webcam
env -u PYTHONPATH $WORKSPACE_ROOT/.venv-lerobot/bin/python ../../experiments/ir_pressure_soak.py \
  --duration-s 10 \
  --max-oak-stall-ms 500 \
  --min-cycles 0 \
  --thermal /dev/video21 \
  --calibration $WORKSPACE_ROOT/webcam-input/.worktrees/ir-hand-pressure-so101-teleop/lerobot_teleoperator_so101_webcam/calibration/oak_flir_hand_pressure_projection.json \
  --sidecar local/datasets/ir_hand_pressure_viability/gate1_smoke.csv
```

The full Gate 1 command is exactly 1800 seconds and requires at least 100
complete open/close/open cycles:

```bash
cd $WORKSPACE_ROOT/webcam-input/.worktrees/ir-hand-pressure-so101-teleop/lerobot_teleoperator_so101_webcam
env -u PYTHONPATH $WORKSPACE_ROOT/.venv-lerobot/bin/python ../../experiments/ir_pressure_soak.py \
  --duration-s 1800 \
  --max-oak-stall-ms 500 \
  --min-cycles 100 \
  --thermal /dev/video21 \
  --calibration $WORKSPACE_ROOT/webcam-input/.worktrees/ir-hand-pressure-so101-teleop/lerobot_teleoperator_so101_webcam/calibration/oak_flir_hand_pressure_projection.json \
  --sidecar local/datasets/ir_hand_pressure_viability/gate1_1800s_100cycles.csv
```

During Gate 1, perform and record these manual injections:

1. Cover the FLIR view until a thermal rejection latches, then uncover it. The
   run must continue, report the rejection, hold the proposal without increased
   closure, and re-arm only after a valid inactive `baseline` reading. A
   successful qualifying run requires this recovery and re-arm after the FLIR
   fault; ending with `pressure_fault_latched` is not success.
2. In a separate fault smoke run, stop the FLIR loopback bridge. This is an
   explicit safe rejection: a raised read reports `thermal_unavailable`; a
   blocked producer that retains its last sample reports `thermal_stale` after
   the thermal age limit. The rejection is command-free, the run continues
   safely, and it finishes nonzero if it cannot recover. Automatic camera
   reconnect is deferred, so restart the bridge and executable after this stop
   condition.
3. Move both hands out of the fresh OAK frames without stopping OAK publication.
   The state must become `HOLD`, skip FLIR estimation, and resume when a valid
   hand and clutch state return. Fresh hand loss is not an OAK watchdog fault.
4. Force device re-enumeration by disconnecting and reconnecting a camera. An
   OAK X_LINK failure or publication stall must return a graceful nonzero final
   summary. A FLIR read failure must remain command-free and end latched unless
   a valid baseline recovery occurs. Restart the executable after re-enumeration.

Stop and reject the run on `oak_failed`, an OAK watchdog exit, sidecar disable,
fault-induced closure increase, unrecovered final pressure latch, insufficient
cycles, cleanup failure, or any setup/runtime error. Preserve the sidecar and
final JSON summary for review. The summary includes counts, rejection reasons,
cleanup failures, and p50/p95/p99/max for OAK age, thermal age, pair skew, loop
period, and control latency.

Out-of-frame projection reports `projection_out_of_fov`; it is never clipped to
a border. A runtime thermal frame that differs from calibration `image_size`
reports unavailable `thermal_shape_mismatch`. Runtime sensor faults enter a
latched safe hold and require a valid inactive `baseline` reading to re-arm.

## Release Gates

- **Gate 0, software:** full webcam and teleoperator tests plus
  `git diff --check`; prove fault transitions cannot increase closure and shadow
  commands are byte-for-byte equivalent to legacy OAK-only commands.
- **Gate 1, read-only soak:** executable available; complete the hardware soak
  and fault injections, then derive age/skew thresholds from measured
  distributions.
- **Gate 2, independent holdout (blocked):** complete the holdout protocol above
  with no clipped out-of-FOV acceptance.
- **Gate 3, incremental signal value (blocked):** against an independent load/force
  reference at fixed finger geometry, show repeatable held-out value beyond
  pinch alone. Otherwise rename the feature to contact/thermal state and stop
  actuation work.
- **Gate 4, low-authority powered gripper (blocked):** arm stationary, no formal
  recording, overdrive capped at `3`, explicit re-arm, and zero fault-induced
  closure increases.
- **Gate 5, teleop and recording release (blocked):** randomized OAK-only versus
  IR-assisted trials and at least 50 non-fragile grasps, with no safety event or
  unhandled camera failure and no worse task success than OAK-only.

Production code rejects `--ir-pressure`, `SO101_IR_PRESSURE=1`, and direct
non-shadow pressure-controller construction until Stage 3 physical
authorization. Only `--ir-pressure-shadow` or
`SO101_IR_PRESSURE_SHADOW=1` remain available. Formal recording remains
blocked until Gate 5 releases it.

## Deferred Work

Broader hand-hull ROI fallback, full 3D OAK-FLIR calibration/extrinsics, robust
regression, and automatic camera reconnect are intentionally deferred. The
current model remains affine `(x, y, z) -> (u, v)`.
