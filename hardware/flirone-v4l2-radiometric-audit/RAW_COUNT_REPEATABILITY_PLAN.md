# Raw-Count Repeatability Protocol

## Scope and Decision Rule

This protocol tests whether the FLIR bridge's **relative raw counts** are
repeatable enough to justify a later low-versus-hard press classifier. It does
not estimate Celsius, force, pressure, or compression in this phase.

Proceed to a classifier experiment only when the effects caused by FFC, bridge
restart, and a hot hand entering a non-target region are all materially smaller
than the low-versus-hard feature difference measured in a later, separate
experiment.

## Fixed Scene and ROIs

Use a stationary high-emissivity target, such as a matte black tape patch,
mounted rigidly in the FLIR view. Keep camera distance, focus, illumination,
and target pose unchanged throughout all runs.

Choose two non-overlapping coordinates in the native `80x60` raw grid:

- `--target-raw-roi`: an interior `10x10` or larger patch on the black target.
- `--control-raw-roi`: an equal-size stationary background/reference patch.

The RGB stream is a `160x128` display of the `80x60` raw grid. A raw ROI
`x,y,width,height` maps approximately to display pixels
`2*x,2*y,2*width,2*height`. Do not choose the bottom display text band or the
center crosshair/marker for either ROI.

For the dynamic test, the hot hand enters a non-target region only. It must
never occlude either raw ROI or move the target.

All run data is stored outside this source repository:

```bash
export SESSION_ROOT=/home/zhuokai/hand-teleop/datasets/ir_raw_repeatability/session-YYYYMMDD
cd /home/zhuokai/hand-teleop/tools/flirone-v4l2-radiometric-audit
```

The recorder creates `runs/<run-id>/` with:

```text
run_manifest.json     predeclared ROIs, phase sequence, and display contract
events.csv            CLOCK_MONOTONIC run and phase boundaries
raw/                  bridge-written uint16 little-endian raw frames + metadata
rgb/                  sampled /dev/video21 display PNGs
rgb_frames.csv        CLOCK_MONOTONIC RGB timestamps and filenames
```

The recorder does not execute `sudo`, change loopback devices, or restart the
camera. It prints the required bridge command for a second terminal. Start it,
confirm `/dev/video21` is available, then type exactly `READY` in the recorder
terminal. After `READY`, it also requires a newly written `raw_frame_*.json`
within 15 seconds; RGB frames alone do not start a valid run.

The RGB recorder logs a transient `/dev/video21` read failure as
`rgb_read_failure` and retries for up to three seconds. Raw metadata remains
the primary measurement for this protocol. A longer RGB outage is explicitly
recorded as `rgb_capture_abort`; it must be reported with the run rather than
silently treated as normal RGB coverage.

## 1. FFC Test

Record the stationary target for 240 seconds. Ensure the camera performs at
least one FFC/shutter event while it is recording.

```bash
python3 record_raw_repeatability_events.py \
  --session-root "$SESSION_ROOT" \
  --run-id ffc_01 \
  --mode ffc \
  --target-raw-roi X,Y,W,H \
  --control-raw-roi X,Y,W,H \
  --display-mode dynamic
```

The subsequent analysis intentionally fails this run if the raw metadata has
no `ffc` frame. Repeat it rather than interpreting an unverified FFC run.

## 2. Bridge-Restart Test

Run the following three times with the same static target and ROIs. Stop the
bridge after each recording, start a fresh bridge for the next one, and do not
move the target or camera between them. Each run captures 30 seconds of warmup,
20 seconds of stable measurement, and 40 seconds of observation.

```bash
python3 record_raw_repeatability_events.py \
  --session-root "$SESSION_ROOT" \
  --run-id restart_01 \
  --mode restart \
  --target-raw-roi X,Y,W,H \
  --control-raw-roi X,Y,W,H \
  --display-mode dynamic
```

Use `restart_02` and `restart_03` for the other two runs. The analysis reports
each stable median relative to `restart_01`; it does not treat an RGB colour
shift as a raw-count result.

## 3. Dynamic-Scene Test and Fixed-Range Display

First run the dynamic protocol in the legacy dynamic display mode. The target
and control regions remain fixed while a hot hand is moved into and out of an
unrelated part of the frame twice.

```bash
python3 record_raw_repeatability_events.py \
  --session-root "$SESSION_ROOT" \
  --run-id dynamic_agc_01 \
  --mode dynamic \
  --target-raw-roi X,Y,W,H \
  --control-raw-roi X,Y,W,H \
  --display-mode dynamic
```

Analyze it and record the emitted `fixed_range_suggestion.raw_low` and
`fixed_range_suggestion.raw_high`:

```bash
python3 analyze_raw_repeatability.py \
  --session-root "$SESSION_ROOT" \
  --summary-path "$SESSION_ROOT/repeatability_summary.json"
```

Then repeat the same dynamic sequence with those exact raw-count bounds. This
is the fixed-range visual validation: if the target raw counts stay stable, its
RGB palette location should also stay stable despite the hot hand elsewhere in
the frame.

```bash
python3 record_raw_repeatability_events.py \
  --session-root "$SESSION_ROOT" \
  --run-id dynamic_fixed_01 \
  --mode dynamic \
  --target-raw-roi X,Y,W,H \
  --control-raw-roi X,Y,W,H \
  --display-mode fixed \
  --fixed-raw-low LOW_COUNT \
  --fixed-raw-high HIGH_COUNT
```

Run the analyzer again after the fixed-range recording. It verifies that every
raw metadata frame uses the mapping declared in the run manifest and reports
target/control deltas for each baseline, hot-hand, and recovery phase.

## Interpretation

Use target and control together:

- Target and control both shift when the hand enters: scene-level/AGC/thermal
  contamination, not evidence for a local press signal.
- Target is stable while dynamic RGB changes in the dynamic run: expected AGC
  artefact; use fixed-range RGB only for visualization.
- Target shifts after FFC or restart by an amount comparable to a future
  low-versus-hard separation: raw counts are not yet suitable for that
  classifier.
- Target remains stable through FFC, restart, and hot-hand tests: raw counts
  pass this repeatability gate, but that still does not prove force or pressure.

Do not run calibration or classifier training as part of this protocol.
