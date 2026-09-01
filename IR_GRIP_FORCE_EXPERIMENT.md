# IR Grip-Force Viability Experiment

## Purpose

This experiment checks whether the FLIR ONE stream adds usable grip-force or
hardness signal beyond SO-101 gripper telemetry during passive grasping.

`/dev/video21` is the `flirone-v4l2` colorized thermal loopback, not a
radiometric temperature stream. Treat it as relative palette-index or intensity
data and analyze baseline-delta changes, not Celsius.

## Current 2026-07-07 Result

The most useful current pathway is the hard-block continuous sweep, not the
fixed low/med/high foam trials.

- SO-101 `present_load` and `present_current` are useful as relative gripper
  effort proxies, but they are raw servo registers rather than calibrated force.
  They can jump during motion because they include friction, backlash,
  controller transients, and quantization.
- Foam fixed-level trials produced weak IR/load relationship. The contact patch
  was too soft/unstable and the block could move.
- Soft-object dataset status: no significant useful IR-load relationship was
  found in the foam/soft trials. The two foam sweep trials in
  `sweep_ir_load_analysis.csv` had pooled Spearman correlations near zero or
  negative: `load_pos` vs `area` `0.057`, `load_pos` vs `mean_delta` `-0.230`,
  and `load_pos` vs `max_delta` `-0.030`.
- Hard-block sweep trials with a tight thermal contact ROI gave the clearest IR
  signal so far.
- `mean_delta` inside the contact ROI is the primary IR feature. It tracks load
  more consistently than thresholded `area_px`.
- `area_px` is currently unreliable because threshold masks can be dominated by
  background/framing artifacts or one-frame segmentation spikes.
- The current useful analysis window is the moving hard-block sweep segment
  where commanded gripper goal moves from about `30` to `25`.

Latest hard-block moving-window result files:

- `/home/zhuokai/hand-teleop/ir-camera-force/local/datasets/ir_grip_force_viability/hard_sweep_rep02_rep03_rep04_goal30_to25_moving_mean_delta_vs_load.png`
- `/home/zhuokai/hand-teleop/ir-camera-force/local/datasets/ir_grip_force_viability/hard_sweep_rep02_rep03_rep04_goal30_to25_moving_mean_delta_vs_load.csv`
- `/home/zhuokai/hand-teleop/ir-camera-force/local/datasets/ir_grip_force_viability/hard_sweep_rep02_rep03_rep04_goal30_to25_moving_mean_delta_vs_load_summary.csv`

Soft-object result files:

- `/home/zhuokai/hand-teleop/ir-camera-force/local/datasets/ir_grip_force_viability/sweep_ir_load_analysis.csv`
- `/home/zhuokai/hand-teleop/ir-camera-force/local/datasets/ir_grip_force_viability/sweep_ir_load_analysis.png`
- `/home/zhuokai/hand-teleop/ir-camera-force/local/datasets/ir_grip_force_viability/soft_sweep_ir_load_summary.csv`

These soft-object artifacts are retained as a negative/low-signal result rather
than as missing data.

For that moving window:

| Trial | Rows | Pearson `load` vs `mean_delta` | Spearman `load` vs `mean_delta` |
|---|---:|---:|---:|
| `rep02` | 8 | 0.871 | 0.922 |
| `rep03` | 26 | 0.742 | 0.770 |
| `rep04` | 29 | 0.739 | 0.766 |
| combined | 63 | 0.503 | 0.472 |

Interpretation: there is a plausible positive relationship between hard-block
compression load and ROI `mean_delta`, but it is not yet a calibrated force
model. More repeats with the same fixture/camera/ROI are needed before using it
for control.

## Outputs

- Dataset root: `/home/zhuokai/hand-teleop/ir-camera-force/local/datasets/ir_grip_force_viability`
- Per trial:
  - `metadata.json`
  - `telemetry.csv`
  - `thermal/`
  - `bird/`
  - `flir_visible/` when captured
- Feature outputs:
  - `ir_features.csv`
  - `ir_features_hard_roi.csv` when extracting with the current hard-block ROI
  - `overlays/`
  - ROI-specific overlay directories such as `overlays_hard_roi/`
- Final outputs:
  - `analysis/summary.json`
  - `analysis/ir_area_vs_current.png`

## Required hardware state

- SO-101 follower arm/gripper connected at
  `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14110850-if00`
- Bird-view RGB camera available on a stable
  `/dev/v4l/by-id/...-video-index0` path chosen from
  `ls -l /dev/v4l/by-id/`
- FLIR ONE loopbacks reserved as:
  - visible RGB: `/dev/video20`
  - thermal colorized output: `/dev/video21`

The robot-moving IR scripts in this package do not expose a
`--robot-use-degrees` or `--robot.use_degrees` CLI flag. They already create
`SO101FollowerConfig(... use_degrees=False ...)` internally, which matches the
package hardware requirement for SO-101 commands here.

`record_ir_grip_trial.py` starts each trial by opening the gripper to
`--open-pos` (`100` by default), then after the hold window it releases to
`--release-pos` (`90` by default) and waits `--release-settle-s` seconds before
disconnecting. This avoids leaving a fixtured block stuck in the gripper after
the measurement.

## Step 0: Bring up FLIR loopbacks

Important bridge state from 2026-07-06:

- The persistent working bridge is
  `/home/zhuokai/hand-teleop/ir-camera-force/hardware/flirone-v4l2`, remote
  `https://github.com/fnoop/flirone-v4l2.git`.
- The checked-out upstream bridge is too old for this FLIR ONE thermal payload.
  This camera reports `ThermalSize=10332`, which is a Gen3/Lepton2-style
  `63 * 164` byte packet stream. The local `src/flirone.c` is patched to:
  - keep visible output on `/dev/video20`
  - keep colorized thermal output on `/dev/video21`
  - decode `ThermalSize == 10332` as 80x60 big-endian Lepton packet rows,
    then upscale to the existing 160x128 RGB loopback
  - keep the old G2 decode path for larger 160x120 payloads
- `/tmp/flirone-v4l2` was the original bring-up copy, but `/tmp` is mounted as
  `tmpfs` on this machine, so it is scratch space and can be cleared after
  reboot or power-off. Use the persistent workspace copy above. The symptom of
  an unpatched bridge is a noisy `/dev/video21` image with `-nan/.../194.9`
  overlay values.

Load the loopback devices:

```bash
sudo modprobe v4l2loopback video_nr=20,21 card_label=FLIR_ONE_VISIBLE,FLIR_ONE_THERMAL exclusive_caps=1,1
```

Start `flirone-v4l2` with the Iron2 palette:

```bash
cd /home/zhuokai/hand-teleop/ir-camera-force/hardware/flirone-v4l2
sudo ./flirone palettes/Iron2.raw
```

Optional live viewer for the thermal loopback:

```bash
ffplay \
  -fflags nobuffer \
  -flags low_delay \
  -framedrop \
  -probesize 32 \
  -analyzeduration 0 \
  -sync ext \
  -f v4l2 \
  -input_format rgb24 \
  -video_size 160x128 \
  -framerate 30 \
  -vf scale=800:640 \
  /dev/video21
```

Expected behavior:

- `/dev/video20` carries FLIR visible RGB
- `/dev/video21` reports `RGB3` at `160x128`
- `/dev/video21` should be interpreted as a colorized relative-intensity stream
- The low-latency viewer above should track camera motion without the large
  delay seen from default `ffplay` buffering

## Step 1: Verify cameras

List the available stable camera symlinks, then paste the exact workspace
camera path into `BIRD_PATH`. Do not use historical OpenCV enumeration such as
`/dev/video2` for this experiment.

```bash
cd /home/zhuokai/hand-teleop/webcam-input/lerobot_teleoperator_so101_webcam
ls -l /dev/v4l/by-id/
BIRD_PATH=/dev/v4l/by-id/<paste-the-workspace-camera-video-index0-symlink>
test -e "$BIRD_PATH"
env -u PYTHONPATH /home/zhuokai/hand-teleop/.venv-lerobot/bin/python verify_ir_grip_setup.py \
  --bird "$BIRD_PATH" \
  --thermal /dev/video21 \
  --flir-visible /dev/video20
```

Pass condition:

- `verify_ir_grip_setup.py` fails before capture unless:
  - `--bird` is a stable `/dev/v4l/by-id/...-video-index0` path
  - `--thermal` is `/dev/video21`
  - `v4l2-ctl` succeeds for the thermal device and reports `RGB3` at `160x128`
- `--allow-unstable-bird-path` and `--allow-thermal-mismatch` are escape
  hatches for deliberate overrides only
- `setup_check/bird.png`, `setup_check/thermal.png`, and
  `setup_check/flir_visible.png` are written under
  `/home/zhuokai/hand-teleop/ir-camera-force/local/datasets/ir_grip_force_viability/setup_check/`

## Step 2: Choose low/med/high gripper targets

```bash
cd /home/zhuokai/hand-teleop/webcam-input/lerobot_teleoperator_so101_webcam
env -u PYTHONPATH /home/zhuokai/hand-teleop/.venv-lerobot/bin/python characterize_ir_grip_current.py \
  --targets 90,80,70,60,50,40 \
  --min-current-gap 10
```

Pass condition:

- `/home/zhuokai/hand-teleop/ir-camera-force/local/datasets/ir_grip_force_viability/grip_targets.json`
  contains `selected_targets.low`, `selected_targets.med`, and
  `selected_targets.high`
- Optional extra probe: `selected_targets.xhigh` may be set manually to `10.0`
  for one stronger compression trial. Keep `xhigh` out of final low/med/high
  monotonic claims unless it proves mechanically stable.

## Step 3: Record one smoke trial

Current next action after the 2026-07-06 handoff: use the hard block, keep the
FLIR thermal setup aimed at the gripper contact area, and record one smoke
trial before starting the full matrix.

If the block flips or slides, strengthen the passive-jaw or table fixture
before collecting analysis data. Visible block motion changes the thermal
contact patch and can make stronger grip levels look smaller than weaker grip
levels.

```bash
cd /home/zhuokai/hand-teleop/webcam-input/lerobot_teleoperator_so101_webcam
BIRD_PATH=/dev/v4l/by-id/<paste-the-workspace-camera-video-index0-symlink>
test -e "$BIRD_PATH"
env -u PYTHONPATH /home/zhuokai/hand-teleop/.venv-lerobot/bin/python record_ir_grip_trial.py \
  --object hard-block \
  --hardness solid \
  --grip-level low \
  --rep 1 \
  --thermal /dev/video21 \
  --bird "$BIRD_PATH" \
  --flir-visible /dev/video20
```

Pass condition:

- Trial folder exists under
  `/home/zhuokai/hand-teleop/ir-camera-force/local/datasets/ir_grip_force_viability/trials/`
- It contains `metadata.json`, `telemetry.csv`, `thermal/*.png`, and `bird/*.png`

## Step 4: Record the passive trial matrix

Use four objects:

- `foam-block`
- `sponge`
- `wood-block`
- `plastic-block`

For each object, record:

- grip levels `low`, `med`, `high`
- reps `1`, `2`, `3`

That yields 36 passive trials total. Keep object pose fixed across reps for the
same object.

`xhigh` is an optional diagnostic level at gripper target `10.0`. Use it only
after a low/med/high set if the object remains stable and you need to test
whether stronger compression makes the contact patch visible.

## Step 4b: Record a continuous hard-block sweep trial

For force-feedback modeling, prefer a continuous slow-close sweep over only
fixed low/med/high/xhigh target points. The sweep records baseline, continuous
closing, terminal hold, and release in one trial named with grip level
`sweep`.

Current recommended hard-block protocol:

- open gripper to `50`
- settle at `50` for `2.5s`
- record `2s` baseline while already at the sweep start
- sweep from `50` to `25` over `14s`
- skip terminal hold with `--hold-s 0` for the cleanest moving-window analysis
- release to `90`

This avoids the bad baseline seen when the gripper moved during baseline, and it
avoids overloading the hard block at target `10`.

```bash
cd /home/zhuokai/hand-teleop/webcam-input/lerobot_teleoperator_so101_webcam
BIRD_PATH=/dev/v4l/by-id/<paste-the-workspace-camera-video-index0-symlink>
test -e "$BIRD_PATH"
env -u PYTHONPATH /home/zhuokai/hand-teleop/.venv-lerobot/bin/python record_ir_grip_sweep.py \
  --object hard-block-passive-jaw-fixtured \
  --hardness solid \
  --rep 5 \
  --thermal /dev/video21 \
  --bird "$BIRD_PATH" \
  --flir-visible /dev/video20 \
  --open-pos 50 \
  --sweep-start-pos 50 \
  --pre-baseline-settle-s 2.5 \
  --target-pos 25 \
  --sweep-s 14 \
  --hold-s 0 \
  --release-pos 90
```

Do not pass `--record-flir-visible` unless continuous FLIR visible frames are
needed; by default `/dev/video20` is captured only as a preflight alignment
snapshot.

The confirmation prompt is case-sensitive; type exactly `YES`.

## Step 5: Record one warmed sanity trial

Warm one object enough that the thermal patch is visually obvious, then run:

```bash
cd /home/zhuokai/hand-teleop/webcam-input/lerobot_teleoperator_so101_webcam
BIRD_PATH=/dev/v4l/by-id/<paste-the-workspace-camera-video-index0-symlink>
test -e "$BIRD_PATH"
env -u PYTHONPATH /home/zhuokai/hand-teleop/.venv-lerobot/bin/python record_ir_grip_trial.py \
  --object wood-block \
  --hardness solid \
  --grip-level high \
  --rep 1 \
  --warmed \
  --thermal /dev/video21 \
  --bird "$BIRD_PATH" \
  --flir-visible /dev/video20
```

The warmed trial is a pipeline sanity check only. Exclude it from passive
grip-force or hardness claims.

## Step 6: Extract features

Recommended extraction when the `flirone-v4l2` palette file is available:

```bash
cd /home/zhuokai/hand-teleop/webcam-input/lerobot_teleoperator_so101_webcam
env -u PYTHONPATH /home/zhuokai/hand-teleop/.venv-lerobot/bin/python extract_ir_grip_features.py \
  --root /home/zhuokai/hand-teleop/ir-camera-force/local/datasets/ir_grip_force_viability \
  --baseline-frames 20 \
  --palette /home/zhuokai/hand-teleop/ir-camera-force/hardware/flirone-v4l2/palettes/Iron2.raw \
  --invert-palette
```

Current hard-block ROI extraction:

```bash
cd /home/zhuokai/hand-teleop/webcam-input/lerobot_teleoperator_so101_webcam
env -u PYTHONPATH /home/zhuokai/hand-teleop/.venv-lerobot/bin/python extract_ir_grip_features.py \
  --root /home/zhuokai/hand-teleop/ir-camera-force/local/datasets/ir_grip_force_viability \
  --trial-glob 'hard-block-passive-jaw-fixtured_solid_sweep_rep*' \
  --baseline-frames 20 \
  --palette /home/zhuokai/hand-teleop/ir-camera-force/hardware/flirone-v4l2/palettes/Iron2.raw \
  --invert-palette \
  --thermal-roi 25,35,115,80 \
  --feature-name ir_features_hard_roi.csv \
  --overlay-dir overlays_hard_roi
```

The ROI is `x=25, y=35, width=115, height=80` in `/dev/video21` thermal frame
coordinates. It is intentionally small enough to ignore most background and
camera-framing artifacts, but should be rechecked if the camera mount changes.

Fallback extraction without palette reconstruction:

```bash
cd /home/zhuokai/hand-teleop/webcam-input/lerobot_teleoperator_so101_webcam
env -u PYTHONPATH /home/zhuokai/hand-teleop/.venv-lerobot/bin/python extract_ir_grip_features.py \
  --root /home/zhuokai/hand-teleop/ir-camera-force/local/datasets/ir_grip_force_viability \
  --baseline-frames 20
```

Pass condition:

- Each trial directory gets `ir_features.csv`
- Hard-block sweep trial directories get `ir_features_hard_roi.csv`
- `overlays/` is populated for visual review
- `overlays_hard_roi/` is populated for ROI-specific visual review

For a readable one-trial review, generate a smoke report:

```bash
cd /home/zhuokai/hand-teleop/webcam-input/lerobot_teleoperator_so101_webcam
env -u PYTHONPATH /home/zhuokai/hand-teleop/.venv-lerobot/bin/python report_ir_grip_trial.py \
  /home/zhuokai/hand-teleop/ir-camera-force/local/datasets/ir_grip_force_viability/trials/hard-block_solid_low_rep01
```

This writes:

- `trial_report.png`: thermal frames, mask overlay, bird view, IR feature
  time series, and gripper telemetry aligned by frame
- `trial_report.json`: baseline-vs-hold summary values

## Step 7: Analyze GO/NO-GO

For the current hard-block sweep data, analyze `mean_delta` first. Keep
`area_px` as a diagnostic only until segmentation is improved.

Current manually generated review artifacts:

- `hard_sweep_rep02_rep03_rep04_goal30_to25_moving_mean_delta_vs_load.png`
- `hard_sweep_rep02_rep03_rep04_goal30_to25_moving_mean_delta_vs_load.csv`
- `hard_sweep_rep02_rep03_rep04_goal30_to25_moving_mean_delta_vs_load_summary.csv`

The fixed-level analyzer below still documents the older low/med/high passive
matrix decision gate. It is not the best evaluator for the continuous sweep
pilot.

```bash
cd /home/zhuokai/hand-teleop/webcam-input/lerobot_teleoperator_so101_webcam
env -u PYTHONPATH /home/zhuokai/hand-teleop/.venv-lerobot/bin/python analyze_ir_grip_experiment.py \
  --root /home/zhuokai/hand-teleop/ir-camera-force/local/datasets/ir_grip_force_viability
```

Actual decision logic in the current analyzer:

- warmed sanity must pass:
  - `hold_mean_area_px >= 25`
  - `hold_max_delta >= 10`
- then either:
  - at least 3 passive objects are strictly monotonic over `low -> med -> high`
    by mean IR contact area
  - or soft-vs-solid passive hold-area effect size is at least `0.8`

Outputs:

- `analysis/summary.json`
- `analysis/ir_area_vs_current.png`

## Software-only verification used for this task

These checks are safe to run without moving hardware beyond each script's
`--help`:

```bash
cd /home/zhuokai/hand-teleop/webcam-input/lerobot_teleoperator_so101_webcam
env -u PYTHONPATH /home/zhuokai/hand-teleop/.venv-lerobot/bin/python -m pytest \
  tests/test_ir_devices.py \
  tests/test_ir_dataset.py \
  tests/test_ir_robot.py \
  tests/test_ir_capture.py \
  tests/test_ir_features.py \
  tests/test_ir_analysis.py \
  -v
```

```bash
cd /home/zhuokai/hand-teleop/webcam-input/lerobot_teleoperator_so101_webcam
env -u PYTHONPATH /home/zhuokai/hand-teleop/.venv-lerobot/bin/python verify_ir_grip_setup.py --help
env -u PYTHONPATH /home/zhuokai/hand-teleop/.venv-lerobot/bin/python characterize_ir_grip_current.py --help
env -u PYTHONPATH /home/zhuokai/hand-teleop/.venv-lerobot/bin/python record_ir_grip_trial.py --help
env -u PYTHONPATH /home/zhuokai/hand-teleop/.venv-lerobot/bin/python extract_ir_grip_features.py --help
env -u PYTHONPATH /home/zhuokai/hand-teleop/.venv-lerobot/bin/python analyze_ir_grip_experiment.py --help
```
