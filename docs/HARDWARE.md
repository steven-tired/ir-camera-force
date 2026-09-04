# Hardware bring-up notes (SO-101 webcam teleop)

Enumerated 2026-06-17.

## Devices
- **Arm (SO-101 follower):** `/dev/ttyACM0` (QinHeng USB-Serial `1a86:55d3`). Only ACM device present.
  - `--robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=so101_follower_1`
- **Hand-tracking camera:** built-in Chicony USB2.0 (uvcvideo). **OpenCV index 0** (`/dev/video0`). Sees the operator.
  - teleoperator: `--teleop.camera_index=0`
- **Workspace camera:** USB webcam model "0825" (uvcvideo). **OpenCV index 2** (`/dev/video2`). Sees the workbench.
  - robot cameras: `index_or_path: 2`, 640x480 @ 30
- **OAK / Intel Movidius MyriadX** (`03e7:2485`): unbooted, NOT a plain /dev/video camera. Unused (RGB-only plan).

## Camera roles
- index 0 = your hand (teleop control input)
- index 2 = the table/workspace (policy observation, recorded into dataset)

## IMPORTANT: normalization
Always pass **`--robot.use_degrees=false`** on every robot command (teleop / record / deploy).
The follower defaults to `use_degrees=True` (body joints in DEGREES), but the so101_webcam
teleoperator outputs normalized **[-100,100]** (RANGE_M100_100). `use_degrees=false` makes the
robot match. Gripper is `0..100` either way. The recorded dataset must use the same mode end-to-end.

For the dedicated IR grip-force scripts in this package, there is no CLI flag to
set this. `characterize_ir_grip_current.py` and `record_ir_grip_trial.py`
already construct `SO101FollowerConfig(... use_degrees=False ...)` internally,
so do not try to add a nonexistent `--robot-use-degrees` flag to those
commands.

## Calibration
Saved to `~/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower_1.json`
(robot class name is `so_follower`; original run wrote `None.json`, copied to the id-matched name).

## Status
- [x] Arm calibrated (`lerobot-calibrate`, id `so101_follower_1`)
- [ ] Teleop tuned (Task 8)
- [ ] Dataset recorded (Task 9)
- [ ] Policy trained / eval success rate: ___

## End-effector IK
- **placo installed** in `.venv-lerobot` via `VIRTUAL_ENV=.venv-lerobot uv pip install -e "./lerobot[placo-dep]"`
  (placo 0.9.15, pin 3.4.0).
  - **Dependency pin required:** `pin==3.4.0` declares `cmeel-urdfdom >= 4` with no upper bound, so the
    resolver picked `cmeel-urdfdom==6.0.0`, whose `liburdfdom_*.so.6.0.0` SONAME doesn't match what
    placo's compiled extension actually links against (`liburdfdom_*.so.4.0`), causing
    `ImportError: liburdfdom_sensor.so.4.0: cannot open shared object file`. Fixed by pinning
    `cmeel-urdfdom==4.0.1`, which in turn requires `cmeel-tinyxml2` providing `libtinyxml2.so.10`
    (the already-installed `cmeel-tinyxml2==11.0.0` only provides `.so.11`) — pinned
    `cmeel-tinyxml2==10.0.0` to match. Verified via `ldd` on
    `cmeel.prefix/lib/liblibplaco.so` that all transitive libs resolve, then `import placo` succeeds.
  - If re-installing from scratch, run, in order:
    ```bash
    VIRTUAL_ENV=.venv-lerobot uv pip install -e "./lerobot[placo-dep]"
    VIRTUAL_ENV=.venv-lerobot uv pip install "cmeel-urdfdom==4.0.1" "cmeel-tinyxml2==10.0.0"
    ```
- **URDF_PATH** (SO-ARM100 clone, not committed to this repo — clone fresh, ~depth 1):
  `$WORKSPACE_ROOT/SO-ARM100/Simulation/SO101/so101_new_calib.urdf`
  ```bash
  cd $WORKSPACE_ROOT
  git clone --depth 1 https://github.com/TheRobotStudio/SO-ARM100.git
  ```
- **target_frame_name = `gripper_frame_link`** — found and worked on the first try, no fallback needed.
- **Kinematics verified** with `lerobot.model.kinematics.RobotKinematics`, motors
  `["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]`:
  - Neutral pose FK: position `[0.391, -0.0, 0.226]` m.
  - IK(FK(q)) round-trip at `q=[10,-20,30,15,-10,5]` deg matched original `q` to ~1e-14 deg.
  - Startup prints benign self-collision warnings for the URDF's neutral pose (cosmetic collision-margin
    quirk in the upstream URDF, not a kinematics error) — `KINEMATICS OK` still prints cleanly.

## IR grip-force viability experiment

- Experiment plan: `docs/superpowers/plans/2026-07-06-ir-grip-force-viability-experiment.md`
- Operator runbook: `../docs/experiments/IR_GRIP_FORCE_EXPERIMENT.md`
- Dataset root: `local/datasets/ir_grip_force_viability`
- Historical device enumeration above listed the workspace camera as
  `/dev/video2`, but do not use `/dev/video2` for this experiment.
- Before the first IR run, list the stable symlinks with:
  ```bash
  ls -l /dev/v4l/by-id/
  BIRD_PATH=/dev/v4l/by-id/<paste-the-workspace-camera-video-index0-symlink>
  test -e "$BIRD_PATH"
  ```
- Reuse that exact `BIRD_PATH` value in `verify_ir_grip_setup.py` and every
  `record_ir_grip_trial.py` command in the runbook.
- FLIR ONE bring-up for this experiment uses `flirone-v4l2` loopbacks on:
  - visible RGB: `/dev/video20`
  - thermal colorized output: `/dev/video21`
- Bring-up commands:
  ```bash
  sudo modprobe v4l2loopback video_nr=20,21 card_label=FLIR_ONE_VISIBLE,FLIR_ONE_THERMAL exclusive_caps=1,1
  cd /tmp/flirone-v4l2
  sudo ./flirone palettes/Iron2.raw
  ffplay -f v4l2 -input_format rgb24 -video_size 160x128 -framerate 30 /dev/video21
  ```
- `/dev/video21` is a colorized `RGB3` `160x128` stream from `flirone-v4l2`, not
  raw radiometric temperature. Analyze it as relative palette-index or
  baseline-delta intensity data, not Celsius.
- Use stable `/dev/v4l/by-id` paths for the bird-view RGB camera.
- The first required pass is `verify_ir_grip_setup.py`, followed by
  `characterize_ir_grip_current.py`.
- `verify_ir_grip_setup.py` now enforces the experiment defaults before
  capture: stable bird path, thermal path `/dev/video21`, and thermal format
  support for `RGB3` at `160x128`. Use `--allow-unstable-bird-path` or
  `--allow-thermal-mismatch` only for deliberate overrides.
- If palette reconstruction is available during feature extraction, use:
  ```bash
  env -u PYTHONPATH $WORKSPACE_ROOT/.venv-lerobot/bin/python extract_ir_grip_features.py \
    --root local/datasets/ir_grip_force_viability \
    --baseline-frames 20 \
    --palette /tmp/flirone-v4l2/palettes/Iron2.raw \
    --invert-palette
  ```
