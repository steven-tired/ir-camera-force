# RealSense D435i to Lepton Calibration Runbook

This runbook executes the corrected author workflow without changing its
calibration algorithm. It uses a five-by-four physical checkerboard (four by
three inner corners), 0.03 m square edge, 30 Lepton intrinsic images, 24 paired
fit images, and a wholly separate 12-pair held-out set. PASS requires all 144
held-out corner errors and `global_max_error_px <= 3.0`; RMS is report-only.

The verifier and paired capture are hard-bound to D435i serial `233522078685`.
Do not add pose scoring, refit on held-out images, delete outliers, reorder
corners, tune thresholds, or claim hardware/time synchronization.

## 1. Start an immutable run

Run this from a new shell. Replace `attempt01` with the next unused attempt
number when retrying; never reuse a prior run directory.

```bash
set -euo pipefail
BASE=/home/zhuokai/hand-teleop/thermal-project
VERIFIER_WT=/home/zhuokai/hand-teleop/thermal-project-calibration
ROOT=/home/zhuokai/hand-teleop/thermal-project-calibration-runs
BASE_COMMIT=74268ab369904935c5b46fd13a14a0f34814bf4b
VERIFIER_COMMIT=3d5352c33770db6f3ed36e70d0ba0c489d9d2f4c
test -z "$(git -C "$VERIFIER_WT" status --porcelain)"
test "$VERIFIER_COMMIT" = "$(git -C "$VERIFIER_WT" rev-parse HEAD)"
git -C "$BASE" merge-base --is-ancestor "$BASE_COMMIT" "$VERIFIER_COMMIT"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-attempt01"
WT="$ROOT/worktrees/$RUN_ID"
RUN="$ROOT/runs/$RUN_ID"

test ! -e "$WT"
test ! -e "$RUN"
mkdir -p "$ROOT/worktrees" "$ROOT/runs"
git -C "$BASE" worktree add --detach "$WT" "$VERIFIER_COMMIT"
mkdir -p \
  "$RUN/raw/intrinsic" \
  "$RUN/raw/pairs/color" \
  "$RUN/raw/pairs/thermal" \
  "$RUN/fit/color" "$RUN/fit/thermal" \
  "$RUN/heldout/color" "$RUN/heldout/thermal" \
  "$RUN/results" "$RUN/logs" "$RUN/manifests" "$RUN/provenance"

printf '%s\n' FAIL >"$RUN/manifests/terminal-status.txt"
finalize_run() {
  rc=$?
  trap - EXIT
  if [ "$rc" -eq 0 ]; then
    rc=1
  fi
  printf '%s\n' FAIL >"$RUN/manifests/terminal-status.txt"
  (cd "$RUN" && find . -type f ! -path './manifests/final.sha256' -print0 \
    | sort -z | xargs -0 sha256sum) >"$RUN/manifests/final.sha256"
  sha256sum "$RUN/manifests/final.sha256" >"$ROOT/runs/$RUN_ID.final.sha256"
  exit "$rc"
}
seal_run_pass() {
  final_manifest_tmp="$ROOT/runs/$RUN_ID.final-manifest.tmp"
  final_seal_tmp="$ROOT/runs/$RUN_ID.final-seal.tmp"
  (cd "$RUN" && find . -type f \
    ! -path './manifests/final.sha256' \
    ! -path './manifests/terminal-status.txt' -print0 \
    | sort -z | xargs -0 sha256sum) >"$final_manifest_tmp"
  pass_status_sha="$(printf 'PASS\n' | sha256sum | awk '{print $1}')"
  printf '%s  %s\n' "$pass_status_sha" './manifests/terminal-status.txt' \
    >>"$final_manifest_tmp"
  final_manifest_sha="$(sha256sum "$final_manifest_tmp" | awk '{print $1}')"
  printf '%s  %s\n' "$final_manifest_sha" "$RUN/manifests/final.sha256" \
    >"$final_seal_tmp"
  mv "$final_manifest_tmp" "$RUN/manifests/final.sha256"
  mv "$final_seal_tmp" "$ROOT/runs/$RUN_ID.final.sha256"
  printf '%s\n' PASS >"$RUN/manifests/terminal-status.txt"
}
trap finalize_run EXIT

printf 'base_commit=%s\nverifier_commit=%s\nrealsense_serial=%s\n' \
  "$BASE_COMMIT" "$VERIFIER_COMMIT" 233522078685 >"$RUN/provenance/source.txt"
cp "$WT/calibration/calibration.xml" "$RUN/provenance/sample-calibration.xml"
cp "$WT/calibration/extrinsic.xml" "$RUN/provenance/sample-extrinsic.xml"
printf '%s\n' \
  '1-10 near 0.25-0.35m: center/roll-,center/roll+,left/yaw-,left/yaw+,right/yaw-,right/yaw+,top/pitch-,top/pitch+,bottom/pitch-,bottom/pitch+' \
  '11-20 middle 0.40-0.55m: same ordered ten poses' \
  '21-30 far 0.60-0.72m: same ordered ten poses' \
  >"$RUN/provenance/intrinsic-slots.txt"
printf '%s\n' \
  '1-8 fit near: center/fronto-parallel,left/yaw-,right/yaw+,top/pitch-,bottom/pitch+,upper-left/roll-,upper-right/roll+,lower-center/yaw+/pitch-' \
  '9-16 fit middle: same ordered eight poses' \
  '17-24 fit far: same ordered eight poses' \
  '25-28 held-out near: center/roll+,left/pitch+,right/pitch-,upper-center/yaw-' \
  '29-32 held-out middle: same ordered four poses' \
  '33-36 held-out far: same ordered four poses' \
  >"$RUN/provenance/paired-slots.txt"
sha256sum "$RUN/provenance"/* >"$RUN/manifests/provenance.sha256"
```

The EXIT trap must remain active until the last acceptance check. Any early
build, capture, calibration, hash, or validation failure leaves a terminal
`FAIL` marker and a complete final hash manifest.

## 2. Build and hash the frozen tools

```bash
cd "$WT"
cmake -S calibration -B calibration/build -DBUILD_TESTING=ON
cmake --build calibration/build \
  --target camera_calibration verify_calibration extrinsic heldout_verify \
  calibration_contract_test heldout_verifier_test -j2
ctest --test-dir calibration/build --output-on-failure
cmake -S stream -B stream/build -DBUILD_TESTING=ON
cmake --build stream/build \
  --target lepton depth_saver thermal_frame_assembler_test stream_contract_test -j2
ctest --test-dir stream/build --output-on-failure
sha256sum \
  "$WT/calibration/build/camera_calibration" \
  "$WT/calibration/build/verify_calibration" \
  "$WT/calibration/build/extrinsic" \
  "$WT/calibration/build/heldout_verify" \
  "$WT/stream/build/depth_saver" \
  >"$RUN/manifests/executables.sha256"
sha256sum -c "$RUN/manifests/executables.sha256"
```

Warm both cameras continuously for 20 minutes before capture. This is a
project repeatability choice, not a claim from the author repository.

## 3. Start the sanctioned Pi stream and verify visibility

Only use the C++ path in `/home/zhuokai/hand-teleop/docs/LEPTON_PI_HOWTO.md`.
Python `spidev` probes are prohibited.

```bash
PI_BINARY_SHA256=4fd0fc67e99a268210b2bf3e09a814ce78a871e316695ac8ced5d31dd0d1760a
ACTUAL_PI_SHA256="$(ssh anujn@192.168.50.2 \
  "sha256sum /home/anujn/Project/LeptonModule/software/build/raspberrypi_video_network | awk '{print \$1}'")"
test "$ACTUAL_PI_SHA256" = "$PI_BINARY_SHA256"
printf '%s  %s\n' "$ACTUAL_PI_SHA256" \
  /home/anujn/Project/LeptonModule/software/build/raspberrypi_video_network \
  >"$RUN/provenance/pi-binary.sha256"
sha256sum "$RUN/provenance"/* >"$RUN/manifests/provenance.sha256"

/home/zhuokai/hand-teleop/scripts/run_lepton_stream.sh start
/home/zhuokai/hand-teleop/scripts/run_lepton_stream.sh status
```

The sanctioned helper leaves the official C++ streamer running on the Pi and
returns control to this original shell, so `$WT`, `$RUN`, and the EXIT trap
remain active here. Keep that managed process alive. From a separate laptop
terminal:

```bash
ssh anujn@192.168.50.2 \
  '~/Project/LeptonModule/software/build/raspberrypi_video_network -ffc-only'
```

In a third laptop terminal, keep the viewer open through the FFC transition:

```bash
cd /home/zhuokai/hand-teleop
env -u PYTHONPATH .venv-lerobot/bin/python \
  webcam-input/.worktrees/ir-hand-pressure-so101-teleop/lerobot_teleoperator_so101_webcam/view_ir_camera.py \
  --lepton-udp 8080
```

Proceed only when the viewer shows complete TLinear frames, telemetry changes
`complete -> imminent/in_progress -> complete`, and `since_last_ffc` resets.
The transport is the footer-telemetry `4 x 10004` byte format.

A 500 W lamp is not required. STOP unless the same physical board is visible
simultaneously in both cameras, all four-by-three inner corners are detected
reliably, and the thermal image is not saturated.

After that visual gate passes, return to the original run shell and record the
actual setup before capturing any calibration image:

```bash
read -r -p 'Heat-source type: ' HEAT_SOURCE_TYPE
read -r -p 'Actual target distance at visibility gate (m): ' ACTUAL_TARGET_DISTANCE_M
read -r -p 'Maximum safe target surface temperature (C): ' MAX_SAFE_SURFACE_TEMPERATURE_C
test -n "$HEAT_SOURCE_TYPE"
python3 -c 'import math,sys; values=[float(x) for x in sys.argv[1:]]; assert all(math.isfinite(x) and x > 0 for x in values)' \
  "$ACTUAL_TARGET_DISTANCE_M" "$MAX_SAFE_SURFACE_TEMPERATURE_C"
printf '%s\n' \
  "heat_source_type=$HEAT_SOURCE_TYPE" \
  "actual_target_distance_m=$ACTUAL_TARGET_DISTANCE_M" \
  "maximum_safe_surface_temperature_c=$MAX_SAFE_SURFACE_TEMPERATURE_C" \
  'corners_visible_in_both_cameras=true' \
  'thermal_saturation=false' \
  >"$RUN/provenance/visibility-and-safety-gate.txt"
sha256sum "$RUN/provenance"/* >"$RUN/manifests/provenance.sha256"
```

## 4. Capture 30 Lepton intrinsic images

Each slot must be a successful complete-corner detection. A failure repeats the
same slot and never advances the index.

```text
slots  1-10  near 0.25-0.35 m    center/roll-, center/roll+, left/yaw-, left/yaw+,
                     right/yaw-, right/yaw+, top/pitch-, top/pitch+,
                     bottom/pitch-, bottom/pitch+
slots 11-20  middle 0.40-0.55 m  same ordered ten poses
slots 21-30  far 0.60-0.72 m     same ordered ten poses
```

```bash
cd "$WT/stream/build"
./lepton -port 8080 -mintemp 27300 -maxtemp 33500
# Press c for exactly 30 successful scheduled views, then exit.

cp thermal_images/thermal_grayimage_*.png "$RUN/raw/intrinsic/"
test "$(find "$RUN/raw/intrinsic" -maxdepth 1 -name 'thermal_grayimage_*.png' | wc -l)" -eq 30
(cd "$RUN" && find raw/intrinsic -type f -print0 | sort -z | xargs -0 sha256sum) \
  >"$RUN/manifests/intrinsic.sha256"

find "$WT/stream/build/thermal_images" -type f -delete
find "$WT/stream/build/images" -type f -delete
```

## 5. Capture 36 paired images

Hold the board stationary during every paired save. This is a pairing
procedure, not a hardware or timestamp synchronization claim.

```text
1-8    fit near       0.25-0.35 m  center/fronto-parallel, left/yaw-,
                                      right/yaw+, top/pitch-, bottom/pitch+,
                                      upper-left/roll-, upper-right/roll+,
                                      lower-center/yaw+/pitch-
9-16   fit middle     0.40-0.55 m  same ordered eight poses
17-24  fit far        0.60-0.72 m  same ordered eight poses
25-28  held-out near  0.25-0.35 m  center/roll+, left/pitch+,
                                      right/pitch-, upper-center/yaw-
29-32  held-out mid   0.40-0.55 m  same ordered four poses
33-36  held-out far   0.60-0.72 m  same ordered four poses
```

```bash
./depth_saver -port 8080 -mintemp 27300 -maxtemp 33500 \
  2>&1 | tee "$RUN/logs/depth_saver.log"
# Hold still and press c for exactly 36 successful scheduled pairs, then exit.

rg -Fx 'RealSense capture contract: serial=233522078685 color=1280x720 RGB8@15 depth=1280x720 Z16@6' \
  "$RUN/logs/depth_saver.log" >"$RUN/provenance/realsense-capture.txt"
sha256sum "$RUN/provenance"/* >"$RUN/manifests/provenance.sha256"

cp images/color_image_*.png "$RUN/raw/pairs/color/"
cp thermal_images/thermal_grayimage_*.png "$RUN/raw/pairs/thermal/"
test "$(find "$RUN/raw/pairs/color" -maxdepth 1 -name 'color_image_*.png' | wc -l)" -eq 36
test "$(find "$RUN/raw/pairs/thermal" -maxdepth 1 -name 'thermal_grayimage_*.png' | wc -l)" -eq 36
(cd "$RUN" && find raw/pairs -type f -print0 | sort -z | xargs -0 sha256sum) \
  >"$RUN/manifests/pairs.sha256"
```

Count only pairs with all 12 corners detected in both images.

## 6. Freeze the split before fitting

Held-out indices remain 25 through 36. Never rename or expose them to fitting.

```bash
(cd "$RUN" && sha256sum -c manifests/pairs.sha256)
cp "$RUN/raw/pairs/color"/color_image_{1..24}.png "$RUN/fit/color/"
cp "$RUN/raw/pairs/thermal"/thermal_grayimage_{1..24}.png "$RUN/fit/thermal/"
cp "$RUN/raw/pairs/color"/color_image_{25..36}.png "$RUN/heldout/color/"
cp "$RUN/raw/pairs/thermal"/thermal_grayimage_{25..36}.png "$RUN/heldout/thermal/"
(cd "$RUN" && find fit heldout -type f -print0 | sort -z | xargs -0 sha256sum) \
  >"$RUN/manifests/frozen-split.sha256"
(cd "$RUN" && sha256sum -c manifests/frozen-split.sha256)
```

## 7. Fit intrinsics and extrinsics

```bash
test "$VERIFIER_COMMIT" = "$(git -C "$WT" rev-parse HEAD)"
sha256sum -c "$RUN/manifests/executables.sha256"
find "$WT/calibration/thermal_images" -type f -delete
find "$WT/calibration/color_images" -type f -delete
cp "$RUN/raw/intrinsic"/thermal_grayimage_*.png "$WT/calibration/thermal_images/"

cd "$WT/calibration/build"
./camera_calibration -r 4 -c 5 -n 30 -pat 1 \
  2>&1 | tee "$RUN/logs/camera_calibration.log"
if rg -q "Pattern not found|Cannot open|must be 160x120" \
  "$RUN/logs/camera_calibration.log"; then
  exit 1
fi
./verify_calibration -n 30 2>&1 | tee "$RUN/logs/verify_calibration.log"
cp "$WT/calibration/calibration.xml" "$RUN/results/calibration.xml"

find "$WT/calibration/thermal_images" -type f -delete
find "$WT/calibration/color_images" -type f -delete
(cd "$RUN" && sha256sum -c manifests/frozen-split.sha256)
cp "$RUN/fit/color"/color_image_*.png "$WT/calibration/color_images/"
cp "$RUN/fit/thermal"/thermal_grayimage_*.png "$WT/calibration/thermal_images/"

./extrinsic -r 4 -c 5 -n 24 2>&1 | tee "$RUN/logs/extrinsic.log"
if rg -q "not found|Failed to load|must be 1280x720|must be 160x120" \
  "$RUN/logs/extrinsic.log"; then
  exit 1
fi
cp "$WT/calibration/extrinsic.xml" "$RUN/results/extrinsic.xml"
```

## 8. Run the non-fitting held-out gate

Recheck source, executable, and frozen-split hashes immediately before the
held-out verifier.

```bash
test "$VERIFIER_COMMIT" = "$(git -C "$WT" rev-parse HEAD)"
sha256sum -c "$RUN/manifests/executables.sha256"
(cd "$RUN" && sha256sum -c manifests/frozen-split.sha256)
./heldout_verify \
  --color-dir "$RUN/heldout/color" \
  --thermal-dir "$RUN/heldout/thermal" \
  --intrinsic "$RUN/results/calibration.xml" \
  --extrinsic "$RUN/results/extrinsic.xml" \
  --output "$RUN/results/heldout_projection_report.json"

python3 -c 'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); assert p["schema_version"]=="thermal-heldout-projection/v1"; assert p["status"]=="pass"; assert p["requested_image_count"]==12; assert p["evaluated_image_count"]==12; assert p["point_count"]==144; assert len(p["images"])==12; assert all(len(x["point_errors_px"])==12 for x in p["images"]); assert not p["failures"]; assert p["global_max_error_px"]<=3.0' \
  "$RUN/results/heldout_projection_report.json"

(cd "$RUN" && find results logs -type f -print0 | sort -z | xargs -0 sha256sum) \
  >"$RUN/manifests/results.sha256"
(cd "$RUN" && sha256sum -c manifests/results.sha256)
seal_run_pass
trap - EXIT
exit 0
```

PASS exists only when `heldout_verify` exits 0, the strict JSON check shows
12 evaluated images, 144 point errors, zero failures, and global maximum at or
below 3.0 px, and both final seals are installed before the terminal marker is
changed from `FAIL` to `PASS`. Never substitute RMS for the maximum-error gate.

## 9. Failure and retry boundary

On failure, allow the EXIT trap to hash and retain the entire run with terminal
status `FAIL`. Record the specific correction. A retry uses a new run ID, new
fit images, and a wholly new held-out set; never delete a failed attempt or
reuse a held-out set once viewed.

An accepted result supports only the actually captured overlap and distance
envelope. The runtime 0.20--0.90 m software guard does not prove geometric
calibration over that full interval.
