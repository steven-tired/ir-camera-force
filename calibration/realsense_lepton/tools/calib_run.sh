#!/usr/bin/env bash
# RealSense D435i <-> Lepton calibration run driver.
#
# THIS IS OPERATOR BOILERPLATE ONLY. It reproduces the non-interactive parts of
# REALSENSE_LEPTON_CALIBRATION.md (runbook Sections 1,2,3-hash,4-tail,5-tail,6,7,8)
# VERBATIM so you do not hand-type them. It changes NO calibration algorithm and
# weakens NO gate: terminal-status starts FAIL and only seal_run_pass (end of §8)
# flips it to PASS.
#
# USAGE — open ONE new dedicated terminal and:
#     source /home/zhuokai/hand-teleop/scripts/calib_run.sh
# This shell becomes your capture shell for the WHOLE run. The EXIT trap is armed
# here; if you close it before PASS, the run is sealed FAIL (intended). Retry with
#     CALIB_ATTEMPT=attempt02 source .../calib_run.sh
#
# After sourcing succeeds it stops at the PHYSICAL VISIBILITY GATE. Then you run,
# in order, the interactive captures and the wrapper functions it prints:
#     calib_gate            # §3 tail: record heat source / distance / safety
#     ./lepton ...          # §4 interactive: press c for 30 intrinsic views
#     calib_after_intrinsic # §4 tail: copy+hash the 30 intrinsic PNGs
#     ./depth_saver ...     # §5 interactive: press c for 36 paired views
#     calib_after_pairs     # §5 tail: copy+verify+hash the 36 pairs
#     calib_freeze          # §6: freeze fit(1-24)/heldout(25-36) split
#     calib_fit             # §7: fit intrinsics + extrinsics
#     calib_heldout         # §8: non-fitting gate -> seal PASS on success
#
# It PRINTS the exact interactive commands (with the frozen -mintemp/-maxtemp and
# the schedule) at each stage so you never guess.

# ---- frozen constants (from the runbook; do not edit) -----------------------
BASE=/home/zhuokai/hand-teleop/thermal-project
VERIFIER_WT=/home/zhuokai/hand-teleop/thermal-project-calibration
ROOT=/home/zhuokai/hand-teleop/thermal-project-calibration-runs
BASE_COMMIT=74268ab369904935c5b46fd13a14a0f34814bf4b
VERIFIER_COMMIT=933c8bc20ab4fe7983f81ab9960ef1e205ea06ea
PI_BINARY_SHA256=4fd0fc67e99a268210b2bf3e09a814ce78a871e316695ac8ced5d31dd0d1760a
RS_SERIAL=233522078685
ATTEMPT="${CALIB_ATTEMPT:-attempt01}"

_die() { echo "CALIB ABORT: $*" >&2; return 1; }

# =============================================================================
# Section 1 — start an immutable run (verbatim guards + trap + provenance)
# =============================================================================
_calib_section1() {
  test -z "$(git -C "$VERIFIER_WT" status --porcelain)" || return $(_die "verifier worktree dirty")
  test "$VERIFIER_COMMIT" = "$(git -C "$VERIFIER_WT" rev-parse HEAD)" || return $(_die "verifier not at pinned commit")
  git -C "$BASE" merge-base --is-ancestor "$BASE_COMMIT" "$VERIFIER_COMMIT" || return $(_die "base not ancestor")

  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$ATTEMPT"
  WT="$ROOT/worktrees/$RUN_ID"
  RUN="$ROOT/runs/$RUN_ID"
  test ! -e "$WT" || return $(_die "worktree exists: $WT")
  test ! -e "$RUN" || return $(_die "run exists: $RUN")
  mkdir -p "$ROOT/worktrees" "$ROOT/runs"
  git -C "$BASE" worktree add --detach "$WT" "$VERIFIER_COMMIT" || return $(_die "worktree add failed")
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
    if [ "$rc" -eq 0 ]; then rc=1; fi
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
    "$BASE_COMMIT" "$VERIFIER_COMMIT" "$RS_SERIAL" >"$RUN/provenance/source.txt"
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
  echo "[§1] run created: $RUN"
}

# =============================================================================
# Section 2 — build + test + hash the frozen tools (verbatim)
# =============================================================================
_calib_section2() {
  ( cd "$WT" &&
    cmake -S calibration -B calibration/build -DBUILD_TESTING=ON &&
    cmake --build calibration/build \
      --target camera_calibration verify_calibration extrinsic resolve_extrinsic heldout_verify \
      calibration_contract_test heldout_verifier_test -j2 &&
    ctest --test-dir calibration/build --output-on-failure &&
    cmake -S stream -B stream/build -DBUILD_TESTING=ON &&
    cmake --build stream/build \
      --target lepton depth_saver thermal_frame_assembler_test stream_contract_test -j2 &&
    ctest --test-dir stream/build --output-on-failure
  ) || return $(_die "build/test failed")
  sha256sum \
    "$WT/calibration/build/camera_calibration" \
    "$WT/calibration/build/verify_calibration" \
    "$WT/calibration/build/extrinsic" \
    "$WT/calibration/build/heldout_verify" \
    "$WT/stream/build/depth_saver" \
    >"$RUN/manifests/executables.sha256" || return $(_die "hash failed")
  sha256sum -c "$RUN/manifests/executables.sha256" || return $(_die "executable hash mismatch")
  echo "[§2] frozen tools built + hashed. WARM BOTH CAMERAS 20 MIN BEFORE CAPTURE."
}

# =============================================================================
# Section 3 (automatable head) — Pi binary hash + start sanctioned streamer
# =============================================================================
_calib_section3_head() {
  local actual
  actual="$(ssh anujn@192.168.50.2 \
    "sha256sum /home/anujn/Project/LeptonModule/software/build/raspberrypi_video_network | awk '{print \$1}'")" \
    || return $(_die "cannot ssh Pi for binary hash")
  test "$actual" = "$PI_BINARY_SHA256" || return $(_die "Pi streamer binary hash mismatch: $actual")
  printf '%s  %s\n' "$actual" \
    /home/anujn/Project/LeptonModule/software/build/raspberrypi_video_network \
    >"$RUN/provenance/pi-binary.sha256"
  sha256sum "$RUN/provenance"/* >"$RUN/manifests/provenance.sha256"
  /home/zhuokai/hand-teleop/scripts/run_lepton_stream.sh start || return $(_die "streamer start failed")
  /home/zhuokai/hand-teleop/scripts/run_lepton_stream.sh status
  echo "[§3] Pi binary verified + streamer started."
}

# ---- run §1,§2,§3-head now, then stop at the physical gate -------------------
if _calib_section1 && _calib_section2 && _calib_section3_head; then
  cat <<EOF

============================================================================
 READY. Run these in TWO OTHER terminals, then pass the PHYSICAL gate:
   term2:  ssh anujn@192.168.50.2 '~/Project/LeptonModule/software/build/raspberrypi_video_network -ffc-only'
   term3:  cd /home/zhuokai/hand-teleop && env -u PYTHONPATH .venv-lerobot/bin/python \\
             webcam-input/.worktrees/ir-hand-pressure-so101-teleop/lerobot_teleoperator_so101_webcam/view_ir_camera.py --lepton-udp 8080

 PROCEED ONLY WHEN: same board simultaneously sharp in D435i RGB AND Lepton LWIR,
 all 4x3 inner corners reliably detected, thermal NOT saturated, and the viewer
 shows FFC complete->imminent->complete with since_last_ffc resetting.

 THEN call, in this shell, in order:
   calib_gate              # record heat source / distance / max-safe-temp
   # §4 capture (interactive):
   cd "\$WT/stream/build" && ./lepton -port 8080 -mintemp 27300 -maxtemp 33500   # press c x30
   calib_after_intrinsic
   # §5 capture (interactive, board STILL each press):
   ./depth_saver -port 8080 -mintemp 27300 -maxtemp 33500 2>&1 | tee "\$RUN/logs/depth_saver.log"   # press c x36
   calib_after_pairs
   calib_freeze
   calib_fit
   calib_heldout           # seals PASS iff global_max_error_px<=3.0
============================================================================
 RUN_ID=$RUN_ID
 WT=$WT
 RUN=$RUN
============================================================================
EOF
else
  echo "CALIB SETUP FAILED — close this shell (run sealed FAIL) and fix, then re-source." >&2
fi

# =============================================================================
# §3 tail — record the visibility/safety gate (call after the physical gate)
# =============================================================================
calib_gate() {
  read -r -p 'Heat-source type: ' HEAT_SOURCE_TYPE
  read -r -p 'Actual target distance at visibility gate (m): ' ACTUAL_TARGET_DISTANCE_M
  read -r -p 'Maximum safe target surface temperature (C): ' MAX_SAFE_SURFACE_TEMPERATURE_C
  test -n "$HEAT_SOURCE_TYPE" || return $(_die "empty heat source")
  python3 -c 'import math,sys; v=[float(x) for x in sys.argv[1:]]; assert all(math.isfinite(x) and x>0 for x in v)' \
    "$ACTUAL_TARGET_DISTANCE_M" "$MAX_SAFE_SURFACE_TEMPERATURE_C" || return $(_die "bad numeric gate input")
  printf '%s\n' \
    "heat_source_type=$HEAT_SOURCE_TYPE" \
    "actual_target_distance_m=$ACTUAL_TARGET_DISTANCE_M" \
    "maximum_safe_surface_temperature_c=$MAX_SAFE_SURFACE_TEMPERATURE_C" \
    'corners_visible_in_both_cameras=true' \
    'thermal_saturation=false' \
    >"$RUN/provenance/visibility-and-safety-gate.txt"
  sha256sum "$RUN/provenance"/* >"$RUN/manifests/provenance.sha256"
  echo "[§3] gate recorded. Now: cd \$WT/stream/build && ./lepton -port 8080 -mintemp 27300 -maxtemp 33500  (press c x30)"
}

# =============================================================================
# §4 tail — copy + count + hash the 30 intrinsic PNGs
# =============================================================================
calib_after_intrinsic() {
  cp "$WT/stream/build"/thermal_images/thermal_grayimage_*.png "$RUN/raw/intrinsic/" || return $(_die "no intrinsic pngs")
  test "$(find "$RUN/raw/intrinsic" -maxdepth 1 -name 'thermal_grayimage_*.png' | wc -l)" -eq 30 \
    || return $(_die "intrinsic count != 30")
  (cd "$RUN" && find raw/intrinsic -type f -print0 | sort -z | xargs -0 sha256sum) \
    >"$RUN/manifests/intrinsic.sha256"
  find "$WT/stream/build/thermal_images" -type f -delete
  find "$WT/stream/build/images" -type f -delete 2>/dev/null
  echo "[§4] 30 intrinsic images stored+hashed. Now: ./depth_saver -port 8080 -mintemp 27300 -maxtemp 33500 2>&1 | tee \$RUN/logs/depth_saver.log  (press c x36, board STILL)"
}

# =============================================================================
# §5 tail — verify RS contract line, copy + count + hash the 36 pairs
# =============================================================================
calib_after_pairs() {
  rg -Fx 'RealSense capture contract: serial=233522078685 color=1280x720 RGB8@15 depth=1280x720 Z16@6' \
    "$RUN/logs/depth_saver.log" >"$RUN/provenance/realsense-capture.txt" \
    || return $(_die "RealSense capture contract line missing from log")
  sha256sum "$RUN/provenance"/* >"$RUN/manifests/provenance.sha256"
  cp "$WT/stream/build"/images/color_image_*.png "$RUN/raw/pairs/color/" || return $(_die "no color pairs")
  cp "$WT/stream/build"/thermal_images/thermal_grayimage_*.png "$RUN/raw/pairs/thermal/" || return $(_die "no thermal pairs")
  test "$(find "$RUN/raw/pairs/color" -maxdepth 1 -name 'color_image_*.png' | wc -l)" -eq 36 || return $(_die "color pairs != 36")
  test "$(find "$RUN/raw/pairs/thermal" -maxdepth 1 -name 'thermal_grayimage_*.png' | wc -l)" -eq 36 || return $(_die "thermal pairs != 36")
  (cd "$RUN" && find raw/pairs -type f -print0 | sort -z | xargs -0 sha256sum) >"$RUN/manifests/pairs.sha256"
  echo "[§5] 36 pairs stored+hashed. Now: calib_freeze"
}

# =============================================================================
# §6 — freeze the fit(1-24)/held-out(25-36) split
# =============================================================================
calib_freeze() {
  (cd "$RUN" && sha256sum -c manifests/pairs.sha256) || return $(_die "pairs hash mismatch")
  cp "$RUN/raw/pairs/color"/color_image_{1..24}.png "$RUN/fit/color/" || return $(_die "copy fit color")
  cp "$RUN/raw/pairs/thermal"/thermal_grayimage_{1..24}.png "$RUN/fit/thermal/" || return $(_die "copy fit thermal")
  cp "$RUN/raw/pairs/color"/color_image_{25..36}.png "$RUN/heldout/color/" || return $(_die "copy heldout color")
  cp "$RUN/raw/pairs/thermal"/thermal_grayimage_{25..36}.png "$RUN/heldout/thermal/" || return $(_die "copy heldout thermal")
  (cd "$RUN" && find fit heldout -type f -print0 | sort -z | xargs -0 sha256sum) >"$RUN/manifests/frozen-split.sha256"
  (cd "$RUN" && sha256sum -c manifests/frozen-split.sha256) || return $(_die "frozen-split verify")
  echo "[§6] split frozen (held-out 25-36 sealed). Now: calib_fit"
}

# =============================================================================
# §7 — fit intrinsics + extrinsics (never touches held-out)
# =============================================================================
calib_fit() {
  test "$VERIFIER_COMMIT" = "$(git -C "$WT" rev-parse HEAD)" || return $(_die "verifier moved")
  sha256sum -c "$RUN/manifests/executables.sha256" || return $(_die "exe hash mismatch")
  find "$WT/calibration/thermal_images" -type f -delete 2>/dev/null
  find "$WT/calibration/color_images" -type f -delete 2>/dev/null
  cp "$RUN/raw/intrinsic"/thermal_grayimage_*.png "$WT/calibration/thermal_images/" || return $(_die "copy intrinsic to calib")
  ( cd "$WT/calibration/build" &&
    ./camera_calibration -r 4 -c 5 -n 30 -pat 1 2>&1 | tee "$RUN/logs/camera_calibration.log" ) || return $(_die "camera_calibration failed")
  if rg -q "Pattern not found|Cannot open|must be 160x120" "$RUN/logs/camera_calibration.log"; then
    return $(_die "camera_calibration log shows failure")
  fi
  ( cd "$WT/calibration/build" && ./verify_calibration -n 30 2>&1 | tee "$RUN/logs/verify_calibration.log" ) || return $(_die "verify_calibration failed")
  cp "$WT/calibration/calibration.xml" "$RUN/results/calibration.xml"
  find "$WT/calibration/thermal_images" -type f -delete
  find "$WT/calibration/color_images" -type f -delete 2>/dev/null
  (cd "$RUN" && sha256sum -c manifests/frozen-split.sha256) || return $(_die "frozen-split verify pre-extrinsic")
  cp "$RUN/fit/color"/color_image_*.png "$WT/calibration/color_images/" || return $(_die "copy fit color to calib")
  cp "$RUN/fit/thermal"/thermal_grayimage_*.png "$WT/calibration/thermal_images/" || return $(_die "copy fit thermal to calib")
  ( cd "$WT/calibration/build" && ./extrinsic -r 4 -c 5 -n 24 2>&1 | tee "$RUN/logs/extrinsic.log" ) || return $(_die "extrinsic failed")
  if rg -q "not found|Failed to load|must be 1280x720|must be 160x120" "$RUN/logs/extrinsic.log"; then
    return $(_die "extrinsic log shows failure")
  fi
  cp "$WT/calibration/extrinsic.xml" "$RUN/results/extrinsic.xml"
  echo "[§7] intrinsics + extrinsics fit. Now: calib_heldout"
}

# =============================================================================
# §8 — non-fitting held-out gate; seal PASS iff global_max_error_px<=3.0
# =============================================================================
calib_heldout() {
  test "$VERIFIER_COMMIT" = "$(git -C "$WT" rev-parse HEAD)" || return $(_die "verifier moved")
  sha256sum -c "$RUN/manifests/executables.sha256" || return $(_die "exe hash mismatch")
  (cd "$RUN" && sha256sum -c manifests/frozen-split.sha256) || return $(_die "frozen-split verify")
  ( cd "$WT/calibration/build" && ./heldout_verify \
      --color-dir "$RUN/heldout/color" \
      --thermal-dir "$RUN/heldout/thermal" \
      --intrinsic "$RUN/results/calibration.xml" \
      --extrinsic "$RUN/results/extrinsic.xml" \
      --output "$RUN/results/heldout_projection_report.json" ) || return $(_die "heldout_verify nonzero")
  python3 -c 'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); assert p["schema_version"]=="thermal-heldout-projection/v1"; assert p["status"]=="pass"; assert p["requested_image_count"]==12; assert p["evaluated_image_count"]==12; assert p["point_count"]==144; assert len(p["images"])==12; assert all(len(x["point_errors_px"])==12 for x in p["images"]); assert not p["failures"]; assert p["global_max_error_px"]<=3.0' \
    "$RUN/results/heldout_projection_report.json" || return $(_die "held-out JSON gate FAILED (max>3px or malformed) — leave FAIL, retry new run")
  (cd "$RUN" && find results logs -type f -print0 | sort -z | xargs -0 sha256sum) >"$RUN/manifests/results.sha256"
  (cd "$RUN" && sha256sum -c manifests/results.sha256) || return $(_die "results hash mismatch")
  seal_run_pass
  trap - EXIT
  echo "[§8] PASS SEALED. terminal-status.txt=PASS. results in $RUN/results/"
  echo "     Stop the streamer when done: /home/zhuokai/hand-teleop/scripts/run_lepton_stream.sh stop"
}
