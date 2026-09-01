# Lepton IR hard-pinch → gripper overdrive — Agent Handoff

_Last updated **2026-07-30** — see the 2026-07-30 single-finger hold-check
section for the newest software state. The frozen 2026-07-24 calibration feeds a
robot-free hand-shadow MVP with optional RGB preview and strict JSONL output.
Stage 0, 1A, 1B, and 1C have bounded acceptance records; corrected Stage 1D
current-device diagnostics have now frozen a bounded trial operating envelope,
but have not established pressure inference or actuator-grade evidence. Stage
1E acquisition and analysis software is implemented and was exercised through
attempt 05; that valid run rejected the frozen tip-to-tip signal hypothesis.
An independent continuous single-index-finger Null/Press feasibility path was
run once, but its formal D435-to-thermal ROI chain failed on every frame; only
a clearly labeled post-hoc thermal-only descriptive salvage is available.
Stage 1F remains a deferred plan only and has not been implemented or run.
Nothing in this handoff authorizes teleop, controller input, pressure inference,
gripper motion, or thermal actuation._

**Read first:** current hand-shadow design
`webcam-input/.worktrees/ir-hand-pressure-so101-teleop/docs/superpowers/specs/2026-07-27-stage1d-hand-associated-robot-free-shadow-design.md`;
Stage 1C evidence
`webcam-input/.worktrees/ir-hand-pressure-so101-teleop/docs/LEPTON_STAGE1C_LIVE_SHADOW_ACCEPTANCE.md`;
operating guide `docs/LEPTON_PI_HOWTO.md`. The older plans and dated sections
below remain background and negative-evidence history.
All code lives on worktree branch **`ir-hand-pressure-so101-teleop`** at
`webcam-input/.worktrees/ir-hand-pressure-so101-teleop/` (do NOT merge the two IR
branches — see memory `ir-two-branches`).

---

## §CURRENT. 2026-07-28 Stage 0 → hand-shadow checkpoint

### Accepted dependency chain

- **Stage 0 is complete only for provenance and its six-item runtime evidence
  contract.** The frozen production extrinsic remains
  `thermal-project-calibration-runs/worktrees/20260724T210232Z-attempt01/calibration/FINAL_flir_brown/extrinsic_refined.xml`,
  SHA-256
  `2ca1ed48450dea16a5778cb5645dd4852d544490e4f47330dd938f743bc6f434`.
  The runtime contract is
  `scratch_lepton/stage0b_runtime_datums.json`, SHA-256
  `22d41109dcaefb29ad770fb5715c35dfd6c13c68195fbcb55e3b9d6fb4ef756b`.
- **Stage 1A** accepts the offline single-point scalar projector only.
- **Stage 1B** accepts the offline sparse wrapper only.
- **Stage 1C** accepts a robot-free live shadow evidence run only. Its final
  100-attempt run produced 60 software-gate-accepted rows and 40 explicit
  blocked rows. This is not hand association, physical validation, teleop, or
  control authorization.
- The abandoned four-blocker Stage 1D design at historical commit `ac0725a`
  was replaced by the smaller hand-shadow MVP at `421e9bb`. Do not resume the
  reverse lookup, dense overlap-mask, raster-radius, or broad dynamic-skew
  program as prerequisites for the first hand trial.

### Stage 1D software checkpoint

Commit `0a24c5f` adds the robot-free hand-shadow runner, its tests, raw SDK-frame
compatibility, the implementation plan, and the corrected design. The runner
uses MediaPipe only for physical-right-hand thumb/index fingertips and writes
strict JSONL; it has no controller or actuation path.

The implemented architecture is deliberately narrow:

```text
D435i raw color/depth
  -> WebcamSource MediaPipe thumb/index pixels
  -> RealSenseRawProjectorCamera color-to-raw-depth SDK association
  -> frozen scalar/sparse color-to-Lepton projection
LeptonUDPSource current thermal frame
  -> exact thermal winner or explicit blocked row
  -> JSONL and optional RGB preview
```

`live_lepton_hand_shadow.py` owns fresh-frame pairing, timing/FFC/depth/parity
gates, projection results, and evidence output. It reuses the Stage 1A/1B
projectors and frozen Stage 0 geometry. It does not use aligned depth,
`pointcloud().map_to(color)`, reverse thermal lookup, fill holes, historical
frames, or implicit collision/occlusion repair.

The color-to-raw-depth audit now follows librealsense v2.58.3 `src/rs.cpp`
semantics: the SDK reads Z16 from the non-negative truncated/floored source
cell, while deprojecting and scoring the fractional SDK coordinate. The same
floored-cell depth is used for the fractional round trip. Schema version 2
records `sdk_depth_uv`, the floored `depth_pixel`, the superseded half-up cell,
`sdk_reprojected_color_uv`, Euclidean `sdk_match_error_px`, and the diagnostic
`source_cell_reprojected_color_uv`. The locked-profile shadow gate accepts
fractional error `<= 0.75 px`; the floored-cell reprojection is not a gate.
Original, 25% inward, and 50% inward samples remain separate diagnostics.

Regression-first verification produced `101 passed` across the five affected
suites. Module compilation, the scoped forbidden-dependency scans, staged
`git diff --check`, and commit-scope review passed. This approves the software
contract only.

### 2026-07-28 schema-v2 close/middle/far evidence

After one approved C++ streamer start and manual FFC, three bounded preview
runs each collected exactly 120 fresh-depth attempts:

| Ruler label | Frame events | Reused depth | Association-eligible | Physical right hand | Original pair OK | 25% inward pair OK | Formal accepted |
|---|---:|---:|---:|---:|---:|---:|---:|
| close, ~0.33 m | 161 | 41 | 104 | 104 | 94 | 99 | 94 |
| middle, ~0.45 m | 159 | 39 | 93 | 86 | 52 | 77 | 52 |
| far, ~0.60 m | 166 | 46 | 70 | 33 | 1 | 4 | 1 |

The observed successful original-tip depth medians do not equal the ruler
labels and are reported separately:

| Ruler label | Thumb median | Index median |
|---|---:|---:|
| close, ~0.33 m | 0.323 m | 0.291 m |
| middle, ~0.45 m | 0.564 m | 0.549 m |
| far, ~0.60 m | 0.764 m | 0.779 m |

The immutable evidence files and hashes are:

- `scratch_lepton/stage1d_hand_shadow_sdk_parity_close01.jsonl`:
  `262bd4c65fdeb37d56601f8f0ae94fe61b095cd97bd1fbcc1d499c2ae777d050`;
- `scratch_lepton/stage1d_hand_shadow_sdk_parity_middle01.jsonl`:
  `591e29b0b9f9e91c807c2ddd2e0e8bc7f13ba957e7043bb9a5dc79503d00884b`;
- `scratch_lepton/stage1d_hand_shadow_sdk_parity_far01.jsonl`:
  `b70dbbcfa0357142c928e241425d314f0a00dfdc3f8ec9432e96201b1ce8560c`.

Attempt blocker counts may overlap when one row has multiple reasons. Close
recorded host skew 16, fractional SDK error 10, and stale D435 completion 1.
Middle recorded fractional SDK error 35, host skew 27, missing physical right
hand 7, depth out of range 3, and stale D435 completion 2. Far recorded
`lepton_ffc_desired` 39, missing physical right hand 37, host skew 18, SDK no
match 17, depth out of range 17, and fractional SDK error 5. There were no
sparse rejection or collision blockers. In accepted rows, the floored-cell
diagnostic exceeded 0.5 px for 165/188 close tips, 86/104 middle tips, and both
far tips, directly confirming why it must not be the SDK parity gate.

**Outcome B — a real association/operating-band blocker remains.** Fractional
parity correction recovered a nontrivial close result, but middle degraded and
far produced only 1/120 accepted attempts. The 25% inward diagnostic was better
at middle/far but is not authorized as a replacement sampling rule. Stage 1D
therefore remains **not accepted as robust across distance**. This is not
physical validation, synchronization evidence, pressure inference, teleop
integration, or control authorization.

### 2026-07-28 frozen Stage 1D hand-shadow envelope

Five later bounded 120-attempt preview runs used the unchanged original-tip
association and projection gates:

| Ruler position | Accepted | Accepted rate | Accepted-tip median depth |
|---|---:|---:|---:|
| 0.30 m | 59/120 | 49.2% | 0.294 m |
| 0.40 m | 28/120 | 23.3% | 0.379 m |
| 0.50 m | 79/120 | 65.8% | 0.481 m |
| 0.60 m | 82/120 | 68.3% | 0.642 m |
| 0.70 m | 44/120 | 36.7% | 0.727 m |

The immutable JSONLs and SHA-256 values are:

- `scratch_lepton/stage1d_envelope_30cm_01.jsonl`:
  `998deba8c9a61acf9ae713b993a36a4230950215060cf5c96fb97927772a33b8`;
- `scratch_lepton/stage1d_envelope_40cm_01.jsonl`:
  `a1992ba94a223d0313395761db120e6b15efea6edd041c8cc86040eb686d6d49`;
- `scratch_lepton/stage1d_envelope_50cm_01.jsonl`:
  `e8ed4698e8c5c0fd67eb0770fcff8cce4ffab72b640fb7529f276f99856cdcde`;
- `scratch_lepton/stage1d_envelope_60cm_01.jsonl`:
  `9cfc66c0c2f54a686f7fd4e038fee6d38af5b3ba9449d3641310abf6f9f94975`;
- `scratch_lepton/stage1d_envelope_70cm_01.jsonl`:
  `c5ab29d22b58fff3a76ba2f3336d333794eb5c5585d38226395fe21ad90a3328`.

This freezes `0.30-0.60 m` as the nominal Stage 1D trial envelope,
`0.40-0.60 m` as the preferred band, and excludes `0.70 m` from the next
trial. It proves only bounded current-device fingertip localization and
thermal-count capture; it does not turn blocked rows into accepted data or
establish contact/press separability.

### Stage 1E software implementation checkpoint

Stage 1E reuses the Stage 1D D435i-to-Lepton hand-shadow path rather than
creating a second projector. The implemented runtime and frozen analyzer are:

- `lerobot_teleoperator_so101_webcam/live_lepton_hand_shadow.py`;
- `lerobot_teleoperator_so101_webcam/analyze_lepton_pinch_signal.py`;
- `lerobot_teleoperator_so101_webcam/tests/test_live_lepton_hand_shadow.py`;
- `lerobot_teleoperator_so101_webcam/tests/test_analyze_lepton_pinch_signal.py`.

Commits `c8d4b71` and `0d49a52` added the robot-free tip-pinch acquisition
mode and offline analyzer. Later bounded fixes were:

- `ccba20e`: make `--manual-ffc` prepare/recover the approved C++ Lepton
  streamer rather than launch a competing `-ffc-only` process;
- `8cb9013`: replace the fixed-duration three-level prompt with operator-paced
  `JUST TOUCH -> PRESS HARD -> RETURN TO JUST TOUCH`;
- `59b5cc1`: require five fresh software-gate-accepted samples per phase, add a
  10 s phase timeout, and record per-tip exact pixel, 3x3 neighbourhood,
  thermal frame median, projected positions, pinch geometry, and movement;
- `a0c5f78`: freeze schema 4 and retain pinch-center motion as a diagnostic
  instead of rejecting otherwise valid hard-press samples.

The acquisition path requires preview plus manual FFC, starts only after a
fresh accepted frame, advances on operator SPACE, writes every accepted and
blocked attempt to JSONL, invalidates undersampled groups explicitly, and
stops after two invalid groups. It remains robot-free and has no controller,
teleop, recorder, gripper, or pressure-apply dependency.

The analyzer verifies the frozen Stage 0 and extrinsic hashes, exact schema-4
protocol metadata, group/sample completeness, and fail-closed acquisition
status before interpreting signal. For each fingertip it retains exact-pixel
and 3x3 values, subtracts the full-frame median, and computes the within-group
`PRESS - (TOUCH before + TOUCH after) / 2` effect. Its frozen decision also
checks leave-one-group-out IR EBA, gain over 2D geometry, per-fingertip
direction consistency, exact-versus-3x3 sign agreement, and return recovery.
It emits only `BLOCKED_ACQUISITION`, `STOP_BEFORE_STAGE1F`, or
`PROCEED_TO_STAGE1F_SHADOW`; it never emits a controller command.

The focused verification recorded at commit `a0c5f78` was 66 passing tests,
plus successful `py_compile` and `git diff --check`. This is software evidence,
not evidence that the tip-to-tip thermal signal works; attempt 05 supplied the
valid negative physical result below.

### Stage 1F software status — deferred plan only

`docs/superpowers/plans/2026-07-28-stage1f-teleop-coexistence-shadow.md` is
guidance for a possible bounded coexistence trial, not implemented Stage 1F
software. There is no Stage 1F-specific runner, launcher, analyzer, test suite,
or `stage1f_teleop_coexistence_01*` evidence on disk.

The plan deliberately proposed no thermal integration into teleop. It would
run the existing OAK-based `teleop_viz_ee.py` as the sole robot-command owner
and the existing D435i/Lepton `live_lepton_hand_shadow.py` in a separate
process with JSONL-only output. It forbids changing `teleop_viz_ee.py`,
`ee_controller.py`, or `record_so101_ee.py`, and clears all IR pressure/sidecar
flags and environment variables.

That plan currently expects a predecessor
`PROCEED_TO_STAGE1F_SHADOW` result, while attempt 05 produced
`STOP_BEFORE_STAGE1F`. Therefore it must not be executed unchanged. The
post-attempt-05 external suggestion that diagnostic-only coexistence could
still be useful is review advice, not implementation or authorization; any
such trial requires a revised prerequisite and explicit user approval first.

### 2026-07-28 Stage 1E acquisition attempt 01 — blocked before capture

The first Stage 1E command stopped before opening a JSONL because the old
automatic FFC path launched `raspberrypi_video_network -ffc-only` alongside
the running streamer. It returned `Lepton configuration read-back mismatch`;
the preserved terminal log is
`scratch_lepton/stage1e_tip_pinch_signal_01.log`, SHA-256
`d24dcb7f09094d13ef743071cd55413ba9c758778c54db4c1a0a9d10e5334502`.
This is acquisition evidence only and contains no contact/press trial.

Root-cause isolation showed that standalone `-ffc-only` could not recover the
current mismatch, while the approved `run_lepton_stream.sh start` path
reconfigured the same C++ streamer, completed manual FFC, and restored
advancing UDP frame counters. Commit `ccba20e` makes `--manual-ffc` use that
single-process recovery path. Any retry must preserve attempt 01 and use a new
numbered filename.

### 2026-07-28 Stage 1E acquisition attempt 02 — usability blocked

Attempt 02 verified that the corrected automatic streamer preparation and
manual FFC path reached live capture, but the operator stopped the incomplete
run because its forced phase gaps were too short, its trials were too long,
and `light_contact` was an ambiguous name for the intended just-touch state.
It is a protocol-usability failure, not signal evidence, and must not be
analyzed or used for a Stage 1E verdict.

The preserved partial files are:

- `scratch_lepton/stage1e_tip_pinch_signal_02.jsonl`, SHA-256
  `7e2cef11450cd38a1aa35db7650b5c8a6b6a4759a220f505bb7e8cbc2261aa24`;
- `scratch_lepton/stage1e_tip_pinch_signal_02.log`, SHA-256
  `e5efc23cf64d59a77793d2d8b48288a0253d0ac8d839cf98c22391aaa048841e`.

The operator-approved replacement has only two physical states, `JUST TOUCH`
and `PRESS HARD`: six self-paced paired groups, SPACE to confirm readiness,
1.0 second recorded for just-touch, hard-press, and return-to-just-touch, then
an untimed separated rest. There is no intermediate `light press` state.
Analysis is paired within group and requires at least five valid groups.

### 2026-07-28 Stage 1E acquisition attempt 03 — completed, acquisition blocked

Commit `8cb9013` implemented the operator-approved two-state, self-paced
protocol. Attempt 03 completed all six groups after the approved C++ streamer
restart/manual-FFC path. The runner reported 605 fresh-depth attempts, 428
software-gate-accepted rows, 177 blocked rows, and
`pinch_signal_protocol_completed=true`.

Immutable evidence:

- `scratch_lepton/stage1e_tip_pinch_signal_03.jsonl`, SHA-256
  `59ce2b52e4b0c6057f50cea9a6f31e7294b57600459945bdaa496696c09fabfb`;
- `scratch_lepton/stage1e_tip_pinch_signal_03.log`, SHA-256
  `a280bf9047dae40a930a28b8d4c0fc4c0b9e6a5d9949e3cdb494010e332fa85c`;
- `scratch_lepton/stage1e_tip_pinch_signal_03_summary.json`, SHA-256
  `0babfa14a2d82a71fb3294daf47f726da0a9d738ecc7f3033089b2240e570ffc`.

The frozen analyzer returned `BLOCKED_ACQUISITION`, not a negative signal
verdict. Only four of six groups met the predeclared minimum of three accepted
rows per recorded phase: group 2 had zero accepted `PRESS HARD` rows and group
4 had only two accepted initial `JUST TOUCH` rows. The partial descriptive
values (`paired IR EBA=0.5833`, median press delta `-17.5` counts) must not be
used to claim separability because the acquisition gate failed. Stage 1E
therefore remains open and Stage 1F is not authorized.

### 2026-07-28 Stage 1E acquisition attempt 04 — position gate invalid

Commit `59b5cc1` replaced time-based phase success with five fresh accepted
samples per phase, a 10 s failure bound, and a predeclared 1.0 native-thermal-
pixel pinch-center gate. The approved Pi C++ streamer/manual-FFC path recovered
successfully before capture. The run then stopped fail-closed after two
consecutive `PRESS HARD` phase timeouts:

- group 1: `JUST TOUCH=5/5`, `PRESS HARD=0/5`; 48 otherwise
  `software_gate_accepted` hard rows moved 4.80--7.52 thermal px;
- group 2: `JUST TOUCH=5/5`, `PRESS HARD=0/5`; 34 otherwise
  `software_gate_accepted` hard rows moved 2.60--5.57 thermal px.

The fixed 1 px whole-pinch-center constraint is therefore incompatible with
the observed physical press gesture in this setup. This is a protocol/config
negative result, not evidence that MediaPipe, D435i association, Lepton input,
or the thermal signal itself failed. No group reached the return phase, so the
frozen analyzer correctly returned `BLOCKED_ACQUISITION`; attempt 04 contains
no pressure-separability result and does not authorize Stage 1F.

Immutable evidence:

- `scratch_lepton/stage1e_tip_pinch_signal_04.jsonl`, SHA-256
  `b185d55184b207f34fb04d56a952b9e9951bc9e1be326e11bf4bc062b4a2db67`;
- `scratch_lepton/stage1e_tip_pinch_signal_04.log`, SHA-256
  `019ba427f26ec880fb9b51a0a348641186b20a944dc4a2e4ea1511fed47fd32b`;
- `scratch_lepton/stage1e_tip_pinch_signal_04_summary.json`, SHA-256
  `c08577665ba1e107c846dd716ace5fc7a6e73d096ccf9535e984e4679cbfd5f5`.

### 2026-07-28 Stage 1E attempt 05 amendment — software ready, not run

Commit `a0c5f78` freezes schema 4 for a separately numbered retry. It keeps the
five-fresh-sample quota and 10 s failure bound, but changes whole-pinch-center
movement from an acquisition gate to `pinch_center_policy=diagnostic_only`.
Every otherwise accepted hard/return row now counts while still recording its
center displacement and all existing per-tip exact-pixel, 3x3, frame-median,
thermal/RGB/raw-depth position, timing, and FFC diagnostics.

The analyzer rejects attempt 04/schema 3 for an attempt 05 verdict and retains
the per-tip centered A-B-A, exact-versus-3x3, geometry, direction, and recovery
checks. Focused verification is 66 passed plus `py_compile` and
`git diff --check`.

### 2026-07-28 Stage 1E acquisition attempt 05 — valid data, signal gate failed

The explicitly authorized attempt 05 completed all six groups with no timeout:
90/90 recording rows satisfied the frozen five-sample phase quotas. The runner
reported 529 fresh-depth attempts, 324 software-gate-accepted rows, 205 blocked
rows, and `pinch_signal_protocol_completed=true`.

The frozen schema-4 analyzer returned `STOP_BEFORE_STAGE1F`, not an acquisition
block:

- paired common-mode-corrected IR EBA was `0.50` versus the `0.75` threshold;
- geometry EBA was also `0.50`, so IR gain over geometry was `0.00`;
- thumb median effect was `-16.5` counts (4/6 strict-sign groups), while index
  median effect was `+64.5` counts (5/6), so the fingertip directions conflict;
- worst-fingertip return recovery ratio was `0.6928` versus the `0.50` limit;
- group primary effects were `+109.5, -1.0, +26.75, +27.75, -3.0, -17.0`
  counts.

Position was successfully retained as a diagnostic rather than an acquisition
gate: group median press-center shifts ranged from 0.28 to 4.73 thermal px.
The exact and 3x3 group-median effects had the same overall sign, but that does
not rescue the failed EBA, per-tip direction, and recovery gates. This is a
valid negative signal result for the frozen attempt 05 hypothesis. It does not
authorize Stage 1F or thermal control.

Immutable evidence:

- `scratch_lepton/stage1e_tip_pinch_signal_05.jsonl`, SHA-256
  `604aca6665910e9ce6329d2aeb5217b0fafa1018930814899af89fe6548df106`;
- `scratch_lepton/stage1e_tip_pinch_signal_05.log`, SHA-256
  `5998b109ab2a712f616ee0f4027bf9a7d020944196135e2efe80499c06809af9`;
- `scratch_lepton/stage1e_tip_pinch_signal_05_summary.json`, SHA-256
  `1cc44e45ceaf3f2332b6f7a7d0cc894b4a64f11e8dcc2485b4b8f299ce47cf88`.

### 2026-07-29 continuous single-index Null/Press path — first session formally incomplete

This is a new robot-free feasibility family, not attempt 06 and not a
replacement verdict for attempt 05. It tests one visible bare index finger on
one fixed rigid nonmetal surface. Null and Press are paired within six blocks;
both begin with light contact. Only the 5 s X phase differs:
`null=keep light contact`, `press=press hard`. Both then return to 5 s light
contact and finish with 5 s lifted/no contact. No scale or force sensor is
required for this instructed-condition feasibility run.

Implemented commits on `ir-hand-pressure-so101-teleop` are:

- `d39ddc9`: frozen block order, 5/5/5/5 s phases, and technical integrity;
- `52a359c`: TIP/DIP/PIP/MCP projection, distal/reference ROIs, and exclusive
  lossless frame archives;
- `7bb30f1`: 0.5 s paired binning, exact 64-sign-flip cluster inference,
  geometry diagnostics, and the 12-native-curve figure;
- `f5ddad6`: an Inferno-rendered PNG beside every authoritative uint16 thermal
  frame;
- `83bf67e`: operator capture CLI with manual FFC per block, 5 s guard, SPACE
  start, automatic phases, ten-second post-A3 rest, and reserve blocks;
- `c57472e`: offline analyzer and positive/negative/confounded/incomplete
  end-to-end fixtures.

The primary value remains distal 3x3 mean minus same-finger reference 5x5 mean.
Each trial is normalized only by subtracting its A1 `[3,5)` median. The primary
test is the predeclared paired exact cluster test, not legacy EBA. A significant
thermal cluster overlapping a significant UV/depth cluster is
`GEOMETRY_CONFOUNDED`, not a clean pressure result.

Fresh focused verification on 2026-07-29:

- new continuous-path suites: `39 passed`;
- new suites plus Stage 1E, Lepton UDP/capture, scalar/sparse projection, live
  shadow, and visualization regressions: `192 passed, 1 unrelated dependency
  deprecation warning`;
- CLI `--help` smoke and module compilation succeeded;
- the four initial UDP failures inside the restricted sandbox were reproduced
  as socket-creation `PermissionError` and passed when the identical command
  was rerun with loopback UDP permitted.

The setup photo is optional. If the supplied path does not exist, capture
continues and records `surface_photo=null` rather than fabricating an image:

```bash
cd /home/zhuokai/hand-teleop/webcam-input/.worktrees/ir-hand-pressure-so101-teleop
env -u PYTHONPATH /home/zhuokai/hand-teleop/.venv-lerobot/bin/python \
  lerobot_teleoperator_so101_webcam/capture_single_finger_curve.py \
  --session-dir /home/zhuokai/hand-teleop/scratch_lepton/single_finger_surface_press_curve_01 \
  --surface-material "rigid matte plastic" \
  --surface-photo /home/zhuokai/hand-teleop/scratch_lepton/single_finger_surface_setup.jpg \
  --preview \
  --manual-ffc
```

The operator presses SPACE only when the preview says the physical right hand
is ready. A1/X/A2/A3 then advance automatically. The capture stores every
in-window Lepton frame in both `raw/thermal_uint16/` and
`rendered/thermal_inferno_auto/`; the rendered images use per-frame automatic
contrast for communication only.

After capture:

```bash
cd /home/zhuokai/hand-teleop/webcam-input/.worktrees/ir-hand-pressure-so101-teleop
MPLCONFIGDIR=/tmp/single-finger-mpl \
env -u PYTHONPATH /home/zhuokai/hand-teleop/.venv-lerobot/bin/python \
  lerobot_teleoperator_so101_webcam/analyze_single_finger_curve.py \
  --session-dir /home/zhuokai/hand-teleop/scratch_lepton/single_finger_surface_press_curve_01
```

Current boundary:

- `software_ready=true`;
- `physical_session_run=true`, but `formal_primary_verdict=INCOMPLETE_FOR_PRIMARY_TEST`;
- no force ground truth;
- no controller input, robot/gripper actuation, pressure estimate, or Stage 1F
  authority;
- attempt 05 remains `STOP_BEFORE_STAGE1F`.

The first physical session is
`scratch_lepton/single_finger_surface_press_curve_01/`. It recorded 2,457
frame rows and successfully wrote all 2,457 authoritative uint16 thermal
frames and all 2,457 rendered thermal images. However, `tracking_valid=false`
for all 2,457 rows. The dominant reasons were 1,750
`TIP:color_to_depth_sdk_no_match` and 588
`TIP:color_to_depth_sdk_match_error_exceeded`. The original READY screen
checked only that MediaPipe saw a physical right hand; it did not check the
complete D435 depth association and thermal ROI chain, so it admitted an
unusable formal run.

Two analyzer defects were found during review:

- capture JSON encoded Python booleans as integer `0/1`, while the offline
  analyzer required identity with `True`; this falsely added artifact-write
  failures when reopening a session;
- the analyzer could not be rerun without overwriting existing outputs.

Commit `51dc23a` preserves booleans, accepts legacy persisted integer flags,
and adds non-destructive `--output-tag`. Reanalysis as `boolfix` removes the
false artifact-write reason but correctly keeps the formal verdict
`INCOMPLETE_FOR_PRIMARY_TEST` with zero selected pairs. The future capture
READY gate now evaluates the same full `build_frame_row` projection/ROI chain
used during recording and ignores SPACE until `tracking_valid=true`.

Because all raw thermal frames remain intact, a separate post-hoc thermal-only
salvage segments the largest right-side hot component, finds its leftmost 3%
tip band, and measures a 3x3 distal patch minus a 5x5 same-finger reference
patch. This rule found a component in all 2,457 frames. Four unique 0.5 s bins
had no frame and were linearly interpolated. Across the six paired primary
blocks, median Press-minus-Null values were `+25.28` counts in X, `+7.45`
counts in A2, and `-12.21` counts in A3. The exact paired whole-curve test
found no significant thermal cluster; its thermal-position diagnostic also
found no significant cluster. This is exploratory, ROI-rule-sensitive
evidence only, not a formal negative result and not a replacement primary
analysis.

Evidence hashes:

- `capture.jsonl`: `1d153da1908ccd15781c300c46bf77a69af5ad5553c458328a84ab1818e6b4bc`;
- `manifest.json`: `5eae3de0d043e81bd153c872292c4c47cc4236a638fc83b4be5a3f37a083fbd2`;
- `analysis_boolfix.json`: `04db721d1537f07d819b23dd7d22385e71459d88b94abaafe261ff3e3e1b2c97`;
- `salvage_thermal_only.json`: `0f4f5edacfe57687082a988d14983e8004fa982d3ce8e8fb1e77f18db7e1a551`;
- `figures_salvage_thermal_only/all_12_curves.png`:
  `25481101d55a341eee3aba1d2be8bdb3500a74453eeecb9aa31b7e1fbe9553cc`;
- `figures_salvage_thermal_only/roi_definition.png`:
  `785708f1f2b178c0bfb112fed07d889a60a32a100a1a2dbdabcc36aad147b432`.

### 2026-07-29 tracked thermal ROI v2 — implemented; old session remains incomplete

The approved noise-reduction path is implemented separately from both the
frozen D435 primary and the first thermal-only salvage:

- A1 initializes the finger axis once from the median of its first 2 s;
- distal and proximal masks are disjoint strips inside a 3x3-eroded hand mask;
- the primary value is
  `median(distal interior) - median(proximal interior)`;
- later frames translate the frozen masks by maximum hand-mask IoU rather than
  redefining the tip independently on every frame;
- center steps over 1 px, component-area changes over 20%, insufficient
  interior support, and distal finger width below approximately 10 px are
  invalid;
- a fixed automatically selected low-variance 15x15 lower-field desk patch is
  recorded as a drift diagnostic only and is not subtracted from the primary;
- missing 0.5 s bins are never interpolated for inference;
- raw curves, display-only five-frame rolling medians, and paired
  Press-minus-Null curves are rendered separately.

The capture READY screen now requires five thermal frames to form a valid A1
anchor and displays the estimated finger width. It accepts SPACE only when the
existing full D435-to-thermal ROI chain is valid and thermal finger width is
approximately 10 px or larger. This changes readiness only; it does not change
the 20 s A1/X/A2/A3 protocol or the raw archives.

Code commit `abbe996`:

- `lerobot_teleoperator_so101_webcam/single_finger_thermal_tracking.py`;
- `analyze_single_finger_thermal_tracking.py`;
- `tests/test_single_finger_thermal_tracking.py`;
- `tests/test_analyze_single_finger_thermal_tracking.py`.

Applied retrospectively to
`scratch_lepton/single_finger_surface_press_curve_01/`, v2 reduced the plotted
range substantially but correctly rejected the old dataset:

- A1 finger widths varied from 4.0 to 16.8 px across the 12 trials;
- 1,115/2,457 frames failed a v2 quality gate: 752 center-step failures, 317
  width failures, and 46 component-area jumps;
- six trials had no valid A1 baseline;
- 230/360 condition/bin locations were missing and no interpolation was used;
- only 0/6 pairs were complete, so no paired primary statistic or phase median
  was computed;
- the desk patch changed by a median absolute 6 counts and maximum 57 counts.

This is useful evidence that the previous large curves were dominated by ROI
instability, but it remains post-hoc and does not create a result for the old
session. The next physical session must move closer and hold the finger
steadier until READY shows a thermal width in the 10--15 px range.

Run the v2 analyzer after a new capture:

```bash
cd /home/zhuokai/hand-teleop/webcam-input/.worktrees/ir-hand-pressure-so101-teleop
MPLCONFIGDIR=/tmp/single-finger-mpl \
env -u PYTHONPATH /home/zhuokai/hand-teleop/.venv-lerobot/bin/python \
  lerobot_teleoperator_so101_webcam/analyze_single_finger_thermal_tracking.py \
  --session-dir /home/zhuokai/hand-teleop/scratch_lepton/<new-session>
```

Final v2 evidence hashes:

- `analysis_tracked_roi_v2.json`:
  `ab26cba8bd45566278dc339abcdb16f1cd720acbac417d084235bcdaba688a90`;
- `figures_tracked_roi_v2/all_12_raw_and_rolling.png`:
  `cedd2b4e38fadad451c79de1a600637021a94529a718eeba4a673a9c56059822`;
- `figures_tracked_roi_v2/paired_press_minus_null.png`:
  `e77f60d85cd6ea2bb563b719b1e18dc455556528d347be895f1f892fb465bb93`;
- `figures_tracked_roi_v2/roi_definition.png`:
  `f12b2ba7e69ced71e9ae13798d8304f2a9a2e2aca8ee6f5849e5993d02266190`.

### 2026-07-30 single-finger hold-check software — implemented, not run

The operator chose to keep investigating the single-finger press signal before
any witness-pad work, on the grounds that it is the minimal measurable version
of the hypothesis. Re-reading the session 01 artifacts identified two separate
acquisition defects rather than a signal result:

- **The D435 chain failed on 100% of frames for a geometric reason.**
  `capture.jsonl` recorded 1,750 `TIP:color_to_depth_sdk_no_match` and 588
  `TIP:color_to_depth_sdk_match_error_exceeded`. Where depth did resolve at all
  the values were min `0.259 m`, median `0.326 m` (`65.535` = invalid), against
  a requested depth profile of `1280x720 @ 6 fps`, whose Min-Z is approximately
  `0.28-0.35 m`. The hand sat on that limit, so the fingertip fell into depth
  holes. **The depth profile was NOT changed:** `1280x720` is pinned by the
  frozen Stage 0 runtime contract `stage0b_runtime_datums.json` and
  `run_session` fail-closes on `Stage 0 runtime metadata mismatch`. Lowering it
  would break the frozen provenance chain and must be a separate, explicitly
  approved decision.
- **ROI instability, not distance, dominated the thermal failures.** The 12 A1
  anchor widths were `4.0, 8.4, 10.0, 10.4, 10.8, 11.2, 11.6, 12.4, 14.4, 14.4,
  15.2, 16.8 px`, so the scale was mostly adequate; 752 of 1,115 invalid frames
  were `center_step_exceeded`, i.e. the whole hand translated while pressing.
  The physical fix is to brace the hand, not to widen the frozen gate.

Two bounded software changes were made on `ir-hand-pressure-so101-teleop`:

1. `lerobot_teleoperator_so101_webcam/capture_single_finger_hold_check.py` —
   a new **thermal-only, robot-free LIGHT/HARD/OFF long-hold sanity check**
   (default 30 s per phase). It uses no D435i, MediaPipe, projection, or Stage 0
   contract, because the v2 primary value
   `median(distal interior) - median(proximal interior)` is computed from
   thermal pixels alone. It archives every frame as lossless uint16 plus an
   Inferno rendering, and plots the primary curve over the three phases with a
   second panel showing distal/proximal/desk change from the first frame, so
   real signal can be told apart from global drift. Its manifest is hardcoded
   `role=sanity_check_not_preregistered` and
   `signal_verdict=not_a_formal_result`. Each frame records both the
   gate-enforcing measurement and an explicitly labelled `ungated_diagnostic`
   one, via a new opt-in `TrialTracker(..., enforce_stability_gates=False)`;
   the frozen v2 primary analysis is unchanged and still always enforces gates.
2. `capture_single_finger_curve.py` gained `--readiness-mode`, defaulting to
   `thermal_only`. The D435 chain is still recorded on every frame but no
   longer blocks SPACE, because at the distance this ROI needs it never became
   valid, and it does not feed the primary value. `d435_and_thermal` restores
   the previous behaviour. The mode is written into both the capture metadata
   row and the manifest.

Verification: `732 passed` across the full suite, plus `py_compile`, CLI
`--help`, `git diff --check`, and an offline synthetic render of the figure.
**This is software evidence only.** No hold check has been captured, no session
02 has been run, attempt 05 remains `STOP_BEFORE_STAGE1F`, and nothing here
authorizes teleop, pressure inference, or actuation. There is still no force
ground truth; `--load-note` only records free text about an independent
reading, and adding a real load cell remains the largest open quality gap.

### 2026-07-30 hold_check_01 — invalid measurement, and a post-hoc effect at the right location

The first hold check ran to completion (`status=complete`, 394 frames, three
30 s phases, `ffc_desired` never asserted). Its headline numbers were
`LIGHT 16.00`, `HARD 15.75`, `OFF 37.00` counts, i.e. a press effect of
`-0.25` counts. **That is not a signal result: the ROIs were on the wrong part
of the hand.**

The frozen v2 rule takes the leftmost 3% of a frame-median+100 threshold blob
as the fingertip. In this session the arm entered from the upper right, the
index finger pointed down-left, and warm background merged into the hand
component, so the rule returned `tip_uv=(52,45)` on the fist while the real
fingertip was at `(71,90)`. Both ROIs sat on the back of the fist about 10 px
apart, which also explains `corr(distal, proximal)=0.63` and
`corr(primary, desk)=0.61`. The reported `finger_width_px=28.0` was the width
of the fist, so the `>= 10 px` readiness gate passed for the wrong reason.

Recomputing the same 394 archived frames with the tip taken from the bottom-most
3% instead puts the ROIs on the pressing finger and gives:

| tip rule | LIGHT | HARD | OFF | HARD-LIGHT | last-10 s diff | corr with desk |
|---|---:|---:|---:|---:|---:|---:|
| frozen (leftmost 3%) | 16.00 | 15.75 | 37.00 | -0.25 | +18.00 | +0.61 |
| bottom-most 3% | 49.00 | 12.00 | 121.50 | **-37.00** | **-52.25** | +0.13 |
| geodesic from arm entry | 25.00 | 29.00 | 59.00 | +4.00 | +9.50 | +0.13 |

The negative sign is the predicted direction: the distal ROI cools relative to
the proximal one under load. Three geometry controls pass. Moving the ROIs away
from the tip (axial 10-18 against 24-32) keeps the effect at `-43` counts, so it
is not a tip-boundary artifact. The tip moved `+0.5 px` in u and `+1.0 px` in v
between LIGHT and HARD, and the distal/proximal pixel counts and interior
support did not change, so it is not an area artifact.

**Two limits keep this exploratory.** First, LIGHT was already falling at
`-1.14 counts/s` and HARD at `-2.23 counts/s`; with no return-to-light phase a
press effect cannot be separated from one continuous settling transient.
Second, the OFF phase is unusable — the near and far ROI variants disagree in
sign there (`+121.5` against `-142`) because the ROI leaves the finger once the
hand lifts. This is post-hoc, ROI-rule-sensitive evidence that the experiment
is worth redoing properly. It is not a formal result and authorizes nothing.

Evidence: `scratch_lepton/single_finger_hold_check_01/`, `capture.jsonl`
SHA-256 `4867ae5c1b7bb14453adc4f6d86afe9d868b13ead9f31103cc3cb4fab535818d`.

### 2026-07-30 hold check v2 — clicked ROIs, template tracking, A-B-A rounds

Two commits on `ir-hand-pressure-so101-teleop` rebuild the sanity check around
those two failures. Software only; **no v2 session has been captured.**

- `5984b16` adds `lerobot_teleoperator_so101_webcam/single_finger_click_roi.py`.
  `rois_from_clicks` takes the fingertip and a point up the same finger, so the
  axis is given rather than inferred, and it segments only a local box around
  that segment instead of thresholding the whole frame. `TemplateTracker`
  follows the ROIs with `TM_CCOEFF_NORMED`, which is invariant to affine
  intensity change, so a fingertip cooling by tens of counts does not drag the
  template; it fails closed on a low score, a match at the search boundary, or
  an ROI leaving the frame. A clicked surface patch is recorded as a drift
  diagnostic and never enters the primary value.
- `e4642fc` rewrites `capture_single_finger_hold_check.py` as
  `experiment_identity=single_finger_hold_check_v2`, schema 2. A round is
  `LIGHT_A / HARD / LIGHT_B` with contact held throughout, repeated `--rounds`
  times (default 3), scored as `HARD - (LIGHT_A + LIGHT_B) / 2` with a
  `return_recovery_ratio`. Rounds are separated by an untimed operator-paced
  rest and re-clicked, so a taped contact patch can cool between presses. Every
  round writes its ROI overlay figure **before** recording, which is the check
  that was missing when session 01 was captured in full at the wrong location.

The operator has fixed the contact location with a patch of black electrical
tape, which supplies mechanical registration and a repeatable high-emissivity
contact surface. Note that thin PVC tape has low thermal mass and poor
conduction, so the contact patch warms under the finger and the heat-sink
effect is expected to shrink across successive rounds; that is why the rest is
untimed and each round is independent.

Verification: `740 passed`, plus `py_compile`, CLI `--help`,
`git diff --check`, and an offline synthetic dry run of both figures. Still
robot-free and thermal-only, with no force ground truth, no pressure inference,
and no Stage 1F authority.

### 2026-07-28 post-attempt-05 review and candidate witness-pad hypothesis

One bounded external GPT Pro review accepted attempt 05 as a valid negative
result for the current tip-to-tip thermal-pressure hypothesis. Its main
recommendation was to retire that v1 signal rather than tune thresholds,
average away the opposite thumb/index directions, or repeat the same screen.
It did not recommend a closed-fist replacement: the relevant contacts would
be internally occluded and MediaPipe fingertip tracking is least reliable when
the fingers overlap.

The review judged that a later process-isolated Stage 1F shadow logger could
still be useful as diagnostic-only infrastructure, but only with explicit
qualifiers and no controller output. This advice does not satisfy or override
the current Stage 1F prerequisite, and no Stage 1F trial is authorized by this
handoff.

The strongest proposed new physical hypothesis is a **visible passive thermal
witness pad** on the index middle phalanx:

- the thumb presses one end of the pad while a separate part remains visible
  to Lepton as the witness ROI;
- an adjacent, physically separate, unpressed pad from the same material is
  the reference ROI;
- the first candidate feature is
  `mean(exposed_witness_ROI) - mean(adjacent_reference_ROI)`;
- the design remains passive: no electronics, active heating, controller
  input, or actuation.

For a first feasibility trial, use one layer of opaque matte-black,
high-quality electrical tape. Scotch Super 88 is preferred because FLIR
reports approximately `0.96` emissivity in both the 3-5 and 8-12 micrometre
bands; Super 33+ is a thinner alternative. Avoid generic glossy or
IR-transmissive vinyl and do not expose bare aluminium or copper to the
thermal camera. Relevant manufacturer guidance:

- FLIR:
  <https://www.flir.com/en-au/discover/rd-science/use-low-cost-materials-to-increase-target-emissivity/>;
- 3M Super 88:
  <https://www.3m.com/3M/en_US/p/dc/v000099244/>;
- 3M Super 33+:
  <https://multimedia.3m.com/mws/media/1983315O/scotch-vinyl-electrical-tape-super-33-datasheet-en-eu.pdf>.

The minimum construction is two non-connected, same-roll, single-layer tape
islands. The pressed island must retain an exposed witness area; the other
island is the unpressed reference. If the exposed area responds too slowly,
a later separately frozen variant may place a thin aluminium or copper thermal
bridge underneath the black tape, but the witness and reference islands must
remain thermally separate. Do not change materials during one frozen trial.
Electrical tape is not a skin-certified medical dressing: use only a small,
short-duration patch after checking tolerance, never wrap it tightly around
the finger, and stop on irritation.

Even a successful witness-pad result would show only that contact-mediated
heat transfer contributes observable information. Without an independent
force reference it must not be called calibrated pressure or force evidence.

### Safety boundary

The new runner imports no teleop, controller, recorder, robot, gripper, or
pressure-apply path. It always produces JSONL and opens a read-only RGB preview
only when explicitly requested. Thermal remains shadow/record-only. Color and
thermal source clocks still have no common time basis; host completion
age/skew is a fail-closed observation gate, not proof of sensor synchronization.

The older scratch files named
`stage1d_dynamic_skew_*` and
`stage1d_refined_xml_independent_34_36_20260727.json` are retained negative or
diagnostic evidence from the stopped four-blocker plan. Do not adopt them as
physical validation and do not bundle them into the hand-shadow MVP commit.

### Exact next action

**Immediate:** run the v2 A-B-A press check on the taped spot:

```bash
cd /home/zhuokai/hand-teleop/webcam-input/.worktrees/ir-hand-pressure-so101-teleop
env -u PYTHONPATH /home/zhuokai/hand-teleop/.venv-lerobot/bin/python \
  lerobot_teleoperator_so101_webcam/capture_single_finger_hold_check.py \
  --session-dir /home/zhuokai/hand-teleop/scratch_lepton/single_finger_hold_check_02 \
  --surface-material "black electrical tape on desk" \
  --rounds 3 \
  --manual-ffc
```

Per round: click the fingertip, a point up the same finger, and a surface patch
off the finger; check the outlined ROIs in the preview, then SPACE. Lift and
rest between rounds so the tape cools.

Read `figures/aba_rounds.png` and the per-round `aba_effect_count` and
`return_recovery_ratio`. A press effect that repeats across rounds and returns
in `LIGHT_B` is worth a frozen protocol. An effect that does not return is
drift. No effect at the correct location retires the single-finger path in
favour of a frozen witness-pad screen. None of these outcomes is a formal
result on its own.

**Standing constraints below are unchanged.**

Preserve the attempt 03, 04, and 05 artifacts and retire the current
tip-to-tip Stage 1E v1 signal. Do not repeat it, tune thresholds after the fact,
average away the opposite fingertip directions, or add a selector to rescue
the result. The only proposed next experiment is a separately specified and
frozen, robot-free, shadow-only witness-pad feasibility screen using one fixed
material and separate visible witness/reference ROIs. No such protocol has yet
been frozen or run.

The preserved Stage 1F teleop-coexistence plan remains conditional future
guidance and is not authorized. Neither a witness-pad experiment nor Stage 1F
may provide controller input or thermal actuation.

---

## §CAL. Lepton↔D435i thermal/RGB calibration — 2026-07-23 → 2026-07-24 (authoritative)

**Bottom line (2026-07-24):** a PROVISIONAL, physically-credible thermal→color extrinsic exists and
is **FROZEN** — `FINAL_flir_brown/extrinsic_refined.xml`. Two GPT Pro reviews accepted it as the
provisional build calibration ("retain this extrinsic… move into validation"). **Calibration work is
DONE for now — stop polishing on these 32 captures (that would turn the calib set into its own test
set); build the pipeline on this and validate end-to-end.** It is **NOT actuator-qualified** — do NOT
let thermal localization directly trigger the SO-101 overdrive; runtime must be depth-based and
fail-closed. The core mystery ("why did calibration keep failing/diverging") is fully solved below.
Open diagnostic-audit items (cheap, do alongside validation, NOT blockers): report epipolar per
thermal-ordering + in native px (thermal & RGB separately); rename/clarify LOO d|T| as full-vector
‖T₋ᵢ−T‖ (max was 0.54 cm, not "<0.5"); per-camera BA RMS; query exact D435i stream K/D (D=0 only if
rectified; RS Brown order [k1,k2,p1,p2,k3]); far-field H∞=Kt·R·Kc⁻¹ check of the 9° rotation;
frame-direction round-trip test.

### What is on disk (2026-07-24)
Run worktree `thermal-project-calibration-runs/worktrees/20260724T210232Z-attempt01/` at verifier
commit **`933c8bc`**. Final PROVISIONAL calibration:
`.../20260724T210232Z-attempt01/calibration/FINAL_flir_brown/`
- `thermal_intrinsics_FLIR_brown.xml` — **FLIR official** Lepton 3.1R Brown-Conrady:
  `fx=104.654 fy=104.483 cx=79.123 cy=55.689`, `D=[-0.39758,0.18069,0.00463,0.00420,-0.03381]`
  (source: FLIR Lepton 3.1R dewarping application note). USE THESE, do not re-fit thermal intrinsics.
- `extrinsic_thermal_to_color.xml` — R,T (thermal→color, `X_c = R X_t + T`), color_K (D435i factory),
  rigRot≈5–9°, |T|≈2.7–3.7 cm. Convention: thermal corners used RAW with the FLIR Brown K,D (no
  undistort); color = D435i factory pinhole, zero distortion.
- A superseded fisheye attempt is under `FINAL_fisheye/` — DO NOT USE (see defect below).
- **`extrinsic_refined.xml` (2026-07-24, current best)** — output of `scripts/refine_extrinsic.py`
  (global-consensus + bundle-adjustment on the 32 pairs, FLIR intrinsics fixed). |T|=3.80 cm,
  rigRot 9.0°, BA reproj RMS 2.03 px, 24/32 inliers. **Now STABLE:** leave-one-out spread dR≈0.3°,
  d|T|≈0.16 cm (the greedy method wobbled ~1 cm/3°). Branch-free symmetric-epipolar accuracy median
  ≈7 color-px (**<1 thermal pixel**; 1 thermal px ≈ 8–11 color px), with a tail on near-frontal pairs.
  Still PROVISIONAL; |T| is ~0.25 cm over the measured upper bound and the epipolar tail is large.

### The solved mystery (root cause)
1. Detection: classic `findChessboardCorners` fails on the cut-out board in RGB → we use
   `findChessboardCornersSB`. SB resolves the checkerboard 180° ambiguity INCONSISTENTLY between the
   160×120 thermal and 1280×720 color views → naive stereoCalibrate gave a decimetre |T|.
2. **The real killer was the THERMAL INTRINSICS.** Freely-fit OpenCV Brown-Conrady on the sparse
   4×3 (12-point) board over a 95° lens is badly conditioned: fx swung 67–296 across subsets with
   low RMS ("low RMS but wrong K"). A `cv2.fisheye` fit gave a STABLE fx≈104 — but GPT Pro proved
   that fitted fisheye is **non-bijective** (its radial map folds at θ≈56°, ~6.4% of pixels have no
   unique inverse) and its `cx=63.6` was wrong (true ≈79–81), which was being absorbed as a false
   ~8° extrinsic rotation.
3. **Fix (confirmed 2026-07-24):** use FLIR's published Brown-Conrady params (fx≈104.65 matches our
   stable fisheye fx; cx=79.1 matches pinhole/geometric). With FLIR intrinsics fixed, rigRot dropped
   13.6°→5.5° (truly parallel mount) and |T|→2.68 cm (within measured 3.05±0.5 cm). This validated
   both GPT Pro's diagnosis and that fisheye was compensating with a bogus rotation.

### Measured rig facts (operator, 2026-07-24)
- Lepton lens ↔ D435i **RGB** lens (4th/rightmost lens) front-glass spacing = **1.2 in ± 0.2 in =
  30.5 mm ± 5.1 mm**.
- **Board square pitch ruler-measured ≈ 30 mm** (both axes), consistent with the assumed 0.03 m used
  in all calibration; ruler precision ≈ ±3 mm → the metric |T| SCALE is supported to within ~10%
  (a board-pitch error transfers directly into |T|, so |T|=3.80 cm carries ≈ ±0.4 cm scale
  uncertainty — which brings it back into agreement with the 30.5 mm front-glass spacing). This
  closes GPT Pro's "measure the actual board pitch" item; note the epipolar metric can NOT validate
  |T| scale (essential-matrix scale ambiguity) — pitch + metric BA are what set it.

### Remaining limitations (why PROVISIONAL, per GPT Pro review 2026-07-24)
- Extrinsic not tight: |T| wobbles ~1 cm and rigRot ~3° with the pair subset; per-pair T spread
  ~1.2–1.7 cm. Held-out reprojection: most pairs 2–8 px, 2 near-frontal outliers ~11/32 px (color-
  only planar-PnP branch ambiguity in the VALIDATION, not necessarily the calibration); does NOT
  meet ≤3 px.
- With a nonzero baseline there is NO depth-independent thermal→RGB pixel map (a thermal pixel is a
  ray; its RGB projection needs depth). Runtime MUST use D435i depth or a tightly-bounded operating
  plane, and fail-closed.
- Resolution reality: 1 thermal pixel ≈ 10–11 RGB-pixel angular intervals; Lepton 8.6 Hz vs RGB
  30 fps → bound timestamp/latency for moving pinches.
- The whole flip-resolution + FLIR-intrinsics compute is **python-only** (`scripts/`), NOT in the
  frozen C++ verifier tools yet.

### Next steps (GPT Pro-recommended order, 2026-07-24)
1. ~~Re-estimate a single global R,T jointly over both orderings AND both planar-PnP branches.~~
   **DONE 2026-07-24** — `scripts/refine_extrinsic.py` (global-consensus + BA); result stable
   (`extrinsic_refined.xml`, LOO dR≈0.3°/d|T|≈0.16 cm). Open item: 25% rejection (>15% budget) — the
   dropped ~8 pairs are near-frontal-ambiguous; a depth-assisted color pose (below) would recover them.
2. Validate over the actual hand ROI + operational depth using depth-resolved / non-planar geometry
   (checkerboard corner RMS does NOT establish few-RGB-pixel accuracy). Also consider using D435i
   DEPTH to deproject the color board corners → unambiguous color pose (removes the planar-PnP branch
   ambiguity that caused the 25% rejection + the epipolar tail).
3. If FLIR defaults still miss the error budget, do a unit-specific Brown calibration with a much
   DENSER target (FLIR note recommends an 8×8 circle grid, ≥6×6 visible, captures to the borders),
   initialized from the FLIR params.
4. Port exact preprocessing + pixel conventions + projection into the frozen C++ path before
   qualifying. Until then: thermal must not directly actuate; require independent RGB/depth pinch
   confirmation; fail-closed on missing/out-of-range depth, timestamp skew, out-of-ROI, low confidence.

### Tooling built during this work (all committed to verifier `933c8bc` unless noted)
- `resolve_extrinsic` (C++, verifier): mount-prior flip resolution + front-face IPPE + outlier
  filter + stereoCalibrate. Reasonable but python fisheye/FLIR experiments went further; see caveats.
- `scripts/preview_capture.py` (workspace): live dual-detection capture. Auto-saves a pair only when
  BOTH cameras SB-detect 12/12 (`--thermal-only` = intrinsic pass, thermal FOV coverage; default =
  paired extrinsic pass). Solved the "blind capture" 5/36→34/36 detection problem.
- `scripts/compute_calib_from_captures.sh` (workspace): selects both-detected pairs, uses a separate
  `intrinsic_thermal/` set if present, runs camera_calibration→resolve_extrinsic→leakage-proof python
  held-out check. NOTE: this still fits thermal intrinsics from the board; the CORRECT path is to
  swap in the FLIR Brown params (above) and fix intrinsics.
- `scripts/calib_run.sh` §2 now also builds `resolve_extrinsic`.

### Capture recipe that finally worked (2026-07-24)
Separate passes with `preview_capture.py`: (1) `--thermal-only` intrinsics — board over the WHOLE
thermal FOV incl. corners/edges AND strong tilts (30–60°); (2) paired extrinsics — board where BOTH
detect, strong tilt diversity (median ~30°). BUT since FLIR publishes the intrinsics, the intrinsic
pass is now only a sanity check — the extrinsic pass (both-detected, tilt-diverse pairs) is what
matters. Watch for keyboard/false-positive SB detections and drop them.

---

## 0. Authoritative status (updated 2026-07-22)

### DONE on 2026-07-22

- **The complete software plan is approved and implemented through the physical gate.** GPT Pro
  task `task-355a910d60834fd8ade67fce19e378d6` returned `APPROVE` with no remaining
  Critical/Important software-plan issue. The only remaining gate is real target visibility,
  fresh capture, and held-out geometric acceptance.
- **Final implementation review is approved.** The first two GPT Pro tasks
  (`task-5d3b02bc4ed543778f48e7606f8de893` and
  `task-f32f197bd426414f967d86e0ceb04fde`) timed out without replies. Round 3
  (`task-50f52d9d1af14a459e552f6576ff775b`) returned five Important findings; all were reproduced
  before fixing: clean-build test targets, foreground SSH shell loss, ledger reserve/identity
  semantics, missing E/F validation, and visibility/safety provenance. Round 4
  (`task-356daae836274a13b13b6cca8a51f106`) returned four further Important findings; all were
  reproduced before fixing: schedule hash self-trust, completed-trial FFC, thermal-distortion
  shape, and PASS-before-seal ordering. Both rounds reported no Critical and no scope expansion.
  Round 5 (`task-b0c0ff0c832e471e8b39555f0bb9a74c`) returned two Important findings and again no
  Critical/scope expansion: active analysis-window FFC could coexist with thermal-valid, and
  valid modalities could lack raw artifact references. Both were reproduced, fixed in schema
  and semantic validation, and committed. Round 6 (`task-baa0555e0a2c407bbbdc77ff01ec1016`)
  reviewed the complete current heads and returned `APPROVE`: Critical None, Important None,
  Scope-expansion None. Its reply is
  `/tmp/webai-ir-final-code-review-round6/reply.md` (run ID
  `209449841c80a2b8a224ae58c18a34be`).
- **Phase 4/5 preregistration artifacts are committed.** The webcam worktree now contains
  `IR_HAND_PINCH_PREREGISTRATION.md`, Draft 2020-12 immutable study-contract and session-ledger
  schemas, positive/negative fixtures, six authoritative realized schedules, and a seal-time
  semantic validator that recomputes the schedule hash and enforces schedule/attempt/FFC and
  valid-modality artifact invariants. The current branch commit is
  `e5a5a37` (`fix(ir): enforce completed modality validity`). Focused contract plus RealSense
  geometry verification is `39 passed`. No pilot data, feature distribution, prediction, or
  performance has been inspected.
- **D435i exact-profile USB 3 preflight passed.** Serial `233522078685` is on a `5000M` bus.
  After five warm-up frames, 30 complete aligned `640x480 BGR8@30` + `640x480 Z16@30`
  framesets took `1.001 s` (`29.98 FPS`); the last depth frame had 213,389 positive pixels.
  Repeat this preflight before every session.
- **The non-fitting held-out verifier and capture serial guard are complete.** Use isolated
  ThermalProject branch `hand-teleop-heldout-verifier` at literal commit
  `933c8bc20ab4fe7983f81ab9960ef1e205ea06ea` (2026-07-24: adds `resolve_extrinsic`, the
  mount-prior flip-resolution fit tool; earlier on this line: 4x3 geometry, shared
  ThermalFrameAssembler for `lepton`, SB detection, D435i zero-coeff distortion accept).
  It compiles the 4x3/0.03 m geometry,
  indices 25-36, 12 images/144 points, and 3.0 px maximum gate; its CLI accepts paths only.
  It rejects malformed R/T/E/F and non-vector/unsupported thermal distortion before opening
  RealSense or exposing held-out data. `depth_saver` calls `enable_device` for the literal D435i
  serial and rejects the wrong active serial/profile before capture.
- **Fresh final software builds/tests passed.** Calibration CTest is `4/4 passed`; stream CTest
  is `2/2 passed`; `camera_calibration`, `verify_calibration`, `extrinsic`, `heldout_verify`,
  `lepton`, and `depth_saver` all built. The verifier and webcam worktrees are clean. The
  original `thermal-project` main worktree still has only the user's pre-existing
  `stream/CMakeLists.txt` change and was not touched. The complete webcam/prereg test command
  is `83 passed`; Python `compileall` is clean; `git diff --check` is clean in both implementation
  worktrees. The runbook literal verifier commit equals the branch HEAD, and corrected base
  `74268ab369904935c5b46fd13a14a0f34814bf4b` is an ancestor of that commit.
- **The exact operator runbook is committed.** Follow
  `lerobot_teleoperator_so101_webcam/REALSENSE_LEPTON_CALIBRATION.md`. It freezes source and
  executable hashes, 30 intrinsic/24 fit/12 wholly-held-out captures, serial/profile log,
  fail-closed PASS/FAIL terminal seal (PASS is written only after both seals are installed),
  and fresh-data retry behavior.
- **Codex weekly-limit guard is active for this handoff.** `scripts/codex_rate_watchdog.py`
  queries `account/rateLimits/read`, selects only the weekly `codex` bucket, reports handoff at
  <=5% and stop at <=2%, and has five passing unit tests. This handoff was refreshed before the
  threshold because the current work session will stop only after the final software review or
  the configured quota threshold. Weekly remaining reached 4% at the latest check, so the
  handoff threshold has fired; do not start another large task in this session. The hard stop
  remains 2%.

- **The ThermalProject `depth_saver` telemetry adaptation is now software-reviewed and live
  verified.** The isolated branch is `hand-teleop-telemetry-footer` at `74268ab`. It receives
  the formal four ordered `10004-byte` datagrams, assembles `61+61+61+57` image packets, and
  excludes the footer telemetry rows. A live test exposed and fixed one over-strict validation
  rule: outside packet 20, the high header nibble is not stable and must be ignored unless it is
  the `0xF` discard marker; packet IDs remain the low 12 bits, while packet 20 alone supplies
  segment `1..4`. Focused CTest passed `1/1` after the fix.
- **The six confirmed upstream calibration defects are fixed without changing the author's
  calibration method.** Commits `415127e` and `d7ee863` retain thermal `calibrateCamera` followed
  by fixed-intrinsic `stereoCalibrate`, the existing XML keys, transform direction, dot-pattern
  path, and unrelated code. The fixes are limited to: `5x4` squares -> `4x3` inner corners at
  `30 mm`; actual `160x120` thermal image size; matching capture filenames; BGR-to-gray corner
  refinement; exact D435i profiles/distortion checks; and minimum input/output validation.
- **Fresh software verification passed.** From clean `/tmp` build directories, all five required
  executables built (`camera_calibration`, `verify_calibration`, `extrinsic`, `lepton`,
  `depth_saver`); calibration CTest was `1/1 passed` and stream CTest was `1/1 passed`. The
  calibration task's independent review found no Critical or Important issue.
- **A non-calibrating live D435i + Lepton smoke test passed.** The sanctioned viewer first proved
  the Pi C++ streamer was producing complete telemetry frames. The corrected C++ `depth_saver`
  then ran for 15 seconds and continuously printed center temperatures around `17-24 C`, which
  proves it passed both RealSense `wait_for_frames()` and complete Lepton assembly. No `c` key
  was pressed, so no fake calibration images/artifacts were saved. The Pi streamer was stopped
  after the test.

### DONE on 2026-07-21

- **Phase 0 is complete.** The sanctioned C++ Pi streamer and laptop viewer produced real
  `160x120 uint16` frames. The current stream was verified as cooked RAW14 TLinear, not AGC
  video. The Lepton and RealSense are rigidly fixed with an approximately `1-2 cm` optical
  baseline. The intended sampled distance guard is configurable and currently `0.20-0.90 m`;
  `20-50 cm` is not a hard operating-distance limit.
- **Phase 1 is complete and live-verified.** The Pi C++ path now sets and reads back
  `AGC=disabled`, radiometry/TLinear enabled at `0.01 K/count`, manual FFC, and footer
  telemetry. The formal UDP format is now four ordered `10004-byte` datagrams per frame
  (`61 x 164` bytes per segment), not the old `60 x 164` layout. The laptop parser exposes
  TLinear/FFC telemetry. Python tests were `25 passed`; local and Pi C++ builds succeeded;
  CTest was `1/1 passed`; a live concurrent manual-FFC gate showed
  `complete -> imminent -> complete` without killing the streamer. The rollback Pi binary is
  preserved.
- **RealSense D435i is selected and working.** Device serial is `233522078685`. Synchronized
  RGB plus Z16 depth capture works, with depth aligned to RGB and converted to metres. The
  author's exact profiles, depth `1280x720@6 Z16` and color `1280x720@15 RGB8`, were also
  started successfully over the current USB 2.1 link, although at reduced throughput.
- **The C++ RealSense SDK blocker is resolved.** `librealsense2` `2.58.1` is installed under
  `/usr/local`; both `pkg-config --modversion realsense2` and
  `/usr/local/lib/cmake/realsense2/realsense2Config.cmake` were verified. It was built with
  `BUILD_ROSBAG2=OFF` because this workflow does not use librealsense `.bag` recording or
  playback; live D435i C++ capture and ROS2 `ros2 bag` are unaffected.
- **The first Phase 2 direct-landmark calibration attempt is complete and preserved as a
  failed gate.** Fourteen valid manually clicked correspondences covered `0.221-0.714 m`.
  Fit error was `RMS=2.375163 px`, `max=3.602714 px`; because the locked maximum is `3 px`,
  the result is **ESCALATE**, not GO. No projection JSON was published and no worst sample was
  removed. The report is
  `lerobot_teleoperator_so101_webcam/calibration/realsense_lepton_attempt2/realsense_lepton_hand_pressure_error_report.json`.
- **The author's ThermalProject calibration method has been audited and selected as the Phase
  2 upgrade path.** Its intended sequence is Lepton intrinsic calibration from thermal
  checkerboard images, paired Lepton/RGB capture, then fixed-intrinsic stereo calibration for
  thermal-to-RGB `R/T`. The upstream calibration and stream targets build now that the C++
  RealSense SDK is installed.
- **A first telemetry-footer receiver adaptation was created in an isolated ThermalProject
  worktree.** Its final review and live acceptance are recorded in the 2026-07-22 section above.

### NOT DONE / still blocking

- **Phase 2 has not passed.** There is no accepted RealSense-to-Lepton projection, no final
  `calibration.xml`, no final `extrinsic.xml`, and no held-out proof that maximum reprojection
  error is `<=3 Lepton pixels`.
- **Physical calibration capture is still blocked on the target/heating setup, not software.**
  The author workflow requires a target visible in both RGB and LWIR. The supplied DXF is a
  `5x4` square board with `30 mm` squares, meaning `4x3` OpenCV inner corners (confirmed
  against the author's DXF `docs/chessboard_pattern.dxf`; 4x3 and 3x4 are the same board rotated
  90 degrees, and 4x3 matches the DXF as-drawn). No suitable
  thermally visible setup has been confirmed available. A 500 W lamp is not mandatory; do not
  proceed until all `4x3` inner corners are simultaneously and reliably visible in RGB and LWIR
  without thermal saturation.
- **Phase 3 and experimental execution remain undone:** no accepted projection/two-level ROI,
  no six-session pilot data, no LOSO classifier result or GO/RETIRE decision, and no teleop
  overdrive activation. Phase 4/5 protocol publication is complete, but their data/analysis are
  not.
- **The completion audit found pre-hardware software still to implement.** The final binary
  session recorder has not yet integrated the frozen D435i `RS1/RS2` features, Lepton
  `delta_T`/FFC validity, realized schedule, append+fsync ledger writes, and terminal sealing in
  one trial path. The deterministic six-fold LOSO/permutation/McNemar analysis harness is also
  not implemented. These can be developed and synthetic-tested before the thermal target is
  available, but require a focused implementation plan/review; do not reuse the foam-oriented
  `analyze_ir_hand_pressure.py` or claim pilot evidence from synthetic fixtures.
- The implementation worktrees are committed and clean, but nothing is merged or release-ready.

### Exact next action

1. Before hardware becomes available, write and GPT Pro-review a focused implementation plan
   for the binary session recorder and frozen Phase 5 analysis harness described above, then
   implement them without inspecting pilot outcomes or inventing time synchronization.
2. For the physical gate, open
   `webcam-input/.worktrees/ir-hand-pressure-so101-teleop/lerobot_teleoperator_so101_webcam/REALSENSE_LEPTON_CALIBRATION.md`.
3. Prove simultaneous complete `4x3` corner visibility in D435i RGB and Lepton without thermal
   saturation. If it fails, stop before creating a run.
4. If visibility passes, execute a new immutable run exactly as written: 30 intrinsic images,
   24 paired fit images, 12 fresh held-out pairs, then accept only JSON with 12 images,
   144 errors, zero failures, and `global_max_error_px <= 3.0`.

---

## 1. Prior software baseline (historical; superseded where noted above)

Commit **`d78531c`** on `ir-hand-pressure-so101-teleop` — 13 files, 475 tests green.
All software upstream of a real thermal frame is built. Run tests with:
```
cd webcam-input/.worktrees/ir-hand-pressure-so101-teleop/lerobot_teleoperator_so101_webcam
env -u PYTHONPATH /home/zhuokai/hand-teleop/.venv-lerobot/bin/python -m pytest tests/ -q
```
(Package `__init__` imports lerobot, so tests need **`.venv-lerobot`**, not `.venv-webcam`.)

- `ir_capture.py` originally implemented `LeptonFrameAssembler` for the old telemetry-off
  layout (VoSPI: 4 segments/frame, 60×164 B packets,
  segment# in packet 20 byte0 high-nibble, big-endian uint16 pixels, repeated-segment =
  new-frame resync) + `LeptonUDPSource` (FrameSource over UDP, timeout→FrameUnavailableError).
  **This wire format was superseded today by the verified footer-telemetry `4 x 10004-byte`
  format described in Section 0.**
- `ir_hand_roi.py` — `select_thermal_blob_roi()`: hottest compact patch, threshold =
  background p25 + 100 counts (NOT median/p99 — those fail when hot region is large),
  area-bounded largest connected component. No cross-camera calibration needed.
- `ir_pressure.py` — `PressureConfig.roi_mode` ("projection"|"blob"), `calibration` now
  optional, `lepton_pressure_config()` factory (blob mode, image 160×120,
  `max_thermal_age_s=0.35` — 0.20 would always be stale at 8.7 Hz, `full_scale_delta=200`
  PROVISIONAL). Blob-mode pressure = frame-internal background delta (EMA baseline invalid
  for a moving ROI) — **must be refit against Phase 2 data.**
- Wiring: `teleop_viz_ee.py --ir-lepton-port` (blob mode DROPS the `--oak` requirement via
  `build_lepton_pressure_source`), `record_so101_ee.py` env `SO101_IR_LEPTON_PORT`,
  `record_ir_hand_pressure_trial.py --lepton-udp`, `view_ir_camera.py --lepton-udp`.
- `scripts/run_lepton_stream.sh` — start|stop|status over ssh (untracked; sits with
  other run_*.sh in the non-git meta-workspace).

---

## 2. Current HARDWARE state (2026-07-21) — camera CONFIRMED GOOD

- Pi **reachable** (`ssh anujn@192.168.50.2`), **I2C `0x2a` responds**, and the **camera is
  confirmed working** (user verified; FFC shutter click on power-up). It is NOT broken.
- The "0 UDP frames + SPI reads hang (exit 124)" I hit earlier was **self-inflicted and is now
  understood**: I violated the howto's **铁律 (iron rule)** by running ad-hoc python `spidev`
  probes (`scripts/lepton_spi_matrix.py`, inline `python3 -c "import spidev"`). Those fight the
  C++ streamer for `spidev0.0` and wedge the SPI controller into **D-state (uninterruptible)**,
  after which the streamer can't produce frames until a reboot. See `LEPTON_PI_HOWTO.md §5b`.
- **Therefore the bring-up is a software-discipline issue, not a hardware repair.** If the
  streamer won't emit frames: `ssh anujn@192.168.50.2 'sudo reboot'`, wait a full 60 s, then
  start the C++ streamer — and never run a python spidev probe again. Physical checks (howto §5:
  J3 power under load, module seat, J5–J9 jumpers) are only a fallback if a clean reboot + C++
  streamer still yields nothing.
- Phase 0/1 was re-verified from the laptop on 2026-07-21: `/usr/sbin/i2cdetect -y 1` saw
  `0x2a`; the existing official C++ streamer stayed up; the sanctioned viewer decoded a real
  160×120 16-bit frame and settled at ~8.0 fps. Observed viewer counts were approximately
  min 31886 / median 32235 / max 33970. Hard-vs-light separability is still untested.
- Intel RealSense D435i serial `233522078685` is currently connected on a `5000M` USB 3 bus.
  Do not reuse historical `/dev/videoN` numbers; enumerate the device or use the locked serial.
  The frozen aligned 640x480 RGB+depth Phase 4 profile passed 30-frame stable-state preflight.

---

## 3. Historical radiometry/range audit (Phase 0/1 has now resolved its runtime boundary)

**Correction:** the previous audit mistakenly answered for the older USB FLIR ONE
`flirone-v4l2` `/dev/video21` pipeline. That pipeline is not the camera path used here. The
current path is **FLIR Lepton 3.1R -> AnujN9/LeptonModule Pi streamer -> raw VoSPI over UDP ->
`LeptonUDPSource`**.

Per user instruction, do **not** fix the selector or continue collecting trials until the
experiment plan has been revised. Rep01 and rep02 remain preserved rejected setup trials;
neither counts toward Phase 2.

### Question 1 — Is this Lepton 3.1R radiometrically calibrated?

**Yes at the camera-model/factory level. The current best interpretation of the UDP pixels is
TLinear temperature in centi-Kelvin (`0.01 K/LSB`), not arbitrary uncalibrated counts.**

Evidence:

- The project author's page explicitly says he selected a Lepton 3.1R because it was
  radiometrically calibrated, verified readings against boiling water and a `-2 C` refrigerator
  using a thermocouple, and describes each pixel as unsigned 16-bit `Kelvin x 100`:
  <https://anujn9.github.io/Thermal/>.
- `ThermalProject/README.md` identifies the hardware and network package as Lepton 3.1R.
  `stream/lepton.cpp` documents `-mintemp/-maxtemp` in centi-Kelvin and computes center
  temperature from the received pixel with `value / 100 - 273` (the precise Celsius offset is
  `273.15`).
- The current Teledyne FLIR Lepton datasheet identifies Lepton 3.1R part `500-0758-03` as
  radiometric. It says the factory defaults for 3.1R have radiometry and TLinear enabled with
  `0.01 K` resolution; e.g. pixel `30000` means `300.00 K = 26.85 C`.
- The exact Pi checkout is `AnujN9/LeptonModule` commit `8c4832c`. Its network streamer copies
  four raw VoSPI segments to UDP without rescaling. Its I2C wrapper only implements connect,
  FFC, and reboot; it does not override radiometry/TLinear settings. Laptop
  `LeptonFrameAssembler` likewise reconstructs big-endian 16-bit pixels without normalization.
- Across the 268 retained rep01/rep02 frames, frame minima were `27811..28134`, medians
  `28726..29188`, and maxima `29859..30610`. Interpreted as TLinear, those are plausible scene
  ranges of `4.96..8.19 C`, `14.11..18.73 C`, and `25.44..32.95 C` respectively.

**Resolved later on 2026-07-21:** the Phase 1 C++ streamer now reads back radiometry, TLinear
enable/resolution, manual FFC, telemetry, and video configuration, while the laptop validates
the footer telemetry. See the authoritative status in Section 0.

### Question 2 — Is the range fixed or dynamic?

There are two different ranges; they must not be conflated:

1. **Temperature encoding and sensor high-gain range: fixed.** TLinear uses `0.01 K/LSB`.
   The project page and FLIR datasheet give the Lepton 3.1R high-gain scene range as
   `-10 C..140 C`. This physical/encoded temperature scale is not recomputed from each frame.
2. **Viewer/colormap display range: implementation-dependent.** Our `view_ir_camera.py` uses
   per-frame percentile autoscaling (`p1..p99`) only to make the image visible; the overlay,
   saved PNG, recorder, and estimator retain the original `uint16` values. In upstream
   `ThermalProject`, `depthimage.cpp` and the point-cloud path default to fixed display bounds
   (`27300..30800`), while standalone `stream/lepton.cpp` defaults to an intended automatic
   display range and accepts fixed `-mintemp/-maxtemp` bounds. `lepton.cpp` also contains an
   apparent typo in the auto-min initialization (`maxValue = 65535` instead of `minValue`), so
   that visualization code must not be used to infer sensor semantics.

**Answer for replanning:** the received raw temperature scale is fixed/radiometric; only the
grayscale/color visualization is dynamic in our viewer. A dynamic display does not invalidate
the original TLinear samples stored in the dataset.

### Constraints for the revised plan

- Do not use the old USB FLIR ONE `/dev/video21` audit to characterize this Lepton 3.1R path.
- Add sanctioned CCI readback of model/radiometry/TLinear/resolution/gain as a hardware gate;
  do not change camera state merely to inspect it.
- Once that gate confirms TLinear `0.01 K`, interpret pixels with
  `temperature_C = uint16 / 100.0 - 273.15`; keep the original `uint16` samples.
- Keep raw-temperature semantics separate from percentile/colormap display scaling.
- Current metadata says `lepton_raw_uint16_counts`; that name is conservative but incomplete if
  TLinear is confirmed. Leave it unchanged until the revised plan decides the schema migration.
- Intrinsic/extrinsic geometric calibration in `ThermalProject/calibration` is separate from the
  Lepton's factory radiometric calibration.

---

## 4. Historical next steps (superseded by Section 0 and the current plan)

### 2026-07-22 RealSense-only comparison baseline

The calibration-independent control baseline now has a minimal software extractor in the
`ir-hand-pressure-so101-teleop` worktree at `webcam_input/pinch_geometry.py`. It computes only
normalized 2D thumb-index aperture (scaled by index-MCP to pinky-MCP palm width) and absolute
D435i fingertip Z difference. It deliberately has no pseudo-3D, classifier, pose gate, smoothing,
or IR input. New TDD coverage is 20 tests; the full `webcam_input/tests` run is 64 passed.
GPT Pro narrow follow-up `task-26721db06e6f453891fda98e4f7c41e9` found no remaining
Critical/Important issues and approved this software contract.

This does **not** mean the comparison experiment is ready. The Phase 4/5 preregistration named in
the authoritative Section 0 is now complete, but the append+fsync+seal recorder is not implemented
and no six-session data exists. The legacy foam/continuous-sweep recorder remains intentionally
unmodified; future capture must record baseline and augmented inputs on the same binary trials.

### Step A — get pixel frames flowing (Phase 0/1) — SANCTIONED PATH ONLY
Use only the official C++ streamer + `view_ir_camera.py` (`LEPTON_PI_HOWTO.md §2–§3`). **Do NOT
run any python `spidev` probe / `run_lepton_stream.sh probe`** (iron rule — see §5 below).
```
# Pi side (foreground so you see its log):
ssh anujn@192.168.50.2 '~/Project/LeptonModule/software/build/raspberrypi_video_network -net 192.168.50.1 -port 8080'
# Laptop side (worktree copy has --lepton-udp; needs .venv-lerobot, the package imports lerobot):
cd /home/zhuokai/hand-teleop
env -u PYTHONPATH .venv-lerobot/bin/python \
  webcam-input/.worktrees/ir-hand-pressure-so101-teleop/lerobot_teleoperator_so101_webcam/view_ir_camera.py \
  --lepton-udp 8080
```
If the streamer prints its banner but no frames arrive: reboot per `LEPTON_PI_HOWTO.md §5b`
(spidev likely wedged), then retry — do not reach for a python probe. Go/no-go: recognizable
hand at ~8.7 Hz; hard pinch → fingertip blob visibly hottest; brief thermal residue after
release. Note raw-count min/med/max from the viewer overlay.

### Step B — Phase 2 hard/light trials (THE decisive physics gate)
```
cd /home/zhuokai/hand-teleop/webcam-input/.worktrees/ir-hand-pressure-so101-teleop/lerobot_teleoperator_so101_webcam
env -u PYTHONPATH /home/zhuokai/hand-teleop/.venv-lerobot/bin/python \
  record_ir_hand_pressure_trial.py --lepton-udp 8080 --surface skin \
  --contact hard_pinch --rep N \
  --bird /dev/v4l/by-id/usb-Intel_R__RealSense_TM__Depth_Camera_435i_Intel_R__RealSense_TM__Depth_Camera_435i-video-index0 \
  --root /home/zhuokai/hand-teleop/datasets/lepton_hard_pinch
# ...and --contact light_pinch, plus confusers (near-pinch/rubbing/warm-object)
```
The first setup trial, `hand-pressure_skin_hard-pinch_sweep_rep01`, is retained as a rejected
geometry example and does **not** count toward the Phase 2 totals: all 134 frames returned
`blob_too_large` (largest hot component ~12026–12835 px vs runtime max 900 px) because the
thermal FOV included the face/full forearm, and `/dev/video0` did not show the finger contact.
Rep02 fixed the RGB view with the D435i, but is also retained as rejected: all 134 thermal frames
still returned `blob_too_large` (~10052–10822 px), and the gradual-release label was inaccurate
because the hand stayed pinched through frame 124 and opened only near frame 133. Neither rep01
nor rep02 counts toward the collection totals. Thermal geometry must be fixed before rep03.

Collect ≥20 hard + ≥20 light + ≥10 confusers across ≥2 sessions. Then build the **new**
analysis: leave-one-session-out separability on blob features (blob_area, delta_mean/p95,
blob_core_mean + ~2 s temporal slopes area_slope/core_slope). **`analyze_ir_hand_pressure.py`
is FLIR-One foam-correlation-specific and NOT reusable — write a new binary hard/light harness.**
Go/no-go: ≥90 % per-trial hard detection, ≤1 false trigger on light+confusers, ≤~700 ms latency.
If it fails: iterate features once, else record a negative result and STOP before runtime.

### Step C — Phase 3 classifier artifact (offline)
Produce `calibration/lepton_hard_pinch_model.json` (threshold or logistic), wire it into
`ir_pressure.py` blob-mode scalar (replace the provisional background-delta), refit
`full_scale_delta`. Replay Phase 2 dataset through `HandPressureEstimator`, agreement ≥95 %.

### Step D — Phase 4 shadow + soak (wiring already done for teleop/record; soak NOT done)
`ir_pressure_soak.py --lepton-udp` is **not implemented** (heavy, calibration-baked — deferred).
Run live shadow: `--ir-pressure-shadow --ir-lepton-port 8080 --ir-sidecar <csv>`; review that
hard-pinch events fire only on real hard pinches. Then Phase 5: flip `--ir-pressure` /
`SO101_IR_PRESSURE=1`, stationary-arm soft-object test first, then A/B pick-place.

---

## 5. Gotchas for the next agent
- **IRON RULE (howto top + §5b): never run ad-hoc python `spidev` reads.** They fight the C++
  streamer for `spidev0.0` and wedge the SPI controller into D-state → streamer stops producing
  frames until a reboot. This is what cost me a whole debugging detour. So: **do not use
  `scripts/lepton_spi_matrix.py`, `run_lepton_stream.sh probe`, or any `python3 -c "import
  spidev"`.** Health-check with `i2cdetect -y 1` (expect `0x2a`) + running the C++ streamer only.
  The violating `probe` subcommand has now been removed from `run_lepton_stream.sh`.
- **If the streamer won't emit frames:** `ssh anujn@192.168.50.2 'sudo reboot'`, wait a full 60 s,
  then start the C++ streamer. Only after a clean reboot still fails do you consider physical
  checks (howto §5). The current deployed build is SPI CE0 / **spidev0.0, mode 3, 20 MHz**;
  follow `LEPTON_PI_HOWTO.md` rather than older notes.
- `run_lepton_stream.sh` now anchors process matching to the full streamer path, avoiding the
  former broad `pkill -f '[r]aspberrypi_video_network'` SSH-shell self-match.
- The authoritative "is it up" check is the streamer log banner + UDP frame arrival in the viewer,
  not `pgrep` (`pgrep -x` warns on the >15-char name but still works for kill).
- Venvs: anything importing the `lerobot_teleoperator_so101_webcam` package (tests, runtime,
  `view_ir_camera.py`, `record_ir_hand_pressure_trial.py`) needs **`.venv-lerobot`** — the
  package `__init__` imports `lerobot`. `.venv-webcam` is only for standalone MediaPipe scripts.
