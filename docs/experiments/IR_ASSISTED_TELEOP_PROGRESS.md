# IR-Assisted SO-101 Teleoperation Progress Record

**Last updated:** 2026-07-20  
**Scope:** History of the FLIR/OAK hand-pressure work from the first SO-101
grip pilot through the latest raw-count FFC test. This is a progress and
evidence record, not a claim that IR pressure is deployable.

## 1. Intended Product and Boundary

The project started as an investigation of whether a FLIR ONE image can add
useful information about how tightly a person is squeezing. The current
product target is deliberately narrow:

- OAK/MediaPipe controls hand pose and **base gripper aperture**.
- IR would optionally produce a conservative `hard` / `not hard` cue that adds
  bounded extra gripper squeeze in an OAK-projected contact ROI.
- It is not a Newton-force estimate, a pressure measurement, or a Celsius
  temperature measurement.
- It is not a LeRobot policy observation in v1 because the FLIR observes the
  human operator rather than the robot at autonomous deployment.
- Any invalid registration, missing OAK landmark, stale thermal frame, FFC,
  release, or low-quality baseline must yield **zero IR extra squeeze** and
  retain OAK-only control.

In this project, "calibration" has three different meanings which must not be
confused:

1. **OAK-to-FLIR registration:** project the OAK contact region into the FLIR
   image.
2. **Operational classifier calibration:** establish an inactive baseline,
   normalize raw features, handle FFC, and select a conservative binary
   threshold.
3. **Temperature calibration:** convert sensor counts to Celsius. This is not
   currently supported and is not required for the binary teleoperation cue.

## 2. Timeline and Evidence

### Phase A: SO-101 Robot-Gripper Feasibility Pilot (early July)

**Question:** Can the existing colorized FLIR stream contain a grip-related
signal when the robot slowly closes around a fixture?

**Method:** Record thermal PNGs and SO-101 telemetry while slowly sweeping the
gripper on a rigid hard block and on foam. The original processing used a
fixed thermal ROI and scalar features such as contact area and `mean_delta`.
`present_load`/`present_current` were used only as relative servo-effort
proxies, never as calibrated force.

**Results:**

- The fixtured hard-block continuous sweep was the clearest early positive
  result. In the narrow goal-position 30-to-25 moving window, `mean_delta`
  versus servo load had Spearman correlations of `0.922`, `0.770`, and `0.766`
  for reps 02--04.
- The foam sweep was a negative/low-signal control. Pooled `mean_delta` versus
  load Spearman was `-0.230`; the two individual sweep values were `-0.359`
  and `-0.309`.

**What this established:** In a highly constrained rigid-fixture setup, the
old pipeline can exhibit a repeatable-looking image/servo relationship.

**What it did not establish:** The result does not transfer to a human hand,
foam compression, a new camera placement, a new thermal baseline, or real-time
control. The input was dynamic-AGC colorized RGB, not raw counts. It cannot be
called force sensing.

**Primary artifacts:**

- `../../docs/experiments/IR_GRIP_FORCE_EXPERIMENT.md`
- `datasets/ir_grip_force_viability/organized_results/final_hard_sweep_goal30_to25/`
- `datasets/ir_grip_force_viability/soft_sweep_ir_load_summary.csv`

### Phase B: Hand-on-Foam Viability and Time-Control Analysis (early/mid July)

**Question:** Is there any hand/foam relation after removing the obvious
confound that both the squeeze proxy and IR can change simply with elapsed
sweep time?

**Method:** Five whole-hand foam sweeps were recorded. A hand-to-foam distance
proxy was used as the reference. The selected IR feature was negative IR area
at a 3-sigma threshold. The primary summary excluded rep01 and compared
time-only, IR-only, and time-plus-IR models.

**Results for reps 02--05:**

- pooled normalized Pearson: `0.601`;
- partial Pearson after removing time: `0.354`;
- incremental R2 from IR after time: `0.088`;
- per-rep partial r: `0.103`, `0.480`, `0.682`, `0.154`.

**What this established:** A weak-to-moderate within-session association may
remain after a time control.

**Limitations identified immediately:** The proxy was not geometric foam
compression or force; per-repetition effect size varied substantially; feature
selection was post-analysis; and the source frames were colorized dynamic-AGC
images. This result was therefore retained as suggestive evidence, not a
deployment calibration.

**Primary artifacts:**

- `datasets/ir_hand_pressure_viability/organized_results/final_hand_foam_time_control/`
- `foam_rep02_rep03_rep04_rep05_hand_distance_time_control_time_control_report.md`

### Phase C: Hysteresis, Drift, and Control-Architecture Review (mid July)

External feedback correctly shifted the criterion from "can a graph correlate?"
to "can a classifier safely affect a gripper?" The three critical control risks
were recorded:

1. **Hysteresis:** the same nominal compression may give different IR states
   during loading and unloading; a memoryless `extra_squeeze = f(IR)` may keep
   a gripper closed after the operator releases.
2. **Drift and FFC:** a moving baseline or shutter event can enter the command
   as false extra squeeze.
3. **Within-trial usability:** a pooled correlation is not enough; a single
   live interaction must be fast, stable, and safe.

This changed the experimental requirement. Future evidence must include
released baselines, hand-near/no-contact controls, loading/unloading, rapid
release, real timestamps, blocked validation, and explicit FFC handling.

### Phase D: Direct Hard/Not-Hard Prompted Classifier Pilot (2026-07-14/15)

**Question:** Can a colorized-IR-only classifier distinguish `hard` from
`not hard` on the same foam under fixed posture?

**Data:** Two separate trials were retained:

- `hard-classifier_s01_fixed-posture_foam_zk_rep01`;
- `oak-squeeze_s01_fixed-posture_foam_zk_rep02`.

Each used randomized prompts `0, 25, 50, 75%`. Analysis defined `hard` as
prompt >=70% and `not hard` as prompt <=50%. Each sample aggregated the final
one second of a hold; frozen frames and detected AGC jumps were excluded.
Validation preserved sequences and trials: no random frame split was used.

**Results:**

- best within-trial blocked balanced accuracy: `0.625` (trial 01) and `0.667`
  (trial 02);
- best train-one-trial/test-the-other balanced accuracy: `0.625`;
- all-feature nearest-centroid: at or near chance across trials.

**Conclusion:** The two-trial pilot does not support controller deployment.
The labels were subjective prompts, not measured force, pressure, or foam
compression. The two trials used different camera eras/setups, which makes
their poor cross-trial transfer useful evidence of fragility but not a final
impossibility proof.

**Primary artifacts:**

- `datasets/ir_hard_classifier/analysis/hard_press_classifier_20260715/`
- `exports/ir_archive/hard_press_20260715/gpt_pro_ir_hard_press_classifier_20260715/`

### Phase E: Fixed-Geometry Foam Experiment and Rep08 (2026-07-15)

**Reason for redesign:** Subjective squeeze prompts were inadequate. The
reference changed to actual foam width reduction measured by two OAK RGB
markers, not to a force label.

**Protocol improvements implemented:**

- foam placed in a repeatable upright fixture;
- two marker dots define `compression_pct = 100 * (d0 - d) / d0`;
- frozen FLIR central, contact, background, room, and warm-reference ROIs;
- released `R`, near/no-contact `N`, just-contact `C0`, and `C10/C20/C30`
  states;
- randomized steady holds, drift baselines, loading/unloading hysteresis, and
  release pulses;
- preflight snapshot, marker visibility gate, thermal-size/ROI checks, actual
  timestamps, repeated-frame flags, and invalid-action logging.

**Data quality:** There are ten trial directories. Nine are explicit incomplete
or diagnostic attempts caused mainly by marker tracking and target-hold gates.
They are retained but do not count as independent classifier evidence. Rep08
is the only complete protocol: 43 actions, five invalid/retried actions, 5,737
synchronized frame indices, and 3,138 valid deduplicated stable-hold frames.

**Rep08 descriptive result:**

- actual compression versus normalized foam-center feature: Spearman
  `rho = -0.4151`, `p = 0.00564`;
- same-recording binary screen with hard >=25% and not-hard <=10%:
  rank AUC `0.7241`.

**Conclusion:** Rep08 is more scientifically relevant than subjective prompts,
but it is still a single recording. The same-recording AUC and p value do not
measure generalization across session, day, camera remount, participant, foam
piece, or camera state. Its FLIR source was still colorized RGB.

**Primary artifacts:**

- `../../docs/experiments/FOAM_COMPRESSION_EXPERIMENT.md`
- `../../docs/experiments/IR_FOAM_CLASSIFIER_LOCAL_ARCHITECTURE.md`
- `datasets/ir_foam_compression/trials/foam-compression_foam-20260715retry2_foam_zhuokai_rep08/`
- `exports/ir_archive/foam_rep08_20260715/gpt_pro_ir_foam_rep08_review_20260715/`

### Phase F: SO-101 Runtime Architecture and Software Hardening (2026-07-09 onward)

In parallel with the empirical work, the SO-101 teleop branch was designed and
implemented as an **optional** IR path rather than a new primary controller.

- OAK image coordinates and depth are propagated with hand landmarks.
- A depth-aware OAK-to-FLIR projection maps pinch/contact ROIs into thermal
  image coordinates.
- The IR runtime has rolling/frozen baseline behavior, quality checks,
  projection bounds, freshness/skew handling, and fallback semantics.
- `WebcamEEController` remains the sole control owner. OAK owns pose/aperture;
  IR can only replace fixed gripper overdrive when calibrated and healthy.
- Recording uses the same control path; no IR feature was added to the
  autonomous-policy observation schema.

The prior software review recorded `450` teleoperator tests and `41` webcam
tests passing on its reviewed worktree. That was a software approval, not a
hardware deployment approval. The original live hardware Gate 1 was not signed
off because the FLIR bridge could not be started non-interactively at that time.
Later standalone raw-count capture proves only the bridge/audit path, not live
SO-101 overdrive behavior.

**Primary artifacts:**

- `docs/superpowers/plans/2026-07-09-real-time-human-hand-ir-pressure-so101-teleop.md`
- `docs/superpowers/specs/2026-07-09-real-time-human-hand-ir-pressure-so101-teleop-design.md`
- `lerobot_teleoperator_so101_webcam/ir_pressure.py`
- `lerobot_teleoperator_so101_webcam/ir_hand_calibration.py`

### Phase G: Raw-Count / Radiometric Feasibility Audit (2026-07-16)

**Reason:** Historical thermal PNGs are dynamic-AGC RGB. Their colors cannot
be recovered as raw counts or Celsius, so a usable classifier must first show
that exported raw counts are repeatable enough to justify a new experiment.

**Implementation:** A separate audit branch added opt-in export of decoded
little-endian `uint16` thermal grids (`80x60` for the connected payload), JSON
metadata, FFC tags, repeat flags, and timing. The default RGB display remains
non-radiometric and dynamic; raw files are the primary scientific measurement.

**Valid FFC run:** `ffc_02` recorded 2,558 raw frames over 269.9 s. After
deduplication there were 2,298 normal frames. Two independent FFC events were
captured, each with 12 FFC-tagged frames.

| FFC event | Target 10 s pre -> 3 s post | Target shift | Control 10 s pre -> 3 s post | Control shift |
| --- | ---: | ---: | ---: | ---: |
| 1 | 3575 -> 3639 | +64 | 3549 -> 3619 | +70 |
| 2 | 3581 -> 3619 | +38 | 3556 -> 3602 | +46 |

The target and stationary control shifted by similar amounts. Their local
normal-window standard deviations were only about 4--6 counts. At 30 seconds,
the FFC residual shifts were still up to +27 counts. This is a camera-global
state transition, not a local target signal.

**Restart runs:**

- `ffc_01` is invalid: it captured RGB but no raw metadata because an older
  bridge without raw export already owned the loopback.
- `restart_01` is a qualified raw stable-window replicate (target/control
  medians 3516/3509; standard deviations 4.79/5.28) but later RGB observation
  timed out.
- `restart_02` is not a clean restart baseline because FFC occurred during its
  stable window and its RGB observation aborted.
- `restart_03` completed its observation and had quiet raw stable windows, but
  it does not create three clean independent restart replicates.

**Current raw-count conclusion:**

- Direct stateless `extra_squeeze = f(raw_count)` is not defensible across an
  FFC event.
- The data do not support a Celsius calibration or force claim.
- A slow, FFC-aware, explicitly re-baselined classifier remains unproven; it
  has not been categorically disproven by one stationary FFC test.

**Primary artifacts:**

- `tools/flirone-v4l2-radiometric-audit/RAW_COUNT_REPEATABILITY_PLAN.md`
- `datasets/ir_raw_repeatability/session-20260716-phone-target/runs/ffc_02/`
- `datasets/ir_raw_repeatability/session-20260716-phone-target/runs/restart_01/`
- `datasets/ir_raw_repeatability/session-20260716-phone-target/runs/restart_02/`
- `datasets/ir_raw_repeatability/session-20260716-phone-target/runs/restart_03/`

### Phase H: Raspberry Pi + Lepton 3.1R Hardware Bring-Up and Repair (2026-07-20)

**Context:** Retried the Raspberry Pi + FLIR Lepton 3.1R path (Matthew's "option 1"),
which had been dead since ~2026-07-16. Goal was to get the radiometric Lepton streaming
again so raw-count / hard-light physics-gate trials can resume.

**Failure signature:** I2C control interface at `0x2a` did not respond (`i2cdetect -y 1`
shows `--`); SPI (VoSPI) output was all zeros; but all power rails read normal at the
breakout header (VIN 3.3 V, VCC28 2.8 V, VCC12 1.2 V). It had streamed one full frame on
Friday, then went dead — a classic "worked, then suddenly and permanently dead."

**Diagnosis (Pi/wiring/software fully cleared):** With the official `i2cdetect` and the
known-good `raspberrypi_video_network` capture binary (not just custom scripts), and a
**Tektronix MSO2012B scope**: SPI clock (SCLK) = clean square wave and I2C clock (SCL) =
clock bursts (the Pi drives both buses); I2C decode showed the Pi correctly sending
`START -> Addr[R] 0x2A -> STOP` **with no data byte and no ACK**; MISO (camera data out) was
flat ~10 mV noise. So the Pi addresses the camera correctly and the module returns nothing.
Correct chip-select is **CE0 / spidev0.0** (not 0.1).

**Root cause — a marginal power/contact fault, NOT a dead module or crystal.** Fixed by full
reassembly: reseating the Lepton **module** in its 32-pin Molex socket, press-checking the
**J5–J9 jumpers** (they route the 25 MHz master clock, 2.8 V / 1.2 V rails, and power-up
sequence to the module — all must be seated), re-plugging the breakout, and reseating the
**J3 power cable**. Two candidates could not be separated, because both contacts were disturbed in the same
session: **(a)** the **J3 power connection** — it measured 3.3 V on a DMM (which draws ~µA)
yet may not have supplied the Lepton's ~150 mA boot/FFC current under load, so the module
never booted while the rail still "looked" fine (VIN on this rig has a history of flaky
contact — measured 1.3 V ↔ 3.2 V jumping in an earlier session, and the original Friday
frame also came only after a power reseat); and **(b)** the **Lepton module ↔ Molex socket
contact** — if the module was pulled and reinserted **under power**, that matches SparkFun's
documented recovery trick for a stuck Lepton. Either restored a marginal contact; the exact
one is unresolved.

**Result — camera confirmed working and stable:** FFC shutter click on power-up, `i2cdetect`
sees `0x2a`, SPI streams full VoSPI (segments 0–7, ~370k–480k non-zero bytes/probe), and the
Pi sender + laptop listener (`scratchpad/lepton_listen.py`) captured a real 160×120 thermal
frame (raw 10324–13399). Verified stable across **multiple full power-cycles** (physical
unplug/replug), not a one-off.

**Useful facts (from the Breakout v2.0 datasheet, `~/Downloads/DS_16912_FLiR...V2.pdf`):**
J2 is a 2×10 header; testable at the header are **pin17 RESET_L**, **pin18 MASTER_CLK
(25 MHz)**, **pin20 PW_DWN_L** (all should be high / present when running), **pin5 SDA /
pin8 SCL / pin7 SPI_CLK / pin12 SPI_MISO / pin10 SPI_CS**. **J3** = power-in (pin1 GND,
pin2 VIN); on R120 boards D1 is reversed so the Lepton must be powered via J3 pin2, not J2
pin2. **J5–J9** = the clock/rail/sequence jumpers.

**Durability follow-up (recommended, not yet done):** resolder / secure the **J3 power line**
and seat the module flat, since the fix is contact-based and can recur. If it goes dark again,
check J3 power (under load) and the module seating **first** before any deeper diagnosis.
Diagnostic lesson: an unloaded DMM voltage reading does **not** prove a rail holds up under
the chip's operating current.

**Status:** the Lepton Pi camera is functional again, so Phase 0/1 live bring-up and the
decisive hard/light physics-gate trials (per the `ir-hand-pressure-so101-teleop` plan) can
resume when the team chooses to.

## 3. Claims Supported Today

| Claim | Status | Reason |
| --- | --- | --- |
| FLIR colorized RGB is calibrated temperature | Not supported | Historical images use dynamic AGC; no per-device radiometric calibration is verified. |
| IR estimates Newton force/pressure | Not supported | No suitable ground-truth force reference exists in the hand/foam data. |
| IR has a useful hard/not-hard classifier for teleop | Not supported | Two-trial cross-session balanced accuracy is 0.625; rep08 has only same-session descriptive evidence. |
| The old hard-block fixture has a positive IR/servo relation | Supported, narrow scope | Three constrained sweeps show high `mean_delta`/load rank correlation. |
| Hand/foam IR contains a possible slow proxy signal | Suggestive only | Time-controlled relation and rep08 geometry relation exist, but are not independently validated. |
| Direct stateless raw IR control survives FFC | Rejected | Both target and control raw counts move by 38--70 counts after FFC. |
| OAK-only teleop remains safe fallback | Supported by design | IR path is optional and must output no overdrive on invalid state. |

## 4. Current Data and Software Location

| Item | Location |
| --- | --- |
| Robot-gripper pilot | `datasets/ir_grip_force_viability/` |
| Hand viability/time-control analysis | `datasets/ir_hand_pressure_viability/` |
| Hysteresis pilot | `datasets/ir_hand_pressure_hysteresis/` |
| Direct hard/not-hard pilot | `datasets/ir_hard_classifier/` |
| Fixed-geometry foam trials | `datasets/ir_foam_compression/` |
| Raw-count repeatability audit | `datasets/ir_raw_repeatability/` |
| Teleop implementation | `webcam-input/lerobot_teleoperator_so101_webcam/` |
| Raw bridge audit branch | `tools/flirone-v4l2-radiometric-audit/` |
| Current external-review package | `exports/ir_archive/teleop_review_20260716/ir_evidence_gpt_pro_20260716_teleop_review.zip` |
| Archived IR review/export packages | `exports/ir_archive/` (dated subfolders; all prior GPT-review packages) |

### Git branch map (webcam-input)

The IR work lives on **two parallel branches** off common base `5a2cf5c`, kept separate on
purpose. They are complementary halves, **not duplicate versions** of the same code:

| Branch | Role | Holds |
| --- | --- | --- |
| `so101-webcam-diffusion` (default checkout) | **Offline experiment & analysis** — the viability study behind this record (Phases D–G) | `record_ir_*`, `analyze_ir_*`, foam-compression + hard-press classifier, dataset processing |
| `ir-hand-pressure-so101-teleop` (git worktree under `webcam-input/.worktrees/`) | **Live runtime integration** — the optional IR-overdrive control path, hardened (43 commits) | `ir_pressure.py`, `ir_hand_calibration.py`, OAK→FLIR projection, gripper overdrive, robot-free soak tests |

One answers "does IR carry a usable signal?" (the negative result); the other builds "the
optional IR-overdrive path into live teleop." **Do not merge until the deployment decision is
settled** (currently OAK-only; IR gated off). The worktree is intentional isolation, not clutter.

## 5. Current Gate and Next Decision

No IR-derived gripper overdrive should be enabled on hardware today.

The current package prepared for GPT Pro review asks for one of three explicit
outcomes:

1. **Retire IR-assisted overdrive and keep OAK-only teleop.**
2. **Run one final falsification experiment** with raw counts, a pre-registered
   OAK geometry label, full-session blocked validation, and a fixed FFC
   re-baseline policy.
3. **Declare data insufficient to choose**, while identifying the minimum
   missing evidence.

Automatic upload of that private package was blocked by the local safety policy;
the verified upload file and prompt are:

- `exports/ir_archive/teleop_review_20260716/ir_evidence_gpt_pro_20260716_teleop_review.zip`
- `exports/ir_archive/teleop_review_20260716/ir_evidence_gpt_pro_20260716/GPT_PRO_REVIEW_PROMPT.md`

Until an independent review or a pre-registered final test changes this gate,
the correct engineering position is: **OAK-only teleop is the usable path; IR
is a research branch with evidence of real confounds and no deployment claim.**
