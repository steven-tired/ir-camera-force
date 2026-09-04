# IR Hand Pinch Loose/Tight Pilot Preregistration

## Status and scope

This preregistration freezes the acquisition and analysis contract for a
RealSense-only baseline and a RealSense-plus-TLinear augmented model. It does
not implement a recorder or classifier, does not claim hardware
synchronization, and does not authorize a pilot before the accepted thermal
calibration and recorder integrity gates pass.

The pilot uses exactly two labels, `loose` and `tight`. No new feature, middle
class, automatic pose gate, imputation, pseudo-time-synchronization, dynamic
threshold, tuned model, or AUC gate may be introduced after acquisition.

## Participant, sessions, and instructions

- Enrol exactly one participant under a non-empty pseudonym.
- Run six sessions on six distinct calendar days, fully restarting the cameras
  and acquisition processes for every session.
- Permit at most four unrecorded familiarization trials per class before each
  session.
- Record exactly 15 completed trials per class in every valid session.
- Instruct `loose` as the minimum grip/contact needed to keep the task stable.
- Instruct `tight` as the maximum comfortable grip/contact without pain.

The labels are subjective task instructions; no force sensor is introduced.
A participant safety stop ends the session and cannot be replaced.

## Trial randomization and technical-abort reserves

The following stored schedules are authoritative (`L=loose`, `T=tight`). Seeds
are provenance only; runtime code must read these strings and must not
regenerate them.

```text
session 1: LLTLLTLTTTLTLTTTLTLLTLLLTTLLTT
session 2: LTTLTTLTLTTLTLLTLTTTLLLTLTLLLT
session 3: LLLLLLTTTLTTLTTTLLLTTTTTLTLLLT
session 4: LLLTTTLLTLLLLLTTTLTTLTTLTTTTLL
session 5: LLTTLLLLTLLLTLLTTTTTTTTLLLTLTT
session 6: TLTTLLLLLTLLTTTLTLLLLLTTTTLTTT
```

Each label has five preallocated reserve tokens. A technical abort consumes
one reserve for the same label and immediately repeats the current scheduled
slot without advancing or reordering the primary schedule. A technical abort
must use exactly one closed reason code:

- `pretrial_ffc_failed`
- `acquisition_process_exit`
- `realsense_disconnect`
- `lepton_stream_disconnect`
- `required_file_write_failed`
- `trial_timing_controller_failed`

Insufficient frames, active analysis-window FFC, landmark or ROI invalidity,
subjective instruction quality, and feature values are completed
modality-invalid outcomes and are never reserve-eligible. A completed
modality-invalid trial is not replaced.

If a session cannot reach 15 completed trials per class, retain the failed
attempt and permit one whole-session retry on another calendar day after a
full restart. The retry decision may use acquisition integrity only. A second
failure retires the study.

## Trial timing and frame attribution

| Stage | Duration | Contract |
| --- | ---: | --- |
| Manual FFC | telemetry-gated | Wait for `complete` and reset `since_last_ffc`; never infer completion from elapsed time. |
| Guard | 1.0 s | No pinch/contact. |
| Formation | 1.0 s | Form the instructed loose or tight pinch. |
| Hold | 3.0 s | Maintain the instructed condition. |
| Analysis | 2.0 s | Use hold interval `[1.0, 3.0)` only. |

The cameras are not hardware-synchronized. A host monotonic clock defines the
trial window, and each stream is aggregated independently within that window.
The only frame-level alignment claim is the RealSense device's aligned
RGB-depth frameset.

## Frozen frame and trial features

Exactly three scalar trial features are permitted:

| Feature | Frame value | Trial aggregate | Minimum valid frames |
| --- | --- | --- | ---: |
| `RS1` | thumb-index RGB distance / landmark 5-to-17 palm scale | median | 30 aligned RGB-depth framesets |
| `RS2` | absolute aligned thumb-index Z difference in metres | median | same aligned framesets as `RS1` |
| `delta_T` | pinch-window TLinear temperature minus adjacent hand-skin baseline | median | 9 thermal frames with no active FFC |

No other temporal summary or thermal feature is permitted.

## Trial and session validity

A trial is completed when the commanded sequence finishes, the label and
required files exist, and pre-trial FFC succeeded. A completed trial may still
be RealSense-invalid, thermal-invalid, or both.

RealSense is fixed to D435i serial `233522078685`, color
`640x480 BGR8 @ 30 Hz`, and depth `640x480 Z16 @ 30 Hz`, aligned to color.
The exact profile must produce at least 30 complete aligned framesets in the
2.0 s analysis window. Expected FPS and the minimum are immutable study
constants, not observed or session-configurable values. Repeat the exact
profile preflight before every session.

Thermal validity requires at least nine valid frames and no active FFC
telemetry during the analysis interval. The recorder must not infer FFC state
from elapsed time.

A session is valid only when it contains exactly 15 completed trials per
class. Completed modality-invalid outcomes remain in the denominators and are
not replaced.

## Frozen models and preprocessing

Use exactly two models:

- RealSense baseline: `RS1`, `RS2`;
- augmented model: `RS1`, `RS2`, `delta_T`.

Within each leave-one-session-out fold and separately for each model, compute
training-set z-scores using population standard deviation (`ddof=0`) and apply
those training statistics to the held-out session. If a training feature has
zero standard deviation, set that standardized feature to zero for the fold.

Use Python 3.12 and exactly `scikit-learn==1.9.0` with
`LogisticRegression` parameters:

```text
solver="lbfgs"
penalty="l2"
C=1.0
fit_intercept=True
max_iter=10000
tol=1e-8
class_weight=None
random_state=None
threshold=0.5
```

Predict `tight` when `p(tight) >= 0.5`; otherwise predict `loose`. A fold fails
if its training population lacks either class, the solver does not converge,
or coefficients or probabilities are non-finite. Its held-out predictions are
then missing.

## LOSO predictions and effective balanced accuracy

Run six-fold leave-one-session-out analysis. In each fold, the baseline trains
on RealSense-valid trials in the other five sessions. The augmented model
trains on trials valid for both modalities in those sessions.

For held-out class `c`, `N_c` is every completed trial of true class `c`, and
`C_c` is every completed trial correctly predicted as `c`. Missing predictions count as incorrect. Class accuracy is `C_c / N_c`; session effective balanced
accuracy (eBA) is the mean of loose and tight class accuracy. The primary
statistic is the mean of the six session eBAs.

## Session-blocked permutation test

Run exactly 19,999 label permutations with seed `164`. Shuffle labels
separately within each session while preserving that session's class counts,
and rerun the complete augmented-model LOSO pipeline for every permutation.
The one-sided p-value is exactly:

```text
(1 + number of permuted mean-session-eBA values >= observed mean-session-eBA) / 20000
```

## Paired McNemar diagnostic

For every completed trial, define baseline and augmented predictions as
correct or incorrect, treating missing predictions as incorrect. Report an
exact two-sided McNemar test. It is a secondary diagnostic, not a decision
gate, because trials are clustered within sessions.

## Deterministic GO/RETIRE rule

Let `eBA_baseline` and `eBA_augmented` denote the two held-out session eBAs.
A tie retires the IR path.

```text
if any session cannot reach 15 completed trials/class after its one allowed whole-session retry:
    RETIRE
run the fixed six-fold LOSO pipeline
if augmented blocked-permutation p >= 0.05:
    RETIRE
if any session has eBA_augmented <= eBA_baseline:
    RETIRE
GO
```

## Permitted pre-analysis quality control

Before the locked analysis, inspect only acquisition integrity, timestamps,
FFC status, file completeness, completed/abort counts, and invalid-reason
counts. Do not inspect feature distributions, feature-label relationships,
predictions, or performance.

## Required acquisition manifest

Before the first familiarization trial, create and hash an immutable study
contract containing the participant pseudonym, session indices, distinct-day
rule, schedules, feature and ROI contracts, units, exact device profiles and
serials, code/config hashes, and software versions. Do not add trial outcomes
to or mutate this contract.

For each session attempt, create a separate trial ledger recording the run and
session IDs, schedule slot and attempt/reserve index, label, monotonic start and
end, FFC telemetry, per-modality validity/reasons, raw artifact paths and
SHA256 values, and completed or technical-abort outcome.

The ledger schema validates content but cannot enforce append-only storage.
The later recorder must create the ledger once, append and fsync every attempt,
forbid in-place edits, and write a terminal seal hash. Pilot start is blocked
until those storage invariants have implementation tests. Validate the study
contract and ledger against their Draft 2020-12 schemas before use and validate
the sealed ledger again after completion. The ledger schema caps technical
aborts at five per label and rejects identical duplicate records. Before
sealing, also run the checked-in semantic validator; it binds every record to
the realized session schedule and rejects duplicate `(scheduled_slot,
attempt_index)` keys, skipped/reordered slots, nonconsecutive attempts, label
mismatches, and cross-slot exhaustion of either label's reserve budget:

```bash
cd $WORKSPACE_ROOT/webcam-input/.worktrees/ir-hand-pressure-so101-teleop/lerobot_teleoperator_so101_webcam
env -u PYTHONPATH $WORKSPACE_ROOT/.venv-lerobot/bin/python \
  validate_ir_hand_pinch_session_ledger.py \
  --ledger SESSION_LEDGER.json \
  --schema ir_hand_pinch_session_ledger.schema.json \
  --schedule ir_hand_pinch_trial_schedule.json
```
