# Local Architecture, Data Inventory, and Computation Pipeline

Last audited: 2026-07-15

## Purpose and Current Decision

The active research question is not force estimation in Newtons. The intended
output is a conservative binary decision for a hand squeezing a fixed foam
object: hard press / not hard press.

The proposed reference label is actual geometric foam compression, not a
subjective squeeze prompt and not a measured force. OAK RGB tracks two black
markers on the foam. FLIR ONE provides a colorized thermal image that may or
may not contain a useful slow proxy for compression.

The local evidence is not sufficient to deploy a classifier. The only complete
fixed-geometry recording is rep08. It can diagnose the next data collection
design, but cannot show generalization across recordings, days, people, camera
placements, or foam pieces.

## Relevant Workspace Structure

/home/zhuokai/hand-teleop is a meta-workspace, not a Git repository itself.
The active implementation is the nested repository below.

~~~text
/home/zhuokai/hand-teleop/
├── project.md
├── tools/flirone-v4l2/
│   └── palettes/Iron2.raw                 # FLIR RGB palette
├── datasets/
│   ├── ir_foam_compression/               # Current fixed-geometry experiment
│   │   ├── preflight/                     # Frozen setup images and reports
│   │   └── trials/
│   │       └── foam-compression_...rep08/ # Only complete geometry-labelled run
│   ├── ir_hard_classifier/                # Earlier subjective-prompt pilots
│   ├── ir_hand_pressure_hysteresis/       # Earlier hand-squeeze temporal pilot
│   ├── ir_hand_pressure_viability/        # Earlier foam/brick viability pilot
│   └── ir_grip_force_viability/           # Separate robot-gripper experiment
├── exports/
│   └── gpt_pro_ir_foam_rep08_review_20260715/
│       ├── rep08_gpt_pro_review_bundle.zip
│       └── materials/
└── webcam-input/lerobot_teleoperator_so101_webcam/
    ├── FOAM_COMPRESSION_EXPERIMENT.md
    ├── IR_FOAM_CLASSIFIER_LOCAL_ARCHITECTURE.md
    ├── view_ir_foam_setup.py               # Interactive dual-camera ROI preview
    ├── record_ir_foam_compression_experiment.py
    ├── analyze_ir_foam_compression.py
    ├── prepare_gpt_pro_rep08_review.py
    ├── record_ir_hard_classifier_experiment.py
    ├── analyze_ir_hard_press_classifier.py
    ├── analyze_ir_oak_squeeze_proxy.py
    └── lerobot_teleoperator_so101_webcam/
        ├── ir_capture.py                   # OpenCV and OAK camera sources
        ├── ir_dataset.py                   # Trial paths, metadata, CSV helpers
        ├── ir_features.py                  # Palette-index and IR features
        └── ir_foam_compression.py          # Marker, geometry, ROI, protocol helpers
~~~

## Local IR Data Inventory

| Dataset | Local size | Trials | Role and current status |
|---|---:|---:|---|
| datasets/ir_foam_compression | 5.9 GiB | 10 | Fixed foam geometry experiment. Nine runs are incomplete diagnostic attempts. rep08 is the only completed protocol. |
| datasets/ir_hard_classifier | 2.2 GiB | 2 | Earlier prompt-labelled hard/not-hard pilot. Labels are subjective 0/25/50/75 percent prompts, not measured compression or force. |
| datasets/ir_hand_pressure_hysteresis | 65 MiB | 4 | Earlier whole-hand foam sweep. Useful for temporal failure diagnosis, but no geometric label. |
| datasets/ir_hand_pressure_viability | 558 MiB | 10 | Earlier hand foam/brick viability recordings. Mixed framing and no objective compression reference. |
| datasets/ir_grip_force_viability | 2.0 GiB | 31 | Separate robot-gripper IR experiment. It is not hand data and must not be pooled with the current hand classifier. |

### Earlier Direct Classifier Pilot

ir_hard_classifier contains:

- hard-classifier_s01_fixed-posture_foam_zk_rep01: bird-camera era setup.
- oak-squeeze_s01_fixed-posture_foam_zk_rep02: later OAK setup.

Its analysis uses only the final one second of a target hold, drops frozen
frames and AGC jumps, and keeps entire sequences together for validation. It
defines hard as prompted level >= 70 percent and not-hard as prompted level
<= 50 percent. Results were weak:

- best blocked balanced accuracy within a trial: 0.625 and 0.667;
- best train-one-trial/test-the-other balanced accuracy: 0.625;
- all-feature nearest-centroid performance at or near chance cross-trial.

These data motivate replacing subjective labels with marker-measured geometry.
They are intentionally not merged with rep08 because camera setup and thermal
analysis regions differ.

### Fixed-Geometry Experiment Attempts

The ir_foam_compression directory preserves failed attempts instead of silently
deleting them.

| Trial | Complete protocol steps | Invalid actions | Interpretation |
|---|---:|---:|---|
| foam-compression_foam-20260715_foam_zhuokai_rep01 | 0 | 2 | Early gate failure. |
| foam-compression_foam-20260715retry1_foam_zhuokai_rep01 | 0 | 2 | Early marker/gate failure. |
| foam-compression_foam-20260715retry2_foam_zhuokai_rep01 | 0 | 0 | Stopped during d0 calibration. |
| retry2 rep02 | 2 | 0 | Partial protocol. |
| retry2 rep03 | 3 | 4 | Partial protocol. |
| retry2 rep04 | 2 | 2 | Partial protocol. |
| retry2 rep05 | 8 | 2 | Partial protocol. |
| retry2 rep06 | 8 | 3 | Partial protocol. |
| retry2 rep07 | 0 | 0 | Stopped at d0 calibration. |
| foam-compression_foam-20260715retry2_foam_zhuokai_rep08 | 43 | 5 | Only complete fixed protocol; primary current data. |

The incomplete trials are retained for hardware and gating diagnosis, but are
not independent classifier evidence.

## Rep08 File Layout and Data Contents

~~~text
datasets/ir_foam_compression/trials/
└── foam-compression_foam-20260715retry2_foam_zhuokai_rep08/
    ├── metadata.json
    ├── telemetry.csv
    ├── frame_features.csv
    ├── events.csv
    ├── thermal/frame_000000.png ... frame_005736.png
    ├── oak_rgb/frame_000000.png ... frame_005736.png
    ├── oak_depth/frame_000000.png ... frame_005736.png
    ├── analysis/
    │   ├── foam_compression_primary_summary.png
    │   ├── foam_compression_step_summary.csv
    │   ├── foam_compression_summary.json
    │   ├── fixed_geometry_binary_screen.csv
    │   └── fixed_geometry_binary_screen.json
    ├── preflight/                           # Generic trial directory
    ├── overlays/
    ├── plots/
    ├── bird/                                # Not used by rep08
    └── flir_visible/                        # Not recorded by rep08

datasets/ir_foam_compression/preflight/
└── foam-compression_foam-20260715retry2_foam_zhuokai_rep08/
    ├── thermal.png                          # Frozen thermal ROIs overlaid
    ├── oak_rgb.png                          # OAK setup snapshot
    ├── oak_markers.png                      # OAK marker detection overlay
    └── preflight_report.json
~~~

| Item | Count or size |
|---|---:|
| Thermal PNG frames | 5,737, about 158 MiB |
| OAK RGB PNG frames | 5,737, about 2.7 GiB |
| OAK depth PNG frames | 5,737, about 49 MiB |
| Synchronized frame indexes | 5,737, 000000 through 005736 |
| telemetry.csv rows | 5,737 data rows |
| frame_features.csv rows | 5,737 data rows |
| events.csv rows | 146 data rows |
| Valid deduplicated stable-hold frames in analysis | 3,138 |
| Final valid protocol steps | 43 |
| Trial storage | about 2.9 GiB |

record_flir_visible was false for rep08, so FLIR visible imagery is not
evidence in this recording.

## Hardware and Physical Setup

### Cameras

- FLIR ONE thermal stream: /dev/video21.
- Optional FLIR visible stream: /dev/video20; not recorded in rep08.
- OAK-D RGB: 640 x 480, with aligned depth.
- Target capture rate: 10 Hz for FLIR and OAK.

The FLIR output is colorized RGB, not raw radiometric temperature. Iron2.raw
maps RGB pixels to nearest palette indices, producing a relative intensity
proxy only.

### Foam Geometry Reference

The foam is upright and constrained only near its lower/back side. Two white
marker tabs with black dots are placed near its upper compressed sides. OAK
detects the dots within two fixed RGB regions. Let d0 be their released
distance after calibration and d be the current distance:

~~~text
compression_pct = 100 * (d0 - d) / d0
~~~

This measures width reduction, not mechanical force. The proposed future labels
are:

~~~text
not hard: compression <= 10 percent
hard:     compression >= 25 percent
ambiguous: 10 to 25 percent, excluded
~~~

### Fixed Thermal Regions

ROIs were chosen before formal capture and stored in metadata.json:

~~~text
foam bounding box:     68,40,28,30
foam center ROI:       75,48,14,18          # primary analysis region
left contact ROI:      68,48,6,18
right contact ROI:     90,48,6,18
background ROI:        5,5,15,15
room reference ROI:    15,15,12,12
warm reference ROI:    130,15,12,12
OAK left marker ROI:   180,90,150,140
OAK right marker ROI:  360,100,80,100
~~~

The foam must span at least 24 thermal pixels. All thermal regions are above
the lower text-overlay boundary y=105. The two reference patches reduce
palette-level/span changes but do not create a temperature measurement.

## Capture Pipeline

### Preflight

view_ir_foam_setup.py is used before recording to inspect FLIR and OAK together
and freeze the ROIs. Formal preflight records snapshots and checks:

- foam bbox lies inside the 160 x 128 thermal image, near center (80, 60), and
  is at least 24 pixels wide;
- both marker dots are detected after two seconds of camera settling;
- no marker loss persists for more than 0.5 seconds during preflight;
- absolute warm-room reference span is at least 5 palette bins;
- no automatic preflight issue is recorded;
- manual checks exclude face, torso, a second hand, reflective objects, and
  thermal overlay contamination.

Rep08 metadata reports no automatic preflight issues.

### Released Calibration and States

Before the action protocol, R means fully released: fingers are at least about
10 mm from foam, but the hand stays in frame. Ten seconds are captured. The
first two seconds are excluded from d0; the remaining marker-distance p95-p5
spread must be at most 5 percent of its median. Rep08 d0 is 149.4123 px.

~~~text
R   released, no contact
N   2-3 mm near foam, no contact (hand-presence control)
C0  just touching, <= 2 percent compression
C10 10 percent geometric compression
C20 20 percent geometric compression
C30 30 percent geometric compression
~~~

### Fixed Protocol

The full plan contains 43 prompted steps:

1. start_drift: 30 s in R.
2. steady_state: ten randomized target states, each preceded by R. R and
   targets hold for 6 s. Targets repeat N, C0, C10, C20, and C30.
3. middle_drift: 20 s in R.
4. hysteresis: C0 -> C10 -> C20 -> C30 -> C20 -> C10 -> C0 -> R. C30 holds
   6 s, other contact states 4 s, final R 12 s.
5. release_pulses: four repetitions of R (8 s) -> C30 (6 s) -> R (12 s).
6. end_drift: 30 s in R.

For C0/C10/C20/C30, OAK must first remain in the target band for one continuous
second before a stable hold begins. The non-release tolerance is plus/minus
3 percentage points. R has a looser plus/minus 5 percentage-point tolerance
because foam recovery can be imperfect; recorded geometric compression is never
forced to zero. During a hold, one out-of-band gap up to 0.5 s is tolerated;
persistent loss invalidates and retries the action. Rep08 completed 43 actions
and recorded 5 invalid retries.

## Per-Frame Computation and Stored Tables

For every shared frame index, the recorder writes raw files and synchronized
rows to telemetry.csv and frame_features.csv.

### Geometry and Hand Fields

telemetry.csv includes:

- true capture and camera timestamps: t_capture, t_thermal, t_oak;
- protocol block, phase, state, target, sequence, step, and attempt;
- marker detection, both marker centers, marker distance, d0_px, and
  compression_pct;
- foam midpoint and rotation derived from marker geometry;
- marker depth values when available;
- OAK hand detection, hand area, hand center, hand-to-foam gap, and hand
  aperture proxies;
- gate state and a repeated-frame flag.

The RGB marker distance is the principal reference. OAK depth is saved as a
secondary record, not the only compression label.

### Thermal Feature Fields

The thermal RGB frame is mapped to a scalar palette-index image using the
fixed Iron2 palette. For each frozen ROI the recorder stores its palette median:

~~~text
foam_center_median
left_contact_median
right_contact_median
background_median
room_reference_median
warm_reference_median
reference_span = warm_reference_median - room_reference_median
~~~

The primary feature is:

~~~text
foam_center_norm =
  (foam_center_median - room_reference_median) / reference_span
~~~

Equivalent normalized contact and background features are stored. Every row
contains thermal_frame_sha1 and frozen_frame_flag, allowing duplicate thermal
frames to be removed later. This feature family is reference-normalized palette
position, not absolute temperature and not a force value.

### Event Audit

events.csv records d0_start, d0_complete, prompts, gate starts, completed
steps, and invalid attempts. Offline analysis excludes all rows from an action
attempt marked invalid. This prevents treating a prompted but unreached
compression as a correct label.

## Offline Analysis Pipeline

Primary command:

~~~bash
cd /home/zhuokai/hand-teleop/webcam-input/lerobot_teleoperator_so101_webcam
env -u PYTHONPATH /home/zhuokai/hand-teleop/.venv-lerobot/bin/python \
  analyze_ir_foam_compression.py \
  --trial /home/zhuokai/hand-teleop/ir-camera-force/local/datasets/ir_foam_compression/trials/foam-compression_foam-20260715retry2_foam_zhuokai_rep08
~~~

The analysis is restricted before examining results:

1. Use only phase == stable_hold rows.
2. Exclude attempts marked invalid in events.csv.
3. Exclude missing markers and frozen frames.
4. Deduplicate remaining thermal data using thermal_frame_sha1.
5. Group by protocol action/attempt.
6. Summarize the final 3 seconds of each valid hold.
7. Compare actual marker compression with the pre-registered primary
   foam_center_norm feature.

Outputs are:

- foam_compression_primary_summary.png: OAK compression versus primary feature
  plus state distributions;
- foam_compression_step_summary.csv: one stable summary per valid action;
- foam_compression_summary.json: trial-level descriptive statistics;
- fixed_geometry_binary_screen.csv/json: auxiliary binary screen using the
  <=10 percent and >=25 percent labels.

The binary screen is a rank AUC on the same single recording. It is not
cross-validated or cross-trial classifier accuracy.

## Current Rep08 Result

| Quantity | Value |
|---|---:|
| Actual compression versus foam_center_norm Spearman rho | -0.4151 |
| Spearman p value | 0.00564 |
| Mean C30 minus C0 primary feature | -2.8397 |
| Usable binary-screen steps | 36 |
| Hard steps, >=25% compression | 7 |
| Not-hard steps, <=10% compression | 29 |
| Same-recording rank AUC, lower IR feature means hard | 0.7241 |

The apparent inverse direction is configuration-specific. It must not be
described as more compression produces more heat. Colorized FLIR behavior,
hand visibility, reference-span behavior, contact transfer, and slow thermal
history can all affect sign and magnitude.

## Known Limitations and Risks

1. Only one complete fixed-geometry recording exists. There is no held-out
   recording for classifier validation.
2. FLIR is colorized RGB, not raw radiometric temperature. Palette conversion
   and reference normalization mitigate but cannot eliminate AGC/span drift.
3. OAK views the front of foam while FLIR has a slightly different view. Marker
   distance is a planar proxy, not a full deformation field.
4. The hand remains in the FLIR view. N is a no-contact hand-presence control,
   but one recording cannot prove hand contour or thermal history is not
   driving the relation.
5. The response can be path-dependent. Hysteresis and release-pulse blocks were
   recorded because a memoryless frame-to-class mapping can fail during release.
6. A p value and same-recording AUC are not a generalization result. Random
   frame splitting is forbidden because adjacent frames are highly correlated.
7. Earlier data have mixed camera framing and labels; they must not be pooled
   without a fixed, justified harmonization plan.

## GPT Pro Package and Future-Plan Request

The external package intentionally contains only rep08, not all local raw data:

~~~text
exports/gpt_pro_ir_foam_rep08_review_20260715/
├── rep08_gpt_pro_review_bundle.zip
└── materials/
    ├── LOCAL_PROJECT_ARCHITECTURE_AND_DATA_PIPELINE.md
    ├── README_FOR_GPT_PRO.md
    ├── GPT_PRO_REVIEW_REQUEST.md
    ├── tables/                           # All rep08 processed CSV/JSON tables
    ├── analysis/                         # All rep08 outputs
    ├── preflight/                        # Setup snapshots and report
    ├── selected_raw/                     # 12 synchronized raw triplets
    └── visuals/                          # Two FLIR/OAK RGB/OAK depth contact sheets
~~~

The raw subset covers released baseline, N, C0, C10/C20/C30 steady states, and
loading/unloading hysteresis. Full rep08 raw data remain local because they
occupy about 2.9 GiB.

Based on this local architecture and rep08-only evidence, design a minimal next
experimental plan that can decide whether a hard/not-hard classifier is viable
without a force sensor. Specify:

1. Whether marker compression is an adequate label and how to define it.
2. The number of independent recordings, re-placements, and participants.
3. A frozen protocol for calibration, R/N/C0 controls, compression states,
   hysteresis, and release dynamics.
4. The raw data and frame/step-level features to retain.
5. Quality gates that reject a recording before analysis.
6. A blocked validation split that prevents temporal and trial leakage.
7. Decision thresholds for continuing, changing the proxy, or stopping this
   classifier direction.

Do not propose random-frame train/test splits or force labels that do not
exist. Distinguish what can be inferred from one trial from what requires
replication.
