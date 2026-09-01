# Fixed-Geometry Foam Compression Experiment

This experiment tests whether the FLIR signal can serve as a slow proxy for
foam compression. It does not estimate hand force in newtons.

## Before Recording

1. Rigidly mount FLIR and OAK. Keep the views close to frontal and fixed.
2. Mount the same foam upright: only the lower/back portion may be constrained.
   Keep its two compressed sides and the central front face unobstructed.
3. Add one white marker tab with a black dot near each top side. The dots must
   remain visible to both OAK and the FLIR visible RGB stream through every
   compression level. Keep each dot near the centre of a small, tight RGB ROI;
   do not let the black foam enter that ROI.
4. Put a room-temperature and a 5--8 C warmer matte-black reference patch in
   the FLIR upper corners. Avoid saturation. Do not move either reference.
5. Exclude face, hair, torso, second hand, hot objects, metal reflections, and
   the FLIR lower text band from the thermal image.
6. Keep the released hand in frame. `R` means both fingers are at least 10 mm
   away; `N` means 2--3 mm away without touching; `C0` means just touching.
7. Before formal data, make one 30% compression for 5 seconds, then release for
   30 seconds. If the marker distance does not return to `d0` within plus/minus
   1%, lower the highest target to 20% and record the residual recovery. `R`
   still means no finger contact, but may retain up to 5% measured geometric
   compression from foam recovery; the actual OAK compression remains in every
   row and is used for analysis.

The foam must span at least 24 thermal pixels. Put its bounding box center near
thermal `(80, 60)` and keep every analysis ROI above `y=105`. The baseline
thermal ROIs are selected before recording. In the RGB-marker mode below, only
the foam-attached ROIs move per frame; the background and two reference patches
remain fixed.

## Live Layout Check

Close any other OAK preview first, then use the live layout tool before a
preflight capture:

```bash
cd /home/zhuokai/hand-teleop/webcam-input/lerobot_teleoperator_so101_webcam

env -u PYTHONPATH QT_QPA_PLATFORM=xcb \
  /home/zhuokai/hand-teleop/.venv-lerobot/bin/python \
  view_ir_foam_setup.py
```

It shows FLIR thermal and OAK RGB together. Select an ROI with a number, then
drag its rectangle in the matching window:

```text
1 foam bounding box       2 foam center        3 left contact
4 right contact           5 background         6 room reference
7 warm reference          8 left OAK marker    9 right OAK marker
s save layout/snapshots   q quit
```

The tool continuously shows black-marker detection and the reference span. Wait
until the marker rate is stable and the thermal status is `thermal ready`, then
press `s`. It writes `setup_layout.json`, `thermal_setup.png`, and
`oak_setup.png` under `datasets/ir_foam_compression/setup/`, and prints the
frozen recorder ROI arguments. Use those printed arguments in the preflight and
formal recording commands below.

## Preflight Preview

From this directory, run the command below after replacing the OAK marker ROIs
and thermal ROIs using the live preview. The numbers are only starting values;
they are not valid until the saved images show the actual setup correctly.

```bash
env -u PYTHONPATH /home/zhuokai/hand-teleop/.venv-lerobot/bin/python \
  record_ir_foam_compression_experiment.py \
  --session-id S01 --participant-id ZK --object-id foam --rep auto --recording-index 1 \
  --thermal-foam-bbox 68,40,28,30 \
  --thermal-foam-roi 75,48,14,18 \
  --thermal-left-contact-roi 68,48,6,18 \
  --thermal-right-contact-roi 90,48,6,18 \
  --thermal-background-roi 5,5,15,15 \
  --thermal-room-reference-roi 15,15,12,12 \
  --thermal-warm-reference-roi 130,15,12,12 \
  --oak-left-marker-roi 180,90,150,140 \
  --oak-right-marker-roi 360,100,80,100 \
  --thermal-roi-tracking flir-visible-markers \
  --flir-visible-left-marker-roi 650,375,45,50 \
  --flir-visible-right-marker-roi 1005,325,50,45 \
  --preflight-only
```

The two FLIR-visible marker ROIs above were verified on the current saved RGB
view: left dot near `(674,399)`, right dot near `(1017,346)`. They are only a
starting point. Re-run preflight whenever the FLIR, foam, tabs, or fixture
move. The recorder uses a tight-ROI, centre-prior dark-dot detector for FLIR
RGB because that stream does not preserve a sufficiently bright white surround
around the dots.

It saves these files under
`/home/zhuokai/hand-teleop/ir-camera-force/local/datasets/ir_foam_compression/preflight/`:

- `thermal.png`: frozen thermal analysis areas.
- `oak_rgb.png` and `oak_markers.png`: marker visibility and detection.
- `flir_visible.png`: RGB marker ROIs and their detected dot centres.
- `preflight_report.json`: technical checks and the manual checklist.

Do not start formal recording until the report has no automatic issues and the
two images confirm the physical checklist. Upload `thermal.png` and
`oak_rgb.png` for review before the first formal recording.

The room and warm reference medians must differ by at least five palette-index
bins. This is a minimum normalization check, not a temperature calibration.

## Formal Recordings

Use exactly the same ROI arguments that passed preflight. Keep the cameras,
foam fixture, reference patches, lighting, and frozen ROIs fixed across all
three recordings. Each recording is still independent in time: stop and restart
the program, fully release the hand, rest at least three minutes, confirm the
thermal baseline and marker distance have recovered, and recalibrate `d0`
before each one. Leave hand-geometry tracking enabled for formal data.

```bash
env -u PYTHONPATH /home/zhuokai/hand-teleop/.venv-lerobot/bin/python \
  record_ir_foam_compression_experiment.py \
  --session-id S01 --participant-id ZK --object-id foam --rep auto --recording-index 1 \
  --thermal-foam-bbox 68,40,28,30 \
  --thermal-foam-roi 75,48,14,18 \
  --thermal-left-contact-roi 68,48,6,18 \
  --thermal-right-contact-roi 90,48,6,18 \
  --thermal-background-roi 5,5,15,15 \
  --thermal-room-reference-roi 15,15,12,12 \
  --thermal-warm-reference-roi 130,15,12,12 \
  --oak-left-marker-roi 180,90,150,140 \
  --oak-right-marker-roi 360,100,80,100 \
  --thermal-roi-tracking flir-visible-markers \
  --flir-visible-left-marker-roi 650,375,45,50 \
  --flir-visible-right-marker-roi 1005,325,50,45
```

For recordings 2 and 3, keep `--rep auto` and only change the recording index:

```text
recording 2: --recording-index 2
recording 3: --recording-index 3
```

The recorder shows and saves a five-second preflight; its initial two seconds
are excluded from marker-rate checks while OAK auto-exposure settles. It then
records a 10-second fully released `d0` calibration. Its first 2 seconds are
retained as raw data but excluded from the `d0` calculation; the final 8 seconds
must still pass the fixed stability criterion. The current black-dot detector
uses `--marker-max-gray 110`, which separates the dots from the gray paper tabs
in this setup. Each
non-release target begins its analysis hold only after OAK marker compression
is inside its plus/minus 3 percentage-point range for 1 continuous second. `R`
uses a plus/minus 5 percentage-point release criterion to accommodate measured
foam recovery after a hold; it never relabels the recorded compression as zero.
A hold tolerates one continuous out-of-range OAK gap of up to 0.5 seconds, but
persistent deviations are marked invalid and retried. Invalid attempts are
excluded by the analysis script.

## Output and Analysis

Each formal trial saves raw thermal PNG, FLIR visible RGB PNG, OAK RGB PNG, OAK
depth PNG, host capture timestamps, marker geometry, hand geometry, per-frame
registration diagnostics, active thermal ROI coordinates, thermal features, and
events under `datasets/ir_foam_compression/trials/<trial-id>/`.

Analyze one completed trial with:

```bash
env -u PYTHONPATH /home/zhuokai/hand-teleop/.venv-lerobot/bin/python \
  analyze_ir_foam_compression.py \
  --trial /home/zhuokai/hand-teleop/ir-camera-force/local/datasets/ir_foam_compression/trials/foam-compression_s01_foam_zk_rep01
```

The pre-registered primary outcome is `foam_center_norm`: the median palette
position in the RGB-marker-registered foam-centre ROI, normalized by the fixed
room and warm reference patches. The RGB-to-thermal mapping is an explicit
frame-normalized approximation under the current near-coincident-lens
assumption, not a calibrated camera transform. The analyzer uses actual OAK
compression, drops duplicate thermal frames, and uses only the last three
seconds of valid gated holds.
