# FLIR Radiometric Feasibility Report

**Date:** 2026-07-16  
**Audit branch:** `codex/flir-radiometric-feasibility`  
**Scope:** FLIR ONE USB bridge and the existing `/dev/video20` and `/dev/video21` loopback pipeline. No classifier training or hand/foam experiment was rerun.

## Final Decision

**FIXED-RANGE NON-RADIOMETRIC OUTPUT ONLY**

The branch now has a live-validated relative raw-count export, but only for one static 15-second scene. It can map the existing palette output from a fixed raw-count range, although that mapping has not yet been validated on a dynamic scene. FFC behavior, restart behavior, and every temperature conversion remain unverified. The conclusion does **not** authorize Celsius output or a force/pressure claim.

## What Is Known

| Question | Evidence | Result |
| --- | --- | --- |
| What is `/dev/video21`? | `v4l2-ctl --device=/dev/video21 --all` | 160x128 `RGB3` V4L2 loopback output; not a raw thermal device. |
| Does the bridge decode 16-bit samples before colorization? | `src/flirone.c` decoders and live 10,332-byte payload | Yes. The validated 10,332-byte path yields 80x60 little-endian samples; a >=39,360-byte path remains a 160x120 little-endian source-level path. |
| Is current RGB dynamically rescaled? | `src/flirone.c` default display mapping | Yes. Every decoded frame uses its own min/max before palette lookup. RGB intensity is therefore not comparable across frames as a temperature or raw-count scale. |
| Is there factory/per-device calibration in the audited path? | `src/plank.h`, `src/flirone.c`, no local `CameraFiles.zip` | No verified source. `plank.h` is documented as values from a FLIR ONE JPEG EXIF example and uses generic reflected temperature/emissivity assumptions. |
| Does the source implement a CameraFiles read path? | `src/flirone.c` `CameraFiles` case | It contains a command skeleton, but the state flow advances past that case; it is not an operational, verified calibration extractor. |

The currently connected USB device identifies as FLIR ONE `09cb:1996`, product revision `1.08`, with three vendor interfaces and bulk endpoints `0x81/0x02`, `0x83/0x04`, and `0x85/0x06`. This identifies a usable bridge transport, not a radiometric calibration guarantee.

## Current Pipeline

```text
FLIR USB endpoint 0x85 payload
  -> bridge decodes uint16 sensor samples
  -> per-frame min/max normalization (current default)
  -> palette conversion and text/marker overlay
  -> /dev/video21: 160x128 RGB3
  -> existing hand/foam recorder
```

The recorder receives the final colorized RGB output. Its historical PNGs cannot be converted back into raw counts or calibrated temperature.

## Prototype Added on This Branch

`src/flirone.c` now adds only opt-in functionality around the existing V4L writing path:

```bash
./flirone palettes/Iron2.raw \
  --raw-dir validation_capture/little_endian_corrected \
  --raw-frame-limit 180
```

For each decoded thermal frame, the prototype writes:

```text
validation_capture/little_endian_corrected/raw_frame_000000.u16le
validation_capture/little_endian_corrected/raw_frame_000000.json
```

The raw file holds the original decoded sensor grid, serialized explicitly as little-endian `uint16` values. The metadata records frame index, monotonic host timestamp, source payload byte order, thermal payload size, dimensions, raw min/median/max, FFC state, repeated-frame flag, dropped-frame observability, calibration status, and display-mapping mode. `camera_timestamp` is explicitly `null` because this bridge payload path does not expose one.

The prototype supports a fixed display range without claiming temperature:

```bash
./flirone palettes/Iron2.raw \
  --fixed-raw-low LOW_COUNT \
  --fixed-raw-high HIGH_COUNT
```

`LOW_COUNT` and `HIGH_COUNT` must be selected from a prior, valid raw capture; they are sensor-count bounds, not degrees Celsius. Without these options, the existing per-frame dynamic normalization is retained. The default overlay now shows raw min/center/max counts rather than unvalidated Celsius. The legacy generic Celsius overlay is available only by explicit opt-in and is not valid for experiments.

## Live Raw-Capture Result

The corrected live capture is stored in `validation_capture/little_endian_corrected/` and was checked by `capture_raw_validation.py`.

| Metric | Result | Interpretation |
| --- | --- | --- |
| Stored frames | 180 JSON/raw pairs | Every raw file had the expected 9,600 bytes for 80x60 `uint16`. |
| Byte order and statistics | All metadata and bytes agreed | The source payload and exported raw array are little-endian. |
| FFC state | 180 normal, 0 FFC | This run did not test FFC resilience. |
| Consecutive repeated frames | 51/180 (28.3%) | Use `repeated_frame_flag` to deduplicate before any temporal analysis. |
| Unique normal frames | 129 over 14.98 s | About 8.6 unique frames/s, consistent with the expected low-rate Lepton stream. |
| Unique-frame raw median | 3495 to 3517 counts | Stable relative response in this stationary scene. |
| Unique-frame median standard deviation | 3.83 counts | About 0.11% of the mean 3504.4 counts. |
| Linear median drift | -0.267 counts/s | Small over this short static observation only. |

The initial directory, `validation_capture/raw/`, is retained as an audit artifact but rejected for science use. The first version read each Lepton2 word as big-endian. Its apparently large signal drift disappeared when those stored words were byte-swapped; the decoder and metadata were then corrected to little-endian and the capture was repeated. The corrected values are consistent with [Teledyne FLIR's statement that Lepton digital output is 14-bit raw](https://oem.flir.com/support/support-center/knowledge-base/how-many-bits-per-pixel-is-the-digital-output/), but that alone does not demonstrate a calibrated temperature relationship.

The corrected stream clears the minimum evidence bar for a stable *relative* raw-count signal. It does not clear a radiometric or control-readiness bar because it contains no reference-temperature sweep, FFC event, restart event, or dynamic hand/foam scene. Fixed mapping bounds must be chosen from a representative dynamic raw capture, not from this narrow stationary range (`p0.5=3477`, `p99.5=3544`).

## Offline Validators

`capture_raw_validation.py` rejects malformed data rather than silently accepting it. It verifies raw file size, `uint16` little-endian metadata, stored statistics, FFC state, repeated-frame flags, and mapping mode.

```bash
python3 capture_raw_validation.py \
  --raw-dir validation_capture/little_endian_corrected \
  --expected-width 80 \
  --expected-height 60 \
  --summary-path validation_capture/little_endian_raw_validation_summary.json
```

`validate_radiometry.py` is deliberately conservative. Given independently measured reference temperatures and raw medians, it reports ordering and a linear empirical fit, but always marks factory radiometry as unvalidated.

```bash
python3 validate_radiometry.py \
  --references-csv reference_measurements.csv \
  --summary-path reference_validation_summary.json
```

Required CSV columns are `reference_c,raw_median`. Do not use the fit as a temperature calibration without repeated FFC/restart checks and controls for emissivity, background radiation, distance, and ambient conditions.

## Remaining Live Validation

The raw stream itself has now been tested. These remaining checks are required before widening the claim:

1. Trigger at least one FFC/shutter event and repeat the stationary scene. Compare pre-FFC, FFC, and post-FFC medians; keep the explicit FFC tags in the analysis.
2. Restart the bridge and repeat the same scene. Compare the restart offset and drift with the short-run standard deviation above.
3. Record a representative dynamic hand-and-foam scene in raw mode. Determine pre-registered `LOW_COUNT` and `HIGH_COUNT` from its deduplicated distribution, then verify that the fixed range remains stable when another hot/cold object enters the frame.
4. Use at least five independently measured reference surfaces. Record repeated holds at each reference, then rerun after FFC and restart.
5. Only after those checks, assess an empirical temperature conversion with `validate_radiometry.py`; it remains an empirical conversion unless a verified per-device calibration source is extracted and independently validated.

Until then, supported use is relative raw-count inspection with duplicate-frame removal. It is not temperature, force, or pressure estimation.

## Repeatability Tooling Implemented

The next live-validation phase is implemented but has not been run. The
protocol and exact commands are in `RAW_COUNT_REPEATABILITY_PLAN.md`.

- `record_raw_repeatability_events.py` creates a fresh run directory, prints
  the separately executed privileged bridge command, saves RGB display frames,
  and records run/phase timestamps with `CLOCK_MONOTONIC`-compatible Python
  timestamps.
- `analyze_raw_repeatability.py` ignores `ffc`, post-FFC discarded, and repeated
  raw frames when computing ROI metrics. It reports FFC pre/post recovery,
  restart offsets, hot-hand target/control phase deltas, and a raw-count fixed
  display-range suggestion from the dynamic capture.
- The recorder and analyzer deliberately do not start hardware, calculate
  temperature, or train a classifier. Their output is a repeatability gate for
  the later classifier plan.

The implementation must not be interpreted as evidence that FFC, restart, or
dynamic-scene stability has passed; those fields remain unmeasured until the
live protocol is completed.

## Commands and Verified Outputs

```text
$ make -B
gcc -I/usr/include/libusb-1.0 -c -o src/flirone.o src/flirone.c
gcc -I/usr/include/libusb-1.0 -o flirone src/flirone.o -lusb-1.0 -lm -Wall

$ python3 -m pytest -q tests
27 passed in 0.18s

$ v4l2-ctl --device=/dev/video20 --all
Card type: FLIR_ONE_VISIBLE; Video Output; 640x480 MJPG

$ v4l2-ctl --device=/dev/video21 --all
Card type: FLIR_ONE_THERMAL; Video Output; 160x128 RGB3

$ python3 capture_raw_validation.py --raw-dir validation_capture/little_endian_corrected --expected-width 80 --expected-height 60
raw_stream_accepted: true
frame_count: 180; deduplicated_normal_frame_count: 129
raw median: 3495..3517; population standard deviation: 3.83 counts
repeated_frame_count: 51; ffc_state_counts: {normal: 180}
```

The live raw files are intentionally Git-ignored as camera data. This report and `validation_capture/little_endian_raw_validation_summary.json` record their reproducible validation result.

## Rollback

The original `tools/flirone-v4l2` worktree was not modified by this audit. This prototype lives only in the separate worktree and branch. To discard it later, from the original repository:

```bash
git worktree remove /home/zhuokai/hand-teleop/tools/flirone-v4l2-radiometric-audit
git branch -D codex/flir-radiometric-feasibility
```

No rollback action was performed during this audit.
