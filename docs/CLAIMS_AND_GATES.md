# Claims and Gates

This is research. Most of what is here did not work, and the parts that did are
narrow. This file separates the three kinds of evidence and states plainly which
applies to what.

- **software** — tests, schema checks, compile checks. Says nothing about a camera.
- **locked inference** — offline analysis of recorded data under a frozen contract.
- **physical** — observed behaviour with a real camera and a real robot.

## What is established

**Software.** The estimators, packet contracts, feature extraction, dataset
readers, protocol validators, and calibration parsers have tests and pass them.

**Locked inference, with negative results.** The grip-force viability experiment
(`docs/IR_GRIP_FORCE_EXPERIMENT.md`, 2026-07-07) concluded:

- SO-101 `present_load` and `present_current` are usable as *relative* gripper
  effort proxies, but they are raw servo registers, not calibrated force. They
  jump during motion because they fold in friction, backlash, controller
  transients, and quantization.
- Fixed-level foam trials produced a **weak** IR-to-load relationship. The contact
  patch was too soft and unstable, and the block could move. The hard-block
  continuous sweep is the more informative protocol.

These negative results are the point of keeping them. Do not re-run the foam
fixed-level protocol expecting a different answer without changing the physical
setup first.

**Calibration.** The FLIR official Brown-Conrady intrinsics are the correct model;
a fisheye model is a non-bijective trap that silently produces plausible-looking
but wrong projections. A provisional RealSense↔Lepton extrinsic with |T| ≈ 2.7 cm
exists and can be improved by a joint solver plus depth validation, without
recapturing the extrinsic.

## What is NOT established

- **No calibrated force.** Nothing here converts a thermal signal into newtons.
- **No autonomous deployment.** No IR signal drives a robot. The gripper adapter
  in `ir_force/gripper_adapter.py` is a **shadow** controller: it produces a
  proposal and an audit trail, and does not actuate.
- **The hand-pinch loose/tight pilot never ran.** Its preregistration
  (`docs/IR_HAND_PINCH_PREREGISTRATION.md`) froze the acquisition and analysis
  contract and explicitly gated the pilot behind thermal-calibration and
  recorder-integrity checks. Treat every number associated with it as absent, not
  as pending-but-probably-fine.
- **`/dev/video21` is not radiometric.** The `flirone-v4l2` loopback is a
  colorized palette stream. Analyse baseline deltas in intensity, never Celsius.

## Preregistration discipline

The pinch pilot's preregistration forbids introducing any new feature, middle
class, automatic pose gate, imputation, pseudo-synchronization, dynamic threshold,
tuned model, or AUC gate **after acquisition**. If the analysis needs one of
those, the protocol is amended and versioned first, and the reason recorded — the
judgement is never adjusted to fit the result.

## Hardware status

**The Lepton hardware has been dead since 2026-07-17** — both SPI and I2C silent.
The `LeptonUDPSource`, blob-ROI, and blob-mode estimator work was completed and
tested in software (469 tests green at the time) but has **never** run against a
live Lepton since. Any Lepton claim here is software-only.

The FLIR ONE path depends on a locally patched `flirone-v4l2`; see
`hardware/flirone-v4l2/UPSTREAM.md` for the patch and its provenance.

## Migration note

This repository was assembled from two divergent branches during the 2026-09-01
repository split. Test results quoted from before that date were produced in the
old workspace layout, not in this tree. `docs/MIGRATION_AUDIT.md` records what was
re-verified here and what was inherited.
