# flirone-v4l2 — upstream and local provenance

**Upstream:** https://github.com/fnoop/flirone-v4l2.git
**Base commit:** `b07e3a1` ("Add README"), branch `master`
**Migrated from:** `tools/flirone-v4l2` (path relative to the pre-split
`hand-teleop` workspace root)
**Migrated on:** 2026-09-01

## What is here that upstream does not have

The checkout was **dirty at migration time**, and the uncommitted work is the
part that actually matters — it is what makes a FLIR ONE Gen 2 produce usable
thermal frames on this rig:

* `src/flirone.c` — 93 insertions / 11 deletions over `b07e3a1`:
  * `VIDEO_DEVICE1`/`VIDEO_DEVICE2` moved from `/dev/video2`/`/dev/video3` to
    `/dev/video20`/`/dev/video21`, so the FLIR loopback devices stop colliding
    with the workspace camera on `/dev/video2`;
  * `decode_g2_thermal()` plus `read_le16`/`read_be16` — the Gen 2 thermal frame
    is split into two half-width halves with different byte order, which the
    upstream decoder does not handle;
  * `thermal_layout_reported` one-shot layout logging.
* `src/flirone_debug.c` — untracked debug utility, migrated as source only.

**This patch was never committed upstream and exists nowhere else.** It was
carried as working-tree state in a clone whose `.git` is not migrated, so this
directory is now its only home.

## Deliberately excluded

* `.git/` — upstream history stays with the upstream URL above.
* `flirone`, `flirone_debug` — compiled x86-64 binaries, rebuildable via `make`.
* `src/flirone.o` — build output.

`palettes/` **is** kept: `experiments/classifier/record_ir_foam_compression_experiment.py`
loads `palettes/Iron2.raw` by path.

## Relationship to `../flirone-v4l2-radiometric-audit`

That is the same upstream on a different branch, and its `src/flirone.c` is a
*divergent* variant of the same patched decoder (1198 lines vs 884 here). They
are kept as sibling trees rather than merged — see `docs/SOURCE_MAP.md`.
