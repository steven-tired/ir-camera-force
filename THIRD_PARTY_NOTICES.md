# Third-party code

Four trees here are third-party, carried in-tree at a recorded upstream commit
rather than as submodules, because in three of the four cases the part that
matters is uncommitted work that exists in no repository. Each `hardware/` tree
keeps its upstream `LICENSE` and carries an `UPSTREAM.md` naming the source
URL, the exact commit, and what was added on top;
`calibration/thermal_project/` is recorded in `docs/CALIBRATION_PROVENANCE.md`
instead.

| Tree | Upstream | Commit | License |
| --- | --- | --- | --- |
| `hardware/flirone-v4l2/` | https://github.com/fnoop/flirone-v4l2 | `b07e3a1` + local patch | GPL-2.0 |
| `hardware/flirone-v4l2-radiometric-audit/` | https://github.com/fnoop/flirone-v4l2 (branch `codex/flir-radiometric-feasibility`) | `e0b603c` | GPL-2.0 |
| `hardware/lepton/` | https://github.com/AnujN9/LeptonModule | `d5b6879` + local patch | BSD-3-Clause (Pure Engineering LLC) |
| `calibration/thermal_project/` | https://github.com/AnujN9/ThermalProject | `933c8bc` + local patch | **none stated upstream** |

## GPL-2.0 and the root Apache-2.0 license

The two `flirone-v4l2` trees are GPL-2.0 and stay GPL-2.0. They build a
standalone V4L2 bridge binary; no code in `ir_force/` links against them or
derives from them — the bridge is consumed at arm's length, through
`/dev/video*`. The root Apache-2.0 license therefore does not apply to those
directories and makes no claim over them.

The local modifications to `hardware/flirone-v4l2/src/flirone.c` (the FLIR ONE
Gen 2 thermal decoder, `decode_g2_thermal()` and the loopback device remap) are
GPL-2.0 like the file they modify, and publishing this repository is what puts
them under that license in the open. `hardware/flirone-v4l2/UPSTREAM.md`
records what the patch changes and that it was never committed upstream.

## FLIR SDK sources under `hardware/lepton/`

`hardware/lepton/software/*/leptonSDKEmb32PUB/` contains FLIR's published
Lepton SDK. Those headers carry FLIR's own proprietary/export-control notices;
they are reproduced verbatim and unmodified from the upstream tree, which FLIR
distributes publicly through GroupGets. This repository redistributes them on
the same terms and claims nothing over them.

## Dependency on mediapipe-so101

This repository consumes the public `mediapipe-so101` gripper contract
(`GripInput` / `GripperController`) at a pinned commit recorded in
`docs/PUBLIC_INTERFACE_LOCK.md`. It is Apache-2.0, the same license as this
repository, and is imported rather than vendored — no code from it is copied
into this tree.

## calibration/thermal_project/ states no license

`AnujN9/ThermalProject` carries no `LICENSE` file, and never has at any commit
in its history. Absent a stated license, the default applies: copyright rests
with the author, Anuj Natraj, and this repository has no grant to redistribute
it. It is carried here because the calibration line was built on it and the
`hand-teleop-heldout-verifier` branch adds ~2600 lines of work original to this
project (`HeldOutVerifier`, `CalibrationContracts`, `resolve_extrinsic`,
`ThermalFrameAssembler` and their tests — see `docs/CALIBRATION_PROVENANCE.md`).

**This is unresolved.** Either the upstream author states a license, or the
tree is removed from what is published and replaced by a pointer plus the
original additions. Until one of those happens, this directory is the part of
the repository that is not clear to redistribute.
