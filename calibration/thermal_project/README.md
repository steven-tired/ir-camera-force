# thermal_project — additions only, not a copy of the upstream tree

The RealSense↔Lepton calibration and thermal-point-cloud rig this project's
calibration line was built on is **Anuj Natraj's ThermalProject**:

> https://github.com/AnujN9/ThermalProject

That repository states no license, and never has at any commit in its history.
Absent a stated license the default applies — copyright rests with its author —
so this repository does not redistribute it. What used to sit here was a full
copy of the `hand-teleop-heldout-verifier` branch; it was removed on 2026-09-04
before publication.

What is kept is the work original to this project, and enough metadata to put
it back together.

## Layout

```
additions/   16 files this project wrote, at their paths within the upstream tree
patches/     the diffs against upstream files, which this project modified
```

`additions/` is the substance: `HeldOutVerifier` and `CalibrationContracts`
(the non-fitting held-out gate that decides whether a calibration run passes),
`resolve_extrinsic` (mount-prior flip resolution), `ThermalFrameAssembler` and
`RealSenseCaptureContract`, and the tests for all of them — about 2600 lines.

`patches/hand-teleop-heldout-verifier.patch` carries the changes to nine
upstream files (the two `CMakeLists.txt`, `cal.cpp`, `extrinsic_cal.cpp`,
`depthimage.cpp`, `lepton.cpp`, and the READMEs) that those additions need in
order to build. `patches/realsense2-optional.patch` makes `find_package(realsense2)`
non-fatal so the stream tools configure on a machine with no RealSense SDK.

## Reconstructing the tree

```bash
git clone https://github.com/AnujN9/ThermalProject.git
cd ThermalProject
git checkout e0dbe7d                      # the base this work sits on
git apply /path/to/patches/hand-teleop-heldout-verifier.patch
cp -r /path/to/additions/. .
```

That reproduces commit `933c8bc` of the `hand-teleop-heldout-verifier` branch
byte for byte — the state every calibration run in `docs/CALIBRATION_PROVENANCE.md`
was produced with. Apply `realsense2-optional.patch` on top if you are building
without the RealSense SDK.

| | |
| --- | --- |
| Upstream | https://github.com/AnujN9/ThermalProject |
| Base commit | `e0dbe7d` ("Updated vcs import command", 2024-08-25) |
| Reconstructed commit | `933c8bc` ("feat(calibration): add resolve_extrinsic") |
| Diff | 25 files changed, 2630 insertions(+), 156 deletions(-) |

## Working with upstream

The additions are a branch off Anuj's tree, not a fork of it — they were always
meant to go back. Two things would each simplify this, in order of how much
they help:

1. **A license on `AnujN9/ThermalProject`.** Any permissive one (Apache-2.0
   matches this repository; MIT and BSD-3-Clause are equally fine) makes the
   whole tree redistributable, and this directory could then carry the working
   checkout again instead of a reconstruction recipe.
2. **Upstreaming the additions.** `HeldOutVerifier`, `CalibrationContracts` and
   `resolve_extrinsic` are not specific to this project's rig — they are a
   held-out gate and a flip resolver for the calibration any user of that repo
   runs. Merged upstream, they stop being an overlay at all.

Until either happens, the recipe above is the interface between the two
projects, and `docs/CALIBRATION_PROVENANCE.md` records which run used which
commit.
