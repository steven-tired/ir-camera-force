# LeptonModule — upstream and local provenance

**Upstream:** https://github.com/AnujN9/LeptonModule.git
**Commit:** `d5b6879` ("Updated main README")
**Migrated from:** `lepton-module-ir-hand-pinch-phase1` (path relative to the
pre-split `hand-teleop` workspace root) — a **git worktree** of
`lepton-module`, on branch `ir-hand-pinch-phase1`
**Migrated on:** 2026-09-01

## Why the worktree and not the main checkout

Both `lepton-module` (master) and the `ir-hand-pinch-phase1` worktree sat at the
same commit `d5b6879`, so their committed content is identical. The worktree
additionally carried uncommitted phase-1 work, which the main checkout does not
have. Copying the worktree therefore captures a strict superset — nothing from
master is lost.

## Uncommitted phase-1 work carried across

Modified:
* `software/CMakeLists.txt`
* `software/raspberrypi_video_network/include/Lepton_I2C.h`
* `software/raspberrypi_video_network/src/Lepton_I2C.cpp`
* `software/raspberrypi_video_network/src/main.cpp`

Untracked:
* `software/raspberrypi_video_network/include/LeptonVoSPI.h`
* `software/raspberrypi_video_network/tests/` (`test_lepton_vospi.cpp`)

As with the FLIR patch, this existed only as working-tree state in a checkout
whose `.git` is not migrated.

## Deliberately excluded

* `.git` (the worktree's linking file) and `build/`.
* `software/raspberrypi_libs/leptonSDKEmb32PUB/Debug/` object files and
  `libLEPTON_SDK.a` — build output, and ignored by upstream's own `.gitignore`.

## Hardware status

The Lepton itself went silent (SPI + I2C) on 2026-07-17 and has not been
revived. This source is preserved for when it is replaced; it is not currently
exercised by any live-camera gate.
