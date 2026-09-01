# Public interface lock

This repository depends on the public `mediapipe-so101` gripper contract. The
dependency is pinned to an exact commit so a change there cannot silently alter
behaviour here.

```
repository  mediapipe-so101
commit      06ff9988359e741be76bd728fa923db17f793f6e
contract    lerobot_teleoperator_so101_webcam.grip.contract
            - GripInput
            - GripperController
            - make_grip_input, STALE_AFTER_S
```

## Installing during local development

```bash
pip install -e /path/to/mediapipe-so101/packages/so101_teleop
```

Check out the pinned commit first. After the public repository is published, this
instruction may be replaced with a Git URL pinned to the same literal commit —
that is a deliberate release change, not an automatic one.

## What the contract guarantees

- MediaPipe owns arm motion and grasp/release authority.
- A gripper controller only decides how far to close while a grasp is active.
- Missing, invalid, or stale input holds the current command.
- Only an explicit MediaPipe release opens the gripper, and it opens to the
  calibrated centre (50.0 for a RANGE_0_100 gripper).

An IR adapter implemented here must satisfy those, and must not need any change
on the public side.
