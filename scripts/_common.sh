# Shared by every wrapper here. Resolves the repository from this script's
# location, so the wrappers work from any cwd and from any checkout path.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# This repo first, then the public one it depends on. The dependency is one-way:
# ir-camera-force -> mediapipe-so101 -> LeRobot.
PUBLIC_REPO="${MEDIAPIPE_SO101_DIR:-$(cd "$REPO/../mediapipe-so101" 2>/dev/null && pwd || true)}"
if [[ -z "${PUBLIC_REPO}" || ! -d "${PUBLIC_REPO}" ]]; then
  echo "mediapipe-so101 not found. Set MEDIAPIPE_SO101_DIR to that checkout." >&2
  exit 2
fi
export PYTHONPATH="$REPO:$PUBLIC_REPO/packages/so101_teleop/src:$PUBLIC_REPO/packages/webcam_input/src:$PUBLIC_REPO/integrations/pressurevision/src${PYTHONPATH:+:$PYTHONPATH}"

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

PYTHON="${SO101_PYTHON:-python3}"
PV_PYTHON="${SO101_PV_PYTHON:-$PYTHON}"
