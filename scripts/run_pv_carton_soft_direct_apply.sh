#!/usr/bin/env bash
# Carton precise-range exploratory launcher, PV in APPLY mode.
# THE ARM MOVES and PressureVision drives the grip. Keep the e-stop within reach.
#
# The controller refuses to start until the Creative camera is recording; OAK is
# probed before the robot process starts, and teleop_viz_ee records into the same
# evidence directory. Etron is best-effort evidence when it is placed usefully.
#
# The PressureVision sender needs its own environment: set SO101_PV_PYTHON.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TELEOP="$REPO/experiments/teleop_viz_ee.py"
SENDER="$PUBLIC_REPO/integrations/pressurevision/tools/serve_pad_pressure.py"
OAK_PROBE_MODULE="lerobot_teleoperator_so101_webcam.programs.oak_probe"

LEVELS="${PV_LEVELS:-}"
MAPPING="${PV_MAPPING:-soft_precise}"
MAX_LEVEL_AGE_MINUTES="${PV_MAX_LEVEL_AGE_MINUTES:-180}"
PV_CAMERA="${PV_CAMERA:-2}"
PV_CROP="${PV_CROP:-40,0,980,720}"
ARM_PORT="${SO101_ARM_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14110850-if00}"
CREATIVE_CAMERA="${SO101_WORKSPACE_CAM:-/dev/v4l/by-id/usb-Creative_Technology_Ltd._Live__Cam_Chat_HD_VF0790_2015103001557-video-index0}"
ETRON_CAMERA="${SO101_SIDE_CAM:-/dev/v4l/by-id/usb-Etron_Technology__Inc._USB2.0_Camera-video-index0}"
EVIDENCE_ROOT="${PV_EVIDENCE_ROOT:-$REPO/local/evidence/pv_carton_soft_precise_apply}"

if [[ -z "${LEVELS}" ]]; then
    echo "PV_LEVELS must point to a fresh fitted levels.json" >&2
    exit 2
fi
if [[ ! -e "${CREATIVE_CAMERA}" ]]; then
    echo "required Creative evidence camera is unavailable: ${CREATIVE_CAMERA}" >&2
    exit 2
fi

mkdir -p "${EVIDENCE_ROOT}"
SESSION="$(mktemp -d "${EVIDENCE_ROOT}/session-XXXXXX")"
CREATIVE_VIDEO="${SESSION}/creative_side.ts"
ETRON_VIDEO="${SESSION}/etron_overview.ts"
PREVIEW_SHARE="${SESSION}/pv_preview.mmap"
SENDER_LOG="${SESSION}/pv_sender.csv"
SENDER_VIDEO="${SESSION}/pv_sender.avi"
SIDECAR="${SESSION}/pv_apply.csv"

creative_pid=0
etron_pid=0
sender_pid=0
cleanup() {
    rc=$?
    trap - EXIT INT TERM
    for pid in "${creative_pid}" "${etron_pid}" "${sender_pid}"; do
        if [[ "${pid}" -gt 0 ]] && kill -0 "${pid}" 2>/dev/null; then
            if kill "${pid}" 2>/dev/null; then :; fi
        fi
    done
    for pid in "${creative_pid}" "${etron_pid}" "${sender_pid}"; do
        if [[ "${pid}" -gt 0 ]]; then
            if wait "${pid}" 2>/dev/null; then :; fi
        fi
    done
    echo "evidence: ${SESSION}"
    exit "${rc}"
}
trap cleanup EXIT INT TERM

ffmpeg -nostdin -hide_banner -loglevel error \
    -f v4l2 -input_format mjpeg -framerate 30 -video_size 1280x720 \
    -i "${CREATIVE_CAMERA}" -an -c:v libx264 -preset ultrafast -tune zerolatency \
    -crf 28 -g 15 -x264-params repeat-headers=1 -f mpegts "${CREATIVE_VIDEO}" \
    >"${SESSION}/creative_ffmpeg.log" 2>&1 &
creative_pid=$!

if [[ -e "${ETRON_CAMERA}" ]] && \
   [[ "$(readlink -f "${CREATIVE_CAMERA}")" != "$(readlink -f "${ETRON_CAMERA}")" ]]; then
    ffmpeg -nostdin -hide_banner -loglevel error \
        -f v4l2 -framerate 30 -video_size 640x480 \
        -i "${ETRON_CAMERA}" -an -c:v libx264 -preset ultrafast -tune zerolatency \
        -crf 28 -g 15 -x264-params repeat-headers=1 -f mpegts "${ETRON_VIDEO}" \
        >"${SESSION}/etron_ffmpeg.log" 2>&1 &
    etron_pid=$!
else
    echo "optional Etron recording skipped: ${ETRON_CAMERA}" >&2
fi

"${PV_PYTHON}" "${SENDER}" \
    --levels "${LEVELS}" \
    --camera "${PV_CAMERA}" \
    --crop "${PV_CROP}" \
    --mjpg \
    --require-scene-match \
    --max-level-age-minutes "${MAX_LEVEL_AGE_MINUTES}" \
    --log "${SENDER_LOG}" \
    --video-out "${SENDER_VIDEO}" \
    --preview-share "${PREVIEW_SHARE}" \
    >"${SESSION}/pv_sender.log" 2>&1 &
sender_pid=$!

# The startup gate: no robot process starts until every evidence stream is
# actually producing bytes. A run whose evidence never started is not evidence.
for _ in $(seq 1 200); do
    for pid in "${creative_pid}" "${sender_pid}"; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "an evidence process exited before the startup gate passed; inspect ${SESSION}" >&2
            exit 1
        fi
    done
    if [[ -s "${CREATIVE_VIDEO}" && -s "${SENDER_LOG}" && -s "${PREVIEW_SHARE}" ]]; then
        break
    fi
    sleep 0.1
done
if [[ ! -s "${CREATIVE_VIDEO}" || ! -s "${SENDER_LOG}" || ! -s "${PREVIEW_SHARE}" ]]; then
    echo "evidence startup gate timed out; inspect ${SESSION}" >&2
    exit 1
fi

"$PYTHON" -m "${OAK_PROBE_MODULE}" >"${SESSION}/oak_probe.log" 2>&1 || true

"$PYTHON" "${TELEOP}" "$@" \
    --oak \
    --pv-pressure \
    --pv-mapping "${MAPPING}" \
    --pv-sidecar "${SIDECAR}" \
    --pv-preview-share "${PREVIEW_SHARE}" \
    --pv-evidence-dir "${SESSION}" \
    --arm-port "${ARM_PORT}"
