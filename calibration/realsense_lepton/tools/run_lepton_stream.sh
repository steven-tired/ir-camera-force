#!/bin/bash
# Manage the FLIR Lepton VoSPI->UDP streamer on the Raspberry Pi.
#
# The Pi streams raw 160x120 uint16 Lepton frames as 4 UDP datagrams/frame to
# this laptop (192.168.50.1:8080). The laptop side (LeptonUDPSource) is fail-soft
# against the streamer dying, so start/stop here is safe to call anytime.
#
# Usage:
#   ./run_lepton_stream.sh start    # launch streamer on the Pi (default)
#   ./run_lepton_stream.sh stop     # kill it
#   ./run_lepton_stream.sh status   # is it running? tail its log
set -euo pipefail

PI=anujn@192.168.50.2
LAPTOP_IP=192.168.50.1
PORT=8080
BIN=/home/anujn/Project/LeptonModule/software/build/raspberrypi_video_network
LOG=/tmp/lepton_stream.log
STREAM_RE="^${BIN}( |$)"

cmd=${1:-start}
case "$cmd" in
  start)
    ssh "$PI" "pkill -f '$STREAM_RE' || true; sleep 1; \
      nohup $BIN -net $LAPTOP_IP -port $PORT > $LOG 2>&1 & sleep 2; \
      pgrep -af '$STREAM_RE' && cat $LOG"
    echo "streamer targeting $LAPTOP_IP:$PORT — view with view_ir_camera.py --lepton-udp $PORT"
    ;;
  stop)
    ssh "$PI" "pkill -f '$STREAM_RE' && echo stopped || echo 'not running'"
    ;;
  status)
    ssh "$PI" "pgrep -af '$STREAM_RE' || echo NOT_RUNNING; echo ---LOG---; tail -n 20 $LOG 2>/dev/null || true"
    ;;
  *)
    echo "usage: $0 {start|stop|status}" >&2
    exit 2
    ;;
esac
