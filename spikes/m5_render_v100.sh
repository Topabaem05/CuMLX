#!/bin/sh
# Render the M5 scenes with Mitsuba cuda_ad_rgb on a shared GPU machine, and return at once.
#
#   sh spikes/m5_render_v100.sh           # copies its own files and starts the run
#   sh spikes/m5_render_v100.sh status    # shows the state of the run
#   sh spikes/m5_render_v100.sh fetch     # copies the images to ~/cumlx-m5/cuda on this Mac
#
# It refuses to start if another job is using the GPU. The host is CUMLX_GPU_HOST, default gpu-host.
# It copies spikes/m5_render.py and src/cumlx from HEAD, and installs only mitsuba==3.9.1 and NumPy in its own folder.
# Delete the remote folder when the test is finished: ssh gpu-host rm -rf cumlx-m5-render
set -u

HOST=${CUMLX_GPU_HOST:-gpu-host}
REMOTE=cumlx-m5-render
REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
LOCAL=${CUMLX_M5_GOLDEN:-$HOME/cumlx-m5/cuda}

case "${1:-start}" in
status)
    ssh "$HOST" "if [ -f $REMOTE/run.pid ] && kill -0 \$(cat $REMOTE/run.pid) 2>/dev/null; then echo running; else echo not running; fi; tail -n 12 $REMOTE/run.log 2>/dev/null"
    exit $?
    ;;
fetch)
    if ! ssh "$HOST" "test -f $REMOTE/cuda/summary.json"; then
        echo "error: $HOST has no results yet" >&2
        exit 1
    fi
    mkdir -p "$LOCAL"
    ssh "$HOST" "cd $REMOTE/cuda && tar -c ." | tar -x -C "$LOCAL" || exit 1
    echo "Copied to $LOCAL"
    exit 0
    ;;
start) ;;
*)
    echo "usage: sh spikes/m5_render_v100.sh [status|fetch]" >&2
    exit 2
    ;;
esac

BUSY=$(ssh "$HOST" "nvidia-smi --query-compute-apps=pid --format=csv,noheader") || {
    echo "error: could not reach $HOST or read its GPU" >&2
    exit 3
}
if [ -n "$BUSY" ]; then
    echo "error: another job is using the GPU on $HOST, pid $BUSY. Nothing was started." >&2
    exit 3
fi
ssh "$HOST" "mkdir -p $REMOTE/CuMLX" || exit 3
git -C "$REPO" archive HEAD spikes/m5_render.py src/cumlx | ssh "$HOST" "tar -x -C $REMOTE/CuMLX" || exit 3
# The login shell gives uv its PATH. Everything after this line runs on the GPU machine and survives this terminal.
ssh "$HOST" "bash -lc 'cd $REMOTE && (uv venv -q --python 3.12 venv && uv pip install -q --python venv/bin/python mitsuba==3.9.1 numpy && PYTHONUNBUFFERED=1 nohup venv/bin/python CuMLX/spikes/m5_render.py render --variant cuda_ad_rgb --seeds 1,2,3 --out cuda) > run.log 2>&1 < /dev/null & echo \$! > $REMOTE/run.pid'" || exit 3
sleep 2
if ! ssh "$HOST" "kill -0 \$(cat $REMOTE/run.pid) 2>/dev/null"; then
    echo "error: the run stopped at once:" >&2
    ssh "$HOST" "cat $REMOTE/run.log" >&2
    exit 1
fi
echo "Started on $HOST, pid $(ssh "$HOST" "cat $REMOTE/run.pid")."
echo "It keeps running if this terminal closes. Tell Claude when you have started it."
