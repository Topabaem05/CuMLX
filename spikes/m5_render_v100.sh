#!/bin/sh
# Render the M5 scenes with Mitsuba cuda_ad_rgb on a shared GPU machine, and return at once.
#
#   sh spikes/m5_render_v100.sh           # copies its own files and starts the run
#   sh spikes/m5_render_v100.sh status    # shows the state of the run
#   sh spikes/m5_render_v100.sh fetch     # copies the images to ~/cumlx-m5/cuda on this Mac
#
# It refuses to start if another job is using the GPU. The host is CUMLX_GPU_HOST, or the content of ~/.config/cumlx/gpu-host. There is no default.
# It copies spikes/m5_render.py and src/cumlx from HEAD, and installs only mitsuba==3.8.0 and NumPy in its own folder.
# 3.8.0, because Dr.Jit 1.4 and later (Mitsuba 3.9) refuse a GPU under compute capability 7.5, and the V100 is 7.0.
# It also renders scalar_rgb on the CPU, to measure what the version difference from the Mac's 3.9.1 does.
# Delete the remote folder when the test is finished: ssh <host> rm -rf cumlx-m5-render
set -u

HOST=${CUMLX_GPU_HOST:-}
HOST_FILE=${XDG_CONFIG_HOME:-$HOME/.config}/cumlx/gpu-host
if [ -z "$HOST" ] && [ -f "$HOST_FILE" ]; then
    HOST=$(cat "$HOST_FILE")
fi
need_host() {
    if [ -z "$HOST" ]; then
        echo "error: no GPU machine is set. Set CUMLX_GPU_HOST to an ssh host with an NVIDIA GPU, or write that host to $HOST_FILE" >&2
        exit 3
    fi
}
REMOTE=cumlx-m5-render
REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
LOCAL=${CUMLX_M5_GOLDEN:-$HOME/cumlx-m5/cuda}

need_host

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
    if ssh "$HOST" "test -f $REMOTE/scalar/summary.json"; then
        SCALAR=$(dirname -- "$LOCAL")/scalar-3.8.0
        mkdir -p "$SCALAR"
        ssh "$HOST" "cd $REMOTE/scalar && tar -c ." | tar -x -C "$SCALAR" || exit 1
        echo "Copied to $SCALAR"
    fi
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
ssh "$HOST" "bash -lc 'cd $REMOTE && (rm -rf venv cuda scalar && uv venv -q --python 3.12 venv && uv pip install -q --python venv/bin/python mitsuba==3.8.0 numpy && { PYTHONUNBUFFERED=1 nohup venv/bin/python CuMLX/spikes/m5_render.py render --variant cuda_ad_rgb --seeds 1,2,3 --out cuda; PYTHONUNBUFFERED=1 nohup venv/bin/python CuMLX/spikes/m5_render.py render --variant scalar_rgb --seeds 21,22,23 --out scalar; }) > run.log 2>&1 < /dev/null & echo \$! > $REMOTE/run.pid'" || exit 3
sleep 2
if ! ssh "$HOST" "kill -0 \$(cat $REMOTE/run.pid) 2>/dev/null"; then
    echo "error: the run stopped at once:" >&2
    ssh "$HOST" "cat $REMOTE/run.log" >&2
    exit 1
fi
echo "Started on $HOST, pid $(ssh "$HOST" "cat $REMOTE/run.pid")."
echo "It keeps running if this terminal closes. Tell Claude when you have started it."
