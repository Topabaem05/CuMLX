#!/bin/sh
# Record the CUDA golden for spikes/m2_scripts.py on a shared GPU machine, and return at once.
#
#   sh spikes/m2_golden_v100.sh           # copies the committed repo and starts the run
#   sh spikes/m2_golden_v100.sh status    # shows the state of the run
#   sh spikes/m2_golden_v100.sh fetch     # copies the reports to ~/cumlx-m2-golden on this Mac
#
# It refuses to start if another job is using the GPU. The host is CUMLX_GPU_HOST, or the content of ~/.config/cumlx/gpu-host. There is no default.
# Delete the remote folder when the test is finished: ssh <host> rm -rf cumlx-m2-golden
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
REMOTE=cumlx-m2-golden
REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
LOCAL=${CUMLX_M2_GOLDEN:-$HOME/cumlx-m2-golden}

need_host

case "${1:-start}" in
status)
    ssh "$HOST" "if [ -f $REMOTE/run.pid ] && kill -0 \$(cat $REMOTE/run.pid) 2>/dev/null; then echo running; else echo not running; fi; tail -n 12 $REMOTE/run.log 2>/dev/null"
    exit $?
    ;;
fetch)
    if ! ssh "$HOST" "test -f $REMOTE/work/summary.json"; then
        echo "error: $HOST has no results yet" >&2
        exit 1
    fi
    mkdir -p "$LOCAL"
    ssh "$HOST" "cd $REMOTE/work && tar -c summary.json reports" | tar -x -C "$LOCAL" || exit 1
    echo "Copied to $LOCAL"
    exit 0
    ;;
start) ;;
*)
    echo "usage: sh spikes/m2_golden_v100.sh [status|fetch]" >&2
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
git -C "$REPO" archive HEAD | ssh "$HOST" "tar -x -C $REMOTE/CuMLX" || exit 3
# The login shell gives uv its PATH. Everything after this line runs on the GPU machine and survives this terminal.
ssh "$HOST" "bash -lc 'cd $REMOTE && PYTHONUNBUFFERED=1 nohup python3 CuMLX/spikes/m2_scripts.py --work \$PWD/work --target cuda > run.log 2>&1 < /dev/null & echo \$! > $REMOTE/run.pid'" || exit 3
sleep 2
if ! ssh "$HOST" "kill -0 \$(cat $REMOTE/run.pid) 2>/dev/null"; then
    echo "error: the run stopped at once:" >&2
    ssh "$HOST" "cat $REMOTE/run.log" >&2
    exit 1
fi
echo "Started on $HOST, pid $(ssh "$HOST" "cat $REMOTE/run.pid")."
echo "It keeps running if this terminal closes. Tell Claude when you have started it."
