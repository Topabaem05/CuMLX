#!/bin/sh
# The M3 gates: third-party CUDA kernels, translated by woof on this Mac, and compared with goldens from the V100.
#
#   sh spikes/m3_kernels.sh                 # makes an environment, installs woof, translates and compares. Returns at once
#   sh spikes/m3_kernels.sh capture         # records the goldens on the V100. It stops if the GPU is busy. Returns at once
#   sh spikes/m3_kernels.sh status          # shows the state of the run
#   sh spikes/m3_kernels.sh tools           # prints the programs that it would run, and runs nothing
#
# The default mode installs metal-woof and runs it with your permissions. `capture` compiles and runs third-party kernels on
# the V100, in a folder that it deletes. Results: summary.json in the folder.
# Delete the folder when the test is finished: rm -rf ~/cumlx-m3
set -u

ROOT=${CUMLX_M3_ROOT:-$HOME/cumlx-m3}
REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
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
PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
export PATH
UV=${CUMLX_M3_UV:-uv}
VENV=$ROOT/woof-venv
PY=$VENV/bin/python
GOLDEN=$REPO/validation/kernel-golden-v100-m3
CAPTURE=${CUMLX_M3_CAPTURE:-$REPO/scripts/kernel_capture_remote.sh}

running() {
    [ -f "$ROOT/run.pid" ] && kill -0 "$(cat "$ROOT/run.pid")" 2>/dev/null
}

case "${1:-start}" in
tools)
    echo "uv $(command -v "$UV" || echo missing)"
    echo "python $PY"
    echo "capture $CAPTURE"
    exit 0
    ;;
status)
    if running; then echo "running, pid $(cat "$ROOT/run.pid")"; else echo "not running"; fi
    [ -f "$ROOT/run.log" ] && tail -n 25 "$ROOT/run.log"
    [ -f "$ROOT/summary.json" ] && echo "summary: $ROOT/summary.json"
    exit 0
    ;;
--capture-work)
    "$REPO/.venv/bin/python" "$REPO/spikes/m3_kernels.py" fetch --root "$ROOT" || exit $?
    "$REPO/.venv/bin/python" "$REPO/spikes/m3_kernels.py" cases --root "$ROOT" || exit $?
    if [ -e "$GOLDEN" ]; then
        echo "$GOLDEN exists. Delete it first if you want a new recording." >&2
        exit 2
    fi
    bash "$CAPTURE" "$HOST" "$GOLDEN" "$ROOT/kernels" "$REPO/validation/kernel-cases-m3/cases.json"
    code=$?
    echo "capture exit code: $code"
    exit "$code"
    ;;
--work)
    [ -x "$PY" ] || "$UV" venv -q --python 3.12 "$VENV" || exit $?  # A second run keeps the environment. uv refuses to replace one by default.
    # The same packages as spikes/run_all.sh. It stays in its own environment: woof adds a .pth file and a cupy facade there.
    "$UV" pip install -q --python "$PY" metal-woof==1.1.5.2 'libclang>=18.1' 'mlx>=0.29' 'nanobind>=3.0,<3.2' || exit $?
    "$PY" "$REPO/spikes/m3_kernels.py" fetch --root "$ROOT" || exit $?
    "$PY" "$REPO/spikes/m3_kernels.py" cases --root "$ROOT" || exit $?
    "$PY" "$REPO/spikes/m3_kernels.py" run --root "$ROOT"
    code=$?
    echo "finished, exit code $code. Summary: $ROOT/summary.json"
    exit "$code"
    ;;
capture | start)
    if running; then
        echo "Already running, pid $(cat "$ROOT/run.pid"). Log: $ROOT/run.log"
        exit 0
    fi
    mkdir -p "$ROOT"
    if [ "${1:-start}" = "capture" ]; then
        need_host
        if [ ! -x "$REPO/.venv/bin/python" ]; then
            echo "error: $REPO/.venv/bin/python not found" >&2
            exit 3
        fi
        echo "Starting the M3 capture on $HOST. It stops if the GPU has compute processes"
        MODE=--capture-work
    else
        if ! command -v "$UV" >/dev/null; then
            echo "error: uv is not on PATH. Install it: https://docs.astral.sh/uv/getting-started/installation/" >&2
            exit 3
        fi
        echo "Starting the M3 translation and comparison in $ROOT"
        MODE=--work
    fi
    nohup sh "$0" "$MODE" > "$ROOT/run.log" 2>&1 < /dev/null &
    echo $! > "$ROOT/run.pid"
    echo "pid $(cat "$ROOT/run.pid"). Log: $ROOT/run.log. Check it with: sh spikes/m3_kernels.sh status"
    exit 0
    ;;
*)
    echo "usage: sh spikes/m3_kernels.sh [start|capture|status|tools]" >&2
    exit 2
    ;;
esac
