#!/bin/sh
# Start spikes/isaac_lab_mac.py in the background and return at once.
#
#   sh spikes/isaac_lab_mac.sh            # starts the run in ~/cumlx-isaaclab
#   sh spikes/isaac_lab_mac.sh status     # shows the state of the run
#
# Delete the folder when the test is finished: rm -rf ~/cumlx-isaaclab
set -u

ISAAC_LAB_COMMIT=c498443e0b833515e1976db9eb3edcd8dabb4cb7
ROOT=${CUMLX_ISAACLAB_ROOT:-$HOME/cumlx-isaaclab}
REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
export PATH

running() {
    [ -f "$ROOT/run.pid" ] && kill -0 "$(cat "$ROOT/run.pid")" 2>/dev/null
}

if [ "${1:-start}" = "status" ]; then
    if running; then
        echo "running, pid $(cat "$ROOT/run.pid")"
    else
        echo "not running"
    fi
    [ -f "$ROOT/run.log" ] && tail -n 12 "$ROOT/run.log"
    exit 0
fi

if running; then
    echo "Already running, pid $(cat "$ROOT/run.pid"). Log: $ROOT/run.log"
    exit 0
fi
if [ ! -x "$REPO/.venv/bin/python" ]; then
    echo "error: $REPO/.venv/bin/python not found" >&2
    exit 3
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv not found" >&2
    exit 3
fi

mkdir -p "$ROOT"
if [ ! -f "$ROOT/IsaacLab/pyproject.toml" ]; then
    echo "Downloading Isaac Lab $ISAAC_LAB_COMMIT ..."
    rm -rf "$ROOT/IsaacLab"
    if ! git -c advice.detachedHead=false clone -q --filter=blob:none --no-checkout \
        https://github.com/isaac-sim/IsaacLab.git "$ROOT/IsaacLab" \
        || ! git -C "$ROOT/IsaacLab" -c advice.detachedHead=false checkout -q "$ISAAC_LAB_COMMIT"; then
        echo "error: could not download Isaac Lab" >&2
        exit 3
    fi
fi

PYTHONUNBUFFERED=1 nohup "$REPO/.venv/bin/python" "$REPO/spikes/isaac_lab_mac.py" \
    --checkout "$ROOT/IsaacLab" --work "$ROOT/work" >"$ROOT/run.log" 2>&1 &
echo $! >"$ROOT/run.pid"
sleep 1
if ! running; then
    echo "error: the run stopped at once:" >&2
    cat "$ROOT/run.log" >&2
    exit 1
fi
echo "Started, pid $(cat "$ROOT/run.pid")."
echo "Log: $ROOT/run.log"
echo "It keeps running if this terminal closes. Tell Claude when you have started it."
