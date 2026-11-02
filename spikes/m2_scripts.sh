#!/bin/sh
# Start spikes/m2_scripts.py in the background and return at once.
#
#   sh spikes/m2_scripts.sh           # starts the run in ~/cumlx-m2-scripts
#   sh spikes/m2_scripts.sh status    # shows the state of the run
#
# Delete the folder when the test is finished: rm -rf ~/cumlx-m2-scripts
set -u

ROOT=${CUMLX_M2_ROOT:-$HOME/cumlx-m2-scripts}
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
mkdir -p "$ROOT"
PYTHONUNBUFFERED=1 nohup "$REPO/.venv/bin/python" "$REPO/spikes/m2_scripts.py" --work "$ROOT" >"$ROOT/run.log" 2>&1 &
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
