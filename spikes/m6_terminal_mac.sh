#!/bin/sh
# M6-1 with terminal-mac: a virtual Mac from Apple's own restore image, shown in a native window. One command, and it returns at once.
#
#   sh spikes/m6_terminal_mac.sh start     # builds terminal-mac at a fixed commit, downloads macOS from Apple, installs it, and opens the window
#   sh spikes/m6_terminal_mac.sh status
#   sh spikes/m6_terminal_mac.sh stop
#
# It builds from source, at the commit that was read before use. terminal-mac (MIT, https://github.com/govijr/terminal-mac) is third party, and an early release candidate.
# A person then does the macOS setup in the window. spikes/m6_vm_reproduce.sh attach runs the work over ssh, so terminal-mac's guest helper is not needed for it.
# While this runs, the VM stops when this Mac's memory in use reaches 38 GB or its free disk falls to 15 GB.
set -u
COMMIT=7009117f9e910c3dd4b4bd863e8bc331bd3c9d96
REPO_URL=https://github.com/govijr/terminal-mac
VM_MEMORY_GIB=6
VM_CPUS=8
VM_DISK_GIB=80
MIN_FREE_START_GB=65
MIN_FREE_GB=15
MAX_USED_GB=38
MAX_USED_AT_START_GB=37
POLL=${CUMLX_VM_POLL:-30}
ROOT=${CUMLX_VM_ROOT:-$HOME/cumlx-vm}
SRC=$ROOT/terminal-mac-src
TM=$SRC/dist/terminal-mac

used_gb() {
    vm_stat | awk '/page size of/ {size=$8} /Pages active/ {a=$3} /Pages wired down/ {w=$4} /occupied by compressor/ {c=$5} END {printf "%.1f", (a+w+c)*size/1e9}'
}
free_gb() {
    df -k "$ROOT" | awk 'NR==2 {print int($4 / 1000000)}'
}
at_least() {
    awk -v a="$1" -v b="$2" 'BEGIN {exit !(a >= b)}'
}
running() {
    [ -f "$ROOT/setup.pid" ] && kill -0 "$(cat "$ROOT/setup.pid")" 2>/dev/null
}
stage() {
    echo "[$(date +%H:%M:%S)] $1"
}

MODE=${1:-}
case "$MODE" in
status)
    if running; then echo "running, pid $(cat "$ROOT/setup.pid")"; else echo "not running"; fi
    [ -f "$ROOT/setup.log" ] && tail -n 6 "$ROOT/setup.log"
    [ -x "$TM" ] && "$TM" status default 2>/dev/null
    if [ -f "$ROOT/setup_state.txt" ]; then
        echo "state: $(cat "$ROOT/setup_state.txt")"
        [ "$(cat "$ROOT/setup_state.txt")" = window_open ]
        exit $?
    fi
    exit 5
    ;;
stop)
    running && kill "$(cat "$ROOT/setup.pid")" 2>/dev/null
    [ -x "$TM" ] && "$TM" stop default
    echo "Stopped."
    exit 0
    ;;
start)
    if [ "$(uname -sm)" != "Darwin arm64" ]; then
        echo "error: this needs an Apple Silicon Mac, found $(uname -sm)" >&2
        exit 3
    fi
    mkdir -p "$ROOT"
    if running; then
        echo "error: a setup is going, pid $(cat "$ROOT/setup.pid")" >&2
        exit 4
    fi
    now=$(free_gb)
    if [ "$now" -lt "$MIN_FREE_START_GB" ]; then
        echo "error: needs $MIN_FREE_START_GB GB free, found $now GB" >&2
        exit 3
    fi
    used=$(used_gb)
    if at_least "$(awk -v a="$used" -v b="$VM_MEMORY_GIB" 'BEGIN {print a + b * 1.0737}')" "$MAX_USED_AT_START_GB"; then
        echo "error: $used GB of memory is in use, and the VM adds $VM_MEMORY_GIB GiB. That reaches $MAX_USED_AT_START_GB GB. Try again when this Mac is less busy." >&2
        exit 3
    fi
    nohup sh "$0" --work >"$ROOT/setup.log" 2>&1 &
    echo $! >"$ROOT/setup.pid"
    echo "Started, pid $(cat "$ROOT/setup.pid"). Log: $ROOT/setup.log"
    echo "A window opens when macOS is installed, in about an hour. See the state with: sh $0 status"
    exit 0
    ;;
--work) ;;
*)
    sed -n 2,11p "$0" >&2
    exit 2
    ;;
esac

state() {
    echo "$1" >"$ROOT/setup_state.txt"
    stage "state: $1"
}

if [ ! -x "$TM" ]; then
    stage "1/3 terminal-mac at $COMMIT"
    [ -d "$SRC" ] && { state error_source_folder_exists; exit 1; }
    git clone -q "$REPO_URL" "$SRC" || { state error_clone; exit 1; }
    git -C "$SRC" -c advice.detachedHead=false checkout -q "$COMMIT" || { state error_commit; exit 1; }
    if [ "$(git -C "$SRC" rev-parse HEAD)" != "$COMMIT" ]; then
        state error_wrong_commit
        exit 1
    fi
    (cd "$SRC" && SIGN_IDENTITY=- bash scripts/build.sh) || { state error_build; exit 1; }
    [ -x "$TM" ] || { state error_build; exit 1; }
fi
"$TM" doctor || { state error_doctor; exit 1; }

stage "2/3 macOS from Apple, into a new VM: $VM_CPUS CPUs, $VM_MEMORY_GIB GiB, a $VM_DISK_GIB GiB sparse disk"
"$TM" up default --detach --cpus "$VM_CPUS" --memory "$VM_MEMORY_GIB" --disk "$VM_DISK_GIB" || { state error_up; exit 1; }

stage "3/3 the window"
"$TM" show default || { state error_show; "$TM" stop default; exit 1; }
state window_open
while :; do
    used=$(used_gb)
    if at_least "$used" "$MAX_USED_GB"; then
        state "stopped_memory_$used"
        "$TM" stop default
        exit 1
    fi
    if [ "$(free_gb)" -le "$MIN_FREE_GB" ]; then
        state stopped_disk
        "$TM" stop default
        exit 1
    fi
    "$TM" status default 2>/dev/null | grep -q '"running"' || { stage "the VM is not running"; exit 0; }
    sleep "$POLL"
done
