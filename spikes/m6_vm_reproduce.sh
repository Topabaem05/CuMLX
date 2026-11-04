#!/bin/sh
# M6-1: reproduce the flagship work on a "new Mac": a fresh macOS 27 VM, on this Mac, from the public repository. One command, and it returns at once.
#
#   sh spikes/m6_vm_reproduce.sh start     # downloads Tart and a macOS image of about 30 GB, then runs for hours. It stops by itself
#   sh spikes/m6_vm_reproduce.sh status    # the state, and the verdict when it is done
#   sh spikes/m6_vm_reproduce.sh stop      # stops the VM
#   sh spikes/m6_vm_reproduce.sh clean     # deletes the VM and the image, and keeps the results
#   sh spikes/m6_vm_reproduce.sh attach IP # the same work in a Mac that is already running, such as one from spikes/m6_terminal_mac.sh. It logs in as CUMLX_VM_USER with CUMLX_VM_PASSWORD, which default to admin and admin
#
# CUMLX_VM_PROXY=1 with attach is for a guest whose NAT does not reach the internet: the guest's web traffic goes through spikes/m6_vm_proxy.py over ssh -R, and no setting on this Mac changes. The result says so.
# It uses Tart (FSL-1.1-ALv2, third party, checked by SHA-256) and an image built by the Tart project. The image has user admin, password admin, on the VM's private network.
# It stops the VM when this Mac's memory in use reaches 38 GB or its free disk falls to 15 GB. Those stops are never a pass.
# A VM is not a new machine: the GPU is Apple's paravirtual device on this same Mac. The results say so.
set -u
TART_VERSION=2.41.1
TART_SHA256=e680a734febccfce7d4e910c1f7f13fc3b00f19a181c945e2d746de3214c32c2
TART_URL=https://github.com/openai/tart/releases/download/$TART_VERSION/tart.tar.gz
IMAGE=ghcr.io/cirruslabs/macos-golden-gate-vanilla:27.2
VM=cumlx-repro
VM_MEMORY_MB=${CUMLX_VM_MEMORY_MB:-6144}
VM_CPUS=${CUMLX_VM_CPUS:-8}
VM_DISK_GB=${CUMLX_VM_DISK_GB:-70}
MIN_FREE_START_GB=65  # The image takes about 35 GB, the guest writes about 15 GB, and 15 GB stays free.
MIN_FREE_GB=15
MAX_USED_GB=38
MAX_USED_AT_START_GB=37  # Memory in use now, plus the VM's memory.
POLL=${CUMLX_VM_POLL:-30}
LIMIT_SECONDS=${CUMLX_VM_LIMIT_SECONDS:-28800}
VM_USER=${CUMLX_VM_USER:-admin}
# Only the environment holds it: it reaches ssh through the askpass script, and the guest script through the command line in the guest.
CUMLX_VM_PASSWORD=${CUMLX_VM_PASSWORD:-admin}
export CUMLX_VM_PASSWORD
case "$CUMLX_VM_PASSWORD" in
*\'*)
    echo "error: a password with a single quote is not supported" >&2
    exit 2
    ;;
esac
PROXY_PORT=${CUMLX_VM_PROXY_PORT:-38128}
GUEST_PROXY_PORT=3128
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO=$(dirname -- "$HERE")
ROOT=${CUMLX_VM_ROOT:-$HOME/cumlx-vm}
TART_HOME=$ROOT/tart-home
RESULT=$ROOT/result
export TART_HOME
TART=${CUMLX_VM_TART:-$ROOT/tart-dist/tart.app/Contents/MacOS/tart}

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
    [ -f "$ROOT/run.pid" ] && kill -0 "$(cat "$ROOT/run.pid")" 2>/dev/null
}
stage() {
    echo "[$(date +%H:%M:%S)] $1"
}
verdict() {
    mkdir -p "$RESULT"
    echo "$1" >"$RESULT/verdict.txt"
    stage "verdict: $1"
}
ask_password() {
    printf '#!/bin/sh\necho "$CUMLX_VM_PASSWORD"\n' >"$ROOT/askpass.sh"
    chmod 700 "$ROOT/askpass.sh"
}
SSH_OPTIONS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PreferredAuthentications=password -o ConnectTimeout=10 -o LogLevel=ERROR"
to_guest() {
    # shellcheck disable=SC2086  # SSH_OPTIONS holds only the fixed words above.
    SSH_ASKPASS="$ROOT/askpass.sh" SSH_ASKPASS_REQUIRE=force DISPLAY=:0 ssh $SSH_OPTIONS "$VM_USER@$GUEST" "$@"
}
start_proxy() {
    python3 "$HERE/m6_vm_proxy.py" "$PROXY_PORT" >"$ROOT/proxy.log" 2>&1 &
    PROXY_PID=$!
    trap 'kill "$PROXY_PID" "${TUNNEL_PID:-}" 2>/dev/null' EXIT
    trap 'exit 143' TERM
    start_tunnel
}
start_tunnel() {
    # exec, so that this pid is ssh itself, and the trap can end it.
    # shellcheck disable=SC2086
    (
        SSH_ASKPASS="$ROOT/askpass.sh" SSH_ASKPASS_REQUIRE=force DISPLAY=:0
        export SSH_ASKPASS SSH_ASKPASS_REQUIRE DISPLAY
        exec ssh $SSH_OPTIONS -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -N -R "$GUEST_PROXY_PORT:127.0.0.1:$PROXY_PORT" "$VM_USER@$GUEST"
    ) >>"$ROOT/proxy.log" 2>&1 &
    TUNNEL_PID=$!
}
stop_vm() {
    if [ "$MODE" = --work-attach ]; then
        "${CUMLX_VM_TERMINAL_MAC:-$ROOT/terminal-mac-src/dist/terminal-mac}" stop default >/dev/null 2>&1
        return 0
    fi
    "$TART" stop "$VM" >/dev/null 2>&1
    wait_pid=${RUN_PID:-}
    [ -n "$wait_pid" ] && kill "$wait_pid" 2>/dev/null
    return 0
}
fetch_result() {
    mkdir -p "$RESULT/guest"
    to_guest 'tar czf - -C "$HOME" repro-result' >"$ROOT/result.tgz" 2>/dev/null && tar xzf "$ROOT/result.tgz" -C "$RESULT/guest"
}
get_tart() {
    [ -n "${CUMLX_VM_TART:-}" ] && return 0
    if [ ! -x "$TART" ]; then
        mkdir -p "$ROOT/tart-dist"
        curl -fsSL -m 300 -o "$ROOT/tart-dist/tart.tar.gz" "$TART_URL" || return 1
    fi
    if [ -f "$ROOT/tart-dist/tart.tar.gz" ]; then
        found=$(python3 -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$ROOT/tart-dist/tart.tar.gz")
        if [ "$found" != "$TART_SHA256" ]; then
            echo "error: Tart's SHA-256 is $found, expected $TART_SHA256"
            return 1
        fi
        tar xzf "$ROOT/tart-dist/tart.tar.gz" -C "$ROOT/tart-dist" || return 1
    fi
    [ -x "$TART" ]
}

MODE=${1:-}
case "$MODE" in
status)
    if running; then echo "running, pid $(cat "$ROOT/run.pid")"; else echo "not running"; fi
    [ -f "$ROOT/run.log" ] && tail -n 6 "$ROOT/run.log"
    if [ -f "$RESULT/verdict.txt" ]; then
        echo "verdict: $(cat "$RESULT/verdict.txt")"
        [ "$(cat "$RESULT/verdict.txt")" = pass ]
        exit $?
    fi
    exit 5
    ;;
stop)
    "$TART" stop "$VM" 2>/dev/null
    running && kill "$(cat "$ROOT/run.pid")" 2>/dev/null
    echo "Stopped."
    exit 0
    ;;
clean)
    if running; then
        echo "error: the run is still going. Stop it first." >&2
        exit 4
    fi
    "$TART" delete "$VM" 2>/dev/null
    "$TART" delete "$IMAGE" 2>/dev/null
    [ -d "$TART_HOME" ] && rm -rf "$TART_HOME"
    echo "Deleted the VM and the image. The results stay in $RESULT"
    exit 0
    ;;
start)
    if [ "$(uname -sm)" != "Darwin arm64" ]; then
        echo "error: this needs an Apple Silicon Mac, found $(uname -sm)" >&2
        exit 3
    fi
    mkdir -p "$ROOT"
    if running; then
        echo "error: a run is going, pid $(cat "$ROOT/run.pid")" >&2
        exit 4
    fi
    if [ -f "$RESULT/verdict.txt" ]; then
        echo "error: $RESULT already holds a verdict. Move it away to run again." >&2
        exit 4
    fi
    now=$(free_gb)
    if [ "$now" -lt "$MIN_FREE_START_GB" ] && [ ! -d "$TART_HOME/vms/$VM" ]; then
        echo "error: needs $MIN_FREE_START_GB GB free, found $now GB" >&2
        exit 3
    fi
    used=$(used_gb)
    vm_gb=$(awk -v m="$VM_MEMORY_MB" 'BEGIN {printf "%.1f", m * 1048576 / 1e9}')
    if at_least "$(awk -v a="$used" -v b="$vm_gb" 'BEGIN {print a + b}')" "$MAX_USED_AT_START_GB"; then
        echo "error: $used GB of memory is in use, and the VM adds $vm_gb GB. That reaches $MAX_USED_AT_START_GB GB. Try again when this Mac is less busy." >&2
        exit 3
    fi
    nohup sh "$0" --work >"$ROOT/run.log" 2>&1 &
    echo $! >"$ROOT/run.pid"
    echo "Started, pid $(cat "$ROOT/run.pid"). Log: $ROOT/run.log"
    echo "See the state with: sh $0 status"
    exit 0
    ;;
attach)
    GUEST=${2:-}
    case "$GUEST" in
    "" | *[!0-9.]*)
        echo "usage: sh $0 attach IP. In the guest, ipconfig getifaddr en0 prints it." >&2
        exit 2
        ;;
    esac
    mkdir -p "$ROOT"
    if running; then
        echo "error: a run is going, pid $(cat "$ROOT/run.pid")" >&2
        exit 4
    fi
    if [ -f "$RESULT/verdict.txt" ]; then
        echo "error: $RESULT already holds a verdict. Move it away to run again." >&2
        exit 4
    fi
    now=$(free_gb)
    if [ "$now" -lt 30 ]; then
        echo "error: needs 30 GB free, found $now GB" >&2
        exit 3
    fi
    used=$(used_gb)
    if at_least "$used" "$MAX_USED_GB"; then
        echo "error: $used GB of memory is in use, at or over $MAX_USED_GB GB" >&2
        exit 3
    fi
    ask_password
    if ! to_guest true 2>/dev/null; then
        echo "error: cannot log in to $VM_USER@$GUEST. In the guest: that account, Remote Login on, and CUMLX_VM_USER and CUMLX_VM_PASSWORD set here." >&2
        exit 3
    fi
    nohup sh "$0" --work-attach "$GUEST" >"$ROOT/run.log" 2>&1 &
    echo $! >"$ROOT/run.pid"
    echo "Started, pid $(cat "$ROOT/run.pid"). Log: $ROOT/run.log"
    echo "See the state with: sh $0 status"
    exit 0
    ;;
--work | --work-attach) ;;
*)
    sed -n 2,13p "$0" >&2
    exit 2
    ;;
esac

ask_password
if [ "$MODE" = --work-attach ]; then
    GUEST=$2
    IMAGE="a Mac that was already running, not an image from this script"
    stage "a Mac that is already running, at $GUEST"
else
stage "1/6 Tart $TART_VERSION"
get_tart || { verdict setup_error_tart; exit 1; }
"$TART" --version

stage "2/6 the image $IMAGE"
"$TART" pull "$IMAGE" || { verdict setup_error_pull; exit 1; }
if [ "$(free_gb)" -le "$MIN_FREE_GB" ]; then
    verdict stopped_disk
    exit 1
fi

stage "3/6 the VM"
"$TART" clone "$IMAGE" "$VM" || { verdict setup_error_clone; exit 1; }
"$TART" set "$VM" --cpu "$VM_CPUS" --memory "$VM_MEMORY_MB" --disk-size "$VM_DISK_GB" || { verdict setup_error_set; exit 1; }
"$TART" run --no-graphics "$VM" >"$ROOT/vm.log" 2>&1 &
RUN_PID=$!

stage "4/6 waiting for the VM"
GUEST=
tries=0
while [ "$tries" -lt 120 ]; do
    GUEST=$("$TART" ip "$VM" --wait 5 2>/dev/null || true)
    [ -n "$GUEST" ] && to_guest true 2>/dev/null && break
    GUEST=
    tries=$((tries + 1))
    sleep "$POLL"
done
if [ -z "$GUEST" ]; then
    verdict setup_error_boot
    stop_vm
    exit 1
fi
fi

COMMIT=$(git -C "$REPO" rev-parse HEAD)
stage "5/6 the guest script. The guest gets CuMLX $COMMIT from the public repository"
# A verdict left by an earlier run must not read as this run's.
to_guest 'rm -rf "$HOME/repro-result"' || { verdict setup_error_copy; stop_vm; exit 1; }
to_guest 'cat > "$HOME/m6_vm_guest.sh"' <"$HERE/m6_vm_guest.sh" || { verdict setup_error_copy; stop_vm; exit 1; }
GUEST_ENV=
if [ -n "${CUMLX_VM_PROXY:-}" ]; then
    start_proxy
    GUEST_ENV="CUMLX_VM_PROXY=http://127.0.0.1:$GUEST_PROXY_PORT"
    mkdir -p "$RESULT"
    echo "The guest reached the internet through a web proxy on the host, over ssh -R, because the VM's NAT did not reach it." >"$RESULT/network.txt"
fi
to_guest "$GUEST_ENV CUMLX_VM_PASSWORD='$CUMLX_VM_PASSWORD' nohup caffeinate -ims sh \"\$HOME/m6_vm_guest.sh\" $COMMIT >\"\$HOME/guest.log\" 2>&1 </dev/null & echo started" || { verdict setup_error_guest; stop_vm; exit 1; }

stage "6/6 running in the guest. Limits on this Mac: $MAX_USED_GB GB of memory, $MIN_FREE_GB GB of free disk"
begin=$(date +%s)
last=
peak=0
while :; do
    used=$(used_gb)
    at_least "$used" "$peak" && peak=$used
    if at_least "$used" "$MAX_USED_GB"; then
        fetch_result
        verdict stopped_memory
        echo "memory in use reached $used GB" >"$RESULT/stop_reason.txt"
        stop_vm
        exit 1
    fi
    if [ "$(free_gb)" -le "$MIN_FREE_GB" ]; then
        fetch_result
        verdict stopped_disk
        echo "free disk fell to $(free_gb) GB" >"$RESULT/stop_reason.txt"
        stop_vm
        exit 1
    fi
    if [ $(($(date +%s) - begin)) -gt "$LIMIT_SECONDS" ]; then
        fetch_result
        verdict stopped_time
        stop_vm
        exit 1
    fi
    if [ -n "${TUNNEL_PID:-}" ] && ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
        start_tunnel
    fi
    now=$(to_guest 'cat "$HOME/repro-result/stage.txt" 2>/dev/null' 2>/dev/null || true)
    if [ -n "$now" ] && [ "$now" != "$last" ]; then
        stage "guest: $now"
        last=$now
    fi
    answer=$(to_guest 'cat "$HOME/repro-result/guest_verdict.txt" 2>/dev/null' 2>/dev/null || true)
    if [ -n "$answer" ]; then
        break
    fi
    sleep "$POLL"
done
fetch_result
echo "$peak" >"$RESULT/host_memory_peak_gb.txt"
echo "$COMMIT" >"$RESULT/cumlx_commit.txt"
echo "$IMAGE" >"$RESULT/image.txt"
verdict "$answer"
stop_vm
exit 0
