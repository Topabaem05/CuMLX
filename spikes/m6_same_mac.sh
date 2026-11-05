#!/bin/sh
# M6-1 on the one Mac that there is: the VM's work again, from the public repository, with an empty home folder. One command, and it returns at once.
#
#   sh spikes/m6_same_mac.sh start [COMMIT]  # by default the commit of this checkout, which must be public
#   sh spikes/m6_same_mac.sh start COMMIT f1f2  # the F1 and F2 work in spikes/m6_f1_f2_work.sh in place of the rest, under validation/m6-f1-f2-limits.json
#   sh spikes/m6_same_mac.sh status          # the state, and the verdict when it is done. Exit 0 pass, 1 not a pass, 5 no verdict yet
#   sh spikes/m6_same_mac.sh stop            # stops the run
#   sh spikes/m6_same_mac.sh clean           # deletes the run's home folder, and keeps the results
#
# It uses no sudo and changes no setting. Everything goes in CUMLX_REPRO_ROOT (default ~/cumlx-repro), and the run's HOME is a folder there, so this person's shell files and caches are not read or changed.
# This is not a new machine: macOS, the Metal driver, the GPU, git and curl are this Mac's. validation/m6-same-mac-limits.json holds the rules, and a pass does not tick M6-1.
# It stops at 38 GB of memory in use or at 15 GB of free disk, and a heavy part starts at 33 GB or less. A stop is never a pass.
set -u
MAX_USED_GB=38
MAX_USED_TO_START_GB=33
MIN_FREE_GB=15
MIN_FREE_START_GB=30
POLL=${CUMLX_REPRO_POLL:-30}
LIMIT_SECONDS=${CUMLX_REPRO_LIMIT_SECONDS:-28800}
ROOM_WAIT_SECONDS=${CUMLX_REPRO_ROOM_WAIT_SECONDS:-3600}
ROOT=${CUMLX_REPRO_ROOT:-$HOME/cumlx-repro}
RESULT=$ROOT/result
MARK=$ROOT/.cumlx-repro
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

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
stop_group() {
    pid=$(cat "$ROOT/run.pid" 2>/dev/null) || return 0
    # The run has its own process group, so this reaches only what the run started.
    kill -TERM "-$pid" 2>/dev/null
    tries=0
    while kill -0 "-$pid" 2>/dev/null && [ "$tries" -lt 20 ]; do
        sleep 1
        tries=$((tries + 1))
    done
    kill -KILL "-$pid" 2>/dev/null
}

MODE=${1:-}
case "$MODE" in
status)
    if running; then echo "running, pid $(cat "$ROOT/run.pid")"; else echo "not running"; fi
    [ -f "$RESULT/stage.txt" ] && echo "stage: $(cat "$RESULT/stage.txt")"
    [ -f "$RESULT/peak_used_gb.txt" ] && echo "peak memory in use: $(cat "$RESULT/peak_used_gb.txt") GB"
    if [ ! -f "$RESULT/verdict.txt" ]; then
        if running; then echo "no verdict yet"; else echo "no verdict: the run ended without one, and that is not a pass. See $ROOT/run.log"; fi
        exit 5
    fi
    verdict=$(cat "$RESULT/verdict.txt")
    echo "verdict: $verdict"
    [ "$verdict" = pass ]
    exit $?
    ;;
stop)
    if running; then
        echo stopped_by_person >"$RESULT/verdict.txt"
        stop_group
    fi
    echo "Stopped."
    exit 0
    ;;
clean)
    if running; then
        echo "error: a run is going, pid $(cat "$ROOT/run.pid")" >&2
        exit 4
    fi
    [ -f "$MARK" ] && rm -rf "${ROOT:?}/home"
    echo "The results stay in $RESULT"
    exit 0
    ;;
start)
    if [ "$(uname -sm)" != "Darwin arm64" ]; then
        echo "error: this needs an Apple Silicon Mac, found $(uname -sm)" >&2
        exit 3
    fi
    COMMIT=${2:-$(git -C "$SCRIPT_DIR/.." rev-parse HEAD 2>/dev/null)}
    if [ -z "$COMMIT" ]; then
        echo "error: name a commit" >&2
        exit 2
    fi
    if [ -e "$ROOT" ] && [ ! -f "$MARK" ] && [ -n "$(ls -A "$ROOT" 2>/dev/null)" ]; then
        echo "error: $ROOT holds other files, and they stay. Set CUMLX_REPRO_ROOT to another folder." >&2
        exit 4
    fi
    mkdir -p "$ROOT"
    if running; then
        echo "error: a run is going, pid $(cat "$ROOT/run.pid")" >&2
        exit 4
    fi
    now=$(free_gb)
    if [ "$now" -lt "$MIN_FREE_START_GB" ]; then
        echo "error: needs $MIN_FREE_START_GB GB free, found $now GB" >&2
        exit 3
    fi
    used=$(used_gb)
    if ! at_least "$MAX_USED_TO_START_GB" "$used"; then
        echo "error: $used GB of memory is in use, above the $MAX_USED_TO_START_GB GB that a heavy job starts at. Try again when this Mac is less busy." >&2
        exit 3
    fi
    touch "$MARK"
    # An earlier run's results stay, under another name, and its verdict cannot be read as this run's.
    [ -d "$RESULT" ] && mv "$RESULT" "$RESULT-$(date +%Y%m%d-%H%M%S)"
    rm -rf "${ROOT:?}/home"
    mkdir -p "$RESULT" "$ROOT/home"
    # macOS's python3 writes a cache under HOME, so even the launcher gets the run's own.
    HOME=$ROOT/home CUMLX_REPRO_ROOT=$ROOT nohup python3 -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' sh "$0" --work "$COMMIT" "${3:-flagship}" >"$ROOT/run.log" 2>&1 &
    echo $! >"$ROOT/run.pid"
    nohup sh "$0" --watch >"$ROOT/watch.log" 2>&1 &
    echo "Started at $COMMIT, pid $(cat "$ROOT/run.pid"). See the state with: sh $0 status"
    exit 0
    ;;
--watch)
    start=$(date +%s)
    peak=0
    pid=$(cat "$ROOT/run.pid")
    caffeinate -ims -w "$pid" 2>/dev/null &
    while kill -0 "$pid" 2>/dev/null; do
        used=$(used_gb)
        peak=$(awk -v a="$used" -v b="$peak" 'BEGIN {print (a > b) ? a : b}')
        echo "$peak" >"$RESULT/peak_used_gb.txt"
        reason=
        at_least "$used" "$MAX_USED_GB" && reason="stopped_memory_$used"
        [ "$(free_gb)" -le "$MIN_FREE_GB" ] && reason=stopped_disk
        [ $(($(date +%s) - start)) -ge "$LIMIT_SECONDS" ] && reason=stopped_time
        if [ -n "$reason" ]; then
            echo "$reason" >"$RESULT/verdict.txt"
            echo "$reason" >"$RESULT/stage.txt"
            stop_group
            exit 0
        fi
        sleep "$POLL"
    done
    exit 0
    ;;
--work) ;;
*)
    sed -n 2,11p "$0" >&2
    exit 2
    ;;
esac

COMMIT=${2:?the commit of CuMLX to test}
PART=${3:-flagship}
OUT=$RESULT
# From here on, nothing of this person's environment reaches the run.
HOME=$ROOT/home
PATH=${CUMLX_REPRO_PATH:-/usr/bin:/bin:/usr/sbin:/sbin}
export HOME PATH
unset VIRTUAL_ENV CONDA_PREFIX PYTHONPATH PYTHONHOME PIP_INDEX_URL PIP_CONFIG_FILE XDG_CONFIG_HOME XDG_CACHE_HOME XDG_DATA_HOME HF_HOME TORCH_HOME CUMLX_MAX_USED_GB CUMLX_START_USED_GB UV_CACHE_DIR UV_PYTHON_INSTALL_DIR UV_INDEX_URL UV_DEFAULT_INDEX
cd "$HOME" || exit 1

stage() { echo "[$(date +%H:%M:%S)] $1"; echo "$1" >"$OUT/stage.txt"; }
finish() {
    # A stop by the watcher or a person has written its verdict first, and it stays.
    [ -f "$OUT/verdict.txt" ] || echo "$1" >"$OUT/verdict.txt"
    stage "finished: $(cat "$OUT/verdict.txt")"
    exit 0
}
wait_for_room() {
    waited=0
    while ! at_least "$MAX_USED_TO_START_GB" "$(used_gb)"; do
        [ "$waited" -ge "$ROOM_WAIT_SECONDS" ] && finish "stopped_memory_before_$1"
        sleep "$POLL"
        waited=$((waited + ${POLL%.*} + 1))
    done
}

stage "1/7 what this machine is"
{
    sw_vers
    uname -m
    sysctl -n machdep.cpu.brand_string hw.memsize hw.ncpu
    system_profiler SPDisplaysDataType
    df -k "$HOME"
    git --version
} >"$OUT/machine.txt" 2>&1
curl -sS -m 30 -o /dev/null https://github.com >"$OUT/internet.txt" 2>&1 || finish setup_error_no_internet

stage "2/7 uv, at a fixed version"
# UV_NO_MODIFY_PATH keeps the installer from editing a shell file, and HOME here is the run's own.
curl -LsSf https://astral.sh/uv/0.12.19/install.sh | UV_NO_MODIFY_PATH=1 sh >"$OUT/uv.txt" 2>&1
PATH=$HOME/.local/bin:$PATH
export PATH
command -v uv >/dev/null 2>&1 || finish setup_error_uv

stage "3/7 CuMLX from its public repository, at $COMMIT"
mkdir -p "$HOME/code/CuMLX"
cd "$HOME/code/CuMLX" || finish setup_error_folder
git init -q . && git remote add origin https://github.com/Topabaem05/CuMLX || finish setup_error_git
git fetch -q --depth 1 origin "$COMMIT" && git checkout -q FETCH_HEAD || finish setup_error_public_commit
echo "$COMMIT" >"$OUT/cumlx_commit.txt"
uv venv -q --python 3.12 .venv || finish setup_error_venv
uv pip install -q --python .venv/bin/python -e '.[dev,capture,diff,render,sim]' rsl-rl-lib==5.5.2 >"$OUT/install.txt" 2>&1 || finish setup_error_install
uv pip freeze --python .venv/bin/python >"$OUT/freeze.txt" 2>&1
PY=$PWD/.venv/bin/python

if [ "$PART" = f1f2 ]; then
    stage "F1 and F2"
    wait_for_room f1f2
    sh spikes/m6_f1_f2_work.sh "$OUT" >"$OUT/f1f2.log" 2>&1
    finish "$(cat "$OUT/f1f2_verdict.txt" 2>/dev/null || echo not_run_no_verdict)"
fi

stage "4/7 the base: doctor, self test, tests without a GPU"
.venv/bin/cumlx doctor --require capture,diff --format json >"$OUT/doctor.json" 2>"$OUT/doctor.err"
echo "$?" >"$OUT/doctor.exit"
.venv/bin/cumlx selftest --format json >"$OUT/selftest.json" 2>"$OUT/selftest.err"
echo "$?" >"$OUT/selftest.exit"
"$PY" -c 'import json, mlx.core as mx; print(json.dumps({"default_device": str(mx.default_device()), "metal": mx.metal.is_available(), "info": {k: str(v) for k, v in mx.metal.device_info().items()}, "sum": float(mx.array([1.0, 2.0]).sum())}))' >"$OUT/mlx_device.json" 2>"$OUT/mlx_device.err"
"$PY" -m pytest -q -m 'not gpu' >"$OUT/tests.txt" 2>&1
echo "$?" >"$OUT/tests.exit"

stage "5/7 F3: the hand-written Metal kernels against the V100 goldens"
wait_for_room f3
"$PY" -m pytest -q -rs tests/test_kernel_gpu.py >"$OUT/f3.txt" 2>&1
echo "$?" >"$OUT/f3.exit"

stage "6/7 F4: Go2 on Genesis through cumlx.sim, then the MLX policy"
wait_for_room f4
"$PY" spikes/m4_sim.py train --task go2_flat --engine genesis --envs 4096 --iterations 300 --seed 1 --log-dir "$HOME/f4" >"$OUT/f4_train.txt" 2>&1
echo "$?" >"$OUT/f4_train.exit"
CK=$HOME/f4/model_299.pt
if [ -f "$CK" ]; then
    "$PY" spikes/m4_sim.py eval --task go2_flat --engine genesis --envs 128 --steps 1000 --seed 7 --checkpoint "$CK" --forward 0.8 --output "$OUT/f4_fwd.json" >"$OUT/f4_fwd.txt" 2>&1
    "$PY" spikes/m4_sim.py eval --task go2_flat --engine genesis --envs 128 --steps 1000 --seed 7 --checkpoint "$CK" --forward -0.8 --output "$OUT/f4_back.json" >"$OUT/f4_back.txt" 2>&1
    "$PY" spikes/m4_sim.py eval --task go2_flat --engine genesis --envs 128 --steps 1000 --seed 7 --checkpoint "$CK" --forward 0 --output "$OUT/f4_stop.json" >"$OUT/f4_stop.txt" 2>&1
    "$PY" spikes/m4_sim.py eval --task go2_flat --engine genesis --envs 128 --steps 1000 --seed 7 --forward 0.8 --output "$OUT/f4_zero.json" >"$OUT/f4_zero.txt" 2>&1
fi

stage "7/7 F5: render on Metal"
wait_for_room f5
"$PY" spikes/m5_render.py render --variant metal_ad_rgb --seeds 11,12,13 --out "$OUT/f5" >"$OUT/f5.txt" 2>&1
echo "$?" >"$OUT/f5.exit"

"$PY" spikes/m6_vm_judge.py "$OUT" --limits validation/m6-same-mac-limits.json >"$OUT/judge.txt" 2>&1
code=$?
case "$code" in
0) finish pass ;;
1) finish fail ;;
*) finish "not_run_exit_$code" ;;
esac
