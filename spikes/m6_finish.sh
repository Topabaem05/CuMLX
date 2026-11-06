#!/bin/sh
# M6-1's remaining flagship work from one command: the CUDA parts on the V100 once its GPU is idle, the Mac parts from nothing on
# this Mac, and then the five verdicts. It returns at once.
#
#   sh spikes/m6_finish.sh start     # this checkout's commit, which must be clean and on origin/main
#   sh spikes/m6_finish.sh status    # the state, and the verdicts when they are written. Exit 0 all five pass, 1 not, 5 not done
#   sh spikes/m6_finish.sh stop      # stops the waiting here and the Mac run. A V100 run that started goes on: see spikes/m6_cuda_v100.sh
#
# The Mac run is spikes/m6_same_mac.sh start COMMIT m6, with its memory and disk limits. It starts as soon as this Mac has room, and
# its F1 part waits for the V100's golden. The V100 run is spikes/m6_cuda_v100.sh, which refuses a busy GPU, so this tries again
# every 10 minutes for up to 30 hours. Results: ~/cumlx-m6-cuda (V100), ~/cumlx-repro/result (Mac), ~/cumlx-m6-finish (verdicts).
set -u
REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
ROOT=${CUMLX_FINISH_ROOT:-$HOME/cumlx-m6-finish}
CUDA_DIR=${CUMLX_M6_CUDA:-$HOME/cumlx-m6-cuda}
REPRO=${CUMLX_REPRO_ROOT:-$HOME/cumlx-repro}
WAIT_GPU_SECONDS=${CUMLX_FINISH_GPU_WAIT_SECONDS:-108000}
POLL=${CUMLX_FINISH_POLL:-600}
SAME_MAC=${CUMLX_FINISH_SAME_MAC:-$REPO/spikes/m6_same_mac.sh}
V100=${CUMLX_FINISH_V100:-$REPO/spikes/m6_cuda_v100.sh}
PY=${CUMLX_FINISH_PYTHON:-$REPO/.venv/bin/python}

running() {
    [ -f "$ROOT/run.pid" ] && kill -0 "$(cat "$ROOT/run.pid")" 2>/dev/null
}
say() { echo "[$(date '+%m-%d %H:%M')] $1"; echo "$1" >"$ROOT/stage.txt"; }

case "${1:-}" in
status)
    if running; then echo "running, pid $(cat "$ROOT/run.pid")"; else echo "not running"; fi
    [ -f "$ROOT/stage.txt" ] && echo "stage: $(cat "$ROOT/stage.txt")"
    [ -f "$ROOT/summary.txt" ] || exit 5
    cat "$ROOT/summary.txt"
    grep -q '^all: pass$' "$ROOT/summary.txt"
    exit $?
    ;;
stop)
    if running; then
        kill "$(cat "$ROOT/run.pid")" 2>/dev/null
        echo stopped_by_person >"$ROOT/stage.txt"
    fi
    sh "$SAME_MAC" stop
    echo "Stopped here. A run on the V100 goes on until it ends; see: sh $V100 status"
    exit 0
    ;;
start)
    if running; then
        echo "error: already running, pid $(cat "$ROOT/run.pid")" >&2
        exit 4
    fi
    if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
        echo "error: the checkout has changes that are not committed. The V100 and the Mac must run the same public commit." >&2
        exit 2
    fi
    COMMIT=$(git -C "$REPO" rev-parse HEAD)
    git -C "$REPO" fetch -q origin main 2>/dev/null
    if ! git -C "$REPO" merge-base --is-ancestor "$COMMIT" origin/main 2>/dev/null; then
        echo "error: $COMMIT is not on origin/main, and the Mac run fetches it from the public repository" >&2
        exit 2
    fi
    if [ -e "$CUDA_DIR" ]; then
        echo "error: $CUDA_DIR exists. Move it away first, so that an earlier run cannot be read as this one." >&2
        exit 4
    fi
    mkdir -p "$ROOT"
    rm -f "$ROOT/summary.txt" "$ROOT"/f*.json
    nohup sh "$0" --work "$COMMIT" >"$ROOT/run.log" 2>&1 &
    echo $! >"$ROOT/run.pid"
    echo "Started at $COMMIT, pid $(cat "$ROOT/run.pid"). See the state with: sh $0 status"
    exit 0
    ;;
--work) ;;
*)
    sed -n 2,12p "$0" >&2
    exit 2
    ;;
esac

COMMIT=${2:?the commit}
cd "$REPO" || exit 1

say "starting the Mac run"
waited=0
while :; do
    CUMLX_F1_GOLDEN=$CUDA_DIR/f1/golden CUMLX_F1_WAIT_SECONDS=$WAIT_GPU_SECONDS CUMLX_REPRO_LIMIT_SECONDS=$((WAIT_GPU_SECONDS + 28800)) sh "$SAME_MAC" start "$COMMIT" m6
    code=$?
    [ "$code" = 0 ] && break
    if [ "$code" != 3 ] || [ "$waited" -ge 7200 ]; then
        say "the Mac run did not start, exit $code"
        break
    fi
    sleep 300
    waited=$((waited + 300))
done

say "waiting for the V100's GPU to be idle"
waited=0
while :; do
    sh "$V100" start "f1 f2 f3 f5"
    code=$?
    [ "$code" = 0 ] && break
    if [ "$code" != 3 ] || [ "$waited" -ge "$WAIT_GPU_SECONDS" ]; then
        say "the V100 run did not start, exit $code"
        break
    fi
    sleep "$POLL"
    waited=$((waited + POLL))
done
# Exit 1 can also mean that the run ended before the launcher looked, so both codes follow the run on the GPU machine.
if [ "$code" = 0 ] || [ "$code" = 1 ]; then
    say "the V100 run is going"
    while :; do
        state=$(sh "$V100" status 2>&1)
        if [ $? = 0 ]; then
            sh "$V100" fetch
            say "the V100 run is done, and its results are in $CUDA_DIR"
            break
        fi
        if echo "$state" | grep -q '^not running'; then
            say "the V100 run ended without finishing: $(echo "$state" | tr '\n' ' ')"
            break
        fi
        sleep 120
    done
fi

say "waiting for the Mac run"
while [ ! -f "$REPRO/result/verdict.txt" ]; do
    sleep 120
done
MAC=$REPRO/result
say "judging"
{
    echo "commit: $COMMIT"
    echo "mac run: $(cat "$MAC/verdict.txt")"
    for f in f1 f2 f3 f4 f5; do
        case $f in
        f1) "$PY" spikes/m6_f1_judge.py --cuda "$CUDA_DIR/f1" --mac "$MAC/f1" --output "$ROOT/f1.json" >/dev/null 2>"$ROOT/f1.err" ;;
        f2) "$PY" spikes/m6_f2_cuda_judge.py --cuda "$CUDA_DIR/f2" --mac "$MAC/f2cuda" --output "$ROOT/f2.json" >/dev/null 2>"$ROOT/f2.err" ;;
        f3) "$PY" spikes/m6_f3_judge.py --cuda "$CUDA_DIR/f3" --mac "$MAC/f3" --output "$ROOT/f3.json" >/dev/null 2>"$ROOT/f3.err" ;;
        f4) "$PY" spikes/m6_f4_judge.py --run "$MAC/f4" --output "$ROOT/f4.json" >/dev/null 2>"$ROOT/f4.err" ;;
        f5) "$PY" spikes/m6_f5_judge.py --cuda "$CUDA_DIR/f5" --mac "$MAC/f5" --output "$ROOT/f5.json" >/dev/null 2>"$ROOT/f5.err" ;;
        esac
        case $? in
        0) echo "$f: pass" ;;
        1) echo "$f: fail" ;;
        *) echo "$f: not_run" ;;
        esac
    done
} >"$ROOT/summary.partial"
if [ "$(grep -c ': pass$' "$ROOT/summary.partial")" = 5 ] && grep -q '^mac run: mac_done$' "$ROOT/summary.partial"; then
    echo "all: pass" >>"$ROOT/summary.partial"
else
    echo "all: not a pass" >>"$ROOT/summary.partial"
fi
mv "$ROOT/summary.partial" "$ROOT/summary.txt"
say "done"
