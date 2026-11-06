#!/bin/sh
# M6-1's remaining flagship work from one command: the CUDA parts on the V100 once its GPU is idle, the Mac parts from nothing on
# this Mac, and then the five verdicts. It returns at once.
#
#   sh spikes/m6_finish.sh start     # this checkout's commit, which must be clean and on origin/main
#   sh spikes/m6_finish.sh status    # the state, and the verdicts when they are written. Exit 0 all five pass, 1 not, 5 not done
#   sh spikes/m6_finish.sh stop      # stops the waiting here and the Mac run. A V100 run that started goes on: see spikes/m6_cuda_v100.sh
#
# The work runs from a snapshot of the commit in ~/cumlx-m6-finish/code, so that a later change to this checkout cannot change it.
# The Mac run is spikes/m6_same_mac.sh start COMMIT m6, with its memory and disk limits, and its F1 part waits for the V100's golden.
# The V100 run is spikes/m6_cuda_v100.sh at the same commit; it refuses a busy GPU, so this tries again every 10 minutes for up to
# 30 hours. Results: ~/cumlx-m6-cuda (V100), ~/cumlx-repro/result (Mac), ~/cumlx-m6-finish (verdicts).
set -u
ROOT=${CUMLX_FINISH_ROOT:-$HOME/cumlx-m6-finish}
CUDA_DIR=${CUMLX_M6_CUDA:-$HOME/cumlx-m6-cuda}
REPRO=${CUMLX_REPRO_ROOT:-$HOME/cumlx-repro}
WAIT_GPU_SECONDS=${CUMLX_FINISH_GPU_WAIT_SECONDS:-108000}
CUDA_RUN_SECONDS=${CUMLX_FINISH_CUDA_RUN_SECONDS:-86400}
POLL=${CUMLX_FINISH_POLL:-600}
SHORT_POLL=${CUMLX_FINISH_SHORT_POLL:-120}

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
    sh "${CUMLX_FINISH_SAME_MAC:-$ROOT/code/spikes/m6_same_mac.sh}" stop
    echo "Stopped here. A run on the V100 goes on until it ends; see: sh spikes/m6_cuda_v100.sh status"
    exit 0
    ;;
start)
    REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
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
    if [ -e "$CUDA_DIR" ] || [ -e "$CUDA_DIR.partial" ]; then
        echo "error: $CUDA_DIR exists. Move it away first, so that an earlier run cannot be read as this one." >&2
        exit 4
    fi
    mkdir -p "$ROOT"
    rm -rf "$ROOT/code" "$ROOT/summary.txt" "$ROOT/cuda_ready.txt" "$ROOT"/f*.json "$ROOT"/f*.err
    mkdir -p "$ROOT/code"
    git -C "$REPO" archive "$COMMIT" spikes src validation | tar -x -C "$ROOT/code" || { echo "error: could not take a snapshot of $COMMIT" >&2; exit 3; }
    CUMLX_FINISH_REPO=$REPO nohup sh "$ROOT/code/spikes/m6_finish.sh" --work "$COMMIT" >"$ROOT/run.log" 2>&1 &
    echo $! >"$ROOT/run.pid"
    echo "Started at $COMMIT, pid $(cat "$ROOT/run.pid"). See the state with: sh $0 status"
    exit 0
    ;;
--work) ;;
*)
    sed -n 2,13p "$0" >&2
    exit 2
    ;;
esac

COMMIT=${2:?the commit}
REPO=${CUMLX_FINISH_REPO:?the git checkout}
CODE=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
SAME_MAC=${CUMLX_FINISH_SAME_MAC:-$CODE/spikes/m6_same_mac.sh}
V100=${CUMLX_FINISH_V100:-$CODE/spikes/m6_cuda_v100.sh}
PY=${CUMLX_FINISH_PYTHON:-$REPO/.venv/bin/python}
READY=$ROOT/cuda_ready.txt
cd "$CODE" || exit 1
NOTES=""
note() { NOTES="$NOTES$1
"; }

say "starting the Mac run"
mac_started=no
waited=0
while :; do
    CUMLX_F1_GOLDEN=$CUDA_DIR/f1/golden CUMLX_F1_READY=$READY CUMLX_F1_WAIT_SECONDS=$((WAIT_GPU_SECONDS + CUDA_RUN_SECONDS)) \
        CUMLX_REPRO_LIMIT_SECONDS=$((WAIT_GPU_SECONDS + CUDA_RUN_SECONDS + 28800)) sh "$SAME_MAC" start "$COMMIT" m6
    code=$?
    if [ "$code" = 0 ]; then
        mac_started=yes
        break
    fi
    if [ "$code" != 3 ] || [ "$waited" -ge 7200 ]; then
        say "the Mac run did not start, exit $code"
        note "mac run: not started, exit $code"
        break
    fi
    sleep 300
    waited=$((waited + 300))
done

fetch() {
    # Up to five tries, two minutes apart: one dropped ssh at the end of a long run must not lose its results.
    tries=0
    while [ "$tries" -lt 5 ]; do
        CUMLX_M6_CUDA=$CUDA_DIR sh "$V100" fetch "$@" && return 0
        [ -e "$CUDA_DIR" ] && return 1
        tries=$((tries + 1))
        sleep "$SHORT_POLL"
    done
    return 1
}

cuda=none
gpu_waited=0
busy_tries=0
while :; do
    say "waiting for the V100's GPU to be idle"
    while :; do
        CUMLX_V100_GIT=$REPO CUMLX_V100_COMMIT=$COMMIT CUMLX_M6_CUDA=$CUDA_DIR sh "$V100" start "f1 f2 f3 f5"
        code=$?
        [ "$code" = 0 ] && break
        if [ "$code" != 3 ] || [ "$gpu_waited" -ge "$WAIT_GPU_SECONDS" ]; then
            say "the V100 run did not start, exit $code"
            break
        fi
        sleep "$POLL"
        gpu_waited=$((gpu_waited + POLL))
    done
    # Exit 1 can also mean that the run ended before the launcher looked, so both codes follow the run on the GPU machine.
    [ "$code" = 0 ] || [ "$code" = 1 ] || break
    say "the V100 run is going"
    waited=0
    while :; do
        state=$(sh "$V100" status 2>&1)
        if [ $? = 0 ]; then
            if fetch; then
                cuda=fetched
                say "the V100 run is done, and its results are in $CUDA_DIR"
            else
                say "the V100 run is done, and its results could not be fetched"
            fi
            break
        fi
        if echo "$state" | grep -q '^not running' || [ "$waited" -ge "$CUDA_RUN_SECONDS" ]; then
            # What finished stays useful: each judge decides its own part, and incomplete.txt keeps this from being a pass.
            if fetch --incomplete; then
                cuda=incomplete
                say "the V100 run did not finish, and what it left is in $CUDA_DIR"
            else
                say "the V100 run did not finish, and nothing could be fetched"
            fi
            break
        fi
        sleep "$SHORT_POLL"
        waited=$((waited + SHORT_POLL))
    done
    # Another job took the GPU in the seconds between the launcher's check and the run's. Wait for the GPU again while there is time.
    if [ -f "$CUDA_DIR/busy_at_start.txt" ] && [ "$gpu_waited" -lt "$WAIT_GPU_SECONDS" ] && [ "$busy_tries" -lt 5 ]; then
        busy_tries=$((busy_tries + 1))
        mv "$CUDA_DIR" "$CUDA_DIR.busy-$busy_tries"
        note_busy="cuda: the GPU was taken before the run began $busy_tries time(s); those copies are in $CUDA_DIR.busy-N"
        cuda=none
        continue
    fi
    break
done
[ -n "${note_busy:-}" ] && say "$note_busy"
[ "$cuda" = fetched ] || note "cuda: $cuda"
[ -f "$CUDA_DIR/busy_at_start.txt" ] && note "cuda: another process took the GPU before the run began: $(cat "$CUDA_DIR/busy_at_start.txt" | tr '\n' ' ')"
[ "$cuda" != none ] && [ "$(cat "$CUDA_DIR/cumlx_commit.txt" 2>/dev/null)" != "$COMMIT" ] && note "cuda: ran $(cat "$CUDA_DIR/cumlx_commit.txt" 2>/dev/null), not $COMMIT"
# The Mac's F1 part waits for this file, so it is written whether or not there are results.
echo "$cuda" >"$READY"

MAC=$REPRO/result
if [ "$mac_started" = yes ]; then
    say "waiting for the Mac run"
    while [ ! -f "$MAC/verdict.txt" ]; do
        state=$(sh "$SAME_MAC" status 2>&1)
        if echo "$state" | grep -q '^not running' && [ ! -f "$MAC/verdict.txt" ]; then
            note "mac run: ended with no verdict"
            break
        fi
        sleep "$SHORT_POLL"
    done
    [ "$(cat "$MAC/cumlx_commit.txt" 2>/dev/null)" = "$COMMIT" ] || note "mac run: ran $(cat "$MAC/cumlx_commit.txt" 2>/dev/null), not $COMMIT"
fi

say "judging"
{
    echo "commit: $COMMIT"
    echo "mac run: $( [ "$mac_started" = yes ] && cat "$MAC/verdict.txt" 2>/dev/null || echo not_started)"
    printf '%s' "$NOTES"
    for f in f1 f2 f3 f4 f5; do
        if [ "$mac_started" != yes ]; then
            echo "$f: not_run"
            continue
        fi
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
if [ "$(grep -c '^f[1-5]: pass$' "$ROOT/summary.partial")" = 5 ] && grep -q '^mac run: mac_done$' "$ROOT/summary.partial" && [ -z "$NOTES" ]; then
    echo "all: pass" >>"$ROOT/summary.partial"
else
    echo "all: not a pass" >>"$ROOT/summary.partial"
fi
mv "$ROOT/summary.partial" "$ROOT/summary.txt"
say "done"
