#!/bin/sh
# The CUDA side of M6-1's flagship work on a shared GPU machine. One command, and it returns at once.
#
#   sh spikes/m6_cuda_v100.sh start [PARTS]   # copies the committed repo and runs the parts, by default "f1 f2 f3 f5"
#   sh spikes/m6_cuda_v100.sh status          # the state of the run. Exit 0 done, 5 running or not done
#   sh spikes/m6_cuda_v100.sh fetch           # copies the results to ~/cumlx-m6-cuda on this Mac
#
# It refuses to start while another process is on the GPU. The host is CUMLX_GPU_HOST, or the content of ~/.config/cumlx/gpu-host. There is no default.
# Everything goes in ~/cumlx-m6 there. Each part installs what it needs into its own --target folder under work/, and the
# Hugging Face cache and gsplat's JIT build stay there too. torch and NumPy are the machine's own (python3, user site). Nothing outside that folder changes.
# The parts are spikes/remote/m6_<part>.sh. Delete the remote folder when the test is finished: ssh <host> rm -rf cumlx-m6
set -u

HOST=${CUMLX_GPU_HOST:-}
HOST_FILE=${XDG_CONFIG_HOME:-$HOME/.config}/cumlx/gpu-host
if [ -z "$HOST" ] && [ -f "$HOST_FILE" ]; then
    HOST=$(cat "$HOST_FILE")
fi
if [ -z "$HOST" ]; then
    echo "error: no GPU machine is set. Set CUMLX_GPU_HOST to an ssh host with an NVIDIA GPU, or write that host to $HOST_FILE" >&2
    exit 3
fi
REMOTE=cumlx-m6
REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
# spikes/m6_finish.sh runs a snapshot of this script, so the git checkout and the commit can be given.
GIT_REPO=${CUMLX_V100_GIT:-$REPO}
LOCAL=${CUMLX_M6_CUDA:-$HOME/cumlx-m6-cuda}

case "${1:-start}" in
status)
    ssh "$HOST" "if [ -f $REMOTE/run.pid ] && kill -0 \$(cat $REMOTE/run.pid) 2>/dev/null; then echo running; else echo not running; fi; cat $REMOTE/out/stage.txt 2>/dev/null; test -f $REMOTE/out/done.txt && echo done"
    ssh "$HOST" "test -f $REMOTE/out/done.txt"
    [ $? = 0 ] && exit 0
    exit 5
    ;;
fetch)
    if ! ssh "$HOST" "test -f $REMOTE/out/done.txt"; then
        echo "error: $HOST has no finished results yet" >&2
        exit 1
    fi
    if [ -e "$LOCAL" ]; then
        echo "error: $LOCAL exists" >&2
        exit 4
    fi
    # Into a side folder first, and then one rename, so that nothing reads a half-copied result.
    rm -rf "$LOCAL.partial"
    mkdir -p "$LOCAL.partial"
    ssh "$HOST" "cd $REMOTE/out && tar -c ." | tar -x -C "$LOCAL.partial" || exit 1
    mv "$LOCAL.partial" "$LOCAL" || exit 1
    echo "Copied to $LOCAL"
    exit 0
    ;;
start) ;;
*)
    sed -n 2,11p "$0" >&2
    exit 2
    ;;
esac

PARTS=${2:-f1 f2 f3 f5}
for part in $PARTS; do
    [ -f "$REPO/spikes/remote/m6_$part.sh" ] || { echo "error: no part $part (spikes/remote/m6_$part.sh)" >&2; exit 2; }
done
BUSY=$(ssh "$HOST" "nvidia-smi --query-compute-apps=pid --format=csv,noheader") || {
    echo "error: could not reach $HOST or read its GPU" >&2
    exit 3
}
if [ -n "$BUSY" ]; then
    echo "error: another process is on the GPU on $HOST, pid $BUSY. Nothing was started." >&2
    exit 3
fi
if ssh "$HOST" "test -f $REMOTE/run.pid && kill -0 \$(cat $REMOTE/run.pid) 2>/dev/null"; then
    echo "error: a run is going on $HOST" >&2
    exit 4
fi
COMMIT=${CUMLX_V100_COMMIT:-$(git -C "$GIT_REPO" rev-parse HEAD)}
ssh "$HOST" "rm -rf $REMOTE/CuMLX $REMOTE/out && mkdir -p $REMOTE/CuMLX $REMOTE/out $REMOTE/work" || exit 3
git -C "$GIT_REPO" archive "$COMMIT" spikes src/cumlx ':(glob)validation/m6-*.json' validation/kernel-cases-m3 | ssh "$HOST" "tar -x -C $REMOTE/CuMLX" || exit 3
echo "$COMMIT" | ssh "$HOST" "cat > $REMOTE/out/cumlx_commit.txt" || exit 3
ssh "$HOST" "cat > $REMOTE/run.sh" <<EOF
#!/bin/bash
cd "\$(dirname "\$0")"
{ nvidia-smi; nvidia-smi --query-compute-apps=pid,used_memory --format=csv; nvcc --version; python3 --version; } > out/machine_start.txt 2>&1
# The launcher looked a moment ago. If another process took the GPU since, this run is not a pass, so it stops here.
BUSY=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [ -n "\$BUSY" ]; then
    echo "\$BUSY" > out/busy_at_start.txt
    echo busy_at_start > out/stage.txt
    date -u +%Y-%m-%dT%H:%M:%SZ > out/done.txt
    exit 0
fi
for part in $PARTS; do
    echo "\$part" > out/stage.txt
    bash CuMLX/spikes/remote/m6_\$part.sh "\$PWD/CuMLX" "\$PWD/out/\$part" "\$PWD/work/\$part" > out/\$part.log 2>&1
    echo \$? > out/\$part.exit
done
nvidia-smi --query-compute-apps=pid,used_memory --format=csv > out/machine_end.txt 2>&1
echo done > out/stage.txt
date -u +%Y-%m-%dT%H:%M:%SZ > out/done.txt
EOF
# The redirection is on the whole background job, so that it holds none of ssh's streams and ssh returns at once.
ssh "$HOST" "bash -lc '(cd $REMOTE && exec nohup bash run.sh) > $REMOTE/run.log 2>&1 < /dev/null & echo \$! > $REMOTE/run.pid'" || exit 3
sleep 2
if ! ssh "$HOST" "kill -0 \$(cat $REMOTE/run.pid) 2>/dev/null"; then
    echo "error: the run stopped at once:" >&2
    ssh "$HOST" "cat $REMOTE/run.log" >&2
    exit 1
fi
echo "Started $PARTS on $HOST at $COMMIT, pid $(ssh "$HOST" "cat $REMOTE/run.pid"). See the state with: sh $0 status"
