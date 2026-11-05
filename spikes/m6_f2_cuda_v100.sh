#!/bin/sh
# F2 on CUDA: gsplat 1.5.3 itself on a shared GPU machine, for validation/m6-f2-cuda-limits.json. One command, and it returns at once.
#
#   sh spikes/m6_f2_cuda_v100.sh           # copies the committed repo and starts the run
#   sh spikes/m6_f2_cuda_v100.sh status    # shows the state of the run
#   sh spikes/m6_f2_cuda_v100.sh fetch     # copies the results to ~/cumlx-m6-f2-cuda on this Mac
#
# It refuses to start if another process is on the GPU. The host is CUMLX_GPU_HOST, or the content of ~/.config/cumlx/gpu-host. There is no default.
# Everything goes in ~/cumlx-m6-f2 there: gsplat and its few dependencies in a --target folder, and gsplat's JIT build in TORCH_EXTENSIONS_DIR inside it.
# torch and NumPy are the machine's own (python3, user site). Nothing outside that folder changes.
# Delete the remote folder when the test is finished: ssh <host> rm -rf cumlx-m6-f2
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
REMOTE=cumlx-m6-f2
REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
LOCAL=${CUMLX_M6_F2_CUDA:-$HOME/cumlx-m6-f2-cuda}

case "${1:-start}" in
status)
    ssh "$HOST" "if [ -f $REMOTE/run.pid ] && kill -0 \$(cat $REMOTE/run.pid) 2>/dev/null; then echo running; else echo not running; fi; cat $REMOTE/out/stage.txt 2>/dev/null; tail -n 5 $REMOTE/run.log 2>/dev/null"
    exit $?
    ;;
fetch)
    if ! ssh "$HOST" "test -f $REMOTE/out/done.txt"; then
        echo "error: $HOST has no finished results yet" >&2
        exit 1
    fi
    mkdir -p "$LOCAL"
    ssh "$HOST" "cd $REMOTE/out && tar -c ." | tar -x -C "$LOCAL" || exit 1
    echo "Copied to $LOCAL"
    exit 0
    ;;
start) ;;
*)
    echo "usage: sh spikes/m6_f2_cuda_v100.sh [status|fetch]" >&2
    exit 2
    ;;
esac

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
COMMIT=$(git -C "$REPO" rev-parse HEAD)
ssh "$HOST" "rm -rf $REMOTE/CuMLX $REMOTE/out && mkdir -p $REMOTE/CuMLX $REMOTE/out" || exit 3
git -C "$REPO" archive HEAD spikes src/cumlx validation/m6-f2-cuda-limits.json | ssh "$HOST" "tar -x -C $REMOTE/CuMLX" || exit 3
echo "$COMMIT" | ssh "$HOST" "cat > $REMOTE/out/cumlx_commit.txt" || exit 3
ssh "$HOST" "cat > $REMOTE/work.sh" <<'EOF'
#!/bin/bash
# Runs on the GPU machine, in ~/cumlx-m6-f2. A failed part does not stop the next one.
cd "$(dirname "$0")"
OUT=$PWD/out
export PYTHONPATH=$PWD/pkgs PATH=$PWD/pkgs/bin:$PATH TORCH_EXTENSIONS_DIR=$PWD/torch_extensions TORCH_CUDA_ARCH_LIST=7.0 PYTHONUNBUFFERED=1
stage() { echo "[$(date +%H:%M:%S)] $1"; echo "$1" > "$OUT/stage.txt"; }
{ nvidia-smi; nvidia-smi --query-compute-apps=pid,used_memory --format=csv; nvcc --version; python3 --version; } > "$OUT/machine_start.txt" 2>&1
stage "install gsplat 1.5.3 into pkgs"
rm -rf pkgs torch_extensions
uv pip install -q --python python3 --target pkgs --no-deps gsplat==1.5.3 ninja jaxtyping wadler-lindig > "$OUT/install.txt" 2>&1
echo $? > "$OUT/install.exit"
python3 -c 'import torch, numpy, gsplat, json; print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "numpy": numpy.__version__, "gsplat": gsplat.__version__, "gpu": torch.cuda.get_device_name(0)}))' > "$OUT/versions.json" 2>> "$OUT/install.txt"
stage "build gsplat's CUDA code"
python3 -c 'import torch, gsplat; gsplat.rasterization(torch.zeros(1, 3, device="cuda") + torch.tensor([0.0, 0.0, 3.0], device="cuda"), torch.tensor([[1.0, 0, 0, 0]], device="cuda"), torch.full((1, 3), 0.2, device="cuda"), torch.full((1,), 0.5, device="cuda"), torch.ones(1, 3, device="cuda"), torch.eye(4, device="cuda")[None], torch.tensor([[[64.0, 0, 32], [0, 64.0, 32], [0, 0, 1]]], device="cuda"), 64, 64); torch.cuda.synchronize(); print("built")' > "$OUT/build.txt" 2>&1
echo $? > "$OUT/build.exit"
stage "renders and gradients"
python3 CuMLX/spikes/m6_f2_cuda_record.py --output "$OUT/record.npz" > "$OUT/record.txt" 2>&1
echo $? > "$OUT/record.exit"
stage "the original program"
python3 CuMLX/spikes/m6_f2_gsplat_program.py --output "$OUT/program.json" > "$OUT/program.txt" 2>&1
echo $? > "$OUT/program.exit"
for seed in 0 1 2 3 4; do
    for mode in right frozen wrong_cameras; do
        stage "training $mode seed $seed"
        python3 CuMLX/spikes/m6_f2_gsplat_train.py --mode $mode --seed $seed --output "$OUT/train_${mode}_$seed.json" > "$OUT/train_${mode}_$seed.txt" 2>&1
        echo $? > "$OUT/train_${mode}_$seed.exit"
    done
done
{ nvidia-smi --query-compute-apps=pid,used_memory --format=csv; } > "$OUT/machine_end.txt" 2>&1
uv pip freeze --python python3 > "$OUT/freeze.txt" 2>&1
stage "done"
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/done.txt"
EOF
ssh "$HOST" "bash -lc 'cd $REMOTE && nohup bash work.sh > run.log 2>&1 < /dev/null & echo \$! > $REMOTE/run.pid'" || exit 3
sleep 2
if ! ssh "$HOST" "kill -0 \$(cat $REMOTE/run.pid) 2>/dev/null"; then
    echo "error: the run stopped at once:" >&2
    ssh "$HOST" "cat $REMOTE/run.log" >&2
    exit 1
fi
echo "Started on $HOST at $COMMIT, pid $(ssh "$HOST" "cat $REMOTE/run.pid"). See the state with: sh $0 status"
