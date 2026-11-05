#!/bin/bash
# F3 on CUDA, under spikes/m6_cuda_v100.sh: SAM2's connected components, built from the original file. For validation/m6-f3-limits.json.
#   bash m6_f3.sh REPO OUT WORK
REPO=$1 OUT=$2 WORK=$3
mkdir -p "$OUT" "$WORK"
export TORCH_EXTENSIONS_DIR=$WORK/torch_extensions TORCH_CUDA_ARCH_LIST=7.0 PYTHONUNBUFFERED=1 PYTHONPATH=$REPO/src
stage() { echo "[$(date +%H:%M:%S)] f3: $1"; echo "f3: $1" > "$OUT/stage.txt"; }
stage "fetch sam2's file at its pinned commit"
python3 "$REPO/spikes/m6_f3_app.py" fetch --out "$WORK/src" > "$OUT/fetch.json" 2> "$OUT/fetch.txt"
echo $? > "$OUT/fetch.exit"
stage "build and run the original"
python3 "$REPO/spikes/m6_f3_app.py" cuda --source "$WORK/src/connected_components.cu" --build "$WORK/build" --output "$OUT/cuda.npz" > "$OUT/cuda.txt" 2>&1
echo $? > "$OUT/cuda.exit"
cp "$OUT/../cumlx_commit.txt" "$OUT/cumlx_commit.txt" 2>/dev/null
stage done
