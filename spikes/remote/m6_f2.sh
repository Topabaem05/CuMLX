#!/bin/bash
# F2 on CUDA, under spikes/m6_cuda_v100.sh: gsplat 1.5.3 itself, for validation/m6-f2-cuda-limits.json. A failed step does not stop the next one.
#   bash m6_f2.sh REPO OUT WORK
REPO=$1 OUT=$2 WORK=$3
mkdir -p "$OUT" "$WORK"
export PYTHONPATH=$WORK/pkgs PATH=$WORK/pkgs/bin:$PATH TORCH_EXTENSIONS_DIR=$WORK/torch_extensions TORCH_CUDA_ARCH_LIST=7.0 PYTHONUNBUFFERED=1
stage() { echo "[$(date +%H:%M:%S)] f2: $1"; echo "f2: $1" > "$OUT/stage.txt"; }
stage "install gsplat 1.5.3"
rm -rf "$WORK/pkgs" "$WORK/torch_extensions"
uv pip install -q --python python3 --target "$WORK/pkgs" --no-deps gsplat==1.5.3 ninja jaxtyping wadler-lindig > "$OUT/install.txt" 2>&1
echo $? > "$OUT/install.exit"
python3 -c 'import torch, numpy, gsplat, json; print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "numpy": numpy.__version__, "gsplat": gsplat.__version__, "gpu": torch.cuda.get_device_name(0)}))' > "$OUT/versions.json" 2>> "$OUT/install.txt"
stage "build gsplat's CUDA code"
python3 -c 'import torch, gsplat; gsplat.rasterization(torch.tensor([[0.0, 0.0, 3.0]], device="cuda"), torch.tensor([[1.0, 0, 0, 0]], device="cuda"), torch.full((1, 3), 0.2, device="cuda"), torch.full((1,), 0.5, device="cuda"), torch.ones(1, 3, device="cuda"), torch.eye(4, device="cuda")[None], torch.tensor([[[64.0, 0, 32], [0, 64.0, 32], [0, 0, 1]]], device="cuda"), 64, 64); torch.cuda.synchronize(); print("built")' > "$OUT/build.txt" 2>&1
echo $? > "$OUT/build.exit"
stage "renders and gradients"
python3 "$REPO/spikes/m6_f2_cuda_record.py" --output "$OUT/record.npz" > "$OUT/record.txt" 2>&1
echo $? > "$OUT/record.exit"
stage "the original program"
python3 "$REPO/spikes/m6_f2_gsplat_program.py" --output "$OUT/program.json" > "$OUT/program.txt" 2>&1
echo $? > "$OUT/program.exit"
for seed in 0 1 2 3 4; do
    for mode in right frozen wrong_cameras; do
        stage "training $mode seed $seed"
        python3 "$REPO/spikes/m6_f2_gsplat_train.py" --mode $mode --seed $seed --output "$OUT/train_${mode}_$seed.json" > "$OUT/train_${mode}_$seed.txt" 2>&1
        echo $? > "$OUT/train_${mode}_$seed.exit"
    done
done
cp "$OUT/../cumlx_commit.txt" "$OUT/cumlx_commit.txt" 2>/dev/null
stage done
