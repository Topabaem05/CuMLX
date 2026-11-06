#!/bin/bash
# A second F1 golden with another prompt, for a check that validation/m6-f1-limits.json does not judge, under spikes/m6_cuda_v100.sh: bitsandbytes 0.50.2 itself, for validation/m6-f1-limits.json. A failed step does not stop the next one.
# FlashAttention-2 needs compute capability 8.0 and the V100 has 7.0, so attention is cumlx.compat.flash_attn_sdpa here too.
#   bash m6_f1.sh REPO OUT WORK
REPO=$1 OUT=$2 WORK=$3
MODEL=Qwen/Qwen2.5-0.5B
REVISION=060db6499f32faf8b98477b0a26969ef7d8b9987
PROMPT="Water boils at a lower temperature on a mountain because"
mkdir -p "$OUT" "$WORK"
export PYTHONPATH=$WORK/pkgs:$REPO/src HF_HOME=$WORK/hf PYTHONUNBUFFERED=1
stage() { echo "[$(date +%H:%M:%S)] f1alt: $1"; echo "f1alt: $1" > "$OUT/stage.txt"; }
stage "install transformers 5.22.1, accelerate 1.15.0 and bitsandbytes 0.50.2"
rm -rf "$WORK/pkgs"
printf 'transformers==5.22.1\naccelerate==1.15.0\nbitsandbytes==0.50.2\n' > "$WORK/req.in"
# torch and what it needs are the machine's own, so the resolved list leaves them out.
# uv's --python python3 can find a Python that uv manages, of another version, so the path is the one that runs the parts below.
PY=$(command -v python3)
{ echo "python: $PY $("$PY" --version 2>&1)" &&
  uv pip compile -q --python "$PY" "$WORK/req.in" -o "$WORK/req.txt" &&
  grep -viE '^(torch|nvidia-|triton|numpy|sympy|mpmath|networkx|jinja2|markupsafe|setuptools|typing-extensions|cuda-)' "$WORK/req.txt" > "$WORK/req-no-torch.txt" &&
  uv pip install -q --python "$PY" --target "$WORK/pkgs" --no-deps -r "$WORK/req-no-torch.txt"; } > "$OUT/install.txt" 2>&1
echo $? > "$OUT/install.exit"
cp "$WORK/req.txt" "$OUT/req.txt" 2>/dev/null
python3 -c 'import torch, transformers, accelerate, bitsandbytes, json; print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__, "accelerate": accelerate.__version__, "bitsandbytes": bitsandbytes.__version__, "gpu": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0)}))' > "$OUT/versions.json" 2>> "$OUT/install.txt"
stage "capture the golden"
python3 -m cumlx run --target cuda "$REPO/spikes/m6_f1_record.py" --substitute flash_attn=cumlx.compat.flash_attn_sdpa --report "$OUT/capture.run-report.json" --format json -- capture --output "$OUT/golden" --prompt "$PROMPT" > "$OUT/capture.json" 2> "$OUT/capture.txt"
echo $? > "$OUT/capture.exit"
cp "$OUT/../cumlx_commit.txt" "$OUT/cumlx_commit.txt" 2>/dev/null
stage done
