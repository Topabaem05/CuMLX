#!/bin/sh
# The Mac side of validation/m6-f1-limits.json, inside spikes/m6_same_mac.sh, which gives it an empty home folder, a public checkout and the limits.
# CUMLX_F1_GOLDEN names the CUDA golden folder from spikes/m6_cuda_v100.sh (its f1/golden). Without it the replay parts do not run, and that is not a pass.
# A failed step does not stop the next one. spikes/m6_f1_judge.py gives the verdict.
#
#   sh spikes/m6_f1_work.sh OUT
set -u
OUT=${1:?the result folder}
MODEL=Qwen/Qwen2.5-0.5B
REVISION=060db6499f32faf8b98477b0a26969ef7d8b9987
SUBS="--substitute flash_attn=cumlx.compat.flash_attn_sdpa --substitute bitsandbytes=cumlx.compat.bitsandbytes_int8"
WORK=$HOME/f1
mkdir -p "$OUT" "$WORK"
say() { echo "[$(date +%H:%M:%S)] f1: $1"; }

say "installing torch 2.15.0, transformers 5.22.1 and accelerate 1.15.0"
uv venv -q --python 3.12 "$WORK/venv" >"$OUT/install.txt" 2>&1
uv pip install -q --python "$WORK/venv/bin/python" -e . torch==2.15.0 transformers==5.22.1 accelerate==1.15.0 >>"$OUT/install.txt" 2>&1
echo "$?" >"$OUT/install.exit"
[ "$(cat "$OUT/install.exit")" = 0 ] || exit 0
uv pip freeze --python "$WORK/venv/bin/python" >"$OUT/freeze.txt" 2>&1
# Neither original package may be here: then the program could not reach them, and the adapters are the only answer.
"$WORK/venv/bin/python" -c 'import importlib.util, importlib.metadata as m, sys
for name in ("bitsandbytes", "flash_attn", "flash-attn"):
    try:
        m.version(name); sys.exit(f"{name} is installed")
    except m.PackageNotFoundError:
        pass
sys.exit(1 if importlib.util.find_spec("bitsandbytes") or importlib.util.find_spec("flash_attn") else 0)' >"$OUT/clean.txt" 2>&1
echo "$?" >"$OUT/clean.exit"
CUMLX=$WORK/venv/bin/cumlx

say "the original program, not changed"
"$CUMLX" run spikes/m6_f1_program.py $SUBS --report "$OUT/program.run-report.json" --format json -- --model "$MODEL" --revision "$REVISION" --output "$OUT/program.json" >"$OUT/program.result.json" 2>"$OUT/program.txt"
echo "$?" >"$OUT/program.exit"
say "the Mac's own golden, for the chain that is reported"
"$CUMLX" run spikes/m6_f1_record.py $SUBS --report "$OUT/capture.run-report.json" --format json -- capture --output "$OUT/mac-golden" >"$OUT/capture.json" 2>"$OUT/capture.txt"
echo "$?" >"$OUT/capture.exit"
GOLDEN=${CUMLX_F1_GOLDEN:-}
# spikes/m6_finish.sh starts this run before the V100 has finished, so the golden may arrive later. CUMLX_F1_WAIT_SECONDS says how long to wait for it.
waited=0
while [ -n "$GOLDEN" ] && [ ! -f "$GOLDEN/summary.json" ] && [ "$waited" -lt "${CUMLX_F1_WAIT_SECONDS:-0}" ]; do
    sleep 60
    waited=$((waited + 60))
done
if [ -n "$GOLDEN" ] && [ -f "$GOLDEN/summary.json" ]; then
    say "replay of the CUDA golden"
    "$CUMLX" run spikes/m6_f1_record.py $SUBS --report "$OUT/replay.run-report.json" --format json -- replay --golden "$GOLDEN" --output "$OUT/replay.json" >"$OUT/replay.result.json" 2>"$OUT/replay.txt"
    echo "$?" >"$OUT/replay.exit"
    for fault in no_outlier_split activation_scale_128 bias_dropped causal_top_left no_softmax_scale; do
        say "control: $fault"
        "$CUMLX" run spikes/m6_f1_record.py $SUBS --report "$OUT/fault_$fault.run-report.json" --format json -- replay --golden "$GOLDEN" --output "$OUT/fault_$fault.json" --fault "$fault" >"$OUT/fault_$fault.result.json" 2>"$OUT/fault_$fault.txt"
        echo "$?" >"$OUT/fault_$fault.exit"
    done
else
    echo "no CUDA golden: CUMLX_F1_GOLDEN=$GOLDEN" >"$OUT/replay.txt"
fi
date -u +%Y-%m-%dT%H:%M:%SZ >"$OUT/f1_mac_done.txt"
