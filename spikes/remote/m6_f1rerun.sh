#!/bin/bash
# F1 on CUDA again, for validation/m6-f1-rerun-limits.json: m6_f1.sh with the golden prompt that the rules name, under spikes/m6_cuda_v100.sh.
#   bash m6_f1rerun.sh REPO OUT WORK
CUMLX_F1_PROMPT=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["prompt"])' "$1/validation/m6-f1-rerun-limits.json") || exit 1
export CUMLX_F1_PROMPT
exec bash "$(dirname "$0")/m6_f1.sh" "$@"
