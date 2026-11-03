#!/bin/bash
# Record op-level goldens of the repository's own kernels on a shared NVIDIA machine, and bring them back.
#   scripts/kernel_capture_remote.sh <ssh-host> <local-output-dir> [<kernels-dir> <cases.json> [cumlx kernel capture flags...]]
# Default kernels: src/cumlx/kernels with validation/kernel-cases/cases.json.
# It stops if another compute process is on the GPU, works in a new folder there, and deletes that folder on exit.
# The remote machine needs nvcc, nvidia-smi, python3 and NumPy. This script installs nothing.
set -euo pipefail

HOST=$1
OUT=$2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
KERNELS=${3:-$ROOT/src/cumlx/kernels}
CASES=${4:-$ROOT/validation/kernel-cases/cases.json}
shift $(( $# < 4 ? $# : 4 ))
[ -e "$OUT" ] && { echo "$OUT exists" >&2; exit 2; }
[ -d "$KERNELS" ] && [ -f "$CASES" ] || { echo "need a kernels folder and a cases file" >&2; exit 2; }

BUSY=$(ssh -o BatchMode=yes "$HOST" 'nvidia-smi --query-compute-apps=pid --format=csv,noheader')
[ -n "$BUSY" ] && { echo "the GPU on $HOST has compute processes: $BUSY" >&2; exit 3; }

WORK=$(ssh -o BatchMode=yes "$HOST" 'mktemp -d "$HOME/cumlx-kernel-capture-XXXXXX"')
cleanup() { ssh -o BatchMode=yes "$HOST" "rm -rf -- '$WORK'" || echo "could not delete $WORK on $HOST" >&2; }
trap cleanup EXIT

export COPYFILE_DISABLE=1
ssh -o BatchMode=yes "$HOST" "mkdir -p '$WORK/kernels' '$WORK/cases'"
tar -C "$ROOT" --no-xattrs -czf - --exclude='__pycache__' src/cumlx | ssh -o BatchMode=yes "$HOST" "tar -C '$WORK' -xzf -"
tar -C "$KERNELS" --no-xattrs -czf - . | ssh -o BatchMode=yes "$HOST" "tar -C '$WORK/kernels' -xzf -"
tar -C "$(dirname "$CASES")" --no-xattrs -czf - . | ssh -o BatchMode=yes "$HOST" "tar -C '$WORK/cases' -xzf -"
STATUS=0
ssh -o BatchMode=yes "$HOST" "cd '$WORK' && PYTHONPATH=src python3 -m cumlx kernel capture kernels \
    --cases 'cases/$(basename "$CASES")' -o golden --require-idle-gpu --format json $*" > "$OUT.result.json" || STATUS=$?
if ssh -o BatchMode=yes "$HOST" "test -f '$WORK/golden/manifest.json'"; then
    mkdir -p "$OUT"
    ssh -o BatchMode=yes "$HOST" "tar -C '$WORK/golden' -czf - ." | tar -C "$OUT" -xzf -
fi
echo "capture exit code: $STATUS; result in $OUT.result.json" >&2
exit "$STATUS"
