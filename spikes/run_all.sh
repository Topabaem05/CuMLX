#!/bin/bash
# Install the M0 candidates into a new venv and run each probe with a time limit. A failed probe is a result, not an error.
set -u
WORK="${1:?usage: run_all.sh WORK_DIR}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$WORK"
uv venv -q --python 3.12 "$WORK/venv"
uv pip install -q --python "$WORK/venv/bin/python" genesis-world torch mitsuba usd-core quadrants mujoco numpy
: > "$WORK/results.jsonl"
for probe in mitsuba usd quadrants genesis; do
  echo "== $probe"
  ( cd "$WORK" && perl -e 'alarm 600; exec @ARGV' "$WORK/venv/bin/python" -I "$HERE/probes/probe_$probe.py" ) 2>&1 | tee -a "$WORK/$probe.log" | grep '^{' >> "$WORK/results.jsonl"
done
cat "$WORK/results.jsonl"
