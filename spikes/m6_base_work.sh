#!/bin/sh
# The base of the m6 part of spikes/m6_same_mac.sh: doctor, the self test and the tests without a GPU, with the checkout's .venv.
#   sh spikes/m6_base_work.sh OUT
set -u
OUT=${1:?the result folder}
mkdir -p "$OUT"
.venv/bin/cumlx doctor --require capture,diff --format json >"$OUT/doctor.json" 2>"$OUT/doctor.err"
echo "$?" >"$OUT/doctor.exit"
.venv/bin/cumlx selftest --format json >"$OUT/selftest.json" 2>"$OUT/selftest.err"
echo "$?" >"$OUT/selftest.exit"
.venv/bin/python -m pytest -q -m 'not gpu' >"$OUT/tests.txt" 2>&1
echo "$?" >"$OUT/tests.exit"
