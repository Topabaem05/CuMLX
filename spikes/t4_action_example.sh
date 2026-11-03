#!/bin/sh
# Prove the CUMLX GitHub Action from a separate private repository: a clean pull request passes, a faulty one fails.
#
#   sh spikes/t4_action_example.sh prepare   # builds the local repository and its branches, with no network
#   sh spikes/t4_action_example.sh           # prepare, then create the private repository and open both pull requests
#   sh spikes/t4_action_example.sh status    # shows the checks of both pull requests
#
# CuMLX is private and its Actions access level is `none`, so the repository holds a copy of the action and the wheel.
# It therefore does not test `uses: Topabaem05/CuMLX@<sha>`. Delete it with: gh repo delete Topabaem05/CuMLX-port-ci-example
set -eu

REPO=${CUMLX_ACTION_EXAMPLE_REPO:-Topabaem05/CuMLX-port-ci-example}
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(dirname -- "$HERE")
OUT=${CUMLX_ACTION_EXAMPLE:-$HOME/code/cumlx-port-ci-example}
PY=$ROOT/.venv/bin/python

commit() {
    git -C "$OUT" add -A
    git -C "$OUT" -c user.name=Topabaem -c user.email=87803226+Topabaem05@users.noreply.github.com commit -q \
        -m "$1" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
}

prepare() {
    if [ -e "$OUT" ]; then
        echo "error: $OUT exists. Remove it first." >&2
        exit 4
    fi
    if [ ! -x "$PY" ]; then
        echo "error: $PY not found. Make it with: python -m venv .venv && .venv/bin/pip install -e '.[dev,capture,diff]' build" >&2
        exit 3
    fi
    work=$(mktemp -d)
    "$PY" -m build --wheel -o "$work" "$ROOT" >/dev/null
    wheel=$(basename "$work"/*.whl)
    mkdir -p "$OUT"
    git -C "$OUT" init -q -b main
    cp -R "$ROOT/examples/port-ci/.github" "$OUT/.github"
    mkdir -p "$OUT/port" "$OUT/.cumlx-action/scripts"
    cp "$ROOT/examples/port-ci/port/__init__.py" "$ROOT/examples/port-ci/port/loader.py" "$OUT/port/"
    cp "$ROOT/action.yml" "$OUT/.cumlx-action/"
    cp "$ROOT/scripts/action_verify.py" "$OUT/.cumlx-action/scripts/"
    cp "$work/$wheel" "$OUT/.cumlx-action/"
    rm -rf "$work"
    "$ROOT/.venv/bin/cumlx" fixture tiny-llama -o "$OUT/original" --format json >/dev/null 2>&1
    "$PY" "$HERE/t4_edit_workflow.py" "$OUT/.github/workflows/verify-port.yml" "$wheel"
    printf '__pycache__/\ncumlx-output/\n' > "$OUT/.gitignore"
    cat > "$OUT/README.md" <<R
# CUMLX port CI example

This private repository proves the CUMLX GitHub Action in a separate repository. The pull request with a defect must fail, and the clean pull request must pass.

\`.cumlx-action/\` holds a copy of the action and the wheel from CuMLX $(git -C "$ROOT" rev-parse --short HEAD), because CuMLX is private and its Actions access level is \`none\`. So this repository does not test \`uses: Topabaem05/CuMLX@<sha>\`.
R
    commit "feat: add the port and the verification workflow"

    git -C "$OUT" checkout -q -b clean-change
    cat > "$OUT/port/loader.py" <<'P'
"""The port under review. CUMLX calls this function and compares the module it returns."""
from cumlx.verify.mlx_port import load_port


def load(path: str):
    model = load_port(path)  # A refactor that changes no computation.
    return model
P
    commit "refactor: name the loaded model"

    git -C "$OUT" checkout -q main
    git -C "$OUT" checkout -q -b injected-fault
    cat > "$OUT/port/loader.py" <<'P'
"""The port under review. This branch has a defect that raises no error: the RoPE base is wrong in layer 1."""
import mlx.nn as nn

from cumlx.verify.mlx_port import load_port


def load(path: str):
    model = load_port(path)
    attention = model.model.layers[1].self_attn
    rope = attention.rope
    attention.rope = nn.RoPE(rope.dims, traditional=rope.traditional, base=rope.base * 10, scale=rope.scale)
    return model
P
    commit "fix: change the RoPE base (injected fault, do not merge)"
    git -C "$OUT" checkout -q main
    echo "prepared $OUT"
}

case ${1:-start} in
prepare)
    prepare
    ;;
status)
    gh pr list --repo "$REPO" --state all --json number,title,headRefName --jq '.[]|[.number,.headRefName,.title]|@tsv'
    for branch in clean-change injected-fault; do
        echo "--- $branch"
        gh pr checks "$branch" --repo "$REPO" || true
    done
    ;;
start)
    [ -e "$OUT/.git" ] || prepare
    echo "creating the private repository $REPO"
    gh repo create "$REPO" --private --description "Proves the CUMLX GitHub Action from a separate repository." --source "$OUT" --remote origin
    git -C "$OUT" push -q -u origin main clean-change injected-fault
    gh pr create --repo "$REPO" --base main --head clean-change --title "Clean refactor: must pass" \
        --body "A refactor that changes no computation. CUMLX must report ok."
    gh pr create --repo "$REPO" --base main --head injected-fault --title "Injected fault: must fail" \
        --body "An injected defect in layer 1 that raises no error. CUMLX must report diverged. Do not merge."
    echo "Now the runs take about 10 minutes. Then: sh spikes/t4_action_example.sh status"
    ;;
*)
    echo "usage: t4_action_example.sh [prepare|status]" >&2
    exit 2
    ;;
esac
