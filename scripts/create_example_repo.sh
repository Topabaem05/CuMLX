#!/bin/bash
# Create the private example repository of roadmap T4: a clean pull request passes the CUMLX action, and a pull request
# with a defect fails. Run it yourself: it creates a repository under your GitHub account. It needs `gh`, and `cumlx`
# with the capture extra. It vendors the action and a wheel, so CuMLX itself can stay private.
#
#   scripts/create_example_repo.sh [owner/name]
#   DRY_RUN=1 scripts/create_example_repo.sh     # build and check everything, and create nothing on GitHub
set -euo pipefail
REPO="${1:-Topabaem05/cumlx-action-example}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
VERSION="$(cd "$ROOT" && python -c "import cumlx; print(cumlx.__version__)")"
WHEEL="cumlx_lab-$VERSION-py3-none-any.whl"

(cd "$ROOT" && python -m build --wheel --outdir "$WORK/dist" >/dev/null)
E="$WORK/example"
mkdir -p "$E/vendor" "$E/.github/actions/cumlx/scripts" "$E/.github/workflows" "$E/port"
cp "$WORK/dist/$WHEEL" "$E/vendor/"
cp "$ROOT/action.yml" "$E/.github/actions/cumlx/action.yml"
cp "$ROOT/scripts/action_verify.py" "$E/.github/actions/cumlx/scripts/"
cp "$ROOT/examples/port-ci/port/__init__.py" "$ROOT/examples/port-ci/port/loader.py" "$E/port/"
sed -e "s#uses: Topabaem05/CuMLX@main#uses: ./.github/actions/cumlx#" \
    -e "s#          token-ids: 1,5,9,23,7,99,3,64,12,8#          token-ids: 1,5,9,23,7,99,3,64,12,8\n          cumlx-package: ./vendor/$WHEEL#" \
    "$ROOT/examples/port-ci/.github/workflows/verify-port.yml" > "$E/.github/workflows/verify-port.yml"
cumlx fixture tiny-llama -o "$E/original" >/dev/null 2>&1
rm -f "$E/original/.cumlx-fixture"
printf '__pycache__/\ncumlx-output/\n' > "$E/.gitignore"
cat > "$E/README.md" <<'EOF'
# CUMLX action example

This repository keeps an MLX port, and checks every pull request with the CUMLX action. Only the verdict `ok` passes.
`.github/actions/cumlx/` and `vendor/` hold a copy of the action and a wheel.
EOF
cd "$E"
git init -q -b main
git add -A
git commit -q -m "Add an MLX port checked by the CUMLX action"
git switch -q -c clean-change
printf '\nThe original model in `original/` is a tiny random model.\n' >> README.md
git commit -q -am "Document the original model"
git switch -q main
git switch -q -c injected-fault
cp "$ROOT/examples/port-ci/port/loader_faulty.py" port/loader.py
git commit -q -am "Change the RoPE base in the port"
git switch -q main
test -n "$(git diff main clean-change -- README.md)"
test -n "$(git diff main injected-fault -- port/loader.py)"
grep -q "cumlx-package: ./vendor/$WHEEL" .github/workflows/verify-port.yml
grep -q "uses: ./.github/actions/cumlx" .github/workflows/verify-port.yml
echo "built: wheel $WHEEL, branches: $(git branch --format='%(refname:short)' | tr '\n' ' ')"
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "dry run: nothing was created on GitHub"
  exit 0
fi
gh repo create "$REPO" --private --source . --remote origin --description "Example: verify an MLX port in CI with the CUMLX action"
git push -q origin main clean-change injected-fault
gh pr create --repo "$REPO" --head clean-change --title "Clean change" --body "This pull request must pass."
gh pr create --repo "$REPO" --head injected-fault --title "Injected fault" --body "This pull request must fail at model.layers.1.self_attn."
echo "created https://github.com/$REPO; check both pull requests"
