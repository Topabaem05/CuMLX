#!/usr/bin/env python3
"""Point the example workflow at the local copy of the action and at the wheel. It works on every system, as `sed -i` does not.

    python spikes/t4_edit_workflow.py WORKFLOW.yml WHEEL_NAME

It exits 4 if the workflow does not have the lines that it expects, so that a changed example cannot pass with no edit.
"""
from __future__ import annotations

from pathlib import Path
import sys

USES_REMOTE = '      - uses: Topabaem05/CuMLX@main\n'
USES_LOCAL = '      - uses: ./.cumlx-action\n'
STEP_ID = '        id: cumlx\n        with:\n'


def edit(text: str, wheel: str) -> str:
    if text.count(USES_REMOTE) != 1 or text.count(STEP_ID) != 1:
        raise ValueError('the workflow must hold the remote `uses` line and the `id: cumlx` step exactly once')
    package = '          cumlx-package: ${{ github.workspace }}/.cumlx-action/' + wheel + '\n'
    return text.replace(USES_REMOTE, USES_LOCAL).replace(STEP_ID, STEP_ID + package)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    path = Path(argv[0])
    try:
        path.write_text(edit(path.read_text(), argv[1]))
    except (OSError, ValueError) as exc:
        print(f'error: {path}: {exc}', file=sys.stderr)
        return 4
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
