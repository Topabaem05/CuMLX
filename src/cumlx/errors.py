"""Structured errors and stable exit codes shared by the CLI and the Python API.

Every failure that a caller (human or AI agent) can act on carries a stable
machine-readable ``code``, a one-sentence ``message`` and, when possible, a
``hint`` that names the exact next action.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Any


class ExitCode(IntEnum):
    OK = 0           # command succeeded; for verification: every compared module passed
    DIVERGED = 1     # verification ran and at least one module exceeded its threshold
    USAGE = 2        # invalid arguments or an unsafe request (for example, output exists)
    ENVIRONMENT = 3  # optional dependency or backend is missing
    INPUT = 4        # input file or model is missing, corrupt or has the wrong schema
    INCOMPLETE = 5   # ran without divergence, but some modules were not verified (not a pass)
    INTERNAL = 10    # unexpected exception: a CUMLX bug


EXIT_CODE_DOCS = {
    ExitCode.OK: 'Success. For diff/conform: every compared item passed.',
    ExitCode.DIVERGED: 'Verification ran and found at least one divergence. Read data.first_divergence.',
    ExitCode.USAGE: 'Invalid arguments or unsafe request. Fix the command; see errors[].hint.',
    ExitCode.ENVIRONMENT: 'A dependency or backend is missing. Run the install command in errors[].hint.',
    ExitCode.INPUT: 'An input path, model or trace is missing, corrupt or has the wrong schema.',
    ExitCode.INCOMPLETE: 'No divergence, but some items were not verified. This is NOT a pass.',
    ExitCode.INTERNAL: 'Unexpected internal error. Report it with the full JSON output.',
}

# code -> (exit code, short description)
ERROR_CODES: dict[str, tuple[ExitCode, str]] = {
    'E_USAGE': (ExitCode.USAGE, 'Invalid command-line arguments.'),
    'E_OUTPUT_EXISTS': (ExitCode.USAGE, 'Output path already exists; pass --overwrite to replace it.'),
    'E_DEP_MISSING': (ExitCode.ENVIRONMENT, 'An optional Python dependency is not installed.'),
    'E_BACKEND_UNAVAILABLE': (ExitCode.ENVIRONMENT, 'The requested compute backend is not available.'),
    'E_INPUT_NOT_FOUND': (ExitCode.INPUT, 'An input path does not exist.'),
    'E_INPUT_INVALID': (ExitCode.INPUT, 'An input file is corrupt or does not match its schema.'),
    'E_MODEL_LOAD': (ExitCode.INPUT, 'The model could not be loaded.'),
    'E_MODEL_RUN': (ExitCode.INPUT, 'The model failed while running on the recorded inputs.'),
    'E_RUN_SCRIPT_FAILED': (ExitCode.INPUT, 'The script given to `cumlx run` failed or ran past its timeout.'),
    'E_RUN_UNSUPPORTED': (ExitCode.ENVIRONMENT, 'The script called a CUDA API with no equivalent on the target.'),
    'E_RUN_MPS_OP_MISSING': (ExitCode.ENVIRONMENT, 'The script used an operation that MPS does not implement.'),
    'E_RUN_MPS_DTYPE': (ExitCode.ENVIRONMENT, 'The script used a data type that MPS does not have.'),
    'E_RUN_MISSING_PACKAGE': (ExitCode.ENVIRONMENT, 'The script imports a package that is not installed.'),
    'E_SIZE_LIMIT': (ExitCode.USAGE, 'The operation would exceed a configured size limit.'),
    'E_INTERNAL': (ExitCode.INTERNAL, 'Unexpected internal error.'),
}


class CumlxError(Exception):
    """An actionable error with a stable code. Never used for verification verdicts."""

    def __init__(self, code: str, message: str, *, hint: str | None = None,
                 details: dict[str, Any] | None = None):
        if code not in ERROR_CODES:
            raise ValueError(f'unknown error code {code!r}')
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.details = details or {}

    @property
    def exit_code(self) -> ExitCode:
        return ERROR_CODES[self.code][0]

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {'code': self.code, 'message': self.message}
        if self.hint:
            out['hint'] = self.hint
        if self.details:
            out['details'] = self.details
        return out


def missing_dependency(module: str, extra: str, *, purpose: str) -> CumlxError:
    return CumlxError(
        'E_DEP_MISSING', f'{purpose} requires the Python module {module!r}, which is not installed.',
        hint=f"python -m pip install 'cumlx-lab[{extra}]'",
        details={'module': module, 'extra': extra})
