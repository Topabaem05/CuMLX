"""``cumlx`` command line. Designed to be driven by AI agents as well as humans.

Agent contract (also returned by ``cumlx describe``):

1. Non-interactive. No prompts, no pagers, no colour codes.
2. ``--format json`` (the default when stdout is not a TTY) prints exactly one
   JSON document (schema ``cumlx.result.v1``) on stdout. Logs go to stderr.
3. Stable exit codes (see ``cumlx describe``). A skip or an unverified module is
   never reported as a pass.
4. Every error has a stable ``code`` and, when possible, a ``hint`` with the
   exact command that fixes it. ``next_steps`` suggests what to run next.
5. Long lists are capped by ``--max-items`` (default 20); full data is in files
   listed under ``artifacts``.
6. Commands that write never overwrite without ``--overwrite``; ``--dry-run``
   validates and plans without heavy work.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
import traceback
from typing import Any, Callable, Sequence

from .. import __version__
from ..errors import ERROR_CODES, EXIT_CODE_DOCS, CumlxError, ExitCode
from .output import Result, to_json, to_text

FORMATS = ('auto', 'json', 'text')


class UsageError(Exception):
    def __init__(self, message: str, prog: str):
        super().__init__(message)
        self.prog = prog


class AgentArgumentParser(argparse.ArgumentParser):
    """argparse that raises instead of printing usage and exiting."""

    def error(self, message: str):  # type: ignore[override]
        raise UsageError(message, self.prog)


# Metadata used by `describe`. Arguments are introspected from argparse so the
# description can never drift from the real parser.
COMMANDS: dict[str, dict[str, Any]] = {
    'describe': dict(
        summary='Print the machine-readable CLI specification (commands, arguments, exit and error codes).',
        requires=[], stage='stable',
        examples=['cumlx describe', 'cumlx describe --command diff']),
    'doctor': dict(
        summary='Report installed dependencies and which commands are ready; with --require, fail when not ready.',
        requires=[], stage='stable',
        examples=['cumlx doctor', 'cumlx doctor --probe --require capture,diff']),
    'fixture': dict(
        summary='Create a tiny random HF model (safetensors) for offline end-to-end tests.',
        requires=['capture'], stage='beta',
        examples=['cumlx fixture tiny-llama -o /tmp/tiny']),
    'capture': dict(
        summary='Run the original PyTorch model and record per-module inputs/outputs as a golden trace.',
        requires=['capture'], stage='beta',
        examples=["cumlx capture /tmp/tiny --token-ids 1,5,9,23 -o /tmp/golden",
                  "cumlx capture Qwen/Qwen2.5-0.5B --prompt 'Hello' -o golden/",
                  "cumlx capture Qwen/Qwen2.5-0.5B --prompt 'Hello' --weights-from-mlx ./qwen-4bit -o golden-q/"]),
    'diff': dict(
        summary='Run an MLX port on the trace inputs, compare every module and locate the first divergence.',
        requires=['diff'], stage='beta',
        examples=['cumlx diff /tmp/golden /tmp/tiny -o /tmp/report',
                  'cumlx diff golden/ ./qwen-mlx --device cpu',
                  'cumlx diff golden/ ./qwen-mlx --tf32 on']),
    'conform': dict(
        summary='Check FP4 (NVFP4/MXFP4) decode conformance of a target against the CUMLX bit-exact codec.',
        requires=['diff'], stage='beta',
        examples=['cumlx conform --target mlx', 'cumlx conform --target mlx --formats mxfp4']),
    'weights': dict(
        summary='Compare port weights with the original safetensors without running a model (fast; names transposes and scale errors).',
        requires=['diff'], stage='beta',
        examples=['cumlx weights Qwen/Qwen2.5-0.5B ./qwen-mlx', 'cumlx weights /tmp/tiny /tmp/tiny -o weights.json']),
    'report': dict(
        summary='Summarize an existing diff report.json (re-read with a different --max-items).',
        requires=[], stage='beta',
        examples=['cumlx report /tmp/report/report.json --max-items 50']),
    'selftest': dict(
        summary='Prove the verifier works here: clean port must pass and injected faults must be located.',
        requires=[], stage='beta',
        examples=['cumlx selftest', 'cumlx selftest --core-only']),
}

AGENT_RULES = [
    'Run `cumlx doctor --require <commands>` first; follow errors[].hint to install what is missing.',
    'Parse stdout as one JSON document; never scrape text output.',
    'Branch on exit_code: 0 ok, 1 diverged, 5 incomplete (not a pass), 2/3/4 fix input or environment.',
    'On divergence read data.first_divergence (module, basis, confidence, explanation) before anything else.',
    'Use --dry-run to validate paths and plans before long runs; use --max-items to bound output size.',
    'Never treat `incomplete` or skipped checks as success.',
    'Run `cumlx weights` before `cumlx diff`: it takes seconds and names transposed or rescaled weights.',
    'Read data.execution: results depend on the device and on TF32, which MLX enables by default on M5 and CUDA.',
    'chain_drift is native-dtype rounding, not a defect. Read W_WEIGHT_ROUNDING_NOISE as a limit on sensitivity.',
]


def _common(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group('output (all commands)')
    g.add_argument('--format', choices=FORMATS, default=os.environ.get('CUMLX_FORMAT', 'auto'),
                   help='json = one JSON document on stdout; auto = json when stdout is not a TTY '
                        '(env CUMLX_FORMAT).')
    g.add_argument('--max-items', type=int, default=int(os.environ.get('CUMLX_MAX_ITEMS', '20')),
                   help='cap list lengths in stdout (-1 = unlimited; env CUMLX_MAX_ITEMS). Default 20.')
    g.add_argument('-v', '--verbose', action='store_true', help='progress logs on stderr')


def build_parser() -> AgentArgumentParser:
    parser = AgentArgumentParser(
        prog='cumlx', description='CUDA->MLX porting parity verifier. Agents: run `cumlx describe`.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--version', action='version', version=f'cumlx {__version__}')
    sub = parser.add_subparsers(dest='command', metavar='COMMAND', parser_class=AgentArgumentParser)

    def add(name: str) -> argparse.ArgumentParser:
        meta = COMMANDS[name]
        p = sub.add_parser(name, help=meta['summary'], description=meta['summary'],
                           epilog='examples:\n  ' + '\n  '.join(meta['examples']),
                           formatter_class=argparse.RawDescriptionHelpFormatter)
        _common(p)
        return p

    p = add('describe')
    p.add_argument('--command', dest='target_command', choices=sorted(COMMANDS),
                   help='describe only this command')

    p = add('doctor')
    p.add_argument('--probe', action='store_true', help='import backends and query devices (slower)')
    p.add_argument('--require', default='', metavar='CMD[,CMD]',
                   help='exit 3 unless these commands are ready, e.g. capture,diff')

    p = add('fixture')
    p.add_argument('kind', choices=['tiny-llama'], help='fixture to create')
    p.add_argument('-o', '--output', required=True, help='directory to create')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--overwrite', action='store_true')

    p = add('capture')
    p.add_argument('model', help='HF Hub id or local directory of the ORIGINAL PyTorch model')
    p.add_argument('-o', '--output', required=True, help='trace directory to create')
    src = p.add_argument_group('inputs (at least one; each value is one sample)')
    src.add_argument('--prompt', action='append', default=[], help='prompt text (repeatable; needs a tokenizer)')
    src.add_argument('--prompts-file', help='UTF-8 file, one prompt per line')
    src.add_argument('--token-ids', action='append', default=[], metavar='ID,ID,...',
                     help='comma-separated token ids (repeatable; no tokenizer needed)')
    p.add_argument('--max-tokens', type=int, default=64, help='truncate each sample to this many tokens')
    p.add_argument('--include', help='regex: only record modules whose name matches (root is always recorded)')
    p.add_argument('--exclude', help='regex: do not record modules whose name matches')
    p.add_argument('--store-dtype', choices=['float32', 'float16'], default='float32')
    p.add_argument('--device', default='cpu', help='torch device for the golden run (cpu, cuda, mps)')
    p.add_argument('--revision', help='HF revision (branch, tag or commit)')
    p.add_argument('--trust-remote-code', action='store_true', help='allow model repo code to run (security risk)')
    p.add_argument('--weights-from-mlx', metavar='PORT',
                   help='quantization-aware golden: load dequantized weights of this MLX port into the PyTorch model')
    p.add_argument('--max-bytes', type=int, default=2 * 1024**3, help='refuse traces larger than this')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='validate arguments and inputs; do not load the model')

    p = add('diff')
    p.add_argument('trace', help='trace directory written by `cumlx capture`')
    p.add_argument('port', help='MLX port: local directory or HF id (mlx-lm format)')
    p.add_argument('-o', '--output', help='report directory to create (report.json, report.md)')
    p.add_argument('--mode', choices=['isolated', 'chain'], default='isolated',
                   help='isolated = also replay each single-input module on its golden input (localizes faults)')
    p.add_argument('--loader', default='mlx_lm', help="'mlx_lm' or 'python:pkg.module:function' returning an nn.Module")
    p.add_argument('--mapping', help='JSON rules mapping PyTorch module names to MLX names')
    p.add_argument('--profile', choices=['auto', 'float32', 'float16', 'bfloat16', 'quantized'], default='auto',
                   help='threshold profile; auto = from the port parameter dtype')
    p.add_argument('--rel-l2', type=float, help='override the rel_l2 threshold')
    p.add_argument('--device', choices=['auto', 'cpu', 'gpu'], default='auto',
                   help='MLX device; auto = the MLX default (GPU on Apple Silicon). Never falls back silently')
    p.add_argument('--tf32', choices=['off', 'on'], default='off',
                   help='off = full float32 matmul. on = the MLX default on M5/CUDA: reduced precision, raises the threshold')
    p.add_argument('--include', help='regex: only compare modules whose name matches (root always compared)')
    p.add_argument('--exclude', help='regex: skip modules whose name matches')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='load trace and port and report the name mapping only')

    p = add('conform')
    p.add_argument('--target', choices=['mlx'], default='mlx', help='implementation to check')
    p.add_argument('--formats', default='nvfp4,mxfp4', help='comma-separated FP4 formats')
    p.add_argument('-o', '--output', help='write full findings JSON to this file')
    p.add_argument('--overwrite', action='store_true')

    p = add('weights')
    p.add_argument('original', help='HF id, local directory or safetensors file of the ORIGINAL checkpoint')
    p.add_argument('port', help='MLX port: local directory or HF id (mlx-lm format)')
    p.add_argument('-o', '--output', help='write every parameter to this JSON file')
    p.add_argument('--loader', default='mlx_lm', help="'mlx_lm' or 'python:pkg.module:function' returning an nn.Module")
    p.add_argument('--mapping', help='JSON rules mapping original parameter names to MLX names')
    p.add_argument('--revision', help='HF revision of the original (branch, tag or commit)')
    p.add_argument('--overwrite', action='store_true')

    p = add('report')
    p.add_argument('path', help='report.json written by `cumlx diff -o`')

    p = add('selftest')
    p.add_argument('--core-only', action='store_true', help='only checks that need no optional dependency')
    p.add_argument('--keep', metavar='DIR', help='keep fixture/trace/report files in this new directory')
    p.add_argument('--device', choices=['auto', 'cpu', 'gpu'], default='auto', help='MLX device for the end-to-end checks')
    return parser


def _detect_format(argv: Sequence[str]) -> str:
    fmt = os.environ.get('CUMLX_FORMAT', 'auto')
    for i, a in enumerate(argv):
        if a == '--format' and i + 1 < len(argv):
            fmt = argv[i + 1]
        elif a.startswith('--format='):
            fmt = a.split('=', 1)[1]
    if fmt not in FORMATS:
        fmt = 'auto'
    if fmt == 'auto':
        try:
            return 'text' if sys.stdout.isatty() else 'json'
        except (AttributeError, ValueError):
            return 'json'
    return fmt


def _describe_parser(parser: argparse.ArgumentParser) -> dict[str, dict]:
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    out = {}
    for name, p in sub.choices.items():
        args = []
        for act in p._actions:
            if isinstance(act, argparse._HelpAction):
                continue
            kind = ('flag' if act.nargs == 0 else 'list' if isinstance(act, argparse._AppendAction) else 'value')
            typ = getattr(act.type, '__name__', None) or 'str'
            if act.nargs == 0:
                typ = 'bool'
            args.append({
                'name': act.dest, 'flags': list(act.option_strings), 'positional': not act.option_strings,
                'kind': kind, 'type': typ, 'required': bool(act.required) or not act.option_strings,
                'default': None if act.default is argparse.SUPPRESS else act.default,
                'choices': list(act.choices) if act.choices else None, 'help': act.help,
            })
        out[name] = {**COMMANDS[name], 'arguments': args}
    return out


def _run_describe(args, parser) -> Result:
    cmds = _describe_parser(parser)
    if args.target_command:
        cmds = {args.target_command: cmds[args.target_command]}
    r = Result('describe', summary=f'{len(cmds)} command(s); stdout schema cumlx.result.v1')
    r.data = {
        'tool': 'cumlx', 'version': __version__,
        'purpose': 'Verify that an MLX port computes the same thing as the original (CUDA/PyTorch) model, '
                   'module by module, and locate the first divergence.',
        'agent_rules': AGENT_RULES,
        'workflow': ['cumlx doctor --require capture,diff',
                     'cumlx capture <original> --prompt TEXT -o golden/',
                     'cumlx diff golden/ <mlx-port> -o report/',
                     'read data.first_divergence; open report/report.md for the full table'],
        'commands': cmds,
        'exit_codes': {str(int(k)): {'name': k.name, 'meaning': v} for k, v in EXIT_CODE_DOCS.items()},
        'error_codes': {k: {'exit_code': int(v[0]), 'meaning': v[1]} for k, v in ERROR_CODES.items()},
        'env': {'CUMLX_FORMAT': 'auto|json|text', 'CUMLX_MAX_ITEMS': 'integer, -1 = unlimited'},
        'result_envelope': ['schema', 'command', 'ok', 'status', 'exit_code', 'summary', 'data', 'artifacts',
                            'warnings', 'errors', 'next_steps', 'truncated', 'cumlx_version', 'elapsed_ms'],
    }
    r.suggest('cumlx doctor', 'check which commands are ready in this environment')
    return r


def _dispatch(args, parser) -> Result:
    if args.command == 'describe':
        return _run_describe(args, parser)
    from . import commands
    handler: Callable[[Any], Result] = getattr(commands, 'run_' + args.command)
    return handler(args)


@contextlib.contextmanager
def _stdout_guard():
    """Send everything written to stdout (Python and native code) to stderr; yield the real stdout.

    Falls back to a Python-level redirect when stdout has no file descriptor (e.g. under pytest).
    """
    try:
        sys.stdout.flush()
        fd = sys.stdout.fileno()
        saved = os.dup(fd)
    except (AttributeError, OSError, ValueError):
        real = sys.stdout
        with contextlib.redirect_stdout(sys.stderr):
            yield lambda text: (real.write(text), real.flush())
        return
    try:
        os.dup2(sys.stderr.fileno(), fd)
        with contextlib.redirect_stdout(sys.stderr):
            yield lambda text: os.write(saved, text.encode())
    finally:
        sys.stdout.flush()
        os.dup2(saved, fd)
        os.close(saved)


def _command_name(argv: Sequence[str]) -> str:
    return next((a for a in argv if a in COMMANDS), 'cumlx')


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    fmt = _detect_format(argv)
    started = time.perf_counter()
    parser = build_parser()
    if not argv or (argv[0].startswith('-') and argv[0] not in ('-h', '--help', '--version')
                    and _command_name(argv) == 'cumlx'):
        argv = ['describe'] + argv  # bare `cumlx` (optionally with output flags) describes itself
    if any(a in ('-h', '--help', '--version') for a in argv):
        parser.parse_args(argv)  # human help/version text on stdout; argparse exits here
        return 0
    name = _command_name(argv)
    with _stdout_guard() as emit:
        try:
            args = parser.parse_args(argv)
            args.argv = list(argv)
            result = _dispatch(args, parser)
        except UsageError as exc:
            result = Result(name, ExitCode.USAGE, summary=f'invalid arguments: {exc}')
            result.errors.append({'code': 'E_USAGE', 'message': str(exc),
                                  'hint': f'run `cumlx describe --command {name}`' if name in COMMANDS
                                  else 'run `cumlx describe`'})
        except CumlxError as exc:
            result = Result(name, exc.exit_code, summary=exc.message)
            result.errors.append(exc.to_dict())
        except KeyboardInterrupt:
            result = Result(name, ExitCode.INTERNAL, summary='interrupted')
            result.errors.append({'code': 'E_INTERNAL', 'message': 'interrupted by user'})
        except Exception as exc:  # noqa: BLE001 - last-resort envelope for agents
            result = Result(name, ExitCode.INTERNAL, summary=f'internal error: {type(exc).__name__}: {exc}')
            result.errors.append({'code': 'E_INTERNAL', 'message': f'{type(exc).__name__}: {exc}',
                                  'hint': 'this is a CUMLX bug; include this JSON in the report',
                                  'details': {'traceback': traceback.format_exc(limit=8).splitlines()[-12:]}})
        if any(e['code'] == 'E_DEP_MISSING' for e in result.errors):
            result.suggest('cumlx doctor', 'see every missing dependency and its install command')
        doc = result.to_dict(version=__version__, elapsed_ms=int((time.perf_counter() - started) * 1000))
        emit((to_json(doc) if fmt == 'json' else to_text(doc)) + '\n')
    return int(result.exit_code)


def entrypoint() -> None:
    sys.exit(main())
