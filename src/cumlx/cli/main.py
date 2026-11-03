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
from collections.abc import Callable, Sequence
import contextlib
import os
import sys
import time
import traceback
from typing import Any

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
        summary='Report installed dependencies and which commands are ready; with --require or --stack, fail when not ready.',
        requires=[], stage='stable',
        examples=['cumlx doctor', 'cumlx doctor --probe --require capture,diff', 'cumlx doctor --stack']),
    'fixture': dict(
        summary='Create a tiny random HF model (safetensors) for offline end-to-end tests.',
        requires=['capture'], stage='beta',
        examples=['cumlx fixture tiny-llama -o /tmp/tiny', 'cumlx fixture tiny-moe -o /tmp/tiny-moe']),
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
        examples=['cumlx conform --target mlx', 'cumlx conform --target mlx --formats mxfp4',
                  'cumlx conform --checkpoint nvidia/model-NVFP4 --against someone/model-MLX-NVFP4']),
    'weights': dict(
        summary='Compare port weights with the original safetensors without running a model (fast; names transposes and scale errors).',
        requires=['diff'], stage='beta',
        examples=['cumlx weights Qwen/Qwen2.5-0.5B ./qwen-mlx', 'cumlx weights /tmp/tiny /tmp/tiny -o weights.json']),
    'scan': dict(
        summary='Triage a CUDA repository or an HF model: read its source as text and rank the work to run it on Apple Silicon.',
        requires=[], stage='beta',
        examples=['cumlx scan ./some-cuda-repo', 'cumlx scan Qwen/Qwen3-0.6B']),
    'recipe': dict(
        summary='Name what to use on Apple Silicon for a CUDA dependency, and show whether each option was run or only read about.',
        requires=[], stage='beta',
        examples=['cumlx recipe', 'cumlx recipe bitsandbytes', 'cumlx recipe ./some-cuda-repo --verified-only', 'cumlx recipe scan.json']),
    'run': dict(
        summary='Run an unmodified PyTorch script that assumes CUDA on MPS, and record each redirected call. Running is not proof.',
        requires=['run'], stage='experimental',
        examples=['cumlx run train.py --target mps --report run-report.json', 'cumlx run train.py -- --epochs 1 --lr 1e-3',
                  'cumlx run --module my.pkg.train', 'cumlx run train.py --substitute flash_attn=mps_flash_attn',
                  'cumlx run train.py --trace-losses --seed 1 --report mps/seed1.json']),
    'rundiff': dict(
        summary='Compare the loss traces of `cumlx run` on CUDA and on the target, and give a proof grade.',
        requires=[], stage='experimental',
        examples=['cumlx rundiff golden-cuda/ runs-mps/', 'cumlx rundiff cuda/seed0.json mps/seed0.json']),
    'port': dict(
        summary='Convert a model with mlx-lm and prove the result: convert, compare weights, capture a golden, diff.',
        requires=['capture', 'diff'], stage='beta',
        examples=['cumlx port Qwen/Qwen2.5-0.5B -o port/', 'cumlx port Qwen/Qwen2.5-0.5B --q-bits 4 -o port-4bit/ --prompt Hello']),
    'tracediff': dict(
        summary='Compare two golden traces of one model, such as a CUDA capture and a CPU capture.',
        requires=[], stage='beta',
        examples=['cumlx tracediff golden-cuda/ golden-cpu/']),
    'report': dict(
        summary='Summarize an existing diff report.json (re-read with a different --max-items).',
        requires=[], stage='beta',
        examples=['cumlx report /tmp/report/report.json --max-items 50']),
    'selftest': dict(
        summary='Prove the verifier works here: clean port must pass and injected faults must be located.',
        requires=[], stage='beta',
        examples=['cumlx selftest', 'cumlx selftest --core-only']),
    'kernel capture': dict(
        summary='Compile the original CUDA kernels with nvcc, run them on a real NVIDIA GPU and record an op-level golden. Executes the code.',
        requires=['nvcc'], stage='experimental',
        examples=['cumlx kernel capture kernels/ --cases cases.json -o kgolden/ --require-idle-gpu',
                  'cumlx kernel capture kernels/ --cases cases.json -o kgolden/ --dry-run']),
    'kernel verify': dict(
        summary='Run a Metal kernel on the inputs of an op-level golden and compare the outputs. An unknown or unmapped kernel is never a pass.',
        requires=['metal'], stage='experimental',
        examples=['cumlx kernel verify foo.metal kgolden/', 'cumlx kernel verify kernels/ kgolden/ -o report/',
                  'cumlx kernel verify translated.metal kgolden/ --kernel foo --engine torch']),
    'kernel port': dict(
        summary='Prepare a .cu file for translation (headers, templates, host code, PTX built-ins) and call a translator. A feature with no Metal equivalent is refused.',
        requires=[], stage='experimental',
        examples=['cumlx kernel port foo.cu -o foo.metal', 'cumlx kernel port foo.cu --preprocess-only -o foo.pre.cu',
                  'cumlx kernel port foo.cu -o foo.metal --instantiate "foo<float>"']),
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
    p.add_argument('--stack', action='store_true',
                   help='also report each platform layer, and exit 3 unless all are ready. No install, no network, and no NVIDIA probe')

    p = add('fixture')
    p.add_argument('kind', choices=['tiny-llama', 'tiny-moe'], help='fixture to create')
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
    p.add_argument('--decode-steps', type=int, default=1,
                   help='also record this many greedy steps on the KV cache (0 = prefill only)')
    p.add_argument('--layers', metavar='N[,N-M]',
                   help='record only these decoder layers and the modules outside the stack, for example 0,11,20-23')
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
    p.add_argument('--checkpoint', metavar='ORIGINAL',
                   help='ModelOpt NVFP4 checkpoint (HF id or path): compare its bytes with the MLX repack given by --against')
    p.add_argument('--against', metavar='REPACK', help='MLX NVFP4 repack of --checkpoint (HF id or path)')
    p.add_argument('--revision', help='HF revision of --checkpoint')
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

    p = add('scan')
    p.add_argument('target', help='local directory, or HF id (only text files are downloaded)')
    p.add_argument('--revision', help='HF revision')
    p.add_argument('--include-tests', action='store_true', help='also read test directories')
    p.add_argument('-o', '--output', help='write every finding to this JSON file')
    p.add_argument('--overwrite', action='store_true')

    p = add('recipe')
    p.add_argument('target', nargs='?', help='a package or recipe name, a directory, an HF id, or a `cumlx scan -o` report. Omit it to list the catalog')
    p.add_argument('--verified-only', action='store_true', help='keep only options that were run on Apple Silicon')
    p.add_argument('--revision', help='HF revision')
    p.add_argument('-o', '--output', help='write every recipe to this JSON file')
    p.add_argument('--overwrite', action='store_true')

    p = add('run')
    p.add_argument('script', nargs='?', help='the Python file to run. It is executed with your permissions')
    p.add_argument('script_args', nargs='*', help='arguments for the script. Put them after --')
    p.add_argument('--module', help='run this module like python -m, instead of a file')
    p.add_argument('--target', choices=['mps', 'cpu', 'cuda'], default='mps', help='the device that stands in for CUDA. cuda redirects nothing and records a golden on a real CUDA machine')
    p.add_argument('--trace-losses', action='store_true', help='record each scalar given to backward() in the report, for `cumlx rundiff`')
    p.add_argument('--seed', type=int, metavar='N', help='seed torch, NumPy and random with N at the start, and add N to each integer seed that the script sets')
    p.add_argument('--report', default='run-report.json', help='where to write the report of redirected calls')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--allow-cpu-fallback', action='store_true', help='let operations that MPS lacks run on the CPU. Off by default, so a missing operation is an error')
    p.add_argument('--substitute', action='append', default=[], metavar='NAME=MODULE', help='serve MODULE when the script imports NAME. Repeatable')
    p.add_argument('--timeout', type=float, help='stop the script after this many seconds')

    p = add('rundiff')
    p.add_argument('golden', help='a run report, or a directory of them, from `cumlx run --trace-losses --target cuda`')
    p.add_argument('candidate', help='a run report, or a directory of them, from the target')

    p = add('port')
    p.add_argument('model', help='HF id or local directory of the ORIGINAL model')
    p.add_argument('-o', '--output', required=True, help='directory to create: mlx/, golden/, report/, port.json')
    p.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], help='dtype of the port; default: as in the checkpoint')
    p.add_argument('--q-bits', type=int, choices=[4, 8], help='quantize the port; the golden then holds its weights')
    p.add_argument('--prompt', action='append', default=[], help='prompt for the golden (repeatable; needs a tokenizer)')
    p.add_argument('--token-ids', action='append', default=[], metavar='ID,ID,...', help='token ids for the golden (repeatable)')
    p.add_argument('--device', choices=['auto', 'cpu', 'gpu'], default='auto', help='MLX device for the comparison')
    p.add_argument('--revision', help='HF revision of the original')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='check the architecture and show the plan; download no weights')

    p = add('tracediff')
    p.add_argument('a', help='first trace directory')
    p.add_argument('b', help='second trace directory')
    p.add_argument('--rel-l2', type=float, default=1e-4, help='report outputs that differ by more than this')

    p = add('report')
    p.add_argument('path', help='report.json written by `cumlx diff -o`')

    p = add('selftest')
    p.add_argument('--core-only', action='store_true', help='only checks that need no optional dependency')
    p.add_argument('--keep', metavar='DIR', help='keep fixture/trace/report files in this new directory')
    p.add_argument('--device', choices=['auto', 'cpu', 'gpu'], default='auto', help='MLX device for the end-to-end checks')

    kernel = sub.add_parser('kernel', help='CUDA kernels: capture a golden, translate, verify',
                            description='CUDA kernels: capture a golden on an NVIDIA machine, translate to Metal, verify on a Mac.')
    ksub = kernel.add_subparsers(dest='kernel_command', metavar='SUBCOMMAND', parser_class=AgentArgumentParser, required=True)

    def add_kernel(name: str) -> argparse.ArgumentParser:
        meta = COMMANDS[f'kernel {name}']
        p = ksub.add_parser(name, help=meta['summary'], description=meta['summary'],
                            epilog='examples:\n  ' + '\n  '.join(meta['examples']),
                            formatter_class=argparse.RawDescriptionHelpFormatter)
        _common(p)
        return p

    p = add_kernel('capture')
    p.add_argument('kernels', help='a folder of .cu files, or one .cu file. It is compiled and executed')
    p.add_argument('--cases', required=True, help='cumlx.kernel-cases.v1 file: the launches to record')
    p.add_argument('-o', '--output', required=True, help='golden directory to create')
    p.add_argument('--header', help='source to put before each kernel; default: common_cuda.h beside the kernels')
    p.add_argument('--arch', help='nvcc -arch value; default: native')
    p.add_argument('--nvcc-flag', action='append', default=[], metavar='FLAG', help='extra nvcc flag, such as --nvcc-flag=-fmad=false. Repeatable')
    p.add_argument('--repeats', type=int, default=3, help='launches per case. The output memory is filled differently before each one')
    p.add_argument('--timeout', type=float, default=60.0, help='seconds allowed for each case to run')
    p.add_argument('--jobs', type=int, default=4, help='parallel compiler processes')
    p.add_argument('--require-idle-gpu', action='store_true', help='refuse to start, and stop, while another compute process is on the GPU. Use it on a shared machine')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='validate the cases and the sources; compile and run nothing')

    p = add_kernel('verify')
    p.add_argument('metal', help='a .metal file, or a folder of them. The file name selects the golden kernel')
    p.add_argument('golden', help='directory written by `cumlx kernel capture`')
    p.add_argument('-o', '--output', help='report directory to create (report.json, report.md)')
    p.add_argument('--kernel', help='the golden kernel that this file translates, when its name differs')
    p.add_argument('--case', action='append', default=[], metavar='ID', help='compare only this case. Repeatable. The verdict then covers only these')
    p.add_argument('--header', help='Metal source to put before the kernel; default: NAME.header.metal, or common_metal.h beside it')
    p.add_argument('--engine', choices=['auto', 'mlx', 'torch', 'woof'], default='auto',
                   help='auto = mx.fast.metal_kernel for a function body, torch.mps.compile_shader for a complete library. woof = its own launcher, for its translations; it runs that package')
    p.add_argument('--repeats', type=int, default=3, help='runs per case. Each run is compared')
    p.add_argument('--overwrite', action='store_true')

    p = add_kernel('port')
    p.add_argument('source', help='the .cu file to translate')
    p.add_argument('-o', '--output', required=True, help='the file to write. It also writes NAME.port.json beside it')
    p.add_argument('--translator', default='woof', help='woof, or python:MODULE:ATTRIBUTE for your own adapter, which runs that code')
    p.add_argument('--preprocess-only', action='store_true', help='write the prepared CUDA source and stop. Needs no translator, and translates nothing')
    p.add_argument('--instantiate', action='append', default=[], metavar='NAME<ARGS>', help='a concrete instance of a template kernel, such as vector_add<float>. Repeatable')
    p.add_argument('--include-dir', action='append', default=[], metavar='DIR', help='where to look for local #include files. Repeatable')
    p.add_argument('--header', help='for a function body: source that goes before it; default: common_cuda.h beside it')
    p.add_argument('--cases', help='for a function body: a cases file that gives its launch constants. Use with --case')
    p.add_argument('--case', help='the case in --cases. It wraps the body in a complete kernel with those constants')
    p.add_argument('--demote-fp64', action='store_true', help='turn double into float. Changes what the kernel computes; `cumlx kernel verify` judges the result')
    p.add_argument('--allow-warp-level', action='store_true', help='pass warp-level built-ins to the translator instead of refusing them. Metal does not promise their meaning; `cumlx kernel verify` judges the result on this device')
    p.add_argument('--overwrite', action='store_true')
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


def _subparsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    return next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction)).choices


def _describe_parser(parser: argparse.ArgumentParser) -> dict[str, dict]:
    parsers: dict[str, argparse.ArgumentParser] = {}
    for name, p in _subparsers(parser).items():
        if name == 'kernel':
            parsers.update({f'kernel {child}': q for child, q in _subparsers(p).items()})
        else:
            parsers[name] = p
    out = {}
    for name, p in parsers.items():
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
    name = f'kernel_{args.kernel_command}' if args.command == 'kernel' else args.command
    handler: Callable[[Any], Result] = getattr(commands, 'run_' + name)
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
    for i, a in enumerate(argv):
        if a == 'kernel' and f"kernel {argv[i + 1] if i + 1 < len(argv) else ''}" in COMMANDS:
            return f'kernel {argv[i + 1]}'
        if a in COMMANDS:
            return a
    return 'cumlx'


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    original = list(argv)
    # What follows -- belongs to the script. argparse handles this mix of positionals differently between
    # Python versions (3.12.3 refuses what 3.13 accepts), and --help or --format there is not ours.
    script_args: list[str] = []
    if argv[:1] == ['run'] and '--' in argv:
        split = argv.index('--')
        argv, script_args = argv[:split], argv[split + 1:]
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
            args.argv = original
            if script_args:
                args.script_args = [*args.script_args, *script_args]
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
