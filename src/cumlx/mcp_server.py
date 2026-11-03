"""MCP server for ``cumlx``: lets a porting agent call the verifier as tools, over stdio (JSON-RPC 2.0).

    python -m cumlx.mcp_server            # or the `cumlx-mcp` console script

Every tool runs one ``cumlx`` command in a subprocess and returns its ``cumlx.result.v1`` envelope, so agents get
the same contract as the CLI. Tool schemas come from the argparse parser and cannot drift from it. A divergence is
a normal result: read ``exit_code`` and ``status``. ``isError`` is set only when the envelope reports an error.

Three arguments run code from elsewhere: ``--trust-remote-code``, ``--loader python:...`` and ``--translator``. Two
commands execute a file: ``run`` and ``kernel capture``. An agent cannot use any of them unless the person who starts
this server passes ``--allow-code-execution``.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import subprocess
import sys
from typing import Any

from . import __version__
from .cli.main import COMMANDS, _describe_parser, build_parser

SERVER_NAME = 'cumlx'
DEFAULT_PROTOCOL = '2025-06-18'
HIDDEN = {'command', 'format', 'verbose'}
CODE_EXECUTION = {'trust_remote_code', 'loader', 'translator'}
EXECUTES_FILES = {'run', 'kernel capture'}  # each runs what the caller points it at, as remote model code would
TOOL_TIMEOUT_SECONDS = 3600
JSON_TYPES = {'bool': 'boolean', 'int': 'integer', 'float': 'number'}


def _schema(arguments: list[dict], *, allow_code: bool) -> tuple[dict, dict]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    by_name: dict[str, dict] = {}
    for a in arguments:
        if a['name'] in HIDDEN or (a['name'] in CODE_EXECUTION and not allow_code):
            continue
        by_name[a['name']] = a
        if a['kind'] == 'list':
            prop: dict[str, Any] = {'type': 'array', 'items': {'type': 'string'}}
        else:
            prop = {'type': JSON_TYPES.get(a['type'], 'string')}
        if a.get('help'):
            prop['description'] = a['help']
        if a.get('choices'):
            prop['enum'] = list(a['choices'])
        if a.get('default') not in (None, False, []):
            prop['default'] = a['default']
        properties[a['name']] = prop
        if a['required']:
            required.append(a['name'])
    schema = {'type': 'object', 'properties': properties, 'additionalProperties': False}
    if required:
        schema['required'] = required
    return schema, by_name


def build_tools(*, allow_code: bool = False) -> dict[str, dict]:
    tools = {}
    for name, spec in _describe_parser(build_parser()).items():
        if name in EXECUTES_FILES and not allow_code:
            continue
        schema, by_name = _schema(spec['arguments'], allow_code=allow_code)
        tools[f"cumlx_{name.replace(' ', '_')}"] = {'command': name, 'arguments': by_name, 'schema': schema,
                                                    'description': COMMANDS[name]['summary']}
    return tools


def _argv(tool: dict, params: dict[str, Any], options: Sequence[str] = ()) -> list[str]:
    """Build the command line. Values go in as separate arguments, never through a shell.

    A positional list goes after --, so that its items stay arguments of the script. ``options`` go before it.
    """
    argv = tool['command'].split()
    trailing: list[str] = []
    unknown = sorted(set(params) - set(tool['arguments']))
    if unknown:
        raise ValueError(f'unknown or disabled argument(s): {unknown}')
    for name, spec in tool['arguments'].items():
        if name not in params:
            continue
        value = params[name]
        if spec['positional'] and isinstance(value, list):
            trailing += [str(item) for item in value]
        elif spec['positional']:
            argv.append(str(value))
        elif spec['kind'] == 'flag':
            if value:
                argv.append(spec['flags'][-1])
        elif spec['kind'] == 'list':
            for item in value:
                argv += [spec['flags'][-1], str(item)]
        else:
            argv += [spec['flags'][-1], str(value)]
    argv += options
    return [*argv, '--', *trailing] if trailing else argv


def call_tool(tool: dict, params: dict[str, Any]) -> dict:
    argv = [sys.executable, '-m', 'cumlx', *_argv(tool, params, ('--format', 'json'))]
    p = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=TOOL_TIMEOUT_SECONDS)
    try:
        doc = json.loads(p.stdout)
    except json.JSONDecodeError:
        tail = (p.stderr or p.stdout)[-2000:]
        return {'content': [{'type': 'text', 'text': f'cumlx exited with {p.returncode} and no JSON result: {tail}'}],
                'isError': True}
    return {'content': [{'type': 'text', 'text': json.dumps(doc, ensure_ascii=False)}],
            'structuredContent': doc, 'isError': doc.get('status') == 'error'}


class Server:
    def __init__(self, *, allow_code: bool = False):
        self.tools = build_tools(allow_code=allow_code)

    def handle(self, message: dict) -> dict | None:
        method = message.get('method')
        msg_id = message.get('id')
        if msg_id is None:  # a notification has no reply
            return None
        try:
            return {'jsonrpc': '2.0', 'id': msg_id, 'result': self._result(str(method), message.get('params') or {})}
        except KeyError as exc:
            return self._error(msg_id, -32601, f'unknown: {exc.args[0]}')
        except (ValueError, TypeError) as exc:
            return self._error(msg_id, -32602, str(exc))
        except subprocess.TimeoutExpired:
            return self._error(msg_id, -32000, f'tool exceeded {TOOL_TIMEOUT_SECONDS} s')

    @staticmethod
    def _error(msg_id, code: int, message: str) -> dict:
        return {'jsonrpc': '2.0', 'id': msg_id, 'error': {'code': code, 'message': message}}

    def _result(self, method: str, params: dict) -> dict:
        if method == 'initialize':
            return {'protocolVersion': params.get('protocolVersion', DEFAULT_PROTOCOL), 'capabilities': {'tools': {}},
                    'serverInfo': {'name': SERVER_NAME, 'version': __version__},
                    'instructions': 'Verify that an MLX port computes the same as the original model. Start with '
                                    'cumlx_doctor, then cumlx_weights, cumlx_capture and cumlx_diff. Branch on exit_code: '
                                    '0 ok, 1 diverged (read data.first_divergence), 5 incomplete (not a pass).'}
        if method == 'ping':
            return {}
        if method == 'tools/list':
            return {'tools': [{'name': n, 'description': t['description'], 'inputSchema': t['schema']}
                              for n, t in self.tools.items()]}
        if method == 'tools/call':
            name = params.get('name')
            if name not in self.tools:
                raise ValueError(f'unknown tool {name!r}')
            return call_tool(self.tools[name], params.get('arguments') or {})
        raise KeyError(method)


def serve(server: Server, stdin=None, stdout=None) -> None:
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            reply: dict | None = Server._error(None, -32700, f'parse error: {exc}')
        else:
            reply = server.handle(message)
        if reply is not None:
            stdout.write(json.dumps(reply, ensure_ascii=False) + '\n')
            stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser(prog='cumlx-mcp', description='MCP stdio server for the cumlx verifier')
    ap.add_argument('--allow-code-execution', action='store_true',
                    help='let agents use `run`, `kernel capture`, --trust-remote-code, --translator and --loader python:... '
                         '(runs code from models or paths)')
    args = ap.parse_args()
    serve(Server(allow_code=args.allow_code_execution))


if __name__ == '__main__':
    main()
