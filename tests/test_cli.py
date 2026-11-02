"""Agent contract of the `cumlx` CLI. Needs only NumPy."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from cumlx.cli import main
from cumlx.errors import ERROR_CODES, ExitCode

ROOT = Path(__file__).resolve().parents[1]
ENVELOPE = ['schema', 'command', 'ok', 'status', 'exit_code', 'summary', 'data', 'artifacts', 'warnings',
            'errors', 'next_steps', 'truncated', 'cumlx_version', 'elapsed_ms']


def run(capsys, *argv):
    code = main(['--format', 'json', *argv] if argv and argv[0].startswith('-') else [*argv, '--format', 'json'])
    out = capsys.readouterr().out
    doc = json.loads(out)  # stdout must be exactly one JSON document
    assert list(doc) == ENVELOPE
    assert doc['exit_code'] == code
    return code, doc


def test_bare_invocation_describes_itself(capsys):
    code = main(['--format', 'json'])
    doc = json.loads(capsys.readouterr().out)
    assert code == 0 and doc['command'] == 'describe'
    assert {'capture', 'diff', 'conform', 'doctor', 'selftest'} <= set(doc['data']['commands'])


def test_describe_is_complete_and_machine_readable(capsys):
    code, doc = run(capsys, 'describe')
    assert code == 0 and doc['ok'] and doc['status'] == 'ok'
    data = doc['data']
    assert data['agent_rules'] and data['workflow']
    assert set(data['exit_codes']) == {str(int(c)) for c in ExitCode}
    assert set(data['error_codes']) == set(ERROR_CODES)
    diff = data['commands']['diff']
    names = {a['name'] for a in diff['arguments']}
    assert {'trace', 'port', 'mode', 'format', 'max_items', 'dry_run'} <= names
    mode = next(a for a in diff['arguments'] if a['name'] == 'mode')
    assert mode['choices'] == ['isolated', 'chain'] and mode['default'] == 'isolated'
    assert all(cmd['examples'] for cmd in data['commands'].values())


def test_describe_single_command(capsys):
    code, doc = run(capsys, 'describe', '--command', 'capture')
    assert code == 0 and list(doc['data']['commands']) == ['capture']


def test_usage_errors_are_json_with_hint(capsys):
    code, doc = run(capsys, 'diff', 'only-one-positional')
    assert code == ExitCode.USAGE and doc['status'] == 'error' and doc['command'] == 'diff'
    err = doc['errors'][0]
    assert err['code'] == 'E_USAGE' and 'describe --command diff' in err['hint']


def test_unknown_command(capsys):
    code, doc = run(capsys, 'frobnicate')
    assert code == ExitCode.USAGE and doc['errors'][0]['code'] == 'E_USAGE'


def test_missing_trace_is_input_error(capsys, tmp_path):
    pytest.importorskip('mlx.core')
    code, doc = run(capsys, 'diff', str(tmp_path / 'nope'), str(tmp_path / 'port'))
    assert code == ExitCode.INPUT and doc['errors'][0]['code'] == 'E_INPUT_NOT_FOUND'


def test_capture_validates_before_loading(capsys, tmp_path):
    code, doc = run(capsys, 'capture', 'some/model', '-o', str(tmp_path / 'g'))
    assert code == ExitCode.USAGE and 'no inputs' in doc['errors'][0]['message']
    code, doc = run(capsys, 'capture', 'some/model', '--token-ids', '1,x', '-o', str(tmp_path / 'g'))
    assert code == ExitCode.USAGE
    code, doc = run(capsys, 'capture', 'some/model', '--token-ids', '1,2', '--include', '(', '-o', str(tmp_path / 'g'))
    assert code == ExitCode.USAGE and 'regex' in doc['errors'][0]['message']


def test_capture_dry_run_does_not_write(capsys, tmp_path):
    out = tmp_path / 'golden'
    code, doc = run(capsys, 'capture', 'some/model', '--token-ids', '1,2,3', '-o', str(out), '--dry-run')
    assert code == 0 and doc['data']['dry_run'] is True and doc['data']['plan']['samples'] == 1
    assert not out.exists()
    assert doc['next_steps'] and '--dry-run' not in doc['next_steps'][0]['command']


def test_refuses_to_overwrite(capsys, tmp_path):
    out = tmp_path / 'golden'
    out.mkdir()
    (out / 'precious.txt').write_text('keep me')
    code, doc = run(capsys, 'capture', 'm', '--token-ids', '1', '-o', str(out), '--dry-run')
    assert code == ExitCode.USAGE and doc['errors'][0]['code'] == 'E_OUTPUT_EXISTS'
    code, doc = run(capsys, 'capture', 'm', '--token-ids', '1', '-o', str(out), '--overwrite', '--dry-run')
    assert code == ExitCode.USAGE and 'refusing' in doc['errors'][0]['hint']
    assert (out / 'precious.txt').read_text() == 'keep me'


def test_doctor_reports_and_require_fails_closed(capsys, monkeypatch):
    code, doc = run(capsys, 'doctor')
    assert code == 0 and 'commands_ready' in doc['data']
    from cumlx.verify import env
    real = env.util.find_spec
    monkeypatch.setattr(env.util, 'find_spec', lambda m: None if m == 'torch' else real(m))
    code, doc = run(capsys, 'doctor', '--require', 'capture')
    assert code == ExitCode.ENVIRONMENT
    err = doc['errors'][0]
    assert err['code'] == 'E_DEP_MISSING' and "cumlx-lab[capture]" in err['hint']
    code, doc = run(capsys, 'doctor', '--require', 'nonsense')
    assert code == ExitCode.USAGE


def test_selftest_core_only(capsys):
    code, doc = run(capsys, 'selftest', '--core-only')
    assert code == 0, doc
    assert {s['id'] for s in doc['data']['steps']['items']} == {'C1', 'C2', 'C3'}


def test_max_items_truncates_and_flags(capsys):
    code, doc = run(capsys, 'selftest', '--core-only', '--max-items', '1')
    steps = doc['data']['steps']
    assert doc['truncated'] is True and len(steps['items']) == 1 and steps['omitted'] == 2


def test_report_command_mirrors_verdict(capsys, tmp_path):
    rep = {'schema': 'cumlx.diff-report.v1', 'verdict': 'diverged',
           'summary': {'pass': 1, 'fail': 1, 'unverified': 0, 'skipped': 0, 'ignored': 0},
           'first_divergence': {'module': 'a.b', 'basis': 'isolated', 'confidence': 'high'},
           'records': [{'sample': 's0000', 'name': 'a.b', 'type': 'Linear', 'status': 'fail',
                        'chain': {'rel_l2': 0.5}, 'isolated': {'rel_l2': 0.5}}]}
    (tmp_path / 'report.json').write_text(json.dumps(rep))
    code, doc = run(capsys, 'report', str(tmp_path))
    assert code == ExitCode.DIVERGED and doc['data']['first_divergence']['module'] == 'a.b'
    assert doc['data']['failures']['items'][0]['isolated_rel_l2'] == 0.5


def test_text_format_for_humans(capsys):
    code = main(['describe', '--format', 'text'])
    out = capsys.readouterr().out
    assert code == 0 and out.startswith('[OK] cumlx describe') and 'exit_code=0' in out


def test_subprocess_stdout_is_pure_json_when_piped():
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
    env.pop('CUMLX_FORMAT', None)
    p = subprocess.run([sys.executable, '-m', 'cumlx', 'selftest', '--core-only', '-v'],
                       capture_output=True, text=True, env=env, check=False)
    doc = json.loads(p.stdout)  # auto format = json because stdout is a pipe
    assert p.returncode == doc['exit_code'] == 0
    assert '[cumlx]' in p.stderr  # verbose logs go to stderr only
