"""spikes/m2_substitutes.py, on a local stub. No third-party code, no CUDA, no network, no extras."""
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPIKE = ROOT / 'spikes' / 'm2_substitutes.py'
LAUNCHER = ROOT / 'spikes' / 'm2_substitutes.sh'


def _driver():
    spec = importlib.util.spec_from_file_location('m2_substitutes', SPIKE)
    module = importlib.util.module_from_spec(spec)
    sys.modules['m2_substitutes'] = module
    spec.loader.exec_module(module)
    return module


CHECK = '''
def check(rec):
    import stubpkg
    rec.set_device('cpu', 'a stub')
    rec.at_most('error', abs(stubpkg.add(2.0, 3.0) - 5.0), 1e-9)
'''
CANNOT = '''
def check(rec):
    rec.set_device('cpu', 'a stub')
    rec.cannot('values', 'the README shows no signature')
'''
UNKNOWN_API = '''
def check(rec):
    import stubpkg
    rec.set_device('cpu', 'a stub')
    rec.attempt('undocumented', lambda: stubpkg.not_a_function(), api='docs')
    rec.at_most('error', abs(stubpkg.add(2.0, 3.0) - 5.0), 1e-9)
'''
README_API_MISSING = '''
def check(rec):
    import stubpkg
    rec.set_device('cpu', 'a stub')
    rec.attempt('documented', lambda: stubpkg.not_a_function())
    rec.at_most('error', abs(stubpkg.add(2.0, 3.0) - 5.0), 1e-9)
'''
CRASH = '''
def check(rec):
    import os
    rec.number('measured_before_the_crash', 7)
    os.abort()
'''
RAISES = '''
def check(rec):
    raise ValueError('a bug in the check')
'''
HANG = '''
def check(rec):
    import time
    time.sleep(60)
'''
OPTIONAL_FAILURE = '''
def check(rec):
    import stubpkg
    rec.set_device('cpu', 'a stub')
    rec.at_most('error', abs(stubpkg.add(2.0, 3.0) - 5.0), 1e-9)
    rec.check('readme_line_as_written', False, 'wrong keyword', required=False)
'''


def _package(root, name, body):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / 'stubpkg.py').write_text(body)
    return folder


def _manifest(tmp_path, cases):
    """A right stub and a wrong one, and one entry for each way that a candidate can end."""
    good = _package(tmp_path, 'good', 'def add(a, b):\n    return a + b\n')
    bad = _package(tmp_path, 'bad', 'def add(a, b):\n    return a + b + 0.5\n')
    entries = []
    for name, (source, path, extra) in cases.items():
        check = tmp_path / f'check_{name}.py'
        check.write_text(source)
        entries.append({'name': name, 'module': 'stubpkg', 'check_file': str(check), 'pythonpath': str({'good': good, 'bad': bad}[path]), 'source': 'a local stub', **extra})
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(entries))
    return manifest


def _run(manifest, work, *extra):
    return subprocess.run([sys.executable, str(SPIKE), '--work', str(work), '--manifest', str(manifest), *extra], capture_output=True, text=True, timeout=120)


def test_each_stub_ends_with_its_own_status_and_one_failure_does_not_stop_the_others(tmp_path):
    manifest = _manifest(tmp_path, {
        'right': (CHECK, 'good', {}),
        'wrong_number': (CHECK, 'bad', {}),
        'crash': (CRASH, 'good', {}),
        'still_runs_after_a_crash': (CHECK, 'good', {}),
        'check_raises': (RAISES, 'good', {}),
        'no_import': (CHECK, 'good', {'module': 'no_such_stub_module'}),
        'no_install': (CHECK, 'good', {'install': [sys.executable, '-c', 'raise SystemExit(7)']}),
        'cannot': (CANNOT, 'good', {}),
        'undocumented_api': (UNKNOWN_API, 'good', {}),
        'readme_api_missing': (README_API_MISSING, 'good', {}),
        'optional_failure': (OPTIONAL_FAILURE, 'good', {}),
    })
    process = _run(manifest, tmp_path / 'work')
    assert process.returncode == 1, process.stdout + process.stderr
    summary = json.loads((tmp_path / 'work' / 'summary.json').read_text())
    assert summary['schema'] == 'cumlx.spike.m2_substitutes.v1'
    assert summary['date']
    by_name = {r['name']: r for r in summary['results']}
    assert {name: r['status'] for name, r in by_name.items()} == {
        'right': 'pass', 'wrong_number': 'fail', 'crash': 'fail', 'still_runs_after_a_crash': 'pass', 'check_raises': 'fail', 'no_import': 'fail', 'no_install': 'fail',
        'cannot': 'could_not_test', 'undocumented_api': 'could_not_test', 'readme_api_missing': 'fail', 'optional_failure': 'pass',
    }
    assert summary['decision_inputs']['wrong_number']['status'] == 'fail'
    assert by_name['right']['numbers']['error'] == 0
    assert by_name['wrong_number']['numbers']['error'] == 0.5
    assert by_name['crash']['child_exit'] != 0
    assert by_name['crash']['numbers']['measured_before_the_crash'] == 7, 'what was measured before a crash must survive'
    assert by_name['no_import']['import'] is False and 'no_such_stub_module' in by_name['no_import']['import_error']
    assert by_name['no_install']['installed'] is False and by_name['no_install']['install_exit'] == 7
    assert by_name['check_raises']['checks'][0]['name'] == 'check_raised' and 'ValueError' in by_name['check_raises']['checks'][0]['detail']['traceback']
    assert by_name['optional_failure']['checks'][-1]['ok'] is False
    for key in ('[right] status pass', '[wrong_number] status fail', '[cannot] status could_not_test', 'installed no', 'import yes', '[done] 3 pass'):
        assert key in process.stdout


def test_a_check_that_hangs_is_a_timeout_failure(tmp_path):
    manifest = _manifest(tmp_path, {'hang': (HANG, 'good', {}), 'next': (CHECK, 'good', {})})
    process = _run(manifest, tmp_path / 'work', '--check-timeout', '3')
    summary = json.loads((tmp_path / 'work' / 'summary.json').read_text())
    assert process.returncode == 1
    assert [r['status'] for r in summary['results']] == ['fail', 'pass']
    assert summary['results'][0]['timed_out'] is True


def test_only_keeps_earlier_results_and_rejects_an_unknown_name(tmp_path):
    manifest = _manifest(tmp_path, {'one': (CHECK, 'good', {}), 'two': (CHECK, 'bad', {})})
    work = tmp_path / 'work'
    _run(manifest, work)
    assert _run(manifest, work, '--only', 'one').returncode == 0
    summary = json.loads((work / 'summary.json').read_text())
    assert {r['name']: r['status'] for r in summary['results']} == {'one': 'pass', 'two': 'fail'}
    assert _run(manifest, work, '--only', 'three').returncode == 2


def test_a_licence_file_is_read_from_the_installed_distribution(tmp_path):
    site = tmp_path / 'site'
    (site / 'stubpkg-1.0.dist-info' / 'licenses').mkdir(parents=True)
    (site / 'stubpkg.py').write_text('def add(a, b):\n    return a + b\n__version__ = "1.0"\n')
    info = site / 'stubpkg-1.0.dist-info'
    (info / 'METADATA').write_text('Metadata-Version: 2.4\nName: stubpkg\nVersion: 1.0\nLicense-Expression: MIT\n')
    (info / 'licenses' / 'LICENSE').write_text('MIT License\n\nCopyright (c) a stub\n')
    (info / 'RECORD').write_text('stubpkg.py,,\nstubpkg-1.0.dist-info/METADATA,,\nstubpkg-1.0.dist-info/licenses/LICENSE,,\n')
    check = tmp_path / 'check.py'
    check.write_text(CHECK)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps([{'name': 'licensed', 'module': 'stubpkg', 'dist': 'stubpkg', 'check_file': str(check), 'pythonpath': str(site)}]))
    assert _run(manifest, tmp_path / 'work').returncode == 0
    result = json.loads((tmp_path / 'work' / 'summary.json').read_text())['results'][0]
    assert result['licence']['metadata'] == 'MIT' and result['licence']['text_found'] is True
    assert result['licence']['files'][0]['first_line'] == 'MIT License'
    assert result['module_version'] == '1.0'


def test_final_status_never_turns_an_unknown_state_into_a_pass():
    driver = _driver()
    good = {'installed': True, 'import': True, 'child_exit': 0, 'finished': True, 'device': {'used': 'mps'}, 'checks': [{'name': 'a', 'ok': True, 'required': True}], 'could_not_test': []}
    assert driver.final_status(good)[0] == 'pass'
    for change, expected in (({'installed': False}, 'fail'), ({'timed_out': True, 'timeout': 1}, 'fail'), ({'import': None}, 'fail'), ({'import': False}, 'fail'),
                             ({'child_exit': -11}, 'fail'), ({'child_exit': None}, 'fail'), ({'finished': False}, 'fail'),
                             ({'checks': [{'name': 'a', 'ok': False, 'required': True}]}, 'fail'), ({'checks': []}, 'could_not_test'), ({'device': None}, 'could_not_test'),
                             ({'could_not_test': [{'name': 'x', 'reason': 'y', 'required': True}]}, 'could_not_test')):
        assert driver.final_status({**good, **change})[0] == expected, change
    assert driver.final_status({**good, 'could_not_test': [{'name': 'x', 'reason': 'y', 'required': False}]})[0] == 'pass'


def test_a_renamed_api_is_not_a_drop_in():
    driver = _driver()
    ok, bad = {'ok': True}, {'ok': False}
    assert driver.substitute_status({'a': ok}, {'b': ok}, True)[0] == 'pass'
    assert driver.substitute_status({'a': ok}, {'b': bad}, True)[0] == 'fail'
    assert driver.substitute_status({'a': bad}, {'b': ok}, True)[0] == 'fail'
    assert driver.substitute_status({'a': ok}, {'b': ok}, False)[0] == 'fail'
    assert driver.substitute_status({}, {}, True)[0] == 'fail'
    assert driver.substitute_status({'a': ok}, {}, True)[0] == 'could_not_test'


def test_every_candidate_is_pinned_and_the_resolver_sees_the_pinned_torch():
    driver = _driver()
    assert driver.TORCH == 'torch==2.15.0'
    names = [c['name'] for c in driver.CANDIDATES]
    assert len(names) == len(set(names))
    for candidate in driver.CANDIDATES:
        assert candidate['check'] in driver.CHECKS
        assert candidate['commit'] and len(candidate['commit']) == 40, candidate['name']
        for requirement in candidate['requirements']:
            assert '==' in requirement or '@' in requirement, requirement
        command = driver.install_plan(candidate, Path('/venv/bin/python'), Path('/repo'))[0]
        assert command[:3] == ['uv', 'pip', 'install']
        assert candidate.get('torch', driver.TORCH) in command and candidate['requirements'][0] in command
        assert command[-2:] == ['-e', '/repo'], 'the editable project must stay out of --only-binary'
        assert ':all:' not in command
        for substitute in candidate['substitutes']:
            assert substitute['script'] in driver.SUB_SCRIPTS
        assert candidate['substitutes'] or candidate.get('no_substitute')
    torches = {c['name']: c.get('torch', driver.TORCH) for c in driver.CANDIDATES}
    assert [name for name, torch in torches.items() if torch != driver.TORCH] == ['mps_flash_attn_torch213']


def test_each_hand_reference_matches_a_direct_computation():
    np = pytest.importorskip('numpy')
    driver = _driver()
    pose = driver._gl_camera(driver._ring_eye(0.0, 0.0, 3.0))
    u, v, depth = driver._project_gl(pose, (0.3, 0.3, 0.0), 150.0, 64.0)
    assert (u, v, depth) == pytest.approx((64 + 150 * 0.1, 64 - 150 * 0.1, 3.0))
    u, v, depth = driver._project_opencv((0.0, 0.0, -4.0), (0.5, 0.25, 0.0), 100.0, 64.0)
    assert (u, v, depth) == pytest.approx((64 - 12.5, 64 - 6.25, 4.0))
    image = driver._gaussian_image(64, 20.0, 40.0, 3.0, (1.0, 1.0, 1.0))
    got_u, got_v, peak = driver._blob_center(image)
    assert (got_u, got_v) == pytest.approx((20.0, 40.0), abs=0.05) and peak == pytest.approx(3.0, abs=0.1)
    assert driver._axis_sigma(image[40, :, 0]) == pytest.approx(3.0, rel=0.02)
    assert driver._clean({'a': float('nan'), 'b': np.float32(1.5), 'c': [float('inf')]}) == {'a': 'nan', 'b': 1.5, 'c': ['inf']}


def test_the_png_fixture_is_a_valid_image(tmp_path):
    np = pytest.importorskip('numpy')
    driver = _driver()
    path = tmp_path / 'image.png'
    driver._write_png(path, np.ones((4, 6, 3)) * 0.5)
    data = path.read_bytes()
    assert data.startswith(b'\x89PNG\r\n\x1a\n') and data[12:16] == b'IHDR' and int.from_bytes(data[16:20], 'big') == 6 and int.from_bytes(data[20:24], 'big') == 4


def test_a_substitute_run_goes_through_cumlx_run(tmp_path):
    torch = pytest.importorskip('torch')
    target = 'mps' if torch.backends.mps.is_available() else 'cpu'
    good = _package(tmp_path, 'pkg', 'def add(a, b):\n    return a + b\n')
    check = tmp_path / 'check.py'
    check.write_text(CHECK)
    drop_in = tmp_path / 'drop_in.py'
    drop_in.write_text('import stuboriginal\nattempt("route", "add", lambda: {"ok": stuboriginal.add(1, 2) == 3})\nattempt("drop_in", "names", lambda: names(stuboriginal, "add"))\n')
    renamed = tmp_path / 'renamed.py'
    renamed.write_text('import stuboriginal\nattempt("route", "add", lambda: {"ok": stuboriginal.add(1, 2) == 3})\nattempt("drop_in", "names", lambda: names(stuboriginal, "original_add"))\n')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps([{'name': 'mapped', 'module': 'stubpkg', 'check_file': str(check), 'pythonpath': str(good),
                                     'substitutes': [{'original': 'stuboriginal', 'module': 'stubpkg', 'script_file': str(drop_in)},
                                                     {'original': 'stuboriginal', 'module': 'stubpkg', 'script_file': str(renamed)}]}]))
    process = _run(manifest, tmp_path / 'work', '--target', target)
    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads((tmp_path / 'work' / 'summary.json').read_text())['results'][0]
    assert result['status'] == 'pass'
    assert [s['status'] for s in result['substitutes']] == ['pass', 'fail'], result['substitutes']
    assert result['substitutes'][0]['recorded_calls'] == ['stuboriginal']
    assert result['substitutes'][1]['drop_in']['names']['missing'] == ['original_add']


def test_the_launcher_starts_in_the_background_and_has_a_status_mode(tmp_path):
    python = ROOT / '.venv' / 'bin' / 'python'
    if not os.access(python, os.X_OK):
        pytest.skip('the launcher needs .venv/bin/python, which this checkout does not have')
    manifest = _manifest(tmp_path, {'right': (CHECK, 'good', {}), 'wrong_number': (CHECK, 'bad', {})})
    assert LAUNCHER.stat().st_mode & stat.S_IRUSR
    env = dict(os.environ, CUMLX_M2_SUBSTITUTES_ROOT=str(tmp_path / 'root'))
    start = time.monotonic()
    started = subprocess.run(['sh', str(LAUNCHER), '--manifest', str(manifest)], env=env, capture_output=True, text=True, timeout=60)
    assert started.returncode == 0, started.stdout + started.stderr
    assert started.stdout.startswith('Starting the M2 substitute evaluation in') and 'Started, pid' in started.stdout
    assert time.monotonic() - start < 30
    for _ in range(240):
        status = subprocess.run(['sh', str(LAUNCHER), 'status'], env=env, capture_output=True, text=True, timeout=60)
        if status.stdout.startswith('not running'):
            break
        time.sleep(0.25)
    assert status.stdout.startswith('not running') and '[done] 1 pass, 1 fail' in status.stdout
    summary = json.loads((tmp_path / 'root' / 'summary.json').read_text())
    assert [r['status'] for r in summary['results']] == ['pass', 'fail']
