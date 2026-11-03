"""Every launcher parses in the shells that it names, and uses no flag that exists on only one of BSD and GNU. Nothing here runs a launcher."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

REPO = Path(__file__).resolve().parent.parent
LAUNCHERS = sorted(path for folder in ('spikes', 'scripts') for path in (REPO / folder).glob('*.sh'))
IDS = [path.name for path in LAUNCHERS]

# What one of the two systems does not have, and what to write instead.
ONE_SYSTEM_ONLY = [
    (r"\bsed\s+-i\b", 'sed -i differs between BSD and GNU. Write to a new file, or use a small script'),
    (r"\bdf\s+-[a-zA-Z]*g\b", 'df -g is BSD only. Use df -k'),
    (r"\bstat\s+-[fc]\b", 'stat -f and stat -c each exist on one system. Use wc -c, or ls, or a small script'),
    (r"\bdate\s+-[jv]\b|\bdate\s+-d\b|\bdate\s+-r\b", 'date -j, -v and -d each exist on one system'),
    (r"\breadlink\s+-f\b", 'readlink -f is missing on older macOS. Use cd and pwd'),
    (r"\bgrep\s+-[a-zA-Z]*P", 'grep -P is GNU only. Use grep -E'),
    (r"\bfree\s+-", 'free is Linux only. Use cumlx.sysmem'),
    (r"\bnproc\b", 'nproc is GNU only'),
    (r"\bsha256sum\b|\bshasum\b", 'each exists on one system by default. Check for both, or use Python'),
    (r"\bmd5\b|\bmd5sum\b", 'each exists on one system'),
]
# A POSIX sh has none of these. dash -n does not reject all of them, because some parse as commands.
BASH_ONLY = [(r"\[\[", '[['), (r"^\s*local\s", 'local'), (r"^\s*declare\s", 'declare'), (r"<\(", '<( )'), (r"&>", '&>'), (r"\$RANDOM", '$RANDOM'), (r"^\s*source\s", 'source')]


def code_lines(text: str):
    """The lines of a script without comments and without here-documents, whose text goes elsewhere."""
    ending = None
    for number, line in enumerate(text.splitlines(), 1):
        if ending is not None:
            if line.strip() == ending:
                ending = None
            continue
        heredoc = re.search(r"<<-?\s*['\"]?(\w+)['\"]?", line)
        if heredoc:
            ending = heredoc.group(1)
        if not line.lstrip().startswith('#'):
            yield number, line


def findings(text: str, rules=ONE_SYSTEM_ONLY) -> list[str]:
    out = []
    for number, line in code_lines(text):
        for pattern, why in rules:
            if re.search(pattern, line):
                out.append(f'line {number}: {why}: {line.strip()[:80]}')
    return out


def test_the_list_of_launchers_is_not_empty():
    assert len(LAUNCHERS) >= 9  # So that an empty list cannot pass the tests below.


@pytest.mark.parametrize('path', LAUNCHERS, ids=IDS)
def test_it_parses_in_the_shell_that_it_names(path):
    shebang = path.read_text().splitlines()[0]
    assert shebang in ('#!/bin/sh', '#!/bin/bash'), f'{path.name}: {shebang}'
    shells = ['sh', 'bash'] if shebang == '#!/bin/sh' else ['bash']
    if shebang == '#!/bin/sh' and shutil.which('dash'):
        shells.append('dash')
    for shell in shells:
        done = subprocess.run([shell, '-n', str(path)], capture_output=True, text=True, timeout=30, check=False)  # -n reads and runs nothing.
        assert done.returncode == 0, f'{shell}: {done.stderr}'


@pytest.mark.parametrize('path', LAUNCHERS, ids=IDS)
def test_it_uses_no_flag_of_only_one_system(path):
    assert not findings(path.read_text())


@pytest.mark.parametrize('path', [p for p in LAUNCHERS if p.read_text().startswith('#!/bin/sh')], ids=lambda p: p.name)
def test_a_sh_launcher_uses_no_bash_syntax(path):
    assert not findings(path.read_text(), BASH_ONLY)


@pytest.mark.parametrize('line', ["sed -i '' s/a/b/ f", 'sed -i s/a/b/ f', 'df -g ~', 'stat -f %z f', 'date -j', 'readlink -f x', 'grep -P x f', 'sha256sum f'])
def test_the_rule_catches_each_flag(line):
    assert findings(f'#!/bin/sh\n{line}\n')


def test_the_rule_reads_only_code():
    assert not findings('#!/bin/sh\n# sed -i is not used here\ncat <<EOF\nsed -i text for a reader\nEOF\nsed s/a/b/ f > g\n')


def test_a_broken_script_fails_the_parse(tmp_path):
    bad = tmp_path / 'bad.sh'
    bad.write_text('#!/bin/sh\nif true; then\n')
    assert subprocess.run(['sh', '-n', str(bad)], capture_output=True, timeout=30, check=False).returncode != 0


@pytest.mark.parametrize('path', LAUNCHERS, ids=IDS)
def test_shellcheck_finds_no_error(path):
    if shutil.which('shellcheck') is None:
        if os.environ.get('CUMLX_REQUIRE_SHELLCHECK'):
            pytest.fail('CUMLX_REQUIRE_SHELLCHECK is set, and shellcheck is not on PATH')
        pytest.skip('shellcheck is not on PATH: not checked')  # CI sets the variable, so a skip there is a failure.
    done = subprocess.run(['shellcheck', '--severity=error', str(path)], capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0, done.stdout


@pytest.mark.parametrize('script, args', [('spikes/run_all.sh', ['{work}']), ('spikes/m3_kernels.sh', [])])
def test_a_missing_uv_stops_with_the_install_hint_and_starts_nothing(tmp_path, script, args):
    bare = '/usr/bin:/bin'
    assert shutil.which('uv', path=bare) is None, 'this machine has uv in the system folders, so the case cannot be set up'
    home = tmp_path / 'home'  # m3_kernels.sh adds folders under HOME and /opt/homebrew to PATH, so name a uv that is not there.
    home.mkdir()
    env = {'PATH': bare, 'HOME': str(home), 'CUMLX_M3_ROOT': str(tmp_path / 'root'), 'CUMLX_M3_UV': str(tmp_path / 'no-uv')}
    if script.endswith('run_all.sh'):
        env.pop('CUMLX_M3_UV')
    argv = [a.format(work=tmp_path / 'work') for a in args]
    done = subprocess.run(['bash', str(REPO / script), *argv], env=env, capture_output=True, text=True, timeout=30, check=False)
    assert done.returncode == 3, done.stderr
    assert 'uv is not on PATH' in done.stderr and 'astral.sh' in done.stderr
    assert not (tmp_path / 'root/run.pid').exists()
    assert not (tmp_path / 'work').exists()
