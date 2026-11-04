"""Keep private identifiers out of every tracked file. The list holds hashes, so this file does not name them."""
from __future__ import annotations

import hashlib
import re
import subprocess
import unicodedata
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PRIVATE_TOKEN_SHA256 = frozenset({
    'c127d810a1477ddaa9d1ef1e0ee5ca8350c59a752c65dd02fd5bf0bd21fa6fec',
    'e54f4c3ee4809b158c86cbb62d50160cdcfc23ba36921721ee9574a9debf09f9',
    '5678d567fd6c3188ddd2f2974c2bd05006bbf4891b10079766fa6e56e47f4351',
    'a4f05557567341c6038cc9048cbd45e6b32c970f05f549bac715f593708e3e95',
    '20c6732dcd871d46c8e7069cfafc5ca8695b3da61c503e3e3bb15692d1a756e4',
})


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


def _private_tokens(text: str, hashes: frozenset[str]) -> list[str]:
    normal = unicodedata.normalize('NFC', text).casefold()
    return sorted({token for token in re.findall(r'[^\W_]+', normal) if _digest(token) in hashes})


def test_a_listed_token_is_found_and_a_longer_word_is_not():
    hashes = frozenset({_digest('example')})
    assert _private_tokens('a path /Example/x-y', hashes) == ['example']
    assert _private_tokens('examples of a path', hashes) == []


def test_the_list_holds_only_hashes():
    assert len(PRIVATE_TOKEN_SHA256) == 5
    assert all(re.fullmatch(r'[0-9a-f]{64}', value) for value in PRIVATE_TOKEN_SHA256)


def _files_to_scan(repo: Path) -> list[str]:
    """Tracked files, and new files that git does not ignore: a check before a commit must read what the commit is about to add."""
    done = subprocess.run(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=repo, capture_output=True, check=True)
    return [name for name in done.stdout.decode().split('\0') if name]


def test_the_scan_reads_a_new_file_and_leaves_an_ignored_one(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / '.gitignore').write_text('ignored.txt\n')
    (tmp_path / 'new.txt').write_text('x')
    (tmp_path / 'ignored.txt').write_text('x')
    assert set(_files_to_scan(tmp_path)) == {'.gitignore', 'new.txt'}


def test_no_tracked_file_holds_a_private_token():
    if not (REPO / '.git').exists():
        pytest.skip('this copy has no git metadata, so it has no tracked files to read')
    found = {}
    for name in _files_to_scan(REPO):
        path = REPO / name
        if path.is_file():
            tokens = _private_tokens(path.read_bytes().decode('utf-8', 'ignore'), PRIVATE_TOKEN_SHA256)
            if tokens:
                found[name] = len(tokens)
    assert not found, f'files that hold a private token: {sorted(found)}'
