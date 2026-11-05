"""`cumlx port --revision`: the converter must get that revision, and each later step must compare against it. Nothing here downloads or converts."""

import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

from cumlx.cli import commands
from cumlx.errors import CumlxError

REVISION = "0123456789abcdef"
MODEL = {"model_type": "qwen2", "mlx_lm_supports": True, "needs_remote_code": False, "family": "causal_lm"}


def port_args(tmp_path, **changes):
    fields = {"model": "org/model", "revision": REVISION, "output": str(tmp_path / "out"), "overwrite": False, "prompt": [], "token_ids": [],
              "dtype": "bfloat16", "q_bits": 8, "dry_run": False, "device": "cpu", "verbose": False, "quiet": True}
    return SimpleNamespace(**{**fields, **changes})


@pytest.fixture
def recorded(monkeypatch, tmp_path):
    calls = SimpleNamespace(convert=None, steps=[], snapshot=[])
    monkeypatch.setattr("cumlx.verify.portability.scan", lambda model, revision=None: {"model": MODEL, "tier": "model"})

    def fake_run(command, **_):
        calls.convert = command
        return subprocess.CompletedProcess(command, 0, "", "")

    def fake_self(*argv):
        calls.steps.append(list(argv))
        return {"exit_code": 0, "status": "ok", "summary": "stub", "data": {"verdict": "ok"}}

    def fake_snapshot(model, revision=None, allow_patterns=None):
        calls.snapshot.append((model, revision, allow_patterns))
        return str(tmp_path / "snapshot")

    monkeypatch.setattr(commands, "subprocess", SimpleNamespace(run=fake_run))  # only in this module, because the test below runs the real converter
    monkeypatch.setattr(commands, "_self", fake_self)
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
    return calls


def test_the_converter_reads_the_pinned_snapshot(recorded, tmp_path):
    assert commands.run_port(port_args(tmp_path)).exit_code == 0
    assert recorded.snapshot[0][:2] == ("org/model", REVISION)
    assert recorded.convert[recorded.convert.index("--hf-path") + 1] == str(tmp_path / "snapshot")


def test_each_later_step_uses_the_same_revision(recorded, tmp_path):
    commands.run_port(port_args(tmp_path))
    by_name = {step[0]: step for step in recorded.steps}
    for name in ("weights", "capture"):
        assert by_name[name][by_name[name].index("--revision") + 1] == REVISION, name


def test_with_no_revision_nothing_is_downloaded_first(recorded, tmp_path):
    commands.run_port(port_args(tmp_path, revision=None))
    assert recorded.snapshot == []
    assert recorded.convert[recorded.convert.index("--hf-path") + 1] == "org/model"
    assert all("--revision" not in step for step in recorded.steps)


def test_a_local_folder_is_not_downloaded(recorded, tmp_path):
    folder = tmp_path / "local"
    folder.mkdir()
    commands.run_port(port_args(tmp_path, model=str(folder)))
    assert recorded.snapshot == []
    assert recorded.convert[recorded.convert.index("--hf-path") + 1] == str(folder)


def test_a_revision_that_cannot_be_fetched_is_an_error_and_writes_nothing(monkeypatch, recorded, tmp_path):
    def refuse(*_, **__):
        raise OSError("no such revision")

    monkeypatch.setattr("huggingface_hub.snapshot_download", refuse)
    with pytest.raises(CumlxError) as caught:
        commands.run_port(port_args(tmp_path))
    assert caught.value.code == "E_MODEL_LOAD" and REVISION in str(caught.value)
    assert recorded.convert is None and not (tmp_path / "out").exists()


def test_each_flag_exists_in_the_real_converter(recorded, tmp_path):
    pytest.importorskip("mlx_lm")
    commands.run_port(port_args(tmp_path))
    help_text = subprocess.check_output([sys.executable, "-m", "mlx_lm", "convert", "--help"], text=True)
    known = set(re.findall(r"--?[a-z][\w-]*", help_text))
    assert "--hf-path" in known
    ours = {item for item in recorded.convert if item.startswith("-") and item != "-m"}
    assert ours <= known, ours - known
