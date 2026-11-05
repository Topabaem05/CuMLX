"""spikes/m6_f1_f2_judge.py on made-up output folders, and the flags that spikes/m6_f1_f2_work.sh passes. Nothing third-party runs."""

import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "spikes"))
import m6_f1_f2_judge as j  # noqa: E402

WORK = REPO / "spikes" / "m6_f1_f2_work.sh"


def good(out: Path) -> Path:
    for name in ("f1_port", "f1_port8", "f1_program_install", "f1_program", "f2_install", "f2_program"):
        (out / f"{name}.exit").write_text("0\n")
    (out / "f1_bad.exit").write_text("1\n")
    (out / "f1_program.json").write_text(json.dumps({"generated_tokens": 20}))
    (out / "f2_program.json").write_text(json.dumps({"first_loss": 0.5, "last_loss": 0.1}))
    (out / "f2_right.json").write_text(json.dumps({"device": "mps", "heldout_psnr_start": 12.0, "heldout_psnr_end": 30.0}))
    (out / "f2_frozen.json").write_text(json.dumps({"device": "mps", "heldout_psnr_start": 12.0, "heldout_psnr_end": 12.0}))
    (out / "f2_wrong_cameras.json").write_text(json.dumps({"device": "mps", "heldout_psnr_start": 12.0, "heldout_psnr_end": 18.0}))
    return out


def change(out: Path, name: str, **fields) -> None:
    path = out / f"{name}.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **fields}))


def test_a_right_run_passes(tmp_path):
    assert j.judge(good(tmp_path))["verdict"] == "pass"


@pytest.mark.parametrize("breaking, broken", [
    (lambda o: (o / "f1_port.exit").write_text("5"), "f1_mac_port"),
    (lambda o: (o / "f1_port8.exit").write_text("1"), "f1_mac_port"),
    (lambda o: (o / "f1_bad.exit").write_text("0"), "f1_mac_port"),
    (lambda o: (o / "f1_program.exit").write_text("1"), "f1_original_program"),
    (lambda o: change(o, "f1_program", generated_tokens=3), "f1_original_program"),
    (lambda o: (o / "f1_program_install.exit").write_text("1"), "f1_original_program"),
    (lambda o: (o / "f2_program.exit").write_text("1"), "f2_original_program"),
    (lambda o: change(o, "f2_program", last_loss=0.9), "f2_original_program"),
    (lambda o: change(o, "f2_right", heldout_psnr_end=24.0), "f2_mac_training"),
    (lambda o: change(o, "f2_right", heldout_psnr_start=28.0), "f2_mac_training"),
    (lambda o: change(o, "f2_right", device="cpu"), "f2_mac_training"),
    (lambda o: change(o, "f2_frozen", heldout_psnr_end=26.0), "f2_mac_training"),
    (lambda o: change(o, "f2_wrong_cameras", heldout_psnr_end=28.0), "f2_mac_training"),
])
def test_each_wrong_run_fails_in_its_part(tmp_path, breaking, broken):
    good(tmp_path)
    breaking(tmp_path)
    result = j.judge(tmp_path)
    assert result["verdict"] == "fail"
    assert result["parts"][broken]["state"] == "fail"


@pytest.mark.parametrize("name, part", [
    ("f1_port.exit", "f1_mac_port"), ("f1_bad.exit", "f1_mac_port"), ("f1_program_install.exit", "f1_original_program"), ("f1_program.exit", "f1_original_program"),
    ("f2_install.exit", "f2_mac_training"), ("f2_right.json", "f2_mac_training"), ("f2_program.exit", "f2_original_program"),
])
def test_a_part_with_no_output_is_not_a_pass(tmp_path, name, part):
    good(tmp_path)
    (tmp_path / name).unlink()
    result = j.judge(tmp_path)
    assert result["parts"][part]["state"] == "not_run"
    assert result["verdict"] != "pass"


def test_a_refused_install_is_a_failure_and_not_a_skip(tmp_path):
    (tmp_path / "f1_program_install.exit").write_text("1")
    assert j.f1_original(tmp_path)["state"] == "fail"


def cli_flags(command: str) -> set[str]:
    text = subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, 'src'); sys.argv = ['cumlx', '{command}', '--help']; from cumlx.cli.main import main; main()"],
                          cwd=REPO, capture_output=True, text=True, check=False).stdout
    assert text, command
    return set(re.findall(r"--[a-z][\w-]*", text))


@pytest.mark.parametrize("command", ["port", "diff", "run"])
def test_each_cumlx_line_uses_only_flags_that_exist(command):
    lines = [line for line in WORK.read_text().splitlines() if re.search(rf"cumlx\"? {command} ", line)]
    assert lines
    for line in lines:
        ours = line.split(" -- ")[0]
        assert set(re.findall(r"(?<=\s)--[a-z][\w-]*", ours)) <= cli_flags(command), line


def test_the_flag_check_can_fail():
    assert "--q-bits" in cli_flags("port")
    assert "--controller" not in cli_flags("port")


def test_the_programs_hold_nothing_for_the_mac():
    for name in ("m6_f1_program.py", "m6_f2_gsplat_program.py"):
        text = (REPO / "spikes" / name).read_text().split('"""', 2)[2]
        assert "mps" not in text.lower() and "metal" not in text.lower(), name


def test_the_rules_say_that_nothing_here_ticks_m6_1():
    assert "nothing here ticks M6-1" in j.LIMITS["scope"]
    assert j.LIMITS["f1"]["revision"] in WORK.read_text()
    assert j.LIMITS["f2"]["metalsplat_commit"] in WORK.read_text()
