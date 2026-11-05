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
    (out / "f1_bad.json").write_text(json.dumps({"status": "diverged"}))
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


def test_exit_1_with_no_diff_result_is_not_a_rejected_port(tmp_path):
    good(tmp_path)
    (tmp_path / "f1_bad.json").unlink()
    result = j.f1_mac_port(tmp_path)
    assert result["state"] == "fail" and "diverged" in result["reasons"][0]


def test_exit_1_with_another_status_is_not_a_rejected_port(tmp_path):
    good(tmp_path)
    change(tmp_path, "f1_bad", status="error")
    assert j.f1_mac_port(tmp_path)["state"] == "fail"


def run_work_with_stubs(tmp_path: Path, fault_exit: int) -> tuple[Path, str]:
    """The script with its .venv and uv as stubs. It finds them by its working folder and by PATH, so no real program runs."""
    bin_dir = tmp_path / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    tools = tmp_path / "tools"
    tools.mkdir()
    calls = tmp_path / "calls"
    (bin_dir / "cumlx").write_text(f'#!/bin/sh\necho "cumlx $1" >>"{calls}"\n[ "$1" = diff ] && exit 1\nexit 0\n')
    (bin_dir / "python").write_text(f'#!/bin/sh\necho "python $1" >>"{calls}"\ncase "$1" in *m6_f1_fault.py) exit {fault_exit} ;; esac\nexit 0\n')
    (tools / "uv").write_text("#!/bin/sh\nexit 1\n")
    for stub in (bin_dir / "cumlx", bin_dir / "python", tools / "uv"):
        stub.chmod(0o755)
    out = tmp_path / "out"
    out.mkdir()
    subprocess.run(["/bin/sh", str(WORK), str(out)], cwd=tmp_path, env={"HOME": str(tmp_path / "home"), "PATH": f"{tools}:/usr/bin:/bin"}, check=True, capture_output=True)
    return out, calls.read_text()


def test_the_script_reaches_the_diff_when_the_wrong_port_exists(tmp_path):
    out, calls = run_work_with_stubs(tmp_path, 0)
    assert "cumlx port" in calls and "cumlx diff" in calls
    assert (out / "f1_bad.exit").read_text().strip() == "1"


def test_a_failed_fault_script_leaves_no_result_for_the_judge(tmp_path):
    out, calls = run_work_with_stubs(tmp_path, 1)
    assert "cumlx port" in calls and "cumlx diff" not in calls
    assert not (out / "f1_bad.exit").exists()
    assert j.f1_mac_port(out)["state"] != "pass"


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


def test_the_record_agrees_with_the_rules_and_ticks_nothing():
    record = json.loads((REPO / "validation" / "m6-f1-f2-run.json").read_text())
    assert record["ticks_m6_1"] is False
    assert record["verdict"] == "fail"
    assert {name: part["state"] for name, part in record["parts"].items()} == {
        "f1_original_program": "fail", "f1_mac_port": "pass", "f2_original_program": "fail", "f2_mac_training": "pass"}
    rule = j.LIMITS["f2"]["mac_training"]
    training = record["measured"]["f2_mac_training"]
    assert training["right"]["heldout_psnr_end"] >= rule["heldout_psnr_at_least_db"]
    assert training["frozen"]["heldout_psnr_end"] < rule["frozen_control_below_db"]
    assert training["right"]["heldout_psnr_end"] - training["wrong_cameras"]["heldout_psnr_end"] >= rule["wrong_cameras_at_least_db_below_right"]
    assert record["measured"]["f1_mac_port"]["wrong_port"]["status"] == "diverged"
    assert record["cumlx_commit"] != record["attempts"][0]["commit"] and record["attempts"][-1]["commit"] == record["cumlx_commit"]
