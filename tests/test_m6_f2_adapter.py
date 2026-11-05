"""spikes/m6_f2_adapter_judge.py on made-up output folders, the hand render on values with a closed form, and the flags of the work script. Nothing third-party runs."""

import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "spikes"))
import m6_f2_adapter_judge as j  # noqa: E402
from m6_f2_hand_render import look_at, quat_wxyz_to_matrix, render  # noqa: E402

WORK = REPO / "spikes" / "m6_f2_adapter_work.sh"
K = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]])
ONE = dict(quats=[[1.0, 0.0, 0.0, 0.0]], colors=[[1.0, 0.5, 0.25]], viewmat=np.eye(4), K=K, width=64, height=64)


def good(out: Path) -> Path:
    for name in ("install", "conformance", "program"):
        (out / f"{name}.exit").write_text("0\n")
    scene = {"image_rel_l2": 1e-5, "alpha_rel_l2": 1e-5, "peak_error_px": 0.0, "device": "mps", "within": True}
    (out / "conformance.json").write_text(json.dumps({
        "conformance": {"a": scene, "b": scene},
        "controls": {name: {"caught": True} for name in j.LIMITS["controls"]["must_fail"]},
        "gradients": {f"color_loss.{name}": {"relative_error": 0.01} for name in j.GRADIENT_NAMES},
        "refusals": {name: "raised: no" for name in j.LIMITS["refusals"]["must_raise"]},
    }))
    (out / "program.json").write_text(json.dumps({"first_loss": 0.5, "last_loss": 0.1}))
    (out / "train_right.json").write_text(json.dumps({"device": "mps", "heldout_psnr_start": 12.0, "heldout_psnr_end": 30.0}))
    (out / "train_frozen.json").write_text(json.dumps({"device": "mps", "heldout_psnr_start": 12.0, "heldout_psnr_end": 12.0}))
    (out / "train_wrong_cameras.json").write_text(json.dumps({"device": "mps", "heldout_psnr_start": 12.0, "heldout_psnr_end": 18.0}))
    return out


def edit(out: Path, name: str, change) -> None:
    path = out / f"{name}.json"
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def test_a_right_run_passes(tmp_path):
    assert j.judge(good(tmp_path))["verdict"] == "pass"


@pytest.mark.parametrize("breaking, broken", [
    (lambda o: edit(o, "conformance", lambda d: d["conformance"]["a"].update(image_rel_l2=0.02)), "conformance"),
    (lambda o: edit(o, "conformance", lambda d: d["conformance"]["a"].update(alpha_rel_l2=0.02)), "conformance"),
    (lambda o: edit(o, "conformance", lambda d: d["conformance"]["b"].update(peak_error_px=1.0)), "conformance"),
    (lambda o: edit(o, "conformance", lambda d: d["conformance"]["b"].update(device="cpu")), "conformance"),
    (lambda o: edit(o, "conformance", lambda d: d["controls"]["half_pixel_shift"].update(caught=False)), "controls"),
    (lambda o: edit(o, "conformance", lambda d: d["controls"].pop("no_eps2d")), "controls"),
    (lambda o: edit(o, "conformance", lambda d: d["gradients"]["color_loss.means"].update(relative_error=0.06)), "gradients"),
    (lambda o: edit(o, "conformance", lambda d: d["gradients"].pop("color_loss.quats")), "gradients"),
    (lambda o: edit(o, "conformance", lambda d: d["refusals"].update(sh_degree="ran")), "refusals"),
    (lambda o: edit(o, "conformance", lambda d: d["refusals"].pop("covars")), "refusals"),
    (lambda o: (o / "program.exit").write_text("4"), "original_program"),
    (lambda o: edit(o, "program", lambda d: d.update(last_loss=0.9)), "original_program"),
    (lambda o: edit(o, "train_right", lambda d: d.update(heldout_psnr_end=24.0)), "training"),
    (lambda o: edit(o, "train_right", lambda d: d.update(heldout_psnr_start=28.0)), "training"),
    (lambda o: edit(o, "train_right", lambda d: d.update(device="cpu")), "training"),
    (lambda o: edit(o, "train_frozen", lambda d: d.update(heldout_psnr_end=26.0)), "training"),
    (lambda o: edit(o, "train_wrong_cameras", lambda d: d.update(heldout_psnr_end=28.0)), "training"),
])
def test_each_wrong_run_fails_in_its_part(tmp_path, breaking, broken):
    good(tmp_path)
    breaking(tmp_path)
    result = j.judge(tmp_path)
    assert result["verdict"] == "fail" and result["parts"][broken]["state"] == "fail"


@pytest.mark.parametrize("name, part", [("program.exit", "original_program"), ("train_right.json", "training"), ("conformance.json", "conformance"), ("conformance.json", "refusals")])
def test_a_part_with_no_output_is_not_a_pass(tmp_path, name, part):
    good(tmp_path)
    (tmp_path / name).unlink()
    result = j.judge(tmp_path)
    assert result["parts"][part]["state"] == "not_run" and result["verdict"] != "pass"


def test_no_output_at_all_is_not_a_pass(tmp_path):
    assert j.judge(tmp_path)["verdict"] == "not_run"


def test_a_refused_install_is_a_failure_and_not_a_skip(tmp_path):
    good(tmp_path)
    (tmp_path / "install.exit").write_text("1")
    assert j.judge(tmp_path)["verdict"] == "fail"


def test_the_hand_render_has_its_closed_form_on_the_axis():
    # The center projects to (32, 32), and sigma is 100 * 0.2 / 2 = 10 px with no perspective term on the axis. Pixel centers are at +0.5.
    image, alpha = render([[0.0, 0.0, 2.0]], scales=[[0.2, 0.2, 0.2]], opacities=[0.5], eps2d=0.0, **ONE)
    assert alpha[32, 32] == pytest.approx(0.5 * np.exp(-0.5 * (0.5 ** 2 + 0.5 ** 2) / 100.0), rel=1e-12)
    assert alpha[32, 42] == pytest.approx(0.5 * np.exp(-0.5 * (10.5 ** 2 + 0.5 ** 2) / 100.0), rel=1e-12)
    assert image[32, 32] == pytest.approx(alpha[32, 32] * np.array([1.0, 0.5, 0.25]))


def test_the_hand_render_adds_eps2d_to_the_variance():
    _, alpha = render([[0.0, 0.0, 2.0]], scales=[[0.2, 0.2, 0.2]], opacities=[0.5], eps2d=4.0, **ONE)
    assert alpha[32, 42] == pytest.approx(0.5 * np.exp(-0.5 * (10.5 ** 2 + 0.5 ** 2) / 104.0), rel=1e-12)


def test_the_hand_render_composites_front_to_back_and_on_the_background():
    fields = dict(ONE, quats=[[1.0, 0.0, 0.0, 0.0]] * 2, colors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    image, alpha = render([[0.0, 0.0, 3.0], [0.0, 0.0, 2.0]], scales=[[0.5, 0.5, 0.5]] * 2, opacities=[0.5, 0.5], background=[0.0, 0.0, 1.0], **fields)
    red, green, blue = image[32, 32]
    assert green > red and alpha[32, 32] == pytest.approx(0.75, rel=0.01)
    assert blue == pytest.approx(1.0 - alpha[32, 32])


def test_the_hand_render_leaves_out_what_the_planes_cut():
    means = [[0.0, 0.0, 3.0]]
    near_image, _ = render(means, scales=[[0.2, 0.2, 0.2]], opacities=[0.5], near_plane=4.0, **ONE)
    far_image, _ = render(means, scales=[[0.2, 0.2, 0.2]], opacities=[0.5], far_plane=2.0, **ONE)
    assert near_image.max() == 0.0 and far_image.max() == 0.0


def test_the_hand_render_skips_an_alpha_below_the_cutoff():
    _, alpha = render([[0.0, 0.0, 3.0]], scales=[[0.2, 0.2, 0.2]], opacities=[0.5], **ONE)
    assert alpha[alpha > 0.0].min() >= 1.0 / 255.0


def test_the_quaternion_is_wxyz():
    assert quat_wxyz_to_matrix(np.array([1.0, 0.0, 0.0, 0.0])) == pytest.approx(np.eye(3))
    half_turn_z = quat_wxyz_to_matrix(np.array([0.0, 0.0, 0.0, 1.0]))
    assert half_turn_z == pytest.approx(np.diag([-1.0, -1.0, 1.0]))


def test_look_at_points_its_z_axis_at_the_target():
    view = look_at([0.0, 0.0, -3.0], [0.0, 0.0, 0.0], [0.0, -1.0, 0.0])
    assert view[:3, :3] @ np.array([0.0, 0.0, 0.0]) + view[:3, 3] == pytest.approx(np.array([0.0, 0.0, 3.0]))
    assert np.linalg.det(view[:3, :3]) == pytest.approx(1.0)


def cli_flags(command: str) -> set[str]:
    text = subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, 'src'); sys.argv = ['cumlx', '{command}', '--help']; from cumlx.cli.main import main; main()"], cwd=REPO, capture_output=True, text=True, check=False).stdout
    assert text, command
    return set(re.findall(r"--[a-z][\w-]*", text))


def test_each_cumlx_line_uses_only_flags_that_exist():
    lines = [line for line in WORK.read_text().splitlines() if 'cumlx" run ' in line]
    assert len(lines) == 2
    for line in lines:
        assert set(re.findall(r"(?<=\s)--[a-z][\w-]*", line.split(" -- ")[0])) <= cli_flags("run"), line


def test_the_programs_hold_nothing_for_the_mac():
    for name in ("m6_f2_gsplat_program.py", "m6_f2_gsplat_train.py"):
        text = (REPO / "spikes" / name).read_text().split('"""', 2)[2]
        assert not re.search(r"\bmps\b|metal", text.lower()), name


def test_the_hand_render_has_no_metalsplat_code():
    text = (REPO / "spikes" / "m6_f2_hand_render.py").read_text().split('"""', 2)[2]
    assert "metalsplat" not in text and "torch" not in text


def test_the_rules_say_that_nothing_here_ticks_m6_1():
    assert "nothing here ticks M6-1" in j.LIMITS["scope"]
    assert j.LIMITS["metalsplat_commit"] in WORK.read_text()
    assert j.LIMITS["adapter"] in WORK.read_text()
