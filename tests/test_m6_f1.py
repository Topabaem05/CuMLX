"""spikes/m6_f1_judge.py on made-up output folders, and the scripts around it. Nothing third-party runs."""

import json
from pathlib import Path
import re
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "spikes"))
import m6_f1_judge as j  # noqa: E402
import m6_f1_record as record  # noqa: E402

WORK = (REPO / "spikes" / "m6_f1_work.sh").read_text()
REMOTE = (REPO / "spikes" / "remote" / "m6_f1.sh").read_text()
LEAVES = [f"model.layers.{i}.{name}" for i in range(24) for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
LAYERS = [f"model.layers.{i}" for i in range(24)]
CALLS = [{"key": f"attn/{i}", "phase": "prefill" if i < 24 else "decode", "layer": i % 24, "causal": True, "softmax_scale": None} for i in range(96)]
SMALL = {"rel_l2": 1e-4, "max_abs": 1e-3, "equal_fraction": 0.99, "finite": True}


def golden() -> dict:
    return {"leaves": LEAVES, "layers": LAYERS, "attention": CALLS, "varlen_calls": 0, "input_ids": [1, 2, 3]}


def replay(**changes) -> dict:
    data = {"input_ids_equal": True,
            "leaves": {name: dict(SMALL) for name in LEAVES},
            "weights": {name: {"shape_equal": True, "cb_differ_fraction": 1e-5, "cb_max_step": 1, "scb_equal": True} for name in LEAVES},
            "layers": {name: dict(SMALL) for name in LAYERS + ["norm", "lm_head"]},
            "attention": {c["key"]: {"phase": c["phase"], "layer": c["layer"], "against_exact": dict(SMALL), "golden_against_exact": dict(SMALL), "against_golden": dict(SMALL)} for c in CALLS},
            "prefill_chain": {"rel_l2": 0.01}}
    data.update(changes)
    return data


def write(path: Path, data) -> None:
    path.write_text(json.dumps(data))


@pytest.fixture
def sides(tmp_path):
    cuda, mac = tmp_path / "cuda", tmp_path / "mac"
    (cuda / "golden").mkdir(parents=True)
    mac.mkdir()
    for name in ("install", "capture", "program"):
        (cuda / f"{name}.exit").write_text("0\n")
    for name in ("install", "clean", "replay", "program"):
        (mac / f"{name}.exit").write_text("0\n")
    write(cuda / "golden" / "summary.json", golden())
    tokens = list(range(100, 120))
    write(cuda / "program.json", {"token_ids": tokens, "text": " Paris", "device": "cuda:0"})
    write(mac / "program.json", {"token_ids": tokens[:5] + [7] * 15, "text": " Paris", "device": "mps:0"})
    calls = [{"kind": "substituted", "name": n, "to": m} for n, m in j.SUBSTITUTES.items()] + [{"kind": "distribution", "name": n, "to": f"module {v}"} for n, v in j.DISTRIBUTIONS.items()]
    write(mac / "program.run-report.json", {"calls": calls})
    write(mac / "replay.json", replay())
    broken = {
        "no_outlier_split": replay(leaves={**replay()["leaves"], LEAVES[3]: {**SMALL, "rel_l2": 0.3}}),
        "activation_scale_128": replay(leaves={**replay()["leaves"], LEAVES[0]: {**SMALL, "rel_l2": 0.0078}}),
        "bias_dropped": replay(leaves={**replay()["leaves"], LEAVES[1]: {**SMALL, "rel_l2": 0.05}}),
        "causal_top_left": replay(attention={**replay()["attention"], "attn/30": {**replay()["attention"]["attn/30"], "against_exact": {**SMALL, "rel_l2": 0.9}}}),
        "no_softmax_scale": replay(attention={**replay()["attention"], "attn/2": {**replay()["attention"]["attn/2"], "against_exact": {**SMALL, "rel_l2": 0.4}}}),
    }
    for fault, data in broken.items():
        (mac / f"fault_{fault}.exit").write_text("0\n")
        write(mac / f"fault_{fault}.json", data)
    return cuda, mac


def test_a_right_pair_passes(sides):
    result = j.judge(*sides)
    assert result["verdict"] == "pass", {k: v["reasons"] for k, v in result["parts"].items()}
    assert result["grade"] == "A" and result["parts"]["first_token"]["equal_tokens"] == 5


def edit(path: Path, change) -> None:
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


@pytest.mark.parametrize("breaking, broken", [
    (lambda c, m: edit(m / "replay.json", lambda d: d["leaves"][LEAVES[5]].update(rel_l2=0.0021)), "leaves"),
    (lambda c, m: edit(m / "replay.json", lambda d: d["leaves"].pop(LEAVES[9])), "leaves"),
    (lambda c, m: edit(m / "replay.json", lambda d: d["leaves"][LEAVES[5]].update(rel_l2=float("nan"))), "leaves"),
    (lambda c, m: edit(m / "replay.json", lambda d: d["weights"][LEAVES[2]].update(scb_equal=False)), "weights"),
    (lambda c, m: edit(m / "replay.json", lambda d: d["weights"][LEAVES[2]].update(cb_max_step=2)), "weights"),
    (lambda c, m: edit(m / "replay.json", lambda d: d["weights"][LEAVES[2]].update(cb_differ_fraction=0.0011)), "weights"),
    (lambda c, m: edit(m / "replay.json", lambda d: d["layers"]["model.layers.7"].update(rel_l2=0.016)), "layers"),
    (lambda c, m: edit(m / "replay.json", lambda d: d["layers"].pop("lm_head")), "layers"),
    (lambda c, m: edit(m / "replay.json", lambda d: d["attention"]["attn/50"]["against_exact"].update(rel_l2=0.02)), "attention"),
    (lambda c, m: edit(m / "replay.json", lambda d: d["attention"]["attn/3"]["golden_against_exact"].update(rel_l2=0.02)), "attention"),
    (lambda c, m: edit(c / "golden" / "summary.json", lambda d: d.update(varlen_calls=2)), "attention"),
    (lambda c, m: write(m / "fault_bias_dropped.json", replay()), "controls"),
    (lambda c, m: (m / "fault_causal_top_left.exit").write_text("1\n"), "controls"),
    (lambda c, m: edit(m / "program.json", lambda d: d.update(token_ids=[1] * 20)), "first_token"),
    (lambda c, m: edit(m / "program.json", lambda d: d.update(token_ids=[100] * 19)), "program_mac"),
    (lambda c, m: edit(m / "program.run-report.json", lambda d: d.update(calls=d["calls"][1:])), "program_mac"),
    (lambda c, m: edit(m / "program.run-report.json", lambda d: d.update(calls=[x for x in d["calls"] if x["kind"] != "distribution"])), "program_mac"),
    (lambda c, m: (c / "program.exit").write_text("1\n"), "program_cuda"),
    (lambda c, m: (m / "clean.exit").write_text("1\n"), "setup"),
    (lambda c, m: edit(m / "replay.json", lambda d: d.update(input_ids_equal=False)), "setup"),
])
def test_each_break_fails_its_part(sides, breaking, broken):
    breaking(*sides)
    result = j.judge(*sides)
    assert result["verdict"] == "fail"
    assert [name for name, item in result["parts"].items() if item["state"] != "pass"] == [broken], {k: v["reasons"] for k, v in result["parts"].items()}


def test_no_golden_is_not_a_pass(sides):
    (sides[0] / "golden" / "summary.json").unlink()
    result = j.judge(*sides)
    assert result["verdict"] != "pass" and result["parts"]["leaves"]["state"] == "not_run"


def test_the_scripts_run_the_controls_and_substitutes_that_the_rules_name():
    loop = re.search(r"for fault in (.*?); do", WORK).group(1).split()
    assert sorted(loop) == sorted(j.FAULTS) == sorted(record.FAULTS)
    for name, module in j.SUBSTITUTES.items():
        assert f"--substitute {name}={module}" in WORK
    assert "--substitute flash_attn=cumlx.compat.flash_attn_sdpa" in REMOTE and "--substitute bitsandbytes" not in REMOTE
    assert "--target cuda" in REMOTE and "bitsandbytes==0.50.2" in REMOTE and "transformers==5.22.1" in REMOTE and "accelerate==1.15.0" in REMOTE
    assert "transformers==5.22.1 accelerate==1.15.0" in WORK
    limits = json.loads((REPO / "validation" / "m6-f1-limits.json").read_text())
    assert limits["program"].startswith("spikes/m6_f1_program.py") and record.MODEL in limits["program"] and record.REVISION in limits["program"]


def test_the_program_and_the_record_load_the_model_alike():
    program = (REPO / "spikes" / "m6_f1_program.py").read_text()
    source = (REPO / "spikes" / "m6_f1_record.py").read_text()
    for text in ("attn_implementation='flash_attention_2'", "BitsAndBytesConfig(load_in_8bit=True)", "device_map={'': 0}"):
        assert text in program and text in source
    assert record.PROMPT in program


def test_the_capture_hooks_save_and_leave_each_output_as_it_is():
    torch = pytest.importorskip("torch")
    saved = {}
    layer = torch.nn.Linear(4, 3)
    x = torch.randn(2, 4)
    want = layer(x)
    layer.register_forward_hook(record.recording_hook(lambda key, value: saved.__setitem__(key, value), "in", "out"))
    got = layer(x)
    assert torch.equal(got, want)
    assert torch.equal(saved["in"], x) and torch.equal(saved["out"], want)


def test_no_forward_hook_in_the_recorder_is_a_lambda():
    source = (REPO / "spikes" / "m6_f1_record.py").read_text()
    assert "register_forward_hook(lambda" not in source


def test_the_rerun_rules_name_a_new_prompt_and_the_scripts_pass_it():
    rerun = json.loads((REPO / "validation" / "m6-f1-rerun-limits.json").read_text())
    prompt = rerun["prompt"]
    alternate = (REPO / "spikes" / "remote" / "m6_f1alt.sh").read_text()
    assert prompt and prompt != record.PROMPT and prompt not in alternate
    assert "validation/m6-f1-limits.json" in rerun["grade"] and "first complete run" in rerun["grade"]
    script = (REPO / "spikes" / "remote" / "m6_f1rerun.sh").read_text()
    assert "validation/m6-f1-rerun-limits.json" in script and "m6_f1.sh" in script and "CUMLX_F1_PROMPT" in script
    assert 'PROMPT_ARGS=(--prompt "$CUMLX_F1_PROMPT")' in REMOTE and '"${PROMPT_ARGS[@]}"' in REMOTE
    assert '--prompt "$CUMLX_F1_PROMPT"' in WORK
    # The reported fault runs on the Mac, and the judge never counts it as a control.
    assert "--fault ieee_row_scale" in WORK and "ieee_row_scale" in record.REPORTED_FAULTS and "ieee_row_scale" not in j.FAULTS


def test_the_reported_fault_divides_the_row_scale_on_the_device():
    torch = pytest.importorskip("torch")
    from cumlx.compat.bitsandbytes_int8 import functional
    real = functional.cuda_row_scale
    try:
        record.apply_fault("ieee_row_scale")
        absmax = torch.tensor([3.0, 0.5, 7.0], dtype=torch.float32)
        assert torch.equal(functional.cuda_row_scale(absmax), 127.0 / absmax)
    finally:
        functional.cuda_row_scale = real


@pytest.fixture
def rerun_sides(sides, tmp_path):
    """The made-up pair laid out as the rerun's folders: the fetched CUDA folder and the Mac result folder."""
    import m6_f1_rerun_record as rerun
    rules = json.loads(rerun.RULES.read_text())
    cuda_f1, mac_f1 = sides
    cuda, mac = tmp_path / "fetched", tmp_path / "result"
    cuda.mkdir()
    mac.mkdir()
    cuda_f1.rename(cuda / "f1rerun")
    mac_f1.rename(mac / "f1")
    for top in (cuda, mac):
        (top / "cumlx_commit.txt").write_text("abc123\n")
    (cuda / "done.txt").write_text("now\n")
    write(cuda / "f1rerun" / "golden" / "summary.json", {**golden(), "prompt": rules["prompt"]})
    (mac / "f1" / "mac-golden").mkdir()
    write(mac / "f1" / "mac-golden" / "summary.json", {"prompt": rules["prompt"]})
    return rerun, rules, cuda, mac


def test_a_rerun_with_every_condition_gets_its_grade(rerun_sides):
    rerun, rules, cuda, mac = rerun_sides
    record = rerun.assess(cuda, mac, rules, True)
    assert record["complete"] and record["verdict"] == "pass" and record["grade"] == "A", record["not_complete_because"]


@pytest.mark.parametrize("breaking", ["commit", "rules", "prompt", "mac_prompt", "busy", "unfinished"])
def test_a_rerun_without_a_condition_is_not_complete_and_gets_no_grade(rerun_sides, breaking):
    rerun, rules, cuda, mac = rerun_sides
    in_commit = breaking != "rules"
    if breaking == "commit":
        (mac / "cumlx_commit.txt").write_text("def456\n")
    elif breaking == "prompt":
        write(cuda / "f1rerun" / "golden" / "summary.json", {**golden(), "prompt": record.PROMPT})
    elif breaking == "mac_prompt":
        write(mac / "f1" / "mac-golden" / "summary.json", {"prompt": record.PROMPT})
    elif breaking == "busy":
        (cuda / "busy_at_start.txt").write_text("123\n")
    elif breaking == "unfinished":
        (cuda / "done.txt").unlink()
    result = rerun.assess(cuda, mac, rules, in_commit)
    assert not result["complete"] and result["grade"] is None and result["not_complete_because"]


def test_a_complete_rerun_that_fails_stays_a_fail(rerun_sides):
    rerun, rules, cuda, mac = rerun_sides
    write(mac / "f1" / "replay.json", replay(leaves={**replay()["leaves"], LEAVES[5]: {**SMALL, "rel_l2": 0.003}}))
    result = rerun.assess(cuda, mac, rules, True)
    assert result["complete"] and result["verdict"] == "fail" and result["grade"] is None
