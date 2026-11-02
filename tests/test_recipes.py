"""The compatibility catalog, `cumlx recipe`, and the scan-to-recipe link."""
import copy
import json
from pathlib import Path

import pytest

from cumlx.cli.main import main
from cumlx.verify.portability import RULES, scan
from cumlx.verify.recipes import declared_dependencies, find, load_catalog, lookup, validate_catalog

REPO = Path(__file__).resolve().parents[1]


def run_cli(capsys, *argv):
    code = main([*argv, '--format', 'json'])
    doc = json.loads(capsys.readouterr().out)
    assert doc['exit_code'] == code
    return doc


def write(root, files):
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def test_catalog_is_valid_and_large_enough():
    catalog = load_catalog()
    assert validate_catalog(catalog, REPO) == []
    assert len(catalog['recipes']) >= 20


def test_every_scan_rule_has_a_recipe():
    covered = {rule for recipe in load_catalog()['recipes'] for rule in recipe['scan_rules']}
    assert {rule.id for rule in RULES} <= covered
    assert 'cuda-source-files' in covered


@pytest.mark.parametrize('breakage,problem', [
    (lambda o: o.update(evidence='probably'), 'bad evidence'),
    (lambda o: o.update(evidence='run', grade=None, source='validation/REPORT.md'), 'needs a measured grade'),
    (lambda o: o.update(evidence='run', grade='A', source='https://example.com'), 'must name a file in validation/'),
    (lambda o: o.update(evidence='run', grade='A', source='validation/no-such-file.json'), 'does not exist'),
    (lambda o: o.update(evidence='docs', grade='A', source='https://example.com'), 'cannot have a measured grade'),
    (lambda o: o.update(evidence='docs', grade=None, source='validation/REPORT.md'), 'needs a URL'),
])
def test_validator_catches_a_false_evidence_claim(breakage, problem):
    catalog = copy.deepcopy(load_catalog())
    breakage(catalog['recipes'][0]['options'][0])
    assert any(problem in p for p in validate_catalog(catalog, REPO))


def test_validator_catches_a_name_claimed_twice():
    catalog = copy.deepcopy(load_catalog())
    catalog['recipes'][1]['aliases'].append(catalog['recipes'][0]['id'])
    assert any('claimed twice' in p for p in validate_catalog(catalog, REPO))


@pytest.mark.parametrize('name,recipe', [('bitsandbytes', 'bitsandbytes'), ('BNB', 'bitsandbytes'), ('flash_attn', 'flash-attn'),
                                         ('flash-attn', 'flash-attn'), ('isaacsim', 'isaac-sim'), ('warp-lang', 'warp')])
def test_names_and_spellings_resolve(name, recipe):
    assert find(name)['id'] == recipe


def test_recipe_by_name_separates_what_ran_from_what_was_read(capsys):
    doc = run_cli(capsys, 'recipe', 'bitsandbytes', '--max-items', '-1')
    assert doc['exit_code'] == 0
    options = doc['data']['recipes']['items'][0]['options']
    assert {o['evidence'] for o in options} == {'run', 'docs'}
    for option in options:
        assert (option['grade'] is not None) == (option['evidence'] == 'run')


def test_verified_only_drops_what_was_not_run(capsys):
    doc = run_cli(capsys, 'recipe', 'bitsandbytes', '--verified-only')
    assert {o['evidence'] for o in doc['data']['recipes']['items'][0]['options']} == {'run'}
    assert run_cli(capsys, 'recipe', 'xformers', '--verified-only')['data']['recipes']['total'] == 0


def test_documents_only_recipe_warns(capsys):
    doc = run_cli(capsys, 'recipe', 'xformers')
    assert 'W_RECIPE_NOT_RUN' in {w['code'] for w in doc['warnings']}
    assert doc['data']['recipes']['items'][0]['best_evidence'] == 'docs'


def test_isaac_sim_is_reported_as_having_no_substitute(capsys):
    doc = run_cli(capsys, 'recipe', 'isaacsim')
    assert doc['data']['recipes']['items'][0]['outcome'] == 'none'
    assert doc['data']['counts']['no_substitute'] == 1
    assert 'W_NO_SUBSTITUTE' in {w['code'] for w in doc['warnings']}


def test_unknown_name_is_an_error_and_never_an_empty_success(capsys):
    doc = run_cli(capsys, 'recipe', 'not-a-real-package')
    assert doc['exit_code'] == 4
    assert doc['errors'][0]['code'] == 'E_INPUT_NOT_FOUND'


def test_catalog_listing_is_capped(capsys):
    doc = run_cli(capsys, 'recipe', '--max-items', '2')
    assert doc['truncated'] is True
    assert doc['data']['recipes']['total'] == len(load_catalog()['recipes'])


def test_directory_gets_a_recipe_for_every_finding(tmp_path, capsys):
    write(tmp_path, {'a.py': 'import warp\nimport cudf\nx.cuda()\n', 'k.cu': '__global__ void f() {}\n', 'requirements.txt': 'onnxruntime-gpu\n'})
    doc = run_cli(capsys, 'recipe', str(tmp_path), '--max-items', '-1')
    assert doc['data']['kind'] == 'scan'
    assert doc['data']['uncovered'] == []
    names = {r['id'] for r in doc['data']['recipes']['items']}
    assert {'warp', 'rapids', 'torch-cuda-device', 'cuda-kernels', 'onnxruntime-gpu'} <= names
    assert 'cupy' not in names  # the matched text narrows a rule that covers several libraries
    assert 'tensorflow-gpu' not in names


def test_scan_report_file_is_accepted(tmp_path, capsys):
    write(tmp_path, {'repo/a.py': 'import warp\n'})
    report = tmp_path / 'scan.json'
    assert run_cli(capsys, 'scan', str(tmp_path / 'repo'), '-o', str(report))['exit_code'] == 0
    doc = run_cli(capsys, 'recipe', str(report))
    assert doc['data']['kind'] == 'scan_report'
    assert [r['id'] for r in doc['data']['recipes']['items']] == ['warp']


def test_other_json_file_is_refused(tmp_path, capsys):
    other = tmp_path / 'other.json'
    other.write_text('{"schema": "other"}')
    doc = run_cli(capsys, 'recipe', str(other))
    assert doc['exit_code'] == 4
    assert doc['errors'][0]['code'] == 'E_INPUT_INVALID'


def test_output_is_not_overwritten(tmp_path, capsys):
    out = tmp_path / 'r.json'
    out.write_text('keep')
    doc = run_cli(capsys, 'recipe', 'warp', '-o', str(out))
    assert doc['errors'][0]['code'] == 'E_OUTPUT_EXISTS'
    assert out.read_text() == 'keep'
    assert run_cli(capsys, 'recipe', 'warp', '-o', str(out), '--overwrite')['exit_code'] == 0
    assert json.loads(out.read_text())['recipes'][0]['id'] == 'warp'


def test_declared_dependencies_read_each_format(tmp_path):
    write(tmp_path, {
        'requirements.txt': '# comment\ntorch>=2\n-r other.txt\n--extra-index-url https://x\nWarp-Lang==1.0  # note\n',
        'requirements-dev.txt': 'pytest\n',
        'environment.yml': 'name: env\nchannels:\n  - conda-forge\ndependencies:\n  - numpy=2\n',
        'pyproject.toml': ('[project]\nname = "x"\nversion = "1"\ndescription = "not-a-dep"\n'
                           'dependencies = [\n  "gsplat[extra]>=1",\n  "triton",\n]\n'
                           'keywords = ["not-a-dep-either"]\n'
                           '[project.optional-dependencies]\ndev = ["mlx[cpu]; sys_platform == \'linux\'"]\n'
                           '[tool.other]\nitems = ["nope"]\n'),
    })
    names = declared_dependencies(tmp_path)
    assert {'torch', 'warp-lang', 'pytest', 'gsplat', 'triton', 'mlx', 'numpy'} <= set(names)
    assert not {'not-a-dep', 'not-a-dep-either', 'nope', 'other.txt', 'x'} & set(names)


def test_dependency_with_no_text_rule_still_gets_a_finding(tmp_path):
    write(tmp_path, {'requirements.txt': 'warp-lang\nnewton\n', 'a.py': 'x = 1\n'})
    report = scan(str(tmp_path))
    ids = {f['id'] for f in report['findings']}
    assert {'dep-warp', 'dep-newton'} <= ids
    assert report['tier'] == 'kernel'
    assert report['uncovered'] == []


def test_a_package_that_runs_on_a_mac_does_not_raise_the_tier(tmp_path):
    write(tmp_path, {'requirements.txt': 'usd-core\n', 'a.py': 'import warp\n'})
    report = scan(str(tmp_path))
    assert report['tier'] == 'kernel'
    assert [f['id'] for f in report['findings']] == ['warp-cuda']


def test_dependency_is_not_counted_twice(tmp_path):
    write(tmp_path, {'requirements.txt': 'bitsandbytes\n', 'a.py': 'import bitsandbytes\n'})
    assert [f['id'] for f in scan(str(tmp_path))['findings']] == ['bitsandbytes']


def test_recipe_lookup_does_not_execute_the_target(tmp_path):
    marker = tmp_path / 'ran'
    write(tmp_path, {'setup.py': f'open({str(marker)!r}, "w")\nfrom torch.utils.cpp_extension import CUDAExtension\n'})
    assert lookup(str(tmp_path))['tier'] == 'kernel'
    assert not marker.exists()


def test_port_names_a_converter_for_a_diffusers_pipeline(tmp_path, capsys):
    pytest.importorskip('mlx_lm')
    write(tmp_path, {'model_index.json': '{"_class_name": "FluxPipeline"}'})
    doc = run_cli(capsys, 'port', str(tmp_path), '-o', str(tmp_path / 'out'))
    assert doc['errors'][0]['code'] == 'E_MODEL_LOAD'
    assert 'cumlx recipe diffusers-model' in doc['errors'][0]['hint']


def test_port_names_a_converter_for_a_vision_model(tmp_path, capsys):
    pytest.importorskip('mlx_lm')
    write(tmp_path, {'config.json': '{"model_type": "no_such_vlm", "vision_config": {}}'})
    doc = run_cli(capsys, 'port', str(tmp_path), '-o', str(tmp_path / 'out'))
    assert 'cumlx recipe vision-model' in doc['errors'][0]['hint']
