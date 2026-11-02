"""The Isaac Lab driver's pure parts, and the real checkpoint in validation/isaac-lab-mac."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / 'validation' / 'isaac-lab-mac'


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _driver():
    return _load(ROOT / 'spikes' / 'isaac_lab_mac.py', 'isaac_lab_mac')


def test_plan_removes_packages_without_a_mac_wheel_and_adds_a_usd_provider():
    driver = _driver()
    pyproject = {
        'project': {'dependencies': ['isaaclab', 'torch==2.11.0', 'usd-exchange==3.0.1', 'omniverseclient==2.74.0', 'Jinja2']},
        'tool': {'uv': {'override-dependencies': ['numpy>=2', 'usd-optimize~=1.3.1', 'usd-exchange==3.0.1']}},
    }
    requirements, overrides = driver.plan_requirements(pyproject)
    assert requirements == ['torch==2.11.0', 'Jinja2', driver.USD_PROVIDER]
    assert overrides == ['numpy>=2']


def test_training_log_is_parsed_and_missing_numbers_stay_missing():
    driver = _driver()
    log = ('\x1b[1m Learning iteration 0/5 \x1b[0m\n  Mean reward: -3.25\n Iteration time: 2.50s\n'
           ' Learning iteration 4/5 \n  Mean reward: 7.00\n Iteration time: 2.00s\n')
    parsed = driver.parse_training(log)
    assert (parsed['reward_first'], parsed['reward_last'], parsed['last_iteration']) == (-3.25, 7.0, 4)
    empty = driver.parse_training("ModuleNotFoundError: No module named 'omni'\n")
    assert empty['reward_last'] is None and empty['iterations_seen'] == 0


def test_recorded_run_shows_learning_and_its_controls_fail():
    record = json.loads((ROOT / 'validation' / 'isaac-lab-mac.json').read_text())
    assert record['supported_by_nvidia'] is False
    runs = record['closed_loop_mlx_controls_newton']['runs']
    trained = [r for r in runs if r['checkpoint'] == 'model_99.pt']
    controls = [r for r in runs if r['checkpoint'] != 'model_99.pt']
    assert len(trained) == 3 and all(r['upright_fraction_last_100'] == 1.0 for r in trained)
    assert len(controls) == 3 and all(r['upright_fraction_last_100'] < 0.1 for r in controls)


@pytest.mark.e2e
def test_real_checkpoint_matches_rsl_rl_and_wrong_policies_fail(e2e):
    import mlx.nn as nn
    import torch

    from cumlx import check_op

    policy = _load(ROOT / 'examples' / 'robot-policy' / 'verify_policy.py', 'verify_policy_real')
    state = policy.actor_state(torch.load(RUN / 'model_99.pt', map_location='cpu', weights_only=True))
    golden = dict(np.load(RUN / 'golden.npz'))
    layers = policy.actor_layers(state)
    inputs = [golden['obs_1'], golden['obs_16'], golden['obs_4096']]
    reference = policy.golden_policy(golden)

    good = check_op(reference, policy.mlx_policy(layers), inputs)
    assert good['verdict'] == 'ok' and good['worst_rel_l2'] < 1e-4

    assert check_op(reference, policy.mlx_policy(layers, activation=nn.relu), inputs)['verdict'] == 'diverged'
    tampered = [(w.clone(), b.clone()) for w, b in layers]
    tampered[1][0][3, 7] *= 1.01
    assert check_op(reference, policy.mlx_policy(tampered), inputs)['verdict'] == 'diverged'
    with pytest.raises(ValueError):
        check_op(reference, policy.mlx_policy(layers), [inputs[1] + 1])


@pytest.mark.e2e
def test_policy_keys_of_old_and_new_rsl_rl_are_read_and_a_normalizer_is_refused(e2e):
    import torch

    policy = _load(ROOT / 'examples' / 'robot-policy' / 'verify_policy.py', 'verify_policy_keys')
    weight, bias = torch.ones(2, 2), torch.zeros(2)
    new = {'mlp.0.weight': weight, 'mlp.0.bias': bias, 'distribution.std_param': torch.ones(1)}
    old = {'actor.0.weight': weight, 'actor.0.bias': bias, 'critic.0.weight': weight, 'critic.0.bias': bias}
    assert len(policy.actor_layers(new)) == 1 and len(policy.actor_layers(old)) == 1
    assert policy.actor_state({'actor_state_dict': new, 'critic_state_dict': {}}) is new
    assert policy.actor_state({'model_state_dict': old}) is old
    with pytest.raises(SystemExit):
        policy.actor_layers({**new, 'obs_normalizer._mean': torch.zeros(2)})
