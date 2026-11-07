"""validation/upstream/config_scan.py flags a config that mlx-lm reads with a different RoPE, and only that."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e
pytest.importorskip('mlx_lm')

SCAN = Path(__file__).resolve().parents[1] / 'validation' / 'upstream' / 'config_scan.py'
LLAMA = {'model_type': 'llama', 'hidden_size': 32, 'num_hidden_layers': 1, 'intermediate_size': 64,
         'num_attention_heads': 2, 'num_key_value_heads': 2, 'rms_norm_eps': 1e-6, 'vocab_size': 64}


def scan():
    spec = importlib.util.spec_from_file_location('config_scan', SCAN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rope_theta_only_in_rope_parameters_is_flagged():
    row = scan().check('x/llama', {**LLAMA, 'rope_parameters': {'rope_theta': 5000000, 'rope_type': 'default'}})
    assert row['findings'] == ['rope_base']
    assert row['mlx_rope_theta'] == 10000


def test_top_level_rope_theta_is_not_flagged():
    row = scan().check('x/llama', {**LLAMA, 'rope_theta': 5000000,
                                   'rope_parameters': {'rope_theta': 5000000, 'rope_type': 'default'}})
    assert row['findings'] == []


def test_scaling_only_in_rope_parameters_is_flagged():
    rp = {'rope_theta': 500000.0, 'rope_type': 'llama3', 'factor': 32.0, 'low_freq_factor': 1.0,
          'high_freq_factor': 4.0, 'original_max_position_embeddings': 8192}
    row = scan().check('x/llama', {**LLAMA, 'rope_theta': 500000.0, 'rope_parameters': rp})
    assert row['findings'] == ['rope_scaling']


def test_a_model_that_reads_rope_parameters_itself_is_not_flagged():
    from mlx_lm.utils import _get_classes

    try:
        _get_classes({'model_type': 'ministral3'})
    except ValueError:
        pytest.skip('this mlx-lm has no ministral3')
    rp = {'rope_theta': 1000000.0, 'rope_type': 'yarn', 'factor': 16.0, 'original_max_position_embeddings': 16384,
          'beta_fast': 32.0, 'beta_slow': 1.0}
    row = scan().check('x/ministral3', {**LLAMA, 'model_type': 'ministral3', 'head_dim': 16, 'rope_parameters': rp})
    assert row['findings'] == [] and row.get('reads_rope_parameters')


def test_a_config_without_rope_parameters_is_skipped():
    assert scan().check('x/llama', {**LLAMA, 'rope_theta': 10000.0}) is None


def test_a_top_level_rope_theta_that_disagrees_with_rope_parameters_is_flagged():
    # transformers 5 takes rope_parameters; mlx-lm takes the top-level value, also with the change of #1964
    row = scan().check('x/llama', {**LLAMA, 'rope_theta': 1000000.0,
                                   'rope_parameters': {'rope_theta': 10000000, 'rope_type': 'default'}})
    assert row['reference'] == 'transformers'
    assert row['transformers_rope_parameters']['rope_theta'] == 10000000
    assert row['findings'] == ['rope_base'] and row['mlx_rope_theta'] == 1000000.0
    assert row['with_pr1964']['base_fixed'] is False
