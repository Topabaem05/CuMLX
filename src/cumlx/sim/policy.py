"""Move a policy trained by rsl_rl to MLX, and prove with `check_op` that it computes the same."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..errors import CumlxError, missing_dependency

ACTOR_PREFIXES = ('mlp.', 'actor.')


def load_actor_layers(checkpoint: str) -> list[tuple[Any, Any]]:
    """(weight, bias) of each linear layer of an rsl_rl actor. It reads with weights_only=True, so the file runs no code.

    A normalizer is refused: leaving one out would give wrong actions and no error.
    """
    import torch
    loaded = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = next((loaded[key] for key in ('actor_state_dict', 'model_state_dict') if key in loaded), loaded)
    normalizers = sorted(k for k in state if 'normalizer' in k)
    if normalizers:
        raise CumlxError('E_INPUT_INVALID', f'this checkpoint has an observation normalizer, which cumlx.sim does not port: {normalizers[:3]}')
    for prefix in ACTOR_PREFIXES:
        numbers = sorted({int(k[len(prefix):].split('.')[0]) for k in state if k.startswith(prefix)})
        if numbers:
            return [(state[f'{prefix}{n}.weight'], state[f'{prefix}{n}.bias']) for n in numbers]
    raise CumlxError('E_INPUT_INVALID', f'no {" or ".join(ACTOR_PREFIXES)} parameters. Keys start with {sorted(state)[:3]}')


def mlx_policy(layers: Sequence[tuple[Any, Any]], activation: Callable[[Any], Any] | None = None) -> Callable[[Any], Any]:
    """The actor in MLX. `activation` exists for a wrong case. The default is ELU, as in rsl_rl's default."""
    import mlx.core as mx
    import mlx.nn as nn
    import torch
    act = activation or nn.elu
    converted = [(mx.array(w.to(torch.float32).numpy()), mx.array(b.to(torch.float32).numpy())) for w, b in layers]

    def run(x: Any) -> Any:
        for i, (w, b) in enumerate(converted):
            x = x @ w.T + b
            if i < len(converted) - 1:
                x = act(x)
        return x
    return run


def rsl_rl_actor(checkpoint: str) -> Callable[[Any], Any]:
    """rsl_rl's own model with the checkpoint's weights. The load is strict. It takes and returns a torch tensor."""
    import torch
    try:
        from rsl_rl.models.mlp_model import MLPModel
        from tensordict import TensorDict
    except ModuleNotFoundError as exc:
        raise missing_dependency(exc.name or 'rsl_rl', 'sim', purpose='The rsl_rl reference') from None
    layers = load_actor_layers(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)['actor_state_dict']
    size = layers[0][0].shape[1]
    actor = MLPModel(TensorDict({'policy': torch.zeros(1, size)}, batch_size=[1]), {'actor': ['policy']}, 'actor', layers[-1][0].shape[0],
                     hidden_dims=[w.shape[0] for w, _ in layers[:-1]], activation='elu', obs_normalization=False,
                     distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': 1.0, 'std_type': 'scalar'})
    actor.load_state_dict(state, strict=True)
    actor.eval()

    def run(x: Any) -> Any:
        return actor(TensorDict({'policy': x}, batch_size=[x.shape[0]]))
    return run


def verify_policy(reference: Callable[[Any], Any], mlx_fn: Callable[[Any], Any], observations: Sequence[np.ndarray], *, rel_l2: float = 1e-4) -> dict[str, Any]:
    """Compare on real observations. `reference` takes torch tensors: pass rsl_rl's own model for grade A. An empty list is incomplete."""
    from .. import check_op
    report = check_op(reference, mlx_fn, list(observations), name='policy', rel_l2=rel_l2)
    return {key: report[key] for key in ('verdict', 'worst_rel_l2', 'failed')} | {'device': report['execution']['device'], 'batches': [len(x) for x in observations]}
