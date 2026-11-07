"""The change of ml-explore/mlx-lm#1964 (head a3117179ce1ccb2ed814c0b21f8636135ed8553d) to mlx_lm/models/base.py.

_flat_rope_config and from_dict restate that pull request's diff, under mlx-lm's MIT licence (Copyright (c) 2023
Apple Inc.). pr1964_check.py and config_scan.py use them in their own process only; nothing here changes mlx-lm.
"""
import inspect


def _flat_rope_config(rope_parameters):
    if not isinstance(rope_parameters, dict):
        return {}
    config = {}
    if "rope_theta" not in rope_parameters and "full_attention" in rope_parameters:
        sliding = rope_parameters.get("sliding_attention") or {}
        if "rope_theta" in sliding:
            config["rope_local_base_freq"] = sliding["rope_theta"]
        rope_parameters = rope_parameters["full_attention"] or {}
    for key in ("rope_theta", "partial_rotary_factor", "original_max_position_embeddings"):
        if key in rope_parameters:
            config[key] = rope_parameters[key]
    rope_type = rope_parameters.get("rope_type") or rope_parameters.get("type", "default")
    if rope_type == "default":
        config["rope_scaling"] = None
    else:
        scaling = {k: v for k, v in rope_parameters.items() if k not in ("rope_theta", "partial_rotary_factor")}
        scaling["type"] = scaling["rope_type"] = rope_type
        config["rope_scaling"] = scaling
    return config

@classmethod
def from_dict(cls, params):
    fields = inspect.signature(cls).parameters
    if "rope_parameters" in params and "rope_parameters" not in fields:
        params = {**_flat_rope_config(params["rope_parameters"]), **params}
    return cls(**{k: v for k, v in params.items() if k in fields})
