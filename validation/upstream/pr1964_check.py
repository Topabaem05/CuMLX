"""Does the change of ml-explore/mlx-lm#1964 (head a3117179ce1ccb2ed814c0b21f8636135ed8553d) fix rope_parameters_repro.py?

    python validation/upstream/pr1964_check.py              # mlx-lm as installed
    python validation/upstream/pr1964_check.py --patched    # with the logic of #1964

_flat_rope_config and from_dict below restate that pull request's change to mlx_lm/models/base.py, from its diff, under mlx-lm's MIT
licence (Copyright (c) 2023 Apple Inc.). They patch the installed mlx-lm in this process only. So this is evidence for the person who
writes the upstream comment, and not a run of the pull request's branch. Run it from the repository root.
"""
import inspect, json, os, runpy, sys, tempfile
from pathlib import Path
import mlx_lm.models.base as base

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

if sys.argv[1:] == ["--patched"]:
    base.BaseModelArgs.from_dict = from_dict
    print("patched with the logic of #1964")
runpy.run_path("validation/upstream/rope_parameters_repro.py", run_name="__main__")

# The survey's models: the RoPE base each one takes from a transformers 5 config with rope_theta 12345
import importlib
from transformers import AutoConfig
for name, cfg_cls in (("llama", "LlamaConfig"), ("qwen2", "Qwen2Config"), ("mistral", "MistralConfig"), ("phi3", "Phi3Config"), ("starcoder2", "Starcoder2Config")):
    import transformers
    cfg = getattr(transformers, cfg_cls)(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    cfg.rope_parameters["rope_theta"] = 12345.0
    d = cfg.to_dict()
    try:
        module = importlib.import_module(f"mlx_lm.models.{name}")
    except ModuleNotFoundError:
        print(f"  {name}: no module of its own in this mlx-lm"); continue
    args = module.ModelArgs.from_dict(d)
    print(f"  {name}: rope_theta in the MLX args = {getattr(args, 'rope_theta', None)}")
