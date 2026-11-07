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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pr1964_logic import from_dict  # noqa: E402

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
