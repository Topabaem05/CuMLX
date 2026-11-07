"""mlx-lm silently uses a default RoPE base for a config written by transformers 5.

transformers 5 writes rope_theta inside rope_parameters. Needs only mlx-lm, transformers and torch.
"""
import atexit
import json
import os
from pathlib import Path
import shutil
import tempfile

os.environ.setdefault("MLX_ENABLE_TF32", "0")  # full float32, so the comparison has no matmul noise

import mlx.core as mx  # noqa: E402
from mlx_lm.utils import load_model  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                     num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=False)
config.rope_parameters["rope_theta"] = 500000.0  # the value in Llama 3
torch.manual_seed(0)
original = LlamaForCausalLM(config).float().eval()
for p in original.parameters():
    if p.ndim > 1:
        torch.nn.init.normal_(p, std=0.25)

path = Path(tempfile.mkdtemp())
atexit.register(shutil.rmtree, path, ignore_errors=True)
original.save_pretrained(path)
saved = json.load(open(path / "config.json"))
print("saved config:", {k: v for k, v in saved.items() if k.startswith("rope")})

port, _ = load_model(path)  # no error and no warning
print("RoPE base in the MLX model:", port.model.layers[0].self_attn.rope.base)

ids = np.array([[1, 5, 9, 23, 7, 44, 3, 12]])
with torch.no_grad():
    expected = original(torch.tensor(ids)).logits.numpy()
actual = np.array(port(mx.array(ids)))
print("relative error of the logits:", float(np.linalg.norm(actual - expected) / np.linalg.norm(expected)))
print("same top-1 token:", (actual.argmax(-1) == expected.argmax(-1)).mean())

saved_flat = {**saved, "rope_theta": 500000.0}
with open(path / "config.json", "w") as f:
    json.dump(saved_flat, f)
port, _ = load_model(path)
print("after adding a top-level rope_theta, base:", port.model.layers[0].self_attn.rope.base)
actual = np.array(port(mx.array(ids)))
print("relative error of the logits:", float(np.linalg.norm(actual - expected) / np.linalg.norm(expected)))
