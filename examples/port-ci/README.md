# CUMLX in the CI of a porting repository

This example shows a repository that keeps an MLX port and checks every pull request with the CUMLX action. It uses a tiny random model, so it needs no download.

| Path | Purpose |
|---|---|
| `original/` | The model that the port must match. A real repository uses an HF id instead |
| `port/loader.py` | The code under review. It returns an MLX `nn.Module` |
| `.github/workflows/verify-port.yml` | Runs the action. Only verdict `ok` passes. Copy it to the root of a porting repository |

Create `original/` once:

```bash
cumlx fixture tiny-llama -o original/
```

`port/loader_faulty.py` is the same port with a wrong RoPE base. [The action test](../../.github/workflows/action-test.yml) runs both in this repository: the clean port must pass, and the faulty port must fail at `model.layers.1.self_attn`.

A pull request that changes the port so it computes something else fails, and the job summary names the first module that differs. `incomplete` also fails, because it is not a pass.
