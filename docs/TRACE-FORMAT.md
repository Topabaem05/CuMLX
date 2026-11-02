# Trace format `cumlx.trace.v1`

A golden trace records what the original model computes, module by module. Any tool can write or read it. The format has no dependency on CUMLX, PyTorch or MLX.

## Layout

```text
<trace>/manifest.json
<trace>/samples/s0000.safetensors
<trace>/samples/s0001.safetensors
```

Each sample is one input. The manifest describes the records. The tensors are in the sample's safetensors file. All tensors are stored in little-endian, and `store_dtype` in the manifest gives the float type (`float32` or `float16`).

## `manifest.json`

```json
{
  "schema": "cumlx.trace.v1",
  "cumlx_version": "0.4.0",
  "created_utc": "2026-11-02T05:00:00Z",
  "source": {
    "framework": "torch", "framework_version": "2.15.0", "transformers_version": "5.22.1",
    "model": "Qwen/Qwen2.5-0.5B", "revision": "060db649...", "architecture": "Qwen2ForCausalLM", "device": "cpu",
    "tied_modules": {"lm_head": "model.embed_tokens"},
    "partial": {"layers": [0, 11], "include": null, "exclude": null}
  },
  "golden": {"kind": "reference", "dtype": "float32", "store_dtype": "float32", "weights_from_mlx": null},
  "samples": [{"id": "s0000", "file": "samples/s0000.safetensors", "sha256": "...", "num_tokens": 7, "decode_steps": 1, "records": []}]
}
```

| Field | Meaning |
|---|---|
| `golden.kind` | `reference`: the original weights. `quantization-aware`: the port's own weights in the original code, which removes quantization loss and rounding from the comparison |
| `source.tied_modules` | A module whose parameter is the same object as another module's |
| `source.partial` | Present when only some modules were recorded. A comparison covers only what was recorded |
| `samples[].sha256` | SHA-256 of the sample file. A reader must reject a mismatch |
| `samples[].decode_steps` | Number of greedy steps recorded on the KV cache. 0 means prefill only. Older traces omit it |

## Records

Records are in completion order, so a child comes before its parent. The root module has the empty name `""`.

```json
{
  "key": "model.layers.0.self_attn#0", "name": "model.layers.0.self_attn", "call": 0, "order": 12,
  "type": "Qwen2Attention", "has_params": true, "replayable": true,
  "replay_kind": "hidden", "phase": "prefill", "replay_note": "...",
  "inputs": [{"tensor": "r00012.in0", "shape": [1, 7, 896], "dtype": "float32", "source": "kw:hidden_states"}],
  "output": {"tensor": "r00012.out", "shape": [1, 7, 896], "dtype": "float32", "source": "tuple[0]"}
}
```

| Field | Meaning |
|---|---|
| `call` | Call index of this module in the sample. Decode calls continue after the prefill calls |
| `phase` | `prefill`, or `decode` for a call on the KV cache. Older traces omit it, which means `prefill` |
| `output.source` | Which part of the return value was recorded: `tensor`, `tuple[0]` or `ModelOutput[0]` |

`replay_kind` says how a reader replays a module on its golden input alone:

| Value | Inputs | How to replay |
|---|---|---|
| `single` | One tensor | Call the module with it |
| `hidden` | Hidden state `[batch, time, dim]` | Call the module with it and a causal mask. The port computes position embeddings itself |
| `hidden_cached` | Hidden state of this step | New KV cache; run this module's earlier steps with their golden inputs; then run this step |
| `tokens_cached` | Token ids of this step | The same as `hidden_cached`, for a module that owns all layer caches (the root or the model body) |
| `experts` | Hidden state, top-k indices, top-k weights | Run the MoE experts with this routing and combine them with these weights |
| `null` | None | Not replayable. A reader compares only the chain output |

## Sample files

Each sample file is standard safetensors:

| Tensor | Shape | Meaning |
|---|---|---|
| `token_ids` | `[1, T]`, int64 | Prefill input |
| `decode_token_ids` | `[1, S]`, int64 | Token of each decode step, chosen greedily from the golden logits. Present when `decode_steps` > 0 |
| `rNNNNN.inM` | Any | Input M of record NNNNN |
| `rNNNNN.out` | Any | Output of record NNNNN |

## Compatibility

A new optional field does not change the schema name. A reader must ignore unknown fields and must use the defaults above for missing ones. A change that a v1 reader would misread needs a new schema name. A reader must reject an unknown schema or `golden.kind`.
