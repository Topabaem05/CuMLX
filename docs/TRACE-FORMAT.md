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

## Op-level golden `cumlx.opgolden.v1`

A trace records a whole model. An op-level golden records one kernel: what it computed on one set of inputs, on a real device. `cumlx kernel capture` writes it on an NVIDIA machine, and `cumlx kernel verify` reads it on a Mac. Any tool can write or read it.

```text
<golden>/manifest.json
<golden>/cases/c0000.safetensors
<golden>/cases/c0001.safetensors
```

```json
{
  "schema": "cumlx.opgolden.v1",
  "cumlx_version": "0.5.0",
  "created_utc": "2026-11-03T05:00:00Z",
  "golden": {"kind": "cuda"},
  "source": {
    "framework": "cuda", "gpu": "Tesla V100-SXM2-16GB", "driver": "580.178.04", "arch": "7.0",
    "compiler": {"name": "nvcc", "version": "12.0.140", "flags": ["-arch=native"]},
    "cuda_runtime": 12000, "cuda_driver_api": 13000, "idle_during_capture": true, "repeats": 3
  },
  "floating_point": {
    "ftz": {"flag": "false (nvcc default)", "load_flushes": false, "input_flushes": false, "output_flushes": false},
    "fmad": {"flag": "true (nvcc default)", "contracts": true},
    "fp64": {"available": true}
  },
  "kernels": {
    "fp4_gemv": {"source_file": "fp4_gemv.cu", "source_sha256": "...", "header_sha256": "...",
                 "form": "body", "features": {"fp64": false, "fp64_literals": 0, "atomics": false, "warp_level": []}}
  },
  "cases": [{
    "id": "fp4_gemv.nvfp4.1x7x17", "kernel": "fp4_gemv",
    "inputs":  [{"name": "x", "dtype": "float32", "shape": [1, 17]}],
    "outputs": [{"name": "y", "dtype": "float32", "shape": [1, 7]}],
    "constants": {"M": 1, "N": 7, "K": 17, "HAS_BIAS": true},
    "grid": [896, 1, 1], "threadgroup": [128, 1, 1],
    "file": "cases/c0000.safetensors", "sha256": "...", "repeat_identical": true
  }]
}
```

| Field | Meaning |
|---|---|
| `golden.kind` | `cuda`: a real CUDA run. It must record `source.compiler`, `source.gpu` and `source.driver`. `fixture`: test data, which never earns a proof grade. A reader must reject any other value |
| `grid`, `threadgroup` | Logical threads, as MLX counts them. CUDA launches `grid / threadgroup` blocks of `threadgroup` threads. The grid must be a whole number of threadgroups |
| `constants` | Integer or boolean values that the kernel reads by name. In MLX they are template arguments |
| `entry` | Optional. The function to launch when the source is a complete kernel, such as `vector_add<float>`. Default: the kernel name |
| `kernels.*.form` | `body`: a function body, as `mx.fast.cuda_kernel` takes. `complete`: a source with `__global__` functions |
| `kernels.*.features` | What a text scan of the original found. A `double` or an unsuffixed floating literal runs in fp64 on CUDA and cannot on Metal |
| `floating_point` | Measured on this device by a probe kernel with the same compiler flags |
| `cases[].repeat_identical` | Every repeat gave the same bytes, although the output memory was filled with a different pattern before each one. False means that the kernel leaves elements unwritten or depends on execution order |
| `cases[].sha256` | SHA-256 of the case file. A reader must reject a mismatch |

Each case file holds `in.<name>` and `out.<name>` for each tensor, in little-endian, with the dtype of the manifest. Output `out.<name>` is the first repeat.

### Cases file `cumlx.kernel-cases.v1`

The input of `cumlx kernel capture`. It has the same case fields, without `file` and `sha256`. An input also says where its data is: `"values": [...]` inline, or `"file": "inputs.safetensors"` with an optional `"tensor"` name. The file must stay beside the cases file. `scripts/make_kernel_cases.py` writes the cases of the FP4 kernels.

### Comparison

`cumlx kernel verify` judges each output by `KERNEL_THRESHOLDS` in `verify/metrics.py`. Integer outputs must match exactly. In a float output, NaN and infinity must sit at the same elements, and the finite elements must satisfy `rel_l2 <= 1e-5` and `max |a - g| / max |g| <= 1e-5`. If all finite golden elements are zero, the limit is `max |a| <= 1e-6`.

## Compatibility

A new optional field does not change the schema name. A reader must ignore unknown fields and must use the defaults above for missing ones. A change that a v1 reader would misread needs a new schema name. A reader must reject an unknown schema or `golden.kind`.
