# Measure a CUDA golden against a CPU golden

The roadmap asks how far a CUDA golden moves from a CPU golden. This procedure answers it on any machine with an NVIDIA GPU. Run it on the GPU machine, then compare the traces anywhere that has `cumlx`.

**Status:** run on one Tesla V100 on 2026-11-02, with Qwen2.5-0.5B. [Result](../validation/cuda-golden-v100.json):

| Comparison | Worst rel_l2 | Above 1e-4 |
|---|---|---|
| CUDA against CUDA, a repeat run | 0.0 | 0 of 1272 outputs |
| CUDA against CPU, same machine | 1.9e-5 | 0 of 1272 |
| V100 CUDA against Apple M5 Pro CPU | 4.5e-5 | 0 of 1272 |
| V100 CPU against Apple M5 Pro CPU | 5.1e-5 | 0 of 1272 |

An MLX bf16 port on the Mac also passes against the CUDA golden: verdict `ok`, top-1 13 of 13. So the threshold policy needs no change for this model, and a golden from a CUDA machine can be used on a Mac. The default `torch.backends.cuda.matmul.allow_tf32=False` was in effect. Other GPU generations and other models were not measured. Run the steps below to measure another machine.

## 1. Install in a new virtual environment

```bash
python3 -m venv cumlx-venv && . cumlx-venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install 'cumlx-lab[capture] @ git+https://github.com/Topabaem05/CuMLX@v0.4.0'   # the repository is private: needs your GitHub access
```

Tesla V100 needs a PyTorch build that still supports `sm_70`. Check it first:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. Record three traces with the same inputs

```bash
MODEL=Qwen/Qwen2.5-0.5B
INPUT='--token-ids 1,5,9,23,7,99,3,64,12,8 --token-ids 151643,872,1234'
cumlx capture $MODEL $INPUT --device cuda -o golden-cuda-a/
cumlx capture $MODEL $INPUT --device cuda -o golden-cuda-b/
cumlx capture $MODEL $INPUT --device cpu  -o golden-cpu/
```

## 3. Compare

```bash
cumlx tracediff golden-cuda-a/ golden-cuda-b/   # CUDA repeatability
cumlx tracediff golden-cuda-a/ golden-cpu/      # CUDA against CPU
```

Read `data.summary.worst_rel_l2`.

| Result | Meaning |
|---|---|
| Exit 0 | The golden moves less than the float32 threshold of `cumlx diff` (1e-4). The policy needs no change |
| Exit 1 | The golden itself moves above the threshold. Use a CPU golden for strict comparison, or raise the threshold with the measured number as its reason |

Keep `torch.backends.cuda.matmul.allow_tf32` off when you record a golden. PyTorch has it off by default.

Record the GPU, driver, PyTorch version and these numbers in the validation report.
