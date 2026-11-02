# Measure a CUDA golden against a CPU golden

The roadmap asks how far a CUDA golden moves from a CPU golden. This procedure answers it on any machine with an NVIDIA GPU. Run it on the GPU machine, then compare the traces anywhere that has `cumlx`.

**Status:** not run yet. The comparison tool was checked with Apple MPS against CPU; see [the validation report](../validation/REPORT.md).

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

Record the GPU, driver, PyTorch version and these numbers in the validation report.
