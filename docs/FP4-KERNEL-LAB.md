# FP4 커널 lab

**Packed FP4 커널 구현 패키지 — CPU 검증 완료 범위와 GPU 미검증 범위를 분리합니다.**

NVFP4/MXFP4 payload를 전체 FP16/FP32 가중치로 펼치지 않고 읽는 Metal·CUDA 커널을 작성했습니다. 현재 상태는 **실험용 소스 구현**입니다. Metal 커널은 Apple M5 Pro 한 기기에서 컴파일되고 GPU 테스트 35개를 통과했습니다(MLX 0.32.2, 2026-11-02). CUDA 커널은 V100 한 대에서 실행했고, 속도는 재지 않았습니다. `allow_unvalidated=True`를 명시해야 GPU 실행을 시도합니다.

이 패키지는 v0.2의 G1/G2 커널 단계를 구현합니다. **`torch.cuda` 호환 backend, CUDA-on-Metal runtime, PTX compiler, NCCL, 임의 CUDA Python wheel 지원은 구현하지 않았습니다.** CUDA 소스는 실제 NVIDIA GPU용이며, Metal 소스는 Apple GPU용입니다. 같은 연산을 양쪽에 작성한 것이지 CUDA 바이너리를 Metal로 번역한 것이 아닙니다.

## 제공 코드

| 커널 | 용도 | 128-thread workgroup 기준 |
|---|---|---|
| `fp4_decode` | 정확성 비교용 전체 FP32 복원 | elementwise; fast path가 호출하지 않음 |
| `fp4_gemv` | M=1 decode | 출력 열 하나, 2개/nibble pair, shared 512 B |
| `fp4_small_m` | 1≤M≤8 | 복원값을 최대 4행 재사용, shared 2048 B |
| `fp4_tiled` | 일반 M의 quantized linear | BM8×BN32×BK32, shared 5120 B |
| `fp4_w4a4` | 양쪽 입력 FP4 | activation·weight 모두 타일 복원 |
| `fp4_quantize` | GPU rowwise quantizer | lane 하나가 block 하나·scale 하나 담당 |
| `rms_norm` | RMSNorm + optional residual | 행별 reduction, shared 512 B |
| `silu_mul` | SiLU(gate) × up | fused elementwise |

각 커널은 `.metal`과 `.cu`로 제공됩니다. 이는 `mx.fast.metal_kernel` / `cuda_kernel`에 전달하는 **함수 본문**입니다. dtype과 상수·buffer signature는 MLX가 생성합니다. 독립 `nvcc file.cu` 프로그램은 아닙니다. [S1, S2]

주요 경로:

```text
src/cumlx/quant.py             CPU codec, canonical tensor, serialization
src/cumlx/reference.py         독립 고정밀 수치 기준
src/cumlx/planning.py          shape dispatch와 launch geometry
src/cumlx/simulation.py        CPU 인덱스·reduction schedule 검사
src/cumlx/mlx_backend.py       MLX JIT launcher, prepared buffers, 검사
src/cumlx/kernels/             16개 커널 본문 + 2개 공통 수학 header
scripts/generate_kernels.py    위 8개 알고리즘의 소스 유지보수 도구
benchmarks/bench_fp4.py        정확성 gate 후 실제 장치 wall-time 측정
```

## 설치

Python 3.10 이상. CPU codec만 사용할 때 MLX는 필요하지 않습니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/cpu_example.py
pytest -q
```

Apple Silicon에서 GPU 실험을 준비할 때:

```bash
python -m pip install -e '.[dev,metal]'
python scripts/env_report.py
pytest -q -m gpu --backend metal --require-gpu
```

NVIDIA/Linux에서 별도 GPU 실험을 준비할 때:

```bash
python -m pip install -e '.[dev,cuda]'
python scripts/env_report.py
pytest -q -m gpu --backend cuda --require-gpu
```

MLX는 **0.32.2 이상 0.33 미만**을 허용합니다. GPU 테스트를 실행한 버전은 0.32.2와 0.32.4이고(M5 Pro 한 기기), 그 밖의 0.32.x는 경고와 함께 실행합니다. CUDA 실행은 V100 한 대에서 기록했습니다(커널 8개, case 29개, `validation/kernel-translation.json`). 설치에는 플랫폼에 맞는 MLX wheel, GPU 및 드라이버/OS 요구사항이 별도로 필요합니다. Mac에서 CUDA extra를 설치하거나 Linux CPU만으로 Metal을 실행하지 않습니다. [S1–S4]

## CPU에서 실제 실행할 수 있는 예제

```python
import numpy as np
from cumlx import quantize, dequantize, reference, save_fp4, load_fp4

rng = np.random.default_rng(28)
x = rng.normal(size=(3, 67)).astype(np.float32)
w = rng.normal(size=(35, 67)).astype(np.float32)
qw = quantize(w, 'nvfp4')
y_reference = reference.linear(x, qw)

save_fp4('weight.cumlx.npz', qw)
restored_container = load_fp4('weight.cumlx.npz')
assert restored_container.checksum == qw.checksum
print(y_reference.shape)  # (3, 35)
```

`reference.linear`은 진단용 CPU full-dequant + float64 dot입니다. GPU 결과를 검증하는 기준이지 GPU kernel의 bitwise simulator나 빠른 경로는 아닙니다.

## GPU 호출 코드 — 구현되어 있으나 하드웨어 실행 미검증

```python
import numpy as np
from cumlx import MLXBackend, quantize

backend = MLXBackend('metal', allow_unvalidated=True)  # 실제 NVIDIA는 'cuda'
mx = backend.mx
rng = np.random.default_rng(28)

# CPU에서 명시적으로 packing → 한 번 준비. 외부 framework zero-copy가 아닙니다.
qw = backend.prepare(quantize(
    rng.normal(size=(512, 1024)).astype(np.float32), 'nvfp4'
))
x = mx.array(rng.normal(size=(3, 1024)).astype(np.float32)).astype(mx.float16)

# Float activation + FP4 weight. dtype=float32이면 W4A32 계산입니다.
y = backend.linear(x, qw, algorithm='auto')
mx.eval(y)
print(y.shape, y.dtype)  # (3, 512), float32

# activation까지 FP4로 quantize한, 별도의 W4A4 경로입니다.
qa = backend.quantize(x, 'nvfp4')
y4 = backend.linear_fp4(qa, qw)
mx.eval(y4)
```

GPU kernels는 **FP32 복원·FP32 누산·FP32 출력**입니다. Native FP4 MMA/Tensor Core/simdgroup matrix 명령을 사용하는 고성능 최종판은 아닙니다. 자동 선택은 shape heuristic이며 benchmark로 선정한 최속 kernel이 아닙니다.

## 형식 계약

`W[N,K]`; 낮은 nibble에 짝수 K, 높은 nibble에 홀수 K. 홀수 마지막 K의 쓰지 않는 high nibble은 0. NVFP4의 block은 K축 16개, scale E4M3, tensor multiplier FP32. MXFP4는 K축 32개, scale E8M0, tensor multiplier 1. [S3, S5]

**자체 rowwise maxabs/RNE 추론 recipe**입니다. TransformerEngine의 2D weight scaling·swizzled/padded scale·RHT·gradient stochastic rounding을 모두 구현한 것이 아닙니다. [S5] 외부 checkpoint는 producer metadata를 해석하는 adapter로 정규화해야 합니다. 현재 로더는 오직 자체 `.npz` container를 읽습니다.

`NxFP4`, `NF4`, 임의 INT4, producer-specific layout은 오류를 반환합니다. NxFP4를 NVFP4로 대체하지 않습니다. 기존 설계의 NxFP 연구 codec 트랙은 아직 미구현입니다.

## 검증과 벤치마크

```bash
pytest -q                         # CPU test + GPU 없으면 GPU만 skip
pytest -q -m gpu --require-gpu     # GPU 없으면 반드시 실패
python benchmarks/bench_fp4.py --backend metal --m 1 --n 512 --k 1024 --output benchmark.json
```

벤치마크는 매회 새 graph 생성 → `mx.eval` → synchronize를 포함한 **Python-to-result wall latency**입니다. GPU profiler의 kernel-only 시간으로 표시하지 않습니다. packed auto/tiled, 가능한 native MLX QMM 및 compiled QMM, 명시적 full-FP32 weight를 사용하는 compiled dense comparator를 구분합니다. 수치 gate를 통과하지 못하면 속도를 채택하지 않습니다.

최신 검증 증거는 [검증 보고서](../validation/REPORT.md)와 원시 로그에 있습니다. GPU skip은 pass가 아니며, C++ host header 검증은 CUDA device compile이 아닙니다. 하드웨어별 성능 수치는 제공하지 않습니다.

MLX는 M5와 CUDA에서 float32 matmul을 기본으로 TF32로 실행합니다. CUMLX 커널은 자체 Metal 코드라 이 설정의 영향을 받지 않지만, native MLX QMM·dense 비교 대상은 영향을 받습니다. 벤치마크를 비교할 때 `MLX_ENABLE_TF32` 값을 함께 기록하십시오.

## 제약과 다음 gate

단일 장치·MLX stream·추론만 지원하는 실험 소스입니다. backward/vmap/TorchTensor/외부 pointer/자동 CPU fallback/동적 vendor checkpoint 변환은 없습니다. FP4 buffer를 변경하지 않아야 하며, 비연속 activation은 MLX의 row-contiguous 준비 과정에서 복사될 수 있습니다. 극단적 subnormal·overflow 영역의 CPU/GPU bitwise 동등성은 인증하지 않았습니다.

다음 gate는 CUDA와 다른 Apple GPU 세대에서 전체 GPU 테스트를 실행하고, benchmark profile별 수치·메모리·성능을 기록하는 것입니다. 그 뒤에만 SIMD reduction, vectorized loads, matrix instructions, MoE gather 및 CUDA-facing frontend를 확장합니다.

[커널 상세 설계](KERNEL-DESIGN.md) · [구현 계획](IMPLEMENTATION-PLAN.md) · [1개월 로드맵 (포팅 검증기)](ROADMAP-1MONTH.md) · [원본 자료](../SOURCES.md)

`../provenance/SOURCE-ARCHIVES.json`은 전달받은 원본 ZIP의 SHA-256을 기록합니다. 바이너리 ZIP과 이전 wheel은 Git에 중복 저장하지 않으며, distributions는 CI artifact로 생성합니다. 과거 설계와 CPU 검증이 실제 GPU 검증을 대신하지 않습니다. CUMLX는 Apple 또는 NVIDIA 공식 프로젝트가 아닙니다.


## GitHub Actions CI

`main` push와 pull request에서 correctness lint, Python 3.10–3.13 CPU tests,
생성 커널 byte 일치, sdist/wheel 빌드 및 checkout 밖의 wheel smoke test를 실행합니다.
CPU job은 GPU 테스트를 **deselect**하며 GPU 성공으로 보고하지 않습니다.

[CI](../.github/workflows/ci.yml) · [실제 GPU 검증](../.github/workflows/gpu.yml) · [CI 실행 범위](CI.md)

GPU workflow는 신뢰하는 `main`에서 수동 실행만 허용합니다.
실제 장치를 가진 self-hosted runner에 `metal` 또는 `cuda` label이 필요합니다.
GPU가 없으면 `--require-gpu`로 실패하며 CPU fallback으로 통과시키지 않습니다.
PyPI 게시, 유료 GPU provisioning, secret 추가, branch protection 변경은 하지 않습니다.
