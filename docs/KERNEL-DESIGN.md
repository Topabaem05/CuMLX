# CUMLX v0.3 — FP4 커널 상세 설계 및 구현

**상태: CPU reference/host helper 검증 + GPU 소스 구현. Metal/CUDA device compile 및 실행 미검증.**

이 문서는 배포 소스와 대응합니다. v0.2의 전체 CUDA 호환 런타임 설계를 완성한 문서가 아니라, 그중 quantized operator provider의 첫 구현입니다. Native NVIDIA와 Metal은 같은 수학을 각각 실행하도록 작성했고, CUDA API를 Metal로 가로채는 기능은 없습니다.

## 1. 수학 및 입력 ABI

기본 연산은 다음과 같습니다.

```text
X[M,K] × W_hat[N,K]^T → Y[M,N]
Y[m,n] = sum_k float32(X[m,k]) * W_hat[n,k] + bias[n]
선택 epilogue: Y <- SiLU(Y)
```

`float16/bfloat16/float32` activation을 받습니다. 커널 안에서 입력을 FP32로 바꾸며 **가중치 복원, FMA 누산, 출력도 FP32**입니다. f16/BF16 activation 경로는 W4A16, f32 activation은 W4A32입니다. 작은 형태를 위해 dtype를 은밀하게 바꾸지 않습니다. output을 half로 사용하려면 호출부가 명시적으로 변환해야 합니다.

CPU oracle은 FP32로 복원된 가중치와 실제 input 값을 **float64 dot**으로 계산한 뒤 float32로 반환합니다. 따라서 reduction 순서가 다른 GPU 결과와 bitwise equality를 요구하지 않고 수치 tolerance로 비교합니다. FP4로 인한 모델 양자화 오차와 커널 구현 오차도 분리합니다.

### Packed weight ABI

| 배열 | dtype·shape | 의미 |
|---|---|---|
| `wq` | uint8 `[N,ceil(K/2)]` | 한 byte에 E2M1 두 코드 |
| `ws` | uint8 `[N,ceil(K/G)]` | encoded block scale, 실수 scale 배열이 아님 |
| `wglobal` | float32 `[1]` | 복원 multiplier; MXFP4는 1 |
| `bias` | float32 `[N]` 또는 미사용 dummy `[1]` | `HAS_BIAS` specialization |
| `x` | float16/BF16/float32 `[M,K]` | row contiguous로 launch; view는 copy 가능 |
| `y` | float32 `[M,N]` | 새 allocation, in-place/out= 미지원 |

주소 계산:

```text
byte_index  = n * ceil(K/2) + floor(k/2)
nibble      = (wq[byte_index] >> (4*(k mod 2))) & 0xF
scale_index = n * ceil(K/G) + floor(k/G)
```

짝수 K가 low nibble입니다. 홀수 마지막 K의 high nibble은 0으로 고정합니다. 한 행의 padding이 다음 행으로 흘러가지 않습니다. GPU quantizer는 lane 하나가 group 하나를 소유하며 G가 짝수이므로 같은 byte를 두 lane이 수정하지 않습니다.

### Scale 해석

NVFP4는 E2M1 코드 × E4M3 block scale × tensor FP32 multiplier, G=16입니다. MXFP4는 E2M1 코드 × E8M0 block scale, G=32입니다. [S3, S5]

```text
NV: W_hat = (decode_E2M1(code) * decode_E4M3(scale_byte)) * global
MX: W_hat = (decode_E2M1(code) * 2^(scale_byte-127)) * 1
```

곱셈 순서를 코드에 고정합니다. GEMM 결과 끝에서 scale을 적용하는 것과 유한 정밀도에서 같다고 가정하지 않습니다. E4M3 negative/NaN scale, E8M0 NaN code255, 비양수·비유한 tensor scale, float32 복원 overflow를 CPU descriptor에서 거부합니다. `-0`의 signbit는 보존합니다.

### 형식과 producer 범위

`FP4Matrix`가 받는 것은 정규화한 자체 row-major 계약입니다. packed bytes가 같아 보인다고 TransformerEngine의 2D weight scaling, swizzled/padded layout, NF4, INT4, NxFP4를 읽지 않습니다. TransformerEngine의 RHT와 stochastic-gradient recipe도 미구현입니다. [S5]

`NxFP4`는 별도 연구 codec이 필요한 이름으로 유지하며 명시적 unsupported 오류를 반환합니다. 저장 로더는 자체 `.npz` container, 고정 metadata, SHA-256과 `allow_pickle=False`만 허용합니다.

## 2. 공통 FP4 decoder와 encoder

E2M1 magnitude는 `0, 0.5, 1, 1.5, 2, 3, 4, 6`, 부호가 별도 1bit입니다. 아래는 실제 Metal header 일부입니다.

```cpp
inline float cumlx_e2m1(unsigned int c) {
    unsigned int e = (c & 7u) >> 1u;
    unsigned int m = c & 1u;
    float v = e == 0u ? 0.5f * float(m)
        : metal::ldexp(1.0f + 0.5f * float(m), int(e) - 1);
    return metal::copysign(v, (c & 8u) ? -1.0f : 1.0f);
}
```

Encoder는 nearest-even을 적용합니다. 아주 큰 finite 값은 거리 계산 전에 magnitude=6으로 포화시킵니다. 이를 먼저 하지 않으면 float32 최대값처럼 큰 수에서 `x-4`와 `x-6`을 구분하지 못해 잘못된 code4를 고를 수 있습니다. 실제 regression을 재현하고 CPU/C++ header 모두 고쳤습니다.

E4M3 scale encoder는 finite code0..126 사이의 binary search + ties-even이며, nonzero block scale이 0으로 반올림될 때 최소 positive code1로 올리는 **CUMLX 자체 규칙**을 사용합니다. MXFP4 block scale은 `amax/6` 이상인 최소 E8M0 power-of-two를 고릅니다.

## 3. GEMV — 한 토큰 decode

실제 파일: `fp4_gemv.metal`, `fp4_gemv.cu`. `BM=1`, `M=1`만 허용합니다.

한 workgroup은 출력 `Y[0,n]` 하나를 계산합니다. 128 lane 중 lane l은 `k=2*l, 2*l+256, ...`의 byte 쌍을 담당합니다. 하나의 byte와 scale을 읽어 두 값을 복원하고 FMA 두 번에 사용합니다. group scale 크기 G=16/32와 byte pair의 경계가 정렬되어 두 nibble이 scale을 공유합니다.

```text
lane l:
  packed byte -> w0,w1
  acc += x[k]   * w0
  acc += x[k+1] * w1  (k+1<K일 때)

128개 partial -> shared-memory tree reduction -> lane0 output
```

Shared memory는 128 float = **512 B**입니다. inactive lane도 partial=0을 기록하고 모든 barrier를 지납니다. K<256에서도 warp 활성 mask를 임의로 가정하지 않습니다. 최종 bias와 SiLU는 lane0에서 같은 커널 안에 적용합니다.

K가 매우 작거나 N이 작아 group 수가 부족한 상황의 최적 성능은 보장하지 않습니다. 이 버전은 K-split 또는 atomic accumulation을 쓰지 않습니다.

## 4. Small-M QMM — 복원값을 여러 행에 재사용

실제 파일: `fp4_small_m.metal`, `.cu`. `BM=4`, 1≤M≤8 허용, 자동 선택은 2≤M≤8입니다.

한 group이 `Y[m0:m0+4,n]`을 담당합니다. 같은 packed pair를 한 번 읽고 최대 4개 activation 행에 재사용합니다. M=5..8에서는 M 방향 group 두 개가 필요해 W를 다시 읽습니다. 공유 메모리는 `4*128*4 = 2048 B`입니다.

아래는 실제 두 커널의 공통 핵심 구조입니다. 전체 원문은 부록 A에 있습니다.

```cpp
for (unsigned int k = lane * 2u; k < K; k += 256u) {
    unsigned int pair = wq[n * WP + k / 2u];
    unsigned int scale = ws[n * WSG + k / WG];
    float w0 = cumlx_restore(pair & 15u, scale, wglobal[0], WF);
    float w1 = k + 1u < K
        ? cumlx_restore(pair >> 4u, scale, wglobal[0], WF) : 0.0f;
    for (unsigned int r = 0u; r < BM; ++r) {
        if (m0 + r < M) {
            accum[r] = metal::fma(float(x[(m0+r)*K+k]), w0, accum[r]);
            if (k + 1u < K)
                accum[r] = metal::fma(float(x[(m0+r)*K+k+1u]), w1, accum[r]);
        }
    }
}
```

이 재사용은 코드 구조상의 개선입니다. native MLX보다 빠르다는 측정 결과가 아닙니다. register pressure 증가, barrier 비용, 캐시 상황을 실제 장치에서 비교해야 합니다.

## 5. Tiled QMM — BM8 × BN32 × BK32

실제 파일: `fp4_tiled.metal`, `.cu`. 모든 양수 M에 강제로 선택할 수 있고 자동 선택은 M≥9입니다.

```text
한 group:
  X tile: 8×32 FP32     = 1024 B
  W tile: 32×32 FP32    = 4096 B
  합계                  = 5120 B

128 lane × 출력 2개 = 8×32 output tile
```

Lane mapping:

```text
row  = lane / 16      -> 0..7
col0 = lane % 16      -> 0..15
col1 = col0 + 16      -> 16..31
```

K 타일마다 activation과 packed weight를 협력 load합니다. packed weight는 **threadgroup tile 안에서만 FP32로 복원**하며 전체 `[N,K]` 복원 allocation을 만들지 않습니다. 모든 load가 끝난 뒤 barrier, lane당 출력 2개의 32-step FMA, 다음 타일로 덮어쓰기 전에 barrier를 수행합니다.

부분 M/N/K는 load 단계에서 0으로 채우고 마지막 store에서 mask를 적용합니다. 경계 lane이 미리 return해서 barrier를 건너뛰는 경로는 없습니다. K-tail도 32회 FMA의 0-padding으로 처리합니다.

이 구현은 scalar FMA 기반입니다. Tensor Core나 Metal matrix 명령을 사용하지 않으므로 큰 prefill에서 native 최적화 GEMM보다 느릴 수 있습니다. 행렬 명령 버전을 붙일 기준선이자 이식 가능한 첫 packed implementation입니다.

## 6. W4A4 — activation도 실제 FP4로 복원

실제 파일: `fp4_w4a4.metal`, `.cu`.

W4A16 path에 이름만 붙인 것이 아닙니다. 입력 activation도 `aq/ascales/aglobal`을 받으며 A tile load에서 E2M1과 scale을 복원합니다. activation group `AG`와 format `AF`를 weight의 `WG/WF`와 별도로 specialization하므로 NV×NV, MX×MX, NV×MX를 시험하도록 구성했습니다.

```text
A_fp4 -> tile-local FP32 decode ---+
                                  +-> FP32 FMA -> Y float32
W_fp4 -> tile-local FP32 decode ---+
```

이는 **FP4로 양자화한 두 피연산자의 의미를 계산**하는 구현입니다. GPU의 native FP4 행렬 명령을 실행한다는 의미는 아닙니다. Quantizer와 GEMM은 별도 launch이며 activation quantization까지 fused라고 주장하지 않습니다.

## 7. GPU block quantizer

실제 파일: `fp4_quantize.metal`, `.cu`.

동적 activation용 경로는 MLX graph로 전체 `amax`를 구하고, 다음 scalar block encoder에 tensor scale을 전달합니다. NV 자동 global은 `amax/2688`, all-zero는 1, 아주 작은 값은 FP32 tiny로 floor합니다. 현재 구현은 자체 deterministic rowwise inference recipe입니다.

작업 분배:

```text
thread_id -> (row, block)
G개 input absolute max
encoded block scale 선택
G/2개 byte에 두 E2M1 코드씩 기록
scale byte 하나 기록
```

하나의 lane만 각 packed byte와 scale을 기록합니다. subgroup이나 atomic을 쓰지 않아 race 소지가 단순합니다. CPU와 GPU의 정규화 연산 순서는 `(value/global)/block_scale`로 맞췄습니다. 포화 영역의 intermediate infinity는 clip으로 처리하고 NaN/Inf 원본 입력은 Python preflight에서 거부합니다.

Finite 검사와 명시적 validation은 host 관측을 위해 synchronize할 수 있습니다. 따라서 현재 `backend.quantize`를 zero-overhead decode loop quantizer라고 홍보하지 않습니다. activation-side graph fusion은 후속 성능 작업입니다. subnormal 처리, division 반올림과 vendor별 compiler 차이는 실제 장치 테스트 전까지 인증하지 않습니다.

## 8. RMSNorm / SiLU-multiply

RMSNorm은 한 group에 한 행입니다.

```text
v[k] = float32(x[k]) + optional float32(residual[k])
mean_square = sum_k v[k]^2 / K
out[k] = v[k] / sqrt(mean_square + epsilon) * float32(weight[k])
```

128 lane의 FP32 sum-of-squares를 tree reduce하고 입력을 다시 읽어 normalized output을 씁니다. 별도의 residual output은 만들지 않습니다. 입력이 커서 float32 square sum이 overflow하는 영역을 안정화하는 scaled-sum 알고리즘은 현재 범위 밖입니다. finite 중간 계산 범위의 테스트를 기준으로 합니다.

SiLU는 overflow를 줄이도록 `z=exp(-abs(x))`를 사용합니다. positive/negative 가지에서 sigmoid를 구성한 뒤 gate×sigmoid×up을 계산합니다. 이 안정화는 dtype 반올림과 ±Infinity/NaN 의미까지 전 범위 인증했다는 뜻은 아닙니다.

## 9. Launch plan과 synchronization

모든 reduction/tile kernel의 threadgroup은 `(128,1,1)`이며 grid는 **논리 thread 수**입니다. CUDA block 수를 직접 grid argument에 전달하지 않습니다. MLX CUDA는 논리 thread extent를 block count로 변환하고, Metal은 thread dispatch를 구성합니다. [S7, S8]

```text
GEMV: groups = N
small-M: groups = N * ceil(M/4)
tiled/W4A4: groups = ceil(M/8) * ceil(N/32)
grid = (groups*128,1,1)
```

전체 group으로 padding하여 모든 barrier에 128 lane이 도달합니다. 1D group index를 `(tile_m,tile_n)`으로 나누기 때문에 CUDA grid.y 크기를 가정하지 않습니다. 값/출력 element count와 logical launch size는 signed-int32 범위를 초과하면 오류입니다.

메모리 ownership·stream 의존성은 **MLX 자체 graph와 stream**에 맡깁니다. 이 버전은 raw CUDA pointer, external event, Torch allocator bridge가 없습니다. `ensure_row_contiguous=True`는 비연속 input을 복사할 수 있습니다. zero-copy 주장은 하지 않습니다.

`DeviceFP4`는 backend instance 소유자를 기록해 다른 session buffer의 우발 사용을 거부합니다. 내부 MLX array에 대한 사용자의 고의 mutation까지 막는 immutable storage/runtime은 아니므로, 준비한 buffer는 수정하지 않아야 합니다. Weight update가 필요하면 CPU FP4Matrix부터 다시 prepare합니다.

## 10. Dispatch와 cache

`plan_linear`의 auto 선택은 M만 사용하는 고정 초기 정책입니다. 작은 K, stride, architecture, register pressure를 반영한 autotuner는 미구현입니다. 사용자는 `algorithm='gemv'|'small_m'|'tiled'`로 유효 범위 안에서 강제 비교할 수 있습니다.

Python launcher는 kernel factory를 backend instance에서 캐시하고 source+header SHA-256을 kernel name에 넣습니다. shape/format/epilogue는 MLX template specialization에 전달합니다. 사용자 커널 cache를 별도로 영구 저장하는 기능은 없으며 MLX compiler cache에 의존합니다. 하드웨어를 'SM100'처럼 허위 표시하지 않습니다.

## 11. 검증 계층과 증거 해석

| 계층 | 이번 수행 여부 | 무엇을 증명하는가 |
|---|---|---|
| CPU codec 및 serialization tests | 수행 | 코드·scale·padding·checksum·오류 정책 |
| Float64 reference comparisons | 수행 | 의도한 수학과 Python 참조 연산 |
| Launch schedule CPU simulation | 수행 | 경계 shape에서 인덱스 분배, store coverage=1 |
| Actual CUDA helper header의 HOST C++ compile/run | 수행 | 수학 helper의 CPU C++ 결과; device compile 아님 |
| Python API import·fail-closed tests | 수행 | MLX 없는 환경의 검사, source registry |
| Metal compiler/NVRTC | 미수행 | 실제 장치 환경 필요 |
| GPU reduction·synchronization·accuracy | 미수행 | 실제 launch 성공 및 수치 필요 |
| GPU performance | 미수행 | 동일 의미 native baseline 비교 필요 |
| CUDA-facing PyTorch | 미구현 | 별도 v0.2 G3/G4 개발 |

GPU test는 실제 장치가 없을 때 skip합니다. `--require-gpu`를 사용하면 skip 대신 실패합니다. GPU가 발견된 뒤 발생한 compiler/numeric 오류를 skip으로 바꾸지 않습니다. CPU simulation의 통과를 Metal race-free나 성능의 실증으로 해석하지 않습니다.

## 12. 성능 측정 규약

`bench_fp4.py`는 입력을 고정하여 각 실행에서 새로운 graph를 만들고 eval+synchronize까지 측정합니다. 값이 이미 평가된 동일 output을 재사용하는 timing은 하지 않습니다. **측정값은 wall latency이며 순수 GPU kernel time이 아닙니다.**

CPU pack, prepare/upload, first call, warm median/min/p95를 나눕니다. 모든 candidate는 같은 packed W와 실제 activation dtype 값을 사용하고, CPU oracle accuracy gate를 통과해야 timed 상태로 기록됩니다. Native QMM은 shape/layout/API 제약으로 실패할 수 있으며 unavailable로 남깁니다.

현재 native MLX QMM comparator에는 global_scale 인자가 없으므로 결과에 global을 별도 곱합니다. [S4] 이 방식은 fused reconstruction과 rounding이 달라질 수 있어 단순히 native라는 이유로 golden bitwise baseline으로 삼지 않습니다. `mx.compile` comparator와 full FP32 weight를 쓰는 dense comparator도 의미·memory 조건을 각각 기록합니다.

Dense comparator는 GPU weight 전체를 펼치므로 절대 peak memory가 packed-only 경로의 증거가 아닙니다. 보고서에 이를 명시합니다. native와 packed path의 메모리를 엄밀히 비교하려면 후보마다 별도 프로세스로 측정하는 후속 harness가 필요합니다. 현재 실행 환경에서는 GPU 측정값을 생성하지 않았습니다.

## 13. 후속 최적화 우선순위 — 아직 구현하지 않은 기능

1. 실제 장치에서 테스트를 통과한 뒤 128-lane shared reduction을 hardware subgroup reduction과 비교합니다. Apple SIMD width나 NVIDIA warp mask가 같다고 가정하지 않습니다.
2. packed load를 byte에서 uint32/uint64로 벡터화하되 alignment, K-tail, scale boundary를 함께 증명합니다. 작은 batch에서 register pressure와 occupancy를 확인합니다.
3. Tiled QMM에 matrix-instruction 후보를 추가합니다. FP16 staging으로 인한 추가 rounding을 새 compute profile로 분리하고 FP32 reconstruction 경로와 동일한 정확도라고 부르지 않습니다.
4. MoE에 expert IDs/offsets와 grouped gather 입력을 추가합니다. 초기에는 expert별 선택 kernel로 정확성을 검증한 뒤 routing/gather/dequant/fused GEMM을 묶습니다. 이번 패키지에는 MoE kernel이 없습니다.
5. Tensor/stream bridge, PrivateUse1 및 dedicated CUDA-facing Torch는 이 kernel provider 위에 별도 개발합니다. CUDA로 인식되는 label만 먼저 위장하지 않습니다.

## 14. 저장 공간 계산 예시

자체 canonical 규약에서 W[4096,4096]을 저장할 때 metadata/container overhead를 제외하면:

```text
FP16: 4096*4096*2                         = 33,554,432 B = 32 MiB
NVFP4: payload/2 + block_scales/16 + 4    =  9,437,188 B ≈ 9 MiB
MXFP4: payload/2 + block_scales/32        =  8,912,896 B = 8.5 MiB
```

이것은 저장량 산술이며 GPU bandwidth 사용량 또는 처리속도 측정이 아닙니다. 커널의 scale 재로드·tile 중복읽기·output/activation traffic까지 포함한 throughput은 실측으로 판단합니다.

## 부록 A — 작성한 Small-M Metal 함수 본문

아래 본문은 패키지 소스에서 그대로 삽입했습니다. MLX launcher가 입력 buffer signature와 BM/M/N/K 등의 상수를 생성합니다.

```cpp
// CUMLX v0.3. Inference-only candidate; physical GPU validation required.
// BM=1: GEMV. BM=4: reuse the decoded pair across up to four activation rows.
// One 128-lane workgroup owns one output column and BM rows. No warp assumptions.
unsigned int lane = thread_position_in_threadgroup.x;
unsigned int group = threadgroup_position_in_grid.x;
unsigned int n = group % N;
unsigned int m0 = (group / N) * BM;
threadgroup float partial[BM * 128];
float accum[BM];
for (unsigned int r = 0u; r < BM; ++r) accum[r] = 0.0f;
for (unsigned int k = lane * 2u; k < K; k += 256u) {
    unsigned int pair = wq[n * WP + k / 2u];
    unsigned int scale = ws[n * WSG + k / WG];
    float w0 = cumlx_restore(pair & 15u, scale, wglobal[0], WF);
    float w1 = k + 1u < K ? cumlx_restore(pair >> 4u, scale, wglobal[0], WF) : 0.0f;
    for (unsigned int r = 0u; r < BM; ++r) {
        if (m0 + r < M) {
            float a0 = float(x[(m0 + r) * K + k]);
            accum[r] = metal::fma(a0, w0, accum[r]);
            if (k + 1u < K) {
                float a1 = float(x[(m0 + r) * K + k + 1u]);
                accum[r] = metal::fma(a1, w1, accum[r]);
            }
        }
    }
}
for (unsigned int r = 0u; r < BM; ++r) partial[r * 128u + lane] = accum[r];
threadgroup_barrier(mem_flags::mem_threadgroup);
// All 128 lanes execute every barrier, including lanes with no K contribution.
for (unsigned int step = 64u; step > 0u; step >>= 1u) {
    if (lane < step) {
        for (unsigned int r = 0u; r < BM; ++r)
            partial[r * 128u + lane] += partial[r * 128u + lane + step];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (lane == 0u) {
    for (unsigned int r = 0u; r < BM; ++r) {
        if (m0 + r < M) {
            float result = partial[r * 128u];
            if (HAS_BIAS) result += bias[n];
            if (APPLY_SILU) result = cumlx_silu(result);
            y[(m0 + r) * N + n] = result;
        }
    }
}
```

## 부록 B — 작성한 Tiled Metal 함수 본문

```cpp
// CUMLX v0.3. Inference-only candidate; physical GPU validation required.
// BM=8, BN=32, BK=32, 128 lanes; each lane accumulates two output columns.
// Shared memory: (8*32 + 32*32)*4 = 5120 bytes, independent of full model size.
unsigned int lane = thread_position_in_threadgroup.x;
unsigned int group = threadgroup_position_in_grid.x;
unsigned int tiles_n = (N + 31u) / 32u;
unsigned int m0 = (group / tiles_n) * 8u;
unsigned int n0 = (group % tiles_n) * 32u;
unsigned int row = lane / 16u, col = lane % 16u;
threadgroup float tile_a[8 * 32];
threadgroup float tile_w[32 * 32];
float acc0 = 0.0f, acc1 = 0.0f;
for (unsigned int k0 = 0u; k0 < K; k0 += 32u) {
    for (unsigned int t = lane; t < 8u * 32u; t += 128u) {
        unsigned int r = t / 32u, kk = t % 32u;
        float value = 0.0f;
        if (m0 + r < M && k0 + kk < K) {
            value = float(x[(m0 + r) * K + k0 + kk]);
        }
        tile_a[t] = value;
    }
    for (unsigned int t = lane; t < 32u * 32u; t += 128u) {
        unsigned int n = n0 + t / 32u, k = k0 + t % 32u;
        float value = 0.0f;
        if (n < N && k < K) {
            unsigned int pair = wq[n * WP + k / 2u];
            unsigned int code = (pair >> ((k & 1u) * 4u)) & 15u;
            value = cumlx_restore(code, ws[n * WSG + k / WG], wglobal[0], WF);
        }
        tile_w[t] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // There is no early return on partial M/N tiles: all threads must synchronize.
    for (unsigned int kk = 0u; kk < 32u; ++kk) {
        float a = tile_a[row * 32u + kk];
        acc0 = metal::fma(a, tile_w[col * 32u + kk], acc0);
        acc1 = metal::fma(a, tile_w[(col + 16u) * 32u + kk], acc1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); // Finish all reads before the next K tile overwrites shared memory.
}
if (m0 + row < M) {
    if (n0 + col < N) {
        if (HAS_BIAS) acc0 += bias[n0 + col];
        if (APPLY_SILU) acc0 = cumlx_silu(acc0);
        y[(m0 + row) * N + n0 + col] = acc0;
    }
    if (n0 + col + 16u < N) {
        if (HAS_BIAS) acc1 += bias[n0 + col + 16u];
        if (APPLY_SILU) acc1 = cumlx_silu(acc1);
        y[(m0 + row) * N + n0 + col + 16u] = acc1;
    }
}
```

전체 CUDA 대응 본문과 GPU block encoder는 `src/cumlx/kernels/`에 있습니다. 원본 자료 S1–S9는 [SOURCES.md](../SOURCES.md)를 참조합니다.
