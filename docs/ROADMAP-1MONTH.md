# CUMLX 1개월 로드맵 — 부족한 점, 개선 방향, 테스팅 마일스톤

작성일: 2026-11-02 · 대상 기간: 2026-11-02 ~ 2026-11-30 (4주 + 릴리스일)
기준 커밋: `22fe8ce` (v0.3.0.dev0)

> **목표 비전:** CUDA를 전제로 만든 AI 모델·오픈소스를 **명령어 한 번**으로 MLX용으로 변환해 Apple Silicon에서 효율적으로 실행한다.
>
> **현재 위치:** NVFP4/MXFP4 packed 커널 **연구실(kernel lab)** 단계. CPU 기준 구현과 GPU 커널 소스는 있지만, 변환 명령어·모델 로더·모델 실행 경로는 없고 GPU 실행은 한 번도 검증하지 않았다.

---

## 0. 요약

| 항목 | 판단 |
|---|---|
| 가장 큰 격차 | "한 번에 변환"할 **제품 층(CLI·체크포인트 어댑터·모델 그래프)이 전혀 없음** |
| 가장 큰 리스크 | GPU 커널 **미검증**(컴파일조차 안 해봄). 그 위에 무엇을 쌓아도 사상누각 |
| 차별점 부재 | `mlx-lm`의 `convert -q`가 이미 "HF → MLX 양자화"를 한 줄로 해준다. CUMLX만의 가치를 정의해야 함 |
| 1개월 현실적 목표 | **"NVIDIA용 NVFP4 체크포인트(및 일반 HF LLM)를 `cumlx convert` 한 줄로 MLX에서 실행"** + **CUDA 의존성 스캐너** + **실제 GPU 검증 증거** |
| 1개월 비목표 | 임의 `.cu` 커스텀 커널 자동 번역, `torch.cuda` 위장 런타임, 학습/backward, 멀티 GPU |

---

## 1. 현재 상태 분석

### 1.1 구성 (총 약 2,400줄)

| 층 | 파일 | 상태 |
|---|---|---|
| FP4 코덱·직렬화 | `src/cumlx/quant.py` | CPU 검증 완료. 단일 행렬 `.npz` 컨테이너만 지원 |
| 고정밀 참조 | `src/cumlx/reference.py` | float64 oracle. 완료 |
| 디스패치 | `src/cumlx/planning.py` | M 값만 보는 고정 휴리스틱 |
| 스케줄 시뮬레이터 | `src/cumlx/simulation.py` | 인덱스·커버리지 검사용(순수 Python 루프) |
| MLX 런처 | `src/cumlx/mlx_backend.py` | 작성됨, **하드웨어 미실행** |
| 커널 | `src/cumlx/kernels/*.metal, *.cu` (8종 × 2) | 작성됨, **컴파일·실행 미검증** |
| 커널 생성기 | `scripts/generate_kernels.py` | CI에서 byte 일치 검사 |
| 벤치마크 | `benchmarks/bench_fp4.py` | 정확성 gate 포함, 실행 기록 없음 |
| CI | `.github/workflows/ci.yml`, `gpu.yml` | CPU CI 정상, GPU는 self-hosted runner 미등록 |

### 1.2 실측 확인 (2026-11-02, Linux x86_64, Python 3.11, MLX 없음)

```text
python -m pytest -q -m 'not gpu'
109 passed, 35 deselected in 3.08s
```

35개 GPU 테스트는 한 번도 실행되지 않았다. 즉 **커널 정확성에 대한 증거는 0**이다.

### 1.3 잘 된 점 (유지할 것)

- 수치 계약이 엄밀하다: nibble 순서, scale 해석, 홀수 K 패딩, `-0` 보존, 포화 처리 등을 문서와 테스트로 고정.
- "검증 안 한 것을 검증했다고 말하지 않는" 문화: `allow_unvalidated=True`, `--require-gpu`, CPU fallback 금지.
- 커널 소스를 생성기로 관리하고 CI에서 drift를 막는다.
- float64 oracle과 tolerance 기반 비교로 커널 오차와 양자화 오차를 분리한다.

---

## 2. 비전 대비 격차

"CUDA 전용 모델 → 명령어 한 번 → MLX에서 효율적으로 실행"을 단계로 나누면 다음과 같다.

| 단계 | 필요 기능 | 현재 | 격차 |
|---|---|---|---|
| ① 진단 | 대상 저장소/모델의 CUDA 의존성 분석 (`torch.cuda`, `.cuda()`, flash-attn, bitsandbytes, triton, xformers, 커스텀 `.cu`) | 없음 | **전무** |
| ② 입력 | HF Hub ID/로컬 경로, `safetensors`, `config.json`, sharded index 로딩 | 자체 `.npz` 단일 행렬만 | **전무** |
| ③ 가중치 변환 | 모델 전체 Linear 일괄 양자화, 사전 양자화 체크포인트(NVIDIA ModelOpt NVFP4 등) 가져오기 | 행렬 1개 CPU 양자화 | 큼 |
| ④ 저장 포맷 | 멀티 텐서 컨테이너, 레이어 메타데이터, mmap/lazy 로딩 | 단일 행렬 `.npz` | 큼 |
| ⑤ 모델 그래프 | 아키텍처(Llama/Qwen/Mistral 등), attention, RoPE, KV cache, sampling | 없음 (Linear/RMSNorm/SiLU 커널만) | **전무** |
| ⑥ 커널 | Packed FP4 GEMV/GEMM, norm, 활성화 | 소스만 있음, 미검증 | 중간 |
| ⑦ 실행 UX | `cumlx convert`, `cumlx run/generate`, `cumlx doctor` | `[project.scripts]` 없음 | **전무** |
| ⑧ 검증 | 커널 정확성 → 레이어 → 모델(perplexity/토큰 일치) → 성능 | CPU 단위 테스트만 | 큼 |

---

## 3. 부족한 점 (상세)

### 3.1 제품·포지셔닝

1. **차별점이 정의되지 않았다.** `mlx-lm`(`mlx_lm.convert --quantize`)이 이미 HF 모델을 한 줄로 MLX 양자화 모델로 바꾼다. MLX 자체도 `mx.quantize`/`quantized_matmul`에 `mxfp4`/`nvfp4` 모드를 제공한다(SOURCES.md S3, S4). CUMLX가 "또 하나의 FP4 커널"이면 존재 이유가 약하다.
   - 후보 차별점 (권장 순):
     - (a) **CUDA 전용으로 배포된 사전 양자화 체크포인트**(NVIDIA ModelOpt NVFP4 등)를 **재양자화 없이** Mac에서 실행. "CUDA용 모델을 MLX로"라는 비전과 정확히 맞는다.
     - (b) **CUDA 의존성 스캐너 + 자동 패치 제안**: 일반 PyTorch 저장소에서 무엇이 막히는지 보고하고, `device="cuda"` → `mps`/MLX 경로로 바꿀 수 있는 부분을 자동 처리.
     - (c) global scale을 포함한 NVFP4 의미의 **정확한 재현**(MLX native QMM은 `global_scale` 인자가 없어 출력에 따로 곱함 — `mlx_backend.py:native_linear`).
2. **"명령어 한 번"에 해당하는 진입점이 없다.** `pyproject.toml`에 `[project.scripts]`가 없다.
3. 패키지 이름이 `cumlx-lab`, README 첫 문장이 "커널 구현 패키지"라서 사용자가 기대하는 "변환 도구"와 다르다.

### 3.2 모델 지원

4. 체크포인트 로더가 자체 `.npz`(`quant.py:load_fp4`) 하나뿐이다. `safetensors`, sharded index, `config.json`을 읽지 못한다.
5. 컨테이너가 **행렬 1개 = 파일 1개**다. 수백 개 Linear를 가진 모델을 담을 포맷이 없다.
6. Attention, RoPE, softmax, embedding, KV cache, MoE 라우팅이 없다. 지금 커널만으로는 어떤 모델도 끝까지 실행할 수 없다.
7. `MLXBackend.linear`가 **2D 입력만** 받는다(`mlx_backend.py:195`). 실제 모델은 `[B, L, D]` 3D 텐서를 쓴다.
8. 출력이 항상 float32다(`mlx_backend.py:170`). fp16/bf16 모델 파이프라인에서는 매 레이어마다 cast가 추가되어 메모리·대역폭을 낭비한다.
9. 크기 한계: `N*K ≤ 2^31-1`(`quant.py:150`, `planning.py:31`). 대형 vocab의 `lm_head`(예: 256k × 16k)는 넘는다. 분할 처리나 int64 인덱스 프로파일이 필요하다.

### 3.3 커널·런타임

10. **GPU 컴파일조차 안 해봤다.** Metal 셰이더 문법 오류, MLX 템플릿 상수 이름 불일치 같은 1차 실패가 남아 있을 가능성이 높다.
11. MLX 버전을 **정확히 일치**(`== 0.32.2`)하지 않으면 생성자가 예외를 던진다(`mlx_backend.py:96`). 사용자 환경에서 MLX가 한 번만 올라가도 도구 전체가 동작하지 않는다. 호환 범위 + 경고 정책으로 바꿔야 한다.
12. `backend.quantize`가 `.item()`으로 **동기화**한다(`mlx_backend.py:233`). 토큰 생성 루프에서 매 스텝 CPU-GPU 왕복이 생겨 lazy graph 이점이 사라진다.
13. 성능 관점의 미완성 (설계 문서 §13에서 스스로 인정):
    - 128-lane shared memory tree reduction만 사용 (SIMD/warp reduction 미사용).
    - packed weight를 **byte 단위**로 읽음 (uint32/uint64 벡터 load 없음). `fp4_tiled`는 원소마다 같은 byte를 다시 읽고 decode한다.
    - GEMV는 출력 열마다 workgroup 1개이며 activation `x`를 shared memory에 staging하지 않는다.
    - Tiled GEMM이 scalar FMA 기반 (simdgroup matrix / Tensor Core 미사용) → prefill에서 native보다 느릴 가능성이 크다.
    - GPU quantizer는 thread 하나가 block 하나를 맡아 인접 thread의 메모리 접근이 G 간격으로 떨어진다(비연속 접근).
14. 디스패치가 M만 본다(`planning.py:33`). K/N 크기·장치 세대를 반영한 autotune이 없다.
15. W4A4에서 activation 양자화와 GEMM이 별도 launch다 (fusion 없음).

### 3.4 검증·CI

16. GPU CI가 self-hosted runner를 전제로 하지만 **등록된 runner가 없다.** 사실상 GPU 테스트는 영원히 안 도는 구조다.
17. GPU 없이도 할 수 있는 검증을 하지 않는다:
    - CUDA: `nvcc`/NVRTC는 GPU 없이 **device 코드 컴파일**이 가능하다. MLX가 생성하는 signature를 흉내 낸 wrapper로 `.cu` 본문을 컴파일하면 문법·타입 오류를 CPU CI에서 잡을 수 있다.
    - Metal: macOS runner에서 `xcrun metal`로 오프라인 컴파일 가능 여부를 확인한다 (GitHub 호스팅 macOS runner의 GPU 실행 가능 여부는 **미확인** — W1에 확인).
18. 모델 수준 지표(perplexity 차이, greedy 토큰 일치율, tok/s, peak memory)를 측정하는 harness가 없다.
19. 성능 회귀 감지 체계가 없다 (기준선 JSON 저장·비교 없음).

### 3.5 코드 품질·개발 경험

20. Ruff 규칙이 문법 오류 수준(E9/F63/F7/F82)뿐이다. `planning.py`, `reference.py`는 한 줄에 여러 문장을 넣는 압축 스타일이라 기여자가 읽기 어렵다.
21. 타입 검사(mypy/pyright)가 없다. `py.typed`를 배포하면서 타입을 검증하지 않는다.
22. CPU `quantize`가 단일 스레드 NumPy이고 전체 FP32 가중치를 메모리에 올린다. 70B급 모델 변환 시 시간·메모리가 문제가 된다 (layer 단위 streaming 필요).
23. 문서가 "하지 않은 것" 목록에 치우쳐 있고, 사용자 관점의 Quickstart(모델 하나 변환해서 돌리기)가 없다.

---

## 4. 개선 방향

### 4.1 원칙

1. **바퀴를 다시 만들지 않는다.** 모델 그래프·tokenizer·sampling은 `mlx-lm`을 재사용하고, CUMLX는 *변환기 + FP4 Linear 제공자 + 진단기*에 집중한다.
2. **검증 먼저, 최적화 나중.** 실제 GPU 정확성 gate를 통과하기 전에는 커널 최적화를 시작하지 않는다 (기존 원칙 유지).
3. **fallback은 명시적으로.** CUMLX 커널이 해당 shape/장치에서 미검증이면 MLX native QMM으로 *명시적으로* 내려가고, 보고서에 그 사실을 기록한다. 사용자 입장에서 "일단 돌아가는" 경험을 확보한다.
4. 기존의 정직한 증거 문화(`validation/REPORT.md`)를 모델 수준까지 확장한다.

### 4.2 목표 아키텍처

```text
cumlx scan <repo|hf-id>      ── CUDA 의존성 보고서 (JSON + 사람이 읽는 표)
        │
cumlx convert <hf-id|path>   ── ① config/safetensors 로딩 (lazy, shard 단위)
        │                       ② 어댑터 선택: dense FP16/BF16 → CUMLX 양자화
        │                                     ModelOpt NVFP4  → 재양자화 없이 정규화
        │                       ③ 멀티 텐서 컨테이너 저장 (safetensors + 메타데이터)
        ▼
cumlx run / generate         ── mlx-lm 아키텍처 로드
                                 └ nn.Linear → CumlxFP4Linear 교체
                                     ├ 검증된 shape → CUMLX 커널
                                     └ 그 외        → MLX native QMM (명시적, 로그 기록)
cumlx doctor                 ── 환경·MLX 버전·GPU·커널 자가 진단
cumlx bench                  ── 레이어/모델 벤치마크, 기준선 비교
```

### 4.3 1개월 범위 결정

| 포함 (MVP) | 제외 (백로그) |
|---|---|
| Apple Silicon + Metal 우선 | CUDA 백엔드 성능 작업 (정확성 검증만) |
| Llama 계열 dense 아키텍처 1~2종 (예: Llama 3.x, Qwen 2.5/3) | MoE, 멀티모달, diffusion, whisper |
| dense HF → NVFP4/MXFP4 변환 | NxFP4/NF4/INT4 |
| ModelOpt NVFP4 체크포인트 import (형식 일치 여부를 W1에 검증) | TransformerEngine 2D/swizzled layout |
| `scan` 정적 분석 (보고만) | 임의 `.cu` → Metal 자동 번역, `torch.cuda` 위장 |
| 추론 전용 | 학습, backward, LoRA |

---

## 5. 주차별 마일스톤

각 주는 **Exit Gate**를 통과해야 다음 주로 넘어간다. Gate를 못 넘으면 범위를 줄이고 다음 주 첫 이틀을 gate 완료에 쓴다.

### W1 (11/02 ~ 11/08) — 실제 GPU 검증과 기반 정리

**목표:** "커널이 실제로 돈다"는 첫 증거를 만든다.

| 작업 | 산출물 |
|---|---|
| Apple Silicon 장비 1대를 self-hosted runner(`metal` label)로 등록 | `gpu.yml` 첫 실행 기록 |
| 35개 GPU 테스트 실행, 실패 원인 분류(컴파일/런처/수치) 후 수정 | `validation/REPORT.md` 갱신 |
| CUDA 본문을 MLX signature wrapper로 감싸 `nvcc -c`(GPU 없이) 컴파일하는 CI job 추가 | `ci.yml`에 `cuda-compile` job |
| MLX 버전 정책 변경: `==` → 테스트된 범위 + 범위 밖은 경고 | `mlx_backend.py` |
| `bench_fp4.py`로 M ∈ {1, 4, 32, 512}, (N,K) ∈ {(4096,4096), (11008,4096), (4096,11008)} 측정 | `validation/bench-w1.json` |
| ModelOpt NVFP4 체크포인트 1개를 받아 텐서 이름·dtype·nibble 순서·scale 의미를 CUMLX 계약과 대조 | `docs/ADAPTER-MODELOPT.md` (일치/불일치 표) |

**테스팅 마일스톤 T1**
- [ ] Metal에서 GPU 테스트 35/35 통과 (skip 0, `--require-gpu`)
- [ ] `test_gpu_decode_all_bytes`, `test_gpu_quantizer`는 bitwise 일치
- [ ] CUDA `.cu` 본문 8개 전부 device compile 성공 (CPU CI)
- [ ] 벤치마크 JSON에 CUMLX vs MLX native QMM vs dense 비교 수치 기록 (정확성 gate 통과한 후보만)

**Exit Gate:** Metal 정확성 테스트 전부 통과. 실패하면 W2의 모델 통합은 MLX native QMM fallback으로만 진행.

---

### W2 (11/09 ~ 11/15) — 변환 파이프라인 (`cumlx convert`)

**목표:** HF 모델 하나를 한 줄로 CUMLX 포맷으로 바꾼다.

| 작업 | 산출물 |
|---|---|
| `[project.scripts] cumlx = "cumlx.cli:main"` 추가, `convert`/`doctor` 서브커맨드 | `src/cumlx/cli.py` |
| `safetensors` + sharded index lazy 로더 (shard 단위 streaming, 전체 FP32 상주 금지) | `src/cumlx/io/hf.py` |
| 멀티 텐서 컨테이너: safetensors 기반, 레이어별 format/shape/global_scale/checksum 메타데이터 | `src/cumlx/io/container.py` |
| 양자화 정책: 어떤 Linear를 FP4로 할지 (embedding·lm_head·norm은 기본 제외, 설정 가능) | `src/cumlx/policy.py` |
| 어댑터 1: dense FP16/BF16 → NVFP4/MXFP4 | `src/cumlx/adapters/dense.py` |
| 어댑터 2: ModelOpt NVFP4 → CUMLX 정규화 (W1 대조 결과가 일치할 때만) | `src/cumlx/adapters/modelopt.py` |
| CPU `quantize` 가속: block 루프 벡터화 + 레이어 단위 병렬 처리 | `quant.py` |

**테스팅 마일스톤 T2**
- [ ] 작은 모델(예: 0.5B~1B 공개 모델) 변환 → 재로드 → 모든 텐서 checksum 일치
- [ ] 변환 전후 각 Linear의 CPU oracle 출력 비교: 상대 오차 분포 기록 (FP4 양자화 오차 기준선)
- [ ] ModelOpt 어댑터: 원본 NVFP4 바이트를 **그대로** 보존 (재양자화 없음, payload/scale bitwise 동일)
- [ ] 7B 모델 변환 시 peak RSS ≤ 원본 BF16 크기의 1.5배, 시간 측정 기록
- [ ] 손상 파일·잘못된 shape·지원하지 않는 dtype에 대해 명확한 오류 메시지 (negative test)

**Exit Gate:** `cumlx convert <소형 모델>`이 끝까지 성공하고 재로드 검증을 통과.

---

### W3 (11/16 ~ 11/22) — 모델 실행 (`cumlx run`)과 정확도

**목표:** 변환한 모델로 실제 텍스트를 생성하고, 품질을 숫자로 증명한다.

| 작업 | 산출물 |
|---|---|
| `CumlxFP4Linear(nn.Module)`: 3D 입력 reshape, bias, 출력 dtype 선택(fp16/bf16) | `src/cumlx/nn.py` |
| `mlx-lm` 모델 로드 후 Linear 교체 훅, 미검증 shape는 native QMM으로 명시적 fallback | `src/cumlx/integrations/mlx_lm.py` |
| `cumlx run <변환 경로> --prompt ...` / `cumlx generate` | `cli.py` |
| 커널 출력 dtype 템플릿화 (float32 고정 해제) | `kernels/*`, 생성기 |
| `backend.quantize`의 `.item()` 동기화 제거 (검증은 debug 모드에서만) | `mlx_backend.py` |
| 모델 평가 harness: perplexity(wikitext-2 일부), greedy 토큰 일치율 | `benchmarks/eval_model.py` |

**테스팅 마일스톤 T3**
- [ ] 레이어 단위: 실제 모델의 모든 Linear shape에 대해 CUMLX 커널 vs CPU oracle tolerance 통과
- [ ] 동일 FP4 가중치에서 CUMLX 경로 vs MLX native QMM 경로의 logits 차이 ≤ 기준 (예: max abs 1e-2, 정해진 프롬프트 세트)
- [ ] Perplexity: BF16 원본 대비 증가율 기록. 목표 NVFP4 ≤ +5%, MXFP4 ≤ +8% (모델별로 조정, **목표치이며 보장 아님**)
- [ ] Greedy decode 128 토큰: 동일 FP4 가중치의 native QMM 경로와 토큰 일치율 ≥ 95%
- [ ] 출력 dtype fp16/bf16에서도 W1 GPU 테스트 전부 재통과
- [ ] 엔드투엔드 smoke: `cumlx convert && cumlx run`이 CI(GPU runner)에서 통과

**Exit Gate:** 소형 모델 1개가 CUMLX 경로로 의미 있는 텍스트를 생성하고 perplexity 목표 이내.

---

### W4 (11/23 ~ 11/29) — 성능, 진단기, 릴리스 준비

**목표:** "효율적"이라는 말을 측정값으로 뒷받침하고, `scan`으로 비전의 진단 단계를 연다.

| 작업 | 산출물 |
|---|---|
| GEMV 최적화 1순위: uint32 벡터 load + SIMD reduction (`simd_sum`) | `fp4_gemv.metal` |
| 디스패치 테이블: W1/W3 벤치마크로 shape별 최속 후보 선택 (CUMLX vs native) | `planning.py` |
| `cumlx scan <path|hf-id>`: AST 기반 CUDA 의존성 탐지 (`torch.cuda.*`, `.cuda()`, `device='cuda'`, `flash_attn`, `bitsandbytes`, `triton`, `xformers`, `apex`, `*.cu`, `load_inline`) + 대체 경로 제안 | `src/cumlx/scan.py` |
| 성능 회귀 기준선 JSON 저장·비교 스크립트 | `benchmarks/compare.py` |
| 사용자 Quickstart 중심 README 재작성, 패키지 이름 검토 | `README.md` |

**테스팅 마일스톤 T4**
- [ ] Decode(M=1) tok/s: CUMLX 경로 ≥ MLX native QMM 경로의 0.9배 (못 미치면 디스패치가 native를 선택하도록 하고 그 사실을 기록)
- [ ] Peak memory: FP4 모델 실행 시 BF16 대비 ≤ 35% (가중치 기준)
- [ ] 최적화된 GEMV가 W1 정확성 테스트 전부 재통과 (최적화가 정확성을 깨지 않음)
- [ ] `scan` 정답 세트: 공개 저장소 10개에 수동 라벨링한 CUDA 의존성 목록 대비 recall ≥ 90%, precision ≥ 80%
- [ ] 성능 회귀 비교: 기준선 대비 10% 이상 느려지면 실패하는 GPU CI step

**Exit Gate (11/30 릴리스):** 아래 "완료 정의"를 모두 만족하면 `v0.4.0` 태그.

### 완료 정의 (Definition of Done, 11/30)

```bash
pip install cumlx-lab
cumlx doctor
cumlx scan ./some-cuda-project
cumlx convert <hf-model-id> --format nvfp4 -o ./model-cumlx
cumlx run ./model-cumlx --prompt "Hello"
```

- 위 다섯 줄이 Apple Silicon Mac에서 동작한다.
- `validation/REPORT.md`에 커밋 SHA별 GPU 테스트 수, perplexity, tok/s, peak memory가 기록되어 있다.
- 미검증 경로는 문서와 런타임 로그 모두에서 "미검증"으로 표시된다.

---

## 6. 테스팅 전략

### 6.1 테스트 피라미드

| 층 | 무엇 | 어디서 | 빈도 |
|---|---|---|---|
| L0 단위 | 코덱, 직렬화, 어댑터 파싱, CLI 인자 | CPU CI | 모든 PR |
| L1 컴파일 | CUDA device compile(`nvcc`), Metal offline compile(가능 시) | CPU CI / macOS runner | 모든 PR |
| L2 커널 정확성 | 기존 35개 + 실제 모델 shape, dtype 3종, format 2종 | GPU runner | main push, 수동 |
| L3 레이어 | Transformer block 1개: CUMLX vs native vs CPU oracle | GPU runner | main push |
| L4 모델 | perplexity, greedy 토큰 일치율, 엔드투엔드 CLI | GPU runner | 일 1회 / 릴리스 |
| L5 성능 | tok/s, prefill latency, peak memory, 기준선 비교 | GPU runner | 일 1회 / 릴리스 |

### 6.2 정확성 기준

| 비교 | 기준 | 근거 |
|---|---|---|
| decode / quantizer GPU vs CPU | bitwise 동일 | 결정적 연산, 이미 테스트로 고정 |
| linear GPU vs float64 oracle | `atol=2e-4, rtol=3e-4` (기존) + K에 비례한 오차 상한 추가 검토 | reduction 순서 차이 |
| CUMLX vs native QMM (동일 FP4 가중치) | logits max abs ≤ 1e-2 | global scale 적용 위치 차이 |
| 모델 perplexity vs BF16 원본 | NVFP4 ≤ +5%, MXFP4 ≤ +8% (목표) | FP4 양자화 오차 |

### 6.3 성능 측정 규칙

- 기존 규칙 유지: 매 반복 새 graph, `mx.eval` + synchronize, 정확성 gate를 통과한 후보만 기록.
- 장비·OS·MLX 버전·커밋 SHA를 결과 JSON에 반드시 포함.
- 모델 수준은 prefill(512 토큰)과 decode(128 토큰)를 분리해 보고.
- 메모리는 후보마다 **별도 프로세스**로 측정 (설계 문서 §12에서 지적한 한계 해소).

---

## 7. 리스크와 대응

| 리스크 | 가능성 | 영향 | 대응 |
|---|---|---|---|
| Apple Silicon GPU runner를 확보하지 못함 | 중 | 치명적 | W1 첫날 확보. 불가하면 개발자 Mac에서 수동 실행 후 원시 로그를 커밋하는 절차로 대체 |
| 커널이 대량으로 실패 | 중 | 큼 | W2·W3는 MLX native QMM fallback으로 진행해 파이프라인 일정을 지킨다. 커널 수정은 병행 |
| CUMLX 커널이 native보다 느림 | 높음 | 중 | 디스패치가 native를 선택하게 한다. 차별점은 속도가 아니라 *변환 범위(ModelOpt import, scan)*에 둔다 |
| ModelOpt NVFP4 layout이 CUMLX 계약과 다름 | 중 | 중 | W1에 대조. 다르면 정규화 어댑터(재배열, 재양자화 없이)를 작성하거나 W2 범위에서 제외 |
| MLX API 변경 | 중 | 중 | 버전 범위 정책 + 주간 최신 MLX 대상 nightly job |
| 7B 이상 변환 시 메모리 부족 | 중 | 중 | shard streaming, 레이어 단위 처리 |
| 범위 확장 압력 (MoE, diffusion 등) | 높음 | 중 | 4.3의 제외 목록을 고정. 요청은 백로그로 이동 |

---

## 8. 1개월 이후 백로그 (우선순위 순)

1. MoE (expert gather + grouped GEMM) — Qwen-MoE, Mixtral, gpt-oss 계열
2. Tiled GEMM에 simdgroup matrix 명령 적용 (prefill 성능)
3. W4A4 activation 양자화 fusion
4. `scan` → 자동 패치(`--fix`): `device='cuda'` 치환, flash-attn → `mx.fast.scaled_dot_product_attention` 대체 안내
5. 비 LLM 모델 (Whisper, Stable Diffusion 계열) 변환 템플릿
6. CUDA 백엔드 성능 작업과 NVIDIA 실장비 검증
7. 대형 행렬 int64 인덱스 프로파일 (`N*K > 2^31`)
8. PyTorch 사용자를 위한 브리지 (설계 v0.2의 G3/G4 — 별도 프로젝트 규모)
