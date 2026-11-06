# CUMLX 1개월 로드맵 — CUDA→MLX 포팅 동등성 검증기

작성일: 2026-11-02 (개정 6: CUDA 측정, 최신 모델, 포팅 검토 반영) · 기간: 2026-11-02 ~ 2026-11-30
베타 `v0.4.0b1`, 정식 `v0.4.0`, 수정 `v0.4.1` (모두 2026-11-02) · 기준 커밋: `22fe8ce` (v0.3.0.dev0)

> **다음 계획:** [플랫폼 로드맵](ROADMAP-PLATFORM.md)
>
> **한 줄 정의:** CUDA로 만든 모델을 *누가 어떻게* MLX로 옮겼든, **원본과 같은 동작을 하는지 레이어 단위로 증명하고, 다르면 어디서 처음 달라졌는지 찾아 주는 도구.**

## 진행 현황 (2026-11-02)

코드와 측정으로 끝낼 수 있는 마일스톤은 모두 끝냈다. 아래 **미완료**는 사람의 결정이나 권한이 필요한 항목이다. 사용법과 측정은 [CLI 문서](CLI.md)에, 원본 결과는 [검증 보고서](../validation/REPORT.md)에 있다.

| 주차 | 항목 | 상태 | 근거 |
|---|---|---|---|
| 베타 | D1~D7, 베타 Gate | ✅ `v0.4.0b1` | 아래 Gate. 실제 모델 Qwen2.5-0.5B, SmolLM2-135M |
| 베타 | Apple Silicon(Metal) | ✅ | M5 Pro. selftest 11/11, e2e와 GPU 테스트 skip 0 |
| 신규 | TF32 대응 | ✅ | MLX 기본 TF32가 false alarm을 일으킴. `--device`, `--tf32`, 실측 기록 |
| W2 | 다중 입력 isolated 재생 | ✅ | attention·decoder layer. F7·F8이 `isolated, high` |
| W2 | decode 단계 capture | ✅ | `--decode-steps`, 포팅본 자체 KV cache 재생, F11 |
| W2 | 양자화 기대 오차 oracle | ✅ | `--weights-from-mlx`. 모든 dtype에 임계값 1e-4 |
| W2 | 정적 가중치 diff | ✅ | `cumlx weights`. 0.5B 모델 약 3.3초 |
| W2 | MoE | ✅ | 라우터·expert 분리 판정, expert별 scale, F9. 작은 Mixtral 공개 모델 2개와 tiny random 모델. 대형 MoE는 미실행 |
| W2 보강 | transformers 4 expert별 `ModuleList` 재생 | ✅ (2026-11-03) | 이전에는 `unverified`. tiny Qwen3-MoE·Mixtral(transformers 4.57.6)에서 커버리지 100%, 한 expert 결함과 F4·F9·F10 지목. CI가 transformers 4를 별도 디렉터리에 설치해 실행 |
| W2 보강 | 학습된 실제 MoE (Granite 3.1 1B-A400M) | ⏳ 중단 | 체크포인트 다운로드와 `weights`까지 실행: 218개 중 170 통과, 48 `unverified`(라우터와 expert의 모듈 경계 차이로 추정, 원인 미확인). capture·diff는 하지 않음 |
| W2 | 임계값 보정 | ✅ | 깨끗한 포팅 10종 측정. 실행별 중앙값 기반 |
| W3 | 공개 포팅본 3~5개 | ✅ | 10개 실행, 9개 ok, 1개 로드 불가. Dory NVFP4 repack 비교 |
| W3 | `conform --checkpoint` | ✅ | 실제 Dory NVFP4, 텐서 288개 일치 |
| W3 | HTML 보고서 | ✅ | `report.html`. 외부 자원 없음 |
| W3 | 자체 커널 dogfooding | ✅ | GPU 테스트 36개, `check_op`으로 Metal 커널과 PyTorch 비교 |
| W3 | 실제 불일치 발견 | ✅ | mlx-lm이 transformers 5의 `rope_parameters`를 놓침. [근거](../validation/upstream/README.md) |
| W3 | **upstream 보고** | ⛔ 미완료 | mlx-lm 정책이 AI의 게시물 작성을 금지. 사용자가 직접 작성해야 함 |
| W3 | CUDA golden 측정 | ✅ | V100, Qwen2.5-0.5B. 반복 실행 동일, CPU와 최대 1.9e-5. [결과](CUDA-GOLDEN.md) |
| W4 | GitHub Action | ✅ 저장소 내부 | `action.yml`, `examples/port-ci`, CI의 `Action test`. 깨끗한 포팅 통과, 결함 포팅 실패 |
| W4 | **별도 예제 저장소 적용** | ✅ 완료 | 공개 Action 사용. PR 이벤트에서 깨끗한 PR 통과, 결함 PR 실패. 예전에 이벤트가 돌지 않던 원인은 못 찾음. `validation/t4-action-example.json` |
| W4 | MCP 서버와 에이전트 스킬 | ✅ | `cumlx-mcp`, `skills/cumlx-verifier` |
| W4 | `check_op` API | ✅ | `cumlx.check_op` |
| W4 | 부분 기록 | ✅ | `--layers`, `W_PARTIAL_TRACE` |
| W4 | 문서 | ✅ | 사용 사례 중심 README, [trace 명세](TRACE-FORMAT.md) |
| 추가 | 최신 Hugging Face 모델 측정 | ✅ | 12개: 7개 ok, 4개 비교 불가(부분 체크포인트), 1개 mlx-lm 로드 불가. [결과](../validation/latest-models.json). 측정 후 삭제 |
| 추가 | CUDA 전용 프로그램 사용 검토 | ✅ | `cumlx scan`, `cumlx port`, [검토 문서](PORTING-CUDA-MODELS.md). Isaac Sim은 Mac 직접 실행 불가, 정책 추론 경로는 직접 확인 |
| 릴리스 | **PyPI 게시** | ⛔ 미완료 | PyPI 토큰이 이 장비에 없음. 이름 `cumlx-lab`은 비어 있음. GitHub 태그와 Release까지 수행 |
| 릴리스 | **보고서와 명세의 외부 공개** | ✅ 완료 | 2026-11-04에 저장소를 공개. 공개 전에 커밋 기록의 로컬 장비 정보를 지움 |

---

## 0. 방향을 바꾼 이유

개정 1에서는 "변환기(`cumlx convert`) + ModelOpt NVFP4 import + CUDA 의존성 스캐너"를 제안했다. 조사 결과 이 영역은 이미 여러 프로젝트가 진행 중이다.

| 영역 | 이미 존재하는 것 |
|---|---|
| HF LLM → MLX 변환·실행 | mlx-lm (`convert`, `load`), mlx-vlm |
| NVIDIA NVFP4 체크포인트 → MLX | mlx-lm·mlx-vlm·oMLX의 계열별 지원, HF의 NVFP4→MLX repack 모음 |
| PyTorch 모델 포팅 | mlx-porting-skill, Claude 포팅 스킬, 마이그레이션 가이드 |
| CUDA 커널 → Metal | kernelport(0.1), K-Search 연구, woof의 프로젝트별 번역 |
| CUDA 전용 op 대체 | mpsops, flash-attn-mps (PyTorch MPS 대상) |

반면 **변환 결과가 맞는지 증명하는 도구는 비어 있다.** 실제로 변환 쪽에서는 버그가 계속 발견되고 있다.

- mlx-vlm: ModelOpt NVFP4 AWQ 로딩·입력 스케일링 수정 (PR #2582)
- oMLX / mlxcel: MoE expert별 global scale 로딩 수정
- woof: 222개 커널 번역·컴파일 성공, 그러나 일부 커널은 값이 틀림
- mlx-serve: 번역 계층이 eval 시점에야 실패

CUMLX가 이미 가진 자산은 커널 속도가 아니라 **검증 체계**다.

- `quant.py`: NVFP4/MXFP4 bit-정확 코덱 (nibble 순서, scale 해석, `-0`, 포화 처리)
- `reference.py`: float64 oracle
- "skip은 pass가 아니다", "암묵적 fallback 금지" 원칙

따라서 CUMLX는 **변환기와 경쟁하지 않고, 모든 변환기 뒤에 붙는 검증 계층**이 된다.

---

## 1. 제품 정의

### 1.1 핵심 명령 (베타 범위)

```bash
# ① 원본(PyTorch, CUDA 또는 CPU)에서 모듈별 입출력을 golden trace로 기록
cumlx capture <hf-id|path> --prompts prompts.txt -o golden/

# ② MLX 포팅본을 같은 입력으로 실행하고 레이어별로 비교
cumlx diff golden/ <mlx-model-path> -o report/

# ③ FP4 형식 정합성 검사 (MLX 내장 dequantize, 체크포인트 bytes)
cumlx conform --target mlx
cumlx conform --checkpoint <path>
```

### 1.2 diff의 두 가지 모드

| 모드 | 방식 | 용도 |
|---|---|---|
| **chain** (전체 실행) | 같은 token id를 MLX 모델에 넣고 모든 모듈 출력을 golden과 비교 | 최종 출력이 같은지 확인. 오차가 누적되어 원인 위치는 흐려짐 |
| **isolated** (모듈 단독 재생) | 각 모듈에 **golden 입력**을 직접 넣고 출력만 비교 | 오차 누적 없이 **잘못된 모듈을 정확히 지목**. 이 도구의 핵심 기능 |

### 1.3 판정 지표

모듈마다 아래 값을 계산하고 dtype별 기본 임계값과 비교한다.

| 지표 | 정의 |
|---|---|
| `rel_l2` | ‖mlx − golden‖₂ / ‖golden‖₂ |
| `max_abs` | max \|mlx − golden\| |
| `cosine` | 두 출력 벡터의 cosine 유사도 |
| `first_divergence` | 실행 순서상 처음으로 임계값을 넘은 모듈 (isolated 모드 기준) |

기본 임계값 (베타 초기값. W2에서 실측으로 보정):

| 비교 조건 | `rel_l2` 상한 |
|---|---|
| fp32 ↔ fp32 | 1e-5 |
| bf16/fp16 포팅 | 1e-2 |
| 양자화 포팅 (4bit/FP4) | **양자화 기대 오차 기반** (W2, §5 참조) |

위 표는 초기 계획이다. 현재 값과 측정 근거는 [CLI 문서](CLI.md#임계값-rel_l2)에 있다.

### 1.4 범위

| 포함 | 제외 |
|---|---|
| PyTorch(HF transformers) → MLX(mlx-lm 형식) 모델 | 변환기 자체 (mlx-lm 등에 맡김) |
| Llama/Qwen 계열 dense → W2부터 MoE | 학습, backward |
| FP4(NVFP4/MXFP4) 정합성 | NF4, INT4 (백로그) |
| 단일 장치 추론 | 멀티 GPU, 분산 |
| 결과 보고 (JSON/Markdown → HTML) | 자동 수정 |

---

## 2. 1주 베타를 위한 설계 결정

1주 안에 핵심 기능이 동작하도록 **새로 만들 것을 최소화**하고, 위험한 의존성을 피한다.

| 결정 | 이유 |
|---|---|
| 기존 GPU 커널(`MLXBackend`)에 **의존하지 않는다** | 커널은 아직 미검증. 검증기는 MLX 기본 op로 동작하는 어떤 모델에든 적용 가능해야 한다 |
| trace 형식 = `safetensors` + `manifest.json` | 공식 safetensors와 상호 호환되는 NumPy 전용 구현(`verify/safetensors_io.py`). 샘플 파일마다 SHA-256을 manifest에 기록해 변조를 거부 |
| golden 생성은 **CPU float32 PyTorch**로 시작 | 베타에서 CUDA 장비가 없어도 된다. CUDA capture는 W3에 추가 (같은 형식) |
| PyTorch 쪽 기록 = `register_forward_hook` | 표준 API, 모델 코드 수정 불필요 |
| MLX 쪽 기록 = **인스턴스 클래스 교체(probe)** | MLX `nn.Module`에는 forward hook이 없다. 기록할 모듈 인스턴스의 `__class__`를 `__call__`만 감싼 하위 클래스로 잠시 바꾼다. 래퍼 모듈과 달리 파라미터·속성 접근·`isinstance`가 그대로 유지된다 (구현 시 변경) |
| 모듈 이름 매핑 = **동일 경로 우선 + 규칙 파일** | mlx-lm은 대부분 HF와 같은 경로(`model.layers.N.self_attn.q_proj`)를 쓴다. 다른 경우만 JSON 규칙(`match`/`replace`/`ignore`)으로 보완 (의존성 없이 처리하려고 YAML 대신 JSON) |
| isolated 모드 베타 범위 = **단일 텐서 입력 모듈** | Linear, Embedding, RMSNorm/LayerNorm, MLP 블록. attention/decoder layer의 재생(hidden state + causal mask)은 앞당겨 구현함. cache를 쓰는 decode 단계는 W2 |
| CI 테스트 모델 = **로컬에서 생성한 tiny random 모델** | 네트워크·대용량 다운로드 없이 CI에서 재현. `transformers` config로 2-layer 모델 생성 → safetensors 저장 → mlx-lm 로드 |
| 의존성은 extra로 분리 | `cumlx-lab[capture]` = torch, transformers / `[diff]` = mlx(Linux는 `mlx[cpu]`), mlx-lm. safetensors는 NumPy만으로 직접 구현해 코어는 numpy만 |
| 검증기 자체를 **오류 주입(fault injection)**으로 검증 | 검증기가 틀리면 의미가 없다. 일부러 망가뜨린 포팅본을 정확히 지목하는지 테스트한다 (§4) |

### 2.1 모듈 구성 (구현 기준)

```text
src/cumlx/
  errors.py              안정된 exit code, error code, hint
  cli/main.py            파서, describe, stdout 보호(JSON 1개만), envelope 출력
  cli/output.py          cumlx.result.v1 envelope, --max-items 절단, text 렌더러
  cli/commands.py        doctor | fixture | capture | diff | conform | report | selftest
  verify/safetensors_io.py  NumPy 전용 safetensors 읽기/쓰기 (BF16 읽기 지원)
  verify/trace.py        manifest v1, 샘플 파일, SHA-256 검증, 출력 디렉터리 안전장치
  verify/capture_torch.py   forward hook 기록기, HF 로더, MLX 가중치 주입(양자화 반영 golden)
  verify/mlx_port.py     mlx-lm/사용자 loader, probe, port dtype 판정, dequantize
  verify/mapping.py      이름 매핑 (동일 경로 + JSON 규칙)
  verify/metrics.py      rel_l2, max_abs, cosine, 임계값 profile
  verify/diff.py         chain / isolated 실행, first_divergence 위치 추정
  verify/conform.py      FP4 decode 전수 비교, NVFP4 global scale 규약 검사
  verify/faults.py       오류 주입 F1~F8
  verify/selftest.py     검증기 자체 검증
  verify/report.py       Markdown 보고서
기존 quant.py, reference.py는 그대로 재사용
```

### 2.2 trace manifest (v1)

```json
{
  "schema": "cumlx.trace.v1",
  "source": {"framework": "torch", "version": "...", "device": "cpu", "dtype": "float32",
             "model": "<hf-id or path>", "revision": "<sha>"},
  "inputs": {"prompts_sha256": "...", "token_ids": "inputs.safetensors"},
  "modules": [
    {"name": "model.layers.0.self_attn.q_proj", "order": 12, "type": "Linear",
     "in": ["t/000012.in.0"], "out": "t/000012.out", "shape": [1, 7, 64], "dtype": "float32"}
  ]
}
```

---

## 3. 일정

### 3.1 베타 주간 (D1~D7: 11/02 ~ 11/08) — 핵심 기능 동작

| 일 | 작업 | 그날의 완료 기준 |
|---|---|---|
| **D1** 11/02 월 | CLI 골격(`[project.scripts] cumlx`), trace 형식·manifest, `metrics.py`, extras 정의. Linux에서 MLX CPU wheel 설치 가능 여부 확인 | 합성 데이터로 metrics·format 단위 테스트 통과 |
| **D2** 11/03 화 | `capture`: forward hook 기록기, tiny random Llama 생성 스크립트 | tiny 모델 capture → manifest·텐서 재로드 일치 |
| **D3** 11/04 수 | `mlx_probe.py`, `mapping.py`, **chain 모드** diff | 같은 tiny 모델의 mlx-lm 버전 chain diff 실행, 모든 모듈 매핑 성공 |
| **D4** 11/05 목 | **isolated 모드** (단일 입력 모듈), `first_divergence`, JSON/Markdown 보고서 | 깨끗한 포팅본에서 false alarm 0 |
| **D5** 11/06 금 | `conform`: FP4 경계값 벡터 → `mx.dequantize`(nvfp4/mxfp4) 대조. 오류 주입 테스트 스위트 | §4 오류 주입 F1~F6 검출 |
| **D6** 11/07 토 | 실제 소형 모델 dogfooding: Qwen2.5-0.5B 또는 Llama-3.2-1B (HF CPU fp32 vs mlx-lm bf16 / 4bit) | 실제 모델에서 보고서 생성, 발견된 버그 수정 |
| **D7** 11/08 일 | Quickstart 문서, 알려진 한계 목록, `v0.4.0b1` 태그 | **베타 Gate 통과** |

#### 베타 Gate (D7) — 모두 만족해야 베타 선언

- [x] `cumlx capture` → `cumlx diff`가 tiny 모델과 실제 0.5~1B 모델에서 끝까지 동작 — Qwen2.5-0.5B, SmolLM2-135M(0.5B 미만 추가 모델)
- [x] isolated 모드: 오류 주입 F1~F6을 **6/6 정확한 모듈로 지목** — 실제 모델에서 F1~F8, F10 41회 전부
- [x] 깨끗한 bf16 포팅본: 기본 임계값에서 **false alarm 0** — 무손실과 float32 원본 양쪽, 공개 포팅본 포함
- [x] `cumlx conform --target mlx`가 nvfp4/mxfp4 양쪽에서 실행되고 결과를 보고 (CPU·GPU)
- [x] CPU CI에서 capture·diff·conform 테스트 통과 — verifier job 포함 전체 통과
- [x] 기존 테스트 109개 계속 통과

#### 베타가 늦어질 때 줄이는 순서

1. D6 실제 모델 → tiny 모델만으로 베타, 실제 모델은 W2 첫날
2. Markdown 보고서 → JSON만
3. `conform` → W2로 이동 (capture + diff만으로도 핵심 가치는 성립)

**절대 줄이지 않는 것:** isolated 모드와 오류 주입 테스트. 이 둘이 없으면 "검증기"가 아니다.

---

### 3.2 W2 (11/09 ~ 11/15) — 베타 강화: 커버리지와 양자화

| 작업 | 설명 |
|---|---|
| 다중 입력 모듈 isolated 재생 | attention, decoder layer: hidden state + mask + position ids/RoPE 입력까지 기록·재생 |
| decode 단계 capture | KV cache를 사용하는 토큰 1개 생성 단계도 기록 (prefill과 별도 trace) |
| **양자화 기대 오차 oracle** | MLX 포팅본의 양자화 가중치를 dequantize → PyTorch 원본 모델에 주입 → 그 출력을 "양자화 golden"으로 사용. **양자화 손실**과 **구현 버그**를 분리한다 |
| 정적 가중치 diff | 실행 없이 MLX 가중치(dequantize)와 HF 가중치를 비교. 전치·scale·순서 오류를 빠르게 검출 |
| MoE 지원 | expert별 모듈 이름, router 출력 기록, expert별 global scale 검사 |
| 임계값 보정 | 깨끗한 포팅본 5개 이상에서 오차 분포 측정 → dtype·모듈 타입별 기본값 갱신 |

**테스팅 마일스톤 T2**
- [x] 오류 주입 F1~F10 전체 10/10 지목 (attention·MoE 관련 F7~F10 포함) — F11까지 11/11. F9는 공개 tiny MoE
- [x] 4bit 포팅본: 양자화 golden 기준 비교에서 깨끗한 포팅 false alarm 0, F4·F5(scale 오류) 검출 — 실제 모델
- [x] 정적 가중치 diff: 전치·scale 오류를 1분 이내 검출 — 0.5B 모델 약 3.3초. 원본 합 약 1GB
- [x] MoE 모델 1개(소형)에서 capture·diff 동작 — 작은 Mixtral 공개 모델 2개, tiny random 저장소 3개, fixture

---

### 3.3 W3 (11/16 ~ 11/22) — 실전 적용 캠페인

| 작업 | 설명 |
|---|---|
| **공개 포팅본 3~5개 검증** | HF의 NVFP4→MLX repack, mlx-lm 커뮤니티 변환본, MoE 포팅본 등 |
| 발견한 문제를 upstream에 보고 | 재현 trace와 보고서를 첨부한 issue/PR |
| CUDA 장비에서 capture | 같은 trace 형식으로 CUDA golden 생성. CPU golden과의 차이 기록 (CUDA 비결정성 측정 포함) |
| `conform --checkpoint` | ModelOpt NVFP4 원본 bytes를 CUMLX 코덱으로 독립 decode → MLX repack 가중치와 bit 비교 |
| HTML 보고서 | 모듈별 오차 그래프, first divergence 강조, trace 링크 |
| **자체 커널 dogfooding** | 기존 FP4 GPU 커널(`MLXBackend`)을 검증기로 검증. Apple Silicon 실기 GPU 테스트 35개 실행 |

**테스팅 마일스톤 T3**
- [x] 공개 포팅본 3개 이상에 대한 검증 보고서 — 9개 ok, 1개 로드 불가
- [ ] **upstream에 보고한 실제 불일치 ≥ 1건** (방향이 맞는지 판단하는 핵심 지표) — 불일치는 찾음(`rope_parameters`), 보고는 사용자가 직접 해야 함. 관련 upstream PR #1964가 열려 있음
- [x] CUDA golden vs CPU golden 차이가 기본 임계값 이내임을 문서화 — V100에서 최대 1.9e-5, Mac CPU와 최대 4.5e-5. [기록](CUDA-GOLDEN.md)
- [x] 기존 GPU 테스트 35개 Metal 실기에서 skip 0으로 실행, 결과 기록 — M5 Pro, MLX 0.32.2와 0.32.4

---

### 3.4 W4 (11/23 ~ 11/29) — 통합과 정식 릴리스 준비

| 작업 | 설명 |
|---|---|
| GitHub Action | 포팅 저장소 CI에서 `cumlx diff`를 돌리고 회귀 시 실패 |
| MCP 서버 / 에이전트 스킬 | 포팅 에이전트(mlx-porting-skill, kernelport 등)가 검증 단계에서 호출할 수 있는 인터페이스 |
| op 단위 API | `cumlx.check_op(torch_fn, mlx_fn, inputs)`: 커스텀 커널 포팅(Metal kernel) 단위 검증 |
| 성능 | 7B 모델 capture가 디스크·시간 면에서 감당 가능하도록 샘플링·부분 기록 옵션 (`--modules`, `--layers`) |
| 문서 | 사용 사례 중심 README 재작성, trace 형식 명세 공개 |

**테스팅 마일스톤 T4**
- [x] GitHub Action을 예제 저장소에 적용, 오류 주입 PR에서 실패하고 깨끗한 PR에서 통과. 2026-11-04에 공개 Action을 커밋으로 고정해 비공개 예제 저장소에 적용함. PR 이벤트로 깨끗한 PR은 `ok`로 통과, 결함 PR은 `diverged`(`model.layers.1.self_attn`)로 실패함. 그 전에 이벤트로 실행이 시작되지 않던 원인은 찾지 못함. [기록](../validation/t4-action-example.json)
- [x] MCP 인터페이스로 에이전트가 capture → diff → 보고서 해석까지 수행하는 데모 — [기록](../validation/mcp-agent-demo.md)
- [x] `check_op`로 기존 `rms_norm`/`silu_mul` Metal 커널 vs PyTorch 비교 통과 — `fp4_linear`도 포함
- [x] 7B 모델: `--layers` 부분 기록으로 trace ≤ 5 GB, capture 10분 이내 (목표치) — 322MB, 12초. 랜덤 가중치의 구조 기준 측정이며 실제 checkpoint는 미실행

### 3.5 정식 릴리스 (`v0.4.0`) — 완료 정의

```bash
python -m pip install -e '.[capture,diff]'     # PyPI 게시 전까지는 저장소에서 설치
cumlx capture Qwen/Qwen2.5-0.5B --prompts-file prompts.txt -o golden/
cumlx diff golden/ ./qwen-mlx-4bit -o report/ --mode isolated
cumlx conform --target mlx
```

- [x] 위 명령이 Apple Silicon Mac에서 문서대로 동작한다. 새 clone과 새 venv에서 README 순서대로 실행해 확인함 (M5 Pro).
- [x] 오류 주입 10종 모두 검출, 깨끗한 포팅본 false alarm 0. F1~F11, 실제 모델 50회.
- [x] 공개 포팅본 검증 보고서 3개 이상. 9개 ok.
- [ ] upstream 보고 1건 이상. 실제 불일치는 찾았고, 보고는 사용자가 직접 작성해야 함 (mlx-lm 정책).
- [x] 검증하지 않은 영역은 보고서와 문서에 "미검증"으로 표시된다.
- [ ] PyPI 게시. 계정과 공개 결정이 필요. `pip install 'cumlx-lab[capture,diff]'`는 아직 동작하지 않음.

---

## 4. 검증기 자체의 검증 — 오류 주입 목록

깨끗한 MLX 포팅본에 일부러 아래 결함을 하나씩 넣고, isolated 모드가 **정확한 모듈**을 `first_divergence`로 지목해야 한다.

| ID | 주입하는 결함 | 기대 지목 위치 | 베타 |
|---|---|---|---|
| F1 | Linear 가중치 전치 (정방 행렬) | 해당 `*_proj` | ✅ |
| F2 | bias 누락 | 해당 Linear | ✅ |
| F3 | RMSNorm epsilon 변경 / weight 누락 | 해당 norm | ✅ |
| F4 | MXFP4 E8M0 scale 지수 +1 | 해당 양자화 Linear | ✅ |
| F5 | NVFP4 E4M3 block scale code +8 (×2). MLX `QuantizedLinear`에는 global scale이 없어서 원안(global scale 누락)은 conform 검사로 옮김 | 해당 양자화 Linear | ✅ |
| F6 | 활성화 함수 교체 (SiLU → GELU) | 해당 MLP | ✅ |
| F7 | RoPE theta 변경 | 해당 attention (isolated, high) | ✅ 앞당김 |
| F8 | attention mask 누락 (causal) | 해당 attention (isolated, high) | ✅ 앞당김 |
| F9 | MoE expert 순서 뒤섞기 | 해당 MoE 블록의 `experts` | ✅ 앞당김 |
| F10 | nibble 순서 반전 (low/high) | 해당 양자화 Linear | ✅ 앞당김 |
| F11 | RoPE가 KV cache offset 무시 (추가) | 해당 attention, `phase: decode` | ✅ |

추가 지표:
- **False alarm rate:** 깨끗한 포팅본에서 임계값을 넘은 모듈 수 / 전체 모듈 수 = 0 (목표)
- **Chain vs isolated 비교:** chain 모드에서는 결함 이후 모든 모듈이 빨갛게 되지만 isolated 모드에서는 결함 모듈 하나만 빨갛게 됨을 보고서로 보여 준다.

---

## 5. 테스팅 전략

| 층 | 무엇 | 어디서 | 빈도 |
|---|---|---|---|
| L0 단위 | metrics, trace 형식, 이름 매핑, FP4 벡터 생성, CLI 인자 | CPU CI | 모든 PR |
| L1 통합(tiny) | tiny random 모델 capture → diff, 오류 주입 F1~F6 | CPU CI (MLX CPU 또는 macOS) | 모든 PR |
| L2 실제 모델 | 0.5~1B 모델 bf16·4bit 포팅 diff | macOS runner / 개발자 Mac | main push, 일 1회 |
| L3 양자화 | 양자화 golden oracle, conform | macOS runner | main push |
| L4 실전 | 공개 포팅본 캠페인 | 수동 | 주 1회 |
| L5 기존 커널 | `MLXBackend` GPU 테스트 35개 | Metal 실기 | W3부터 주 1회 |

**양자화 포팅의 판정 원칙:** 4bit/FP4 포팅본을 fp32 원본과 직접 비교하면 양자화 손실 때문에 항상 오차가 크다. 그래서 W2부터는 *같은 양자화 가중치를 PyTorch에 넣어 만든 golden*과 비교한다. 이 비교에서 남는 오차가 **구현 버그**다.

---

## 6. 기존 코드의 위치 변화

| 기존 자산 | 새 역할 |
|---|---|
| `quant.py` (FP4 코덱) | `conform`의 독립 decoder, 오류 주입 F4·F5·F10 생성기 |
| `reference.py` (float64 oracle) | `check_op`과 metrics 정밀도 기준 |
| `kernels/`, `mlx_backend.py` | 검증 대상(dogfooding). 개발 우선순위는 낮춤 |
| `planning.py`, `simulation.py` | 유지, 변경 없음 |
| CI 원칙 (skip ≠ pass, fallback 금지) | 검증기 보고서 원칙으로 확장: 매핑 실패·미기록 모듈은 "미검증"으로 표시하고 통과로 세지 않음 |

개정 1의 지적 사항 중 아래는 여전히 유효하며 W3~W4 또는 백로그에서 처리한다.

- ~~MLX 버전 정확 일치 강제~~ → 0.32.2 이상 0.33 미만 허용으로 완화함 (`check_mlx_version`)
- GPU self-hosted runner 미등록 → 백로그로 옮김. 사용자 장비에 상주 서비스를 설치하는 일이라 사용자가 결정한다. GPU 테스트는 로컬에서 실행함
- ~~Ruff 규칙 확대, 타입 검사 도입~~ → 검증기·CLI에 확대, mypy는 새 모듈부터 CI 적용함

---

## 7. 리스크와 대응

| 리스크 | 가능성 | 영향 | 대응 |
|---|---|---|---|
| Linux에서 MLX CPU wheel을 쓸 수 없음 | ~~중~~ 해소 | 중 | D1 확인 완료: `mlx[cpu]` 0.32.4 동작 |
| 개발 환경에서 HF Hub 접근 불가 | 해소 | 중 | Mac에서 D6 실행 완료 |
| 장치별 정밀도 차이가 false alarm을 일으킴 | 확인됨 | 큼 | M5의 MLX 기본 TF32. 기본으로 끄고, 측정값을 보고서에 기록 (`--tf32`) |
| 임계값이 단일 기기·2개 모델 측정값 | 중 | 중 | 다른 GPU 세대와 아키텍처에서 재측정 (W3 캠페인) |
| tiny fixture가 결함을 가림 | 확인됨 | 중 | HF 기본 초기화(std 0.02)에서는 attention이 거의 균등해 RoPE 결함이 안 보였다. `initializer_range=0.25`, 랜덤 norm/bias로 해결. 실제 모델 검증이 필요한 이유 |
| HF ↔ mlx-lm 모듈 경로가 모델마다 다름 | 높음 | 중 | 동일 경로 우선 + YAML 규칙. 매핑 실패는 "미검증"으로 보고 (조용히 건너뛰지 않음) |
| 다중 입력 모듈 재생이 어려움 (mask, cache) | 높음 | 중 | 베타는 단일 입력 모듈로 한정, W2에서 모델 계열별 adapter |
| trace 용량이 큼 | 높음 | 중 | 짧은 프롬프트, fp16 저장 옵션, `--layers` 부분 기록 |
| 임계값이 너무 빡빡하거나 느슨함 | 확인됨 | 큼 | 고정 임계값은 양쪽으로 틀렸다. 실행별 중앙값 기반으로 변경. 결함이 임계값을 올리지 못하도록 회귀 테스트 추가 |
| 공개 포팅본에서 버그를 못 찾음 | 중 | 중 | 그래도 "검증 통과" 보고서는 가치가 있음. 대상 범위를 MoE·NVFP4 repack 같은 위험 영역으로 좁힘 |
| 기존 도구가 같은 기능을 추가 | 중 | 중 | trace 형식을 공개 명세로 만들고 다른 도구에 연동되는 쪽을 택함 (경쟁보다 통합) |

---

## 8. 백로그 (1개월 이후)

1. 비 LLM 모델 (Whisper, diffusion, vision) capture adapter
2. CUDA 커스텀 커널(`.cu`) ↔ Metal 커널 op 단위 자동 비교 (kernelport 등과 연동)
3. 자동 원인 추정: 오차 패턴으로 "전치 의심", "scale 지수 오류 의심" 등 힌트 제시
4. NF4/INT4/GGUF 형식 conform
5. 기존 FP4 커널 성능 작업 (SIMD reduction, 벡터 load, matrix 명령)
6. GPU self-hosted runner 등록과 정기 실행

---

## 참고 자료

- mlx-vlm ModelOpt NVFP4 AWQ 수정: https://github.com/Blaizzy/mlx-vlm/pull/2582
- oMLX ModelOpt NVFP4 expert scale 수정: https://github.com/jundot/omlx/pull/4661
- mlxcel NVFP4 per-expert global_scale 이슈: https://github.com/lablup/mlxcel/issues/2351
- NVIDIA NVFP4 → MLX 모음: https://huggingface.co/collections/Brooooooklyn/nvidia-nvfp4-mlx
- mlx-porting-skill: https://github.com/Amal-David/mlx-porting-skill
- kernelport: https://pypi.org/project/kernelport/
- K-Search CUDA→MLX: https://bair.berkeley.edu/blog/2026/07/29/cuda-to-mlx-k-search/
- woof CUDA→Metal 번역: https://github.com/jacobbieker/woof/pull/139
- mlx-serve 번역 계층 실패 사례: https://github.com/ddalcu/mlx-serve/pull/820
- mpsops: https://github.com/mpsops
