# CUMLX

**CUDA에서 돌던 것을 Apple Silicon에서 돌게 하고, 원본과 얼마나 같은지 증명 등급으로 보입니다.**

CUMLX의 중심은 검증기입니다. MLX 포팅본이 원본 PyTorch 모델과 같은 계산을 하는지 모듈 단위로 비교하고, 다르면 처음 달라진 모듈을 찾습니다. 그 위에 CUDA를 전제한 PyTorch 프로그램, CUDA 커널, 로봇 시뮬레이션, 카메라 센서를 Mac에서 돌리는 경로를 붙였습니다. 직접 다 만들지 않고 오픈소스(mlx-lm, metalsplat, woof, Genesis, MuJoCo, Mitsuba)를 묶습니다. RTX와는 비교하지 않았습니다.

포팅이 성공했다는 메시지나 그럴듯한 생성 결과는 증거가 아닙니다. 가중치 전치, RoPE 오프셋, 잘못된 양자화 scale은 오류 없이 조용히 틀린 값을 냅니다. 아래는 RoPE base를 일부러 틀린 작은 포팅본에서 실제로 나온 판정입니다([기록](https://github.com/Topabaem05/CuMLX/blob/main/validation/t4-action-example.json)).

```text
diverged: 50 pass, 8 fail, 0 unverified, 6 skipped, 16 chain drift; first divergence at model.layers.1.self_attn (isolated, high)
```

## 증명 등급

물리와 렌더링은 구현이 다르면 값이 정확히 같을 수 없어서, 모든 결과에 등급을 붙입니다([로드맵 2절](https://github.com/Topabaem05/CuMLX/blob/main/docs/ROADMAP-PLATFORM.md#2-증명-등급)). 등급 B의 허용치는 측정 전에 정하고, 결과를 본 뒤 바꾸지 않습니다.

| 등급 | 뜻 | 방법 |
|---|---|---|
| **A 수치 동등** | 같은 입력에 같은 값 (float32 오차 수준) | `cumlx diff`, `check_op`, CUDA golden |
| **B 통계 동등** | 측정 지표가 허용 범위 안 | 궤적 오차, 정책 보상, PSNR, 사전 정의한 허용치 |
| **C 실행 확인** | 끝까지 돌지만 같다는 증거는 없음 | 스모크 테스트 |
| **미검증** | 실행하지 않음 | - |

**golden**은 원본 프로그램을 실행해 기록한 기준값입니다. 모델은 `cumlx capture`의 모듈별 입출력, 학습 스크립트는 `cumlx run --trace-losses`의 손실 기록, 커널은 `cumlx kernel capture`의 연산 단위 기록입니다.

## 현재 상태

[플랫폼 로드맵](https://github.com/Topabaem05/CuMLX/blob/main/docs/ROADMAP-PLATFORM.md)의 완료 조건 25개 중 24개를 채웠습니다. 남은 것은 M6(통합 릴리스)의 "기함 작업을 새 Mac에서 처음부터 재현"입니다. 프로젝트에 Mac이 한 대(Apple M5 Pro, 48 GB)뿐이라, 같은 Mac의 결과로는 체크하지 않기로 했습니다.

**기함 작업**은 플랫폼이 쓸모 있는지 판단하려고 로드맵에서 정한 다섯 작업이고, 목표 등급도 거기서 정했습니다. 2026-11-06의 첫 완전 실행에서 F2, F3, F4, F5는 목표 등급에 도달했고 F1은 실패했습니다([기록](https://github.com/Topabaem05/CuMLX/blob/main/validation/m6-finish-run.json)). **F1은 2026-11-07에 새 규칙의 첫 완전 실행에서 A를 받았습니다**([기록](https://github.com/Topabaem05/CuMLX/blob/main/validation/m6-f1-rerun.json)). 판정 규칙은 실행 전에 커밋했습니다. F1, F2, F3, F5는 이 Mac의 빈 환경(공개 저장소, 빈 홈 폴더)과 Tesla V100의 CUDA를 비교했고, F4는 CUDA 부분이 없어 Mac에서 Genesis와 MuJoCo, MLX와 rsl_rl을 비교했습니다. 첫 완전 실행은 V100과 Mac을 같은 커밋(4bf9b41)으로 돌렸습니다.

| 작업 | Mac 경로 | 목표 | 결과 | 근거 (rel_l2는 상대 L2 오차) | 범위 |
|---|---|---|---|---|---|
| F1 FlashAttention + bitsandbytes로 LLM 추론 | `cumlx run` + 어댑터 2개 | A | **A** (새 규칙, 2026-11-07) | 새 프롬프트의 V100 golden과 int8 층 168개 최악 rel_l2 3.0e-7(허용 2e-3), int8 가중치 일치, 디코더 층 최악 7.5e-3(허용 1.5e-2), 토큰 20개 모두 같음, 대조군 5개 모두 검출. 2026-11-06의 첫 완전 실행은 int8 층 하나가 2.48e-3이라 실패로 남음 | Qwen2.5-0.5B, 판정한 프롬프트 1개 |
| F2 3D Gaussian Splatting 학습 | `cumlx run` + gsplat 어댑터 | B | **B** | 실제 gsplat 1.5.3과 렌더 rel_l2 최악 3.6e-7(허용 1e-3), 기울기 2.6e-6(허용 5%), 시드 5개 평균 held-out PSNR Mac 43.15 dB, CUDA 43.50 dB(허용 차이 3.2 dB) | 합성 장면, 밀도 조절과 SH 없음 |
| F3 커스텀 `.cu` 커널 앱 (SAM2 연결 요소) | `cumlx kernel port` + woof(커널 6개), 호스트 함수는 Python으로 옮긴 드라이버 | A | **A** | 레이블과 개수가 원본 CUDA 확장과 정확히 같음, 드라이버 결함 3개 모두 검출(번역한 커널 안의 결함은 이 시험에 없음) | 합성 마스크 24개 |
| F4 Isaac Lab 스타일 사족 보행 학습과 추론 | `cumlx.sim` + MLX | 추론 A, 물리 B | **A, B** | MLX 정책이 rsl_rl과 절대 오차 최악 1.7e-6(허용 1e-5), Genesis와 MuJoCo 궤적 오차 중앙값 0.024(허용 0.1) | go2_flat, 학습 시드 1개 |
| F5 물리 기반 카메라 센서 | `cumlx.render` | B | **B** | RGB는 장치 간 PSNR 최악 45.65 dB(허용 40 dB), 깊이는 정확한 광선 계산 대비 최악 3.4e-5 m(허용 1 mm), 세그멘테이션 불일치 0 | 상자 장면 하나, 시점 6개. CUDA는 Mitsuba 3.8.0, Metal은 3.9.1 |

F1의 기록은 다음과 같습니다.

- **2026-11-06의 첫 완전 실행은 실패로 남습니다.** int8 층 하나(`model.layers.14.self_attn.o_proj`)가 2.48e-3(허용 2e-3)이었습니다. 원인은 bitsandbytes가 행 scale `127/absmax`를 CUDA의 근사 나눗셈 `__fdividef`로 구하는데, 어댑터는 장치에서 float32로 나눈 것입니다. 어댑터는 V100에서 잰 `__fdividef` 표를 쓰도록 고쳤습니다. 같은 golden을 Mac에서만 다시 돌린 3번째 시도(d1717d6)는 허용치 안이었지만, 첫 완전 실행이 아니고 독립 시험도 아니라서 등급이 되지 않습니다.
- **새 시험에서 A를 받았습니다.** [새 규칙](https://github.com/Topabaem05/CuMLX/blob/main/validation/m6-f1-rerun-limits.json)은 허용치와 대조 시험을 그대로 두고, 어떤 실행도 쓰지 않은 프롬프트로 두 장치를 다시 돌립니다. 측정 전에 커밋했습니다. V100과 Mac 모두 그 커밋(0f7856c)으로 돌았고, 첫 완전 실행에서 모든 부분과 대조군 5개가 통과했습니다. 그 전에 Mac 쪽은 디스크가 모자라 시작 전에 한 번 멈췄는데, 규칙상 완전 실행이 아닙니다.
- **이 golden은 고친 어댑터와 고치기 전 어댑터를 가르지 못합니다.** 고치기 전의 나눗셈(판정하지 않는 재생 결함 `ieee_row_scale`)도 이 golden에서는 int8 층 최악 1.3e-3으로 허용치 안이었습니다. 이 A는 지금 어댑터가 새 입력에서 맞다는 것을 보이고, `__fdividef` 표가 필요하다는 것을 보이지는 않습니다.
- **FlashAttention-2 바이너리는 어디서도 돌지 않았습니다.** V100(sm70)이 지원하지 않아 CUDA 기준 실행도 어텐션은 CUMLX의 어댑터를 썼습니다. 실제 CUDA 라이브러리와 비교한 것은 bitsandbytes이고, 어텐션의 기준은 float64 정확 계산입니다. F1은 bfloat16 프로그램이므로 목표 A는 bfloat16 반올림 수준의 동등입니다([규칙](https://github.com/Topabaem05/CuMLX/blob/main/validation/m6-f1-limits.json)).

F2는 시드마다 PSNR이 약 2 dB씩 달라 학습 비교 자체는 약하고, 렌더와 기울기 비교가 강한 근거입니다. 등급과 한계는 [결과 보고서](https://github.com/Topabaem05/CuMLX/blob/main/docs/PLATFORM-REPORT.md)에 있습니다.

## 시작하기

Apple Silicon Mac 또는 Linux, Python 3.10 이상이 필요합니다. PyPI에 올리기 전까지는 저장소에서 설치합니다.

```bash
git clone https://github.com/Topabaem05/CuMLX.git && cd CuMLX
python -m pip install -e '.[capture,diff]'
cumlx doctor --require capture,diff,weights     # 빠진 것과 설치 명령을 알려 줍니다
cumlx selftest                                  # 이 환경에서 검증기가 동작하는지 수 초 안에 증명합니다
```

`./my-port`는 mlx-lm이 불러올 수 있는 MLX 포팅본 디렉터리입니다. 아직 없으면 아래의 `cumlx port`로 만듭니다. mlx-lm 형식이 아닌 포팅본은 `--loader python:패키지.모듈:함수`로 불러옵니다.

```bash
cumlx weights Qwen/Qwen2.5-0.5B ./my-port                        # 가중치 확인 (수 초)
cumlx capture Qwen/Qwen2.5-0.5B --prompt "Hello" -o golden/      # 원본의 모듈별 입출력 기록
cumlx diff golden/ ./my-port -o report/                          # 포팅본 코드 확인
```

이 golden은 이 컴퓨터의 PyTorch(기본 CPU)로 기록합니다. CUDA와 비교하려면 NVIDIA 장비에서 `cumlx capture <원본> --prompt "Hello" --device cuda -o golden-cuda/`로 기록하고 Mac에서 `cumlx diff golden-cuda/ ./my-port`를 실행합니다([절차](https://github.com/Topabaem05/CuMLX/blob/main/docs/CUDA-GOLDEN.md)).

주요 종료 코드는 `0` 통과, `1` 불일치, `5` 미완료(통과가 아님)이고, `2`~`4`는 사용법·환경·입력 오류입니다. `5`는 파라미터가 있는데 비교하지 못한 모듈(`unverified`)이 있다는 뜻이며, `--mapping rules.json`으로 이름을 연결하거나 명시적으로 제외합니다. 대부분의 오류에는 `errors[].hint`에 해결 명령이 있고, 전체 목록은 `cumlx describe`에 있습니다. `report/`에는 `report.html`, `report.md`, `report.json`이 생깁니다.

Linux에서 MLX는 CPU로만 돌고, `diff`와 `weights`는 CI의 작은 fixture로만 시험했습니다. `capture --device cuda`, `run --target cuda`, `kernel capture`는 Linux의 V100에서 썼습니다. `kernel verify`는 Apple Silicon 전용입니다. `cumlx.render`의 기본은 Metal이고, F5에서는 V100의 Mitsuba 3.8.0 `cuda_ad_rgb`로도 돌았습니다. Linux에서 `diff` extra와 `cuda` extra를 함께 설치하지 마십시오.

## `cumlx diff` 결과를 읽는 법

`diff`는 기본으로 **isolated** 모드입니다. 각 모듈에 원본의 golden 입력을 그대로 넣어 비교하므로 오차가 누적되지 않고, `first_divergence`가 잘못된 모듈을 가리킵니다. 그 모듈을 포함해 재생하는 상위 모듈(attention, decoder layer)도 함께 실패할 수 있습니다. diff는 포팅본 전체를 네이티브 dtype으로 이어서 돌리는 **chain** 실행도 함께 하며, `--mode chain`은 chain만으로 판정합니다.

| `basis`, `confidence` | 뜻 |
|---|---|
| `isolated`, high | 그 모듈이 자기 golden 입력에서 틀렸습니다. RoPE, mask, residual 결함도 포함합니다 |
| `chain+isolated`, medium | 재생할 수 없는 모듈의 하위 모듈은 맞고, 그 모듈 자체 코드가 틀렸습니다 |
| `chain`, medium | `--mode chain`의 결과, 네이티브 dtype 실행의 NaN·shape 오류, 또는 재생할 수 없는 모듈에 isolated로 통과한 하위 모듈도 없는 경우입니다. chain 모드였다면 `--mode isolated`로 다시 실행하십시오 |

| 경고·필드 | 뜻 |
|---|---|
| `chain_drift` | 모듈은 golden 입력에서 통과했지만 chain 실행의 출력이 임계값을 넘었습니다. 네이티브 dtype의 누적 반올림이며 실패로 세지 않습니다 |
| `W_WEIGHT_ROUNDING_NOISE` | 가중치 반올림 때문에 임계값이 올라갔습니다. 더 작은 결함은 통과할 수 있으니 `--weights-from-mlx`를 쓰십시오 |
| `W_TF32_ACTIVE` | 행렬곱이 실제로 TF32로 실행됐다고 측정됐습니다(`--tf32 on`, M5나 CUDA). 임계값이 그 오차만큼 올라갑니다 |
| `W_PARTIAL_TRACE` | `--layers` 등으로 일부만 기록했습니다. 판정은 기록된 모듈에 대한 것입니다 |
| `data.execution` | 실행 장치와 측정한 float32 행렬곱 오차입니다 |

**MLX는 M5와 CUDA에서 float32 행렬곱을 기본으로 TF32로 실행합니다.** dtype은 float32로 보이지만 오차가 약 1e-3입니다. CUMLX는 기본으로 이 모드를 끄고 비교하며, 실제 오차를 측정해 보고합니다. 사용자의 기본 실행 환경 그대로 재려면 `--tf32 on`을 쓰십시오. `kernel verify`와 `rundiff`의 결과는 [CLI 문서](https://github.com/Topabaem05/CuMLX/blob/main/docs/CLI.md)를 보십시오.

## 무엇을 쓰는가

### 모델을 MLX로 옮겼다

시작하기의 `weights`, `capture`, `diff` 세 명령을 실행합니다. mlx-lm이 구현한 모델은 변환과 증명을 명령 하나로 끝냅니다. 변환은 mlx-lm이 하고, CUMLX는 결과를 검증합니다.

```bash
cumlx port Qwen/Qwen3-0.6B -o port/              # 변환 + 가중치 비교 + golden + diff
cumlx port Qwen/Qwen3-0.6B --q-bits 4 -o port4/  # 양자화
```

### 양자화했거나 bf16/f16으로 내렸다

양자화 포팅본을 원본과 그대로 비교하면 양자화 손실 때문에 항상 다릅니다. 포팅본의 가중치를 원본 코드에 넣은 golden과 비교하면, 남는 차이는 구현 버그입니다. bf16/f16 포팅본은 먼저 기본 golden으로 비교하고, `W_WEIGHT_ROUNDING_NOISE`가 나오면 이 golden을 씁니다. 가중치 자체는 `cumlx weights`로 따로 확인합니다.

```bash
cumlx capture <원본> --prompt "Hello" --weights-from-mlx ./my-port-4bit -o golden-p/
cumlx diff golden-p/ ./my-port-4bit
```

### 생성할 때만 답이 이상하다

capture는 기본으로 KV cache 위의 생성 한 단계도 기록합니다. prefill은 맞는데 캐시 처리(위치 오프셋, mask)만 틀리면 `first_divergence.phase`가 `decode`로 나옵니다.

### MoE 모델을 옮겼다

라우터와 expert를 따로 판정합니다. expert는 원본의 라우팅 결과로 재생하므로 라우터 결함과 expert 결함을 구분합니다. `cumlx weights`는 expert 순서가 바뀌면 어느 위치로 갔는지 알려 줍니다. mlx-lm의 `switch_mlp` 구조를 지원합니다.

### 공개된 NVFP4 repack을 믿어도 되는지 확인한다

모델을 실행하지 않고 ModelOpt 원본의 bytes와 MLX repack을 비교합니다. `global_scale` 규약 오류(2688배)도 지목합니다.

```bash
cumlx conform --checkpoint nvidia/모델-NVFP4 --against 누군가/모델-MLX-NVFP4
```

### CUDA 프로젝트를 옮기려 한다

먼저 작업량을 봅니다. 소스를 텍스트로만 읽고 실행하지 않습니다. `model`, `device`, `library`, `kernel`, `platform`으로 분류하고, `platform`은 RTX, Omniverse처럼 옮길 수 없는 경우입니다.

```bash
cumlx scan ./some-cuda-repo
cumlx recipe ./some-cuda-repo    # 발견한 항목마다 무엇으로 대체할지 안내
```

`cumlx recipe`는 선택지마다 근거를 표시합니다. `evidence: run`은 Apple Silicon에서 실행한 결과가 `validation/`에 있다는 뜻이고, `evidence: docs`는 문서만 읽었다는 뜻입니다. `--verified-only`를 주면 실행한 것만 남습니다.

### CUDA를 전제한 PyTorch 프로그램을 수정 없이 돌린다 (실험 단계)

`cumlx run`은 스크립트를 MPS에서 실행하고, 바꾼 호출을 모두 `run-report.json`에 기록합니다. 그 안에서 `torch.cuda.is_available()`은 True이고 장치 이름은 MPS로 표시합니다. MPS에 없는 연산은 조용히 CPU로 가지 않고 오류로 끝나며, 스트림이나 `cuda:1` 같은 호출은 `E_RUN_UNSUPPORTED`입니다. 컴파일된 CUDA 확장과 `python -I` 하위 프로세스에는 적용되지 않고, `torch.version.cuda`는 None 그대로입니다.

원래 패키지의 API를 Mac에서 주는 CUMLX 모듈을 **어댑터**라고 부릅니다. `--substitute`로 지정할 때만 쓰입니다.

```bash
cumlx run train.py -- --lr 1e-3
cumlx run infer.py --substitute flash_attn=cumlx.compat.flash_attn_sdpa --substitute bitsandbytes=cumlx.compat.bitsandbytes_int8
cumlx run train.py --substitute gsplat=cumlx.compat.gsplat_metalsplat
```

- `cumlx.compat.flash_attn_sdpa`: `flash_attn_func`, `flash_attn_varlen_func`를 float32 정확 어텐션으로 줍니다. dropout, sliding window, softcap, ALiBi, 확률 반환, paged cache, `flash_attn_with_kvcache`와 packed 함수 4개는 거부합니다.
- `cumlx.compat.bitsandbytes_int8`: Transformers의 `load_in_8bit=True`가 쓰는 LLM.int8을, V100(sm70)에서 잰 bitsandbytes 0.50.2의 CUDA 계산을 따라 줍니다. 반올림 한 번 수준의 알려진 차이가 2개 있고, 4비트, 학습, 이미 양자화된 체크포인트, `bitsandbytes.optim`의 optimizer는 거부합니다.
- `cumlx.compat.gsplat_metalsplat`: `gsplat.rasterization`을 metalsplat 위에서 줍니다. 핀홀 카메라와 RGB만 다루며, metalsplat은 따로 설치합니다.

`flash_attn_sdpa`와 `bitsandbytes_int8`는 `importlib.metadata`에 `flash-attn 2.8.3+cumlx`, `bitsandbytes 0.50.2+cumlx`로 보입니다. `+cumlx`는 원본 패키지가 아니라는 표시입니다. 돌아간다는 것은 맞다는 증거가 아니므로 `W_RUN_NOT_VERIFIED`가 항상 붙습니다. 자세한 것은 [CLI 문서](https://github.com/Topabaem05/CuMLX/blob/main/docs/CLI.md#cumlx-run)에 있습니다.

학습 스크립트는 NVIDIA 장비와 Mac에서 손실을 기록하고 `cumlx rundiff`로 등급을 매깁니다. 등급 B는 시드 3개 이상이 필요하고, 등급 A는 처음 두 스텝만 봅니다.

```bash
cumlx run train.py --target cuda --trace-losses --seed 0 --report golden/seed0.json   # NVIDIA 장비. 시드 1, 2도 같이
cumlx run train.py --trace-losses --seed 0 --report mps/seed0.json                    # Mac
cumlx rundiff golden/ mps/
```

### CUDA 커널을 옮긴다 (실험 단계)

```bash
cumlx kernel capture kernels/ --cases cases.json -o kgolden/ --require-idle-gpu   # NVIDIA 장비에서 원본을 실행해 golden 기록
cumlx kernel port foo.cu -o foo.metal --translator woof                            # 헤더, 템플릿, 호스트 코드, PTX 내장 함수 처리 후 번역
cumlx kernel verify foo.metal kgolden/                                             # golden 입력으로 Metal 커널 실행
```

CUMLX는 번역기를 설치하지 않습니다. woof(`metal-woof`)는 조건을 받아들이는 사람이 직접 설치합니다. 번역기가 없으면 `E_KERNEL_TRANSLATOR_MISSING`(exit 3)이고, Metal에 없는 기능은 `E_KERNEL_UNPORTABLE`로 이유와 함께 거부합니다. 호스트 코드(`main`, ATen)는 번역하지 않고 떼어 내므로, F3처럼 앱을 돌리려면 호스트 함수는 사람이 옮깁니다. 입력이 닿지 않은 경로의 결함은 잡을 수 없으므로, 통과는 case가 실행한 범위만 보장합니다. 직접 쓴 Metal 연산은 `cumlx.check_op(torch_fn, mlx_fn, inputs)`로 비교합니다.

`__nv_bfloat16` 커널은 출력 옆에 쓰는 `cumlx_bf16.h`로 옮깁니다. MSL `bfloat`와 Metal float 산술은 비정규수를 0으로 만들어 CUDA와 다르므로, 이 헤더는 값을 16비트로 저장하고 정수 산술로 계산합니다. M5 Pro에서 CUDA 모델과 변환은 2^32개 전부, 덧셈·뺄셈·곱셈은 bf16 쌍 2^32개 전부 비트까지 같았습니다. 비교 대상은 모델이며 CUDA 실행이 아닙니다. 나눗셈과 `__nv_bfloat162`는 거부합니다. [CLI 문서](https://github.com/Topabaem05/CuMLX/blob/main/docs/CLI.md)를 보십시오.

### 로봇 정책을 학습하고 돌린다

`cumlx.sim`은 Genesis와 MuJoCo 위에 Isaac Lab 스타일 작업(cartpole, Go2 평지)을 정의합니다. Isaac Lab 자체나 PhysX가 아닙니다. `.[sim,metal]` extra와 학습용 rsl-rl-lib를 설치하고, 학습한 정책은 MLX로 옮겨 rsl_rl과 비교합니다. [examples/robot-policy/](https://github.com/Topabaem05/CuMLX/blob/main/examples/robot-policy/)를 보십시오.

**Isaac Sim은 Mac에서 직접 실행할 수 없습니다.** `uv pip install isaacsim`은 macOS에서 실패합니다(macOS wheel 없음). RTX 장비에서 실행하고 Mac에서 스트리밍하거나, 학습된 정책만 MLX로 옮겨 쓰십시오. Isaac Sim과 Isaac Lab을 Mac에 올리던 작업은 라이선스 조건 때문에 2026-11-04에 중단했고, [측정 기록](https://github.com/Topabaem05/CuMLX/blob/main/validation/archive-orchardsim/)만 남아 있습니다.

### 카메라 센서를 만든다

`cumlx.render`는 Mitsuba Metal로 RGB, 깊이, 세그멘테이션과 레이캐스트 센서를 만듭니다(`.[render]` extra, USD는 `.[usd]`). 물리와 함께 쓰는 예제는 [examples/physics-camera/](https://github.com/Topabaem05/CuMLX/blob/main/examples/physics-camera/)에 있습니다. USD는 메시, 카메라 하나, 조명 2종, 확산 색만 읽고 나머지는 `E_RENDER_USD_UNSUPPORTED`로 거부합니다. RTX 센서와 다른 점은 [렌더링 문서](https://github.com/Topabaem05/CuMLX/blob/main/docs/RENDER-SENSORS.md)에 있습니다.

### 포팅 저장소의 CI에 넣는다

```yaml
- uses: Topabaem05/CuMLX@v0.5.0
  with:
    original: Qwen/Qwen2.5-0.5B
    port: ./my-port
    prompt: Hello
```

`ok`만 통과합니다. 불일치와 `incomplete`는 job을 실패시키고, job summary에 처음 달라진 모듈이 표시됩니다. [예제](https://github.com/Topabaem05/CuMLX/blob/main/examples/port-ci/)가 있고, `scripts/create_example_repo.sh`는 이를 별도 저장소로 만듭니다.

### Rust에서 쓴다

`describe`, `tracediff`와, MLX 없이 파일을 직접 읽는 `weights`가 Rust로도 있습니다. [rust/](https://github.com/Topabaem05/CuMLX/blob/main/rust/README.md)의 `cumlx-rs`가 이 세 명령을 실행하고, 정상 입력에서 같은 JSON 문서(`cumlx.result.v1`), 같은 종료 코드와 오류 코드를 냅니다. 형식이 깨진 manifest처럼 결과가 다른 입력은 rust/README.md에 적었습니다. 라이브러리 crate `cumlx`는 trace 형식, safetensors, 지표와 임계값 profile을 Rust에서 쓰게 합니다.

```bash
cargo build --release --manifest-path rust/Cargo.toml
rust/target/release/cumlx-rs describe
rust/target/release/cumlx-rs tracediff golden-cuda/ golden-cpu/
rust/target/release/cumlx-rs weights ./original ./port-mlx
```

torch, MLX 또는 nvcc가 필요한 `capture`, `diff`, `run`, `kernel capture` 같은 명령은 실행하지 않고 종료 코드 3과 `E_PYTHON_ONLY`로 끝나며, `hint`에 Python `cumlx` 명령을 적습니다. `scan`, `recipe`, `rundiff`, `report`, `kernel port`는 아직 옮기지 않았습니다. `weights`는 로컬 원본과 양자화하지 않은 로컬 포팅본만 읽고, Hugging Face id, `--mapping`, `--loader python:...`는 `E_PYTHON_ONLY`로 거부합니다. 두 구현이 같은 입력에서 같은 판정과 1e-12 이내의 같은 숫자를 내는지 [tests/test_rust_parity.py](https://github.com/Topabaem05/CuMLX/blob/main/tests/test_rust_parity.py)가 확인합니다.

### AI 에이전트에게 맡긴다

`--format json`(stdout이 터미널이 아니면 기본)에서 CLI는 stdout에 JSON 문서 하나만 내보내고, 대부분의 오류에 해결 명령을 붙이고, 다음 명령을 제안합니다. `cumlx describe --format json`이 모든 명령과 오류 코드를 알려 줍니다. `cumlx-mcp`는 같은 명령을 MCP 도구로 제공하며, 코드를 실행하는 `run`, `kernel capture`와 `--trust-remote-code`, `--loader`, `--translator`는 서버를 `--allow-code-execution`으로 시작해야 보입니다. [에이전트 스킬](https://github.com/Topabaem05/CuMLX/blob/main/skills/cumlx-verifier/SKILL.md)과 [AGENTS.md](https://github.com/Topabaem05/CuMLX/blob/main/AGENTS.md)를 보십시오.

## 검증 범위

장비는 Apple M5 Pro 한 대(Metal, CPU), Linux MLX CPU(CI, 작은 fixture), Tesla V100 한 대(CUDA)입니다. 기록마다 범위가 다릅니다([검증 보고서](https://github.com/Topabaem05/CuMLX/blob/main/validation/REPORT.md)).

- 깨끗한 실제 포팅본 38회(Qwen2.5-0.5B, SmolLM2-135M, 공개된 작은 MoE, 입력 5개): false alarm 0. 2회는 시작하지 못했습니다(CPU의 `gather_mm`).
- 결함 주입 50회(F1~F11, Qwen2.5-0.5B와 SmolLM2-135M, MoE 결함은 작은 MoE): 놓친 것 0, 3회는 해당 없음(bias가 없는 층).
- 공개된 `mlx-community` 포팅본 10개 중 9개 ok, 1개는 불러오지 못함.
- 2026-11-02 기준 최근 Hugging Face 모델 12개 중 7개(Gemma4, LFM2, Mixtral MoE, Qwen3, Llama)가 구현과 4비트 변환 모두 ok, 4개는 부분 체크포인트로 거부, 1개는 mlx-lm이 불러오지 못함.
- 기함 작업은 위 표의 범위에서만 시험했습니다.

**미검증:** 다른 Mac과 다른 Apple GPU 세대, V100 외의 NVIDIA 장비, FlashAttention-2 바이너리, 패딩 입력, 음성·이미지 모델, RTX 센서. 전체 목록은 [미검증 목록](https://github.com/Topabaem05/CuMLX/blob/main/docs/UNVERIFIED.md)에 있습니다. 임계값은 위 범위의 측정값입니다.

## 문서

| 문서 | 내용 |
|---|---|
| [플랫폼 결과 보고서](https://github.com/Topabaem05/CuMLX/blob/main/docs/PLATFORM-REPORT.md) | 조건과 기함 작업별 상태, 등급, 실패 |
| [검증 보고서](https://github.com/Topabaem05/CuMLX/blob/main/validation/REPORT.md) | 검증기의 실제 모델 실행, 결함 주입, CUDA golden 기록 |
| [미검증 목록](https://github.com/Topabaem05/CuMLX/blob/main/docs/UNVERIFIED.md) | 확인하지 못한 범위 |
| [CLI](https://github.com/Topabaem05/CuMLX/blob/main/docs/CLI.md) | 명령, 판정, 임계값, 어댑터, 측정 기록 |
| [AGENTS.md](https://github.com/Topabaem05/CuMLX/blob/main/AGENTS.md) | 에이전트 사용법과 개발 안내 |
| [Trace 형식](https://github.com/Topabaem05/CuMLX/blob/main/docs/TRACE-FORMAT.md) | golden trace 공개 명세 |
| [CUDA golden 측정](https://github.com/Topabaem05/CuMLX/blob/main/docs/CUDA-GOLDEN.md) | V100 측정 결과와 다른 NVIDIA 장비에서 실행할 절차 |
| [CUDA 프로젝트 포팅 검토](https://github.com/Topabaem05/CuMLX/blob/main/docs/PORTING-CUDA-MODELS.md) | 변환 가능 범위, Isaac Sim 검토, `scan`과 `port` |
| [렌더링과 센서](https://github.com/Topabaem05/CuMLX/blob/main/docs/RENDER-SENSORS.md) | `cumlx.render`, RTX 센서와 다른 점 |
| [플랫폼 로드맵](https://github.com/Topabaem05/CuMLX/blob/main/docs/ROADMAP-PLATFORM.md) | Apple Silicon 실행, 오픈소스 통합, 마일스톤 |
| [1개월 로드맵](https://github.com/Topabaem05/CuMLX/blob/main/docs/ROADMAP-1MONTH.md) | 검증기 마일스톤 진행 상태 |
| [FP4 커널 lab](https://github.com/Topabaem05/CuMLX/blob/main/docs/FP4-KERNEL-LAB.md) | 실험용 NVFP4/MXFP4 Metal·CUDA 커널. 설계는 [KERNEL-DESIGN](https://github.com/Topabaem05/CuMLX/blob/main/docs/KERNEL-DESIGN.md), 계획은 [IMPLEMENTATION-PLAN](https://github.com/Topabaem05/CuMLX/blob/main/docs/IMPLEMENTATION-PLAN.md) |
| [CI](https://github.com/Topabaem05/CuMLX/blob/main/docs/CI.md) | 자동 검사와 GPU 검증 경계 |
| [PyPI 릴리스](https://github.com/Topabaem05/CuMLX/blob/main/docs/RELEASE.md) | 릴리스 절차와 소유자가 할 일 |
| [Rust 코어](https://github.com/Topabaem05/CuMLX/blob/main/rust/README.md) | `cumlx-rs`: 옮긴 것, 옮기지 않은 것, Python과 같다는 증명 |
| [변경 기록](https://github.com/Topabaem05/CuMLX/blob/main/CHANGELOG.md) | 릴리스별 변경 |

CUMLX는 Apple 또는 NVIDIA의 공식 프로젝트가 아닙니다. MIT 라이선스입니다.
