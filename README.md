# CUMLX

**MLX 포팅본이 원본 모델과 같은 계산을 하는지 모듈 단위로 증명하고, 다르면 처음 달라진 모듈을 찾습니다.**

지금의 CUMLX는 mlx-lm이든 직접 작성한 코드든, 어떻게 만든 포팅본이든 그 뒤에 붙여 쓰는 검증 도구입니다. 다음 목표는 RTX·CUDA에서 돌던 것을 Apple Silicon에서 돌게 만드는 것까지 넓히는 것입니다. [플랫폼 로드맵](docs/ROADMAP-PLATFORM.md)에 있으며 아직 구현하지 않았습니다. 포팅이 성공했다는 메시지나 그럴듯한 생성 결과는 증거가 아닙니다. 가중치 전치, RoPE 오프셋, 잘못된 양자화 scale은 오류 없이 조용히 틀린 값을 냅니다.

```text
$ cumlx diff golden/ ./my-port
diverged: 50 pass, 8 fail, 0 unverified, 4 skipped; first divergence at `model.layers.1.self_attn` (isolated, high)
```

## 시작하기

Apple Silicon Mac 또는 Linux, Python 3.10 이상이 필요합니다.

```bash
python -m pip install -e '.[capture,diff]'
cumlx doctor --require capture,diff,weights     # 빠진 것과 설치 명령을 알려 줍니다
cumlx selftest                                  # 이 환경에서 검증기가 동작하는지 증명합니다 (수 초)
```

```bash
cumlx weights Qwen/Qwen2.5-0.5B ./my-port                        # 가중치 확인 (수 초)
cumlx capture Qwen/Qwen2.5-0.5B --prompt "Hello" -o golden/      # 원본의 모듈별 입출력 기록
cumlx diff golden/ ./my-port -o report/                          # 포팅본 코드 확인
```

종료 코드는 `0` 통과, `1` 불일치, `5` 미완료(통과가 아님)입니다. `report/`에 `report.html`, `report.md`, `report.json`이 생깁니다.

## 언제 무엇을 쓰는가

### 모델을 쉽게 옮기고 싶다

mlx-lm이 구현한 모델은 변환과 증명을 명령 하나로 끝냅니다. 변환은 mlx-lm이 하고, CUMLX는 결과를 검증합니다.

```bash
cumlx port Qwen/Qwen3-0.6B -o port/              # 변환 + 가중치 비교 + golden + diff
cumlx port Qwen/Qwen3-0.6B --q-bits 4 -o port4/  # 양자화
```

CUDA 전용 프로젝트는 먼저 작업량을 봅니다. 소스를 텍스트로만 읽고 실행하지 않습니다.

```bash
cumlx scan ./some-cuda-repo
cumlx recipe ./some-cuda-repo    # 발견한 항목마다 무엇으로 대체할지 안내
cumlx run train.py                # CUDA 전제 PyTorch 스크립트를 수정 없이 MPS에서 실행하고, 바꾼 호출을 기록 (실험 단계)
cumlx rundiff golden/ mps/        # CUDA와 MPS의 손실 기록을 비교해 등급을 매김 (실험 단계)
```

`model`, `device`, `library`, `kernel`, `platform` 중 하나로 분류합니다. `platform`은 RTX, Omniverse처럼 옮길 수 없는 경우입니다.

`cumlx recipe`는 선택지마다 근거를 표시합니다. `evidence: run`은 Apple Silicon에서 실행한 결과가 `validation/`에 있다는 뜻이고, `evidence: docs`는 문서만 읽었다는 뜻입니다. `--verified-only`를 주면 실행한 것만 남습니다.

**Isaac Sim은 Mac에서 직접 실행할 수 없습니다.** `pip install isaacsim`은 macOS에서 실패합니다(`Didn't find wheel`). RTX 장비에서 실행하고 Mac에서 스트리밍하거나, [학습된 정책만 MLX로 옮겨](examples/robot-policy/) 쓰십시오. Isaac Sim 없는 Isaac Lab cartpole 학습은 Newton으로 Mac CPU에서 돌았습니다. Unitree Go2 평지 보행도 같은 경로로 학습했고, MLX로 돌린 정책이 시뮬레이션에서 명령 속도를 따라갑니다(시드 1개). NVIDIA가 지원하지 않는 구성입니다([cartpole](validation/isaac-lab-mac.json), [Go2](validation/isaac-lab-go2.json) 측정값). Apple GPU 물리(warp-metal)와 Isaac Sim 연결 작업은 비공개 저장소 [OrchardSim](https://github.com/Topabaem05/OrchardSim)으로 옮겼습니다. [검토 문서](docs/PORTING-CUDA-MODELS.md)를 보십시오.

### 모델을 MLX로 옮겼다

위 세 명령을 실행합니다. `diff`는 각 모듈에 **원본의 입력을 그대로 넣어** 비교하므로, 오차가 누적되지 않고 잘못된 모듈 하나만 실패합니다. attention과 decoder layer도 따로 재생합니다.

### 양자화했거나 bf16/f16으로 내렸다

양자화 포팅본을 원본과 그대로 비교하면 양자화 손실 때문에 항상 다릅니다. 포팅본의 가중치를 원본 코드에 넣은 golden과 비교하면, 남는 차이는 구현 버그입니다.

```bash
cumlx capture <원본> --prompt "Hello" --weights-from-mlx ./my-port-4bit -o golden-p/
cumlx diff golden-p/ ./my-port-4bit
```

가중치 자체는 `cumlx weights`로 따로 확인합니다.

### 생성할 때만 답이 이상하다

capture는 기본으로 KV cache 위의 생성 한 단계도 기록합니다. prefill은 맞는데 캐시 처리(위치 오프셋, mask)만 틀리면 `first_divergence.phase`가 `decode`로 나옵니다.

### MoE 모델을 옮겼다

라우터와 expert를 따로 판정합니다. expert는 원본의 라우팅 결과로 재생하므로, 라우터 결함과 expert 결함을 구분합니다. `cumlx weights`는 expert 순서가 바뀌면 어느 위치로 갔는지 알려 줍니다. 현재 mlx-lm의 `switch_mlp` 구조를 지원합니다.

### 공개된 NVFP4 repack을 믿어도 되는지 확인한다

모델을 실행하지 않고, ModelOpt 원본의 bytes와 MLX repack을 비교합니다. `global_scale` 규약 오류(2688배)도 지목합니다.

```bash
cumlx conform --checkpoint nvidia/모델-NVFP4 --against 누군가/모델-MLX-NVFP4
```

### Metal 커널을 직접 작성했다

```python
import numpy as np, torch
from cumlx import check_op

x = np.random.default_rng(0).normal(size=(4, 8)).astype(np.float32)
report = check_op(torch.nn.functional.silu, my_metal_silu, inputs=[x])
assert report['verdict'] == 'ok', report
```

### 포팅 저장소의 CI에 넣는다

```yaml
- uses: Topabaem05/CuMLX@v0.5.0
  with:
    original: Qwen/Qwen2.5-0.5B
    port: ./my-port
    prompt: Hello
```

`ok`만 통과합니다. 불일치와 `incomplete`는 job을 실패시키고, job summary에 처음 달라진 모듈이 표시됩니다. 이 저장소는 비공개이므로, 다른 저장소에서 쓰려면 먼저 Settings → Actions → Access에서 접근을 허용해야 합니다. [예제](examples/port-ci/)가 있고, `scripts/create_example_repo.sh`는 이를 별도 저장소로 만듭니다.

### AI 에이전트에게 맡긴다

CLI는 stdout에 JSON 문서 하나만 내보내고, 오류마다 해결 명령을 붙이고, 다음 명령을 제안합니다. `cumlx-mcp`는 같은 명령을 MCP 도구로 제공합니다. [에이전트 스킬](skills/cumlx-verifier/SKILL.md)과 [AGENTS.md](AGENTS.md)를 보십시오.

## 결과를 읽는 법

`data.first_divergence`를 먼저 봅니다.

| `basis`, `confidence` | 뜻 |
|---|---|
| `isolated`, high | 그 모듈이 자기 golden 입력에서 틀렸습니다. RoPE, mask, residual 결함도 포함합니다 |
| `chain+isolated`, medium | 재생할 수 없는 모듈의 하위 모듈은 맞고, 그 모듈 자체 코드가 틀렸습니다 |
| `chain`, medium | chain 모드 결과이거나, 네이티브 dtype 실행의 NaN·shape 오류입니다 |

| 경고·필드 | 뜻 |
|---|---|
| `chain_drift` | 네이티브 dtype의 누적 반올림입니다. 결함이 아닙니다 |
| `W_WEIGHT_ROUNDING_NOISE` | 가중치 반올림 때문에 임계값이 올라갔습니다. 더 작은 결함은 통과할 수 있으니 `--weights-from-mlx`를 쓰십시오 |
| `W_TF32_ACTIVE` | `--tf32 on`으로 실행했습니다. 임계값이 크게 올라갑니다 |
| `W_PARTIAL_TRACE` | `--layers` 등으로 일부만 기록했습니다. 판정은 기록된 모듈에 대한 것입니다 |
| `data.execution` | 실행 장치와 측정한 float32 행렬곱 오차입니다 |

### M5의 float32 주의점

**MLX는 M5와 CUDA에서 float32 행렬곱을 기본으로 TF32로 실행합니다.** dtype은 float32로 보이지만 오차가 약 1e-3입니다. CUMLX는 기본으로 이 모드를 끄고 비교하며, 실제 오차를 측정해 보고합니다. 사용자의 기본 실행 환경 그대로 재려면 `--tf32 on`을 쓰십시오.

## 검증 범위

자세한 수치와 원본 결과는 [검증 보고서](validation/REPORT.md)에 있습니다.

**실행한 범위:** Apple M5 Pro(Metal, CPU), Linux MLX CPU(CI, tiny fixture), Tesla V100(CUDA golden). 원본은 Qwen2.5-0.5B 계열, SmolLM2-135M, Qwen3-0.6B와 2026년 공개 모델 7개(Gemma4, LFM2, Mixtral MoE, Qwen3, Llama)입니다. 깨끗한 실행 38회의 false alarm은 0이고, 결함 주입 50회를 전부 정확한 모듈로 지목했습니다.

**미검증:** V100 외의 NVIDIA 장비, M5 외의 Apple GPU 세대, 패딩 입력, 음성·이미지 모델. 임계값은 위 범위의 측정값입니다.

## 문서

| 문서 | 내용 |
|---|---|
| [변경 기록](CHANGELOG.md) | 릴리스별 변경 |
| [CLI](docs/CLI.md) | 명령, 판정, 임계값, 측정 기록 |
| [AGENTS.md](AGENTS.md) | 에이전트 사용법과 개발 안내 |
| [Trace 형식](docs/TRACE-FORMAT.md) | golden trace 공개 명세 |
| [CUDA golden 측정](docs/CUDA-GOLDEN.md) | V100 측정 결과와 다른 NVIDIA 장비에서 실행할 절차 |
| [CUDA 프로젝트 포팅 검토](docs/PORTING-CUDA-MODELS.md) | 변환 가능 범위, Isaac Sim 검토, `scan`과 `port` |
| [플랫폼 로드맵](docs/ROADMAP-PLATFORM.md) | 다음 계획: Apple Silicon 실행, 오픈소스 통합, 마일스톤 |
| [1개월 로드맵](docs/ROADMAP-1MONTH.md) | 검증기 마일스톤 진행 상태 |
| [FP4 커널 lab](docs/FP4-KERNEL-LAB.md) | 실험용 NVFP4/MXFP4 Metal·CUDA 커널 |
| [CI](docs/CI.md) | 자동 검사와 GPU 검증 경계 |

CUMLX는 Apple 또는 NVIDIA의 공식 프로젝트가 아닙니다. MIT 라이선스입니다.
