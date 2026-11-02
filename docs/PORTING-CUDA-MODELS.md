# CUDA 전용 모델과 프로젝트를 Apple Silicon으로 옮기는 방법 검토

작성일: 2026-11-02. 질문은 두 가지입니다. CUDA 전용 오픈소스 모델을 쉽게 변환하는 코드를 CUMLX에 넣을 수 있는가? Isaac Sim 같은 복잡한 프로젝트도 쉽게 변환할 수 있는가?

## 결론

1. **범용 CUDA 변환기는 만들지 않습니다.** 대신 `cumlx scan`(작업량 분류)과 `cumlx port`(변환과 증명을 명령 하나로)를 추가했습니다.
2. **mlx-lm이 구현한 모델은 이미 쉽게 변환됩니다.** `cumlx port`로 변환하고 바로 검증합니다.
3. **손으로 쓴 CUDA 커널이 있으면 자동 변환이 되지 않습니다.** 커널을 하나씩 다시 쓰고 `cumlx.check_op`으로 검증해야 합니다.
4. **Isaac Sim은 Mac에서 직접 실행할 수 없습니다.** NVIDIA RTX와 Omniverse 런타임에 묶여 있기 때문입니다. RTX 장비에서 실행하고 Mac에서 스트리밍하거나, 학습된 정책만 MLX로 옮겨 쓸 수 있습니다. 정책 경로는 직접 확인했습니다.

## 난이도 4단계

`cumlx scan <경로 또는 HF id>`는 소스를 텍스트로만 읽고 다음 단계로 분류합니다. 대상 코드는 실행하지 않습니다.

| 단계 | 뜻 | 할 일 | 자동화 |
|---|---|---|---|
| `model` | mlx-lm이 구현한 아키텍처 | `cumlx port` | 전부 |
| `device` | CUDA는 장치 문자열뿐 | 장치 설정 변경 후 검증 | 대부분 |
| `library` | CUDA 전용 라이브러리 사용 | 대체 라이브러리로 교체 | 일부 |
| `kernel` | 손으로 쓴 CUDA 커널 | Metal이나 MLX로 재작성, `check_op` 검증 | 없음 |
| `platform` | NVIDIA 하드웨어나 폐쇄 런타임 필요 | 그 부분을 NVIDIA 장비에 두거나 교체 | 불가 |

### 실제 저장소 분류 결과

[validation/scan-cuda-repos.json](../validation/scan-cuda-repos.json)에 원본 결과가 있습니다. 빌드나 실행은 하지 않았습니다. 테스트, 문서, 도구, 예제 폴더는 제외했습니다.

| 저장소 | 단계 | CUDA 소스 파일 | 해석 |
|---|---|---|---|
| isaac-sim/IsaacSim | `platform` | 14 | 1141개 파일이 Omniverse를 사용합니다 |
| NVIDIA/warp | `kernel` | 15 | Warp 자체는 Mac에서 CPU로 실행됩니다 |
| Dao-AILab/flash-attention | `kernel` | 613 | MLX에는 이미 `mx.fast.scaled_dot_product_attention`이 있어 옮길 이유가 적습니다 |
| NVlabs/nvdiffrast | `kernel` | 5 | 래스터화 커널의 재작성이 필요합니다 |
| facebookresearch/sam2 | `kernel` | 1 | 커널 하나뿐입니다. 선택 기능인지 먼저 확인할 가치가 있습니다 |

분류는 시작점입니다. sam2처럼 파일이 하나뿐인 경우도 `kernel`로 나옵니다. 의존 패키지 안의 CUDA는 찾지 못합니다.

## 왜 범용 변환기를 만들지 않는가

- **변환 자체는 이미 여러 프로젝트가 합니다.** 모델은 mlx-lm과 mlx-vlm이 맡습니다. 커널은 K-Search, KForge, kernelport가 있습니다. 이 도구들은 직접 실행해 보지 않았습니다.
- **커널 번역은 기계적이지 않습니다.** 공유 메모리, 스레드 그룹, TF32 같은 차이 때문에 문법을 바꿔도 값이 달라질 수 있습니다. 앞선 사례로 woof는 커널 222개가 컴파일에 성공했지만 일부 값이 틀렸습니다.
- **CUMLX의 가치는 변환 결과의 증명입니다.** 이번 검증에서도 mlx-lm이 오류 없이 틀린 RoPE base를 쓰는 사례를 찾았습니다. 변환기를 하나 더 만들어도 이 문제는 줄지 않습니다.

그래서 코드는 변환 위에 증명을 붙이는 쪽으로 추가했습니다.

```bash
cumlx scan Qwen/Qwen3-0.6B                       # 작업량 확인 (가중치 다운로드 없음)
cumlx port Qwen/Qwen3-0.6B -o port/              # mlx-lm 변환 + 가중치 비교 + golden 기록 + diff
cumlx port Qwen/Qwen3-0.6B --q-bits 4 -o port4/  # 양자화. golden에 포팅본 가중치를 넣음
```

`cumlx port`는 mlx-lm이 구현하지 않은 아키텍처와 원격 코드가 필요한 모델을 거부합니다. 이때 변환된 척하지 않고 `cumlx scan`을 안내합니다.

## Isaac Sim

**Mac에서 직접 실행할 수는 없습니다. 다른 방식으로 사용할 수 있습니다.** 아래 표의 "직접 실행"은 2026-11-02에 M5 Pro에서 실제로 돌려 본 결과입니다.

| 항목 | 결과 | 확인 방법 |
|---|---|---|
| Isaac Sim을 Mac에서 실행 | **불가.** x86_64 RTX GPU와 DGX Spark만 지원합니다 | [공식 요구 사항](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html) |
| `pip install isaacsim` | **설치되는 듯 보이지만 빈 패키지입니다.** 최신판 6.2의 wheel은 Linux와 Windows용뿐이라, Mac에서는 코드가 없는 6.0.0.0 소스 패키지가 골라집니다 | PyPI 파일 목록과 패키지 내용 확인 |
| RTX 센서와 렌더링 | 불가. RT core가 필요합니다 | 공식 요구 사항 |
| 소스 변환 | 불가. 저장소는 Apache 2.0이지만 Omniverse Kit SDK 등 다른 라이선스의 구성 요소가 필요하고, 파일 1141개가 Omniverse를 사용합니다 | 저장소 `LICENSE`, `cumlx scan` |
| 보유한 Tesla V100에서 실행 | 요구 사항 미달. RT core가 없고 공식 최소 사양은 GeForce RTX 4080입니다 | 공식 요구 사항. 실행하지 않음 |

### 쓸 수 있는 방법

| 방법 | 무엇을 얻는가 | 상태 |
|---|---|---|
| **A. RTX 장비에서 실행하고 Mac에서 스트리밍** | Isaac Sim 전체 기능. [공식 스트리밍 클라이언트](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/manual_livestream_clients.html)가 macOS용으로 있습니다 | 문서로 확인. RTX 장비가 없어 실행하지 못함 |
| **B. 학습은 NVIDIA 장비, 정책 추론은 Mac** | 학습된 로봇 정책을 MLX로 실행 | **직접 실행.** 아래 참조 |
| **C. Mac에서 다른 물리 엔진 사용** | 렌더링 없는 물리 시뮬레이션 | **직접 실행.** 아래 참조 |

**B. 정책을 MLX로 옮기기.** [examples/robot-policy/verify_policy.py](../examples/robot-policy/verify_policy.py)는 rsl_rl 기본 구조(관측 48, 은닉 512-256-128, ELU, 행동 12)의 정책을 MLX로 옮기고 `cumlx.check_op`으로 비교합니다. 배치 크기 1, 16, 4096에서 최악 오차는 5.2e-7이고, 활성화 함수를 잘못 옮기면 실패합니다. 랜덤 가중치로 실행했습니다. `--checkpoint`로 실제 체크포인트를 넣을 수 있지만 실제 체크포인트로는 실행하지 않았습니다.

**C. Mac의 물리 엔진.** 모두 CPU입니다.

| 엔진 | 결과 (M5 Pro) | 한계 |
|---|---|---|
| MuJoCo 3.17 | 카트폴 약 72만 step/s | 단일 환경 CPU. GPU 병렬 학습은 아님 |
| Newton 1.7 / Warp 1.19 | 상자 낙하 시뮬레이션이 바닥에서 멈춤(z = 0.2). 커널 컴파일 포함 약 50 step/s | `cuda:0`은 거부. Metal 미지원 |
| Isaac Sim의 RTX 센서 | 대체 없음 | 렌더링이 필요하면 A를 쓰십시오 |

공개 fork [IsaacLab-MLX](https://github.com/RobotFlow-Labs/IsaacLab-mlx)는 cartpole, 2중 진자, 쿼드콥터 경로를 Mac에서 실행한다고 설명합니다. 전체 Isaac Sim 호환, RTX 센서, Warp 커널은 지원하지 않는다고 명시합니다. 저장소 설명만 읽었고 실행하지 않았습니다.

### 다른 CUDA 전용 프로그램

같은 순서로 판단하십시오.

1. `cumlx scan`으로 단계를 봅니다.
2. `platform`이면 NVIDIA 장비에서 실행하고 Mac은 원격 클라이언트로 씁니다. 신경망 부분만 분리할 수 있으면 B를 적용합니다.
3. `kernel`이면 그 커널이 선택 기능인지 먼저 확인합니다. 필요하면 다시 쓰고 `check_op`으로 검증합니다.

## 다음 단계 후보

아래는 필요할 때 결정할 항목입니다. 지금 구현하지 않았습니다.

| 후보 | 효과 | 비고 |
|---|---|---|
| 실제 Isaac Lab 정책으로 실행 | B 경로의 실전 확인 | 예제는 준비됨. 체크포인트 필요 |
| mlx-lm 미지원 아키텍처용 포팅 템플릿 | `model` 밖의 모델 작업량 감소 | `--loader python:...`와 연결 |
| 커널 번역 도구와 연동 | `kernel` 단계 자동화 | kernelport 등을 먼저 평가 |
| 의존 패키지까지 스캔 | 분류 정확도 향상 | 설치 없이 메타데이터 읽기 필요 |
