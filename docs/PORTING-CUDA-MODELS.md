# CUDA 전용 모델과 프로젝트를 Apple Silicon으로 옮기는 방법 검토

작성일: 2026-11-02. 질문은 두 가지입니다. CUDA 전용 오픈소스 모델을 쉽게 변환하는 코드를 CUMLX에 넣을 수 있는가? Isaac Sim 같은 복잡한 프로젝트도 쉽게 변환할 수 있는가?

## 결론

1. **범용 CUDA 변환기는 만들지 않습니다.** 대신 `cumlx scan`(작업량 분류)과 `cumlx port`(변환과 증명을 명령 하나로)를 추가했습니다.
2. **mlx-lm이 구현한 모델은 이미 쉽게 변환됩니다.** `cumlx port`로 변환하고 바로 검증합니다.
3. **손으로 쓴 CUDA 커널이 있으면 자동 변환이 되지 않습니다.** 커널을 하나씩 다시 쓰고 `cumlx.check_op`으로 검증해야 합니다.
4. **Isaac Sim은 변환할 수 없습니다.** 코드 문제가 아니라 NVIDIA RTX와 Omniverse 런타임에 묶여 있기 때문입니다. 시뮬레이션은 NVIDIA 장비에 두고, 학습된 정책만 옮기는 구성이 현실적입니다.

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

**변환할 수 없습니다.** 근거는 다음과 같습니다.

| 구성 요소 | 상태 (2026-11-02 확인) | 근거 |
|---|---|---|
| Isaac Sim | macOS 미지원. x86_64 RTX GPU와 DGX Spark만 지원 | [공식 요구 사항](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html) |
| RTX 센서와 렌더링 | RT core가 필요합니다. RT core가 없는 A100, H100도 지원하지 않습니다 | 공식 요구 사항 |
| 소스 코드 | 저장소는 Apache 2.0이지만, 빌드와 실행에 Omniverse Kit SDK 등 다른 라이선스의 구성 요소가 필요합니다 | 저장소 `LICENSE` |
| Omniverse 의존 | 파일 1141개가 사용합니다 (테스트, 문서, 도구 제외) | `cumlx scan` |
| Warp | Mac에서 CPU만 지원합니다. `cuda:0`은 거부합니다 | 직접 실행 (Warp 1.19.0, M5 Pro) |
| Newton | macOS는 CPU 전용입니다 | [Newton README](https://github.com/newton-physics/newton) |

### 현실적인 구성

1. **시뮬레이션은 NVIDIA 장비, 정책 추론은 Mac.** Isaac Lab에서 학습한 정책 네트워크는 대개 작은 신경망입니다. 이 부분은 MLX로 옮길 수 있고, `cumlx.check_op`이나 사용자 loader와 `cumlx diff`로 검증할 수 있습니다. 이 경로는 아직 실행해 보지 않았습니다.
2. **Mac에서 물리 시뮬레이션이 필요하면 CPU로 실행합니다.** Warp와 Newton은 Mac CPU에서 동작합니다. Warp의 CPU 실행은 직접 확인했습니다. Newton은 README로만 확인했습니다. 대규모 병렬 학습 속도는 기대하기 어렵습니다.
3. **IsaacLab-MLX 참고.** 공개 fork [IsaacLab-MLX](https://github.com/RobotFlow-Labs/IsaacLab-mlx)가 cartpole, 2중 진자, 쿼드콥터 경로를 Mac에서 실행한다고 설명합니다. 전체 Isaac Sim 호환, RTX 센서, Warp 커널은 지원하지 않는다고 명시합니다. 저장소 설명만 읽었고 실행하지 않았습니다.

## 다음 단계 후보

아래는 필요할 때 결정할 항목입니다. 지금 구현하지 않았습니다.

| 후보 | 효과 | 비고 |
|---|---|---|
| 정책 네트워크 예제 | 1번 구성의 검증 방법 제시 | 공개된 Isaac Lab 정책으로 실행 필요 |
| mlx-lm 미지원 아키텍처용 포팅 템플릿 | `model` 밖의 모델 작업량 감소 | `--loader python:...`와 연결 |
| 커널 번역 도구와 연동 | `kernel` 단계 자동화 | kernelport 등을 먼저 평가 |
| 의존 패키지까지 스캔 | 분류 정확도 향상 | 설치 없이 메타데이터 읽기 필요 |
