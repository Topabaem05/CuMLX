# CUMLX 플랫폼 로드맵 — RTX·CUDA 작업을 Apple Silicon에서 실행하기

작성일: 2026-11-02 · 기간: 2026-11-03 ~ 2027-01-29 (13주, 목표치) · 기준: `v0.4.1`
이전 로드맵: [1개월 로드맵](ROADMAP-1MONTH.md) (검증기). 이 문서는 그 뒤를 잇습니다.

> **한 줄 정의:** RTX·CUDA에서 돌던 것을 **Apple Silicon에서 돌게 만들고, 같은 결과인지 증명한다.**

지금까지 CUMLX는 옮긴 결과를 검증하는 도구였습니다. 이제 옮기는 일까지 맡습니다. 직접 다 만들지 않고, 이미 있는 오픈소스를 묶고 빈 곳만 구현합니다. 검증기는 그대로 핵심입니다. 옮겼다는 말을 믿을 수 있게 만드는 부분이기 때문입니다.

**이 문서의 상태:** M0을 실행해 결과를 반영한 계획입니다. M1 이후의 기능은 아직 구현하지 않았습니다. M0에서 실행하지 않은 후보는 문서로만 확인했습니다.

## 1. 원칙

1. **속이지 않습니다.** 호환 계층은 명시적으로 켜야 하고, 무엇을 바꿨는지 모두 기록합니다. MPS를 NVIDIA 장치라고 보고하지 않습니다.
2. **조용히 대체하지 않습니다.** 지원하지 않는 기능은 오류를 내고, 대안이 있으면 안내합니다.
3. **"돌아간다"와 "같다"를 구분합니다.** 모든 결과에 증명 등급을 붙입니다(2절).
4. **있는 것을 씁니다.** 오픈소스는 선택적 의존성으로 연결합니다. 코드를 복사해 넣지 않습니다.
5. **어댑터로 격리합니다.** 작은 프로젝트에 의존하므로, 언제든 교체할 수 있어야 합니다.
6. **라이선스를 먼저 봅니다.** 라이선스 표기가 없는 프로젝트는 사용자가 직접 설치하는 경우에만 연결합니다.

### 하지 않는 일

| 하지 않는 일 | 이유 |
|---|---|
| CUDA 바이너리, PTX 실행 또는 NVIDIA 드라이버 에뮬레이션 | 폐쇄 바이너리를 옮길 방법이 없습니다. 소스가 있는 것만 다룹니다 |
| Omniverse Kit 또는 Isaac Sim을 macOS용으로 옮기기 | Kit은 폐쇄 구성 요소이고 macOS용이 없습니다. Linux 컨테이너에서 CPU PhysX까지 도는 것은 [확인](../validation/isaac-sim.json)했지만, RTX와 카메라는 안 되고 NVIDIA 지원 구성도 아닙니다. "Isaac Sim 호환"이라고 주장하지 않습니다 |
| RTX 센서와 수치까지 같은 렌더링 | 렌더러가 다르면 같은 값이 나오지 않습니다. 등급 B로 비교합니다 |
| NVIDIA 대비 속도 보장 | 측정한 값만 보고합니다 |
| 다중 GPU, 분산 학습 | 범위 밖입니다 |

## 2. 증명 등급

물리와 렌더링은 구현이 다르면 값이 정확히 같을 수 없습니다. 그래서 등급을 나누고 보고서에 표시합니다.

| 등급 | 뜻 | 방법 | 적용 대상 |
|---|---|---|---|
| **A 수치 동등** | 같은 입력에 같은 값 (float32 오차 수준) | `cumlx diff`, `check_op`, CUDA golden | 모델, 커널, 연산 |
| **B 통계 동등** | 측정 지표가 허용 범위 안 | 궤적 오차, 정책 보상, PSNR, 사전 정의한 허용치 | 물리, 렌더링, 학습 |
| **C 실행 확인** | 끝까지 돌지만 같다는 증거는 없음 | 스모크 테스트 | 기준을 만들 수 없는 경우 |
| **미검증** | 실행하지 않음 | - | 표시만 함 |

등급 B의 허용치는 측정 전에 정합니다. 결과를 본 뒤 바꾸지 않습니다.

## 3. 구조

```text
                       cumlx scan   무엇을 옮겨야 하는지 분류           (있음)
                            │
                       cumlx recipe 무엇으로 옮길지 안내                (M1)
        ┌───────────────┬───┴───────────┬────────────────┐
        ▼               ▼               ▼                ▼
    cumlx port      cumlx run       cumlx kernel      cumlx.sim, cumlx.render
    모델 변환       PyTorch 호환     CUDA 커널 번역    물리, 레이트레이싱
    (있음)          (M2)            (M3)              (M4, M5)
        │               │               │                │
    mlx-lm, mlx-vlm  mps-* 패키지    woof 번역기       Genesis, Quadrants
    mflux            metalsplat      kernelport        Mitsuba, Dr.Jit
                     mlx3d           Slang             OpenUSD, MuJoCo
        └───────────────┴───────┬───────┴────────────────┘
                                ▼
          검증: diff, check_op, conform, tracediff, CUDA golden (있음)
                     + 등급 B 지표 (M4)
```

**CUDA golden은 이미 강점입니다.** 보유한 V100에서 기준을 기록해 Mac에서 비교하는 경로를 확인했습니다. 커널과 연산으로 확장합니다.

## 4. 통합 후보

상태 열의 "문서"는 문서로만 확인, "실행"은 이 Mac에서 직접 실행했다는 뜻입니다. 모두 2026-11-02 기준입니다. 측정값은 [validation/spikes.json](../validation/spikes.json)에 있습니다.

| 영역 | 프로젝트 | 라이선스 | 상태 | 쓰는 방법 |
|---|---|---|---|---|
| **물리** | [Genesis](https://github.com/Genesis-Embodied-AI/genesis-world) ★30k | Apache-2.0 | **실행**: Metal에서 Go2 4096개, Isaac USD 불러오기 | 1순위 시뮬레이터 |
| 물리 | [MuJoCo](https://github.com/google-deepmind/mujoco) ★16k | Apache-2.0 | **실행** (CPU 약 72만 step/s) | 등급 B의 기준 물리 |
| 물리 | [Newton](https://github.com/newton-physics/newton), [Warp](https://github.com/NVIDIA/warp) | Apache-2.0 | **실행** (공식판은 Mac에서 CPU만. warp-metal 경로는 OrchardSim) | 호환 대상, Metal 없음 |
| **GPU 언어** | [Quadrants](https://github.com/Genesis-Embodied-AI/quadrants) ★228 | Apache-2.0 | **실행**: Metal과 CPU 차이 7.0e-8 | Warp 스타일 커널의 이식 대상 |
| **레이트레이싱** | [Mitsuba 3.9](https://github.com/mitsuba-renderer/mitsuba3), [Dr.Jit](https://github.com/mitsuba-renderer/drjit) | BSD 3조항 형식, BSD-3-Clause | **실행**: `metal_ad_rgb` 렌더링. 하드웨어 레이트레이싱 사용 여부는 문서로만 확인 | RTX 대체 1순위 |
| 렌더링 | [LuisaRender](https://github.com/LuisaGroup/LuisaRender) ★629 | BSD-3-Clause | Mac 지원 미확인 | M5에서 평가 |
| 렌더링 | Genesis Nyx (`gs-nyx` 0.1.9) | 표기 없음 | PyPI에 Linux, Windows wheel만 있음 | 제외 |
| **장면** | [OpenUSD](https://github.com/PixarAnimationStudios/OpenUSD) ★7.6k | TOST 1.0 | **실행**: `usd-core` 읽기, 쓰기 | Isaac 자산 읽기 |
| **커널 번역** | [woof fork](https://github.com/jacobbieker/woof) `backend/translate` ★1 | Apache-2.0 | **실행**: 전용 형식 커널은 정확, 외부 커널 17개는 그대로 실패 (M0) | 번역기 1순위. 전처리 계층을 직접 만들어야 함 |
| 커널 증명 | [kernelport](https://github.com/PS12007/kernelport) ★0 | MIT | **실행**: Metal 포트 12개 PASS, 틀린 포트 FAIL. 원본 CUDA 검증은 미실행 | 연동 후보. 겹치는 부분이 큼 |
| 커널 작성 | [Slang](https://github.com/shader-slang/slang) ★5.8k | Apache-2.0 WITH LLVM-exception | 문서: Metal, CUDA 동시 출력 | 새로 쓰는 커널 |
| **PyTorch 대체** | [mps-bitsandbytes 등 mpsops](https://pypi.org/project/mpsops/) ★11 | PyPI 표기는 MIT. `mps-flash-attn`만 저장소에 라이선스 파일이 있습니다 (2026-11-03 확인) | 문서 | 사용자 직접 설치만. 파일이 없는 것은 포함 배포 불가 |
| 3D | [mlx3D](https://github.com/amirhossein-razlighi/mlx3D) ★43 | MIT | 문서: PyTorch3D 스타일, 3DGS 커널 | PyTorch3D, nvdiffrast 대체 후보 |
| 3DGS | [metalsplat](https://github.com/tchauffi/metalsplat) ★15, [msplat](https://github.com/rayanht/msplat) ★102 | MIT, Apache-2.0 | 문서 | gsplat 대체 |
| **모델** | [mlx-lm](https://github.com/ml-explore/mlx-lm), [mlx-vlm](https://github.com/Blaizzy/mlx-vlm), [mflux](https://github.com/mflux-community/mflux) | MIT | mlx-lm **실행** | `cumlx port` 확장 |
| 런타임 | [PyTorch MPS](https://github.com/pytorch/pytorch) | BSD 계열 | **실행** | `compile_shader`로 커널 연결 |
| 참고 | [IsaacLab-MLX](https://github.com/RobotFlow-Labs/IsaacLab-mlx) ★13 | BSD-3 | 문서: 2026-03 이후 갱신 없음 | 설계 참고. 의존하지 않음 |

**쓰지 않는 것:**

- **MoltenVK:** Khronos가 지원을 중단했습니다.
- **ZLUDA, chipStar, SYCLomatic:** Metal 경로가 없거나 Mac 지원을 확인하지 못했습니다.

## 5. 마일스톤

날짜는 목표치입니다. 각 마일스톤은 완료 조건을 모두 만족해야 끝납니다.

### M0. 가정 검증 (완료 2026-11-02)

Apple M5 Pro 한 대에서 직접 실행했습니다. 재현 방법은 [spikes/](../spikes/), 측정값은 [validation/spikes.json](../validation/spikes.json)에 있습니다.

| 확인할 것 | 계속 진행 조건 | 측정 결과 | 판정 |
|---|---|---|---|
| Mitsuba `metal_ad_rgb` | Cornell box 렌더링, CPU 대비 평균 차이 5% 이내 | 차이 0.008%, PSNR 52.1 dB. Metal 약 240 Mpath/s, CPU `scalar_rgb` 약 4 Mpath/s | **통과** |
| Genesis 병렬 물리 | 4096개 환경, CPU 대비 3배 이상 | Go2 임의 행동에서 Metal 8.1만~13.3만, CPU 1.3만~1.4만 env-step/s. 6.3~10.1배 | **통과** |
| Quadrants Metal | CPU와 오차 1e-5 이내 | 7.0e-8 | **통과** |
| OpenUSD | Isaac 자산을 읽어 Genesis로 불러옴 | Isaac Lab cartpole의 관절 2개 로드. 막대가 중력으로 쓰러지고, 중력을 끄면 그대로 있음 | **통과** |
| woof 번역기 | `.cu` 하나를 앱 없이 번역, 실행 | CUMLX 커널 3개 정확 (최악 5.0e-8), 틀린 커널은 1.08로 검출 | **통과, 단 아래 제약** |
| kernelport | 예제 커널을 Metal에서 PASS | 12개 PASS. 일부러 틀린 포트는 0/21로 FAIL | **통과, 단 아래 제약** |
| 라이선스 | 각 후보를 `LICENSE`로 확인 | 4절에 반영. `gs-nyx`, `mps-*` 일부는 표기 없음 | 표기 없는 것은 제외 또는 직접 설치만 |

**결과가 계획을 바꾼 부분:**

- **M3의 위험이 가장 큽니다.** woof는 자기 형식의 커널만 바로 처리합니다. 외부 커널 17개 중 **그대로 Metal에서 컴파일된 것은 0개**입니다. 빠진 헤더는 빈 파일로 넘어가도, 다음 벽이 있습니다.

  | 커널 | 멈춘 이유 | 필요한 일 |
  |---|---|---|
  | kernelport 10개 | 템플릿이라 인스턴스가 생기지 않음. `vector_add`를 `float`로 풀면 오차 0으로 실행됨 | 템플릿 인스턴스 생성 |
  | kernelport 2개 | 생성된 MSL 오류 (`rsqrtf` 없음, `block_sum` 호출 불일치) | 내장 함수 보완 |
  | nvdiffrast 4개 | `__match_any_sync`, `__frcp_rz` 미지원. 앞의 것은 warp 단위 동기화 | Metal 대응 설계 또는 거부 |
  | sam2 1개 | 호스트 코드와 섞여 있고 `ATen` 헤더를 포함 | 장치 코드 분리 |

  실패는 모두 이유를 밝힌 오류였습니다. 잘못 번역된 코드가 조용히 실행된 경우는 없었습니다. 외부 커널의 값은 아직 비교하지 못했습니다.
- **M4는 환경 수가 많아야 이득입니다.** 512개에서는 Metal이 CPU의 1.0~1.9배(임의 행동)에 그칩니다.
- **Genesis 처리량은 흔들립니다.** 같은 장면을 반복하면 최대 3배까지 차이가 납니다(서 있는 장면, Metal 4096개: 13만~39.7만). 원인은 찾지 못했고, 최고값은 첫 실행의 값입니다. 앞으로는 여러 번 재서 범위로 보고합니다.
- **Genesis Metal과 CPU의 통계는 가깝습니다.** 임의 행동 4096개의 평균 높이 차이는 0.08%이고, 같은 백엔드 반복은 소수 7자리까지 같습니다. 등급 B의 첫 근거지만, 허용치는 아직 정하지 않았습니다.
- **조건 하나를 바꿨습니다.** Mitsuba의 비교 기준을 `llvm_ad_rgb`에서 `scalar_rgb`로 바꿨습니다. 이 Mac에 `libLLVM`이 없기 때문입니다. Genesis 비교는 CPU 512개 대 Metal 4096개에서 같은 환경 수로 고쳤습니다. 3배 기준은 그대로입니다.

**M0 이후 Isaac Sim 시험 (2026-11-02):**

- Linux arm64 컨테이너에서 Isaac Sim 6.2의 Kit과 CPU PhysX가 돌았습니다. RTX와 카메라는 실패했습니다. 재현 스크립트는 OrchardSim의 `spikes/isaac_sim_container_record.sh`이고, EULA 동의는 실행하는 사람이 직접 설정해야 합니다.
- V100에서도 Kit과 CPU PhysX만 돕니다. RTX는 `Skipping unsupported non-RTX GPU`로 거부됩니다. 따라서 M5의 V100 비교는 Mitsuba CUDA로만 가능합니다.
- Mac 컨테이너(arm64)와 V100 서버(x86_64)의 CPU PhysX 값이 출력된 모든 자릿수까지 같습니다. 작은 장면 2개의 결과입니다. 이 경로는 M4에서 Isaac 쪽 기준 물리로 쓸 수 있습니다.
- 같은 cartpole USD의 막대 각도가 1초 후 PhysX 3.31 rad, Genesis 1.97 rad로 다릅니다. 원인은 조사하지 않았습니다. M4의 등급 B 비교에서 먼저 풀어야 합니다.
- **Isaac Lab을 Isaac Sim 없이 Mac CPU에서 실행했습니다.** Newton으로 cartpole을 100회 학습하는 데 약 18초 걸렸고, 처리량은 환경 256개에서 약 4만 step/s입니다. 시드 3개 모두 보상이 올랐고, 마지막 에피소드 길이는 300, 293, 246입니다(최대 300). 수동 구성이며 NVIDIA가 지원하지 않습니다. `uv sync`는 macOS를 거부합니다. 측정값은 `validation/isaac-lab-mac.json`, 재현은 OrchardSim의 `spikes/isaac_lab_mac.sh`입니다.
- **같은 경로로 Unitree Go2 평지 보행을 학습했습니다.** CPU 물리라 약 3,700 step/s이고, 환경 4096개 × 300회에 약 2.2시간 걸렸습니다. MLX로 돌린 정책은 시뮬레이션에서 명령 +0.8 m/s에 +0.74, -0.8에 -0.87, 0에 0.00으로 따라가고, 넘어짐은 환경당 0.07회 이하입니다. 대조군은 모두 약 0 m/s입니다. 환경 1024개 × 300회 정책은 서기만 하고 걷지 못했습니다. 시드 1개, CPU 물리이며 실제 로봇이나 PhysX와는 비교하지 않았습니다. 측정값은 `validation/isaac-lab-go2.json`입니다.
- 그 체크포인트의 MLX 정책이 rsl_rl 원본 출력과 7.9e-8까지 일치합니다. 이 정책이 Newton 환경을 직접 제어해 시드 3개에서 막대를 100% 세웠습니다. 힘을 주지 않는 대조군은 0%, 1회차 체크포인트는 5%입니다.

**범위 결정 (2026-11-03):** Isaac Sim과 Isaac Lab의 macOS 작업을 비공개 저장소 [OrchardSim](https://github.com/Topabaem05/OrchardSim)으로 분리했습니다. CuMLX에는 검증기와 레시피에 필요한 결과만 남깁니다.

**확인하지 못한 것:** 실제 CUDA에서의 kernelport 원본 검증, 복잡한 USD 자산(cartpole 하나만 시험), Mitsuba가 실제로 하드웨어 레이트레이싱을 쓰는지, 정책 추론을 포함한 처리량, `mps-*`, metalsplat, mlx3D 실행(M2 첫 작업으로 넘김).

- [x] 표의 항목을 모두 실행하고 `validation/spikes.json`에 기록
- [x] 이 로드맵을 결과에 맞게 수정

### M1. 레시피 카탈로그 (완료 2026-11-02) → `v0.5.0`

`cumlx scan`은 무엇이 문제인지 말해 주지만, 해결은 사람이 찾아야 합니다. 그 간격을 메웁니다.

- `cumlx recipe <패키지 또는 scan 결과>`: 대체품, 설치 명령, 증명 등급, 알려진 한계를 출력
- 카탈로그는 데이터 파일이고 항목마다 근거와 확인 날짜를 기록
- `cumlx scan`이 의존성 파일(`requirements`, `pyproject`)도 읽고 레시피를 연결
- `cumlx port` 확장: mlx-lm 미지원 모델이면 mlx-vlm, mflux로 안내

완료 조건:

- [x] 레시피 20개 이상, 항목마다 증명 등급과 근거 표시. 33개입니다. 그중 14개는 실행한 선택지가 있고, 19개는 문서만 확인했습니다
- [x] 실행으로 확인한 레시피와 문서만 확인한 레시피가 출력에서 구분됨 (`evidence`, `best_evidence`, `W_RECIPE_NOT_RUN`)
- [x] 스캔한 저장소 5개에서 모든 발견 항목에 레시피 또는 "대체 없음"이 나옴. 6개 저장소에서 미연결 0건입니다 ([측정값](../validation/recipe-survey.json)). 레시피가 그 저장소에서 실제로 동작하는지는 증명하지 않습니다

### M2. PyTorch 호환 실행 (11/17 ~ 11/28) → `v0.6.0`

소스 수정 없이 CUDA 전제의 PyTorch 스크립트를 Mac에서 실행합니다. 가장 많은 프로젝트에 효과가 있습니다.

```bash
cumlx run train.py --target mps --report run-report.json
```

- 첫 작업: M0에서 실행하지 않은 `mps-*`, metalsplat, msplat, mlx3D를 실행해 레시피에 넣을지 결정. **실행했습니다(2026-11-03).** 레시피에 넣은 것은 metalsplat, torch 2.13의 mps-flash-attn, torch 자체 sdpa와 torchvision의 `deform_conv2d`입니다. 기준은 CPU와 손 계산이며 CUDA와는 비교하지 않았습니다. `validation/m2-substitutes.json`
- `.cuda()`, `device="cuda"`, `torch.cuda.*`, `autocast("cuda")`를 MPS로 연결
- `flash_attn`, `bitsandbytes`, `pytorch3d`, `gsplat` 등의 import 연결은 `--substitute`로 직접 지정할 때만 합니다. **자동 연결은 하지 않기로 했습니다.** 평가한 후보 중 원래 패키지의 API를 그대로 제공하는 것이 없습니다
- 바꾼 호출을 모두 `run-report.json`에 기록
- MPS에 없는 연산은 조용히 CPU로 넘기지 않고 오류 또는 명시적 허용

```text
호환 계층이 지키는 규칙
- torch.cuda.get_device_name()은 "MPS (cumlx run)"을 돌려줍니다. NVIDIA라고 하지 않습니다.
- `cumlx run` 없이는 아무것도 바뀌지 않습니다.
```

완료 조건:

- [x] 수정하지 않은 오픈소스 스크립트 5개 실행, `run-report.json` 생성. nanoGPT 학습과 샘플링, minGPT, PyTorch-GAN, pytorch-tutorial CNN이 MPS에서 끝까지 돌았습니다(시드별 13회). 등급 C입니다. 측정값은 `validation/m2-scripts.json`입니다
- [x] 그중 3개 이상을 V100 CUDA golden과 비교해 등급 A 또는 B. 5개 중 4개가 통과했습니다. `cnn_mnist`와 `gan_mnist`는 등급 A(첫 두 스텝, 최악 8.4e-8), `mingpt_adder`와 `nanogpt_train`은 등급 B(시드 3개, 4구간)입니다. 시드 값을 어긋나게 하면 등급 A가 B로 떨어지고, 기준 손실에 1.2를 곱하면 diverged입니다. `nanogpt_sample`은 손실 기록이 없어 비교하지 못했습니다. 등급 A는 처음 두 스텝만, 등급 B는 허용치(표준오차 3배와 평균 1% 중 큰 값)보다 큰 차이만 봅니다. 손실만 비교하고 저장한 모델은 비교하지 않았습니다. [기록](../validation/m2-rundiff.json)
- [x] 지원하지 않는 호출이 오류로 끝나는 테스트 (조용한 CPU 폴백 없음)
- [x] 호환 계층을 끄면 원래 동작과 같음을 테스트로 확인. `cumlx`를 import해도 torch가 바뀌지 않고, `uninstall()`이 모든 속성을 되돌립니다

### M3. CUDA 커널 번역 (12/01 ~ 12/19) → `v0.7.0`

`kernel` 단계의 핵심입니다. 자동 번역 뒤 실제 CUDA 결과로 증명합니다.

```bash
cumlx kernel capture kernels/ --cases cases.json -o kgolden/   # NVIDIA 장비: 원본 커널을 실행해 기록
cumlx kernel port kernels/foo.cu -o foo.metal                  # 번역
cumlx kernel verify foo.metal kgolden/                         # Mac: 비교
```

- woof 번역기를 어댑터로 연결. `metal-woof`는 netCDF4, cdsapi 같은 날씨 모델 의존성까지 설치하므로 분리 방법을 먼저 정함
- 번역 전 처리 계층을 직접 구현: 표준 헤더 대체, 템플릿 인스턴스 생성, 호스트 코드 분리, PTX 내장 함수 대응 (M0에서 드러난 벽)
- warp 단위 동작처럼 Metal 대응이 없는 기능은 이유를 밝히고 거부
- 연산 단위 golden 형식 `cumlx.opgolden.v1`을 trace 명세에 추가
- `torch.mps.compile_shader`, `mx.fast.metal_kernel`로 실행
- fp64, FTZ, 실행 순서 차이는 보고서에 명시

**기존 자산:** FP4 커널 lab에는 같은 연산의 `.cu`와 `.metal`이 8쌍 있습니다. 번역 결과를 손으로 쓴 Metal과 비교하는 시험 세트로 씁니다.

**현재 상태 (2026-11-03):** 제3자 코드 없이 만들 수 있는 부분을 구현했습니다. 측정값은 [validation/kernel-translation.json](../validation/kernel-translation.json)에 있습니다.

- `cumlx kernel capture`, `verify`, `port`와 `cumlx.opgolden.v1`을 구현했습니다.
- FP4 커널 8개를 Tesla V100(driver 580.178.04, nvcc 12.0.140)에서 처음 실행해 29개 case를 기록했습니다. 손으로 쓴 Metal 커널이 M5 Pro에서 29개 모두 합격입니다(최악 `rel_l2` 7.5e-8, 허용치 1e-5). 허용치는 측정 전에 커밋했습니다.
- 전처리한 lab 커널 8개를 nvcc로 다시 컴파일한 결과는 원본과 byte까지 같습니다. 바꾼 내장 함수는 정수 출력이 같고, 부동소수 출력은 최대 1 ULP 차이입니다.
- 번역기 연결은 **직접 쓴 stub으로만** 시험했습니다. woof를 실행하지 않았으므로, 자동 번역의 범위와 정확도는 아직 측정하지 않았습니다.
- 외부 커널을 돌릴 준비를 마쳤습니다. **아직 측정이 아닙니다.** [spikes/m3_kernels.sh](../spikes/m3_kernels.sh)가 고정한 커밋에서 받은 zoo 12개와 sam2 1개를 V100에서 기록하고, woof로 번역해 비교합니다. 소스는 저장소에 넣지 않습니다. 전처리만 실행한 결과, zoo 11개와 sam2는 통과하고 `rmsnorm`은 `__shfl_xor_sync` 때문에 거부되며(`--allow-warp-level`을 주면 번역기로 넘기고 값으로 판정합니다), nvdiffrast 4개는 거부됩니다. [기록](../validation/kernel-m3.json) nvdiffrast 거부 이유 중 `operator`의 device global과 `make_float4 -> make_float4`류의 재귀는 이 계층의 오류여서 고쳤습니다. 다시 전처리한 결과 파일마다 거부가 2건씩 줄어 8, 10, 16, 24건이 됐고 새 이유는 없습니다. 통과하는 커널의 전처리 결과 해시는 그대로입니다. **4개 파일 모두 여전히 거부됩니다.** 전부 `__match_any_sync`를 쓰기 때문입니다. `zero_value` 전체 특수화 3건은 이 계층이 처리하지 못하는 것이고, Metal의 한계는 아닙니다

완료 조건:

- [ ] V100에서 원본을 실행해 기록한 커널 golden 20개 이상. **미달: 커널 8개**(case 29개)입니다. 그 외에 시험용으로 직접 쓴 커널 3개가 있지만 세지 않습니다. zoo 12개와 sam2 1개의 case 71개를 준비했고, V100이 유휴일 때 `sh spikes/m3_kernels.sh capture`로 기록하면 21개가 됩니다. 2026-11-03에는 V100이 다른 작업으로 사용 중이었습니다
- [ ] 그중 번역 후 등급 A가 70% 이상, 실패한 커널은 이유를 분류해 보고. **부분 측정:** woof 1.1.5.2를 실행했습니다. V100 기준값이 있는 저장소 커널 8개 중 6개가 등급 A입니다. `fp4_gemv`와 `silu_mul`은 `__half` 케이스에서 번역에 실패합니다(woof에 이 타입이 없습니다). 조건은 21개 중 15개이므로 외부 13개 중 9개가 더 필요하고, 외부 커널은 기준값이 없어 아직 비교하지 못했습니다.
- [ ] kernelport zoo 12개를 수정 없이 번역, 실행하고 V100 기준값과 비교. **번역까지 실행, 비교는 아직:** woof가 11개를 번역했고 Metal 컴파일도 통과했습니다(실행은 안 했습니다). `rmsnorm`은 `__shfl_xor_sync`로 거부됩니다. V100 기준값이 없어 값 비교는 0개입니다
- [ ] 실제 저장소 하나의 커널 전체를 처리 (sam2 또는 nvdiffrast). **위험:** M0에서 둘 다 실패했습니다. 일정 안에 안 되면 미달로 보고하고 기준을 낮추지 않습니다. **미실행.** `__match_any_sync`와 `__frcp_rz`는 이제 이유를 밝혀 거부합니다. nvdiffrast 4개는 전처리에서 거부되는 것을 실행해 확인했습니다. 저장소는 sam2로 정했습니다(커널 6개). woof 번역과 Metal 컴파일은 통과했고, V100 기준값이 없어 실행과 값 비교는 아직입니다. `merge`는 스레드 순서에 따라 다른 유효한 forest를 만들 수 있어, 순서가 결과를 바꾸지 못하는 입력에서만 비교합니다
- [x] 값이 틀린 번역을 일부러 넣으면 찾아냄 (결함 주입 5종). 8종 9건을 V100 golden으로 시험해 틀린 경우 9건을 모두 찾았고, 같은 커널의 맞는 경우 9건은 모두 합격입니다. 결함 K7은 처음에 통과했습니다. 입력이 그 경로를 실행하지 않았기 때문이고, 허용치는 그대로 두고 case를 추가했습니다

### M4. 로봇 시뮬레이션 (12/22 ~ 2027/01/09) → `v0.8.0`

Isaac Sim 대체가 아니라, **Isaac Lab 스타일 학습 작업을 Mac에서 돌리는 경로**입니다.

- `cumlx.sim`: Isaac Lab 스타일 작업 정의를 Genesis 또는 MuJoCo로 연결. Isaac Lab + Newton CPU 경로가 돌므로, 작업 정의를 옮기기 전에 이 경로로 어디까지 되는지 먼저 확인합니다
- USD, URDF, MJCF 자산 불러오기
- 등급 B 지표: 같은 초기 상태와 행동의 궤적 오차, 같은 정책의 평균 보상
- 학습한 정책을 MLX로 내보내고 `check_op`으로 검증 (이미 있는 `examples/robot-policy` 확장)

**현재 상태 (2026-11-04):** 아래 완료 조건의 학습과 측정은 OrchardSim에서 했습니다. `cumlx.sim`은 그와 별도로 이제 있습니다. cartpole과 Go2 평지 작업을 Genesis와 MuJoCo에 연결하고, MJCF와 URDF를 읽습니다. Isaac Lab 작업이나 자산이 아니라 그 형식을 따른 작업입니다. USD는 이 환경들에 읽을 수단이 없어 돌려 보지 않았습니다. Go2는 환경 4096개 × 300회를 Metal에서 약 5분 만에 학습했고(MLX 정책 +0.81, -0.86, 0.00 m/s, 대조군 0.00, 넘어짐 0), MLX 정책은 rsl_rl과 최악 6.1e-7입니다. 시드 1개이고 시뮬레이션 결과입니다. Genesis와 MuJoCo 비교에서 cartpole은 처음 쓴 허용치를 **넘었습니다**(레일 끝에 닿은 뒤 갈라짐). 그 전 구간은 약 2e-6까지 일치합니다. Go2는 1초 동안 허용치 안입니다(중앙값 0.024). [기록](../validation/m4-sim.json)

완료 조건:

- [x] cartpole, 사족 보행 작업이 Mac GPU에서 학습 완료. 두 작업 모두 물리를 warp-metal로 Apple GPU에서 돌렸고, 판정 기준은 실행 전에 커밋했습니다. cartpole은 시드 3개 모두 통과했습니다(세운 비율 1.0, 0.9997, 1.0, 대조군 0.051 이하). 학습기는 CPU입니다. Go2는 학습기가 MPS이고, 새 시드 2개 중 1개만 통과했습니다. 시드 2는 +0.87, -0.87, 0.00 m/s입니다. **시드 1은 학습은 끝났지만 +0.8 m/s에서 넘어짐이 0.14로 기준 0.1을 넘어 실패입니다.** 시드 42는 기준을 쓰기 전의 실행입니다(+0.87, -0.88, 0.00). 시뮬레이션 결과이고 PhysX나 실제 로봇과는 비교하지 않았습니다. OrchardSim `validation/m4-cartpole-metal.json`, `validation/m4-go2-seeds.json`
- [x] 기준 물리(MuJoCo)와의 궤적 오차를 사전에 정한 허용치로 등급 B 판정. MuJoCo Warp on `metal:0`과 MuJoCo C(float64)를 같은 시작과 행동으로 비교했습니다. 256개 world, 200 step, 시작 3개, 반복 2회로 6회 모두 통과했습니다. 최악 qpos 오차의 중앙값 7.7e-5~1.0e-4(허용 1e-2), p90 7.9e-3~3.9e-2(허용 5e-2)이고, CPU의 Warp 대조군과 비슷합니다. 주입한 결함 2종은 걸러집니다. **모델은 Go2가 아니라 humanoid입니다.** 드문 world는 1.31까지 벗어나며 이 허용치는 그 값을 제한하지 않습니다. OrchardSim `validation/m4-trajectory.json`
- [x] 실제 Isaac Lab 체크포인트 하나를 MLX에서 실행, 등급 A. cartpole 4-32-32-1, rsl_rl 출력과 최악 7.9e-8. Go2 사족 보행 정책도 MLX에서 최악 2.48e-7이지만, 기준은 rsl_rl 출력이 아니라 직접 만든 torch 구현입니다
- [ ] 처리량(env-step/s)을 여러 번 측정해 범위로 공개. 목표: 4096 환경에서 10만 step/s 이상. **측정은 끝났고 공개는 아직입니다.** 저장소가 비공개입니다. Go2 평지, 물리 Metal, MLX 정책 추론 포함, 학습기 제외, 5회 반복입니다. Genesis는 173,832~327,866(중앙값 181,172)으로 목표를 넘습니다. Isaac Lab은 29,771~38,849(중앙값 30,994)로 **목표에 못 미칩니다.** Genesis는 Isaac Lab과 다른 엔진이고 작업과 모델도 다릅니다. 첫 반복이 다른 값의 약 2배인데 원인은 찾지 못했습니다. 다른 작업이 함께 돌았습니다. OrchardSim `validation/m4-throughput.json`

### M5. 레이트레이싱과 센서 (01/12 ~ 01/23) → `v0.9.0`

RTX가 필요하던 렌더링 경로를 Metal 레이트레이싱으로 대체합니다.

- `cumlx.render`: USD 장면을 Mitsuba Metal로 렌더링
- `cumlx.sim`의 카메라 센서로 연결 (RGB, 깊이, 세그멘테이션)
- LiDAR 같은 레이캐스트 센서를 Metal 레이트레이싱으로 구현

**현재 상태 (2026-11-04):** `cumlx.render`는 Python 사전으로 만든 Mitsuba 장면에서 RGB, 깊이, 세그멘테이션을 만듭니다. `cumlx.render.usd`는 USD의 작은 일부(메시, 원근 카메라 1개, RectLight, DistantLight, 확산 색)를 Mitsuba 장면으로 바꾸고, 그 밖의 것은 이름을 밝혀 거부합니다. 같은 장면을 Mitsuba로 직접 쓴 것과 Metal에서 비교해 두 장면 모두 등급 B입니다(최악 58.78 dB, 68.48 dB, MSE 비 1.004, 허용치는 M5 그대로, 일부러 넣은 결함 6건은 모두 실패, [기록](../validation/m5-usd.json)). 이 저장소에서 만든 작은 장면이고 다른 도구의 USD 자산은 돌려 보지 않았으며, Sensors나 `cumlx.sim`과는 아직 연결하지 않았습니다. `cumlx.sim` 카메라는 작업의 링크를 상자로 그려 물리 상태와 맞춥니다(21개 프레임 모두 통과, 처음 쓴 무게중심 기준은 실패, [기록](../validation/m4-sim.json)). 레이캐스트 센서는 `cumlx.render.lidar`에 있고, 빔 폭과 반사율, 잡음이 없는 기하 측정입니다. 정확한 거리와 최악 5.6e-6 m 차이입니다(허용 1e-3 m, 틀린 경우 3종은 실패, [기록](../validation/m5-lidar.json)). Mitsuba의 Metal 변형은 Metal 레이트레이싱 API를 씁니다. 이 GPU에서 그 API가 하드웨어로 가속된다는 부분은 Apple 문서의 내용이고, 카운터로 확인하지는 않았습니다.

완료 조건:

- [ ] 같은 장면의 Mitsuba Metal과 V100 CUDA 결과 비교, 등급 B (PSNR 허용치는 측정 전에 지정). 허용치는 [측정 전에 커밋](../validation/m5-render-limits.json)했고, Metal 쪽 이미지와 비교 도구는 준비했습니다. **V100 이미지가 아직 없습니다.** Mitsuba 3.9.1은 V100을 거부합니다(Dr.Jit 1.4 이상은 compute capability 7.5 이상 필요, V100은 7.0). 첫 실행은 70분간 CPU 100%로 멈춰 중단했고 그 원인은 찾지 못했습니다. [바꾼 절차](../validation/m5-render-limits.json)는 V100에 3.8.0을 쓰고, 허용치는 그대로이며, 두 버전의 `scalar_rgb` 비교를 대조군으로 둡니다. 대조군이 통과하지 못하면 이 항목은 달성이 아닙니다. 3.8.0에는 Metal이 없어 양쪽 버전을 맞출 수 없습니다. 3.8.0이 V100의 OptiX에서 돌지는 아직 모릅니다. Metal과 CPU `scalar_rgb` 비교는 두 장면 모두 등급 B이고, 주입한 결함 2종은 두 장면 모두 실패합니다. 이것으로 이 항목을 대신하지 않습니다
- [ ] 256×256 RGB 센서 프레임 속도 측정 및 공개. **측정은 끝났고 공개는 아직입니다.** 저장소가 비공개입니다. 초당 프레임: Cornell box 16 spp 127~134, 64 spp 50~53. 매 프레임 메시를 옮기는 장면 16 spp 224~314, 64 spp 167~225. 각 5회 반복 [측정값](../validation/m5-render.json). 다른 작업이 함께 돌았습니다. 장면이 작습니다
- [x] 물리와 카메라를 함께 쓰는 예제 하나 실행. [drop_box.py](../examples/physics-camera/drop_box.py): Genesis와 Mitsuba가 한 프로세스에서 Metal로 돕니다. 정육면체 하나의 장면이고, 61개 프레임 모두 물리 상태와 일치합니다(최악 0.48 px). 대조군은 움직이지 않고, 틀린 자세와 틀린 카메라는 실패합니다
- [x] RTX 센서와 다른 점을 문서로 명시. [docs/RENDER-SENSORS.md](RENDER-SENSORS.md). RTX 쪽은 문서만 읽었고 직접 비교하지 않았습니다

### M6. 통합 릴리스 (01/26 ~ 01/29) → `v1.0.0`

- `cumlx doctor`가 전체 스택을 점검. `cumlx doctor --stack`으로 만들었습니다. 이 Python 환경의 패키지와 장치만 보고, 값이 맞는지는 보지 않습니다
- 기함 작업 5개를 문서대로 재현 (아래 표)
- 위 내용으로 README 재작성. 현재 기능과 한계는 반영했고, 전체 재작성은 V100 결과가 나온 뒤에 합니다

완료 조건:

- [ ] 기함 작업을 새 Mac에서 처음부터 재현. **미실행:** 새 Mac이 없습니다. 지금까지의 기록은 [보고서의 기함 작업 표](PLATFORM-REPORT.md#기함-작업)에 있고, 처음부터 재현된 작업은 없습니다
- [ ] 결과 보고서 공개, 모든 항목에 증명 등급 표시. [보고서](PLATFORM-REPORT.md)는 작성했습니다. **공개는 아직입니다.** 저장소가 비공개입니다
- [ ] 미검증 영역 목록 공개. [목록](UNVERIFIED.md)은 작성했습니다. **공개는 아직입니다.** 저장소가 비공개입니다

### 기함 작업

플랫폼이 실제로 쓸모 있는지 판단하는 기준입니다.

| 작업 | 원래 필요한 것 | Mac 경로 | 목표 등급 |
|---|---|---|---|
| FlashAttention + bitsandbytes로 LLM 추론 | CUDA | `cumlx run` 또는 `cumlx port` | A |
| 3D Gaussian Splatting 학습 | CUDA 래스터라이저 | `cumlx run` + metalsplat | B |
| CuPy 또는 커스텀 `.cu` 커널 앱 | CUDA | `cumlx kernel` | A |
| Isaac Lab 사족 보행 정책 학습과 추론 | Isaac Sim, RTX | `cumlx.sim` + MLX | 추론 A, 물리 B |
| 물리 기반 카메라 센서 | RTX | `cumlx.render` | B |

## 6. 위험

| 위험 | 가능성 | 대응 |
|---|---|---|
| 커널 자동 번역의 범위가 좁음 | 높음 (M0 확인) | 전처리 계층 구현. 안 되는 커널은 이유를 밝혀 거부. 일정이 넘치면 M3의 범위를 줄이고 보고 |
| 문서와 달리 Metal 경로가 불완전 | 중간 | M0에서 네 후보는 실행으로 확인. `mps-*`, metalsplat, mlx3D는 M2에서 먼저 실행 |
| 작은 프로젝트가 중단됨 (woof ★1, kernelport ★0, mpsops) | 높음 | 버전 고정, 어댑터 격리. 라이선스가 허용하면 fork 가능 |
| 자동 번역이 값을 틀림 | 높음 | CUDA golden으로 항상 검증. 틀린 결과는 FAIL로 보고 |
| CUDA에 맞춘 알고리즘이 Metal에서 느림 | 중간 | 측정하고 공개. 속도를 약속하지 않음 |
| 호환 계층이 기대를 부풀림 | 중간 | 보고서에 대체 내역과 등급을 항상 표시 |
| 메모리 부족 (48GB 통합 메모리, 16GB V100) | 중간 | 작은 작업부터. 필요 메모리를 보고 |
| GPU 세대 차이 (M5만 보유) | 중간 | 다른 세대는 미검증으로 표시. TF32처럼 세대별 차이를 측정으로 확인 |
| 범위가 넓어 끝나지 않음 | 높음 | M1~M3을 먼저 완료. M4·M5는 M0 결과에 따라 축소 가능 |

## 7. 결정

결정한 것 (2026-11-02):

- **호환 계층 정책:** `cumlx run` 안에서 `torch.cuda.is_available()`은 True를 돌려줍니다. 장치 이름은 MPS로 표시하고, 이 동작을 `run-report.json`에 기록합니다.
- **M0 실행:** 외부 패키지를 설치해 실행하는 것을 허용했습니다.

아직 결정하지 않은 것:


| 항목 | 내용 | 권장 |
|---|---|---|
| **제품 범위** | 검증기 하나로 둘지, 패키지를 나눌지 | `cumlx-lab`에 extras로 시작, 커지면 분리 |
| **kernelport와의 관계** | 연동 또는 자체 구현 | 연동. 번역은 woof, 포트 검증과 시험 케이스는 kernelport를 쓰고, CUDA golden과 보고서는 CUMLX가 맡음. 단, wheel에 zoo가 없고 프로젝트가 작아 버전을 고정 |
| **M5 기준 장비** | RTX 센서와 직접 비교하려면 RTX 장비가 필요. V100은 Isaac Sim이 RTX를 거부함(실행 확인) | 우선 Mitsuba CUDA 비교만. 장비가 생기면 추가 |
