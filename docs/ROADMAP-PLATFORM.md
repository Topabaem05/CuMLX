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
| 물리 | [Newton](https://github.com/newton-physics/newton), [Warp](https://github.com/NVIDIA/warp) | Apache-2.0 | **실행** (Mac은 CPU만) | 호환 대상, Metal 없음 |
| **GPU 언어** | [Quadrants](https://github.com/Genesis-Embodied-AI/quadrants) ★228 | Apache-2.0 | **실행**: Metal과 CPU 차이 7.0e-8 | Warp 스타일 커널의 이식 대상 |
| **레이트레이싱** | [Mitsuba 3.9](https://github.com/mitsuba-renderer/mitsuba3), [Dr.Jit](https://github.com/mitsuba-renderer/drjit) | BSD 3조항 형식, BSD-3-Clause | **실행**: `metal_ad_rgb` 렌더링. 하드웨어 레이트레이싱 사용 여부는 문서로만 확인 | RTX 대체 1순위 |
| 렌더링 | [LuisaRender](https://github.com/LuisaGroup/LuisaRender) ★629 | BSD-3-Clause | Mac 지원 미확인 | M5에서 평가 |
| 렌더링 | Genesis Nyx (`gs-nyx` 0.1.9) | 표기 없음 | PyPI에 Linux, Windows wheel만 있음 | 제외 |
| **장면** | [OpenUSD](https://github.com/PixarAnimationStudios/OpenUSD) ★7.6k | TOST 1.0 | **실행**: `usd-core` 읽기, 쓰기 | Isaac 자산 읽기 |
| **커널 번역** | [woof fork](https://github.com/jacobbieker/woof) `backend/translate` ★1 | Apache-2.0 | **실행**: 전용 형식 커널은 정확, 외부 커널 17개는 그대로 실패 (M0) | 번역기 1순위. 전처리 계층을 직접 만들어야 함 |
| 커널 증명 | [kernelport](https://github.com/PS12007/kernelport) ★0 | MIT | **실행**: Metal 포트 12개 PASS, 틀린 포트 FAIL. 원본 CUDA 검증은 미실행 | 연동 후보. 겹치는 부분이 큼 |
| 커널 작성 | [Slang](https://github.com/shader-slang/slang) ★5.8k | Apache-2.0 WITH LLVM-exception | 문서: Metal, CUDA 동시 출력 | 새로 쓰는 커널 |
| **PyTorch 대체** | [mps-bitsandbytes 등 mpsops](https://pypi.org/project/mpsops/) ★11 | **표기 없음** | 문서 | 사용자 직접 설치만. 포함 배포 불가 |
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

- Linux arm64 컨테이너에서 Isaac Sim 6.2의 Kit과 CPU PhysX가 돌았습니다. RTX와 카메라는 실패했습니다. 재현 스크립트는 `spikes/isaac_sim_container.sh`이고, EULA 동의는 실행하는 사람이 직접 설정해야 합니다.
- V100에서도 Kit과 CPU PhysX만 돕니다. RTX는 `Skipping unsupported non-RTX GPU`로 거부됩니다. 따라서 M5의 V100 비교는 Mitsuba CUDA로만 가능합니다.
- Mac 컨테이너(arm64)와 V100 서버(x86_64)의 CPU PhysX 값이 출력된 모든 자릿수까지 같습니다. 작은 장면 2개의 결과입니다. 이 경로는 M4에서 Isaac 쪽 기준 물리로 쓸 수 있습니다.
- 같은 cartpole USD의 막대 각도가 1초 후 PhysX 3.31 rad, Genesis 1.97 rad로 다릅니다. 원인은 조사하지 않았습니다. M4의 등급 B 비교에서 먼저 풀어야 합니다.
- **Isaac Lab을 Isaac Sim 없이 Mac CPU에서 실행했습니다.** Newton으로 cartpole을 100회 학습하는 데 약 18초 걸렸고, 처리량은 환경 256개에서 약 4만 step/s입니다. 시드 3개 모두 보상이 올랐고, 마지막 에피소드 길이는 300, 293, 246입니다(최대 300). 수동 구성이며 NVIDIA가 지원하지 않습니다. `uv sync`는 macOS를 거부합니다. 측정값은 `validation/isaac-lab-mac.json`, 재현은 `spikes/isaac_lab_mac.sh`입니다.
- **같은 경로로 Unitree Go2 평지 보행을 학습했습니다.** CPU 물리라 약 3,700 step/s이고, 환경 4096개 × 300회에 약 2.2시간 걸렸습니다. MLX로 돌린 정책은 시뮬레이션에서 명령 +0.8 m/s에 +0.74, -0.8에 -0.87, 0에 0.00으로 따라가고, 넘어짐은 환경당 0.07회 이하입니다. 대조군은 모두 약 0 m/s입니다. 환경 1024개 × 300회 정책은 서기만 하고 걷지 못했습니다. 시드 1개, CPU 물리이며 실제 로봇이나 PhysX와는 비교하지 않았습니다. 측정값은 `validation/isaac-lab-go2.json`입니다.
- 그 체크포인트의 MLX 정책이 rsl_rl 원본 출력과 7.9e-8까지 일치합니다. 이 정책이 Newton 환경을 직접 제어해 시드 3개에서 막대를 100% 세웠습니다. 힘을 주지 않는 대조군은 0%, 1회차 체크포인트는 5%입니다.

**범위 결정 (2026-11-03):** Isaac Sim과 Isaac Lab의 macOS 작업이 커지면 별도 저장소로 분리합니다. CuMLX에는 검증기와 레시피에 필요한 결과만 남깁니다.

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

- 첫 작업: M0에서 실행하지 않은 `mps-*`, metalsplat, mlx3D를 실행해 레시피에 넣을지 결정
- `.cuda()`, `device="cuda"`, `torch.cuda.*`, `autocast("cuda")`를 MPS로 연결
- `flash_attn`, `bitsandbytes`, `pytorch3d`, `gsplat` 등을 import 시점에 레시피의 대체품으로 연결. 지금은 `--substitute`로 직접 지정할 때만 연결합니다. 자동 연결은 `mps-*` 평가 후 결정합니다
- 바꾼 호출을 모두 `run-report.json`에 기록
- MPS에 없는 연산은 조용히 CPU로 넘기지 않고 오류 또는 명시적 허용

```text
호환 계층이 지키는 규칙
- torch.cuda.get_device_name()은 "MPS (cumlx run)"을 돌려줍니다. NVIDIA라고 하지 않습니다.
- `cumlx run` 없이는 아무것도 바뀌지 않습니다.
```

완료 조건:

- [ ] 수정하지 않은 오픈소스 스크립트 5개 실행, `run-report.json` 생성. 재현 스크립트는 `spikes/m2_scripts.sh`이며, 아직 실행하지 않았습니다
- [ ] 그중 3개 이상을 V100 CUDA golden과 비교해 등급 A 또는 B. 비교 도구 `cumlx rundiff`와 허용치는 준비했습니다(fixture만, CUDA 없음). 실제 비교는 아직 하지 않았습니다
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

완료 조건:

- [ ] V100에서 원본을 실행해 기록한 커널 golden 20개 이상
- [ ] 그중 번역 후 등급 A가 70% 이상, 실패한 커널은 이유를 분류해 보고
- [ ] kernelport zoo 12개를 수정 없이 번역, 실행하고 V100 기준값과 비교
- [ ] 실제 저장소 하나의 커널 전체를 처리 (sam2 또는 nvdiffrast). **위험:** M0에서 둘 다 실패했습니다. 일정 안에 안 되면 미달로 보고하고 기준을 낮추지 않습니다
- [ ] 값이 틀린 번역을 일부러 넣으면 찾아냄 (결함 주입 5종)

### M4. 로봇 시뮬레이션 (12/22 ~ 2027/01/09) → `v0.8.0`

Isaac Sim 대체가 아니라, **Isaac Lab 스타일 학습 작업을 Mac에서 돌리는 경로**입니다.

- `cumlx.sim`: Isaac Lab 스타일 작업 정의를 Genesis 또는 MuJoCo로 연결. Isaac Lab + Newton CPU 경로가 돌므로, 작업 정의를 옮기기 전에 이 경로로 어디까지 되는지 먼저 확인합니다
- USD, URDF, MJCF 자산 불러오기
- 등급 B 지표: 같은 초기 상태와 행동의 궤적 오차, 같은 정책의 평균 보상
- 학습한 정책을 MLX로 내보내고 `check_op`으로 검증 (이미 있는 `examples/robot-policy` 확장)

완료 조건:

- [ ] cartpole, 사족 보행 작업이 Mac GPU에서 학습 완료. cartpole과 Go2 모두 Newton으로 **CPU에서만** 학습했으므로 이 항목은 미달입니다
- [ ] 기준 물리(MuJoCo)와의 궤적 오차를 사전에 정한 허용치로 등급 B 판정
- [x] 실제 Isaac Lab 체크포인트 하나를 MLX에서 실행, 등급 A. cartpole 4-32-32-1, rsl_rl 출력과 최악 7.9e-8. Go2 사족 보행 정책도 MLX에서 최악 2.48e-7이지만, 기준은 rsl_rl 출력이 아니라 직접 만든 torch 구현입니다
- [ ] 처리량(env-step/s)을 여러 번 측정해 범위로 공개. 목표: 4096 환경에서 10만 step/s 이상. M0의 Go2 임의 행동은 8.1만~13.3만이며, 정책 추론은 포함하지 않았습니다

### M5. 레이트레이싱과 센서 (01/12 ~ 01/23) → `v0.9.0`

RTX가 필요하던 렌더링 경로를 Metal 레이트레이싱으로 대체합니다.

- `cumlx.render`: USD 장면을 Mitsuba Metal로 렌더링
- `cumlx.sim`의 카메라 센서로 연결 (RGB, 깊이, 세그멘테이션)
- LiDAR 같은 레이캐스트 센서를 Metal 레이트레이싱으로 구현

완료 조건:

- [ ] 같은 장면의 Mitsuba Metal과 V100 CUDA 결과 비교, 등급 B (PSNR 허용치는 측정 전에 지정)
- [ ] 256×256 RGB 센서 프레임 속도 측정 및 공개
- [ ] 물리와 카메라를 함께 쓰는 예제 하나 실행
- [ ] RTX 센서와 다른 점을 문서로 명시

### M6. 통합 릴리스 (01/26 ~ 01/29) → `v1.0.0`

- `cumlx doctor`가 전체 스택을 점검
- 기함 작업 5개를 문서대로 재현 (아래 표)
- 위 내용으로 README 재작성

완료 조건:

- [ ] 기함 작업을 새 Mac에서 처음부터 재현
- [ ] 결과 보고서 공개, 모든 항목에 증명 등급 표시
- [ ] 미검증 영역 목록 공개

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
