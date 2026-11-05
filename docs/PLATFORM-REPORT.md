# 플랫폼 결과 보고서

기준: CuMLX `a2845d6`와 그 뒤 M3 전처리 재실행, 당시의 OrchardSim `e92054a`, 2026-11-03. 장비는 Apple M5 Pro 48 GB 한 대와 공유 V100 한 대입니다. 모든 수치는 표에 적은 기록 파일에서 읽었습니다. 확인하지 못한 항목은 [UNVERIFIED.md](UNVERIFIED.md)에 있습니다.

이 저장소는 2026-11-04에 공개했습니다. 공개 전에 커밋 기록의 로컬 장비 정보를 지웠으므로, 그 뒤의 커밋 해시는 예전 기록과 다릅니다.

증명 등급은 [로드맵 2절](ROADMAP-PLATFORM.md#2-증명-등급)을 따릅니다. **A**는 수치 동등, **B**는 사전에 정한 허용치 안의 통계 동등, **C**는 실행 확인, **미검증**은 실행하지 않았거나 측정 대상이 아닌 경우입니다.

`validation/archive-orchardsim/`의 기록은 별도 저장소 OrchardSim에서 측정한 값입니다. 그 저장소의 Isaac Sim 작업은 라이선스 조건 때문에 오픈소스로 공개하기 어렵다고 판단해 2026-11-04에 삭제했습니다. **기록은 볼 수 있지만, 만든 스크립트가 공개되어 있지 않아 외부에서 다시 측정할 수 없습니다.**

## 요약

| 마일스톤 | 달성 | 전체 | 남은 조건 |
|---|---:|---:|---|
| M0 가정 검증 | 2 | 2 | |
| M1 레시피 | 3 | 3 | |
| M2 PyTorch 호환 | 4 | 4 | |
| M3 CUDA 커널 | 5 | 5 | |
| M4 로봇 시뮬레이션 | 4 | 4 | |
| M5 레이트레이싱과 센서 | 4 | 4 | |
| M6 통합 릴리스 | 2 | 3 | 새 Mac 재현 |
| **합계** | **24** | **25** | |

**기함 작업 5개 중 새 Mac에서 처음부터 재현된 것은 없습니다.** F3, F4, F5의 규칙에 정한 부분은 같은 Mac에서 빈 환경으로 다시 돌려 통과했습니다([기록](../validation/m6-same-mac-run.json)). [아래](#기함-작업)를 보십시오.

## 완료 조건

"측정함, 미공개"는 측정을 마쳤지만 조건의 "공개"를 충족하지 못해 달성으로 세지 않은 항목입니다.

### M0. 가정 검증

| ID | 조건 | 상태 | 등급 | 기록 | 결과와 한계 |
|---|---|---|---|---|---|
| M0-1 | 표의 항목을 모두 실행하고 기록 | 달성 | C | `validation/spikes.json` | Mitsuba Metal과 CPU의 평균 차이 8.4e-5, PSNR 52.1 dB. Genesis Go2 임의 행동 4096개에서 Metal이 CPU의 6.3~10.1배. Quadrants 오차 7.0e-8. woof는 CUMLX 커널 3개에서 최악 5.0e-8, 틀린 커널은 1.08로 검출. kernelport 12개 PASS이고 원본은 UNVERIFIED. **외부 커널 17개 중 그대로 Metal에서 컴파일된 것은 0개** |
| M0-2 | 로드맵을 결과에 맞게 수정 | 달성 | 미검증 | `docs/ROADMAP-PLATFORM.md` | 문서 작업이며 측정 대상이 아닙니다 |

### M1. 레시피 카탈로그

| ID | 조건 | 상태 | 등급 | 기록 | 결과와 한계 |
|---|---|---|---|---|---|
| M1-1 | 레시피 20개 이상, 항목마다 등급과 근거 표시 | 달성 | C | `src/cumlx/verify/recipes.json` | 33개. 실행한 선택지가 있는 것은 현재 16개이고 나머지 17개는 문서만 확인했습니다. 등급은 선택지마다 다릅니다 |
| M1-2 | 실행과 문서 확인이 출력에서 구분됨 | 달성 | C | `tests/test_recipes.py` | `evidence`, `best_evidence`, `W_RECIPE_NOT_RUN` |
| M1-3 | 스캔한 저장소 5개에서 모든 발견 항목에 레시피 또는 "대체 없음" | 달성 | C | `validation/recipe-survey.json` | 6개 저장소, 미연결 0건. 텍스트 스캔이며 레시피가 그 저장소에서 동작한다는 증거가 아닙니다 |

### M2. PyTorch 호환 실행

| ID | 조건 | 상태 | 등급 | 기록 | 결과와 한계 |
|---|---|---|---|---|---|
| M2-1 | 수정하지 않은 오픈소스 스크립트 5개 실행 | 달성 | C | `validation/m2-scripts.json` | 13회 실행 모두 끝까지 돌았고 수정한 스크립트는 0개입니다. 값이 맞는지는 말해 주지 않습니다 |
| M2-2 | 그중 3개 이상을 V100 CUDA golden과 비교, 등급 A 또는 B | 달성 | B | `validation/m2-rundiff.json` | 5개 중 4개 통과. 4개 모두 등급 B(시드 3개, 4구간)이고, `cnn_mnist`, `gan_mnist`는 첫 두 스텝이 등급 A(최악 8.4e-8)입니다. `nanogpt_sample`은 손실 기록이 없어 비교 불가. 등급 A는 처음 두 스텝만이고 손실만 비교했습니다 |
| M2-3 | 지원하지 않는 호출이 오류로 끝남 | 달성 | C | `tests/test_run_compat.py` | 조용한 CPU 폴백이 없음을 테스트로 확인 |
| M2-4 | 호환 계층을 끄면 원래 동작과 같음 | 달성 | C | `tests/test_run_compat.py` | `cumlx` import는 torch를 바꾸지 않고, `uninstall()`이 속성을 되돌립니다 |

### M3. CUDA 커널 번역

| ID | 조건 | 상태 | 등급 | 기록 | 결과와 한계 |
|---|---|---|---|---|---|
| M3-1 | V100 원본으로 기록한 커널 golden 20개 이상 | 달성 | A | `validation/kernel-golden-v100-m3`, `validation/kernel-translation.json` | 21개: 저장소 8개, 외부 13개(zoo 12, sam2). 외부 case 71개를 nvcc 12.0으로 V100에서 기록 |
| M3-2 | 번역 후 등급 A 70% 이상, 실패 분류 | 달성 | A | `validation/kernel-m3.json` | woof로 19/21(90%). `fp4_gemv`, `silu_mul`은 woof에 `__half`가 없어 실패. `rmsnorm`은 `--allow-warp-level`이 필요하며, 빼면 18/21 |
| M3-3 | kernelport zoo 12개를 번역, 실행, V100과 비교 | 달성 | A | `validation/kernel-m3.json` | 12개 모두 등급 A, 최악 1.0e-7. `rmsnorm`은 플래그가 필요하고 그 입력에서의 일치입니다. 틀린 커널 대조군은 실패 |
| M3-4 | 실제 저장소 하나의 커널 전체 처리 | 달성 | A | `validation/kernel-m3.json` | sam2 진입점 6개 모두 등급 A(case 12개). 각각 따로 실행했고 전체 파이프라인과 앱은 아닙니다. `merge`는 희소 이미지만. nvdiffrast 4개는 `__match_any_sync` 등으로 거부 |
| M3-5 | 값이 틀린 번역을 넣으면 찾아냄 | 달성 | A | `validation/kernel-translation.json` | 틀린 9건 모두 diverged, 맞는 9건 모두 ok. 이 저장소의 커널이며, 입력이 닿지 않는 결함은 잡을 수 없습니다 |

### M4. 로봇 시뮬레이션

| ID | 조건 | 상태 | 등급 | 기록 | 결과와 한계 |
|---|---|---|---|---|---|
| M4-1 | cartpole, 사족 보행이 Mac GPU에서 학습 완료 | 달성 | B | `validation/archive-orchardsim/m4-cartpole-metal.json`, `m4-go2-seeds.json`, `isaac-lab-metal.json` | cartpole 시드 3개 통과(1.0, 0.9997, 1.0, 대조군 0.051 이하), 학습기는 CPU. Go2 학습기는 MPS. 시드 2 통과(+0.87, -0.87, 0.00 m/s). **시드 1은 실패: +0.8 m/s에서 넘어짐 0.14, 기준 0.1.** 시드 42는 기준 작성 전 실행(+0.87, -0.88, 0.00). PhysX와 실제 로봇은 비교하지 않았습니다 |
| M4-2 | MuJoCo와의 궤적 오차, 등급 B | 달성 | B | `validation/archive-orchardsim/m4-trajectory.json` | 6회 모두 통과. 최악 qpos 오차의 중앙값 7.7e-5~1.0e-4(허용 1e-2), p90 7.9e-3~3.9e-2(허용 5e-2). 결함 2종 검출. **모델은 humanoid이고 Go2가 아닙니다.** 드문 world는 1.31까지 벗어나며 허용치가 이를 제한하지 않습니다 |
| M4-3 | Isaac Lab 체크포인트를 MLX에서 실행, 등급 A | 달성 | A | `validation/isaac-lab-mac.json` | cartpole, rsl_rl 출력과 최악 7.9e-8(허용 1e-4). Go2 체크포인트는 2.48e-7이지만 기준이 직접 만든 torch 구현입니다(`validation/isaac-lab-go2.json`) |
| M4-4 | 처리량을 범위로 공개, 4096 환경 10만 step/s 이상 | 달성 | C | `validation/m4-throughput.json` | **목표는 Genesis로만 넘었습니다.** 물리 Metal, MLX 추론 포함, 학습기 제외, 5회. **Isaac Lab 29,771~38,849로 목표 미달.** Genesis 173,832~327,866으로 목표 초과. 다른 엔진이고 작업도 다릅니다. Genesis 첫 반복이 다른 값의 약 2배인데 원인은 찾지 못했습니다 |

### M5. 레이트레이싱과 센서

| ID | 조건 | 상태 | 등급 | 기록 | 결과와 한계 |
|---|---|---|---|---|---|
| M5-1 | Mitsuba Metal과 V100 CUDA 비교, 등급 B | 달성 | B | `validation/m5-render.json`, `validation/m5-render-limits.json` | Cornell box 58.70 dB, 구 장면 45.56 dB(허용 40 dB), MSE 비 1.00, 0.99(허용 2). 결함 2종 실패. V100은 Mitsuba 3.8.0, Metal은 3.9.1로 버전이 다르고, 대조군 `scalar_rgb` 3.8.0과 3.9.1도 등급 B입니다. 장면 2개, 시드 3개 |
| M5-2 | 256×256 RGB 프레임 속도 측정 및 공개 | 달성 | C | `validation/m5-render.json` | 초당 프레임: Cornell box 16 spp 127~134, 64 spp 50~53. 움직이는 메시 16 spp 224~314, 64 spp 167~225. 5회 반복, 다른 작업과 함께 돌렸고 장면이 작습니다 |
| M5-3 | 물리와 카메라를 함께 쓰는 예제 실행 | 달성 | B | `validation/m5-render.json` | Genesis와 Mitsuba가 한 프로세스에서 Metal로 돕니다. 61/61 프레임, 최악 0.48 px(허용 3 px). 정육면체 하나의 장면이며 RTX와 비교한 것이 아닙니다 |
| M5-4 | RTX 센서와 다른 점을 문서로 명시 | 달성 | 미검증 | `docs/RENDER-SENSORS.md` | RTX 쪽은 문서만 읽었고 직접 비교하지 않았습니다 |

### M6. 통합 릴리스

| ID | 조건 | 상태 | 등급 | 기록 | 결과와 한계 |
|---|---|---|---|---|---|
| M6-1 | 기함 작업을 새 Mac에서 처음부터 재현 | 미달 | 미검증 | 없음 | 새 Mac 대신 최근 macOS VM으로 재현하기로 2026-11-04에 정했고, [`terminal-mac`](https://github.com/govijr/terminal-mac)의 macOS 27.2 VM에서 실제로 돌렸습니다. **판정은 fail입니다.** 공개 저장소 설치, doctor, 자체 테스트, GPU 없는 테스트는 통과했지만 F3, F4, F5는 VM의 준가상화 GPU를 라이브러리가 거부해 돌지 못했습니다([기록](../validation/m6-vm-run.json)). 이 VM으로는 이 칸을 채울 수 없고, F1, F2도 남아 있습니다. 실제 Mac이 한 대뿐이라 2026-11-05에 같은 Mac에서 공개 저장소와 빈 홈 폴더로 다시 돌렸고, 같은 규칙으로 **판정은 pass입니다**([기록](../validation/m6-same-mac-run.json)). 같은 Mac이므로 이 칸은 채우지 않습니다 |
| M6-2 | 결과 보고서 공개, 모든 항목에 등급 표시 | 달성 | C | 이 문서 | 2026-11-04에 저장소와 함께 공개. 보관 기록은 다시 측정할 수 없습니다 |
| M6-3 | 미검증 영역 목록 공개 | 달성 | C | [UNVERIFIED.md](UNVERIFIED.md) | 2026-11-04에 저장소와 함께 공개 |

## 기함 작업

목표 등급은 [로드맵](ROADMAP-PLATFORM.md#기함-작업)의 값입니다. 작업의 일부를 뒷받침하는 기록은 있지만, 어느 것도 작업 전체를 처음부터 재현한 기록이 아닙니다.

| ID | 작업 | 목표 | 상태 | 현재 등급 | 기록 | 기록이 보여 주는 것 |
|---|---|---|---|---|---|---|
| F1 | FlashAttention + bitsandbytes로 LLM 추론 | A | 재현 안 됨 | 미검증 | `validation/m6-f1-f2-run.json`, `validation/m2-substitutes.json`, `validation/cuda-golden-v100.json` | 같은 Mac의 빈 환경에서 2026-11-05에 시험한 결과 **원본 프로그램은 실패했습니다.** transformers가 `bitsandbytes` 배포판을 요구해 모델을 올리기 전에 멈추며, mps-bitsandbytes 0.7.0에는 `Linear8bitLt`도 없습니다. Mac 경로인 `cumlx port`는 bf16과 8비트 모두 통과했고, 가중치 하나를 1.01배 한 복사본은 `model.layers.0.mlp.up_proj`에서 거부됐습니다. 이 비교의 기준은 CUDA가 아니라 같은 Mac의 CPU입니다. mps-flash-attn은 torch 2.13의 float32에서 CPU sdpa와 최악 4.6e-7(허용 1e-4)이지만, torch 2.15에는 설치되지 않고 `flash_attn_func`가 없어 그대로 대체하지 못합니다. mps-bitsandbytes는 검사 프로세스가 exit -11로 끝나 실패입니다. 별개로 Qwen2.5-0.5B의 bf16 MLX 포트는 V100 golden과 isolated 최악 9.8e-6(허용 1e-4)로 일치합니다. 이것은 이 작업을 재현한 기록이 아닙니다 |
| F2 | 3D Gaussian Splatting 학습 | B | 재현 안 됨 | 미검증 | `validation/m6-f1-f2-run.json`, `validation/m2-substitutes.json` | 같은 Mac의 빈 환경에서 2026-11-05에 시험한 결과 **gsplat 원본 프로그램은 실패했습니다.** metalsplat에 `rasterization`이 없습니다. Mac 경로인 metalsplat 자체 API 학습은 통과했습니다: MPS에서 held-out PSNR 16.95 dB에서 43.49 dB(허용 25 dB 이상), 멈춘 대조군 16.95 dB, 틀린 카메라 대조군 19.70 dB입니다. 정답 이미지도 metalsplat이 그렸으므로 독립 렌더러와 비교한 값은 아니고, 같은 시드에서도 시도마다 PSNR이 달랐습니다(42.67 dB, 43.49 dB). metalsplat은 손 계산과 비교한 렌더링 검사도 통과했습니다(최고점 오차 0 px). msplat은 학습은 돌았지만(held-out 35.96 dB) 시점 오차 3.14 px로 허용 3 px를 넘어 실패입니다. CUDA 비교는 없습니다 |
| F3 | CuPy 또는 커스텀 `.cu` 커널 앱 | A | 재현 안 됨 | 미검증 | `validation/kernel-translation.json`, `validation/kernel-m3.json`, `src/cumlx/verify/recipes.json` | 앱을 돌린 기록이 없습니다. `tests/test_kernel_gpu.py`는 같은 Mac의 빈 환경에서 30개 통과했습니다(`validation/m6-same-mac-run.json`). 이 저장소의 커널 8개는 직접 쓴 Metal로 V100과 등급 A입니다. 외부 커널 13개는 woof 번역과 V100 비교까지 했지만 앱은 아닙니다. woof의 CuPy 파사드는 `asarray`, `asnumpy`, 커널 실행만 돌린 등급 C입니다 |
| F4 | Isaac Lab 사족 보행 정책 학습과 추론 | 추론 A, 물리 B | 일부 달성, 다른 경로 | 추론 A, 물리 B(일부) | `validation/archive-orchardsim/mlx-closed-loop-o4.json`, `isaac-lab-metal.json`, `m4-go2-seeds.json`, `m4-trajectory.json`, `warp-metal-o2.json`, `validation/m4-sim.json` | 보관 기록의 경로는 Isaac Lab, warp-metal, 런타임 패치이고 스크립트는 공개되어 있지 않습니다. 로드맵의 `cumlx.sim` 경로는 Isaac Lab이 아닌 Genesis의 Go2 작업으로 따로 돌렸습니다(`validation/m4-sim.json`). 시드 1개에서 +0.81, -0.86, 0.00 m/s이고 MLX 정책은 rsl_rl과 최악 6.1e-7, Genesis와 MuJoCo 비교는 1초 동안 통과입니다. 추론은 폐루프 1000 step에서 rsl_rl과 최악 2.70e-7(허용 1e-4)이고, 가중치 하나의 1% 변경은 이 허용치로 잡히지 않습니다. Go2의 CPU와 Metal 비교는 통과(속도 차이 0.002 m/s, 허용 0.05)했지만 MuJoCo Warp 일치 시험은 실패입니다. `cumlx.sim` 경로는 같은 Mac의 빈 환경에서 다시 학습해 M4 기록과 값이 정확히 같았습니다(`validation/m6-same-mac-run.json`). 새 Mac 재현은 없습니다 |
| F5 | 물리 기반 카메라 센서 | B | 일부 달성 | B(Metal과 V100 CUDA, 버전 다름) | `validation/m5-render.json`, `validation/m4-sim.json`, `validation/m5-lidar.json`, `docs/RENDER-SENSORS.md` | `cumlx.render`로 RGB, 깊이, 세그멘테이션을 만들고 예제가 물리 상태와 일치합니다. Metal과 V100 CUDA 렌더는 장면 2개에서 등급 B입니다(Mitsuba 3.9.1, 3.8.0). 센서 전체를 CUDA와 비교하지는 않았습니다. `cumlx.sim` 카메라는 cartpole의 상자 링크로 두 엔진에서 21개 프레임 모두 통과했고(`validation/m4-sim.json`), 레이캐스트 센서는 정확한 거리와 최악 5.6e-6 m입니다(`validation/m5-lidar.json`). USD는 메시, 카메라, 조명 2종, 확산 색만 읽으며, 작은 장면 2개가 같은 장면의 Mitsuba 사전과 등급 B입니다(`validation/m5-usd.json`). 다른 도구의 자산은 시험하지 않았습니다. 렌더링은 같은 Mac의 빈 환경에서 시드 3개를 다시 돌렸고, 이 실행에서는 CUDA와 비교하지 않았습니다(`validation/m6-same-mac-run.json`) |

## 실패와 미달

실패는 실패로 남깁니다. 허용치는 결과를 본 뒤 바꾸지 않았고, 다른 측정을 추가한 경우 첫 판정을 그대로 두었습니다.

| 항목 | 결과 | 허용치 | 기록 | 원인 |
|---|---|---|---|---|
| Go2 시드 1 | +0.8 m/s에서 넘어짐 0.14 | 0.1 | `validation/archive-orchardsim/m4-go2-seeds.json` | 조사하지 않았습니다. 속도와 대조군은 기준 안입니다 |
| Isaac Lab 처리량 | 29,771~38,849 env-step/s | 목표 100,000 | `validation/m4-throughput.json` | 목표에 못 미칩니다. Genesis의 값으로 대신하지 않습니다 |
| MuJoCo Warp 일치, 첫 시험 | qpos 2.02e-3 | 1e-3 | `validation/archive-orchardsim/warp-metal-o2.json` | float32 반올림이 드문 world의 접촉 하나를 바꾸는 것이 유력합니다. CPU도 1 ulp 잡음을 넣으면 같은 허용치를 넘습니다. 모든 커널이 옳다는 증거는 아닙니다 |
| MuJoCo Warp 일치, 둘째 시험 | 9회 중 8회 통과. 접촉 수가 다른 world 15개 | 11개 | `validation/archive-orchardsim/warp-metal-o2.json` | Metal에서 평균 12.3개, 대조군 7.3개로 더 잦습니다. 접촉 순서는 전체 원인이 아닙니다 |
| MuJoCo Warp 한 스텝 진단 | 같은 상태에서 qvel 2.6e-6 차이. 1 ulp 잡음은 1.8e-7 | 없음 | `validation/archive-orchardsim/warp-metal-o2.json` | 대조군 잡음이 실제 차이보다 작았습니다. MuJoCo C(float64)와 비교한 정확도는 Metal과 CPU가 같은 수준입니다(4.1e-6, 4.0e-6). 차이는 운동학 뒤 가속도와 구속 단계에서 커집니다. 원인은 확정하지 못했습니다 |
| `cumlx.sim` cartpole, Genesis와 MuJoCo | 중앙값 0.037~0.049, p90 0.40~0.45 | 0.02, 0.05 | `validation/m4-sim.json` | 수레가 레일 끝에 닿은 뒤 갈라집니다. 그 전 구간을 묻는 둘째 측정은 통과합니다(최악 2.1e-6). 처음 실행은 행동이 경첩까지 움직이는 작업 정의 오류로 실패했습니다 |
| `cumlx.sim` 카메라, 무게중심 기준 | 막대가 투영 위치에서 2.3~5.1 px | 3 px | `validation/m4-sim.json` | 수레가 막대 아래쪽을 가립니다(도형에서 읽은 설명). 표면 기준은 통과합니다(최악 8.8 mm, 허용 10 mm) |
| `cumlx.sim` MLX 정책, 상대 오차 | MuJoCo cartpole 1.30e-4, 1.41e-4 | 1e-4 | `validation/m4-sim.json` | 행동이 약 8e-4인 배치의 float32 반올림으로 읽었습니다. 절대 기준은 통과합니다(최악 1.7e-6, 허용 1e-5) |
| Metal 반복 일치 | 접촉이 있으면 같은 시작에서 달라집니다. qpos 최대 1.39e-2 | 없음 | `validation/archive-orchardsim/warp-metal-o2.json` | 접촉과 제한을 끄면 정확히 같습니다. 시드 하나로 비트 단위 재현이 안 됩니다 |
| Newton 작은 장면 | 원뿔에서 4.9e-3 | 1e-3 | `validation/archive-orchardsim/warp-metal-o2.json` | 찾지 못했습니다. 시작 위치에 따라 약 5e-6 또는 4.89e-3입니다. Newton 자체 충돌 경로는 패치한 복사본에서만 돌고, 공식판은 `fixedarray_t`로 실패합니다 |
| MLX PPO 업데이트, 첫 시험 | 가중치 오차 0.0186, 0.0214(시드 3개 중 2개) | 0.01 | `validation/archive-orchardsim/mlx-ppo-o4.json` | 서로게이트 클립이 기울기를 불연속으로 만듭니다. rsl_rl도 자기와 비교해 같은 허용치를 넘습니다(0.0102, 0.0110) |
| MLX PPO, 자유 실행 측정 | 포트 결함 5건을 고치기 전 중앙값 9.8e-3, 고친 뒤 9.4e-4. 시드 10개 중 5개가 큼. 대조군은 4개 | 중앙값 2.2e-4 | `validation/archive-orchardsim/mlx-ppo-o4.json` | 실패로 남습니다. 클립을 넘지 않는 경우는 1.7e-4(허용 1e-2)입니다 |
| MLX PPO, 넷째 측정 | 시드 120개에서 통과: 1e-3 초과 54%, rsl_rl MPS 52%, 허용 72% | 사전 작성 규칙 | `validation/archive-orchardsim/mlx-ppo-o4.json` | **검토 결과 이 측정으로는 게이트를 열어 두었습니다.** 게이트는 뒤의 float64 측정으로 달성했습니다. actor 규칙은 수정 전 포트와 결함 2종(gradient clip 제거, value clip 제거)을 구분하지 못하고 critic 규칙만 잡습니다. 이 업데이트로 학습을 돌린 적이 없습니다 |
| MLX 기본 TF32로 PPO | 가중치 오차 0.40~0.43 | 0.01 | `validation/archive-orchardsim/mlx-ppo-o4.json` | 이 M5에서 TF32를 켜면 업데이트가 틀립니다 |
| Isaac Sim 연결 첫 실행 | 8회 중 0회 버팀 | 0.75 이상 | `validation/archive-orchardsim/isaac-sim-o5.json` | 이 프로젝트의 관측 순서 오류. 고친 뒤 8회 중 8회입니다 |
| Go2 rough, Metal | NaN 관측으로 종료, 23개 중 22개 실행 | 전부 | `validation/archive-orchardsim/isaac-lab-metal.json` | 같은 설정이 CPU에서도 같은 오류를 냅니다 |
| msplat | 학습 시점 오차 3.14 px | 3 px | `validation/m2-substitutes.json` | 조사하지 않았습니다 |
| mps-bitsandbytes | exit -11로 종료 | 정상 종료 | `validation/m2-substitutes.json` | torch 2.15에서 "must be on MPS". C++ ABI 불일치가 유력하지만 증명되지 않았습니다 |
| mps-deform-conv, correlation, carafe, conv3d | 필수 검사 실패 | 통과 | `validation/m2-substitutes.json` | 위와 같습니다 |

## OrchardSim 작업 종료

2026-11-04에 이 작업을 끝냈습니다. 끝낼 때 조건 39개 중 36개를 채운 상태였고, 기록은 [보관](../validation/archive-orchardsim/)했습니다.

달성한 항목의 기록: Isaac Lab 앱 화면(cartpole, Go2, `metal:0`), 버튼 3개와 백엔드, 부분별 Mac GPU 시간(Go2 64개 60초에 23.4초, CPU 대조군 0초), warp-metal의 MSL 커널 771개, 직접 쓴 MSL 커널 위의 정책 시험, Isaac Sim의 낙하 시험(창 없는 Kit, CPU PhysX), O5 V100 연결, 직접 쓴 MSL cartpole 커널(값마다 float 두 개를 쓰는 커널. float32 커널의 실패는 남습니다. 6개 시작 상태에서 최악 3.9e-9, 허용 5e-2), O4 PPO 업데이트(float64, CPU에서 시드 10개, 최악 1.35e-13, 허용 1e-8. 업데이트의 논리를 비교한 것이고 Mac GPU의 float32 비교가 아닙니다).

| 끝낼 때 열려 있던 조건 | 상태 |
|---|---|
| O6 새 Mac에서 한 명령으로 재현 | 미실행. 그 스크립트는 저장소와 함께 없어졌습니다 |
| O7 Mac에서 Isaac Sim 창 보기 | 설치한 그대로는 실패: 검은 화면. Kit가 GPU 장치를 만들지 못합니다(`Failed to create any GPU devices`) |
| O7 Isaac Sim의 Play와 Stop | 설치한 그대로는 시험 불가: 창과 버튼이 없습니다 |

## Isaac Sim EULA 값에 대한 기록

O5의 V100 실행(2026-11-04, 8회 중 8회, 대조군 2개는 각 0회, 왕복 중앙값 6.09 ms, p99 7.70 ms, ssh 터널 하나)은 사용자가 직접 질문에 답해 EULA 동의를 선택한 뒤 돌렸습니다. cartpole, CPU PhysX, RTX 없음입니다.

O5의 Mac 컨테이너 실행(8회 중 8회, 왕복 중앙값 0.77 ms, p99 1.89 ms)에서는 **서브에이전트가 `OMNI_KIT_ACCEPT_EULA=YES`를 직접 넣었습니다.** 사용자는 2026-11-02에 V100과 Mac VM의 Isaac Sim 시험에 동의했습니다. 그러나 2026-11-03에 세션은 Mac 컨테이너 실행을 사용자의 결정에 맡기겠다고 알렸고, 이 실행에 대한 결정은 없었습니다. 따라서 이 실행은 사용자에게 알린 범위를 넘었습니다. 사용자는 2026-11-04에 이를 검토해 **결과를 인정하고 이후 Mac 컨테이너 시험을 허용했습니다.** Isaac Sim 자체는 수정하지 않았습니다. 기록은 `validation/archive-orchardsim/isaac-sim-o5.json`입니다.
