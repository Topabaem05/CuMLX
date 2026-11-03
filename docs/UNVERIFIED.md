# 미검증 영역

기준: CuMLX `4580acf`, OrchardSim `e92054a`, 2026-11-03. 기록의 `not_shown`, `not_verified`, 로드맵의 내용을 플랫폼 층별로 모았습니다. 결과 전체는 [PLATFORM-REPORT.md](PLATFORM-REPORT.md)에 있습니다.

**이 저장소는 비공개이므로 이 목록은 아직 공개되지 않았습니다.** `OrchardSim` 표시가 있는 기록은 별도의 비공개 저장소에 있습니다.

여기에 없다고 해서 검증되었다는 뜻은 아닙니다. 기록이 다루는 범위 밖은 모두 미검증입니다.

## 공통

- 장비는 Apple M5 Pro 48 GB 한 대와 V100 한 대입니다. 다른 Apple Silicon과 다른 NVIDIA 세대는 측정하지 않았습니다.
- 대부분의 측정은 다른 작업이 함께 도는 Mac에서 했습니다. 시간과 처리량은 벤치마크가 아닙니다.
- 새 Mac에서 처음부터 재현한 적이 없습니다. OrchardSim `spikes/reproduce_go2_metal.sh`는 거부 동작만 시험했습니다.
- `cumlx doctor --stack`은 패키지와 장치가 있는지만 봅니다. nvcc와 NVIDIA 장비는 찾지 않고, Genesis, warp-metal, 번역기는 import하지 않습니다.

## 모델 검증기

- V100 기준은 모델 하나와 V100 한 대입니다. 다른 NVIDIA 세대와 `torch.backends.cuda.matmul.allow_tf32=True`는 측정하지 않았습니다. `validation/cuda-golden-v100.json`
- FP4 커널 실험실의 `MLXBackend`는 한 장치에서만 시험했고 속도는 측정하지 않았습니다.
- 실제 체크포인트 실행에서 돌리지 못한 경우가 2건 있습니다. `validation/dogfood-macos-m5-pro.json`

## PyTorch 호환 실행 (M2)

- **스크립트 5개 중 어느 것도 V100 CUDA와 비교하지 않았습니다.** 등급 C입니다. `validation/m2-scripts.json`
- `cumlx rundiff`의 허용치 근거는 fixture의 CPU와 MPS입니다. CUDA가 참여하지 않았습니다. `validation/run-compare-limits.json`
- 대체 패키지의 기준은 CPU의 torch 또는 손 계산입니다. 어느 것도 원래 패키지 이름으로 그대로 대체되지 않습니다. `validation/m2-substitutes.json`
- mps-deform-conv, correlation, carafe, conv3d, mps-bitsandbytes의 실패 원인: C++ ABI 불일치가 유력하지만 증명되지 않았습니다. torch 2.9.1 실행으로 시험할 수 있습니다.
- mlx3D는 GPU 사용을 확인하지 못해 시험을 끝내지 못했습니다.

## CUDA 커널 (M3)

- **제3자 커널은 V100에서도 Metal에서도 돌지 않았습니다.** `validation/kernel-m3.json`
- 생성한 하네스를 nvcc로 컴파일한 적이 없습니다.
- woof가 번역한 외부 커널 12개는 컴파일만 확인했고 값은 비교하지 못했습니다. V100 기준값이 아직 없습니다.
- 번역 예측은 측정이 아닙니다.
- case는 float32만 다룹니다. zoo의 float16, bfloat16과 가장 큰 실행은 포함하지 않았습니다.
- 이 저장소의 커널 8개는 이식하기 쉽게 쓴 것이고 warp 단위 동작을 쓰지 않습니다. 비정상 수와 오버플로 영역은 인증하지 않았습니다. `validation/kernel-translation.json`
- nvdiffrast의 `zero_value` 전체 특수화 3건은 이 계층이 처리하지 못해 거부됩니다. Metal에서 되는지는 시험하지 않았습니다.
- 입력이 닿지 않는 결함은 잡을 수 없으므로, 통과는 case가 실행한 범위만 뜻합니다.

## 물리와 학습 (M4, OrchardSim)

- `cumlx.sim`은 Isaac Lab이 아닙니다. 작업 2개가 그 형식을 따를 뿐이고, Isaac Lab 작업 정의를 그대로 읽지 못합니다. USD 자산은 돌려 보지 않았고, MuJoCo 쪽 Go2 학습도 없습니다. `validation/m4-sim.json`
- Genesis와 MuJoCo의 관절 제한 처리: cartpole이 레일 끝에 닿으면 두 엔진이 갈라져 처음 쓴 허용치를 넘었습니다. 일치 시험은 한 시간 간격에서 두 엔진이 같은 적분을 한다는 뜻이고, 어느 쪽이 정확한지는 말해 주지 않습니다. 시간 간격 대조군이 그보다 크게 벗어납니다.
- 작은 행동에서 MLX 정책: 균형을 잡은 cartpole의 행동은 약 8e-4라서 상대 오차가 1e-4를 넘습니다(MuJoCo 정책 1.41e-4). 절대 오차는 1e-6 이하입니다. float32 반올림으로 읽었습니다.
- PhysX와의 비교, 실제 로봇. 어떤 결과도 sim-to-real 결과가 아닙니다.
- 같은 cartpole USD가 1초 후 PhysX 3.31 rad, Genesis 1.97 rad로 다르며, 원인은 조사하지 않았습니다. `validation/isaac-sim.json`
- 궤적 시험은 humanoid 모델입니다. Isaac Lab 작업과 Go2 모델은 다루지 않았고, 드문 world의 오차(0.3~1.3)는 허용치가 제한하지 않습니다. OrchardSim `validation/m4-trajectory.json`
- 다른 cartpole 작업, 좌우와 회전 명령, +0.8, -0.8, 0 외의 속도. OrchardSim `validation/m4-cartpole-metal.json`, `m4-go2-seeds.json`
- 보행 품질(발 미끄러짐, 토크, 에너지, 접촉 시점), 방향 드리프트, 긴 실행. `validation/isaac-lab-go2.json`
- Newton 접촉으로 한 전체 학습과 그 접촉의 4096 환경 처리량. 패치의 슬롯 충돌이 불가능하다는 증명도 없습니다. OrchardSim `validation/warp-metal-o2.json`
- warp-metal의 모든 커널이 옳은지. 일치 실패가 커널 오류 없이 설명된다는 것만 보였고, 시험은 humanoid 60 step입니다.
- Metal 반복 불일치의 전체 원인. 접촉 순서를 고정해도 반복이 일치하지 않습니다.
- 스모크 행렬은 작업이 돈다는 것만 보입니다. 학습 여부나 CPU와의 일치는 보이지 않으며, 등록된 146개 중 23개만 시험했습니다. OrchardSim `validation/isaac-lab-o1.json`, `isaac-lab-metal.json`
- MLX PPO 업데이트를 넣은 학습과 실제 rollout. 데이터는 합성입니다. OrchardSim `validation/mlx-ppo-o4.json`
- 폐루프 등급 A의 허용치 1e-4는 가중치 하나의 1% 변경을 잡지 못합니다. OrchardSim `validation/mlx-closed-loop-o4.json`
- Genesis 처리량이 첫 반복에서 약 2배인 원인. 학습기를 포함한 처리량은 다른 기록에 있습니다. OrchardSim `validation/m4-throughput.json`
- Genesis와 Newton의 일치. OrchardSim `validation/genesis-o6.json`
- 지원: NVIDIA는 macOS를 지원 대상으로 올리지 않았고, 이후의 Isaac Lab 커밋이 이 경로를 깨뜨릴 수 있습니다.

## Isaac Sim (OrchardSim O5)

- **V100에서 Isaac Sim과 MLX 정책의 연결은 돌리지 않았습니다.** 연결은 Mac 컨테이너에서만 돌았습니다.
- 컨테이너 실행의 EULA 값은 서브에이전트가 넣었고, 사용자가 아직 검토하지 않았습니다. [보고서](PLATFORM-REPORT.md#isaac-sim-eula-값에-대한-기록)를 보십시오.
- 사족 보행, RTX, 카메라, GPU PhysX. 컨테이너와 V100 모두 RTX를 쓰지 못합니다.
- 컨테이너 연결은 세션 3개, 에피소드 8개씩, 시드 1개입니다. 지연은 한 장비의 Docker 포트를 거친 값이고 네트워크 수치가 아닙니다. OrchardSim `validation/isaac-sim-o5.json`
- 자유 낙하에서 정확한 값과의 0.10 m 차이는 원인을 조사하지 않았습니다.

## 렌더링과 센서 (M5)

- **Mitsuba Metal과 V100 CUDA 비교는 아직 없습니다.** Mitsuba 3.9.1은 V100을 거부합니다(compute capability 7.5 이상 필요, V100은 7.0). 3.8.0 경로는 아직 돌리지 않았고, 돌아도 양쪽 버전이 다릅니다. 첫 실행이 70분간 멈춘 원인도 찾지 못했습니다. `validation/m5-render.json`
- RTX와의 직접 비교. RTX 쪽은 문서만 읽었습니다. `docs/RENDER-SENSORS.md`
- 하드웨어 레이트레이싱: Metal 레이트레이싱 API를 쓰는 것은 확인했지만, 전용 회로가 동작했음을 보여 주는 카운터는 읽지 않았습니다. 하드웨어 가속은 Apple 문서의 내용입니다.
- 만들지 않은 것: USD 장면, 모션 벡터, 렌즈 왜곡, 시맨틱 레이블, 병렬 환경 렌더링, 로봇 메시를 그리는 카메라. `cumlx.sim` 카메라의 링크는 상자입니다.
- 레이캐스트 센서는 기하만 잽니다. 빔 폭, 반사율, 잡음, 회전 중 움직임이 없고, RTX LiDAR와 비교하지 않았습니다. `validation/m5-lidar.json`
- `cumlx.sim` 카메라에서 0.3 m 틀린 카메라는 21개 중 12개 프레임에서만 잡힙니다. `validation/m4-sim.json`
- 프레임 속도는 작은 장면입니다. 1, 4 spp 이미지는 디노이저 없이 쓸 수 없습니다.

## 통합 (M6)

- 기함 작업 5개 전체의 재현. [보고서](PLATFORM-REPORT.md#기함-작업)를 보십시오.
- 복잡한 USD 자산. cartpole 하나만 시험했습니다.
- README는 V100 결과가 나온 뒤에 다시 씁니다. 아직 쓰지 않았습니다.
