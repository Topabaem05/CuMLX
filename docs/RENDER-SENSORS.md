# 카메라 센서: Mitsuba Metal과 RTX 센서의 차이

`cumlx.render`는 Mitsuba 3.9.1로 Apple GPU에서 RGB, 깊이, 세그멘테이션 이미지를 만듭니다. Isaac Sim의 RTX 센서를 대체하지 않습니다. 이 문서는 무엇이 같고 무엇이 다른지 적습니다. 측정값은 [validation/m5-render.json](../validation/m5-render.json)에 있습니다.

아래 표의 **측정**은 이 저장소에서 실행한 결과이고, **문서**는 NVIDIA 또는 Apple 문서를 읽은 내용입니다. RTX 쪽은 전부 문서입니다. 보유한 Tesla V100은 Isaac Sim이 RTX를 거부하므로(`Skipping unsupported non-RTX GPU`, [측정](../validation/isaac-sim.json)) RTX 센서를 직접 실행한 적이 없고, **두 경로의 출력을 픽셀 단위로 비교한 적도 없습니다.**

## 차이

| 항목 | RTX 센서 (Isaac Sim, 문서) | `cumlx.render` (측정) |
|---|---|---|
| 하드웨어 | RTX 하드웨어의 GPU에서 렌더 시점에 시뮬레이션 | Apple GPU의 Metal 레이트레이싱 API. 아래 "하드웨어 레이트레이싱" 참고 |
| 렌더러 | Omniverse RTX | Mitsuba 경로 추적. 이미지에 노이즈가 있고 디노이저를 켜지 않음 |
| 재질 | 비가시 재질(Non-Visual Materials) 포함, 라이다 반환은 방사율과 반사율에 따라 달라짐 | Mitsuba BSDF만. USD와 MDL 재질을 읽지 않음 |
| RGB | `rgb` annotator | 있음. 256×256에서 측정 |
| 깊이 | `distance_to_camera`, `distance_to_image_plane` | 카메라 원점까지의 거리 하나. 이미지 평면 거리는 없음 |
| 세그멘테이션 | `semantic_segmentation` | 도형 번호. 의미 라벨은 없음 |
| 모션 벡터 | `motion_vectors` | 없음 |
| 렌즈 왜곡 | OpenCV 핀홀, 피시아이 모델 | 없음. 이상적인 핀홀뿐 |
| 라이다 | RTX Lidar: 프로필, 스캔 속도, 재질 응답, 여러 프레임에 걸친 스캔 | 없음 |
| 레이더, 음향 | RTX Radar, RTX Acoustic | 없음 |
| 장면 | USD | Python 사전으로 만든 Mitsuba 장면. USD를 읽지 않음 |
| 여러 환경 | Isaac Lab과 연결 | 한 번에 장면 하나. 병렬 환경 렌더링은 없음 |
| 물리 연결 | Isaac Sim 내부 | 매 프레임 메시 꼭짓점을 옮기고 가속 구조를 다시 만듦. Genesis 정육면체 하나로 실행 |

## 하드웨어 레이트레이싱

M0에서 미검증으로 남긴 질문입니다. 2026-11-03에 다음을 확인했습니다.

- **측정:** `metal_ad_rgb`의 실제 커널은 Metal 레이트레이싱 API를 씁니다. Dr.Jit의 Trace 로그에 생성된 MSL이 있고, `raytracing::intersector<raytracing::triangle_data, raytracing::instancing>`와 `instance_acceleration_structure`를 사용합니다. CPU에서 교차를 계산한 뒤 넘기는 구조가 아닙니다.
- **측정:** 이 장비의 GPU는 Apple M5 Pro이고, `supportsRaytracing`이 true, `MTLGPUFamily.apple9`가 true입니다.
- **문서:** Apple은 M3 제품군부터 Mac에 하드웨어 가속 레이트레이싱이 들어왔고, M5에는 3세대 레이트레이싱 엔진이 있다고 설명합니다.
- **측정하지 못한 것:** Metal API는 레이트레이싱 하드웨어가 없는 Apple6 GPU에서도 동작합니다. 따라서 API 사용만으로 하드웨어 사용이 증명되지는 않습니다. 교차 계산이 전용 회로에서 돌았음을 보여 주는 카운터는 읽지 않았습니다. 결론은 "Metal 레이트레이싱 API를 쓰고, 이 GPU에서 그 API는 Apple 문서상 하드웨어로 가속된다"까지입니다.

## 함께 쓸 때 주의할 점

- **깊이의 기준점:** Mitsuba의 `depth` AOV는 near clip 지점부터 잽니다. 그대로 쓰면 기본값에서 약 1 cm 짧습니다. `Sensors`는 이 거리를 더해 카메라 원점 기준으로 돌려줍니다. 첫 실행의 프레임 검사가 이 오차를 찾았습니다.
- **세그멘테이션은 픽셀당 1 샘플입니다.** 두 도형 번호의 평균은 도형이 아니기 때문입니다. 그래서 윤곽이 거칠고, 픽셀 안의 임의 위치를 샘플링합니다.
- **`Sensors`에는 JIT 변형이 필요합니다.** `scalar_rgb`는 거부합니다.
- **`llvm_ad_rgb`는 libLLVM이 없는 Mac에서 쓸 수 없습니다** (M0 측정).

## 근거

- [Isaac Sim RTX Lidar](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/sensors/isaacsim_sensors_rtx_lidar.html), [Camera](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/sensors/isaacsim_sensors_camera.html), [Sensors](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/sensors/index.html). 2026-11-03에 읽음, Isaac Sim 6.0 문서.
- [Apple M3 발표](https://www.apple.com/newsroom/2023/10/apple-unveils-m3-m3-pro-and-m3-max-the-most-advanced-chips-for-a-personal-computer/), [Apple M5 발표](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/), [Metal feature sets](https://developer.apple.com/metal/feature-sets).
