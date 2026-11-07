# `cumlx` CLI — 포팅 동등성 검증기

검증한 범위만 적습니다. 기준일은 2026-11-02이고, 환경은 Apple M5 Pro(Metal, CPU)와 Linux x86_64 CPU입니다.

## 설치

```bash
python -m pip install -e '.[dev]'                  # 핵심 기능 (NumPy만): describe, doctor, report, selftest --core-only
python -m pip install -e '.[dev,capture,diff]'     # + torch, transformers, mlx, mlx-lm
cumlx doctor --probe                               # 사용할 수 있는 명령과 설치 방법을 보고
```

Linux에서는 `diff` extra가 `mlx[cpu]`를 설치합니다. 같은 환경에 기존 `cuda` extra(`mlx[cuda12]`)를 함께 설치하지 마십시오.

## 명령

| 명령 | 하는 일 | 필요 extra |
|---|---|---|
| `describe` | 명령·인자·exit code·error code의 기계용 명세 | - |
| `doctor` | 의존성과 준비된 명령 보고. `--require`로 미준비 시 exit 3. `--stack`은 플랫폼 층(MLX, mlx-lm, torch MPS, Metal 장치, Genesis, Mitsuba Metal, warp-metal, 번역기)을 이 Python 환경 기준으로 보고하고, 하나라도 준비되지 않으면 exit 3. `--env PYTHON`을 반복하면 각 Python이 자기 프로세스에서 감지하고, 어느 한 곳에서 `ok`인 층을 준비로 셉니다. 답하지 못하는 환경도 exit 3입니다. 이 결과는 부품들이 한 프로세스에서 함께 돈다는 뜻이 아닙니다. 설치와 네트워크 접근은 하지 않고, nvcc는 찾지 않습니다 | - |
| `fixture tiny-llama`, `fixture tiny-moe` | 오프라인 테스트용 2-layer 랜덤 Llama 또는 Qwen3 MoE 생성 | capture |
| `capture` | 원본 PyTorch 모델의 모듈별 입출력을 golden trace로 기록 | capture |
| `diff` | MLX 포팅본을 같은 입력으로 실행해 비교하고 처음 어긋난 모듈을 찾음 | diff |
| `conform` | MLX의 NVFP4/MXFP4 decode를 CUMLX bit-정확 코덱과 CPU·GPU에서 전수 비교. `--checkpoint`와 `--against`를 주면 ModelOpt 원본과 MLX repack의 bytes를 비교 | diff |
| `tracediff` | golden trace 두 개를 비교. CUDA와 CPU capture의 차이처럼 기준 자체의 흔들림을 측정 | - |
| `rundiff` | `cumlx run`의 손실 기록을 CUDA와 대상 장치에서 비교해 증명 등급을 매김 | - |
| `scan` | CUDA 저장소나 HF 모델의 소스를 텍스트로 읽어, Apple Silicon으로 옮기는 작업량을 분류. 대상 코드는 실행하지 않음 | - |
| `run` | CUDA를 전제한 PyTorch 스크립트를 수정 없이 MPS에서 실행하고, 바꾼 호출을 모두 기록. 실행은 증명이 아닙니다 | `torch` |
| `recipe` | CUDA 의존성에 대해 Apple Silicon에서 쓸 대체품을 안내. 선택지마다 직접 실행했는지 문서만 읽었는지 표시 | - |
| `kernel capture` | NVIDIA 장비에서 원본 `.cu` 커널을 nvcc로 컴파일해 실행하고 연산 단위 golden을 기록. 코드를 실행합니다 | `nvcc`, `nvidia-smi` |
| `kernel verify` | Metal 커널을 golden 입력으로 실행해 비교. 모르는 커널과 매핑되지 않은 커널은 합격이 아닙니다 | `metal` 또는 `torch` |
| `kernel port` | `.cu` 파일의 헤더, 템플릿, 호스트 코드, PTX 내장 함수를 처리하고 번역기를 호출. Metal 대응이 없는 기능은 이유를 밝혀 거부 | 번역기 |
| `port` | mlx-lm으로 변환한 뒤 가중치 비교, golden 기록, diff까지 한 번에 실행 | capture, diff |
| `weights` | 모델을 실행하지 않고 원본 safetensors와 포팅본 가중치를 비교. 전치·부호·scale 오류를 이름으로 지목 | diff |
| `report` | 기존 `report.json` 요약 (`--max-items`로 다시 읽기) | - |
| `selftest` | 검증기 자체 검증: 깨끗한 포팅 통과 + 오류 주입 11종 위치 검출 + 정적 가중치 비교 | 전체 기능은 capture+diff |

## 빠른 시작 (오프라인)

```bash
cumlx fixture tiny-llama -o /tmp/tiny
cumlx capture /tmp/tiny --token-ids 1,5,9,23,7,99,3 -o /tmp/golden
cumlx diff /tmp/golden /tmp/tiny -o /tmp/report      # 원본 가중치를 mlx-lm이 그대로 읽으므로 ok
cumlx selftest
```

## AI 에이전트 친화 설계

| 설계 | 이유 |
|---|---|
| stdout이 TTY가 아니면 자동으로 JSON 1개만 출력 | 에이전트가 텍스트를 긁지 않고 그대로 파싱 |
| 실행 중 stdout(파이썬·네이티브 출력 포함)을 stderr로 돌림 | 라이브러리 로그가 JSON을 오염시키지 않음 |
| 결과 envelope 고정 (`cumlx.result.v1`) | 모든 명령을 같은 코드로 처리 |
| 안정된 exit code, `incomplete`(5)를 pass와 구분 | 분기 처리가 쉽고, 미검증을 성공으로 오인하지 않음 |
| 오류마다 `code`, 대부분에 `hint`(정확한 해결 명령) | 에이전트가 스스로 복구 |
| `next_steps`에 다음 명령 제안 | 워크플로를 문서 없이 이어감 |
| `--max-items`로 목록 길이 제한, `truncated` 표시, 전체는 파일 | 컨텍스트 창 절약 |
| `--dry-run`, `--overwrite` 없이는 덮어쓰지 않음, CUMLX가 만든 디렉터리만 삭제 | 되돌리기 어려운 실수 방지 |
| `describe`가 argparse에서 직접 명세를 생성 | 문서와 실제 인자가 어긋나지 않음 |
| `selftest`로 환경에서 검증기 동작을 증명 | 결과를 믿어도 되는지 먼저 확인 |

## cumlx run

CUDA를 전제한 PyTorch 스크립트를 수정하지 않고 MPS에서 실행합니다.

```bash
cumlx run train.py --report run-report.json -- --lr 1e-3    # 스크립트 인자는 -- 뒤에
cumlx run --module my.pkg.train -- --epochs 1
cumlx run train.py --target cpu                             # MPS가 없는 Linux 등
cumlx run infer.py --substitute flash_attn=cumlx.compat.flash_attn_sdpa --substitute bitsandbytes=cumlx.compat.bitsandbytes_int8   # 아래 어댑터
cumlx run train.py --substitute gsplat=cumlx.compat.gsplat_metalsplat   # gsplat.rasterization만, 핀홀과 RGB. 나머지 옵션은 오류. metalsplat 설치 필요
```

**스크립트가 받는 값.** `torch.cuda.is_available()`은 True입니다. `get_device_name()`은 `MPS (cumlx run)`을 돌려주며, NVIDIA 이름을 쓰지 않습니다. `cumlx run` 없이는 torch에 아무 변화가 없습니다.

| 호출 | 처리 | 보고서의 `kind` |
|---|---|---|
| `.cuda()`, `device='cuda'`, `.to('cuda')`, `torch.device('cuda')`, `autocast('cuda')`, `GradScaler`, `torch.load`, `torch.cuda.FloatTensor` 등 | MPS로 연결 | `mapped` |
| `torch.cuda.is_available`, `is_initialized` | True를 돌려줌 | `policy` |
| `max_memory_allocated`, `memory_stats`, `get_device_properties` 등 | MPS에 같은 값이 없어 비슷한 값을 돌려줌. `W_RUN_APPROXIMATE` | `approximate` |
| `pin_memory`, `ipc_collect`, `reset_peak_memory_stats` 등 | 이 장치에서는 할 일이 없음 | `noop` |
| `utilization`, `temperature`, `memory_snapshot`, `stream`, `cuda:1` 등 | **오류.** 조용히 넘어가지 않습니다. `E_RUN_UNSUPPORTED` | `refused` |

**오류로 끝나는 경우.** MPS에 없는 연산은 `E_RUN_MPS_OP_MISSING`이고, `--allow-cpu-fallback`을 줄 때만 CPU로 넘어갑니다. 환경에 `PYTORCH_ENABLE_MPS_FALLBACK`이 있어도 이 옵션 없이는 지웁니다. float64는 `E_RUN_MPS_DTYPE`이고, 정밀도를 대신 낮추지 않습니다. 없는 패키지는 `E_RUN_MISSING_PACKAGE`이며 `cumlx recipe`를 안내합니다. 그 밖의 스크립트 실패는 `E_RUN_SCRIPT_FAILED`입니다.

**어댑터.** `--substitute NAME=MODULE`은 스크립트가 `NAME`을 import할 때 `MODULE`을 주고, `NAME.sub`는 `MODULE.sub`로 갑니다. `NAME`은 최상위 패키지 이름이어야 하며, 점이 있는 `NAME`은 `E_USAGE`입니다. CUMLX에 있는 어댑터는 셋입니다.

| 어댑터 | 주는 것 | 거부하는 것 | 근거 |
|---|---|---|---|
| `cumlx.compat.flash_attn_sdpa` | `flash_attn_func`, `flash_attn_varlen_func`, `bert_padding`. float32 정확 어텐션, FlashAttention-2의 오른쪽 아래 정렬 causal mask, grouped-query head | dropout, sliding window, softcap, ALiBi, 확률 반환, paged cache, `flash_attn_with_kvcache`, packed 함수 4개 | float64 정확 어텐션 대비 bfloat16 반올림 수준(`validation/m6-finish-run.json`) |
| `cumlx.compat.bitsandbytes_int8` | `Int8Params`, `Linear8bitLt`, `matmul`과 int8 함수. bitsandbytes 0.50.2의 CUDA 계산을 따르며, 행 scale은 V100에서 잰 `__fdividef(127, abs(h))` 표(`fdividef_127_sm70.bin`, 측정 기록 `validation/bnb-fdividef-sm70.json`)에서 읽습니다. int8 곱은 1024씩 끊어 float32로 정확히 계산합니다 | 4비트, 학습(`has_fp16_weights=True`), 기울기, 이미 양자화된 체크포인트, `bitsandbytes.optim`의 optimizer | 고친 뒤 같은 golden에서 V100의 bitsandbytes와 int8 층 168개 최악 1.65e-7(독립 시험이 아님, `validation/m6-finish-run.json`). 새 프롬프트의 golden에서 V100과 최악 3.0e-7, F1 A(`validation/m6-f1-rerun.json`). 남은 차이: outlier 열 합의 순서(cuBLAS), dequant의 FMA |
| `cumlx.compat.gsplat_metalsplat` | `gsplat.rasterization`, 핀홀과 RGB, classic과 antialiased. metalsplat의 rasterize kernel을 gsplat 1.5.3의 alpha 상한 0.999로 다시 컴파일합니다 | SH, 다른 render mode와 카메라, covars, sparse grad 등 나머지 옵션 전부 | 실제 gsplat 1.5.3과 렌더 최악 3.6e-7, 기울기 2.6e-6 |

`flash_attn_sdpa`와 `bitsandbytes_int8`를 `--substitute`로 주면, `importlib.metadata`가 `flash-attn 2.8.3+cumlx`, `bitsandbytes 0.50.2+cumlx`를 보고합니다. 버전을 확인하는 라이브러리(Transformers 등)가 찾을 수 있게 하는 것이고, `+cumlx`는 원본이 아니라는 표시입니다. 보고할 때마다 `run-report.json`에 `distribution`으로 남습니다. 다른 모듈을 `--substitute`로 줄 때는 이렇게 하지 않습니다.

**출력.** 스크립트의 stdout은 stderr로 보내므로, stdout의 JSON 하나는 깨지지 않습니다. `run-report.json`에는 호출마다 처음 호출한 파일과 줄, 횟수가 있습니다.

**적용 범위.** DataLoader 워커와 스레드, 스크립트가 띄운 Python 하위 프로세스에도 적용됩니다. 호환 계층의 비용은 호출당 약 0.6~1.5µs입니다(M5 Pro, 호출 종류 3개, 2회 측정).

**한계.**

- 실행은 결과가 맞다는 증명이 아닙니다. 항상 `W_RUN_NOT_VERIFIED`를 표시하고, 등급은 C를 넘지 않습니다. CUDA golden과 비교해야 더 높은 등급이 됩니다.
- `torch.backends.cudnn.benchmark` 같은 설정은 그대로 두며, MPS에서는 효과가 없습니다. 보고서에도 남지 않습니다.
- `torch.version.cuda`는 None 그대로입니다. 이 값으로 분기하는 코드는 CUDA가 없다고 판단합니다.
- 컴파일된 CUDA 확장과 `python -I` 하위 프로세스에는 적용되지 않습니다.
- `torch.cuda.memory` 같은 하위 모듈을 직접 부르면 바꾸지 않습니다.
- MCP에서는 `--allow-code-execution`을 줄 때만 보입니다. 파일을 실행하는 명령이기 때문입니다.

## cumlx rundiff

`cumlx run`이 끝까지 돌아도 결과가 같다는 뜻은 아닙니다. 이 명령은 같은 스크립트를 CUDA와 대상 장치에서 돌린 손실 기록을 비교합니다.

```bash
# CUDA 장비에서 (아무것도 바꾸지 않고 기록만 합니다)
cumlx run train.py --target cuda --trace-losses --seed 0 --report golden/seed0.json
# Mac에서
cumlx run train.py --trace-losses --seed 0 --report mps/seed0.json
# 시드 1, 2도 같게 실행한 뒤
cumlx rundiff golden/ mps/
```

- `--trace-losses`는 `Tensor.backward()`에 들어온 스칼라를 모두 기록합니다. `GradScaler`를 쓰면 스케일 전 값을 기록합니다.
- `--seed N`은 시작할 때 torch, NumPy, random을 N으로 시드하고, 스크립트가 정하는 정수 시드에 N을 더합니다. 스크립트가 직접 만든 `torch.Generator`와 NumPy `RandomState`는 바꾸지 않습니다.

| 결과 | 조건 | exit |
|---|---|---|
| 등급 A | 같은 시드에서 1, 2번째 손실의 상대 차이가 모두 1e-4 이하 | 0 |
| 등급 B | 시드 3개 이상에서, 학습을 4등분한 구간의 평균 손실이 모두 허용 범위 안 | 0 |
| `diverged` | 한 구간이라도 허용 범위 밖이거나, 대상 쪽에만 NaN이나 Inf가 있음 | 1 |
| `incomplete` | 시드 부족, 시드가 스크립트에 닿지 않음, 잡음이 너무 큼, backward 횟수가 다름, 프로세스 여러 개 | 5 |
| `E_RUN_NOT_COMPARABLE` | 스크립트나 인자가 다르거나, 실패한 실행이 섞임 | 4 |

허용 범위는 차이의 표준오차 3배와 golden 평균의 1% 중 큰 값입니다. 허용 범위가 golden 평균의 10%를 넘으면 합격시키지 않습니다. 이 값은 오픈소스 스크립트를 재기 전에 고정했고, 결과를 본 뒤 바꾸지 않습니다. 근거는 [fixture 측정값](../validation/run-compare-limits.json)입니다. 여기서 CPU와 MPS의 차이는 2번째 스텝에서 최대 1.7e-7이고, 학습률이 5% 틀린 경우도 `diverged`입니다. 이 측정에 CUDA는 없었습니다.

**한계.** 등급 A는 처음 두 스텝만 보장합니다(`W_RUNDIFF_FIRST_STEPS_ONLY`). 등급 B는 잡음보다 작은 차이를 찾지 못합니다. 한 실행의 손실은 하나로 묶으므로 GAN처럼 옵티마이저가 둘이면 평균으로 비교합니다(`W_RUNDIFF_ONE_STREAM`). 손실만 비교하므로, 샘플링 결과나 저장한 모델은 비교하지 않습니다. golden이 CUDA가 아니면 `W_RUNDIFF_GOLDEN_NOT_CUDA`를 표시합니다.

## cumlx kernel

CUDA 커널을 Metal로 옮기고 실제 CUDA 결과로 증명합니다. 형식은 [`cumlx.opgolden.v1`](TRACE-FORMAT.md)입니다.

```bash
cumlx kernel capture kernels/ --cases cases.json -o kgolden/ --require-idle-gpu   # NVIDIA 장비
cumlx kernel port kernels/foo.cu -o foo.metal                                     # 번역기가 필요합니다
cumlx kernel verify foo.metal kgolden/ -o report/                                 # Mac
```

**capture.** 각 case를 nvcc로 컴파일해 실행합니다. 출력 메모리를 반복마다 다른 값으로 채우므로, 쓰지 않은 원소가 있으면 `repeat_identical`이 false가 됩니다. 컴파일이나 실행에 실패한 case는 golden에 넣지 않고 exit 5로 보고합니다. nvcc가 없으면 exit 3입니다. 공유 장비에서는 `--require-idle-gpu`를 쓰십시오. 다른 프로세스가 GPU에 있으면 시작하지 않고, 도중에 나타나면 golden을 쓰지 않고 멈춥니다. 원격 실행은 `scripts/kernel_capture_remote.sh`가 맡습니다. MCP에서는 `--allow-code-execution`을 줄 때만 보입니다.

**verify.** 입력이 함수 본문이면 `mx.fast.metal_kernel`로, `kernel void`가 있는 완전한 라이브러리면 `torch.mps.compile_shader`로 실행합니다. 파일 이름이 golden의 커널 이름이 되고, 다르면 `--kernel`로 지정합니다.

| 결과 | 조건 | exit |
|---|---|---|
| `ok`, 등급 A | 모든 case의 모든 출력이 허용치 안. 등급은 golden이 실제 CUDA일 때만 붙습니다 | 0 |
| `diverged` | 한 출력이라도 허용치 밖. `data.first_divergence`를 읽으십시오 | 1 |
| `incomplete` | 모르는 커널, golden의 커널을 덮지 못한 폴더, 컴파일 실패, 버퍼 불일치, float64 | 5 |
| 오류 | golden 손상 또는 해시 불일치 (4), Metal 또는 의존성 없음 (3) | 3, 4 |

허용치는 `rel_l2 ≤ 1e-5`, `최대 오차 / 최대 golden ≤ 1e-5`이며, 정수는 정확히 같아야 하고 NaN과 무한대는 같은 위치에 있어야 합니다. 측정 전에 커밋했습니다(`3072384`). MLX는 출력 버퍼를 비우지 않고 재사용하므로, 실행 전에 같은 크기의 버퍼를 NaN으로 채워 돌려줍니다. 커널이 쓰지 않은 원소는 `nan_mismatch`로 나타납니다. 재사용을 보장하는 API는 없어서 이 동작은 측정으로만 확인했습니다. 원소를 쓰지 않는 커널은 이 단계가 있을 때 5/5(`nan_mismatch`), 없을 때도 5/5(`rel_l2`) 실패했습니다. 즉 이 단계는 실패의 원인을 분명하게 할 뿐이며, 필요하다는 것을 보인 것은 아닙니다.

**`data.execution.differences`는 항상 네 항목을 말합니다.** 값은 같은 엔진으로 실행한 시험 커널로 그때그때 잽니다.

| 항목 | V100 (측정) | M5 Pro (측정) | 영향 |
|---|---|---|---|
| fp64 | 있음 | 없음 | 원본에 `double`이나 접미사 없는 실수 리터럴이 있으면 `W_KERNEL_FP64` |
| FTZ | 서브노멀 유지 | 복사는 유지, 산술 입력과 결과는 0으로 | `W_KERNEL_FTZ_DIFFERS`, 출력마다 `subnormal_flushed` |
| FMA 축약 | `a*b+c`를 합침 | 합침 | 다르면 `W_KERNEL_FMAD_DIFFERS` |
| 실행 순서 | warp 32 | SIMD 그룹 32 | atomic, warp 단위 코드, 반복 결과 차이가 있으면 `W_KERNEL_EXEC_ORDER` |

`--engine woof`는 woof가 번역한 라이브러리를 woof의 실행기로 돌립니다. woof의 스칼라 구조체와 threadgroup 메모리는 `torch.mps.compile_shader`가 설정하지 못합니다. 서명은 `NAME.port.json`의 `metal_signatures`에서 읽습니다. 패키지가 없으면 `E_KERNEL_TRANSLATOR_MISSING`(exit 3)입니다. 지금까지 stub으로만 시험했습니다.

**port.** 번역 전에 다음을 처리합니다. 결과는 `NAME.port.json`에 모두 기록합니다.

| 단계 | 하는 일 |
|---|---|
| 표준 헤더 | `cmath`, `cstdint`는 정규화하고 `cuda_runtime.h`, `cfloat`, `climits`는 없앱니다. 쓰는 상수만 정의합니다 |
| 호스트 코드 분리 | `main`, `ATen`, `iostream`과 호스트 함수를 없애고, 장치 코드가 쓰는 선언만 남깁니다 |
| 템플릿 | 호스트의 `foo<float><<<`나 `--instantiate`에서 구체 타입을 찾아 인스턴스를 만듭니다. 못 찾으면 거부합니다 |
| PTX 내장 함수 | `direct`(번역기가 처리), `rewrite`(`rsqrtf`, `__clz` 등을 이식 가능한 함수로 교체), `refuse` 세 가지입니다 |
| bfloat16 | `__nv_bfloat16`과 그 내장 함수를 `cumlx_bf16.h`의 이름으로 바꾸고, 그 헤더를 출력 옆에 씁니다. 아래 **bfloat16**을 보십시오 |

**거부(`E_KERNEL_UNPORTABLE`, exit 4).** 이유를 모두 `errors[].details.refused`에 적습니다. warp 단위 동작(`__match_any_sync`, `__shfl_*`, `__ballot_sync`), 반올림 모드(`__frcp_rz` 등), `double`(`--demote-fp64`로 명시 가능), 텍스처, 전역 `__device__` 변수, 재귀, `printf`, 인라인 PTX, 장치 코드의 커널 실행, 모르는 `__` 이름, 찾을 수 없는 지역 헤더, bfloat16 나눗셈(`__hdiv`)과 쌍(`__nv_bfloat162`)이 대상입니다.

**bfloat16.** Apple Silicon Mac의 GPU는 모두 MSL `bfloat`를 지원합니다(Apple6 이상, MSL 3.1, macOS 14). 그러나 CUMLX는 그것을 쓰지 않습니다. MSL 명세는 bfloat에 0 방향 반올림과 비정규수 flush를 허용하고, M5 Pro에서 `bfloat(float)`는 비정규수를 0으로 만듭니다. Metal의 float32 덧셈과 곱셈도 safe와 fast 두 수학 모드에서 비정규수 입력과 결과를 0으로 만들므로(2026-11-07 측정), float로 계산하는 에뮬레이션도 CUDA와 다릅니다. 그래서 `cumlx_bf16.h`는 값을 16비트로 저장하고 모든 연산을 32비트 정수 산술로 합니다.

| CUDA | Metal에서 계산하는 것 | 근거 |
|---|---|---|
| `__float2bfloat16`(`_rn`), `_rz`, `_rd`, `_ru` | CUDA 12.0 `cuda_bf16.hpp`의 sm70 경로. 비정규수를 지키고, NaN은 0x7FFF | float32 비트 패턴 2^32개 전부에서 모델과 비트 일치. V100(sm70, nvcc 12.0) golden 65536개와 NaN payload까지 비트 일치, grade A(`validation/kernel-golden-v100-bf16`). MSL `bfloat` 변환은 같은 golden에서 `rn`이 다름 |
| `__hadd`, `__hsub`, `__hmul`, `__hfma`, `+ - *`, `+= -= *=` | sm80 이상의 `fma.rn.bf16`: 정확한 결과를 가장 가까운 짝수로 한 번 반올림, 비정규수 유지 | 덧셈·뺄셈·곱셈은 bf16 쌍 2^32개 전부, fma는 263824개 조합에서 모델과 비트 일치. RTX A4000(sm86, nvcc 12.4) golden 262144개와 NaN payload까지 비트 일치, grade A(`validation/kernel-golden-a4000-bf16`) |
| `__hneg`, `__habs`, 비교 연산자 | 부호 비트 변경, 정수 순서 비교(NaN은 false) | sm86 golden과 NaN이 아닌 값은 비트 일치. NaN 입력에 CUDA는 0x7FFF를 내고 헤더는 payload를 지킵니다(PTX가 정하지 않음, NaN 종류로 비교) |
| `__bfloat162float`, `__bfloat16_as_ushort`, `__ushort_as_bfloat16` | 비트 이동 | |
| `/`, `__hdiv`, `__nv_bfloat162` | 거부. `operator/`는 삭제되어 컴파일되지 않습니다 | CUDA의 반올림을 확인하지 않음 |

근거는 [validation/bf16-exhaustive.json](../validation/bf16-exhaustive.json)(`scripts/bf16_exhaustive.py`, M5 Pro, 243초)입니다. 같은 검사에서 MSL `bfloat` 변환은 1671만 개, Metal float 덧셈 후 반올림은 108만 쌍이 달라, 이 검사가 차이를 볼 수 있음을 보였습니다. 이 전수 검사의 비교 대상은 모델입니다. 실제 CUDA 실행과는 두 번 비교했습니다. 변환은 V100(sm70, nvcc 12.0) golden(`validation/kernel-golden-v100-bf16`), 변환과 산술은 RTX A4000(sm86, nvcc 12.4) golden(`validation/kernel-golden-a4000-bf16`, 입력은 `validation/kernel-cases-bf16/cases-sm80.json`)이며, 둘 다 grade A입니다. CUDA 12.0은 sm70에 bf16 산술이 없어 V100으로는 산술을 확인할 수 없습니다.

`kernel verify`는 커널이 `cumlx_bf16`를 쓰면 같은 헤더를 앞에 붙이고 `W_KERNEL_BF16_EMULATED`와 `data.execution.bf16_header_sha256`를 남깁니다. bfloat16 출력은 비트로 비교합니다. golden에서 NaN이 아닌 원소는 16비트가 모두 같아야 하고, NaN은 NaN이기만 하면 됩니다(PTX가 연산의 NaN을 정하지 않으므로, `nan_payload_differs`에 셉니다). bfloat16 버퍼는 `--engine torch`로 돌립니다. `mx.fast.metal_kernel`은 그 버퍼를 `uint16_t`로 선언해 정수가 조용히 float로 바뀌므로, `--engine mlx`는 `bind` 단계의 미완료로 끝납니다. golden과 cases의 `bfloat16`은 비트 패턴(uint16)으로 저장합니다.

`--allow-warp-level`은 warp 단위 내장 함수를 거부하지 않고 번역기로 넘기며 `W_KERNEL_WARP_LEVEL_PASSED`를 표시합니다. Metal은 SIMD 그룹 32도, `threadIdx.x % 32`와 같은 lane도 보장하지 않으므로, 결과는 `cumlx kernel verify`로 그 장치에서만 판단합니다. 기본값은 거부입니다.

**번역기.** `TranslatorAdapter`로 연결합니다. 기본값은 `woof`이고, CUMLX는 설치하지 않습니다. 없으면 `E_KERNEL_TRANSLATOR_MISSING`(exit 3)이며 아무것도 쓰지 않습니다. `--preprocess-only`는 번역기 없이 전처리 결과만 씁니다. woof는 완전한 커널만 받으므로, 함수 본문은 `--cases cases.json --case ID`로 상수를 주어 감쌉니다. 번역 결과는 항상 `W_KERNEL_NOT_VERIFIED`입니다.

**검증 기록 (2026-11-03, [측정값](../validation/kernel-translation.json)).** Tesla V100-SXM2-16GB, driver 580.178.04, nvcc 12.0.140, `-arch=native`, 유휴 상태. Metal은 Apple M5 Pro, MLX 0.32.4, 안전 수학 모드입니다.

| 측정 | 결과 |
|---|---|
| 손으로 쓴 Metal 커널 8개, case 29개 | MLX 29/29, PyTorch 29/29 합격. 최악 `rel_l2` 7.5e-8 |
| V100 반복성 | case 28개가 별도 실행에서 byte까지 같음 |
| 전처리 후 감싼 lab 커널 8개를 nvcc로 다시 실행 | 원본과 byte까지 같음 |
| 바꾼 내장 함수와 원본, 같은 V100 | 정수 48개 동일. 실수 72개 중 7개가 1 ULP 차이(`rsqrtf`, `__frsqrt_rn`, `__fdividef`) |
| 틀린 번역 9건, 8종 | 9건 모두 `diverged`, 같은 커널의 맞는 번역 9건 모두 `ok` |

| 결함 | 종류 | 커널 | 실패한 case | 첫 실패 |
|---|---|---|---|---|
| K1 | 내장 함수 대응 오류 | `silu_mul` | 3/3 | NaN 위치 불일치 172개 |
| K2 | 비트 배치 | `fp4_gemv` | 4/4 | `rel_l2` 1.8 |
| K3 | float를 half로 낮춤 | `fp4_tiled` | 4/4 | `rel_l2` 2.2e-4 |
| K4 | 축소 단계 누락 | `rms_norm` | 3/3 | `rel_l2` 0.34 |
| K5 | 템플릿 상수 | `fp4_w4a4` | 1/3 | `rel_l2` 1.0 |
| K6 | 정수 인코딩 | `fp4_quantize` | 2/2 | 정수 98개 불일치 |
| K7 | 누산기 유실 | `fp4_small_m` | 1/4 | K > 256인 case만 |
| K8 | 템플릿 상수 | `fp4_decode` | 2/6 | `rel_l2` 926 |
| K9 | 항 누락 (epsilon) | `rms_norm` | 1/3 | 입력이 작은 case만, `rel_l2` 2.6 |

**한계.** 이 저장소의 커널만 실행했고, 이 커널들은 처음부터 이식을 염두에 두고 썼으며 warp 단위 동작이 없습니다. V100과 M5 Pro 한 대씩입니다. 번역기는 stub으로만 시험했으므로 woof의 실제 호출은 확인하지 않았습니다. 전처리는 토큰과 선언 단위로 동작해서 매크로가 기능을 숨길 수 있습니다. 호출에서 인자를 추론하는 템플릿 함수는 템플릿으로 남기고 `W_KERNEL_TEMPLATE_KEPT`를 표시합니다. 서브노멀만 있는 출력은 FTZ 때문에 `diverged`가 됩니다. 입력이 실행하지 않는 경로의 오류는 찾지 못합니다. K7이 그 예입니다.

## 권장 순서

1. `cumlx weights <원본> <포팅본>`: 수 초 안에 가중치 자체를 확인합니다.
2. `cumlx capture <원본> ... -o golden/` 후 `cumlx diff golden/ <포팅본>`: 포팅본의 코드를 확인합니다.
3. 양자화 포팅본이거나 `W_WEIGHT_ROUNDING_NOISE`가 나오면 `capture --weights-from-mlx <포팅본>`으로 golden을 다시 만듭니다. 포팅본의 가중치를 넣은 golden에서는 모든 모듈을 `rel_l2 ≤ 1e-4`로 판정합니다.

## diff 판정

- **isolated** (기본): 각 모듈에 golden 입력을 넣어 **float32 산술로** 비교합니다. attention·decoder layer는 hidden state와 causal mask로 재생합니다. 저정밀 포팅본은 재생하는 동안만 float32로 승격하고 끝나면 복구합니다.
- **chain**: token id로 포팅본 전체를 **네이티브 dtype으로** 실행하고 모든 모듈 출력을 비교합니다. 결함 뒤의 모든 출력이 함께 실패합니다.
- `chain_drift`: 모듈은 golden 입력에서 통과했지만 chain 출력이 임계값을 넘은 경우입니다. 실패로 세지 않습니다. 저정밀 산술의 누적 반올림이며, torch 자체 bf16도 같은 크기로 벗어납니다(아래 대조 실험). NaN과 shape 불일치는 chain에서도 실패입니다.
- `data.summary.native_output`: 네이티브 dtype 출력의 top-1이 golden과 일치한 토큰 위치 수입니다.
- `first_divergence.basis`:
  - `isolated` (high): 그 모듈 자체가 다릅니다. RoPE, mask, residual 결함도 여기에 해당합니다.
  - `chain+isolated` (medium): 재생할 수 없는 모듈의 하위 모듈은 맞지만 chain 출력이 다릅니다. 그 모듈의 자체 코드가 원인입니다.
  - `chain` (medium): chain 모드이거나, isolated 모드에서 네이티브 dtype 출력이 NaN 또는 shape 불일치입니다.
- 상태: `pass`, `fail`, `unverified`(파라미터가 있는데 비교 못 함 → exit 5), `skipped`(파라미터 없는 보조 모듈), `ignored`(사용자가 제외).
- **decode 단계:** capture는 기본으로 KV cache 위의 생성 한 단계를 함께 기록합니다(`--decode-steps`). 다음 토큰은 golden logits에서 고르므로 양쪽 입력이 같습니다. diff는 모듈마다 새 KV cache를 만들어 prefill의 golden 입력으로 채운 뒤 다음 입력을 넣습니다. 모든 입력이 golden이라 실패하면 그 모듈의 캐시 처리(위치 오프셋, mask, 캐시 갱신)가 틀린 것입니다. `first_divergence.phase`가 `decode`로 나옵니다. KV cache가 없는 포팅본은 decode 기록이 `unverified`가 되고 `W_DECODE_NOT_RUN`이 나옵니다.
- **MoE:** HF의 `mlp.experts`를 mlx-lm의 `mlp.switch_mlp`로 자동 매핑합니다. expert는 원본의 라우팅(top-k 인덱스, 가중치)으로 재생하므로 라우터 결함과 expert 결함을 구분합니다. chain 출력은 형태가 달라 비교할 수 없어서 `--mode chain`에서는 experts가 `unverified`입니다. transformers 4의 expert별 모듈(`experts.3`, `experts.3.up_proj`)은 포팅본의 `switch_mlp` 3번 expert로 매핑하고, 원본이 그 expert에 보낸 토큰만 재생합니다. 그래서 한 layer의 한 expert 결함도 그 projection을 지목합니다(`mlx_name`이 `switch_mlp[3].up_proj`).
- **모듈 경계 차이:** 출력이 golden의 상수배이고 부모 모듈이 일치하면, 그 상수를 부모에서 적용하는 구현 차이입니다. 예를 들어 transformers는 Gemma 임베딩을 임베딩 안에서 스케일하고 mlx-lm은 모델 본체에서 스케일합니다. `boundary_scale`을 기록하고 `W_MODULE_BOUNDARY_DIFFERS`를 내며, 실패로 세지 않습니다. 진짜 scale 결함은 부모 출력도 바꾸므로 실패로 남습니다.
- **mask:** 재생에는 포팅본이 자체 실행에서 그 모듈에 넘긴 mask를 그대로 씁니다. 하이브리드 모델의 합성곱 층은 causal mask를 받지 않기 때문입니다.
- **일부만 기록:** `capture --layers 0,13,27`은 그 decoder layer와 스택 밖의 모듈만 기록해 큰 모델의 trace를 줄입니다. diff는 `W_PARTIAL_TRACE`를 내고, 판정은 기록된 모듈에 대한 것입니다. 기록하지 않은 layer의 결함은 그 layer를 포함하는 모듈의 실패로 나타납니다.
- weight tying: `lm_head`가 `embed_tokens`와 가중치를 공유하면 capture가 이를 기록하고, diff는 `embed_tokens.as_linear`를 재생해 실제로 비교합니다.

### 실행 장치와 TF32

`--device`는 `auto`(MLX 기본: Apple Silicon은 GPU), `cpu`, `gpu`입니다. GPU를 요청했는데 없으면 CPU로 넘어가지 않고 `E_BACKEND_UNAVAILABLE`로 실패합니다.

**MLX는 float32 matmul·attention을 M5와 CUDA에서 기본으로 TF32(가수 10비트)로 실행합니다.** 입출력 dtype은 float32 그대로입니다. [MLX 문서](https://ml-explore.github.io/mlx/build/html/usage/precision.html)에 나와 있는 동작이며 버그가 아닙니다.

| M5 Pro GPU, MLX 0.32.4 | `MLX_ENABLE_TF32=1` (MLX 기본) | `=0` |
|---|---|---|
| float32 GEMM (M≥2) rel_l2 | 7.9e-4 | 1.4e-7 |
| `fast.scaled_dot_product_attention` (GPU vs CPU) | 1.1e-3 | 1.3e-7 |
| GEMV (M=1) rel_l2 | 6.4e-8 | 6.4e-8 |

검증기는 기본으로 TF32를 끕니다(`--tf32 off`). 그리고 실제 오차를 측정해 `data.execution`에 기록합니다. MLX 0.32는 이 설정을 프로세스당 한 번, 첫 matmul에서 읽습니다. 그래서 이미 TF32로 matmul을 실행한 프로세스에서 끄도록 요청하면 측정값으로 알아내고 `E_BACKEND_UNAVAILABLE`로 거부합니다.

`--tf32 on`은 사용자가 실제로 배포하는 MLX 기본 동작을 측정합니다. 임계값은 측정 오차의 60배 이상으로 올라가고(M5 Pro에서 4.5e-2) `W_TF32_ACTIVE` 경고가 나옵니다. 이 모드에서는 작은 결함을 놓칠 수 있습니다.

### 임계값 (`rel_l2`)

| 포팅본 profile | chain | leaf 상한 | composite 상한 |
|---|---|---|---|
| float32 | 1e-4 | 1e-4 | 1e-4 |
| float16 | 5e-3 | 2e-3 | 1e-2 |
| bfloat16 | 3e-2 | 1.5e-2 | 2e-1 |
| quantized (reference golden) | 3e-2 | 3e-2 | 3e-2 |
| 포팅본 가중치 golden (모든 dtype) | profile 값 | 1e-4 | 1e-4 |

isolated 임계값은 그 실행에서 측정한 **leaf 오차의 중앙값**으로 정합니다. leaf는 중앙값의 6배, composite(다른 파라미터 모듈을 포함한 모듈)는 100배이며, 하한 1e-4와 위 상한 사이로 제한합니다.

- 가중치 반올림은 모든 leaf에 거의 고르게 나타납니다(최댓값/중앙값 ≤ 3.0). 결함이 있는 leaf 하나는 중앙값을 움직이지 못하므로, **결함이 자기 임계값을 느슨하게 만들 수 없습니다.**
- 무손실 포팅본(예: bf16 원본 체크포인트의 bf16 포팅)은 하한 1e-4를 유지합니다.
- float32 원본을 bf16으로 내린 포팅본은 composite 임계값이 약 0.15까지 올라가고 `W_WEIGHT_ROUNDING_NOISE`가 나옵니다. 이때 그보다 작은 composite 결함은 reference golden으로 구분할 수 없으므로 `--weights-from-mlx`를 쓰십시오.

## 검증 기록 (2026-11-02)

환경: Apple M5 Pro, macOS 27.0.1, Python 3.13, torch 2.15.0, transformers 5.22.1, mlx 0.32.4, mlx-lm 0.32.2. 원본 결과는 [validation/dogfood-macos-m5-pro.json](../validation/dogfood-macos-m5-pro.json)이고 `scripts/dogfood_real_model.py`로 재현합니다.

### 깨끗한 실제 포팅본 (입력 5개, 최대 48 토큰, decode 1단계, GPU)

| 포팅본 / golden | 판정 | isolated 최악 | chain 최악 | native top-1 |
|---|---|---|---|---|
| Qwen2.5-0.5B float32 / reference | ok | 1.3e-5 | 2.8e-5 | 64/64 |
| Qwen2.5-0.5B bf16 / reference | ok (drift 487) | 1.3e-5 | 0.35 | 62/64 |
| Qwen2.5-0.5B float16 / reference | ok (drift 174) | 1.3e-5 | 5.4e-2 | 64/64 |
| Qwen2.5-0.5B 4bit / 포팅본 가중치 | ok | 1.6e-5 | 2.4e-5 | 64/64 |
| Qwen2.5-0.5B 8bit / 포팅본 가중치 | ok | 2.0e-5 | 5.3e-5 | 64/64 |
| Qwen2.5-0.5B 4bit+bf16 / 포팅본 가중치 | ok (drift 803) | 1.3e-5 | 0.17 | 63/64 |
| SmolLM2-135M float32 / reference | ok | 6.3e-6 | 1.2e-5 | 94/94 |
| SmolLM2-135M bf16 / reference | ok (drift 196) | 6.3e-6 | 0.14 | 91/94 |
| SmolLM2-135M 4bit / 포팅본 가중치 | ok | 2.4e-6 | 1.3e-5 | 94/94 |

나머지 포팅본과 CPU 실행, float32 원본 시나리오, 공개 tiny MoE를 합쳐 **깨끗한 실행 38회 전부 ok, 파라미터 커버리지 100%**입니다. 공개 tiny MoE의 CPU 실행 2회는 MLX `gather_mm`이 CPU에서 float32만 지원해 실행하지 못했고, 통과로 세지 않았습니다.

float32 원본(가중치를 1% 흔들어 bf16으로 표현할 수 없게 만든 경우)의 float16·bf16 포팅 8회도 전부 ok입니다.

| 포팅본 | 중앙값 | isolated 최악 | 적용 임계값 (leaf / composite) |
|---|---|---|---|
| Qwen2.5 float16 | 2.0e-4 | 2.9e-3 | 1.2e-3 / 1.0e-2 |
| Qwen2.5 bf16 | 1.6e-3 | 7.5e-2 | 9.4e-3 / 1.6e-1 |
| SmolLM2 float16 | 1.9e-4 | 3.5e-3 | 1.1e-3 / 1.0e-2 |
| SmolLM2 bf16 | 1.5e-3 | 2.6e-2 | 9.1e-3 / 1.5e-1 |

### 결함 주입 (실제 모델, 중간 layer)

50회 전부 정확한 모듈을 `isolated, high`로 지목했습니다. SmolLM2의 F2 3회는 attention에 bias가 없어 적용할 수 없었고, 통과로 세지 않았습니다.

| 결함 | Qwen2.5 rel_l2 | SmolLM2 rel_l2 | 비고 |
|---|---|---|---|
| F1 전치 / F2 bias 누락 | 0.50 / 0.40 | 1.78 / 적용 불가 | float32·float16·bf16 포팅 |
| F3 norm eps / F6 활성화 교체 | 0.37 / 0.29 | 6.1e-3 / 0.20 | float32·float16·bf16 포팅 |
| F7 RoPE ×10 / F8 mask 누락 | 4.2e-2 / 0.48 | 2.2e-2 / 1.05 | |
| F11 RoPE가 캐시 offset 무시 | 0.61 | 0.36 | prefill은 정상, `phase: decode`로 지목 |
| F4 MXFP4 scale / F5 NVFP4 scale | 3.9e-3 / 1.6e-3 | 8.9e-3 / 2.0e-3 | 포팅본 가중치 golden에서만 구분 가능 |
| F10 nibble 반전 | 1.5 | 2.1 | NVFP4·MXFP4 |
| F9 expert 순서 회전 | - | - | 공개 tiny MoE 3개. `mlp.experts`를 지목, `weights`가 expert 위치를 명시 |

검출 한계(Qwen2.5, 무손실 float32·bf16, 기본 임계값): RoPE base ×1.01, 가중치 ×1.01, 가중치 ×1.002를 모두 정확한 모듈로 지목했습니다.

### 공개 포팅본

[validation/public-ports.json](../validation/public-ports.json)은 `mlx-community`의 포팅본 10개 결과입니다. 9개는 `ok`이고, `TinyLlama-1.1B-Chat-v1.0-4bit`는 2024년 파일명(`weights.00.safetensors`)이라 mlx-lm이 로드하지 못해 **검증하지 못했습니다.**

Dory1.0-4B NVFP4의 MLX repack(`smdesai/...`)은 ModelOpt 원본과 비교했습니다([결과](../validation/checkpoint-dory-nvfp4.json)). 양자화 텐서 288개의 payload·scale bytes가 모두 같고, 복호값 차이는 최대 9.7e-8입니다. 다른 텐서 219개도 동일합니다.

### 실제 불일치

mlx-lm 0.32.2는 transformers 5가 쓴 config의 `rope_parameters`를 많은 모델에서 읽지 않아, 조용히 기본 RoPE base를 씁니다. 2026-11-07에는 사람들이 Hugging Face에 올린 finetune의 MLX 포팅본 3개(MiniCPM5-2B 2개, Voltaire-2.5)에서 `model.layers.0.self_attn`이 틀린 것을 확인했고, config에 최상위 `rope_theta`만 더하면 통과했습니다. 근거와 재현은 [validation/upstream](../validation/upstream/README.md)에 있습니다. mlx-lm 정책상 AI가 upstream 게시물을 작성할 수 없어 아직 보고하지 않았습니다.

### 장치 간 golden 차이

Qwen2.5-0.5B, 입력 2개, `cumlx tracediff`로 잰 값입니다. 전부 float32 임계값 1e-4 안입니다. V100 측정의 [원본 결과](../validation/cuda-golden-v100.json)가 있습니다.

| 비교 | 최악 rel_l2 |
|---|---|
| CPU 반복, MPS 반복, V100 CUDA 반복 | 0.0 |
| V100 CUDA vs V100 CPU | 1.9e-5 |
| MPS vs CPU (M5 Pro) | 4.3e-5 |
| V100 CUDA vs M5 Pro CPU | 4.5e-5 |
| V100 CPU vs M5 Pro CPU | 5.1e-5 |

M5 Pro의 MLX bf16 포팅본은 V100 CUDA golden으로도 `ok`입니다(top-1 13/13). CUDA 장비에서 받은 golden을 Mac에서 그대로 쓸 수 있습니다. 모델 하나, V100 한 대의 결과입니다.

### 최신 Hugging Face 모델

[validation/latest-models.json](../validation/latest-models.json)은 2026-11-02 기준 최근 모델 12개의 결과입니다. 같은 체크포인트를 transformers와 mlx-lm으로 각각 로드해 비교하므로, mlx-lm 구현 자체를 검증합니다. 측정한 모델은 바로 삭제했습니다.

| 모델 | 구조 | 구현 판정 | 4bit 판정 | isolated 최악 | native top-1 |
|---|---|---|---|---|---|
| Octen/Octen-WebGR-0.6B | Qwen3 | ok | ok | 7.4e-6 | 54/55 |
| LiquidAI/LFM2.5-230M | LFM2 하이브리드 | ok | ok | 6.3e-6 | 58/60 |
| udgamlabs/Laghu | Gemma4 | ok | ok | 임베딩 외 임계값 이내 (leaf 1e-04, composite 1e-04) | 73/74 |
| udgamlabs/Vemana-Base | Gemma4 | ok | ok | 임베딩 외 임계값 이내 (leaf 1e-04, composite 1e-04) | 106/106 |
| nkthebass/tinybrainbot-61m-a10m-v4.1-moe-base | Mixtral MoE | ok | ok | 2.7e-6 | 85/85 |
| nkthebass/tinybrainbot-360m-a75m-v5.1-instruct | Mixtral MoE | ok | ok | 2.4e-6 | 68/68 |
| nkthebass/tinybrainbot-100m-v5.1-base | Llama | ok | ok | 1.5e-5 | 37/38 |

Gemma4의 임베딩은 상수배 차이(transformers가 임베딩 안에서 스케일)라서 `boundary_scale`로 기록됩니다.

나머지 5개는 비교하지 못했습니다. 통과가 아닙니다.

- Ling-3.0-flash-UBTree, MiniCPM-V-4.7-1B-DSpark, Qwen3.8-Flash-Next-DFlash2, North-Mini-Code-1.0-eagle: `embed_tokens`나 `lm_head`가 없는 draft 또는 부분 체크포인트입니다. 표준 모델로 로드하면 그 가중치가 랜덤이 되므로 capture가 거부합니다.
- Climate-1000-Granite-0.35b: mlx-lm 0.32.2가 `linear_attention` 층을 구현하지 않아 로드 오류를 냅니다.

### 큰 모델의 capture

Qwen2.5-7B 구조(파라미터 76억 개, 랜덤 가중치, bf16), `--layers 0,13,27`, 입력 3개, 32 토큰, decode 1단계: trace 322MB, capture 12.1초입니다([결과](../validation/large-capture-qwen2.5-7b.json)). 실제 checkpoint로는 실행하지 않았습니다. 실제 capture는 float32로 로드하므로 메모리 약 30GB가 필요합니다.

### `drift` 판정의 근거: torch 자체 bf16과의 대조

| Qwen2.5 bf16, 입력 3개 | logit rel_l2 | top-1 | 모듈 출력 rel_l2 중앙값 |
|---|---|---|---|
| torch bf16 vs torch float32 | 0.037~0.056 | 19/20 | 1.6e-2 |
| MLX bf16 CPU vs torch float32 | 0.041~0.059 | 19/20 | 3.6e-2 |
| MLX bf16 GPU vs torch float32 | 0.040~0.045 | 18/20 | 1.7e-2 |

MLX의 drift는 torch 자체 bf16과 같은 크기입니다.

### 정적 가중치 비교

Qwen2.5-0.5B의 float32·float16·bf16·4bit·8bit 포팅 모두 290개 파라미터 통과, 프로세스 전체 약 3.3초(M5 Pro, 캐시된 체크포인트)입니다. SmolLM2와 공개 포팅본, 공개 tiny MoE도 통과했습니다. MoE는 expert별로 판정합니다.

### conform 발견 사항 (MLX 0.32.4)

- `mx.dequantize(..., mode='nvfp4', global_scale=G)`는 `code × block_scale × G/2688`을 계산합니다. NVIDIA ModelOpt의 `weight_scale_2`처럼 복원 배율을 그대로 넘기면 가중치가 1/2688이 됩니다. 변환하려면 `G = multiplier × 2688`로 넘깁니다. 2의 거듭제곱이 아닌 값에서는 float32 반올림 때문에 최대 1 ULP 차이가 납니다.
- `mx.quantize` 결과의 코드 차이는 전부 부호 있는 0의 차이(MLX는 작은 음수를 +0, CUMLX는 -0으로 기록)였고, 복원값은 같았습니다.
- **Metal GPU는 subnormal을 0으로 flush합니다.** MXFP4 decode의 64개 값(크기 ≤ 3.5e-38, scale code 0·1)이 부호를 유지한 0이 됩니다. CPU stream은 값을 유지합니다. scale 2^-127 자체가 먼저 flush되므로 결과가 정규수여도 0이 됩니다. `decode_bitwise.mxfp4.gpu`(`device_behavior`)로 보고하며 실패로 세지 않습니다.
- `mx.dequantize`는 scale의 dtype으로 산술합니다. bf16 scale이면 `dtype=float32`를 줘도 결과가 0.49% 벗어나지만, `quantized_matmul`은 float32 정밀도입니다(7.6e-8). 그래서 golden용 가중치는 scale을 float32로 올린 뒤 복원합니다.

이 내용은 MLX 버그라고 단정한 것이 아니라 **규약과 장치 동작에 대한 기록**입니다. upstream에 보고하기 전에 MLX 문서와 소스 코드로 의도를 확인해야 합니다.

### 관찰: 네이티브 저정밀 양자화의 CPU와 GPU 차이

4bit+bf16 포팅본의 네이티브 top-1은 GPU에서 63/64, CPU에서 54/64입니다(SmolLM2 92/94, 76/94). CPU 실행은 349초로 GPU(4초)보다 훨씬 느립니다. isolated float32 재생은 양쪽 모두 1.3e-5이므로 가중치와 로직 문제는 아닙니다. 원인은 조사하지 않았습니다.

### Linux

원격 CI는 Linux MLX CPU에서 tiny fixture로 e2e 테스트와 selftest를 실행합니다. 실제 모델은 Linux에서 실행하지 않았습니다.

## 미완료와 한계

- 패딩 입력은 기록하지 않으므로, 재생은 항상 causal mask를 가정합니다.
- 학습된 실제 MoE 모델은 검증하지 않았습니다. 공개 tiny random 모델과 fixture만 실행했습니다.
- transformers 4로 기록한 expert별 `ModuleList` 구조(`mlp.experts.3.up_proj`, Mixtral의 `w1`/`w3`/`w2`)는 expert 하나씩 재생합니다. 2026-11-03 Linux CPU에서 transformers 4.57.6으로 기록한 tiny Qwen3-MoE와 tiny Mixtral만 확인했습니다. 학습된 transformers 4 체크포인트로는 실행하지 않았습니다.
- bf16 MoE 포팅과 expert별 global scale을 쓰는 NVFP4 MoE 포팅은 CPU에서 실행되지 않습니다(MLX `gather_mm`과 `gather_qmm`의 CPU 구현 제한). `E_MODEL_RUN`으로 보고합니다.
- `hidden` 재생은 `(x, mask, cache)` 형식을 대상으로 합니다. 형식이 다른 모듈은 재생 오류로 기록되고 chain으로 판정합니다.
- 이름 매핑은 동일 경로와 JSON 규칙만 지원합니다. mlx-lm 외의 포팅은 `--loader python:pkg.mod:fn`이 필요합니다.
- `weights`는 safetensors 체크포인트만 읽습니다. 양자화 가중치는 방향과 scale만 판정하므로 작은 값 오류는 놓칠 수 있습니다.
- `conform --checkpoint`는 row-major low-nibble-first 레이아웃의 ModelOpt NVFP4만 읽습니다.
- CUDA golden은 Qwen2.5-0.5B와 V100 한 대로만 측정했습니다. 다른 장비는 [측정 절차](CUDA-GOLDEN.md)를 따르십시오.
- PyTorch `Conv1d`, `Conv2d` 모듈을 직접 재생하지 않습니다(입력 레이아웃이 다름). 그 모듈을 포함한 상위 모듈로 판정합니다. 비전 모델은 범위 밖입니다.
- 임계값은 두 모델과 M5 Pro 한 기기의 측정값입니다. 다른 아키텍처와 GPU 세대에서는 재확인이 필요합니다.
