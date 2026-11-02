# `cumlx` CLI — 포팅 동등성 검증기

상태: **베타 준비 (v0.4.0.dev0)**. 2026-11-02, Linux x86_64 CPU 환경에서 검증한 범위만 적습니다.
Apple Silicon(Metal)에서 실행한 기록과 실제 공개 모델로 dogfooding한 기록은 아직 없습니다.

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
| `doctor` | 의존성과 준비된 명령 보고. `--require`로 미준비 시 exit 3 | - |
| `fixture tiny-llama` | 오프라인 테스트용 2-layer 랜덤 Llama 생성 | capture |
| `capture` | 원본 PyTorch 모델의 모듈별 입출력을 golden trace로 기록 | capture |
| `diff` | MLX 포팅본을 같은 입력으로 실행해 비교하고 처음 어긋난 모듈을 찾음 | diff |
| `conform` | MLX의 NVFP4/MXFP4 decode를 CUMLX bit-정확 코덱과 전수 비교 | diff |
| `report` | 기존 `report.json` 요약 (`--max-items`로 다시 읽기) | - |
| `selftest` | 검증기 자체 검증: 깨끗한 포팅 통과 + 오류 주입 8종 위치 검출 | 전체 기능은 capture+diff |

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
| 오류마다 `code` + `hint`(정확한 해결 명령) | 에이전트가 스스로 복구 |
| `next_steps`에 다음 명령 제안 | 워크플로를 문서 없이 이어감 |
| `--max-items`로 목록 길이 제한, `truncated` 표시, 전체는 파일 | 컨텍스트 창 절약 |
| `--dry-run`, `--overwrite` 없이는 덮어쓰지 않음, CUMLX가 만든 디렉터리만 삭제 | 되돌리기 어려운 실수 방지 |
| `describe`가 argparse에서 직접 명세를 생성 | 문서와 실제 인자가 어긋나지 않음 |
| `selftest`로 환경에서 검증기 동작을 증명 | 결과를 믿어도 되는지 먼저 확인 |

## diff 판정

- **chain**: token id로 포팅본 전체를 실행하고 모든 모듈 출력을 비교합니다. 결함 뒤의 모든 출력이 함께 실패합니다.
- **isolated** (기본): 단일 입력 모듈(Linear, Embedding, Norm, MLP 등)에 golden 입력을 직접 넣어 비교합니다. 결함 모듈 하나만 실패합니다.
- `first_divergence.basis`:
  - `isolated` (high): 그 모듈 자체가 다릅니다.
  - `chain+isolated` (medium): 하위 모듈은 맞지만 입력이 이미 틀렸습니다. 부모 모듈 안의 기록되지 않은 코드(RoPE, attention, mask, residual)가 원인입니다.
  - `chain` (medium): isolated 재생이 불가하거나 chain 모드입니다.
- 임계값(`rel_l2`): float32 1e-4, float16 5e-3, bfloat16 3e-2, quantized 3e-2. **베타 초기값이며 W2에 실측으로 보정합니다.**
- 상태: `pass`, `fail`, `unverified`(파라미터가 있는데 비교 못 함 → exit 5), `skipped`(파라미터 없는 보조 모듈), `ignored`(사용자가 제외).

## 검증 기록 (2026-11-02, Linux x86_64, Python 3.13, torch 2.15.0, transformers 5.22.1, mlx 0.32.4 CPU, mlx-lm 0.32.2)

| 항목 | 결과 |
|---|---|
| 깨끗한 tiny Llama 포팅 (float32) | isolated·chain 모두 `ok`, 최대 chain rel_l2 9.8e-7 |
| F1 Linear 전치, F2 bias 누락, F3 norm eps, F6 활성화 교체 | isolated, high로 정확한 모듈 지목 |
| F7 RoPE base, F8 causal mask 누락 | chain+isolated, medium으로 정확한 attention 모듈 지목 |
| F4 MXFP4 scale 지수 +1, F5 NVFP4 scale code +8 | 양자화 반영 golden에서 isolated, high로 지목 |
| 양자화 포팅 vs 일반 golden | `diverged` + `W_QUANTIZED_PORT_REFERENCE_GOLDEN` 경고 |
| `conform --target mlx` | MXFP4·NVFP4 decode 전수 bit 일치. **NVFP4 `global_scale`은 복원 배율이 아니라 amax 규약**(내부에서 /2688)으로 확인 |
| pytest | 전체 의존성 153 passed / numpy만 135 passed, 18 skipped (e2e) |

### conform 발견 사항 (MLX 0.32.4)

- `mx.dequantize(..., mode='nvfp4', global_scale=G)`는 `code × block_scale × G/2688`을 계산합니다. NVIDIA ModelOpt의 `weight_scale_2`처럼 복원 배율을 그대로 넘기면 가중치가 1/2688이 됩니다. 변환하려면 `G = multiplier × 2688`로 넘깁니다. 2의 거듭제곱이 아닌 값에서는 float32 반올림 때문에 최대 1 ULP 차이가 납니다.
- `mx.quantize` 결과의 코드 차이는 전부 부호 있는 0의 차이(MLX는 작은 음수를 +0, CUMLX는 -0으로 기록)였고, 복원값은 같았습니다.

이 내용은 MLX 버그라고 단정한 것이 아니라 **규약 차이**에 대한 기록입니다. upstream에 보고하기 전에 MLX 문서와 소스 코드로 의도를 확인해야 합니다.

## 미완료와 한계

- 실제 공개 모델(예: Qwen2.5-0.5B)로 dogfooding(D6)은 하지 못했습니다. 이 개발 환경의 네트워크 정책이 `huggingface.co`를 차단하기 때문입니다.
- Apple Silicon Metal에서는 실행하지 않았습니다. Linux MLX CPU 백엔드에서만 확인했습니다.
- isolated 재생은 단일 텐서 입력 모듈만 지원합니다(attention·decoder layer는 chain 기준). W2 범위입니다.
- 이름 매핑은 동일 경로와 JSON 규칙만 지원합니다. mlx-lm 외의 포팅은 `--loader python:pkg.mod:fn`이 필요합니다.
- KV cache를 쓰는 decode 단계는 기록하지 않습니다(prefill만).
