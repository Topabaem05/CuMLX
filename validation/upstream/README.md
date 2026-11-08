# mlx-lm에서 찾은 불일치: transformers 5 config의 RoPE base

CUMLX로 찾은 실제 불일치의 근거입니다. **아직 upstream에 보고하지 않았습니다.**

mlx-lm의 [기여 정책](https://github.com/ml-explore/mlx-lm/blob/main/CONTRIBUTING.md)은 버그 리포트, PR 설명, 사람에 대한 답변을 AI로 작성하는 것을 금지합니다. 그래서 보고는 사람이 직접 작성해야 합니다. 이 문서는 그 판단에 필요한 사실만 정리합니다.

## 현상

transformers 5는 `rope_theta`를 config의 `rope_parameters` 안에 저장합니다. mlx-lm의 많은 모델은 최상위 `rope_theta`만 읽습니다. 그래서 값이 없으면 **오류나 경고 없이** 모델의 기본값을 씁니다.

재현은 `rope_parameters_repro.py`입니다. mlx-lm, transformers, torch만 필요합니다. 2026-11-02, mlx-lm 0.32.2, transformers 5.22.1, mlx 0.32.4에서 실행한 결과는 다음과 같습니다.

```text
saved config: {'rope_parameters': {'rope_theta': 500000.0, 'rope_type': 'default'}}
RoPE base in the MLX model: 10000
relative error of the logits: 0.17350445687770844
same top-1 token: 1.0
after adding a top-level rope_theta, base: 500000.0
relative error of the logits: 4.7141412551354733e-07
```

## 모델별 결과

`rope_survey.py`는 `rope_theta=12345`인 tiny 모델을 transformers 5로 저장하고, mlx-lm으로 로드한 뒤 `cumlx diff`를 실행합니다.

| 모델 | 포팅본의 RoPE base | 결과 |
|---|---|---|
| llama | 10000 | 조용히 틀림. `self_attn`을 `isolated`로 지목 |
| qwen2 | 1000000 | 조용히 틀림. `self_attn`을 `isolated`로 지목 |
| mistral | 10000 | 조용히 틀림. `self_attn`을 `isolated`로 지목 |
| phi3 | 10000 | 조용히 틀림. `self_attn`을 `isolated`로 지목 |
| starcoder2 | 100000 | 조용히 틀림. `self_attn`을 `isolated`로 지목 |
| qwen3 | 12345 | 정상 (수정됨, upstream PR #1898) |
| stablelm, granite, exaone4 | - | 로드 오류: `rope_theta` 인자 누락 |
| gemma, gemma2, cohere | - | 로드 오류: `lm_head.weight`. 이 조사의 tied 설정 때문일 수 있으며 RoPE 문제인지 확인하지 않았습니다 |

## 공개된 실제 포팅본에서 확인 (2026-11-07)

위의 조사는 tiny 랜덤 모델이었습니다. 이번에는 Hugging Face에 사람들이 올린 MLX 포팅본을 직접 검사했습니다.

**1단계: config만 읽어 후보 찾기.** `config_scan.py`는 `mlx` 태그가 붙은 최근 저장소의 `config.json`만 내려받습니다. 그리고 mlx-lm의 `ModelArgs.from_dict`로 실제로 쓸 RoPE 설정을 만들어 config와 비교합니다. 가중치는 받지 않고 모델도 실행하지 않습니다. 결과는 `config-scan.json`에 있습니다.

2026-11-07(UTC), mlx-lm 0.32.2(최신 릴리스)로 최근 저장소 6000개를 나열해 config 4864개를 읽었습니다. 그중 587개에 `rope_parameters`가 있었고, 8개(올린 사람 7명)에서 mlx-lm이 transformers와 다른 RoPE base를 씁니다. 기준은 transformers 5.22.1이 같은 config에서 실제로 쓰는 값입니다.

| 포팅본 | 모델 | config의 transformers_version | transformers의 base | mlx-lm의 base | #1964의 변경으로 |
|---|---|---|---|---|---|
| rahulol/Hermit-MiniCPM5-2B-MLX-4bit | llama | 5.21.0 | 5000000 | 10000 | 고쳐짐 |
| dalatexcoder/MiniCPM5-2B-ARA-MLX-4Bit-optiq-mixed | llama | 5.18.0 | 5000000 | 10000 | 고쳐짐 |
| dalatexcoder/MiniCPM5-2B-Heretic-ARA-MLX-4Bit | llama | 5.18.0 | 5000000 | 10000 | 고쳐짐 |
| junafinity/Nanbeige-4.3-33B-Heretic-Uncensored-MLX-4bit | llama | 5.20.2 | 120000000 | 10000 | 고쳐짐 |
| kilianbalaguer9/Voltaire-2.5-MLX-4bit | llama | 5.19.0 | 130000 | 10000 | 고쳐짐 |
| melmass/Llama-3-8B-LLM2Vec-Kimodo-MLX-4bit | llama | 5.1.0 | 500000 | 10000 | 고쳐짐 |
| masahiroid/Sentiment-Analysis-7B-4bit-mlx | mistral | 5.2.0 | 1000000 | 10000 | 고쳐짐 |
| imgovind/wrightproduct-v1 | qwen3_moe | 5.20.2 | 10000000 | 1000000 | **고쳐지지 않음** |

마지막 config는 최상위 `rope_theta`(1000000)와 `rope_parameters.rope_theta`(10000000)를 둘 다 가지며, 두 값이 다릅니다. transformers 5는 `rope_parameters`를 쓰고(원본 Qwen3-30B-A3B-Instruct-2507도 10000000), mlx-lm은 최상위 값을 씁니다. #1964는 `{**flat, **params}` 순서로 합치므로 최상위 값이 이깁니다. 이 포팅본은 78 GB라서 모델을 돌려 확인하지 않았고, config 수준의 결과입니다.

config에 `rope_parameters` 필드를 직접 읽는 모델(gemma4_text, laguna, mellum, ministral3 등)은 판정에서 뺐습니다. `rope_scaling`만 빠진 실제 포팅본은 이번 범위에 없었습니다. 195개는 mlx-lm 0.32.2가 읽지 못했습니다. 대부분 mlx-lm에 없는 모델 종류(k2_horizon 48, lfm3 40, qwen3_5_text 31 등)이고, flint 외에는 2026-11-07의 main에도 없습니다. 다른 도구로 만든 포팅본으로 보이며 이 조사의 대상이 아닙니다.

**2단계: 포팅본 자체로 확인.** `port_rope_check.py`는 원본의 config로 transformers 모델을 만들고, 공개된 포팅본의 가중치를 모두 넣어 golden을 기록합니다(`--weights-from-mlx`와 같은 golden이고, 원본 가중치를 받지 않습니다). 그리고 공개된 포팅본을 `cumlx diff`로 비교합니다. 대조군은 같은 포팅본에서 config에 최상위 `rope_theta` 한 줄만 더한 것입니다.

| 포팅본 (원본) | 원본을 저장한 transformers | MLX의 RoPE base | 공개된 그대로 | 대조군 (최상위 `rope_theta`만 추가) |
|---|---|---|---|---|
| [rahulol/Hermit-MiniCPM5-2B-MLX-4bit](https://huggingface.co/rahulol/Hermit-MiniCPM5-2B-MLX-4bit) (SonGT1/MiniCPM5-abliterated-2B) | 5.21.0 | 10000 (원래 5000000) | **diverged**: `model.layers.0.self_attn` (isolated, high), rel_l2 0.066, 336개 실패 | **ok**: 2036 pass, 0 fail, 최악 8.0e-6 |
| [dalatexcoder/MiniCPM5-2B-ARA-MLX-4Bit-optiq-mixed](https://huggingface.co/dalatexcoder/MiniCPM5-2B-ARA-MLX-4Bit-optiq-mixed) (dalatexcoder/MiniCPM5-2B-heretic-ara) | 5.18.0 | 10000 (원래 5000000) | **diverged**: `model.layers.0.self_attn` (isolated, high), 334개 실패 | **ok**: 2036 pass, 0 fail, 최악 5.1e-6 |
| [kilianbalaguer9/Voltaire-2.5-MLX-4bit](https://huggingface.co/kilianbalaguer9/Voltaire-2.5-MLX-4bit) (원본 이름 없음) | 5.19.0 | 10000 (원래 130000) | **diverged**: `model.layers.0.self_attn` (isolated, high), 200개 실패 | **ok**: 1172 pass, 0 fail, 최악 4.0e-6 |

앞의 두 포팅본은 원본과 config가 같습니다(양자화 항목 제외). Voltaire는 카드에 원본이 없어서, 포팅본 자신의 config에서 양자화 항목을 뺀 것을 원본 config로 썼습니다. 세 포팅본 모두 대조군이 통과하므로 원인은 RoPE base 하나입니다. 기록은 `ports/`에 있습니다.

**짧은 입력에서는 잘 드러나지 않습니다.** 위 diff의 프롬프트 2개에서 top-1은 첫째 포팅본이 12개 위치 중 9개, 둘째가 12개 중 12개, 셋째가 11개 중 11개 golden과 같았습니다. `long_context_effect.py`로 첫째 포팅본을 이 저장소의 영어 문서(겹치는 문장 없음)에 돌리고, 대조군과 비교했습니다.

| 입력 길이 (token) | 16 | 64 | 256 | 1024 | 2048 |
|---|---|---|---|---|---|
| top-1이 다른 위치의 비율 | 31% | 36% | 36% | 39% | 46% |
| logits 상대 오차 (뒤쪽 절반) | 0.19 | 0.45 | 0.35 | 0.46 | 0.51 |

기준은 golden과 같다고 확인한 대조군입니다. 이 측정은 작업 품질을 잰 것이 아니고, 한 포팅본과 한 텍스트의 결과입니다.

### 큰 포팅본: 앞쪽 layer만 잘라서 확인 (2026-11-08)

7B 이상은 float32 golden이 메모리에 들어가지 않습니다(7B는 약 28 GB, 이 Mac의 다른 프로그램이 이미 20~30 GB를 씁니다). 2B도 전체 golden은 이번에 41 GB까지 올라가 중단했습니다. 그래서 `port_rope_check.py`에 두 옵션을 더했습니다.

- `--layers K`: 앞쪽 K개 decoder layer, embedding, 마지막 norm, head만 담은 포팅본을 만듭니다. 그 텐서가 든 shard만 내려받습니다. 원본 config에도 같은 K를 적용하고, 대조군도 같은 잘린 포팅본입니다. RoPE base 결함은 layer 0에서 드러나므로 이것으로 위치를 찾을 수 있습니다. 판정은 그 K개 layer에만 해당합니다.
- `--vocab-from-prompts`: embedding과 head에서 프롬프트에 나온 token id의 행만 남기고 0부터 다시 번호를 붙입니다. 4-bit 행은 행마다 따로 양자화되어 있어서 남긴 행의 값은 바뀌지 않습니다. 그러므로 모든 activation은 자르지 않은 모델과 같고, logits는 남긴 token의 것입니다. 33B는 vocabulary가 166,912개라서 float32 embedding과 head만 9.6 GB였습니다.

**방법 검증.** 이미 전체로 확인한 Voltaire에 `--layers 2`를 쓰자, 전체 검사처럼 `model.layers.0.self_attn`에서 diverged, 대조군 ok였습니다. rel_l2는 전체 검사와 같은 0.039입니다. `--vocab-from-prompts`를 더해도 rel_l2(0.038986802532664726), 통과/실패/chain drift 수(92/24/63), 대조군의 최악 오차(2.94e-6)가 출력된 모든 자릿수까지 같았습니다. 결함이 없는 포팅본(mlx-community/Qwen2.5-0.5B-Instruct-4bit, 최상위 `rope_theta` 있음)은 두 옵션으로 ok였습니다. top-1 일치 수는 vocabulary를 자르면 남긴 token 안에서만 비교하므로 달라질 수 있습니다(Voltaire 10/11 → 11/11).

| 포팅본 (원본) | 잘라서 검사한 범위 | MLX의 RoPE base | 공개된 그대로 | 대조군 |
|---|---|---|---|---|
| [dalatexcoder/MiniCPM5-2B-Heretic-ARA-MLX-4Bit](https://huggingface.co/dalatexcoder/MiniCPM5-2B-Heretic-ARA-MLX-4Bit) (dalatexcoder/MiniCPM5-2B-Heretic-ARA) | 42개 중 layer 2개 | 10000 (원래 5000000) | **diverged**: `model.layers.0.self_attn` (isolated, high), rel_l2 0.068, 24개 실패 | **ok**: 116 pass, 최악 4.6e-6 |
| [masahiroid/Sentiment-Analysis-7B-4bit-mlx](https://huggingface.co/masahiroid/Sentiment-Analysis-7B-4bit-mlx) (microsoft/Sentiment-Analysis-7B) | 32개 중 layer 2개 | 10000 (원래 1000000) | **diverged**: `model.layers.0.self_attn` (isolated, high), rel_l2 0.136, 24개 실패 | **ok**: 116 pass, 최악 2.6e-6 |
| [junafinity/Nanbeige-4.3-33B-Heretic-Uncensored-MLX-4bit](https://huggingface.co/junafinity/Nanbeige-4.3-33B-Heretic-Uncensored-MLX-4bit) (Luni/Nanbeige-4.3-33B-Heretic) | 52개 중 layer 2개, 166,912개 중 token 11개 | 10000 (원래 120000000) | **diverged**: `model.layers.0.self_attn` (isolated, high), rel_l2 0.109, 24개 실패 | **ok**: 116 pass, 최악 2.9e-6 |

최대 메모리 사용량은 24 GB, 30 GB, 28 GB였습니다(다른 프로그램 포함). Nanbeige는 원본과 config가 `eos_token_id`만 다릅니다. 이 값은 생성을 멈추는 token이고 diff의 계산에 쓰이지 않습니다.

7B는 MLX에서는 전체로 돌릴 수 있어서 `long_context_effect.py`로 출력 영향을 쟀습니다. 기준인 대조군은 앞쪽 2개 layer에서만 golden과 같다고 확인했습니다.

| 입력 길이 (token) | 16 | 64 | 256 | 1024 | 2048 |
|---|---|---|---|---|---|
| top-1이 다른 위치의 비율 (Sentiment-Analysis-7B) | 19% | 22% | 18% | 22% | 24% |
| logits 상대 오차 (뒤쪽 절반) | 0.13 | 0.16 | 0.18 | 0.18 | 0.18 |

**확인에서 뺀 것.** melmass/Llama-3-8B-LLM2Vec-Kimodo-MLX-4bit는 mlx-lm 포팅본이 아닙니다. 구조가 양방향 encoder(`LlamaBiModel`)이고, `lm_head`가 없고, 텐서 이름에 `model.`이 없고, 양자화 설정이 `config.json`이 아니라 `kimodo_mlx.json`에 있습니다(가중치를 받지 않고 원격 header만 읽음). 전용 loader가 읽으므로, scan의 표시는 mlx-lm 결함을 보이지 않습니다. 그 loader가 `rope_parameters`를 읽는지는 확인하지 않았습니다. imgovind/wrightproduct-v1(78 GB)은 여전히 config 수준의 결과입니다.

정리하면 scan이 표시한 8개 중 6개를 포팅본 자체로 확인했고, 1개는 mlx-lm 포팅본이 아니며, 1개는 config 수준입니다.

## 실제 영향과 한계

- 영향을 받는 조건은 **그 모델 구현의 기본값과 다른 `rope_theta`를 쓰는 checkpoint를 transformers 5로 다시 저장한 경우**입니다. Llama 3(500000)의 finetune이 해당할 수 있습니다.
- transformers 4로 저장한 공개 원본은 최상위 `rope_theta`를 가지므로 영향을 받지 않습니다. 예를 들어 openbmb/MiniCPM5-2B의 공식 config(transformers 5.6.2)와 공식 MLX 포팅본은 최상위 `rope_theta`가 있어 정상입니다. 영향을 받는 것은 그 원본을 transformers 5의 새 버전으로 다시 저장한 finetune과, 그것을 변환한 포팅본입니다.
- 2026-11-02의 공개 포팅본 캠페인(`mlx-community` 10개)에서는 이 문제가 나타나지 않았습니다. 2026-11-07에 실제 finetune의 포팅본 3개, 2026-11-08에 3개(7B와 33B 포함, 앞쪽 layer만)를 더 확인했습니다(위).

## upstream 상태 (2026-11-02 확인)

- 관련 PR이 이미 열려 있습니다: [ml-explore/mlx-lm#1964](https://github.com/ml-explore/mlx-lm/pull/1964) "Read RoPE settings from transformers 5 rope_parameters". 2026-10-11에 열렸고, 리뷰와 댓글이 없습니다. 새 이슈는 중복이 될 수 있습니다.
- 이 PR 브랜치는 실행하지 않았습니다. 코드는 읽었고, `BaseModelArgs.from_dict`에서 값을 복원합니다.
- main의 `llama.py`와 `qwen2.py`는 2026-11-02 기준으로 기본값을 그대로 쓰며, `rope_parameters`를 읽지 않습니다.

제안: 이슈를 새로 열기보다 #1964에 직접 쓴 짧은 확인 댓글을 다는 편이 낫습니다. 올리기 전에 PR 브랜치에서 이 재현을 실행하십시오.

### #1964의 변경으로 재현이 고쳐지는지 (2026-11-07 확인)

`pr1964_check.py`는 #1964(head `a3117179`)가 `mlx_lm/models/base.py`에 더한 두 함수를 diff에서 읽어 그대로 옮기고, 설치된 mlx-lm 0.32.2를 이 프로세스 안에서만 바꿔 `rope_parameters_repro.py`와 모델 조사를 다시 돌립니다. PR 브랜치를 설치해 돌린 것은 아닙니다. 2026-11-07, 이 Mac(mlx-lm 0.32.2, transformers 5.22.1, mlx 0.32.4)의 결과는 다음과 같습니다.

| | mlx-lm 0.32.2 그대로 | #1964의 변경 |
|---|---|---|
| 재현: MLX 모델의 RoPE base | 10000 | 500000.0 |
| 재현: logits 상대 오차 | 0.1735 | 4.7e-7 |
| `rope_theta=12345`인 config의 `ModelArgs.rope_theta`: llama, qwen2, phi3, starcoder2 | 10000, 1000000, 10000, 100000 | 모두 12345.0 |

mistral은 mlx-lm 0.32.2에 모듈이 따로 없어 조사하지 않았습니다. 이 표는 댓글을 쓰는 사람이 확인할 근거이고, 댓글 자체는 mlx-lm 정책에 따라 사람이 직접 써야 합니다. 2026-11-07 기준으로 #1964는 열려 있고 댓글이 없습니다.

## 별개의 관찰

- `hf-internal-testing/tiny-random-MixtralForCausalLM`과 `tiny-random-Qwen2MoeForCausalLM`은 mlx-lm 0.32.2에서 `Received N parameters not in model`로 로드되지 않습니다(`experts.gate_up_proj` 형식). transformers 5.22.1의 `save_pretrained`는 expert별 이름으로 저장하므로 이 형식은 일부 저장소에만 있습니다. `gpt_oss`에 대한 비슷한 이슈는 #1994입니다.
- transformers 5가 저장한 `qwen3_moe` config는 `num_local_experts`와 `rope_parameters`를 쓰기 때문에 mlx-lm 0.32.2에서 `ModelArgs` 인자 누락으로 로드되지 않습니다. 관련 PR은 #1968입니다.
