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

## 실제 영향과 한계

- 영향을 받는 조건은 **그 모델 구현의 기본값과 다른 `rope_theta`를 쓰는 checkpoint를 transformers 5로 다시 저장한 경우**입니다. Llama 3(500000)의 finetune이 해당할 수 있습니다.
- transformers 4로 저장한 공개 원본은 최상위 `rope_theta`를 가지므로 영향을 받지 않습니다. 이번 공개 포팅본 캠페인에서는 이 문제가 나타나지 않았습니다.
- 위 실행은 tiny 랜덤 모델입니다. 실제 finetune checkpoint로는 재현하지 않았습니다.

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
