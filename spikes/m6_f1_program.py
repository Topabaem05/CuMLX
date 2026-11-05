"""An ordinary LLM inference program for a CUDA machine, with FlashAttention and 8-bit bitsandbytes. It has no Mac-specific line, because F1 asks whether such a program runs as it is."""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--revision', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    revision=args.revision,
    attn_implementation='flash_attention_2',
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map={'': 0},
)
inputs = tokenizer('The capital of France is', return_tensors='pt').to('cuda')
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
generated = out[0, inputs['input_ids'].shape[1]:].tolist()
with open(args.output, 'w') as handle:
    json.dump({'generated_tokens': len(generated), 'token_ids': generated, 'text': tokenizer.decode(generated), 'device': str(out.device)}, handle)
print(tokenizer.decode(generated))
