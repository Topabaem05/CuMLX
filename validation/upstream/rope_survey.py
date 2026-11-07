# Save a tiny model with a non-default rope_theta through transformers 5, load it with mlx-lm, and run cumlx diff.

import atexit
import json
import pathlib
import shutil
import tempfile

from mlx_lm.utils import load_model
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from cumlx.verify import diff
from cumlx.verify.capture_torch import capture_model, load_hf_model
from cumlx.verify.trace import TraceWriter, prepare_output_dir

THETA=12345.0
CASES={'llama':dict(attention_bias=False),'qwen2':{},'qwen3':dict(head_dim=16),'mistral':dict(head_dim=16),'phi3':dict(pad_token_id=0),'gemma':dict(head_dim=16),'gemma2':dict(head_dim=16),'smollm3':dict(),'starcoder2':dict(),'stablelm':dict(),'granite':dict(),'cohere':dict(),'glm':dict(head_dim=16),'exaone4':dict()}
base=dict(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,tie_word_embeddings=False,max_position_embeddings=64)
tok=[1,5,9,23,7,44,3,12]
out={}
for mt,extra in CASES.items():
    d=pathlib.Path(tempfile.mkdtemp())
    atexit.register(shutil.rmtree,d,ignore_errors=True)
    try:
        cfg=AutoConfig.for_model(mt,**{**base,**extra})
        rp=getattr(cfg,'rope_parameters',None)
        if isinstance(rp,dict): rp['rope_theta']=THETA
        else: cfg.rope_theta=THETA
        m=AutoModelForCausalLM.from_config(cfg,dtype=torch.float32).eval()
        with torch.no_grad():
            for n,p in m.named_parameters():
                if 'norm' not in n and p.ndim>1: torch.nn.init.normal_(p,std=0.25)
        m.save_pretrained(d/'m',safe_serialization=True)
        j=json.load(open(d/'m/config.json')); saved={k:j.get(k) for k in('rope_theta','rope_parameters')}
    except BaseException as e:
        out[mt]=('skip-hf',type(e).__name__,str(e)[:90]); continue
    try:
        pm,_=load_model(d/'m')
    except BaseException as e:
        out[mt]=('LOAD-FAIL',type(e).__name__,str(e)[:140],saved); continue
    try:
        tm=load_hf_model(str(d/'m'))
        g=prepare_output_dir(d/'g',overwrite=True); w=TraceWriter(g,source={'framework':'torch'},golden={'kind':'reference','dtype':'float32'},max_bytes=1<<30)
        capture_model(tm,[np.asarray(tok)],w,decode_steps=1); w.finish()
        r=diff.run_diff(str(g),'x',model=pm)
        fd=r['first_divergence']; base_port=None
        for n,mod in pm.named_modules():
            if hasattr(mod,'rope') and hasattr(mod.rope,'base'): base_port=mod.rope.base; break
        out[mt]=(r['verdict'],'rope base in port',base_port,'expected',THETA,fd and (fd['module'],fd['basis'],round(fd['metrics'].get('rel_l2') or 0,4)),saved.get('rope_theta'),'unv',r['summary']['unverified'])
    except BaseException as e:
        out[mt]=('diff-error',type(e).__name__,str(e)[:160])
for k,v in out.items(): print(k,v)
