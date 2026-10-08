"""Task 3 GPU helpers; leaves Task 1/2 code unchanged."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from common.data import repo_path
from common.generation import batch_generate
from common.logging_utils import set_seed


def configure(seed):
    if not torch.cuda.is_available():raise RuntimeError('GPU required.')
    set_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha(path):return hashlib.sha256(repo_path(path).read_bytes()).hexdigest()

def dump(path, value):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(path)

def append(path, value):
    with Path(path).open('a') as f:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')

def records(path):
    path=Path(path)
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []

def generate(model,tok,prompts,cfg,cap):
    previous=model.config.use_cache
    model.config.use_cache=True
    try:
        with torch.autocast('cuda',dtype=torch.float16):
            result=batch_generate(model,tok,prompts,max_prompt_length=int(cfg['max_prompt_length']),
                                  max_new_tokens=int(cap),**cfg['generation'])
    finally:model.config.use_cache=previous
    result={k:v.clone() if torch.is_tensor(v) else v for k,v in result.items()}
    # Left-padded prompts plus response tokens through the first EOS only.
    result['attention_mask'][:,result['prompt_width']:] = result['response_mask'].long()
    return result

def slice_batch(batch,start,end):
    n=batch['sequences'].shape[0]
    return {k:(v[start:end] if torch.is_tensor(v) and v.ndim>0 and v.shape[0]==n else v)
            for k,v in batch.items()}

def scores(model,batch,entropy=False):
    attn=batch['attention_mask'];positions=(attn.cumsum(-1)-1).clamp_min(0)
    outputs=model(input_ids=batch['sequences'],attention_mask=attn,position_ids=positions,use_cache=False)
    logits=outputs.logits[:,batch['prompt_width']-1:-1,:]
    ids=batch['response_ids'];logs=[];ent=[]
    for start in range(0,ids.shape[1],32):
        part=logits[:,start:start+32].float();labels=ids[:,start:start+32]
        lp=-F.cross_entropy(part.reshape(-1,part.shape[-1]),labels.reshape(-1),reduction='none')
        logs.append(lp.reshape(labels.shape))
        if entropy:
            with torch.no_grad():
                all_lp=F.log_softmax(part,-1);ent.append(-(all_lp.exp()*all_lp).sum(-1))
    return torch.cat(logs,-1),torch.cat(ent,-1) if entropy else None

def scored_microbatches(model,batch,micro=1,entropy=False):
    logs=[];ents=[]
    for start in range(0,batch['sequences'].shape[0],micro):
        lp,ent=scores(model,slice_batch(batch,start,start+micro),entropy)
        logs.append(lp)
        if entropy:ents.append(ent)
    return torch.cat(logs),torch.cat(ents) if entropy else None

def trainable_state(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}

def restore(model,state):
    parameters=dict(model.named_parameters())
    with torch.no_grad():
        for n,v in state.items():parameters[n].copy_(v.to(parameters[n].device))
