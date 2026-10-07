from __future__ import annotations
import argparse
import json
import subprocess
import sys
import numpy as np
import torch
from common.data import load_yaml,read_jsonl,prompt_messages,repo_path
from common.metrics import masked_mean
from common.models import load_policy,load_tokenizer
from task2_ppo.ppo import compute_gae,shaped_rewards,normalize_advantages,ppo_policy_loss
from task2_ppo.runtime import cuda_required,policy_scores,dump,sha


def load_cached_rollouts(path):
    rows=torch.load(repo_path(path),map_location='cpu',weights_only=False)
    if not isinstance(rows,list) or not rows:raise ValueError('Empty/invalid cache.')
    for r in rows:
        if 'old_logprobs' not in r:r['old_logprobs']=r['old_policy_logprobs']
        if 'ref_logprobs' not in r:r['ref_logprobs']=r['reference_logprobs']
    return rows


def cached_study(config_path):
    cuda_required();cfg=load_yaml(config_path)
    cache=load_cached_rollouts(cfg['cached_rollouts'])
    tok=load_tokenizer(cfg['base_model'])
    model=load_policy(cfg,adapter_path=cfg['paths']['ppo_midpoint_policy'],trainable=False)
    prompt_rows=read_jsonl(cfg['paths']['rl_prompt_train'])+read_jsonl(cfg['paths']['rl_prompt_eval'])
    prompts={r['prompt_id']:r for r in prompt_rows if r.get('prompt_id') is not None}
    # Reconstruct one fixed, padded cached batch. Never generate new responses here.
    encodings=[]
    for row in cache:
        if row['prompt_id'] not in prompts:raise RuntimeError('Cached prompt ID absent from released pools.')
        messages=prompt_messages(prompts[row['prompt_id']])
        prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        ids=tok(prompt,add_special_tokens=False,truncation=True,
                max_length=int(cfg['max_prompt_length']))['input_ids']
        response=tok(row['response'],add_special_tokens=False)['input_ids']
        if row['terminated_with_eos']:response.append(tok.eos_token_id)
        if len(response)!=len(row['old_logprobs']):
            raise RuntimeError(f"Cached response {row['source_index']} cannot be reconstructed losslessly: "
                               f"{len(response)} vs {len(row['old_logprobs'])}. Do not silently truncate.")
        if isinstance(row['response_tokens'],int) and len(response)!=row['response_tokens']:
            raise RuntimeError('Cache length mismatch.')
        encodings.append((ids,response))
    width=max(len(p) for p,r in encodings);steps=max(len(r) for p,r in encodings)
    seq=[];attn=[];responses=[];masks=[];old=[];ref=[];values=[];terminal=[]
    for row,(p,r) in zip(cache,encodings):
        n=len(r);left=width-len(p);right=steps-n
        seq.append([tok.pad_token_id]*left+p+r+[tok.pad_token_id]*right)
        attn.append([0]*left+[1]*(len(p)+n)+[0]*right)
        responses.append(r+[tok.pad_token_id]*right);masks.append([1]*n+[0]*right)
        for destination,key in [(old,'old_logprobs'),(ref,'ref_logprobs'),(values,'values')]:
            destination.append(torch.cat([row[key].float(),torch.zeros(right)]))
        terminal.append(float(row['effective_terminal_reward']))
    mask=torch.tensor(masks,dtype=torch.float32,device='cuda')
    old=torch.stack(old).cuda();ref=torch.stack(ref).cuda();values=torch.stack(values).cuda()
    shaped=shaped_rewards(torch.tensor(terminal,device='cuda'),old,ref,mask,cfg['kl_beta'])
    advantages,returns=compute_gae(shaped,values,mask,cfg['gamma'],cfg['gae_lambda'])
    advantages=normalize_advantages(advantages,mask)
    # Score individually to avoid materializing all 32 vocabulary tensors at once.
    new=torch.zeros_like(old)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        for i,(p,r) in enumerate(encodings):
            n=len(r)
            batch={'sequences':torch.tensor([p+r],device='cuda'),
                'attention_mask':torch.ones(1,len(p)+n,dtype=torch.long,device='cuda'),
                'prompt_width':len(p),'response_ids':torch.tensor([r],device='cuda')}
            scores,_=policy_scores(model,batch);new[i,:n]=scores[0]
    results=[]
    for epsilon in cfg['clip_values']:
        loss,ratio,fraction=ppo_policy_loss(new,old,advantages,mask,float(epsilon))
        raw=ratio*advantages;clipped=ratio.clamp(1-float(epsilon),1+float(epsilon))*advantages
        results.append({'epsilon':float(epsilon),'clipped_surrogate':-loss.item(),
            'clip_fraction':fraction.item(),'affected_token_fraction':fraction.item(),
            'surrogate_changed_token_fraction':masked_mean((raw>clipped).float(),mask).item()})
    result={'num_cached_rollouts':len(cache),'valid_tokens':int(mask.sum()),
        'cache_sha256':sha(cfg['cached_rollouts']),
        'source_indices':[r['source_index'] for r in cache],
        'prompt_ids':[r['prompt_id'] for r in cache],
        'candidate_policy':'supplied PPO midpoint',
        'advantages':'Cached effective reward + reference KL shaping; GAE; valid-token normalization',
        'conditions':results}
    dump(repo_path(cfg['results_dir'])/'cached_clipping_summary.json',result)
    print(json.dumps(result,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/ppo.yaml')
    p.add_argument('--stage',choices=['cache','train','evaluate','all'],default='all')
    a=p.parse_args();cfg=load_yaml(a.config)
    if a.stage in ['cache','all']:cached_study(a.config)
    for epsilon in cfg['clip_values']:
        name=f'clip_{float(epsilon):.2f}'.replace('.','p');out=f'outputs/task2_ppo/{name}'
        if a.stage in ['train','all']:
            subprocess.run([sys.executable,'-u','-m','task2_ppo.continue_train',
                '--config',a.config,'--run-name',name,'--output',out,
                '--updates',str(cfg['fork_updates']),'--clip-epsilon',str(epsilon)],check=True)
        if a.stage in ['evaluate','all']:
            subprocess.run([sys.executable,'-u','-m','task2_ppo.evaluate',
                '--config',a.config,'--adapter',out,'--name',name],check=True)

if __name__=='__main__':main()
