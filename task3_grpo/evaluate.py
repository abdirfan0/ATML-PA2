"""Batched, resumable held-out evaluation with explicit timing and batch seed rule."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import gc
import json
import time
import numpy as np
import torch
from common.data import load_yaml,read_jsonl,prompt_messages,repo_path
from common.generation import score_reward_pairs
from common.models import load_policy,load_reward_model,load_tokenizer,reference_mode
from common.logging_utils import set_seed
from task3_grpo.runtime import configure,sha,dump,append,records,generate,scored_microbatches


def evaluate(config_path,adapter,name,batch_size=4,limit=None):
    cfg=load_yaml(config_path);configure(cfg['seed'])
    root=repo_path(cfg['results_dir']);root.mkdir(parents=True,exist_ok=True)
    rows=read_jsonl(cfg['paths']['rl_prompt_eval'])
    if limit is not None:rows=rows[:int(limit)]
    if not rows or batch_size<1:raise ValueError('Nonempty evaluation and positive batch size required.')
    cap=int(cfg.get('eval_max_completion_length',cfg['cache_generation_cap']))
    signature={'config':cfg,'adapter':str(repo_path(adapter)),
        'adapter_sha256':sha(repo_path(adapter)/'adapter_model.safetensors'),
        'eval_data_sha256':sha(cfg['paths']['rl_prompt_eval']),
        'row_indices':list(range(len(rows))),'prompt_ids':[r['prompt_id'] for r in rows],
        'generation_batch_size':batch_size,'score_microbatch_size':1,
        'reward_batch_size':batch_size,'max_new_tokens':cap,
        'seed_rule':'config seed + first original row index in fixed batch; fixed ordered batches across policies',
        'deterministic_algorithms':True,'tf32':False,
        'code_sha256':{'evaluate':sha('task3_grpo/evaluate.py'),'runtime':sha('task3_grpo/runtime.py')}}
    mp=root/f'{name}_eval_run.json';gp=root/f'{name}_generations.jsonl';rp=root/f'{name}_rewards.jsonl'
    sp=root/f'{name}_eval_summary.json';tp=root/f'{name}_eval_timing.json'
    if mp.exists():
        if json.loads(mp.read_text())!=signature:raise RuntimeError('Evaluation signature differs; use a new name.')
        if sp.exists():print('Completed evaluation:',sp,flush=True);return json.loads(sp.read_text())
    else:dump(mp,signature)
    previous=json.loads(tp.read_text()) if tp.exists() else {}
    timing={key:float(previous.get(key,0.)) for key in ['model_loading_seconds','generation_seconds','policy_reference_scoring_seconds','reward_scoring_seconds']}
    existing=records(gp);done={r['row_index'] for r in existing}
    if len(done)!=len(existing):raise RuntimeError('Duplicate generation records.')
    if len(done)<len(rows):
        started=time.perf_counter();tok=load_tokenizer(cfg['base_model']);model=load_policy(cfg,adapter_path=adapter)
        timing['model_loading_seconds']+=time.perf_counter()-started
        for start in range(0,len(rows),batch_size):
            indices=list(range(start,min(start+batch_size,len(rows))))
            if set(indices)<=done:continue
            if done.intersection(indices):raise RuntimeError('Incomplete generation batch; remove only that batch records before retrying.')
            set_seed(int(cfg['seed'])+start)
            prompts=[prompt_messages(rows[i]) for i in indices]
            torch.cuda.synchronize();began=time.perf_counter()
            batch=generate(model,tok,prompts,cfg,cap)
            torch.cuda.synchronize();timing['generation_seconds']+=time.perf_counter()-began
            began=time.perf_counter()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                lp,ent=scored_microbatches(model,batch,entropy=True)
                with reference_mode(model):ref,_=scored_microbatches(model,batch)
            torch.cuda.synchronize();timing['policy_reference_scoring_seconds']+=time.perf_counter()-began
            new=[]
            for j,i in enumerate(indices):
                n=batch['response_lengths'][j]
                kl=float((lp[j,:n]-ref[j,:n]).mean());entropy=float(ent[j,:n].mean())
                if not np.isfinite([kl,entropy]).all():raise RuntimeError('Nonfinite evaluation scores.')
                new.append({'row_index':i,'prompt_id':rows[i]['prompt_id'],'prompt_messages':prompts[j],
                    'response':batch['responses'][j],'response_tokens':batch['response_ids'][j,:n].cpu().tolist(),
                    'response_length':n,'terminated_with_eos':batch['terminated_with_eos'][j],
                    'hit_generation_cap':batch['truncated'][j],'sampled_kl_token_mean':kl,
                    'entropy_token_mean':entropy,'batch_seed':int(cfg['seed'])+start})
            # Write a complete batch atomically to keep batch-seeded resumption stable.
            existing.extend(new);tmp=gp.with_suffix('.tmp')
            tmp.write_text(''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n' for x in existing));tmp.replace(gp)
            dump(tp,timing)
            print(f'{name}: generation {indices[-1]+1}/{len(rows)}; generation seconds={timing["generation_seconds"]:.1f}',flush=True)
            del batch,lp,ent,ref
        del model;gc.collect();torch.cuda.empty_cache()
    generated=sorted(records(gp),key=lambda r:r['row_index']);done={r['row_index'] for r in records(rp)}
    pending=[r for r in generated if r['row_index'] not in done]
    if pending:
        began=time.perf_counter();rm,rtok=load_reward_model(cfg)
        timing['model_loading_seconds']+=time.perf_counter()-began
        for start in range(0,len(pending),batch_size):
            chunk=pending[start:start+batch_size]
            torch.cuda.synchronize();began=time.perf_counter()
            values=score_reward_pairs(rm,rtok,[r['prompt_messages'] for r in chunk],[r['response'] for r in chunk],
                                     max_length=int(cfg.get('reward_max_length',1280))).cpu().tolist()
            torch.cuda.synchronize();timing['reward_scoring_seconds']+=time.perf_counter()-began
            for r,v in zip(chunk,values):
                if not np.isfinite(v):raise RuntimeError('Nonfinite reward.')
                append(rp,{'row_index':r['row_index'],'reward':v})
            dump(tp,timing);print(f'{name}: reward scoring {min(start+batch_size,len(pending))}/{len(pending)}',flush=True)
        del rm;gc.collect();torch.cuda.empty_cache()
    rewards=records(rp)
    for label,data in [('generation',generated),('reward',rewards)]:
        ids=[r['row_index'] for r in data]
        if len(ids)!=len(rows) or set(ids)!=set(range(len(rows))):raise RuntimeError(f'Incomplete/duplicate {label}.')
    lens=np.array([r['response_length'] for r in generated],dtype=float)
    summary={'name':name,'num_eval_prompts':len(rows),
        'reward_model_score_mean':float(np.mean([r['reward'] for r in rewards])),
        'response_length_mean':float(lens.mean()),'response_length_std':float(lens.std()),
        'generation_cap_rate':float(np.mean([r['hit_generation_cap'] for r in generated]))}
    for key in ['sampled_kl_token_mean','entropy_token_mean']:
        summary[key]=float(sum(r[key]*r['response_length'] for r in generated)/lens.sum())
    timing['generated_tokens']=int(lens.sum());timing['generation_tokens_per_second']=float(lens.sum()/max(timing['generation_seconds'],1e-9))
    dump(tp,timing);dump(sp,summary);print(json.dumps({'summary':summary,'timing':timing},indent=2),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/grpo.yaml')
    p.add_argument('--adapter',required=True);p.add_argument('--name',default='standard')
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--limit',type=int)
    a=p.parse_args();evaluate(a.config,a.adapter,a.name,a.batch_size,a.limit)

if __name__=='__main__':main()
