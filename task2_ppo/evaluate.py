from __future__ import annotations
import argparse
import gc
import json
import numpy as np
import torch
from common.data import load_yaml, read_jsonl, prompt_messages, repo_path
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import set_seed
from common.metrics import masked_mean, sampled_kl
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from task2_ppo.runtime import (cuda_required, sha, dump, append, records,
                              normal_tensors, policy_scores)


def evaluate(config_path, adapter, name):
    cuda_required(); cfg=load_yaml(config_path)
    root=repo_path(cfg['results_dir']);root.mkdir(parents=True,exist_ok=True)
    rows=read_jsonl(cfg['paths']['rl_prompt_eval'])
    gp=root/f'{name}_generations.jsonl';rp=root/f'{name}_rewards.jsonl'
    mp=root/f'{name}_eval_run.json';sp=root/f'{name}_eval_summary.json'
    signature={'config':cfg,'adapter':str(repo_path(adapter)),
        'adapter_sha256':sha(repo_path(adapter)/'adapter_model.safetensors'),
        'eval_data_sha256':sha(cfg['paths']['rl_prompt_eval']),
        'row_indices':list(range(len(rows))),
        'prompt_ids':[r.get('prompt_id') for r in rows],
        'seed_rule':'config seed + original row index', 'generation_batch_size':1,
        'kl_averaging':'response-token weighted across examples',
        'entropy':'full categorical entropy of raw policy over response states',
        'code_sha256':{'evaluate':sha('task2_ppo/evaluate.py'),
                       'runtime':sha('task2_ppo/runtime.py')}}
    if mp.exists():
        if json.loads(mp.read_text()) != signature:
            raise RuntimeError('Existing evaluation differs; use another name.')
    else:dump(mp,signature)
    done={r['row_index'] for r in records(gp)}
    if len(done)<len(rows):
        tok=load_tokenizer(cfg['base_model'])
        model=load_policy(cfg,adapter_path=adapter,trainable=False)
        model.config.use_cache=True
        for i,row in enumerate(rows):
            if i in done:continue
            set_seed(int(cfg['seed'])+i);messages=prompt_messages(row)
            generated=normal_tensors(batch_generate(model,tok,[messages],
                max_prompt_length=int(cfg['max_prompt_length']),
                max_new_tokens=int(cfg['eval_max_response_length']),**cfg['generation']))
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                policy,entropy=policy_scores(model,generated,entropy=True)
                with reference_mode(model):ref,_=policy_scores(model,generated)
                mask=generated['response_mask']
                kl=sampled_kl(policy,ref,mask).item()
                ent=masked_mean(entropy,mask).item()
            if not np.isfinite([kl,ent]).all():raise RuntimeError(f'Non-finite scores: {i}')
            n=generated['response_lengths'][0]
            append(gp,{'row_index':i,'prompt_id':row.get('prompt_id'),
                'prompt_messages':messages,'response':generated['responses'][0],
                'response_tokens':generated['response_ids'][0,:n].cpu().tolist(),
                'response_length':n,'sampled_kl_token_mean':kl,
                'entropy_token_mean':ent,'seed':int(cfg['seed'])+i,
                'terminated_with_eos':generated['terminated_with_eos'][0],
                'hit_generation_cap':generated['truncated'][0]})
            print(f'{name}: generation {i+1}/{len(rows)}; tokens={n}',flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    generations=records(gp);done={r['row_index'] for r in records(rp)}
    pending=[r for r in generations if r['row_index'] not in done]
    if pending:
        rm,tok=load_reward_model(cfg)
        for j,r in enumerate(pending):
            score=score_reward_pairs(rm,tok,[r['prompt_messages']],[r['response']],
                                     max_length=int(cfg['reward_max_length']))[0].item()
            if not np.isfinite(score):raise RuntimeError('Non-finite reward.')
            effective=score-float(cfg['missing_eos_penalty'])*(not r['terminated_with_eos'])
            append(rp,{'row_index':r['row_index'],'reward':score,'effective_reward':effective})
            print(f'{name}: reward scoring {j+1}/{len(pending)}',flush=True)
    rewards=records(rp);expected=set(range(len(rows)))
    for label,data in [('generations',generations),('rewards',rewards)]:
        ids=[r['row_index'] for r in data]
        if len(ids)!=len(expected) or set(ids)!=expected:
            raise RuntimeError(f'Incomplete or duplicate {label}.')
    lengths=np.array([r['response_length'] for r in generations],dtype=float)
    summary={'name':name,'num_eval_prompts':len(rows),
        'reward_model_score_mean':float(np.mean([r['reward'] for r in rewards])),
        'effective_reward_mean':float(np.mean([r['effective_reward'] for r in rewards])),
        'response_length_mean':float(lengths.mean()),'response_length_std':float(lengths.std()),
        'generation_cap_rate':float(np.mean([r['hit_generation_cap'] for r in generations]))}
    for key in ['sampled_kl_token_mean','entropy_token_mean']:
        summary[key]=float(sum(r[key]*r['response_length'] for r in generations)/lengths.sum())
    dump(sp,summary);print(json.dumps(summary,indent=2),flush=True);return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/ppo.yaml')
    p.add_argument('--adapter',required=True);p.add_argument('--name',default='standard')
    a=p.parse_args();evaluate(a.config,a.adapter,a.name)

if __name__=='__main__':main()
