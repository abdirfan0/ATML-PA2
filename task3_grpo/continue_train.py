"""On-policy GRPO continuation from the fixed supplied midpoint."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import random
import time
import numpy as np
import torch
from torch.optim import AdamW
from common.data import load_yaml,read_jsonl,prompt_messages,repo_path
from common.generation import score_reward_pairs
from common.models import load_policy,load_reward_model,load_tokenizer,trainable_parameters,reference_mode
from common.metrics import masked_mean
from common.logging_utils import set_seed
from task3_grpo.grpo import group_relative_advantages,grpo_policy_loss,mask_truncated_sequences
from task3_grpo.runtime import configure,sha,dump,generate,scored_microbatches,slice_batch,scores,trainable_state,restore


def run_grpo(config_path,output=None,updates=None,loss_type='grpo',run_name='standard'):
    cfg=load_yaml(config_path)
    if updates is not None:cfg['updates']=int(updates)
    configure(cfg['seed'])
    root=repo_path(cfg['results_dir']);root.mkdir(parents=True,exist_ok=True)
    out=repo_path(output or cfg['output']);mp=root/f'{run_name}_run.json';sp=root/f'{run_name}_summary.json'
    state_path=root/f'{run_name}_state.pt';log_path=root/f'{run_name}_training.jsonl'
    rows=read_jsonl(cfg['paths']['rl_prompt_train']);groups=int(cfg['prompts_per_update']);K=int(cfg['num_generations'])
    count=int(cfg['updates'])*groups
    order=np.random.default_rng(int(cfg['seed'])).permutation(len(rows)).tolist()
    if count>len(order):raise ValueError('Update budget exceeds prompt pool.')
    selected=order[:count];cap=int(cfg['max_completion_length'])
    signature={'config':cfg,'loss_type':loss_type,'output':str(out),'selected_indices':selected,
        'prompt_ids':[rows[i]['prompt_id'] for i in selected],
        'train_data_sha256':sha(cfg['paths']['rl_prompt_train']),
        'midpoint_policy_sha256':sha(repo_path(cfg['paths']['grpo_midpoint_policy'])/'adapter_model.safetensors'),
        'reference':'Frozen original base policy, LoRA disabled',
        'maximum_generated_token_allowance':int(cfg['updates'])*groups*K*cap,
        'actual_generated_tokens':'Recorded separately; EOS can shorten completions.',
        'rollout_seed_rule':'config seed + zero-based update index; identical K-completion batch layout',
        'deterministic_algorithms':True,'tf32':False,'initial_amp_scale':1024,
        'code_sha256':{f:sha(f) for f in ['task3_grpo/continue_train.py','task3_grpo/grpo.py','task3_grpo/runtime.py']}}
    if mp.exists():
        if json.loads(mp.read_text())['signature']!=signature:raise RuntimeError('Run differs; use a new name/output.')
        if sp.exists():print('Completed run:',sp,flush=True);return json.loads(sp.read_text())
    else:
        if out.exists() or state_path.exists():raise RuntimeError('Existing output without matching metadata.')
        dump(mp,{'signature':signature,'gpu':torch.cuda.get_device_name(0),'torch_version':torch.__version__})
    tok=load_tokenizer(cfg['base_model']);policy=load_policy(cfg,adapter_path=cfg['paths']['grpo_midpoint_policy'],trainable=True)
    for p in trainable_parameters(policy):p.data=p.data.float()
    policy.eval()  # Disable dropout for stable old/new likelihood ratios.
    rm,rtok=load_reward_model(cfg)
    parameters=trainable_parameters(policy);optimizer=AdamW(parameters,lr=float(cfg['learning_rate']),weight_decay=0.)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024)
    logs=[];begin_update=0;elapsed_before=0.
    if state_path.exists():
        state=torch.load(state_path,map_location='cpu',weights_only=False)
        restore(policy,state['policy']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        logs=state['logs'];begin_update=state['completed_updates'];elapsed_before=state['elapsed_seconds']
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);del state
    torch.cuda.reset_peak_memory_stats();began=time.perf_counter()
    for update in range(begin_update,int(cfg['updates'])):
        set_seed(int(cfg['seed'])+update)
        indices=selected[update*groups:(update+1)*groups]
        prompt_group=[prompt_messages(rows[i]) for i in indices]
        prompts=[p for p in prompt_group for _ in range(K)]
        ids=torch.arange(groups,device='cuda').repeat_interleave(K)
        torch.cuda.synchronize();started=time.perf_counter()
        batch=generate(policy,tok,prompts,cfg,cap)
        torch.cuda.synchronize();generation_seconds=time.perf_counter()-started
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            old,ent=scored_microbatches(policy,batch,entropy=True)
            with reference_mode(policy):ref,_=scored_microbatches(policy,batch)
        rewards=score_reward_pairs(rm,rtok,prompts,batch['responses'],max_length=int(cfg.get('reward_max_length',1280)))
        if not torch.isfinite(rewards).all():raise RuntimeError('Nonfinite rollout rewards.')
        advantage=group_relative_advantages(rewards,ids).detach()
        mask=batch['response_mask']
        loss_mask=mask_truncated_sequences(mask,batch['truncated']) if cfg['mask_truncated_completions'] else mask
        active_tokens=float(loss_mask.sum());num_sequences=len(prompts)
        stds=[float(rewards[ids==i].std(unbiased=False)) for i in range(groups)]
        diagnostics=[];skipped=0
        if active_tokens>0:
            for epoch in range(int(cfg['policy_epochs'])):
                optimizer.zero_grad(set_to_none=True);policy_term=kl_term=clip_fraction=0.
                for i in range(num_sequences):
                    local_mask=loss_mask[i:i+1]
                    n=float(local_mask.sum())
                    if n==0:continue
                    with torch.autocast('cuda',dtype=torch.float16):
                        new,_=scores(policy,slice_batch(batch,i,i+1))
                        # Sequence mean for the surrogate; active-token mean for KL.
                        micro_loss,d=grpo_policy_loss(new,old[i:i+1],advantage[i:i+1],local_mask,ref[i:i+1],
                             float(cfg['clip_epsilon']),float(cfg['kl_beta'])*num_sequences*n/active_tokens,loss_type,cap)
                        loss=micro_loss/num_sequences
                        local_policy=d['policy_term']/num_sequences
                        local_kl=d['sampled_kl']*n/active_tokens
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite policy loss.')
                    scaler.scale(loss).backward()
                    policy_term+=float(local_policy.detach());kl_term+=float(local_kl.detach())
                    clip_fraction+=float(d['clip_fraction'])*n/active_tokens
                    del new,loss,micro_loss,local_policy,local_kl,d
                scaler.unscale_(optimizer)
                norm=torch.nn.utils.clip_grad_norm_(parameters,float(cfg['max_grad_norm']))
                previous=scaler.get_scale();scaler.step(optimizer);scaler.update()
                skip=int(scaler.get_scale()<previous);skipped+=skip
                diagnostics.append({'policy_term':policy_term,'kl_penalty_estimator':kl_term,
                    'policy_loss':policy_term+cfg['kl_beta']*kl_term,'clip_fraction':clip_fraction,
                    'policy_gradient_norm':float(norm) if torch.isfinite(norm) else None,'skipped_step':bool(skip)})
        lengths=batch['response_lengths'];completion_records=[]
        for i in range(num_sequences):
            group=i//K;n=lengths[i];active=float(loss_mask[i].sum());a=float(advantage[i])
            denominator=max(active,1.) if loss_type=='grpo' else cap
            weight=1./(num_sequences*denominator) if active>0 else 0.
            completion_records.append({'source_index':indices[group],'prompt_id':rows[indices[group]]['prompt_id'],
                'generation_index':i%K,'prompt_messages':prompts[i],'response':batch['responses'][i],
                'response_tokens':batch['response_ids'][i,:n].cpu().tolist(),'response_length':n,
                'reward':float(rewards[i]),'relative_advantage':a,
                'terminated_with_eos':batch['terminated_with_eos'][i],'hit_generation_cap':batch['truncated'][i],
                'active_loss_tokens':int(active),'length_bin':'short_le128' if n<=128 else 'long_gt128',
                'normalization_token_weight':weight,
                'policy_logp_gradient_mean_abs_at_ratio1':abs(a)*weight,
                'policy_logp_gradient_l1_at_ratio1':abs(a)*weight*active})
        def conditioned(key):
            return {label:{'num_active_completions':len(v),key:float(np.mean(v)) if v else None}
                    for label in ['short_le128','long_gt128']
                    for v in [[r[key] for r in completion_records if r['length_bin']==label and r['active_loss_tokens']>0]]}
        record={'update':update+1,'source_indices':indices,'prompt_ids':[rows[i]['prompt_id'] for i in indices],
            'raw_reward_mean':float(rewards.mean()),'group_reward_std_mean':float(np.mean(stds)),
            'uninformative_group_fraction':float(np.mean(np.array(stds)<=1e-6)),
            'sampled_kl_token_mean':float(masked_mean(old-ref,mask)),
            'entropy_token_mean':float(masked_mean(ent,mask)),
            'response_length_mean':float(np.mean(lengths)),'generated_tokens':int(sum(lengths)),
            'generation_cap_rate':float(np.mean(batch['truncated'])),
            'active_loss_tokens':int(active_tokens),'no_active_tokens':active_tokens==0,
            'generation_seconds':generation_seconds,'skipped_policy_steps':skipped,
            'optimization_epochs':diagnostics,
            'policy_loss':float(np.mean([d['policy_loss'] for d in diagnostics])) if diagnostics else None,
            'clip_fraction':float(np.mean([d['clip_fraction'] for d in diagnostics])) if diagnostics else None,
            'length_conditioned_statistics':conditioned('policy_logp_gradient_l1_at_ratio1'),
            'completions':completion_records,'elapsed_seconds':elapsed_before+time.perf_counter()-began,
            'peak_vram_gib':torch.cuda.max_memory_allocated()/2**30}
        logs.append(record)
        state={'completed_updates':update+1,'policy':trainable_state(policy),'optimizer':optimizer.state_dict(),
            'scaler':scaler.state_dict(),'logs':logs,'elapsed_seconds':record['elapsed_seconds'],
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
            'python_rng':random.getstate(),'numpy_rng':np.random.get_state()}
        tmp=state_path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(state_path);del state
        tmp=log_path.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in logs));tmp.replace(log_path)
        print(f'{run_name}: {update+1}/{cfg["updates"]} reward={record["raw_reward_mean"]:.3f} group_std={record["group_reward_std_mean"]:.3f} active_tokens={int(active_tokens)} skipped={skipped}',flush=True)
        del batch,old,ref,ent,rewards,advantage,mask,loss_mask
    out.mkdir(parents=True,exist_ok=True);policy.save_pretrained(out);tok.save_pretrained(out)
    steps=sum(len(r['optimization_epochs']) for r in logs)
    summary={'run_name':run_name,'loss_type':loss_type,'updates':len(logs),'adapter_path':str(out),
        'generated_tokens':sum(r['generated_tokens'] for r in logs),
        'maximum_generated_token_allowance':signature['maximum_generated_token_allowance'],
        'active_loss_tokens':sum(r['active_loss_tokens'] for r in logs),
        'no_active_token_updates':sum(r['no_active_tokens'] for r in logs),
        'skipped_policy_steps':sum(r['skipped_policy_steps'] for r in logs),
        'policy_skip_fraction':sum(r['skipped_policy_steps'] for r in logs)/max(steps,1),
        'wall_seconds':elapsed_before+time.perf_counter()-began,
        'peak_vram_gib':max(r['peak_vram_gib'] for r in logs)}
    dump(sp,summary);print(json.dumps(summary,indent=2),flush=True);return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/grpo.yaml')
    p.add_argument('--output');p.add_argument('--updates',type=int);p.add_argument('--run-name',default='standard')
    p.add_argument('--loss-type',choices=['grpo','dr_grpo'],default='grpo')
    a=p.parse_args();run_grpo(a.config,a.output,a.updates,a.loss_type,a.run_name)

if __name__=='__main__':main()
