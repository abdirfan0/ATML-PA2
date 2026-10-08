"""Exact token-controlled normalization forks on shared frozen-midpoint rollouts.

The standard experiment remains on-policy. This supplementary intervention is
fixed-rollout training: the behavior policy stays at the supplied midpoint and
both forks use its same saved log probabilities in the clipped likelihood ratio.
No claim is made that later batches were sampled from either updated fork.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, gc, json, time
import numpy as np
import torch
from torch.optim import AdamW
from common.data import load_yaml, read_jsonl, prompt_messages, repo_path
from common.generation import score_reward_pairs
from common.models import load_policy, load_tokenizer, load_reward_model, reference_mode, trainable_parameters
from common.logging_utils import set_seed
from task3_grpo.runtime import configure, sha, dump, records, generate, scored_microbatches, slice_batch, scores, trainable_state, restore
from task3_grpo.grpo import group_relative_advantages, mask_truncated_sequences, grpo_policy_loss

TENSOR_KEYS=['sequences','attention_mask','response_ids','response_mask']

def save_rows(path, data):
    tmp=path.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n' for x in data));tmp.replace(path)

def unpack(record,device='cuda'):
    batch=dict(record['batch'])
    for key in TENSOR_KEYS:
        batch[key]=torch.tensor(batch[key],device=device,dtype=torch.float32 if key=='response_mask' else torch.long)
    return batch

def collect(config):
    cfg=load_yaml(config);configure(cfg['seed']);root=repo_path(cfg['results_dir']);root.mkdir(parents=True,exist_ok=True)
    path=root/'shared_midpoint_rollouts.jsonl';mp=root/'shared_midpoint_rollouts_run.json';sp=root/'shared_midpoint_rollouts_summary.json'
    count=int(cfg['fork_updates']);groups=int(cfg['prompts_per_update']);K=int(cfg['num_generations']);cap=int(cfg['max_completion_length'])
    rows=read_jsonl(cfg['paths']['rl_prompt_train']);selected=np.random.default_rng(int(cfg['seed'])).permutation(len(rows)).tolist()[:count*groups]
    assert len(selected)==count*groups
    signature={'config':cfg,'updates':count,'selected_indices':selected,'prompt_ids':[rows[i]['prompt_id'] for i in selected],
        'midpoint_sha256':sha(repo_path(cfg['paths']['grpo_midpoint_policy'])/'adapter_model.safetensors'),
        'data_sha256':sha(cfg['paths']['rl_prompt_train']),'seed_rule':'config seed + update index; same K-completion layout',
        'behavior':'Frozen supplied midpoint for every batch; shared unchanged by both forks',
        'code_sha256':{'matched_normalization':sha('task3_grpo/matched_normalization.py'),'runtime':sha('task3_grpo/runtime.py'),'grpo':sha('task3_grpo/grpo.py')}}
    if mp.exists():
        if json.loads(mp.read_text())!=signature:raise RuntimeError('Shared-rollout signature changed; preserve existing artifacts and use a fresh results directory.')
        if sp.exists():
            assert read_jsonl(path) and sha(path)==json.loads(sp.read_text())['rollouts_sha256']
            print('Shared rollouts already complete.',flush=True);return
    else:
        if path.exists():raise RuntimeError('Rollouts without metadata.')
        dump(mp,signature)
    cached=records(path);assert [x['update'] for x in cached]==list(range(1,len(cached)+1))
    tok=load_tokenizer(cfg['base_model']);model=load_policy(cfg,adapter_path=cfg['paths']['grpo_midpoint_policy'],trainable=True)
    for p in trainable_parameters(model):p.data=p.data.float();p.requires_grad_(False)
    model.eval();rm,rtok=load_reward_model(cfg);began=time.perf_counter()
    for update in range(len(cached),count):
        set_seed(int(cfg['seed'])+update);indices=selected[update*groups:(update+1)*groups]
        prompts=[prompt_messages(rows[i]) for i in indices for _ in range(K)]
        batch=generate(model,tok,prompts,cfg,cap)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            old,entropy=scored_microbatches(model,batch,entropy=True)
            with reference_mode(model):ref,_=scored_microbatches(model,batch)
        reward=score_reward_pairs(rm,rtok,prompts,batch['responses'],max_length=int(cfg.get('reward_max_length',1280)))
        ids=torch.arange(groups,device='cuda').repeat_interleave(K);adv=group_relative_advantages(reward,ids)
        assert all(torch.isfinite(x).all() for x in [old,ref,reward,adv,entropy])
        packed={k:(v.detach().cpu().tolist() if torch.is_tensor(v) else v) for k,v in batch.items()}
        cached.append({'update':update+1,'source_indices':indices,'prompt_ids':[rows[i]['prompt_id'] for i in indices],
            'prompt_messages':prompts,'batch':packed,'old_logp':old.cpu().tolist(),'reference_logp':ref.cpu().tolist(),
            'rewards':reward.cpu().tolist(),'advantages':adv.cpu().tolist(),'entropy':entropy.cpu().tolist(),
            'generated_tokens':sum(batch['response_lengths'])})
        save_rows(path,cached);print(f'Shared frozen-midpoint rollouts: {update+1}/{count}',flush=True)
        del batch,old,ref,reward,adv,entropy
    summary={'updates':count,'num_completions':count*groups*K,'generated_tokens_once':sum(x['generated_tokens'] for x in cached),
        'generated_token_budget_per_condition':sum(x['generated_tokens'] for x in cached),'rollouts_sha256':sha(path),
        'collection_seconds_this_session':time.perf_counter()-began,
        'accounting':'Completions generated once and reused by both forks. Each fork consumes the exact same tokenized rollout pool; do not count collection cost twice.'}
    dump(sp,summary);print(json.dumps(summary,indent=2),flush=True)
    del model,rm;gc.collect();torch.cuda.empty_cache()

def train(config,kind):
    cfg=load_yaml(config);configure(cfg['seed']);root=repo_path(cfg['results_dir']);path=root/'shared_midpoint_rollouts.jsonl'
    pool=records(path);pool_summary=json.loads((root/'shared_midpoint_rollouts_summary.json').read_text())
    pool_meta=json.loads((root/'shared_midpoint_rollouts_run.json').read_text())
    assert pool_meta['config']==cfg
    assert pool_meta['midpoint_sha256']==sha(repo_path(cfg['paths']['grpo_midpoint_policy'])/'adapter_model.safetensors')
    assert sha(path)==pool_summary['rollouts_sha256'];assert len(pool)==int(cfg['fork_updates'])
    name='matched_canonical' if kind=='grpo' else 'matched_dr_grpo';out=repo_path('outputs/task3_grpo')/name
    mp=root/f'{name}_run.json';sp=root/f'{name}_summary.json';lp=root/f'{name}_training.jsonl';statepath=root/f'{name}_state.pt'
    signature={'config':cfg,'loss_type':kind,'shared_rollouts_sha256':sha(path),
        'midpoint_sha256':sha(repo_path(cfg['paths']['grpo_midpoint_policy'])/'adapter_model.safetensors'),
        'generated_token_budget':pool_summary['generated_token_budget_per_condition'],
        'protocol':'Fixed-rollout controlled fork, frozen-midpoint behavior likelihoods, no independent online resampling',
        'code_sha256':{'matched_normalization':sha('task3_grpo/matched_normalization.py'),'runtime':sha('task3_grpo/runtime.py'),'grpo':sha('task3_grpo/grpo.py')}}
    if mp.exists():
        if json.loads(mp.read_text())!=signature:raise RuntimeError('Matched fork signature changed.')
        if sp.exists():print('Completed matched fork:',name,flush=True);return
    else:
        if out.exists() or statepath.exists():raise RuntimeError('Existing matched output without metadata.')
        dump(mp,signature)
    tok=load_tokenizer(cfg['base_model']);model=load_policy(cfg,adapter_path=cfg['paths']['grpo_midpoint_policy'],trainable=True)
    for p in trainable_parameters(model):p.data=p.data.float()
    model.eval();parameters=trainable_parameters(model);opt=AdamW(parameters,lr=float(cfg['learning_rate']),weight_decay=0.)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024);logs=[];prior_time=0.
    if statepath.exists():
        state=torch.load(statepath,map_location='cpu',weights_only=False);restore(model,state['policy']);opt.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler']);logs=state['logs'];prior_time=state['elapsed_seconds'];del state
    began=time.perf_counter();torch.cuda.reset_peak_memory_stats();cap=int(cfg['max_completion_length'])
    for record in pool[len(logs):]:
        set_seed(int(cfg['seed'])+record['update']-1);batch=unpack(record);old=torch.tensor(record['old_logp'],device='cuda');ref=torch.tensor(record['reference_logp'],device='cuda');adv=torch.tensor(record['advantages'],device='cuda')
        mask=mask_truncated_sequences(batch['response_mask'],batch['truncated']) if cfg['mask_truncated_completions'] else batch['response_mask']
        active=float(mask.sum());N=len(record['advantages']);epochs=[];completion_stats=[]
        for epoch in range(int(cfg['policy_epochs']) if active else 0):
            opt.zero_grad(set_to_none=True);policy_total=kl_total=clips=0.
            for i in range(N):
                n=float(mask[i].sum());weight=0. if not n else 1/(N*(n if kind=='grpo' else cap))
                if not n:
                    if epoch==0:completion_stats.append({'generation_index':i,'length':batch['response_lengths'][i],'active_tokens':0,'token_weight':0.,'mean_abs_policy_logp_derivative':0.,'policy_logp_derivative_l1':0.})
                    continue
                with torch.autocast('cuda',dtype=torch.float16):
                    new,_=scores(model,slice_batch(batch,i,i+1))
                    micro,d=grpo_policy_loss(new,old[i:i+1],adv[i:i+1],mask[i:i+1],ref[i:i+1],float(cfg['clip_epsilon']),float(cfg['kl_beta'])*N*n/active,kind,cap)
                    loss=micro/N
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite matched loss.')
                if epoch==0:
                    with torch.no_grad():
                        ratio=(new.float()-old[i:i+1]).exp();a=float(adv[i]);epsilon=float(cfg['clip_epsilon'])
                        blocked=(ratio>1+epsilon) if a>0 else (ratio<1-epsilon)
                        derivative=abs(a)*ratio*weight*(~blocked)*mask[i:i+1]
                        completion_stats.append({'generation_index':i,'length':batch['response_lengths'][i],'active_tokens':int(n),'advantage':a,'token_weight':weight,
                          'mean_abs_policy_logp_derivative':float(derivative.sum()/n),'policy_logp_derivative_l1':float(derivative.sum())})
                scaler.scale(loss).backward();policy_total+=float(d['policy_term'])/N;kl_total+=float(d['sampled_kl'])*n/active;clips+=float(d['clip_fraction'])*n/active
                del new,loss,micro,d
            scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(parameters,float(cfg['max_grad_norm']));previous=scaler.get_scale();scaler.step(opt);scaler.update()
            epochs.append({'policy_loss':policy_total+float(cfg['kl_beta'])*kl_total,'policy_term':policy_total,'kl_estimator':kl_total,'clip_fraction':clips,'gradient_norm':float(norm) if torch.isfinite(norm) else None,'skipped':scaler.get_scale()<previous})
        logs.append({'update':record['update'],'source_indices':record['source_indices'],'prompt_ids':record['prompt_ids'],'generated_rollout_tokens':record['generated_tokens'],
            'active_loss_tokens':int(active),'optimization_epochs':epochs,'completion_statistics':completion_stats,
            'elapsed_seconds':prior_time+time.perf_counter()-began,'peak_vram_gib':torch.cuda.max_memory_allocated()/2**30})
        state={'policy':trainable_state(model),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'logs':logs,'elapsed_seconds':logs[-1]['elapsed_seconds']}
        tmp=statepath.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(statepath);del state;save_rows(lp,logs)
        print(f'{name}: update {record["update"]}/{len(pool)}, active tokens={int(active)}',flush=True)
        del batch,old,ref,adv,mask
    out.mkdir(parents=True,exist_ok=True);model.save_pretrained(out);tok.save_pretrained(out)
    summary={'run_name':name,'protocol':signature['protocol'],'updates':len(logs),'loss_type':kind,'adapter_path':str(out),
        'shared_rollouts_sha256':signature['shared_rollouts_sha256'],'generated_token_budget':signature['generated_token_budget'],
        'consumed_rollout_tokens':sum(x['generated_rollout_tokens'] for x in logs),'active_loss_tokens':sum(x['active_loss_tokens'] for x in logs),
        'new_tokens_generated_in_fork':0,'skipped_steps':sum(e['skipped'] for x in logs for e in x['optimization_epochs']),
        'wall_seconds':prior_time+time.perf_counter()-began,'peak_vram_gib':max(x['peak_vram_gib'] for x in logs)}
    assert summary['consumed_rollout_tokens']==summary['generated_token_budget'];dump(sp,summary);print(json.dumps(summary,indent=2),flush=True)
    del model,opt;gc.collect();torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/grpo.yaml');p.add_argument('--stage',choices=['collect','train','all'],default='all');a=p.parse_args()
    if a.stage in ['collect','all']:collect(a.config)
    if a.stage in ['train','all']:
        for kind in ['grpo','dr_grpo']:train(a.config,kind)

if __name__=='__main__':main()
