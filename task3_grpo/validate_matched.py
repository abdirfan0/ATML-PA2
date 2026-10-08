"""CPU tests plus integrity validation of exact-token fixed-rollout forks."""
import argparse, hashlib, json
import numpy as np
from common.data import load_yaml, repo_path


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/grpo.yaml');p.add_argument('--tests-only',action='store_true');a=p.parse_args()
    if a.tests_only:
        import torch
        from task3_grpo.grpo import grpo_policy_loss
        from task3_grpo.matched_normalization import unpack
        for kind in ['grpo','dr_grpo']:
            mask=torch.tensor([[1.,1.,0.],[1.,1.,1.]])
            new=torch.tensor([[.4,-.4,0.],[.4,-.4,.1]],requires_grad=True)
            old=torch.zeros_like(new);adv=torch.tensor([1.,-1.])
            loss,_=grpo_policy_loss(new,old,adv,mask,old,.2,0.,kind,4)
            loss.backward()
            ratio=new.detach().exp();blocked=torch.where(adv[:,None]>0,ratio>1.2,ratio<.8)
            denom=mask.sum(-1) if kind=='grpo' else torch.full((2,),4.)
            predicted=-adv[:,None]*ratio/(2*denom[:,None])*(~blocked)*mask
            torch.testing.assert_close(new.grad,predicted)
        batch={'sequences':[[0,7,8,9]],'attention_mask':[[0,1,1,1]],'response_ids':[[9]],'response_mask':[[1.]],'prompt_width':3}
        out=unpack({'batch':json.loads(json.dumps(batch))},device='cpu')
        assert out['sequences'].dtype==torch.long and out['response_mask'].dtype==torch.float32
        assert out['prompt_width']==3
        print('PASS: actual-ratio clipped-surrogate derivatives and shared-rollout tensor reconstruction.')
        return
    cfg=load_yaml(a.config);root=repo_path(cfg['results_dir'])
    def read(n):return json.loads((root/n).read_text())
    def rows(n):return [json.loads(x) for x in (root/n).read_text().splitlines() if x.strip()]
    pool=rows('shared_midpoint_rollouts.jsonl');summary=read('shared_midpoint_rollouts_summary.json');meta=read('shared_midpoint_rollouts_run.json')
    digest=hashlib.sha256((root/'shared_midpoint_rollouts.jsonl').read_bytes()).hexdigest()
    assert digest==summary['rollouts_sha256'];assert len(pool)==int(cfg['fork_updates'])
    assert [x['update'] for x in pool]==list(range(1,len(pool)+1))
    K=int(cfg['num_generations']);cap=int(cfg['max_completion_length']);total=active=0
    for row in pool:
        b=row['batch'];mask=np.array(b['response_mask']);lengths=mask.sum(-1).astype(int)
        assert np.array_equal(lengths,b['response_lengths']);assert row['generated_tokens']==int(lengths.sum())
        ids=np.array(b['response_ids']);assert ids.shape==mask.shape==np.array(row['old_logp']).shape==np.array(row['reference_logp']).shape
        assert len(lengths)==K*len(row['source_indices'])
        for i in range(0,len(lengths),K):
            rewards=np.array(row['rewards'][i:i+K],dtype=np.float32)
            np.testing.assert_allclose(row['advantages'][i:i+K],(rewards-rewards.mean())/(rewards.std()+1e-6),rtol=2e-5,atol=2e-5)
        for n,trunc,eos in zip(lengths,b['truncated'],b['terminated_with_eos']):
            if trunc:assert n==cap and not eos
        total+=int(lengths.sum());active+=int(sum(n for n,t in zip(lengths,b['truncated']) if not t))
    assert total==summary['generated_token_budget_per_condition']==summary['generated_tokens_once']
    signatures=[];conditions={}
    for name in ['matched_canonical','matched_dr_grpo']:
        m=read(name+'_run.json');s=read(name+'_summary.json');logs=rows(name+'_training.jsonl');signatures.append(m)
        assert m['shared_rollouts_sha256']==s['shared_rollouts_sha256']==digest
        assert s['generated_token_budget']==s['consumed_rollout_tokens']==total
        assert s['active_loss_tokens']==active and s['new_tokens_generated_in_fork']==0
        assert len(logs)==s['updates']==len(pool)
        for filename,h in m['code_sha256'].items():
            path={'matched_normalization':'task3_grpo/matched_normalization.py','runtime':'task3_grpo/runtime.py','grpo':'task3_grpo/grpo.py'}[filename]
            assert hashlib.sha256(repo_path(path).read_bytes()).hexdigest()==h
        for row,log in zip(pool,logs):
            assert row['source_indices']==log['source_indices'] and row['prompt_ids']==log['prompt_ids']
            assert row['generated_tokens']==log['generated_rollout_tokens']
            for e in log['optimization_epochs']:assert e['gradient_norm'] is not None and np.isfinite(e['gradient_norm'])
        assert not s['skipped_steps'],'Skipped optimization steps; inspect before accepting comparison.'
        stats={}
        cs=[c for l in logs for c in l['completion_statistics'] if c['active_tokens']]
        for label in ['short_le128','long_gt128']:
            group=[c for c in cs if (c['length']<=128)==(label=='short_le128')]
            stats[label]={'num_completions':len(group)}
            for k in ['token_weight','mean_abs_policy_logp_derivative','policy_logp_derivative_l1']:
                stats[label][k]=float(np.mean([c[k] for c in group])) if group else None
        conditions[name]={'training':s,'length_conditioned_actual_ratio_statistics':stats}
        if (root/(name+'_eval_summary.json')).exists():conditions[name]['evaluation']=read(name+'_eval_summary.json')
    assert {k:v for k,v in signatures[0].items() if k!='loss_type'}=={k:v for k,v in signatures[1].items() if k!='loss_type'}
    evaluated=[(root/(n+'_eval_run.json')).exists() for n in conditions]
    if all(evaluated):
        ems=[read(n+'_eval_run.json') for n in conditions]
        for key in ['config','eval_data_sha256','row_indices','prompt_ids','generation_batch_size','score_microbatch_size','reward_batch_size','max_new_tokens','seed_rule','deterministic_algorithms','tf32']:
            assert ems[0][key]==ems[1][key],key
        for n in conditions:
            gen=rows(n+'_generations.jsonl');reward=rows(n+'_rewards.jsonl');es=read(n+'_eval_summary.json')
            assert sorted(x['row_index'] for x in gen)==sorted(x['row_index'] for x in reward)==list(range(200))
            assert es['num_eval_prompts']==len(gen)==len(reward)==200
            np.testing.assert_allclose(es['reward_model_score_mean'],np.mean([x['reward'] for x in reward]))
    report={'passed':True,'protocol':'Fixed-rollout normalization intervention; same frozen-midpoint completions and behavior log probabilities. Later batches are not sampled from updated forks.',
        'exact_generated_token_budget_per_condition':total,'exact_active_loss_tokens_per_condition':active,
        'shared_rollouts_sha256':digest,'collection_cost_accounting':'Generated once, consumed unchanged by each condition. Do not count generation cost twice.',
        'all_evaluations_present':all(evaluated),'conditions':conditions}
    (root/'matched_normalization_validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
