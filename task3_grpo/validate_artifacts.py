"""CPU integrity checks and matched-rollout normalization statistics."""
import argparse, hashlib, json
import numpy as np
from common.data import load_yaml, repo_path

def rows(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def read(p): return json.loads(p.read_text())
def close(a,b): np.testing.assert_allclose(a,b,rtol=2e-5,atol=2e-5)

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/grpo.yaml');a=p.parse_args()
 cfg=load_yaml(a.config);r=repo_path(cfg['results_dir']);metas={};evals={};logs_all={};runs={}
 for name,count in [('standard',20),('canonical',8),('dr_grpo',8)]:
  logs=rows(r/f'{name}_training.jsonl');logs_all[name]=logs;s=read(r/f'{name}_summary.json');m=read(r/f'{name}_run.json')['signature'];metas[name]=m
  assert len(logs)==count==s['updates'];assert [x['update'] for x in logs]==list(range(1,count+1))
  for f,h in m['code_sha256'].items(): assert hashlib.sha256(repo_path(f).read_bytes()).hexdigest()==h,f
  K=int(m['config']['num_generations']);cap=int(m['config']['max_completion_length'])
  assert [i for x in logs for i in x['source_indices']]==m['selected_indices']
  for x in logs:
   cs=x['completions'];assert len(cs)==K*len(x['source_indices']);stds=[]
   for pid in x['prompt_ids']:
    group=[c for c in cs if c['prompt_id']==pid];assert sorted(c['generation_index'] for c in group)==list(range(K))
    re=np.array([c['reward'] for c in group],dtype=np.float32);std=float(re.std());stds.append(std)
    close([c['relative_advantage'] for c in group],(re-re.mean())/(std+1e-6))
   close(x['group_reward_std_mean'],np.mean(stds));close(x['uninformative_group_fraction'],np.mean(np.array(stds)<=1e-6));close(x['raw_reward_mean'],np.mean([c['reward'] for c in cs]))
   for c in cs:
    n=c['response_length'];active=c['active_loss_tokens'];assert n==len(c['response_tokens']) and 0<n<=cap
    assert active==(0 if c['hit_generation_cap'] else n)
    if c['hit_generation_cap']: assert n==cap and not c['terminated_with_eos']
    w=0. if not active else 1/(len(cs)*(active if m['loss_type']=='grpo' else cap))
    close(c['normalization_token_weight'],w);close(c['policy_logp_gradient_mean_abs_at_ratio1'],abs(c['relative_advantage'])*w);close(c['policy_logp_gradient_l1_at_ratio1'],abs(c['relative_advantage'])*w*active)
   assert x['generated_tokens']==sum(c['response_length'] for c in cs);assert x['active_loss_tokens']==sum(c['active_loss_tokens'] for c in cs)
   assert x['no_active_tokens']==(x['active_loss_tokens']==0)
   for e in x['optimization_epochs']:
    for k in ['policy_loss','policy_gradient_norm','kl_penalty_estimator','clip_fraction']: assert e[k] is not None and np.isfinite(e[k])
  for k in ['generated_tokens','active_loss_tokens','skipped_policy_steps']: assert s[k]==sum(x[k] for x in logs)
  assert s['no_active_token_updates']==sum(x['no_active_tokens'] for x in logs)
  g=sorted(rows(r/f'{name}_generations.jsonl'),key=lambda x:x['row_index']);rw=rows(r/f'{name}_rewards.jsonl');es=read(r/f'{name}_eval_summary.json');em=read(r/f'{name}_eval_run.json');evals[name]=em
  assert len(g)==len(rw)==es['num_eval_prompts']==200
  assert [x['row_index'] for x in g]==sorted(x['row_index'] for x in rw)==list(range(200));assert [x['prompt_id'] for x in g]==em['prompt_ids']
  lens=np.array([x['response_length'] for x in g])
  for x in g:
   assert len(x['response_tokens'])==x['response_length'];assert x['batch_seed']==int(cfg['seed'])+(x['row_index']//4)*4
  close(es['reward_model_score_mean'],np.mean([x['reward'] for x in rw]));close(es['response_length_mean'],lens.mean());close(es['response_length_std'],lens.std());close(es['generation_cap_rate'],np.mean([x['hit_generation_cap'] for x in g]))
  for k in ['sampled_kl_token_mean','entropy_token_mean']: close(es[k],sum(x[k]*x['response_length'] for x in g)/lens.sum())
  runs[name]={'updates':count,'eval_rows':len(g),'generated_tokens':s['generated_tokens'],'active_loss_tokens':s['active_loss_tokens'],'no_active_token_updates':s['no_active_token_updates'],'skipped_policy_steps':s['skipped_policy_steps'],'maximum_online_clip_fraction':max(e['clip_fraction'] for x in logs for e in x['optimization_epochs'])}
 for k in ['config','selected_indices','prompt_ids','midpoint_policy_sha256','maximum_generated_token_allowance']: assert metas['canonical'][k]==metas['dr_grpo'][k],k
 for k in ['config','eval_data_sha256','row_indices','prompt_ids','generation_batch_size','score_microbatch_size','reward_batch_size','max_new_tokens','seed_rule','deterministic_algorithms','tf32']: assert evals['standard'][k]==evals['canonical'][k]==evals['dr_grpo'][k],k
 ignored=['elapsed_seconds','peak_vram_gib','generation_seconds']
 for x,y in zip(logs_all['standard'][:8],logs_all['canonical']): assert {k:v for k,v in x.items() if k not in ignored}=={k:v for k,v in y.items() if k not in ignored}
 shared=[c for x in logs_all['canonical'] for c in x['completions'] if c['active_loss_tokens']];stats={}
 for label in ['short_le128','long_gt128']:
  cs=[c for c in shared if c['length_bin']==label];out={'num_completions':len(cs)}
  for kind in ['canonical','dr_grpo']:
   w=np.array([1/(int(cfg['num_generations'])*(c['active_loss_tokens'] if kind=='canonical' else int(cfg['max_completion_length']))) for c in cs]);adv=np.abs([c['relative_advantage'] for c in cs]);lengths=np.array([c['active_loss_tokens'] for c in cs])
   out[kind]={'mean_token_weight':float(w.mean()),'mean_abs_token_derivative':float((w*adv).mean()),'mean_derivative_l1':float((w*adv*lengths).mean())}
  stats[label]=out
 report={'passed':True,'runs':runs,'checks':['training source hashes','within-prompt advantages','truncation masks','normalization derivatives','row-level summary recomputation','matched fork settings and prompts','matched evaluation settings and IDs','canonical reproduces standard first eight updates'],
 'limitations':['Actual generated-token counts differ; the forks share a maximum allowance, not exact token matching.','Raw sampled log-probability differences under tempered/top-p decoding are diagnostics, not unbiased full-policy KL.','Selected-token log-probability derivatives are not model-parameter gradient norms.'],
 'fixed_rollout_normalization':{'source':'Canonical active completions: identical lengths and advantages under both formulas','length_bins':stats}}
 (r/'artifact_validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
