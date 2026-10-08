"""Recompute every matched evaluation metric from saved CPU records."""
import argparse, hashlib, json
import numpy as np
from common.data import load_yaml,repo_path
from task3_grpo.validate_matched import main as validate

def main():
    validate()
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/grpo.yaml');a=p.parse_args();cfg=load_yaml(a.config);r=repo_path(cfg['results_dir']);checked={}
    def read(n):return json.loads((r/n).read_text())
    def rows(n):return [json.loads(x) for x in (r/n).read_text().splitlines() if x.strip()]
    signatures=[]
    for name in ['matched_canonical','matched_dr_grpo']:
        em=read(name+'_eval_run.json');signatures.append(em);es=read(name+'_eval_summary.json')
        for key,path in [('evaluate','task3_grpo/evaluate.py'),('runtime','task3_grpo/runtime.py')]:assert hashlib.sha256(repo_path(path).read_bytes()).hexdigest()==em['code_sha256'][key]
        g=sorted(rows(name+'_generations.jsonl'),key=lambda x:x['row_index']);rew=rows(name+'_rewards.jsonl')
        assert [x['row_index'] for x in g]==sorted(x['row_index'] for x in rew)==list(range(200))
        assert [x['prompt_id'] for x in g]==em['prompt_ids'];assert em['generation_batch_size']==4
        n=np.array([x['response_length'] for x in g])
        for x in g:
            assert len(x['response_tokens'])==x['response_length']
            assert x['batch_seed']==int(cfg['seed'])+(x['row_index']//4)*4
            assert 0<x['response_length']<=em['max_new_tokens']
            assert not(x['terminated_with_eos'] and x['hit_generation_cap'])
        calculated={'reward_model_score_mean':np.mean([x['reward'] for x in rew]),'response_length_mean':n.mean(),'response_length_std':n.std(),
                    'generation_cap_rate':np.mean([x['hit_generation_cap'] for x in g])}
        for key in ['sampled_kl_token_mean','entropy_token_mean']:calculated[key]=sum(x[key]*x['response_length'] for x in g)/n.sum()
        for key,value in calculated.items():
            assert np.isfinite(value);np.testing.assert_allclose(value,es[key],rtol=1e-8,atol=1e-10)
        timing=read(name+'_eval_timing.json');assert timing['generated_tokens']==int(n.sum())
        np.testing.assert_allclose(timing['generation_tokens_per_second'],n.sum()/timing['generation_seconds'])
        checked[name]={'all_eval_metrics_recomputed':True,'source_hashes_match':True,'eval_rows':len(g)}
    ga=sorted(rows('matched_canonical_generations.jsonl'),key=lambda x:x['row_index']);gb=sorted(rows('matched_dr_grpo_generations.jsonl'),key=lambda x:x['row_index'])
    assert [x['prompt_messages'] for x in ga]==[x['prompt_messages'] for x in gb]
    summary={'passed':True,'checks':checked,'matched_full_prompt_messages':True,'protocol':'Exact-token fixed-rollout comparison; retain original online forks as separate experiments.'}
    (r/'final_matched_artifact_validation.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
