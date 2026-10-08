from __future__ import annotations
import argparse
import subprocess
import sys
import json
import numpy as np
from common.data import load_yaml,repo_path


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/grpo.yaml')
    p.add_argument('--stage',choices=['train','evaluate','summarize','all'],default='all')
    p.add_argument('--batch-size',type=int,default=4)
    a=p.parse_args();cfg=load_yaml(a.config);root=repo_path(cfg['results_dir'])
    names=['canonical','dr_grpo']
    for name,loss in zip(names,['grpo','dr_grpo']):
        out=f'outputs/task3_grpo/{name}'
        if a.stage in ['train','all']:
            subprocess.run([sys.executable,'-u','-m','task3_grpo.continue_train',
                '--config',a.config,'--run-name',name,'--output',out,'--updates',str(cfg['fork_updates']),
                '--loss-type',loss],check=True)
        if a.stage in ['evaluate','all']:
            subprocess.run([sys.executable,'-u','-m','task3_grpo.evaluate',
                '--config',a.config,'--adapter',out,'--name',name,'--batch-size',str(a.batch_size)],check=True)
    if a.stage in ['summarize','all']:
        metas=[json.loads((root/f'{n}_run.json').read_text())['signature'] for n in names]
        for key in ['config','selected_indices','prompt_ids','midpoint_policy_sha256','maximum_generated_token_allowance']:
            if metas[0][key]!=metas[1][key]:raise RuntimeError(f'Forks differ: {key}')
        report={'controlled_maximum_token_allowance':metas[0]['maximum_generated_token_allowance'],
            'token_budget_convention':'Same updates, prompts, K and generation cap; report actual EOS-dependent token counts separately.',
            'length_statistic':'Derivative of the policy surrogate with respect to selected-token log probability at ratio=1, excluding KL; mean absolute token derivative and total L1 mass. This is not a model-parameter gradient norm.',
            'conditions':{}}
        for n in names:
            logs=[json.loads(x) for x in (root/f'{n}_training.jsonl').read_text().splitlines()]
            completions=[c for r in logs for c in r['completions']]
            bins={}
            for label in ['short_le128','long_gt128']:
                selected=[c for c in completions if c['length_bin']==label and c['active_loss_tokens']>0]
                bins[label]={'num_active_completions':len(selected)}
                for key in ['response_length','relative_advantage','normalization_token_weight',
                            'policy_logp_gradient_mean_abs_at_ratio1','policy_logp_gradient_l1_at_ratio1']:
                    bins[label][key]=float(np.mean([r[key] for r in selected])) if selected else None
            report['conditions'][n]={'training':json.loads((root/f'{n}_summary.json').read_text()),
                'evaluation':json.loads((root/f'{n}_eval_summary.json').read_text()),'length_bins':bins}
        (root/'normalization_summary.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))

if __name__=='__main__':main()
