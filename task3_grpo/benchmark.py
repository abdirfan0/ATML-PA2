"""Measure evaluation throughput on eight prompts before a full GPU batch."""
from __future__ import annotations
import argparse
import json
from common.data import load_yaml,repo_path
from task3_grpo.evaluate import evaluate
from task3_grpo.runtime import dump


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/grpo.yaml')
    p.add_argument('--limit',type=int,default=8)
    a=p.parse_args();cfg=load_yaml(a.config);root=repo_path(cfg['results_dir'])
    results={}
    for batch_size in [1,4]:
        name=f'benchmark_b{batch_size}_n{a.limit}'
        evaluate(a.config,cfg['paths']['grpo_midpoint_policy'],name,batch_size,a.limit)
        results[str(batch_size)]=json.loads((root/f'{name}_eval_timing.json').read_text())
    report={'conditions':results,
        'generation_tokens_per_second_ratio_b4_over_b1':
            results['4']['generation_tokens_per_second']/results['1']['generation_tokens_per_second'],
        'scope':'Throughput benchmark only. Batched sampling changes the random-number layout, so generated responses and token counts may differ. All final policy comparisons must use the same batch size.'}
    dump(root/'evaluation_benchmark.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
