"""Equal-generation CPU diagnostic from the supplied K=8 cache."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import numpy as np
from common.data import load_yaml, read_jsonl, repo_path

EPS = 1e-6

def load_k8_cache(path):
    by_prompt = defaultdict(list)
    for row in read_jsonl(path):
        by_prompt[int(row['source_index'])].append(row)
    for index, group in by_prompt.items():
        group.sort(key=lambda x: int(x['generation_index']))
        if len(group) != 8 or [r['generation_index'] for r in group] != list(range(8)):
            raise ValueError(f'Expected exactly eight unique ordered completions: {index}')
        if len({r['prompt_id'] for r in group}) != 1:
            raise ValueError('Inconsistent prompt IDs.')
    return dict(sorted(by_prompt.items()))

def regroup_equal_generation_budget(by_prompt, k):
    if k not in [2, 4, 8]:
        raise ValueError('K must be 2, 4 or 8.')
    return [group[start:start+k] for group in by_prompt.values() for start in range(0, 8, k)]

def statistics(groups):
    advantages, stds = [], []
    for group in groups:
        rewards = np.array([r['reward'] for r in group], dtype=float)
        std = rewards.std(ddof=0)
        stds.append(float(std))
        advantages.extend(((rewards - rewards.mean()) / (std + EPS)).tolist())
    return {'num_groups':len(groups), 'num_completions':sum(map(len, groups)),
            'informative_group_rate':float(np.mean(np.array(stds) > EPS)),
            'uninformative_group_fraction':float(np.mean(np.array(stds) <= EPS)),
            'mean_group_reward_std':float(np.mean(stds)),
            'group_relative_signal_variance':float(np.var(advantages, ddof=0))}

def main():
    ap = argparse.ArgumentParser();ap.add_argument('--config',default='configs/grpo.yaml')
    args = ap.parse_args();cfg = load_yaml(args.config)
    grouped = load_k8_cache(cfg['group_cache'])
    # One difficulty assignment per prompt, independent of regrouping K.
    means = {i:float(np.mean([r['reward'] for r in g])) for i,g in grouped.items()}
    threshold = float(np.median(list(means.values())))
    bins = {'lower_mean_reward':[i for i in grouped if means[i] <= threshold],
            'higher_mean_reward':[i for i in grouped if means[i] > threshold]}
    if any(not x for x in bins.values()):raise ValueError('Difficulty bins must both be nonempty.')
    conditions = []
    for k in cfg['group_sizes']:
        result = {'K':int(k), **statistics(regroup_equal_generation_budget(grouped,k))}
        result['difficulty_bins'] = {
            name:statistics(regroup_equal_generation_budget({i:grouped[i] for i in ids},k))
            for name,ids in bins.items()}
        conditions.append(result)
    assert len({r['num_completions'] for r in conditions}) == 1
    result = {'cache_sha256':hashlib.sha256(repo_path(cfg['group_cache']).read_bytes()).hexdigest(),
        'num_prompts':len(grouped), 'epsilon':EPS,
        'partition_rule':'For each prompt, sort generation_index 0..7, partition consecutive K-sized blocks; reuse all eight completions at every K.',
        'reward_rule':'Use supplied raw reward; all cached completions retained, including capped responses. No new generations.',
        'difficulty_rule':'Full K=8 mean reward per prompt, pooled median threshold; lower <= median, higher > median. Proxy for reward difficulty, not ground-truth correctness.',
        'difficulty_threshold':threshold,
        'difficulty_bin_source_indices':bins,
        'prompt_records':[{'source_index':i,'prompt_id':g[0]['prompt_id'],'mean_reward_k8':means[i]} for i,g in grouped.items()],
        'conditions':conditions}
    root = repo_path(cfg['results_dir']);root.mkdir(parents=True,exist_ok=True)
    (root/'group_size_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__ == '__main__':main()
