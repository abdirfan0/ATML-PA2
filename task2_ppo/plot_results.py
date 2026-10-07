"""CPU-only plots and tables from saved PPO results; never reruns models."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/task2_ppo'
FIGURES = ROOT / 'report/figures'
NAMES = ['standard', 'clip_0p05', 'clip_0p20', 'clip_0p50',
         'kl_0p00', 'kl_0p10', 'kl_0p20']

def read(name):
    return json.loads((RESULTS / name).read_text())

def lines(name):
    return [json.loads(x) for x in (RESULTS / name).read_text().splitlines() if x.strip()]

def save(fig, name):
    fig.tight_layout()
    for suffix in ['png', 'pdf']:
        fig.savefig(FIGURES / f'{name}.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)

def table(name, rows):
    with (RESULTS / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False})
    summaries, training, comparison = {}, {}, []
    for name in NAMES:
        summary = read(f'{name}_eval_summary.json')
        run = read(f'{name}_summary.json')
        log = lines(f'{name}_training.jsonl')
        generations = lines(f'{name}_generations.jsonl')
        rewards = lines(f'{name}_rewards.jsonl')
        assert len(generations) == len(rewards) == summary['num_eval_prompts'] == 200
        assert len(log) == run['updates']
        summaries[name], training[name] = summary, log
        steps = sum(len(x['optimization_epochs']) for x in log)
        row = dict(summary)
        row.update({k: run[k] for k in ['updates', 'clip_epsilon', 'kl_beta',
                    'wall_seconds', 'peak_vram_gib', 'skipped_policy_steps',
                    'skipped_value_steps', 'policy_skip_fraction']})
        row['critic_skip_fraction'] = run['skipped_value_steps'] / steps
        row['mean_online_clip_fraction'] = float(np.mean([x['clip_fraction'] for x in log]))
        comparison.append(row)
    table('ppo_comparison.csv', comparison)
    cache = read('cached_clipping_summary.json')['conditions']
    table('cached_clipping_comparison.csv', cache)

    log = training['standard']
    updates = [x['update'] for x in log]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    fields = [('raw_reward_mean', 'Raw reward'),
              ('effective_reward_mean', 'Effective reward'),
              ('sampled_kl_token_mean', 'Sampled KL diagnostic'),
              ('entropy_token_mean', 'Token entropy'),
              ('policy_loss', 'Policy loss (epoch mean)'),
              ('value_loss', 'Critic MSE (epoch mean)'),
              ('clip_fraction', 'Online clip fraction'),
              ('response_length_mean', 'Response length (tokens)')]
    for ax, (key, title) in zip(axes.flat, fields):
        ax.plot(updates, [x[key] for x in log], marker='o', markersize=3)
        ax.set(title=title, xlabel='Update')
        ax.grid(alpha=.2)
    axes.flat[6].set_ylim(-.01, 1)
    axes.flat[7].set_ylim(0, 512)
    ax = axes.flat[8]
    for key, label in [('policy_gradient_norm', 'Policy'), ('value_gradient_norm', 'Critic')]:
        values = [np.mean([d[key] for d in x['optimization_epochs'] if d[key] is not None])
                  if any(d[key] is not None for d in x['optimization_epochs']) else np.nan
                  for x in log]
        ax.plot(updates, values, marker='o', markersize=3, label=label)
    ax.set(title='Gradient norms before clipping', xlabel='Update', yscale='log')
    ax.legend()
    ax.grid(alpha=.2)
    fig.suptitle('Standard PPO: 20 updates; 0/40 policy and 2/40 critic steps skipped', y=1.02)
    save(fig, 'task2_standard_training')

    metrics = [('reward_model_score_mean', 'Mean reward score'),
               ('sampled_kl_token_mean', 'Sampled KL diagnostic'),
               ('entropy_token_mean', 'Token entropy'),
               ('response_length_mean', 'Mean response length (tokens)'),
               ('generation_cap_rate', 'Generation cap rate')]
    for prefix, labels, title in [('clip', ['0.05', '0.20', '0.50'], 'Clipping epsilon'),
                                  ('kl', ['0.00', '0.10', '0.20'], 'KL beta')]:
        names = [n for n in NAMES if n.startswith(prefix + '_')]
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for ax, (key, label) in zip(axes.flat, metrics):
            ax.bar(labels, [summaries[n][key] for n in names], color='#3676a8')
            ax.set(title=label, xlabel=title)
            ax.grid(axis='y', alpha=.2)
            if 'kl_token' in key:
                ax.axhline(0, color='black', linewidth=.7)
                ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            if key == 'response_length_mean': ax.set_ylim(0, 768)
            if key == 'generation_cap_rate': ax.set_ylim(0, 1)
        ax = axes.flat[5]
        positions = np.arange(3)
        selected = [next(r for r in comparison if r['name'] == n) for n in names]
        ax.bar(positions - .18, [r['policy_skip_fraction'] for r in selected], .36, label='Policy')
        ax.bar(positions + .18, [r['critic_skip_fraction'] for r in selected], .36, label='Critic')
        ax.set(xticks=positions, xticklabels=labels, ylim=(0, 1),
               title='Fraction of optimization steps skipped', xlabel=title)
        ax.legend()
        fig.suptitle('8-update forks; 200 fixed evaluation prompts each', y=1.02)
        save(fig, f'task2_{prefix}_comparison')

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for ax, key, title in zip(axes,
        ['clip_fraction', 'surrogate_changed_token_fraction', 'clipped_surrogate'],
        ['Ratios outside clip interval', 'Clipped surrogate actually changes', 'Mean clipped surrogate']):
        ax.bar([str(x['epsilon']) for x in cache], [x[key] for x in cache], color='#3676a8')
        ax.set(title=title, xlabel='Clipping epsilon')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Fixed cached rollouts: 32 responses, 8,814 valid tokens', y=1.03)
    save(fig, 'task2_cached_clipping')
    print('Saved two CSV tables and four figures (PNG and PDF).')

if __name__ == '__main__':
    main()
