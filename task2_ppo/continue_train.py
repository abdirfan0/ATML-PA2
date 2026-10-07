from __future__ import annotations
import argparse
import json
import time
import random
import numpy as np
import torch
from torch.optim import AdamW
from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import set_seed
from common.metrics import masked_mean, sampled_kl
from common.models import (load_policy, load_reward_model, load_tokenizer,
    load_value_model, trainable_parameters, value_parameter_groups, reference_mode)
from task2_ppo.ppo import (compute_gae, shaped_rewards, ppo_policy_loss,
    value_mse_loss, normalize_advantages)
from task2_ppo.runtime import (sha, dump, policy_scores, critic_scores,
    normal_tensors, trainable_state, restore_trainable, cuda_required)


def run_ppo(config_path, output=None, updates=None, clip_epsilon=None,
            kl_beta=None, run_name='standard'):
    cuda_required()
    cfg = load_yaml(config_path)
    for key, value in [('updates', updates), ('clip_epsilon', clip_epsilon),
                       ('kl_beta', kl_beta)]:
        if value is not None:
            cfg[key] = value
    set_seed(int(cfg['seed']))
    root = repo_path(cfg['results_dir']); root.mkdir(parents=True, exist_ok=True)
    out = repo_path(output or cfg['output'])
    state_path = root / f'{run_name}_state.pt'
    meta_path = root / f'{run_name}_run.json'
    summary_path = root / f'{run_name}_summary.json'
    rows = read_jsonl(cfg['paths']['rl_prompt_train'])
    order = np.random.default_rng(int(cfg['seed'])).permutation(len(rows)).tolist()
    count = int(cfg['updates']) * int(cfg['prompts_per_update'])
    if count > len(order):
        raise ValueError('Requested budget exceeds fixed prompt pool.')
    selected = order[:count]
    signature = {
        'config': cfg, 'output': str(out), 'selected_indices': selected,
        'prompt_ids': [rows[i].get('prompt_id') for i in selected],
        'train_data_sha256': sha(cfg['paths']['rl_prompt_train']),
        'midpoint_policy_sha256': sha(repo_path(cfg['paths']['ppo_midpoint_policy']) / 'adapter_model.safetensors'),
        'reference': 'Frozen original base policy; LoRA disabled',
        'scoring': 'Raw response-token policy log probabilities',
        'entropy': 'Full categorical entropy of raw policy over response states',
        'rollout_seed_rule': 'config seed + update index',
        'initial_amp_scale': 1024,
        'code_sha256': {p: sha(p) for p in [
            'task2_ppo/continue_train.py', 'task2_ppo/ppo.py', 'task2_ppo/runtime.py']},
    }
    if meta_path.exists():
        if json.loads(meta_path.read_text())['signature'] != signature:
            raise RuntimeError('Existing run differs; use a new run name/output.')
        if summary_path.exists():
            print('Completed run exists:', summary_path, flush=True)
            return json.loads(summary_path.read_text())
    else:
        if out.exists() or state_path.exists():
            raise RuntimeError('Existing output without matching run metadata.')
        dump(meta_path, {'signature': signature, 'gpu': torch.cuda.get_device_name(0),
                         'torch_version': torch.__version__})

    tok = load_tokenizer(cfg['base_model'])
    policy = load_policy(cfg, adapter_path=cfg['paths']['ppo_midpoint_policy'], trainable=True)
    critic = load_value_model(cfg, cfg['paths']['ppo_midpoint_value'],
                              train_mode=cfg['value_train_mode'])
    reward, rtok = load_reward_model(cfg)
    for model in [policy, critic]:
        for p in trainable_parameters(model):
            p.data = p.data.float()
        # Disable dropout for old/new likelihood consistency; gradients still work.
        model.eval()
    pp = trainable_parameters(policy); vp = trainable_parameters(critic)
    popt = AdamW(pp, lr=float(cfg['policy_learning_rate']), weight_decay=0.0)
    vopt = AdamW(value_parameter_groups(critic, float(cfg['value_lora_learning_rate']),
                                       float(cfg['value_head_learning_rate'])), weight_decay=0.0)
    pscale = torch.amp.GradScaler('cuda', init_scale=1024)
    vscale = torch.amp.GradScaler('cuda', init_scale=1024)
    logs, elapsed_before, start_update = [], 0.0, 0
    if state_path.exists():
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        restore_trainable(policy, state['policy']); restore_trainable(critic, state['critic'])
        popt.load_state_dict(state['policy_optimizer']); vopt.load_state_dict(state['value_optimizer'])
        pscale.load_state_dict(state['policy_scaler']); vscale.load_state_dict(state['value_scaler'])
        logs = state['logs']; start_update = state['completed_updates']
        elapsed_before = state['elapsed_seconds']
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        del state
        print(f'Resuming after {start_update} completed updates.', flush=True)
    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    for update in range(start_update, int(cfg['updates'])):
        set_seed(int(cfg['seed']) + update)
        n = int(cfg['prompts_per_update'])
        indices = selected[update*n:(update+1)*n]
        prompts = [prompt_messages(rows[i]) for i in indices]
        policy.config.use_cache = True
        generated = normal_tensors(batch_generate(policy, tok, prompts,
            max_prompt_length=int(cfg['max_prompt_length']),
            max_new_tokens=int(cfg['max_response_length']), **cfg['generation']))
        policy.config.use_cache = False
        mask = generated['response_mask']
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            old, ent = policy_scores(policy, generated, entropy=True)
            with reference_mode(policy):
                ref, _ = policy_scores(policy, generated)
            values = critic_scores(critic, generated)
            raw_reward = score_reward_pairs(reward, rtok, prompts,
                generated['responses'], max_length=int(cfg['reward_max_length']))
            penalty = torch.tensor([not e for e in generated['terminated_with_eos']],
                                    device=mask.device, dtype=torch.float32)
            effective = raw_reward - float(cfg['missing_eos_penalty']) * penalty
            shaped = shaped_rewards(effective, old, ref, mask, cfg['kl_beta'])
            advantages, returns = compute_gae(shaped, values, mask,
                gamma=float(cfg['gamma']), lam=float(cfg['gae_lambda']))
            advantages = normalize_advantages(advantages, mask).detach()
            returns = returns.detach()
        diagnostics = []; skipped_policy = skipped_value = 0
        for epoch in range(int(cfg['ppo_epochs'])):
            popt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                new, _ = policy_scores(policy, generated)
                ploss, ratio, clip = ppo_policy_loss(new, old, advantages, mask,
                                                    eps=float(cfg['clip_epsilon']))
            if not torch.isfinite(ploss):
                raise RuntimeError(f'Non-finite policy loss at update {update}.')
            pscale.scale(ploss).backward(); pscale.unscale_(popt)
            pnorm = torch.nn.utils.clip_grad_norm_(pp, float(cfg['max_grad_norm']))
            scale = pscale.get_scale(); pscale.step(popt); pscale.update()
            skipped_policy += int(pscale.get_scale() < scale)
            del new
            vopt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                predicted = critic_scores(critic, generated)
                vloss = value_mse_loss(predicted, returns, mask)
            if not torch.isfinite(vloss):
                raise RuntimeError(f'Non-finite value loss at update {update}.')
            vscale.scale(float(cfg['value_coef']) * vloss).backward(); vscale.unscale_(vopt)
            vnorm = torch.nn.utils.clip_grad_norm_(vp, float(cfg['max_grad_norm']))
            scale = vscale.get_scale(); vscale.step(vopt); vscale.update()
            skipped_value += int(vscale.get_scale() < scale)
            diagnostics.append({'policy_loss': ploss.item(), 'value_loss': vloss.item(),
                'clip_fraction': clip.item(),
                'policy_gradient_norm': float(pnorm) if torch.isfinite(pnorm) else None,
                'value_gradient_norm': float(vnorm) if torch.isfinite(vnorm) else None})
            del predicted, ploss, vloss
        torch.cuda.synchronize()
        record = {'update': update+1, 'source_indices': indices,
            'prompt_ids': [rows[i].get('prompt_id') for i in indices],
            'raw_reward_mean': raw_reward.mean().item(),
            'effective_reward_mean': effective.mean().item(),
            'sampled_kl_token_mean': sampled_kl(old, ref, mask).item(),
            'entropy_token_mean': masked_mean(ent, mask).item(),
            'response_length_mean': mask.sum(-1).mean().item(),
            'generated_tokens': int(mask.sum()),
            'generation_cap_rate': float(np.mean(generated['truncated'])),
            'skipped_policy_steps': skipped_policy, 'skipped_value_steps': skipped_value,
            'optimization_epochs': diagnostics,
            'elapsed_seconds': elapsed_before + time.perf_counter()-began,
            'peak_vram_gib': torch.cuda.max_memory_allocated()/2**30,
            'rollouts': [{'source_index':i, 'prompt_messages':p, 'response':r,
                'response_tokens':generated['response_ids'][j,:generated['response_lengths'][j]].cpu().tolist()}
                for j,(i,p,r) in enumerate(zip(indices,prompts,generated['responses']))]}
        for key in ['policy_loss','value_loss','clip_fraction']:
            record[key] = float(np.mean([d[key] for d in diagnostics]))
        logs.append(record)
        state = {'completed_updates':update+1, 'logs':logs,
            'elapsed_seconds':record['elapsed_seconds'], 'policy':trainable_state(policy),
            'critic':trainable_state(critic), 'policy_optimizer':popt.state_dict(),
            'value_optimizer':vopt.state_dict(), 'policy_scaler':pscale.state_dict(),
            'value_scaler':vscale.state_dict(), 'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all(), 'python_rng':random.getstate(),
            'numpy_rng':np.random.get_state()}
        tmp = state_path.with_suffix('.tmp'); torch.save(state,tmp); tmp.replace(state_path)
        del state
        log_path = root/f'{run_name}_training.jsonl'
        log_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in logs))
        print(f"{run_name}: {update+1}/{cfg['updates']} reward={record['raw_reward_mean']:.3f} "
              f"KL={record['sampled_kl_token_mean']:.5f} length={record['response_length_mean']:.0f} "
              f"skipped={skipped_policy}/{skipped_value}",flush=True)
    out.mkdir(parents=True,exist_ok=True)
    policy.save_pretrained(out); tok.save_pretrained(out)
    critic.save_pretrained(out/'critic')
    summary = {'run_name':run_name,'updates':len(logs),'clip_epsilon':cfg['clip_epsilon'],
        'kl_beta':cfg['kl_beta'],'adapter_path':str(out),
        'generated_tokens':sum(r['generated_tokens'] for r in logs),
        'skipped_policy_steps':sum(r['skipped_policy_steps'] for r in logs),
        'skipped_value_steps':sum(r['skipped_value_steps'] for r in logs),
        'wall_seconds':elapsed_before+time.perf_counter()-began,
        'peak_vram_gib':max(r['peak_vram_gib'] for r in logs),
        'stability_statistic':'fraction of policy optimization steps skipped by AMP',
        'policy_skip_fraction':sum(r['skipped_policy_steps'] for r in logs)/(len(logs)*int(cfg['ppo_epochs']))}
    dump(summary_path,summary); print(json.dumps(summary,indent=2),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/ppo.yaml');p.add_argument('--output')
    p.add_argument('--updates',type=int);p.add_argument('--clip-epsilon',type=float)
    p.add_argument('--kl-beta',type=float);p.add_argument('--run-name',default='standard')
    a=p.parse_args();run_ppo(a.config,a.output,a.updates,a.clip_epsilon,a.kl_beta,a.run_name)

if __name__=='__main__':main()
