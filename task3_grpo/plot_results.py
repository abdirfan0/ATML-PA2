"""Generate Task 3 figures, comparison tables and qualitative candidates on CPU."""
import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common.data import load_yaml, repo_path

def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def table(path,records):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records)

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/grpo.yaml');a=p.parse_args()
 cfg=load_yaml(a.config);r=repo_path(cfg['results_dir']);figdir=repo_path('report/figures');figdir.mkdir(parents=True,exist_ok=True)
 def save(fig,name):
  fig.tight_layout();fig.savefig(figdir/f'{name}.png',dpi=180);fig.savefig(figdir/f'{name}.pdf');plt.close(fig)
 records=[]
 for name in ['standard','canonical','dr_grpo']:
  es=read(r/f'{name}_eval_summary.json');s=read(r/f'{name}_summary.json')
  records.append({**es,**{k:s[k] for k in ['updates','generated_tokens','active_loss_tokens','wall_seconds','peak_vram_gib','no_active_token_updates','skipped_policy_steps']}})
 table(r/'grpo_comparison.csv',records)
 logs=rows(r/'standard_training.jsonl');fig,axs=plt.subplots(3,3,figsize=(12,9));x=[l['update'] for l in logs]
 fields=[('raw_reward_mean','Raw reward'),('sampled_kl_token_mean','Raw sampled log-probability difference'),('group_reward_std_mean','Within-group reward standard deviation'),('uninformative_group_fraction','Uninformative-group fraction'),('policy_loss','Policy loss'),('entropy_token_mean','Full-policy token entropy'),('response_length_mean','Response length (tokens)'),('generation_cap_rate','Generation-cap fraction')]
 for ax,(key,title) in zip(axs.flat,fields):
  ax.plot(x,[l[key] if l[key] is not None else np.nan for l in logs],marker='.',linewidth=1);ax.set_title(title,fontsize=10);ax.set_xlabel('Update');ax.grid(alpha=.2)
 axs.flat[8].plot(x,[l['optimization_epochs'][0]['policy_gradient_norm'] if l['optimization_epochs'] else np.nan for l in logs],marker='.');axs.flat[8].set_title('Policy gradient norm before clipping');axs.flat[8].set_xlabel('Update')
 save(fig,'task3_standard_training')
 groups=read(r/'group_size_summary.json');conditions=groups['conditions'];out=[]
 for c in conditions:
  for bin_name,data in [('all',c),*c['difficulty_bins'].items()]:
   out.append({'K':c['K'],'difficulty_bin':bin_name,**{k:data[k] for k in ['num_groups','num_completions','informative_group_rate','mean_group_reward_std','group_relative_signal_variance']}})
 table(r/'group_size_comparison.csv',out)
 fig,axs=plt.subplots(1,3,figsize=(12,3.5))
 for ax,key,title in zip(axs,['informative_group_rate','mean_group_reward_std','group_relative_signal_variance'],['Informative-group rate','Mean reward standard deviation','Relative-advantage variance']):
  for label in ['all','lower_mean_reward','higher_mean_reward']:
   d=[o for o in out if o['difficulty_bin']==label];ax.plot([o['K'] for o in d],[o[key] for o in d],marker='o',label=label)
  ax.set_title(title);ax.set_xlabel('K');ax.set_xticks([2,4,8]);ax.grid(alpha=.2)
 axs[0].legend(fontsize=7);save(fig,'task3_group_size')
 comparison=records[1:];fig,axs=plt.subplots(1,3,figsize=(10,3.5))
 for ax,key,title in zip(axs,['reward_model_score_mean','sampled_kl_token_mean','response_length_mean'],['Held-out reward','Raw sampled log-probability difference','Mean response length (tokens)']):
  ax.bar(['Canonical','Dr. GRPO'],[v[key] for v in comparison],color=['#4477aa','#ee7733']);ax.set_title(title,fontsize=9)
 save(fig,'task3_normalization_comparison')
 validation=read(r/'artifact_validation.json');bins=validation['fixed_rollout_normalization']['length_bins'];fig,axs=plt.subplots(1,2,figsize=(9,3.5));labels=list(bins);indices=np.arange(len(labels))
 for ax,key,title in zip(axs,['mean_abs_token_derivative','mean_derivative_l1'],['Mean absolute token derivative','Mean total derivative L1 mass']):
  for i,kind in enumerate(['canonical','dr_grpo']):ax.bar(indices+(i-.5)*.34,[bins[b][kind][key] for b in labels],width=.34,label=kind)
  ax.set_xticks(indices,['≤128 tokens','>128 tokens']);ax.set_title(title);ax.legend(fontsize=8)
 fig.suptitle('Same saved rollouts; policy surrogate derivatives at ratio=1, excluding KL',fontsize=10);save(fig,'task3_fixed_rollout_normalization')
 generations={n:{g['row_index']:g for g in rows(r/f'{n}_generations.jsonl')} for n in ['canonical','dr_grpo']}
 rewards={n:{g['row_index']:g['reward'] for g in rows(r/f'{n}_rewards.jsonl')} for n in generations}
 diffs=sorted(range(200),key=lambda i:rewards['dr_grpo'][i]-rewards['canonical'][i]);ids=list(dict.fromkeys(diffs[:3]+diffs[-3:]))
 examples=[{'row_index':i,'prompt_id':generations['canonical'][i]['prompt_id'],'prompt_messages':generations['canonical'][i]['prompt_messages'],**{n:{'response':generations[n][i]['response'],'reward':rewards[n][i],'response_length':generations[n][i]['response_length']} for n in generations}} for i in ids]
 (r/'qualitative_candidates.json').write_text(json.dumps({'selection_rule':'Three smallest and three largest Dr. GRPO minus canonical reward changes; candidates for human review, not representative quality labels.','examples':examples},indent=2,ensure_ascii=False)+'\n')
 matched=read(r/'matched_normalization_validation.json')
 matched_records=[]
 for name,condition in matched['conditions'].items():
  training=condition['training']
  matched_records.append({**condition['evaluation'],'protocol':'fixed_rollout',
      **{k:training[k] for k in ['updates','generated_token_budget','consumed_rollout_tokens','active_loss_tokens','wall_seconds','peak_vram_gib','skipped_steps']}})
 table(r/'matched_normalization_comparison.csv',matched_records)
 fig,axs=plt.subplots(1,3,figsize=(10,3.5))
 for ax,key,title in zip(axs,['reward_model_score_mean','sampled_kl_token_mean','response_length_mean'],['Held-out reward','Raw sampled log-probability difference','Mean response length (tokens)']):
  ax.bar(['Canonical','Dr. GRPO'],[v[key] for v in matched_records],color=['#4477aa','#ee7733']);ax.set_title(title,fontsize=9)
 fig.suptitle(f"Fixed-rollout forks: identical {matched['exact_generated_token_budget_per_condition']:,}-token training pool",fontsize=10)
 save(fig,'task3_matched_normalization_comparison')
 fig,axs=plt.subplots(1,2,figsize=(9,3.5));labels=['short_le128','long_gt128'];indices=np.arange(2)
 gradient_rows=[]
 for name,c in matched['conditions'].items():
  for label,data in c['length_conditioned_actual_ratio_statistics'].items():gradient_rows.append({'name':name,'length_bin':label,**data})
 table(r/'matched_gradient_comparison.csv',gradient_rows)
 for ax,key,title in zip(axs,['mean_abs_policy_logp_derivative','policy_logp_derivative_l1'],['Mean absolute token derivative','Mean total derivative L1 mass']):
  for i,(name,c) in enumerate(matched['conditions'].items()):
   ax.bar(indices+(i-.5)*.34,[c['length_conditioned_actual_ratio_statistics'][b][key] for b in labels],width=.34,label=name.replace('matched_',''))
  ax.set_xticks(indices,['≤128 tokens','>128 tokens']);ax.set_title(title);ax.legend(fontsize=8)
 fig.suptitle('Actual-ratio policy surrogate derivatives, excluding KL; same rollout pool',fontsize=10)
 save(fig,'task3_matched_gradient_allocation')
 names=['matched_canonical','matched_dr_grpo']
 gg={n:{g['row_index']:g for g in rows(r/f'{n}_generations.jsonl')} for n in names}
 rr={n:{g['row_index']:g['reward'] for g in rows(r/f'{n}_rewards.jsonl')} for n in names}
 ordered=sorted(range(200),key=lambda i:rr[names[1]][i]-rr[names[0]][i]);ids=list(dict.fromkeys(ordered[:3]+ordered[-3:]))
 examples=[{'row_index':i,'prompt_id':gg[names[0]][i]['prompt_id'],'prompt_messages':gg[names[0]][i]['prompt_messages'],**{n:{'response':gg[n][i]['response'],'reward':rr[n][i],'response_length':gg[n][i]['response_length'],'hit_generation_cap':gg[n][i]['hit_generation_cap']} for n in names}} for i in ids]
 (r/'matched_qualitative_candidates.json').write_text(json.dumps({'selection_rule':'Three smallest and three largest matched Dr. GRPO minus matched canonical reward changes; human-review candidates, not representative quality labels.','examples':examples},indent=2,ensure_ascii=False)+'\n')
 print('Saved six PNG/PDF figure pairs, four CSV tables, and original/matched qualitative candidates.')

if __name__=='__main__':main()
