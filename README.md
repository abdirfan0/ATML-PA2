# ATML PA2 - LLM Post-Training

<!-- FINAL_STUDENT_SETUP -->

## Quick start

```bash
git clone https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining.git
cd ATML-PA2-LLM-PostTraining
python -m pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
```

The fixed datasets, cached diagnostics, and supplied
continuation checkpoints are downloaded from:

https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets

Pinned release revision:

`0b350481fb03f5525a35bcdec4131bd4fe487f98`

---
# ATML PA2 - LLM Post-Training

This is the **student starter repository** for ATML PA2. The released code is intentionally incomplete: Tasks 1-3 provide model/data loading, objective helpers, checkpoint restoration, and experiment entry points, but **you must implement the training loops and ablation orchestration yourself**. Each of Tasks 1-3 also contains one deliberate algorithmic defect in its core objective code; identifying and correcting these defects is part of validating your implementation.

Task 4 supplies the fixed AI safety judge and response-generation utilities, but you must write the evaluation/aggregation code. Task 5 supplies the exact RLVR verifier, the fixed pairwise AI judge used for RLAIF evaluation, and data/model loaders; you must implement the requested evaluation and analysis.

## 1. Clone and install

```bash
git clone https://github.com/COURSE_ORG/ATML-PA2-LLM-PostTraining.git
cd ATML-PA2-LLM-PostTraining
python -m pip install -r requirements.txt
```

## 2. Download the course assets

The large course-created checkpoints and fixed data are distributed as a GitHub Release asset rather than normal Git files. After cloning, run:

```bash
python -m scripts.download_assets
python -m scripts.validate_assets
```

If your instructor provides a direct asset URL separately, use:

```bash
python -m scripts.download_assets --url '<ASSET_URL>'
```

Public base/reward/judge models are downloaded from Hugging Face at runtime and are **not** included in the course asset archive.

The installer also materializes the fixed 100-example Task 5 transfer set from the official SVAMP challenge-set source if it is not already present. The tiny Task 1 word-limit prompt set is tracked directly in this repository.

## 3. Environment check

```bash
python -m scripts.check_environment
```

Run commands from the repository root. The reference environment used to prepare the release pins Transformers 4.57.1, TRL 0.27.2, PEFT 0.17.1, and Tokenizers 0.22.1.

## 4. Supplied course checkpoints

After `download_assets`, these directories should exist:

```text
checkpoints/ppo_midpoint_policy/
checkpoints/ppo_midpoint_value/
checkpoints/grpo_midpoint_policy/
checkpoints/rlvr_policy/
checkpoints/rlaif_policy/
```

PPO and GRPO begin from the supplied continuation checkpoints. RLVR and RLAIF are supplied frozen evaluation policies; students do not retrain them.

The PPO value checkpoint is intentionally released as the exact staff midpoint state, including its imperfect held-out value calibration. Treat critic behavior as an analysis variable rather than assuming a perfect baseline, and start every PPO fork from the identical supplied policy/value state. The default continuation generation cap is 512 tokens for feasibility; frozen evaluation uses the larger cap specified in `configs/ppo.yaml`.

## 5. Task entry points

### Task 1 - DPO

```bash
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.ablate_beta --config configs/dpo.yaml
python -m task1_dpo.analyze_length --config configs/dpo.yaml
```

### Task 2 - PPO

```bash
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml
python -m task2_ppo.ablate_kl --config configs/ppo.yaml
```

### Task 3 - GRPO

```bash
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml
python -m task3_grpo.compare_normalization --config configs/grpo.yaml
```

### Task 4 - Safety calibration

The judge loader/parser are supplied. You must implement the requested generation aggregation and evaluation.

```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml
python -m task4_safety.judge_responses --config configs/feedback.yaml
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml
python -m task4_safety.evaluate_safety --config configs/feedback.yaml
```

### Task 5 - RLVR vs RLAIF

The exact verifier and pairwise AI judge are supplied; you implement the evaluation/analysis.

```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.score_perturbations --config configs/feedback.yaml
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
```

## 6. Reproducibility rules

- Do not alter course-provided data, cached rollouts, or supplied checkpoints.
- Start every short fork from the **same supplied midpoint checkpoint**.
- Keep prompt IDs, generated-token/update budgets, seed, and evaluation procedure matched across ablations.
- Commit your code, configs, small JSON/CSV logs, and figures. Do not commit downloaded checkpoints, raw course assets, or model caches.
- Record peak VRAM and wall-clock time for the standard PPO and GRPO continuations.

See the assignment manual for the required experiments, metrics, and report questions.

## Task 1: DPO experiments

Implemented experiments:
- Standard DPO: one epoch on 1,446 eligible preference pairs.
- Beta study: 0.03, 0.10, and 0.30, each on the same 600 eligible pairs.
- Length study: training on 1,442 eligible length-control pairs,
  followed by length-stratified and word-limit evaluation.

### Setup

```bash
git clone https://github.com/abdirfan0/ATML-PA2.git
cd ATML-PA2
python -m pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
```

### Inspect saved evidence on CPU

Results and logs are in results/task1_dpo/.
The clean notebook is notebooks/01_DPO.ipynb.
Its Drive/path cell must be adapted to your local checkout or Colab location.

```bash
python -m task1_dpo.plot_results
```

Figures are saved as PNG and PDF in report/figures/.

### Reproduce experiments on a CUDA GPU

For a fresh clone, first preserve the published results because training
rejects existing run logs. Existing trained outputs must also be moved
aside if reproducing in a previously used checkout.

```bash
mv results/task1_dpo results/task1_dpo_published

python -m task1_dpo.train --config configs/dpo.yaml --run-name standard --output outputs/task1_dpo/standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard

python -m task1_dpo.ablate_beta --config configs/dpo.yaml --stage train
python -m task1_dpo.ablate_beta --config configs/dpo.yaml --stage evaluate

python -m task1_dpo.analyze_length --config configs/dpo.yaml --stage all
python -m task1_dpo.plot_results
```

Completed evaluation records are reused only when settings and adapter
signatures match. Exact outputs can vary across GPU hardware.

### Evidence and conventions

Run metadata records hyperparameters, original row indices, filtering,
dataset hashes, code hashes, and training hardware. Standard and beta
forks use different training budgets. AMP-skipped updates are recorded.

Response length includes generated EOS tokens and excludes padding.
Length dispersion is reported using population standard deviation.
Sampled KL is the raw response-token log-probability difference from the
frozen reference, aggregated with response-token weighting. With
temperature/top-p sampling, it is a diagnostic rather than an unbiased
estimate of full-policy KL.

Qualitative examples displayed in the notebook:
- Evaluation row 41: standard versus beta 0.03.
- Word-limit prompt 5: standard versus length-balanced.

### Attribution

Based on the course starter code:
https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining

Implementation and debugging used ChatGPT/Codex coding assistance.
The starter's existing attribution and dependency information are retained.

## Task 2: PPO experiments

Task 2 is implemented in `task2_ppo/`. The standard continuation uses
20 updates; each clipping or KL fork uses 8 updates from the same supplied
midpoint. Every policy is evaluated on the same 200 prompts with fixed
decoding settings and per-prompt seeds.

### Inspect saved results on CPU

```bash
python -m task2_ppo.validate_objective
python -m task2_ppo.plot_results
```

`notebooks/02_PPO.ipynb` displays tables, figures, and selected responses.
GPU execution flags are disabled by default. The qualitative cell saves
`results/task2_ppo/qualitative_examples.json`.

### Reproduce GPU experiments

First install `requirements.txt`, download the pinned course assets with
`python -m scripts.download_assets`, and run `python -m scripts.validate_assets`.
Use a fresh checkout. On an existing checkout, preserve both the previous
`results/task2_ppo/` and `outputs/task2_ppo/` directories before starting
a fresh reproduction. Matching incomplete runs resume automatically;
runs with mismatched metadata are rejected.

```bash
python -m task2_ppo.validate_objective
python -u -m task2_ppo.analyze_clipping --config configs/ppo.yaml --stage cache
python -u -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard --output outputs/task2_ppo/standard
python -u -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -u -m task2_ppo.analyze_clipping --config configs/ppo.yaml --stage train
python -u -m task2_ppo.ablate_kl --config configs/ppo.yaml --stage train
python -u -m task2_ppo.analyze_clipping --config configs/ppo.yaml --stage evaluate
python -u -m task2_ppo.ablate_kl --config configs/ppo.yaml --stage evaluate
python -m task2_ppo.plot_results
```

### Recorded limitations and conventions

- Online clip fractions were zero in all recorded optimization epochs.
  Cached-rollout clipping diagnostics are reported separately.
- `clip_0p20` and `kl_0p10` use identical settings but produced different
  adapters and mean rewards (1.5029 and 1.5508). The source of this
  run-to-run variation has not been established; fixed seeds did not
  provide bitwise reproducibility. Small differences between conditions
  should not be treated as definitive parameter effects.
- AMP skipped two critic steps per run: 2/40 for standard PPO and 2/16
  for each short fork. No policy steps were skipped.
- Sampled KL uses raw response-token log-probability differences under
  temperature/top-p decoding. It is a diagnostic, not an unbiased
  full-policy KL estimate; small negative values are possible.
- Response lengths include generated EOS tokens and exclude padding.
  Length standard deviations use the population convention. Evaluation
  uses a 768-token cap; training uses a 512-token cap.
- Gradient norms are recorded before gradient clipping. Training wall time
  excludes initial model loading and held-out evaluation; peak VRAM is
  PyTorch's peak allocated CUDA memory.
- Evaluation processes one prompt at a time and may take substantially
  longer than the short training continuations.
- Saved metadata records configurations, prompt IDs, indices, and hashes.
  The uploaded-artifact checks are recorded in
  `results/task2_ppo/artifact_validation.json`.

The pipeline used coding assistance from ChatGPT/Codex.

<!-- TASK3_REPRODUCTION -->

## Task 3 reproduction

Run commands from the repository root. The clean notebook is `notebooks/03_GRPO.ipynb`; setup and GPU execution are disabled there by default. The Python modules implement the experiment pipeline.

For a fresh environment and course assets:

```bash
python -m pip uninstall -y torchao
python -m pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
python -m task3_grpo.validate_objective
python -m task3_grpo.validate_matched --tests-only
```

Standard online GRPO and the supplied-cache group-size diagnostic:

```bash
python -u -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
python -u -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard --batch-size 4
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml
```

Original online normalization forks (equal maximum allowances; actual generated-token counts differ):

```bash
python -u -m task3_grpo.compare_normalization --config configs/grpo.yaml --stage train
python -u -m task3_grpo.compare_normalization --config configs/grpo.yaml --stage evaluate --batch-size 4
python -m task3_grpo.compare_normalization --config configs/grpo.yaml --stage summarize
```

Supplementary exact-token normalization intervention:

```bash
python -u -m task3_grpo.matched_normalization --config configs/grpo.yaml --stage all
python -u -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/matched_canonical --name matched_canonical --batch-size 4
python -u -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/matched_dr_grpo --name matched_dr_grpo --batch-size 4
```

This intervention generates eight batches once using the frozen supplied midpoint, then initializes each fork from that midpoint and trains on the same batches. Rewards, advantages, response lengths, truncation masks and behavior log probabilities are shared. The only fork setting changed is sequence normalization. Later batches are not sampled from the updated forks; this is fixed-rollout training, separately labelled from online GRPO. The exact consumed rollout-token and active-loss-token totals are checked. Shared collection cost must be counted once, separately from fork optimization and evaluation. Fork optimization timings exclude model loading and shared rollout collection, whereas standard continuation timing includes its online rollout work.

CPU validation and figures:

```bash
python -m task3_grpo.validate_artifacts --config configs/grpo.yaml
python -m task3_grpo.finalize_matched --config configs/grpo.yaml
python -m task3_grpo.plot_results --config configs/grpo.yaml
```

The source-hash, prompt-ID, token-budget and row-level metric checks write JSON validation records in `results/task3_grpo/`. Figures are PNG/PDF pairs in `report/figures/`; CSV tables and full qualitative candidates remain in the results directory. Candidate selection uses the three largest and smallest reward differences and is not a representative sample or a correctness annotation.

Raw sampled policy/reference log-probability differences under temperature/top-p decoding are diagnostics, not unbiased full-policy KL estimates. Gradient-allocation statistics concern selected-token log probabilities, excluding the KL term; they are not model-parameter gradient norms. Masked batches perform no optimization step. The standard run retains 20 rollout updates, including two batches with no active-loss tokens. Saved state files contain optimizer and adapter state for resumption; do not commit `results/task3_grpo/*_state.pt` or temporary files.

Matching completed runs are reused and partial runs resume. Changed metadata or training source hashes are rejected. For an independent reproduction, use a fresh checkout or preserve prior results and output directories first. All final evaluations use ordered fixed prompt IDs, batch size 4, the same decoding settings and the same batch-seed rule. The batch-size throughput benchmark is optional:

```bash
python -u -m task3_grpo.benchmark --config configs/grpo.yaml --limit 8
```

The course starter supplies model/data loading and objective scaffolding. Task 3 continuation, grouping corrections, validation, orchestration and analysis scripts were developed with ChatGPT assistance. The student is responsible for interpreting results and writing the report.
