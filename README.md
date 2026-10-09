# ATML PA2: LLM Post-Training

Student implementation and saved evidence for Tasks 1–5. Python modules implement
the experiments; clean notebooks inspect saved results on CPU and provide optional
GPU reproduction. GPU execution is disabled by default in every clean notebook.

## Setup

```bash
git clone https://github.com/abdirfan0/ATML-PA2.git
cd ATML-PA2
python -m pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
python -m scripts.check_environment
```

Run commands from the repository root. Assets are downloaded from the Hugging
Face dataset `AbDu11aHHH/ATML-PA2-assets`, pinned to revision
`0b350481fb03f5525a35bcdec4131bd4fe487f98`. The downloader does not use GitHub
Release archives or accept `--url`. Raw datasets, caches, and weights are ignored
by Git. Public base/reward/judge models are loaded at GPU inference time.

Core requirements pin Transformers 4.57.1, Tokenizers 0.22.1, TRL 0.27.2, and
PEFT 0.17.1. Torch and several other dependencies have version ranges, so this
is not a complete environment lock. Recorded versions/hardware are in run
metadata; exact responses across versions or hardware are not guaranteed.
If an incompatible optional `torchao` package blocks PEFT in Colab, uninstall
it before installing the requirements:

```bash
python -m pip uninstall -y torchao
python -m pip install -r requirements.txt
```

Use Colab or an existing Jupyter/IPython environment for notebooks. Their path
cells detect the checkout or mount the original Drive location; adapt the Drive
fallback if necessary. Install requirements and download assets before CPU
diagnostics that read course data or caches. Saved-text inspection needs no GPU.

## Independent GPU reproduction

Use an A100 for the GPU pipelines. A fresh clone contains published result logs
but no trained student adapters or optimizer states. Preserve published Task
1–3 logs before independent training: otherwise completed-run reuse or metadata
rejection can prevent a fresh run. On a fresh clone, run this once:

```bash
mkdir ../ATML-PA2-published-evidence
mv results/task1_dpo results/task2_ppo results/task3_grpo ../ATML-PA2-published-evidence/
```

Keep that backup. In a previously used checkout, also preserve the existing
`outputs/task1_dpo`, `outputs/task2_ppo`, and `outputs/task3_grpo` directories.
Matching incomplete runs can resume when their states are available. Task 4–5
commands below use separate reproduction directories. Do not use their evaluation
labels to tune earlier policy training.

Fixed prompt IDs, selected indices, budgets, decoding settings, source/data
hashes and applicable hardware details are in run metadata. Tables and logs are
under `results/`; PNG/PDF figure pairs are under `report/figures/`. All five clean
notebooks are under `notebooks/` and inspect that evidence.

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
Preserve published results even in a fresh clone, as described above. Also preserve the previous
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

Matching completed runs are reused and partial runs resume. Changed metadata or training source hashes are rejected. For an independent reproduction, preserve prior results even in a fresh clone, and preserve existing output directories first. All final evaluations use ordered fixed prompt IDs, batch size 4, the same decoding settings and the same batch-seed rule. The batch-size throughput benchmark is optional:

```bash
python -u -m task3_grpo.benchmark --config configs/grpo.yaml --limit 8
```

The course starter supplies model/data loading and objective scaffolding. Task 3 continuation, grouping corrections, validation, orchestration and analysis scripts were developed with ChatGPT assistance. The student is responsible for interpreting results and writing the report.

## Task 4: Safety calibration

Inspect saved results with `notebooks/04_Safety.ipynb`, or run:

```bash
python -m task4_safety.finalize --config configs/feedback.yaml
```

For fresh GPU generation, first reproduce the standard Task 1–3 adapters above.
The pipeline uses all 450 XSTest prompts, fixed categorical judging, greedy
decoding, batch size 4, and a 256-token response cap:

```bash
python -m task4_safety.validate_pipeline
python -u -m task4_safety.pipeline --config configs/feedback.yaml --stage all --batch-size 4 --output results/task4_safety_reproduction
```

Label all 240 items in `results/task4_safety_reproduction/manual_audit_blind.csv`
before viewing AI labels or the audit key. Keep IDs and full text unchanged.
The fixed audit covers 30 SAFE and 30 UNSAFE prompt IDs across four policies.
After labeling, recompute summaries and figures on CPU:

```bash
python -c 'from pathlib import Path; from common.data import load_yaml; from task4_safety.pipeline import dataset,summarize; summarize(Path("results/task4_safety_reproduction"),dataset(load_yaml("configs/feedback.yaml")))'
python -m task4_safety.finalize --config configs/feedback.yaml --output results/task4_safety_reproduction --figures report/reproduction_figures
```

Class-inconsistent AI labels remain visible. Full-set and audit-subset metrics
use separate class-specific denominators. Manual disagreements are retained;
qualitative safety candidates require student review. See
[Task 4 details](task4_safety/REPRODUCTION.md).

## Task 5: Feedback-source evaluation

Inspect saved results with `notebooks/05_Feedback.ipynb`, or run:

```bash
python -m task5_feedback.finalize --config configs/feedback.yaml
```

Fresh GPU reproduction evaluates untouched SFT and two supplied frozen adapters
on 300 GSM8K and 100 SVAMP prompts; no training is performed. Generation is
greedy with batch size 16 and a 512-token response cap. All 100 controlled
diagnostic responses and 200 diagnostic pairs are scored.

```bash
python -m task5_feedback.validate_pipeline
python -u -m task5_feedback.pipeline --config configs/feedback.yaml --stage all --batch-size 16 --output results/task5_feedback_reproduction
python -m task5_feedback.finalize --config configs/feedback.yaml --output results/task5_feedback_reproduction --figures report/reproduction_figures
```

The unchanged verifier uses the last designated `####` number. Terminal format
is measured separately. The fixed pairwise judge uses one hash-selected A/B
orientation, maps unparsed outputs to TIE, and does not retain raw judge text.
Decisive judgments on identical responses are counted in `pair_diagnostics.csv`.
Overall verifier–judge agreement includes shared ties; decisive-subset agreement
includes its denominator. Transfer differences compare different prompt
distributions. See [Task 5 details](task5_feedback/REPRODUCTION.md).

The legacy Task 4/5 starter helper modules are retained for supplied loaders and
scorers. Use the implemented `pipeline` and `finalize` commands above rather
than the starter TODO entry points.

## Task 6 and submission

Task 6 synthesizes the existing evidence; no additional training is required.
The student writes the report and its interpretation under the manual's AI-use
policy. Every reported value should trace to a saved file and command. Freeze
the submitted code state with a meaningful commit and provide an accessible
repository link. Materially reused course code and ChatGPT/Codex coding
assistance are attributed above; the PDF report language and analysis must be
the student's own.
