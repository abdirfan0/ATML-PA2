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
