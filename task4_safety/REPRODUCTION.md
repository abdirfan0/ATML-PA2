# Task 4 reproduction

The pipeline compares the untouched SFT base policy with the standard final DPO,
PPO and GRPO adapters. It uses all 450 released XSTest prompts, one greedy
response per policy and prompt, batch size 4, a 256-token prompt budget and a
256-token response cap. The released categorical judge source is unchanged.
Judge confidence is audit-only. No safety results are used to select or train
the preceding adapters.

## Inspect saved evidence on CPU

Run `notebooks/04_Safety.ipynb`, or from the repository root:

```bash
python -m task4_safety.finalize --stage all
```

The finalizer checks source and dataset hashes, ordered IDs, response hashes,
class-specific rate denominators, category tables, fixed audit selection,
manual labels, confusion counts and disagreements. It creates two supplemental
tables, four PNG/PDF figure pairs and qualitative review candidates.
It does not replace or relabel student judgments. The final audit comprises
30 SAFE and 30 UNSAFE prompt IDs repeated across four policies: 240 responses.
Exact duplicate prompt-response pairs use consistent student-finalized labels.

The recorded GPU packages and adapter hashes remain in `run.json`.
CPU finalization does not compare installed CPU packages against GPU packages
and does not load or rehash adapter weight files. Downloaded course assets and
adapter outputs are needed for fresh generation, not for reviewing saved text.

## Fresh GPU run

Install `requirements.txt`. Obtain the released assets with
`python -m scripts.download_assets` and check them with
`python -m scripts.validate_assets`. Reproduce the required standard Task 1–3
adapters at the configured paths. In Colab, remove the incompatible optional
`torchao` package if present before installing the pinned dependencies.

Preserve previous results and use a fresh output directory:

```bash
python -m task4_safety.pipeline --config configs/feedback.yaml --stage all --batch-size 4 --output results/task4_safety_reproduction
```

This runs generation, fixed judging, audit-sheet creation, summarization and
validation. Fill every label in that directory's `manual_audit_blind.csv`
without first viewing AI labels or the audit key. Use the five released labels.
Do not edit IDs, prompts or responses. Then, on CPU:

```bash
python -c 'from pathlib import Path; from common.data import load_yaml; from task4_safety.pipeline import dataset,summarize; summarize(Path("results/task4_safety_reproduction"),dataset(load_yaml("configs/feedback.yaml")))'
python -m task4_safety.finalize --output results/task4_safety_reproduction --figures report/reproduction_figures --stage all
```

Matching GPU runs can resume. A changed source, configuration, asset hash or
package version is rejected by the GPU pipeline's metadata checks.

## Evidence limits

The fixed AI judge is an evaluation instrument, not ground truth. Raw labels,
including class-inconsistent labels, are retained. Class-specific rates need
not sum to one when those inconsistent labels exist. The audit-subset metrics
are explicitly separated from full-dataset metrics. Responses are evaluated
as shown, including truncation. Manual disagreements are retained rather than
treated automatically as either judge errors or student errors.

`qualitative_candidates.json` supplies raw four-policy responses for student
review. Potential harmful-compliance candidates require human assessment.
There are no manual UNSAFE_COMPLIANCE labels in the final 240-response audit,
and the AI's UNSAFE_COMPLIANCE labels in this run occur only on SAFE prompts.
Do not silently promote those labels to verified harmful-compliance examples
or claim that they establish its absence throughout the full dataset.
The student selects examples, identifies policy differences and judge errors,
and writes the report interpretation.
