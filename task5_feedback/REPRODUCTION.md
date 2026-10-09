# Task 5 reproduction

This task evaluates three fixed policies; it does not train new policies. SFT is
the untouched Qwen2.5-1.5B-Instruct model. RLVR and RLAIF use the released adapters
in `checkpoints/rlvr_policy` and `checkpoints/rlaif_policy`.

## Inspect saved evidence on CPU

From the repository root, install the requirements needed for CPU inspection
(NumPy, pandas, PyYAML, and matplotlib) and download/validate the released datasets
using the repository asset scripts. Then run:

```bash
python -m task5_feedback.finalize --config configs/feedback.yaml
```

This checks saved metadata, source/dataset hashes, prompt order, response hashes,
verifier rewards, judge cache bindings, and all seven CSV/JSON tables. It writes
`final_artifact_validation.json`, `pair_diagnostics.csv`, raw qualitative
candidates, and four PNG/PDF figure pairs. It requires no GPU or adapter weights;
adapter hashes are checked for presence in the original run metadata rather than
recomputed without weights. Original responses, pair labels, and run metadata
are preserved. Use `--stage validate` to omit plots and supplements.

`notebooks/05_Feedback.ipynb` provides the same CPU inspection workflow. Its GPU
cell is disabled by default.

## Reproduce inference on an A100

Use a fresh runtime, install the repository's pinned requirements, and run:

```bash
python -m scripts.download_assets
python -m scripts.validate_assets
python -m task5_feedback.validate_pipeline
python -m task5_feedback.pipeline --config configs/feedback.yaml --stage all --batch-size 16 --output results/task5_feedback_reproduction
python -m task5_feedback.finalize --config configs/feedback.yaml --output results/task5_feedback_reproduction --figures report/reproduction_figures
```

If an incompatible optional `torchao` installation blocks PEFT, uninstall
`torchao` before reinstalling the pinned requirements, as in the working notebook.
Inference uses greedy decoding, the released message lists unchanged, a
512-token prompt budget, a 512-token response cap, and batch size 16 for all
policies. Batch size can affect numerical generation behavior, so retain it for
matched comparisons. Exact responses across hardware or dependency versions are
not guaranteed. Preserve existing evidence before generating new runs; metadata
changes are rejected for an existing run directory.

## Evidence and metric definitions

- GSM8K: 300 released prompts; SVAMP: 100 released prompts. All three policies
  produce responses on every prompt, totaling 1,200 responses.
- Policy comparison: all three unordered policy pairs per prompt, totaling
  1,200 comparisons. A win counts as 1, a tie as 0.5, and a loss as 0. The
  SFT-against-SFT score of 0.5 is a defined baseline, not an additional judged pair.
- Diagnostics: 20 released problems with five validated variants each, totaling
  100 responses and 200 unordered comparisons. Clean-reference comparisons use
  20 pairs for each perturbation and each feedback mechanism.
- `S_reason` is clean-preferred rate for corrupted reasoning with a correct
  final answer. `S_outcome` is clean-preferred rate for the wrong-final variant.
  Tie and wrong-preference rates are reported separately. Filler has unchanged
  mathematical content, so its better/wrong fields are left unavailable.
- The unchanged verifier extracts the last designated `####` number. Terminal
  standalone format compliance is a separate measurement. Failure types record
  missing final fields, wrong final numbers, correct nonterminal fields, and
  correct terminal fields; they do not establish reasoning correctness.
- Transfer differences are GSM8K minus SVAMP, with the sign preserved. These
  compare different datasets and do not isolate causal effects of feedback.

## Fixed judge limitations

The course pairwise judge, rubric, parser, and hash-selected A/B orientation are
unchanged. One ordering is evaluated per pair. Unparsed outputs are mapped to
`TIE`, and raw judge text is not retained, so explicit ties and parser failures
cannot be separated from these artifacts. `pair_diagnostics.csv` also counts
identical response pairs and decisive AI preferences on identical text.
Overall verifier–judge agreement includes shared ties; the verifier-decisive
subset is reported with its denominator. Neither pairwise scores nor exact
verifier rewards assess all aspects of reasoning quality.

Qualitative candidates contain released diagnostic text and recorded labels.
Select and explain examples in your own report, as required by the manual.
