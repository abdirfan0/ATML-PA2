# Course data

Install fixed datasets, caches, checkpoints, and manifests with:

```bash
python -m scripts.download_assets
python -m scripts.validate_assets
```

The downloader uses the pinned Hugging Face course-asset revision specified in
`configs/assets.yaml` and `scripts/download_assets.py`.

Only `word_limit_prompts.jsonl` (10 Task 1 prompts) is tracked directly in this
repository. `math_transfer_eval.jsonl` (the 100-example Task 5 SVAMP subset) is
downloaded unchanged with the other course assets and is ignored by Git.
Fixed source indices and dataset hashes are retained in run metadata.

Do not edit course data or use safety/math evaluation results to tune earlier
training. Save generated evidence under `results/` and adapters under the ignored
`outputs/` directory.
