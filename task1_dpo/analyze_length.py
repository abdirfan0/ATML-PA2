from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from common.data import (
    load_yaml, repo_path, read_jsonl, prompt_messages,
)
from common.dpo_preprocessing import load_dpo_rows
from common.generation import batch_generate
from common.logging_utils import set_seed, save_json
from common.metrics import word_count, word_limit_compliance
from common.models import load_policy, load_tokenizer, reference_mode
from task1_dpo.train import make_collate, sequence_logp


def records(path):
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text().splitlines()
        if line.strip()
    ]


def append(path, record):
    with path.open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def evaluate_length(cfg):
    directory = repo_path(cfg["results_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(cfg["base_model"])

    rows, filtering = load_dpo_rows(
        cfg["paths"]["dpo_length_eval"],
        tokenizer,
        int(cfg["max_sequence_length"]),
    )
    indices = filtering["selected_indices"]
    word_prompts = read_jsonl(cfg["paths"]["word_limit_prompts"])
    collate = make_collate(tokenizer, int(cfg["max_sequence_length"]))

    combined = {}
    policies = {
        "standard": cfg["standard_output"],
        "length_balanced": cfg["length_output"],
    }

    for name, adapter in policies.items():
        prefix = directory / f"length_study_{name}"
        preference_path = Path(f"{prefix}_preferences.jsonl")
        word_path = Path(f"{prefix}_word_limits.jsonl")
        metadata_path = Path(f"{prefix}_run.json")

        metadata = {
            "adapter": str(repo_path(adapter)),
            "adapter_sha256": hashlib.sha256(
                (repo_path(adapter) / "adapter_model.safetensors").read_bytes()
            ).hexdigest(),
            "config": cfg,
            "eval_indices": indices,
            "eval_sha256": hashlib.sha256(
                repo_path(cfg["paths"]["dpo_length_eval"]).read_bytes()
            ).hexdigest(),
            "word_prompts_sha256": hashlib.sha256(
                repo_path(cfg["paths"]["word_limit_prompts"]).read_bytes()
            ).hexdigest(),
            "word_prompt_seed_rule": "config seed + word-prompt index",
        }

        if metadata_path.exists():
            if json.loads(metadata_path.read_text()) != metadata:
                raise RuntimeError(f"Existing settings differ for {name}.")
        else:
            save_json(metadata_path, metadata)

        model = load_policy(cfg, adapter_path=adapter, trainable=False)
        model.config.use_cache = True

        done = {r["row_index"] for r in records(preference_path)}
        pending = [
            (i, row) for i, row in zip(indices, rows) if i not in done
        ]

        for offset in tqdm(
            range(0, len(pending), int(cfg["batch_size"])),
            desc=f"{name}: length-stratified preferences",
        ):
            items = pending[offset:offset + int(cfg["batch_size"])]
            chosen, rejected = collate([row for _, row in items])
            chosen = {k: v.cuda() for k, v in chosen.items()}
            rejected = {k: v.cuda() for k, v in rejected.items()}

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                pc = sequence_logp(model, chosen)
                pr = sequence_logp(model, rejected)
                with reference_mode(model):
                    rc = sequence_logp(model, chosen)
                    rr = sequence_logp(model, rejected)
                margins = pc - pr - rc + rr

            for position, (index, row) in enumerate(items):
                margin = float(margins[position].item())
                if not np.isfinite(margin):
                    raise RuntimeError(f"Non-finite margin: row {index}")
                append(preference_path, {
                    "row_index": index,
                    "prompt_id": row.get("prompt_id"),
                    "length_stratum": row["length_stratum"],
                    "preference_margin": margin,
                    "correct": int(margin > 0),
                })

        done = {r["prompt_index"] for r in records(word_path)}

        for index, row in tqdm(
            list(enumerate(word_prompts)),
            desc=f"{name}: word-limit prompts",
        ):
            if index in done:
                continue

            set_seed(int(cfg["seed"]) + index)
            messages = prompt_messages(row)
            prompt_text = "\n".join(
                str(message["content"]) for message in messages
            )
            generated = batch_generate(
                model, tokenizer, [messages],
                max_prompt_length=int(cfg["max_sequence_length"]),
                max_new_tokens=int(cfg["max_generation_tokens"]),
                **cfg["generation"],
            )
            response = generated["responses"][0]
            compliance = word_limit_compliance(prompt_text, response)
            if compliance is None:
                raise RuntimeError(f"Cannot parse word limit: prompt {index}")

            append(word_path, {
                "prompt_index": index,
                "prompt_id": row.get("prompt_id"),
                "prompt_messages": messages,
                "response": response,
                "response_length": generated["response_lengths"][0],
                "word_count": word_count(response),
                "word_limit_compliant": compliance,
                "hit_generation_cap": generated["truncated"][0],
            })

        preferences = records(preference_path)
        words = records(word_path)

        assert len(preferences) == len(indices)
        assert {r["row_index"] for r in preferences} == set(indices)
        assert len(words) == len(word_prompts)
        assert {r["prompt_index"] for r in words} == set(range(len(word_prompts)))

        strata = {}
        for stratum in sorted({row["length_stratum"] for row in rows}):
            subset = [
                r for r in preferences if r["length_stratum"] == stratum
            ]
            strata[stratum] = {
                "num_pairs": len(subset),
                "preference_accuracy": float(np.mean([
                    r["correct"] for r in subset
                ])),
            }

        lengths = np.array([r["response_length"] for r in words])
        combined[name] = {
            "strata": strata,
            "word_prompt_count": len(words),
            "word_limit_compliance": float(np.mean([
                r["word_limit_compliant"] for r in words
            ])),
            "word_prompt_response_length_mean": float(lengths.mean()),
            "word_prompt_response_length_std": float(lengths.std()),
            "word_prompt_generation_cap_rate": float(np.mean([
                r["hit_generation_cap"] for r in words
            ])),
        }

        del model
        gc.collect()
        torch.cuda.empty_cache()

    save_json(directory / "length_study_summary.json", combined)
    print(json.dumps(combined, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dpo.yaml")
    parser.add_argument(
        "--stage", choices=["train", "evaluate", "all"], default="all"
    )
    args = parser.parse_args()
    cfg = load_yaml(args.config)

    if args.stage in {"train", "all"}:
        summary = repo_path(cfg["results_dir"]) / "length_balanced_summary.json"
        if summary.exists():
            print("Length-balanced training already completed.", flush=True)
        else:
            subprocess.run([
                sys.executable, "-m", "task1_dpo.train",
                "--config", args.config,
                "--run-name", "length_balanced",
                "--dataset", cfg["paths"]["dpo_length_train"],
                "--output", cfg["length_output"],
            ], check=True)

    if args.stage in {"evaluate", "all"}:
        evaluate_length(cfg)


if __name__ == "__main__":
    main()
