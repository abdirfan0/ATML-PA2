from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from common.data import (
    load_yaml, repo_path, prompt_messages_from_preference,
)
from common.dpo_preprocessing import load_dpo_rows
from common.generation import (
    batch_generate, response_token_logprobs, score_reward_pairs,
)
from common.logging_utils import set_seed, save_json
from common.metrics import sampled_kl
from common.models import (
    load_policy, load_reward_model, load_tokenizer, reference_mode,
)
from task1_dpo.train import make_collate, sequence_logp


def append_record(path, record):
    with path.open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_records(path):
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def evaluate(config_path, adapter, name, beta=None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    directory = repo_path(cfg["results_dir"])
    directory.mkdir(parents=True, exist_ok=True)

    # Use the actual training beta when its run metadata is available.
    training_metadata = directory / f"{name}_run.json"
    if beta is None and training_metadata.exists():
        beta = json.loads(
            training_metadata.read_text()
        )["effective_beta"]
    beta = float(cfg["beta"] if beta is None else beta)

    tokenizer = load_tokenizer(cfg["base_model"])
    rows, preprocessing = load_dpo_rows(
        cfg["paths"]["dpo_standard_eval"],
        tokenizer,
        int(cfg["max_sequence_length"]),
    )
    indices = preprocessing["selected_indices"]

    prefix = directory / name
    metadata_path = Path(f"{prefix}_eval_run.json")
    preference_path = Path(f"{prefix}_preferences.jsonl")
    generation_path = Path(f"{prefix}_generations.jsonl")
    reward_path = Path(f"{prefix}_rewards.jsonl")

    adapter_file = repo_path(adapter) / "adapter_model.safetensors"
    signature = {
        "adapter": str(repo_path(adapter)),
        "adapter_sha256": hashlib.sha256(
            adapter_file.read_bytes()
        ).hexdigest(),
        "config": cfg,
        "beta": beta,
        "eval_dataset_sha256": hashlib.sha256(
            repo_path(cfg["paths"]["dpo_standard_eval"]).read_bytes()
        ).hexdigest(),
        "eval_indices": indices,
        "generation_batch_size": 1,
        "generation_seed_rule": "config seed + original row index",
        "kl_averaging": "response-token weighted mean across all examples",
        "reward_max_length": (
            int(cfg["max_sequence_length"])
            + int(cfg["max_generation_tokens"]) + 64
        ),
    }

    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != signature:
            raise RuntimeError(
                "Existing evaluation uses different settings or adapter. "
                "Use a different --name."
            )
    else:
        save_json(metadata_path, signature)

    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    model.config.use_cache = True

    # 1. Teacher-forced preference evaluation.
    completed = {
        record["row_index"] for record in read_records(preference_path)
    }
    collate = make_collate(tokenizer, int(cfg["max_sequence_length"]))

    pending = [
        (index, row)
        for index, row in zip(indices, rows)
        if index not in completed
    ]

    for offset in tqdm(
        range(0, len(pending), int(cfg["batch_size"])),
        desc="Held-out preferences",
    ):
        items = pending[offset:offset + int(cfg["batch_size"])]
        chosen, rejected = collate([row for _, row in items])
        chosen = {key: value.cuda() for key, value in chosen.items()}
        rejected = {key: value.cuda() for key, value in rejected.items()}

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            policy_chosen = sequence_logp(model, chosen)
            policy_rejected = sequence_logp(model, rejected)

            with reference_mode(model):
                ref_chosen = sequence_logp(model, chosen)
                ref_rejected = sequence_logp(model, rejected)

            margins = (
                policy_chosen - policy_rejected
                - ref_chosen + ref_rejected
            )
            losses = -F.logsigmoid(beta * margins)

        for position, (index, _) in enumerate(items):
            margin = float(margins[position].item())
            loss = float(losses[position].item())
            if not np.isfinite(margin) or not np.isfinite(loss):
                raise RuntimeError(f"Non-finite preference result: row {index}")

            append_record(preference_path, {
                "row_index": index,
                "preference_margin": margin,
                "correct": int(margin > 0),
                "dpo_loss": loss,
                "chosen_response_tokens":
                    int(chosen["response_mask"][position].sum().item()),
                "rejected_response_tokens":
                    int(rejected["response_mask"][position].sum().item()),
            })

    # 2. Generate once per prompt; score policy/reference on that response.
    completed = {
        record["row_index"] for record in read_records(generation_path)
    }

    for index, row in tqdm(
        list(zip(indices, rows)), desc="Generate + sampled KL"
    ):
        if index in completed:
            continue

        # Per-prompt seeds make interrupted runs reproducible.
        set_seed(int(cfg["seed"]) + index)
        prompt = prompt_messages_from_preference(row)

        generated = batch_generate(
            model, tokenizer, [prompt],
            max_prompt_length=int(cfg["max_sequence_length"]),
            max_new_tokens=int(cfg["max_generation_tokens"]),
            **cfg["generation"],
        )

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            policy_logp = response_token_logprobs(
                model,
                generated["sequences"],
                generated["attention_mask"],
                generated["prompt_width"],
                generated["response_ids"],
            )[0]

            with reference_mode(model):
                ref_logp = response_token_logprobs(
                    model,
                    generated["sequences"],
                    generated["attention_mask"],
                    generated["prompt_width"],
                    generated["response_ids"],
                )[0]

            kl = sampled_kl(
                policy_logp, ref_logp, generated["response_mask"]
            ).item()

        if not np.isfinite(kl):
            raise RuntimeError(f"Non-finite KL: row {index}")

        append_record(generation_path, {
            "row_index": index,
            "seed": int(cfg["seed"]) + index,
            "prompt_messages": prompt,
            "response": generated["responses"][0],
            "response_tokens": generated["response_ids"][0][
                :generated["response_lengths"][0]
            ].cpu().tolist(),
            "response_length": generated["response_lengths"][0],
            "sampled_kl_token_mean": kl,
            "terminated_with_eos": generated["terminated_with_eos"][0],
            "hit_generation_cap": generated["truncated"][0],
        })

    # Release the policy before loading the frozen reward model.
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Reward scoring reuses saved generations.
    generations = read_records(generation_path)
    completed = {
        record["row_index"] for record in read_records(reward_path)
    }
    pending = [
        record for record in generations
        if record["row_index"] not in completed
    ]

    if pending:
        reward_model, reward_tokenizer = load_reward_model(cfg)

        for record in tqdm(pending, desc="Reward-model scoring"):
            score = score_reward_pairs(
                reward_model,
                reward_tokenizer,
                [record["prompt_messages"]],
                [record["response"]],
                max_length=signature["reward_max_length"],
            )[0].item()

            if not np.isfinite(score):
                raise RuntimeError(
                    f"Non-finite reward: row {record['row_index']}"
                )

            append_record(reward_path, {
                "row_index": record["row_index"],
                "reward": score,
            })

    preferences = read_records(preference_path)
    rewards = read_records(reward_path)

    expected = set(indices)
    for label, records in [
        ("preferences", preferences),
        ("generations", generations),
        ("rewards", rewards),
    ]:
        recorded = [record["row_index"] for record in records]
        if len(recorded) != len(expected) or set(recorded) != expected:
            raise RuntimeError(f"Incomplete or duplicate {label} records.")

    lengths = np.array([
        record["response_length"] for record in generations
    ], dtype=float)

    summary = {
        "name": name,
        "beta": beta,
        "num_eval_pairs": len(preferences),
        "heldout_dpo_loss": float(np.mean([
            record["dpo_loss"] for record in preferences
        ])),
        "heldout_preference_accuracy": float(np.mean([
            record["correct"] for record in preferences
        ])),
        "sampled_kl_token_mean": float(
            sum(
                record["sampled_kl_token_mean"] * record["response_length"]
                for record in generations
            ) / lengths.sum()
        ),
        "reward_model_score_mean": float(np.mean([
            record["reward"] for record in rewards
        ])),
        "response_length_mean": float(lengths.mean()),
        "response_length_std": float(lengths.std()),
        "generation_cap_rate": float(np.mean([
            record["hit_generation_cap"] for record in generations
        ])),
    }

    save_json(Path(f"{prefix}_eval_summary.json"), summary)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dpo.yaml")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--name", default="standard")
    parser.add_argument("--beta", type=float)
    args = parser.parse_args()
    evaluate(args.config, args.adapter, args.name, args.beta)


if __name__ == "__main__":
    main()
