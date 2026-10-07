from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.logging_utils import set_seed
from common.models import load_policy, load_tokenizer, trainable_parameters
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    from common.dpo_preprocessing import load_dpo_rows

    tokenizer = load_tokenizer(cfg["base_model"])
    rows, preprocessing = load_dpo_rows(
        path,
        tokenizer,
        int(cfg["max_sequence_length"]),
        max_examples=max_examples,
    )
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "preprocessing": preprocessing,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }



def sequence_logp(model, batch):
    """Sum next-token log probabilities over response tokens only."""
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    # Left padding must not change real-token position indices.
    positions = batch["attention_mask"].cumsum(-1) - 1
    positions = positions.clamp_min(0)

    logits = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=positions,
        use_cache=False,
    ).logits

    targets = batch["input_ids"][:, 1:]
    mask = batch["response_mask"][:, 1:]
    scores = logits[:, :-1, :]

    # Small chunks avoid a large temporary FP32 vocabulary tensor.
    def token_logp(chunk, labels):
        return -F.cross_entropy(
            chunk.float().reshape(-1, chunk.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).reshape(labels.shape)

    total = torch.zeros(
        targets.shape[0], device=targets.device, dtype=torch.float32
    )

    for offset in range(0, targets.shape[1], 32):
        chunk = scores[:, offset:offset + 32, :]
        labels = targets[:, offset:offset + 32]
        response_mask = mask[:, offset:offset + 32]

        if torch.is_grad_enabled() and chunk.requires_grad:
            logp = checkpoint(
                token_logp, chunk, labels, use_reentrant=False
            )
        else:
            logp = token_logp(chunk, labels)

        total = total + (logp * response_mask).sum(-1)

    return total


def run_training(
    config_path: str,
    run_name: str,
    dataset_path: str | None = None,
    output_path: str | None = None,
    beta: float | None = None,
    max_examples: int | None = None,
):
    import hashlib
    import json
    import math
    import subprocess
    import time

    from tqdm.auto import tqdm
    from common.models import reference_mode

    bundle = prepare_dpo_run(
        config_path, dataset_path, beta, max_examples
    )
    cfg = bundle["cfg"]
    rows = bundle["rows"]
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    loader = bundle["loader"]
    beta_value = bundle["beta"]

    if not torch.cuda.is_available():
        raise RuntimeError("Use a GPU runtime for this training run.")
    if not rows:
        raise RuntimeError("Training dataset is empty.")

    output = repo_path(output_path or cfg["standard_output"])
    result_dir = repo_path(cfg["results_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    log_path = result_dir / f"{run_name}_training.jsonl"
    config_path_out = result_dir / f"{run_name}_run.json"
    summary_path = result_dir / f"{run_name}_summary.json"

    # Protect existing results from accidental overwriting.
    if log_path.exists() or output.exists():
        raise FileExistsError(
            f"Run {run_name!r} already has logs or an output directory. "
            "Use a new run name and output path."
        )

    # Validate every row before beginning optimization.
    for index, row in enumerate(rows):
        try:
            prompt = prompt_messages_from_preference(row)
            chosen, rejected = preference_responses(row)
            encode_prompt_response(
                tokenizer, prompt, chosen, int(cfg["max_sequence_length"])
            )
            encode_prompt_response(
                tokenizer, prompt, rejected, int(cfg["max_sequence_length"])
            )
        except Exception as exc:
            raise RuntimeError(
                f"Training row {index} cannot be encoded: {exc}"
            ) from exc

    # FP32 trainable adapters allow stable AdamW and AMP gradient scaling.
    for parameter in trainable_parameters(model):
        parameter.data = parameter.data.float()

    parameters = trainable_parameters(model)
    optimizer = AdamW(
        parameters,
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    scaler = torch.amp.GradScaler("cuda")

    accumulation = int(cfg["grad_accum_steps"])
    epochs = int(cfg["epochs"])
    max_norm = float(cfg["max_grad_norm"])

    dataset_file = repo_path(
        dataset_path or cfg["paths"]["dpo_standard_train"]
    )
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        commit = None

    metadata = {
        "run_name": run_name,
        "config": cfg,
        "effective_beta": beta_value,
        "dataset": str(dataset_file),
        "dataset_sha256": hashlib.sha256(
            dataset_file.read_bytes()
        ).hexdigest(),
        "row_indices": bundle["preprocessing"]["selected_indices"],
        "preprocessing": bundle["preprocessing"],
        "num_examples": len(rows),
        "output": str(output),
        "git_commit": commit,
        "train_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "objective_sha256": hashlib.sha256(
            repo_path("task1_dpo/dpo.py").read_bytes()
        ).hexdigest(),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
    }
    config_path_out.write_text(json.dumps(metadata, indent=2))

    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    updates = 0
    skipped_updates = 0
    examples_seen = 0
    cumulative_loss = 0.0

    def save_checkpoint(epoch, batch_index):
        output.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output)
        tokenizer.save_pretrained(output)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "batch_index": batch_index,
                "updates": updates,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
            },
            output / "training_state.pt",
        )

    for epoch in range(epochs):
        window_loss = 0.0
        window_accuracy = 0.0
        window_examples = 0

        progress = tqdm(loader, desc=f"{run_name}: epoch {epoch + 1}")

        for batch_index, (chosen, rejected) in enumerate(progress):
            chosen = {k: v.cuda() for k, v in chosen.items()}
            rejected = {k: v.cuda() for k, v in rejected.items()}

            # Correctly normalize the final, possibly partial accumulation window.
            window_start = (batch_index // accumulation) * accumulation
            window_end = min(window_start + accumulation, len(loader))
            window_size = sum(
                min(
                    int(cfg["batch_size"]),
                    len(rows) - i * int(cfg["batch_size"]),
                )
                for i in range(window_start, window_end)
            )
            batch_size = chosen["input_ids"].shape[0]

            with torch.no_grad(), reference_mode(model):
                with torch.autocast("cuda", dtype=torch.float16):
                    ref_chosen = sequence_logp(model, chosen)
                    ref_rejected = sequence_logp(model, rejected)

            with torch.autocast("cuda", dtype=torch.float16):
                policy_chosen = sequence_logp(model, chosen)
                policy_rejected = sequence_logp(model, rejected)
                loss, diagnostics = dpo_loss(
                    policy_chosen, policy_rejected,
                    ref_chosen, ref_rejected,
                    beta_value,
                )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at batch {batch_index}."
                )

            scaler.scale(loss * batch_size / window_size).backward()

            loss_value = loss.detach().item()
            accuracy = diagnostics["preference_accuracy"].item()
            examples_seen += batch_size
            cumulative_loss += loss_value * batch_size
            window_loss += loss_value * batch_size
            window_accuracy += accuracy * batch_size
            window_examples += batch_size

            if batch_index + 1 == window_end:
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, max_norm
                )

                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.get_scale() < old_scale
                optimizer.zero_grad(set_to_none=True)

                updates += int(not skipped)
                skipped_updates += int(skipped)

                record = {
                    "epoch": epoch + 1,
                    "batch_index": batch_index,
                    "optimizer_updates": updates,
                    "examples_seen": examples_seen,
                    "loss": window_loss / window_examples,
                    "training_preference_accuracy":
                        window_accuracy / window_examples,
                    "gradient_norm_before_clipping":
                        float(gradient_norm.item())
                        if torch.isfinite(gradient_norm) else None,
                    "amp_skipped_update": skipped,
                    "amp_scale": scaler.get_scale(),
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_vram_gib":
                        torch.cuda.max_memory_allocated() / 2**30,
                }
                with log_path.open("a") as stream:
                    stream.write(json.dumps(record) + "\n")

                progress.set_postfix(
                    loss=f"{record['loss']:.4f}",
                    updates=updates,
                    skipped=skipped_updates,
                )
                window_loss = window_accuracy = 0.0
                window_examples = 0

                if not skipped and updates % 25 == 0:
                    save_checkpoint(epoch, batch_index)

        save_checkpoint(epoch, batch_index)

    torch.cuda.synchronize()
    summary = {
        "run_name": run_name,
        "num_examples": len(rows),
        "epochs": epochs,
        "optimizer_updates": updates,
        "skipped_updates": skipped_updates,
        "mean_training_loss": cumulative_loss / examples_seen,
        "wall_seconds": time.perf_counter() - started,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "adapter_path": str(output),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
