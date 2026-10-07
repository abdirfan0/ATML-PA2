from __future__ import annotations

import argparse
import json
import subprocess
import sys

from common.data import load_yaml, repo_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dpo.yaml")
    parser.add_argument(
        "--stage", choices=["train", "evaluate", "all"], default="train"
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    results = repo_path(cfg["results_dir"])

    for beta in cfg["betas"]:
        beta = float(beta)
        name = f"beta_{beta:.2f}".replace(".", "p")
        output = f"outputs/task1_dpo/{name}"

        print(f"\n=== {name}: {args.stage} ===", flush=True)

        if args.stage in {"train", "all"}:
            summary_path = results / f"{name}_summary.json"
            metadata_path = results / f"{name}_run.json"

            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                metadata = json.loads(metadata_path.read_text())
                assert metadata["effective_beta"] == beta
                assert summary["num_examples"] == int(
                    cfg["short_ablation_examples"]
                )
                assert repo_path(
                    output + "/adapter_model.safetensors"
                ).exists()
                print("Completed training already exists; skipping.", flush=True)
            else:
                subprocess.run([
                    sys.executable, "-m", "task1_dpo.train",
                    "--config", args.config,
                    "--run-name", name,
                    "--beta", str(beta),
                    "--max-examples", str(cfg["short_ablation_examples"]),
                    "--output", output,
                ], check=True)

        if args.stage in {"evaluate", "all"}:
            subprocess.run([
                sys.executable, "-m", "task1_dpo.evaluate",
                "--config", args.config,
                "--adapter", output,
                "--name", name,
                "--beta", str(beta),
            ], check=True)


if __name__ == "__main__":
    main()
