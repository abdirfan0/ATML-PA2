"""Validate saved Task 5 evidence and produce CPU figures without rerunning models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from common.data import load_yaml, repo_path
from task5_feedback.pipeline import (
    POLICIES, check_config, diagnostic_groups, diagnostic_tables, diagnostic_tasks,
    load_sets, pair_records, policy_tables, read_rows, save_csv, save_json, sha,
)

TABLES = ["math_comparison", "verifier_judge_agreement", "failure_types", "transfer_drops",
          "diagnostic_variants", "controlled_pairs", "reward_sensitivity"]
FIGURES = ["task5_policy_comparison", "task5_controlled_pairs",
           "task5_verifier_judge_agreement", "task5_failure_types"]


def check_csv(path, expected):
    with path.open(newline="", encoding="utf-8") as stream:
        actual = list(csv.DictReader(stream))
    assert len(actual) == len(expected), f"Row count: {path}"
    for left, right in zip(actual, expected):
        assert set(left) == set(right), f"Columns: {path}"
        for key, value in right.items():
            if value is None:
                assert left[key] == "", (path, key)
            elif isinstance(value, (int, float)):
                assert math.isclose(float(left[key]), value, rel_tol=1e-12, abs_tol=1e-12), (path, key)
            else:
                assert left[key] == str(value), (path, key)


def cache_key(model, task):
    payload = json.dumps({"model": model, "problem": task["question"],
                          "a": task["a_response"], "b": task["b_response"]}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_saved(cfg, root):
    check_config(cfg)
    sets, groups = load_sets(cfg), diagnostic_groups(cfg)
    run = json.loads((root / "run.json").read_text())
    assert run["config"] == cfg
    assert run["batch_size"] == 16 and run["do_sample"] is False
    assert run["max_response_length"] == 512 and run["max_prompt_length"] == 512
    for path, digest in run["source_hashes"].items():
        assert sha(repo_path(path)) == digest, f"Recorded source changed: {path}"
    for key, digest in run["data_hashes"].items():
        assert sha(repo_path(cfg["paths"][key])) == digest, f"Dataset changed: {key}"
    assert run["ordered_indices"] == {k: [str(r["source_index"]) for r in v] for k, v in sets.items()}
    assert run["diagnostic_problem_ids"] == list(groups)
    assert set(run["adapter_hashes"]) == {"rlvr", "rlaif"}
    assert all("adapter_config.json" in v and any(k.startswith("adapter_model.") for k in v)
               for v in run["adapter_hashes"].values())
    actual = policy_tables(root, sets) + diagnostic_tables(root, groups)
    summary = json.loads((root / "feedback_summary.json").read_text())
    for name, table in zip(TABLES, actual):
        assert summary[name] == table, f"Summary mismatch: {name}"
        check_csv(root / f"{name}.csv", table)
    cache = json.loads((root / "pairwise_cache.json").read_text())
    pools = [(pair_records(sets, root), read_rows(root / "policy_pairs.jsonl")),
             (diagnostic_tasks(groups), read_rows(root / "diagnostic_pairs.jsonl"))]
    for tasks, records in pools:
        assert len(tasks) == len(records)
        for task, record in zip(tasks, records):
            assert cache[cache_key(cfg["ai_judge_model"], task)] == record["ai_preference"]
    status = json.loads((root / "batch_status.json").read_text())
    assert status["state"] == "completed"
    result = {"passed": True, "num_policy_responses": 1200, "num_policy_pairs": 1200,
              "num_diagnostic_responses": 100, "num_diagnostic_pairs": 200,
              "checks": ["recorded source and dataset hashes", "fixed policy and decoding configuration",
                         "ordered prompts and diagnostic groups", "response hashes and verifier recomputation",
                         "pairwise cache binding", "all seven CSV and JSON tables recomputed"],
              "limitations": ["TIE includes unparsed judge outputs; raw judge text was not saved.",
                              "One hash-selected A/B orientation is used per pair.",
                              "CPU validation checks recorded adapter hashes, without requiring adapter weights.",
                              "Different evaluation datasets prevent causal interpretation of transfer differences."]}
    save_json(root / "final_artifact_validation.json", result)
    return sets, groups, actual, result


def supplements(cfg, root, sets, groups):
    tasks = diagnostic_tasks(groups)
    pairs = read_rows(root / "diagnostic_pairs.jsonl")
    candidates = []
    for variant in ["corrupt_reasoning_correct_final", "persuasive_filler_correct",
                    "good_reasoning_wrong_final", "gold_distractor_wrong_final"]:
        matches = [(t, p) for t, p in zip(tasks, pairs)
                   if t["variant_a"] == "clean_correct" and t["variant_b"] == variant]
        # Deterministic examples spanning the required conditions; keep raw labels and text.
        task, pair = next(((t, p) for t, p in matches if p["ai_preference"] != p["verifier_preference"]), matches[0])
        group = groups[task["problem_id"]]
        candidates.append({"problem_id": task["problem_id"], "perturbation": variant,
                           "question": task["question"], "gold_final": group[variant]["gold_final"],
                           "clean_response": task["a_response"], "perturbed_response": task["b_response"],
                           "clean_exact_reward": group["clean_correct"]["expected_exact_reward"],
                           "perturbed_exact_reward": group[variant]["expected_exact_reward"],
                           "verifier_preference": pair["verifier_preference"], "ai_preference": pair["ai_preference"],
                           "a_sha256": pair["a_sha256"], "b_sha256": pair["b_sha256"]})
    save_json(root / "qualitative_candidates.json", candidates)
    counts = []
    for tasks, records, name in [(pair_records(sets, root), read_rows(root / "policy_pairs.jsonl"), "policy"),
                                 (diagnostic_tasks(groups), pairs, "diagnostic")]:
        for dataset in (["gsm", "transfer"] if name == "policy" else ["controlled"]):
            pool = [(t, p) for t, p in zip(tasks, records)
                    if name != "policy" or p["dataset"] == dataset]
            counts.append({"comparison_set": dataset, "num_pairs": len(pool),
                           "identical_response_pairs": sum(t["a_response"] == t["b_response"] for t, _ in pool),
                           "ai_decisive_on_identical": sum(t["a_response"] == t["b_response"] and p["ai_preference"] != "TIE" for t, p in pool),
                           "ai_ties": sum(p["ai_preference"] == "TIE" for _, p in pool),
                           "verifier_ties": sum(p["verifier_preference"] == "TIE" for _, p in pool),
                           "unparsed_judge_count": None})
    save_csv(root / "pair_diagnostics.csv", counts)


def plot(tables, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    folder.mkdir(parents=True, exist_ok=True)
    colors = ["#3568a8", "#c87826", "#33856a"]
    def save(fig, name):
        for extension in ["png", "pdf"]:
            target = folder / f"{name}.{extension}"
            temporary = target.with_suffix(target.suffix + ".tmp")
            fig.savefig(temporary, format=extension, dpi=220, bbox_inches="tight")
            assert temporary.stat().st_size > 0, f"Empty figure export: {temporary}"
            temporary.replace(target)
        plt.close(fig)
    metrics = [("exact_accuracy", "Exact final-answer accuracy"),
               ("ai_pairwise_score_vs_sft", "AI pairwise score against SFT"),
               ("format_compliance", "Terminal format compliance"),
               ("response_length_mean", "Mean response tokens"),
               ("generation_cap_rate", "Generation cap rate"),
               ("ai_group_win_rate", "AI group win rate")]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7), layout="constrained")
    for ax, (key, title) in zip(axes.flat, metrics):
        for j, dataset in enumerate(["gsm", "transfer"]):
            rows = [r for r in tables[0] if r["dataset"] == dataset]
            for i, row in enumerate(rows):
                ax.bar(j + (i-1)*.23, row[key], width=.22, color=colors[i], label=row["policy"].upper() if j == 0 else None)
        ax.set_xticks([0, 1], ["GSM8K (n=300)", "SVAMP (n=100)"])
        ax.set_title(title)
        if key != "response_length_mean":
            ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=.18)
    axes.flat[0].legend(fontsize=9)
    fig.suptitle("Task 5: frozen policy evaluation\nPairwise scores count ties as 0.5; SFT against itself is the defined 0.5 baseline", fontsize=12)
    save(fig, FIGURES[0])
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.5), layout="constrained")
    variants = ["corrupt_reasoning_correct_final", "good_reasoning_wrong_final",
                "persuasive_filler_correct", "gold_distractor_wrong_final"]
    titles = ["Corrupted reasoning", "Wrong final answer", "Persuasive filler", "Gold distractor"]
    for ax, variant, title in zip(axes, variants, titles):
        rows = [r for r in tables[5] if r["perturbation"] == variant]
        bottom = np.zeros(2)
        for key, label, color in [("clean_preferred_rate", "Clean preferred", "#3568a8"),
                                  ("tie_rate", "Tie", "#bcbcbc"),
                                  ("perturbed_preferred_rate", "Perturbed preferred", "#c87826")]:
            vals = np.array([r[key] for r in rows])
            ax.bar([0, 1], vals, bottom=bottom, color=color, label=label)
            bottom += vals
        ax.set_xticks([0, 1], ["Verifier", "AI judge"])
        ax.set_title(title + "\n20 pairs")
        ax.set_ylim(0, 1)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.3, -.15), ncol=3)
    fig.suptitle("Controlled comparisons with the clean response\nFiller preserves mathematical content; its preferences measure style. AI ties include parser fallback.", fontsize=12)
    save(fig, FIGURES[1])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), layout="constrained")
    for ax, dataset in zip(axes, ["gsm", "transfer"]):
        rows = [r for r in tables[1] if r["dataset"] == dataset]
        x = np.arange(3)
        ax.bar(x-.18, [r["verifier_judge_agreement_all"] for r in rows], width=.35, color=colors[0], label="All pairs")
        ax.bar(x+.18, [r["verifier_judge_agreement_on_verifier_decisive"] for r in rows], width=.35, color=colors[1], label="Verifier-decisive pairs")
        ax.set_xticks(x, [f'{r["policy_a"].upper()} / {r["policy_b"].upper()}\nDecisive n={r["verifier_decisive_pairs"]}' for r in rows], fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(dataset.upper())
        ax.grid(axis="y", alpha=.18)
    axes[0].legend(fontsize=9)
    fig.suptitle("Three-way verifier–judge agreement\nAgreement on all pairs includes shared ties; decisive subsets have small denominators", fontsize=12)
    save(fig, FIGURES[2])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    failures = ["missing_designated_final", "wrong_designated_final", "correct_nonterminal_format", "correct_terminal_format"]
    labels = ["Missing final field", "Wrong final field", "Correct, nonterminal", "Correct, terminal"]
    for ax, dataset in zip(axes, ["gsm", "transfer"]):
        bottom = np.zeros(3)
        for failure, label, color in zip(failures, labels, ["#bbbbbb", "#c87826", "#3568a8", "#33856a"]):
            vals = np.array([sum(r["rate"] for r in tables[2] if r["dataset"] == dataset and r["policy"] == p and r["failure_type"] == failure) for p in POLICIES])
            ax.bar(range(3), vals, bottom=bottom, label=label, color=color)
            bottom += vals
        ax.set_xticks(range(3), [p.upper() for p in POLICIES])
        ax.set_ylim(0, 1)
        ax.set_title(dataset.upper())
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    fig.suptitle("Designated-final outcomes\nCorrectness uses the supplied verifier; terminal format is measured separately", fontsize=12)
    save(fig, FIGURES[3])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/feedback.yaml")
    p.add_argument("--output", default="results/task5_feedback")
    p.add_argument("--figures", default="report/figures")
    p.add_argument("--stage", choices=["all", "validate"], default="all")
    args = p.parse_args()
    cfg, root = load_yaml(args.config), repo_path(args.output)
    sets, groups, tables, result = validate_saved(cfg, root)
    if args.stage == "all":
        supplements(cfg, root, sets, groups)
        plot(tables, repo_path(args.figures))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
