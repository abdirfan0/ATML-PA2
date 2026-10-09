"""Frozen-policy Task 5 evaluation; course verifier and pairwise judge are unchanged."""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import itertools
import json
import math
import os
import re
import time
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from common.data import load_yaml, read_jsonl, repo_path
from task5_feedback.rlvr import exact_reward, extract_designated_final

POLICIES = {"sft": None, "rlvr": "checkpoints/rlvr_policy", "rlaif": "checkpoints/rlaif_policy"}
VARIANTS = ["clean_correct", "corrupt_reasoning_correct_final", "good_reasoning_wrong_final", "persuasive_filler_correct", "gold_distractor_wrong_final"]
FIXED = {"task5_feedback/rlvr.py": "f8865f7d6c67dc4d0565b182411383f0ec2ed9d01a0f99198ad2db0c31ec1bff",
         "task5_feedback/rlaif.py": "13e10a76c225aae8db5e17900bb070f89e4ebf51ab5b78dd86a5423740c1a1f9"}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_json(path, data):
    atomic(path, json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def save_rows(path, rows):
    atomic(path, "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows))


def read_rows(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save_csv(path, rows, fields=None):
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic(path, stream.getvalue())


def text_sha(text):
    return hashlib.sha256(str(text).encode()).hexdigest()


def check_config(cfg):
    for path, digest in FIXED.items():
        assert sha(repo_path(path)) == digest, f"Keep the supplied verifier/judge unchanged: {path}"
    assert cfg["base_model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg["ai_judge_model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert int(cfg["math_max_new_tokens"]) == 512
    for name, adapter in POLICIES.items():
        assert cfg["policies"][name] == adapter


def load_sets(cfg, limit=0):
    result = {}
    for name, key in [("gsm", "gsm_eval"), ("transfer", "math_transfer_eval")]:
        rows = read_jsonl(cfg["paths"][key])
        assert len(rows) == (300 if name == "gsm" else 100)
        for r in rows:
            assert isinstance(r["messages"], list) and r["messages"]
            assert str(r["gold_final"]).strip() and r["question"].strip()
        assert len({str(r["source_index"]) for r in rows}) == len(rows)
        result[name] = rows[:limit] if limit else rows
    return result


def diagnostic_groups(cfg, limit=0):
    rows = read_jsonl(cfg["paths"]["task5_diagnostics"])
    assert len(rows) == 100
    groups = {}
    for row in rows:
        pid = str(row["problem_id"])
        group = groups.setdefault(pid, {})
        assert row["variant_type"] not in group
        assert row.get("manual_validation") is True
        assert exact_reward(row["response"], row["gold_final"]) == row["expected_exact_reward"]
        group[row["variant_type"]] = row
    assert len(groups) == 20
    for group in groups.values():
        assert set(group) == set(VARIANTS)
        assert len({r["question"] for r in group.values()}) == 1
        assert len({str(r["gold_final"]) for r in group.values()}) == 1
    return dict(list(groups.items())[:limit]) if limit else groups


def signature(cfg, sets, groups, batch_size):
    source = ["task5_feedback/pipeline.py", "common/data.py", "common/models.py", "common/generation.py", *FIXED]
    adapters = {}
    for name, folder in POLICIES.items():
        if folder:
            path = repo_path(folder)
            weights = sorted(path.glob("adapter_model.*"))
            assert weights, f"Missing {name} adapter weights"
            adapters[name] = {p.name: sha(p) for p in [path / "adapter_config.json", *weights]}
    pkgs = {}
    for name in ["torch", "transformers", "peft", "bitsandbytes", "numpy"]:
        try:
            pkgs[name] = version(name)
        except PackageNotFoundError:
            pkgs[name] = "not installed"
    return {"config": cfg, "batch_size": batch_size, "do_sample": False,
            "max_response_length": 512, "max_prompt_length": 512,
            "source_hashes": {p: sha(repo_path(p)) for p in source}, "adapter_hashes": adapters,
            "data_hashes": {k: sha(repo_path(cfg["paths"][k])) for k in ["gsm_eval", "math_transfer_eval", "task5_diagnostics"]},
            "ordered_indices": {k: [str(r["source_index"]) for r in v] for k, v in sets.items()},
            "diagnostic_problem_ids": list(groups), "packages": pkgs}


def bind_run(root, sig):
    path = root / "run.json"
    if path.exists():
        assert json.loads(path.read_text()) == sig, "Metadata changed; preserve previous results and use a fresh output directory."
    else:
        assert not list(root.glob("generated_*.jsonl")), "Responses exist without run metadata."
        save_json(path, sig)


def configure(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    assert torch.cuda.is_available(), "Select an A100 GPU runtime."
    assert "A100" in torch.cuda.get_device_name(0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("GPU:", torch.cuda.get_device_name(0), flush=True)


def score_response(text, gold):
    pred = extract_designated_final(text)
    # Terminal-format compliance is separate from the supplied verifier's correctness.
    terminal = bool(re.search(r"(?:^|\n)\s*####\s*[-+]?\d[\d,]*(?:\.\d+)?\s*$", text))
    reward = exact_reward(text, gold)
    failure = "missing_designated_final" if pred is None else ("wrong_designated_final" if not reward else
              ("correct_terminal_format" if terminal else "correct_nonterminal_format"))
    return {"designated_final": pred, "exact_reward": reward,
            "format_compliant": terminal, "failure_type": failure}


def check_prefix(saved, rows, policy, dataset):
    assert len(saved) <= len(rows)
    for i, (r, s) in enumerate(zip(saved, rows)):
        assert r["row_index"] == i and r["dataset"] == dataset and r["policy"] == policy
        assert r["source_index"] == str(s["source_index"])
        assert r["messages"] == s["messages"] and r["question"] == s["question"]
        assert r["gold_final"] == str(s["gold_final"])
        assert isinstance(r["response"], str) and 0 < r["response_tokens"] <= 512
        assert r["clipped_at_max"] == (r["response_tokens"] == 512 and not r["terminated_with_eos"])
        for key, value in score_response(r["response"], s["gold_final"]).items():
            assert r[key] == value


def generate(cfg, root, sets, batch_size):
    import torch
    from common.generation import batch_generate
    from common.models import load_tokenizer, load_policy
    for name, adapter in POLICIES.items():
        pending = []
        for dataset, rows in sets.items():
            path = root / f"generated_{dataset}_{name}.jsonl"
            saved = read_rows(path)
            check_prefix(saved, rows, name, dataset)
            if len(saved) < len(rows):
                assert len(saved) % batch_size == 0
                pending.append((dataset, rows, path, saved))
        if not pending:
            print(name, "responses already complete", flush=True)
            continue
        started = time.perf_counter()
        tok = load_tokenizer(cfg["base_model"])
        model = load_policy(cfg, adapter_path=adapter, trainable=False)
        model.config.use_cache = True
        model.generation_config.use_cache = True
        for rows in sets.values():
            for row in rows:
                assert len(tok.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=True)) <= 512, "Prompt exceeds budget; never silently truncate a math problem."
        loaded = time.perf_counter()
        sessions = []
        for dataset, rows, path, saved in pending:
            prior = len(saved)
            t0 = time.perf_counter()
            for start in range(prior, len(rows), batch_size):
                chunk = rows[start:start+batch_size]
                gen = batch_generate(model, tok, [r["messages"] for r in chunk], max_prompt_length=512,
                                     max_new_tokens=512, do_sample=False, temperature=0.0, top_p=1.0)
                for i, row in enumerate(chunk):
                    text = gen["responses"][i]
                    saved.append({"row_index": start+i, "source_index": str(row["source_index"]),
                                  "policy": name, "dataset": dataset, "messages": row["messages"],
                                  "question": row["question"], "gold_final": str(row["gold_final"]),
                                  "response": text, "response_tokens": int(gen["response_lengths"][i]),
                                  "terminated_with_eos": bool(gen["terminated_with_eos"][i]),
                                  "clipped_at_max": bool(gen["truncated"][i]), **score_response(text, row["gold_final"])})
                del gen
                save_rows(path, saved)
                print(f"{dataset}/{name}: generation {len(saved)}/{len(rows)}; session seconds={time.perf_counter()-t0:.1f}", flush=True)
            sessions.append({"dataset": dataset, "start_row": prior, "new_rows": len(saved)-prior,
                             "generation_seconds_this_session": time.perf_counter()-t0})
        save_json(root / f"generation_{name}_timing.json", {"model_loading_seconds": loaded-started,
                  "sessions": sessions, "peak_vram_gib": torch.cuda.max_memory_allocated()/2**30,
                  "model_commit_hash": getattr(model.config, "_commit_hash", None)})
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def pair_records(sets, root):
    tasks = []
    for dataset, rows in sets.items():
        by_policy = {}
        for name in POLICIES:
            generated = read_rows(root / f"generated_{dataset}_{name}.jsonl")
            assert len(generated) == len(rows)
            check_prefix(generated, rows, name, dataset)
            by_policy[name] = generated
        for i, row in enumerate(rows):
            for a, b in itertools.combinations(POLICIES, 2):
                x, y = by_policy[a][i], by_policy[b][i]
                tasks.append({"dataset": dataset, "row_index": i, "source_index": str(row["source_index"]),
                              "question": row["question"], "policy_a": a, "policy_b": b,
                              "a_sha256": text_sha(x["response"]), "b_sha256": text_sha(y["response"]),
                              "verifier_preference": "A" if x["exact_reward"] > y["exact_reward"] else
                              ("B" if x["exact_reward"] < y["exact_reward"] else "TIE"),
                              "a_response": x["response"], "b_response": y["response"]})
    return tasks


def diagnostic_tasks(groups):
    tasks = []
    for pid, group in groups.items():
        for a, b in itertools.combinations(VARIANTS, 2):
            x, y = group[a], group[b]
            ra, rb = exact_reward(x["response"], x["gold_final"]), exact_reward(y["response"], y["gold_final"])
            tasks.append({"problem_id": pid, "question": x["question"], "variant_a": a, "variant_b": b,
                          "a_sha256": text_sha(x["response"]), "b_sha256": text_sha(y["response"]),
                          "verifier_preference": "A" if ra > rb else ("B" if ra < rb else "TIE"),
                          "a_response": x["response"], "b_response": y["response"]})
    return tasks


def public_task(task):
    return {k: v for k, v in task.items() if k not in {"a_response", "b_response"}}


def check_pairs(saved, tasks):
    assert len(saved) <= len(tasks)
    for r, t in zip(saved, tasks):
        assert public_task(t) == {k: v for k, v in r.items() if k != "ai_preference"}
        assert r["ai_preference"] in {"A", "B", "TIE"}


def judge(cfg, root, sets, groups):
    from task5_feedback.rlaif import PairwiseAIJudge
    import torch
    pools = [("policy_pairs.jsonl", pair_records(sets, root)),
             ("diagnostic_pairs.jsonl", diagnostic_tasks(groups))]
    pending = []
    for filename, tasks in pools:
        saved = read_rows(root / filename)
        check_pairs(saved, tasks)
        if len(saved) < len(tasks):
            pending.append((filename, tasks, saved))
    if not pending:
        print("All pairwise judge results already complete", flush=True)
        return
    # Preserve the fixed class, rubric, parser, hash-based A/B orientation and cache.
    ai = PairwiseAIJudge(cfg, root / "pairwise_cache.json")
    started = time.perf_counter()
    counts = []
    for filename, tasks, saved in pending:
        prior = len(saved)
        for task in tasks[prior:]:
            preference = ai.compare(task["question"], task["a_response"], task["b_response"])
            saved.append({**public_task(task), "ai_preference": preference})
            save_rows(root / filename, saved)
            if len(saved) % 20 == 0 or len(saved) == len(tasks):
                print(f"{filename}: {len(saved)}/{len(tasks)}; session seconds={time.perf_counter()-started:.1f}", flush=True)
        counts.append({"file": filename, "start_row": prior, "new_comparisons": len(saved)-prior})
    save_json(root / "judge_timing.json", {"seconds_this_session": time.perf_counter()-started,
              "counts": counts, "model_commit_hash": getattr(ai.model.config, "_commit_hash", None)})
    del ai
    gc.collect()
    torch.cuda.empty_cache()


def pair_rates(rows, field):
    n = len(rows)
    assert n
    return {"num_pairs": n, "a_preferred_rate": sum(r[field] == "A" for r in rows)/n,
            "tie_rate": sum(r[field] == "TIE" for r in rows)/n,
            "b_preferred_rate": sum(r[field] == "B" for r in rows)/n}


def policy_tables(root, sets):
    tasks = pair_records(sets, root)
    pairs = read_rows(root / "policy_pairs.jsonl")
    check_pairs(pairs, tasks)
    assert len(pairs) == len(tasks)
    result, agreements, failures = [], [], []
    for dataset, rows in sets.items():
        for policy in POLICIES:
            generated = read_rows(root / f"generated_{dataset}_{policy}.jsonl")
            relevant = [r for r in pairs if r["dataset"] == dataset and policy in {r["policy_a"], r["policy_b"]}]
            scores = []
            for r in relevant:
                outcome = r["ai_preference"]
                scores.append(.5 if outcome == "TIE" else float((outcome == "A") == (r["policy_a"] == policy)))
            versus = [r for r in relevant if "sft" in {r["policy_a"], r["policy_b"]}]
            vs_scores = [.5 if r["ai_preference"] == "TIE" else float(
                         (r["ai_preference"] == "A") == (r["policy_a"] == policy)) for r in versus]
            n = len(generated)
            result.append({"dataset": dataset, "policy": policy, "num_prompts": n,
                           "exact_accuracy": sum(r["exact_reward"] for r in generated)/n,
                           "format_compliance": sum(r["format_compliant"] for r in generated)/n,
                           "response_length_mean": sum(r["response_tokens"] for r in generated)/n,
                           "generation_cap_rate": sum(r["clipped_at_max"] for r in generated)/n,
                           "ai_pairwise_score_vs_sft": .5 if policy == "sft" else sum(vs_scores)/len(vs_scores),
                           "ai_group_win_rate": sum(scores)/len(scores)})
            for failure, count in Counter(r["failure_type"] for r in generated).items():
                failures.append({"dataset": dataset, "policy": policy, "failure_type": failure, "count": count, "rate": count/n})
        for a, b in itertools.combinations(POLICIES, 2):
            pool = [r for r in pairs if r["dataset"] == dataset and r["policy_a"] == a and r["policy_b"] == b]
            decisive = [r for r in pool if r["verifier_preference"] != "TIE"]
            agreements.append({"dataset": dataset, "policy_a": a, "policy_b": b,
                               **pair_rates(pool, "ai_preference"),
                               "verifier_judge_agreement_all": sum(r["ai_preference"] == r["verifier_preference"] for r in pool)/len(pool),
                               "verifier_decisive_pairs": len(decisive),
                               "verifier_judge_agreement_on_verifier_decisive":
                               sum(r["ai_preference"] == r["verifier_preference"] for r in decisive)/len(decisive) if decisive else None,
                               "verifier_tie_rate": sum(r["verifier_preference"] == "TIE" for r in pool)/len(pool)})
    drops = []
    for policy in POLICIES:
        a = next(r for r in result if r["dataset"] == "gsm" and r["policy"] == policy)
        b = next(r for r in result if r["dataset"] == "transfer" and r["policy"] == policy)
        drops.append({"policy": policy, **{k+"_gsm_minus_transfer": a[k]-b[k] for k in
                      ["exact_accuracy", "format_compliance", "ai_pairwise_score_vs_sft", "response_length_mean"]}})
    return result, agreements, failures, drops


def diagnostic_tables(root, groups):
    tasks = diagnostic_tasks(groups)
    pairs = read_rows(root / "diagnostic_pairs.jsonl")
    check_pairs(pairs, tasks)
    assert len(pairs) == len(tasks)
    variant_table, controlled = [], []
    for variant in VARIANTS:
        rewards, scores = [], []
        for pid, group in groups.items():
            r = group[variant]
            rewards.append(exact_reward(r["response"], r["gold_final"]))
            relevant = [p for p in pairs if p["problem_id"] == pid and variant in {p["variant_a"], p["variant_b"]}]
            wins = sum(.5 if p["ai_preference"] == "TIE" else float(
                       (p["ai_preference"] == "A") == (p["variant_a"] == variant)) for p in relevant)
            assert len(relevant) == 4
            scores.append(wins/4)
        variant_table.append({"variant_type": variant, "num_responses": len(groups),
                              "exact_reward_mean": sum(rewards)/len(rewards),
                              "ai_group_win_rate_mean": sum(scores)/len(scores)})
    for variant in VARIANTS[1:]:
        pool = [p for p in pairs if p["variant_a"] == "clean_correct" and p["variant_b"] == variant]
        for mechanism, field in [("exact_verifier", "verifier_preference"), ("ai_pairwise", "ai_preference")]:
            rates = pair_rates(pool, field)
            filler = variant == "persuasive_filler_correct"
            controlled.append({"perturbation": variant, "mechanism": mechanism, "num_pairs": len(pool),
                               "clean_preferred_rate": rates["a_preferred_rate"], "tie_rate": rates["tie_rate"],
                               "perturbed_preferred_rate": rates["b_preferred_rate"],
                               "better_response_rate": None if filler else rates["a_preferred_rate"],
                               "wrong_preference_rate": None if filler else rates["b_preferred_rate"],
                               "expected_relation": "equivalent mathematical content; report style preference, not a correctness ranking" if filler else "clean diagnostically better"})
    sensitivities = []
    for mechanism in ["exact_verifier", "ai_pairwise"]:
        reason = next(r for r in controlled if r["mechanism"] == mechanism and r["perturbation"] == "corrupt_reasoning_correct_final")
        outcome = next(r for r in controlled if r["mechanism"] == mechanism and r["perturbation"] == "good_reasoning_wrong_final")
        filler = next(r for r in controlled if r["mechanism"] == mechanism and r["perturbation"] == "persuasive_filler_correct")
        distractor = next(r for r in controlled if r["mechanism"] == mechanism and r["perturbation"] == "gold_distractor_wrong_final")
        sensitivities.append({"mechanism": mechanism, "S_reason": reason["clean_preferred_rate"],
                              "reason_tie_rate": reason["tie_rate"], "reason_wrong_preference_rate": reason["perturbed_preferred_rate"],
                              "S_outcome": outcome["clean_preferred_rate"], "outcome_tie_rate": outcome["tie_rate"],
                              "outcome_wrong_preference_rate": outcome["perturbed_preferred_rate"],
                              "filler_preferred_rate": filler["perturbed_preferred_rate"], "filler_tie_rate": filler["tie_rate"],
                              "clean_preferred_over_distractor_rate": distractor["clean_preferred_rate"]})
    return variant_table, controlled, sensitivities


def summarize(root, sets, groups):
    comparison, agreement, failures, drops = policy_tables(root, sets)
    variants, controlled, sensitivities = diagnostic_tables(root, groups)
    tables = {"math_comparison": comparison, "verifier_judge_agreement": agreement,
              "failure_types": failures, "transfer_drops": drops, "diagnostic_variants": variants,
              "controlled_pairs": controlled, "reward_sensitivity": sensitivities}
    for name, table in tables.items():
        save_csv(root / f"{name}.csv", table)
    save_json(root / "feedback_summary.json", {**tables, "protocol": {
        "generation": "Greedy, unchanged released message lists, common 512-token response cap; no training.",
        "policy_pairs": "All three unordered policy pairs per prompt; report wins + half ties against SFT.",
        "diagnostics": "All ten pairs per five-response group; clean reference compared separately with each of four perturbations.",
        "filler": "Mathematical content unchanged; preference reported as style susceptibility, not proof of correctness.",
        "agreement": "Three-way verifier/AI ranking agreement; report all pairs and verifier-decisive subset.",
        "format": "Terminal standalone #### number field; supplied exact verifier may accept a nonterminal field.",
        "limits": ["Pairwise judge is not ground truth.", "Supplied judge maps unparsed outputs to TIE; cache does not retain raw judge outputs.",
                   "Hash-based orientation chooses one A/B ordering per pair, not a two-order adjudication.",
                   "Cross-dataset drops compare different prompt distributions; no causal optimizer-only attribution."]}})
    print("Saved seven CPU tables and feedback_summary.json", flush=True)


def validate(root, sets, groups):
    stored = json.loads((root / "feedback_summary.json").read_text())
    actual = policy_tables(root, sets) + diagnostic_tables(root, groups)
    names = ["math_comparison", "verifier_judge_agreement", "failure_types", "transfer_drops",
             "diagnostic_variants", "controlled_pairs", "reward_sensitivity"]
    for name, table in zip(names, actual):
        assert stored[name] == table, f"Summary mismatch: {name}"
    result = {"passed": True, "num_policy_responses": sum(len(v) for v in sets.values())*3,
              "num_policy_pairs": sum(len(v) for v in sets.values())*3,
              "num_diagnostic_responses": len(groups)*5, "num_diagnostic_pairs": len(groups)*10,
              "checks": ["fixed verifier and judge source hashes", "unchanged dataset hashes and ordered indices",
                         "identical prompts across the three fixed policies", "exact reward and terminal format recomputation",
                         "pair results bound to response hashes", "all comparison/diagnostic tables recomputed"]}
    save_json(root / "artifact_validation.json", result)
    print(json.dumps(result, indent=2), flush=True)


def benchmark(cfg, root):
    import torch
    from common.models import load_policy, load_tokenizer
    from common.generation import batch_generate
    rows = load_sets(cfg)["gsm"][:16]
    tok = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, trainable=False)
    model.config.use_cache = True
    model.generation_config.use_cache = True
    # Tiny warmup excludes model loading and CUDA initialization from comparisons.
    batch_generate(model, tok, [rows[0]["messages"]], max_prompt_length=512, max_new_tokens=4, do_sample=False)
    results = []
    for size in [4, 8, 16]:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        tokens = 0
        responses = []
        try:
            for offset in range(0, len(rows), size):
                gen = batch_generate(model, tok, [r["messages"] for r in rows[offset:offset+size]],
                                     max_prompt_length=512, max_new_tokens=512, do_sample=False)
                tokens += sum(gen["response_lengths"])
                responses.extend(gen["responses"])
                del gen
            torch.cuda.synchronize()
            seconds = time.perf_counter()-started
            results.append({"batch_size": size, "num_prompts": len(rows), "seconds": seconds,
                            "generated_tokens": tokens, "tokens_per_second": tokens/seconds,
                            "peak_vram_gib": torch.cuda.max_memory_allocated()/2**30,
                            "response_hashes": [text_sha(r) for r in responses]})
            print(json.dumps({k: v for k, v in results[-1].items() if k != "response_hashes"}, indent=2), flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"Batch {size}: OOM; keep smaller successful batch", flush=True)
            torch.cuda.empty_cache()
            break
    save_json(root / "generation_benchmark.json", {"conditions": results,
              "scope": "Throughput-only SFT benchmark on the same 16 prompts. All final policy comparisons must use one batch size; floating-point batching may change greedy outputs."})
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/feedback.yaml")
    p.add_argument("--stage", choices=["preflight", "benchmark", "generate", "judge", "summarize", "validate", "all"], default="preflight")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output", default="results/task5_feedback")
    a = p.parse_args()
    assert a.batch_size > 0 and a.limit >= 0
    cfg = load_yaml(a.config)
    check_config(cfg)
    sets, groups = load_sets(cfg, a.limit), diagnostic_groups(cfg, 2 if a.limit else 0)
    root = repo_path(a.output)
    if a.limit or a.stage == "benchmark":
        assert root != repo_path("results/task5_feedback"), "Use a separate output for smoke or benchmark."
    if a.stage == "benchmark":
        configure(int(cfg["seed"]))
        benchmark(cfg, root)
        return
    sig = signature(cfg, sets, groups, a.batch_size)
    if a.stage == "preflight":
        print(json.dumps({"dataset_rows": {k: len(v) for k, v in sets.items()}, "diagnostic_groups": len(groups),
                          "variants": VARIANTS, "policies": POLICIES, "packages": sig["packages"],
                          "decoding": "Greedy; 512 response tokens; released messages unchanged", "frozen_source_hashes": FIXED}, indent=2))
        return
    bind_run(root, sig)
    if a.stage in ["generate", "judge", "all"]:
        configure(int(cfg["seed"]))
    if a.stage in ["generate", "all"]:
        generate(cfg, root, sets, a.batch_size)
    if a.stage in ["judge", "all"]:
        judge(cfg, root, sets, groups)
    if a.stage in ["summarize", "all"]:
        summarize(root, sets, groups)
    if a.stage in ["validate", "all"]:
        validate(root, sets, groups)


if __name__ == "__main__":
    main()


