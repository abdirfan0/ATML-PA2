"""CPU validation using synthetic policy outputs, not real experiment results."""
import tempfile
import unittest
from pathlib import Path

from common.data import load_yaml
from task5_feedback.rlvr import exact_reward, extract_designated_final
from task5_feedback.pipeline import (POLICIES, diagnostic_groups, diagnostic_tasks, diagnostic_tables,
                                    pair_records, policy_tables, public_task, save_rows,
                                    score_response, summarize, validate, bind_run, check_pairs)


class FeedbackTests(unittest.TestCase):
    def test_designated_final(self):
        self.assertEqual(extract_designated_final("Candidate 9 is discarded.\n#### 10"), "10")
        self.assertEqual(exact_reward("Candidate 9 is discarded.\n#### 10", "9"), 0)
        self.assertEqual(exact_reward("#### 9\n#### 10", "9"), 0)
        self.assertEqual(exact_reward("Answer is 9", "9"), 0)
        self.assertEqual(exact_reward("#### 1,000", "1000"), 1)

    def test_format_separate_from_verifier(self):
        r = score_response("#### 9\nExtra prose", "9")
        self.assertEqual(r["exact_reward"], 1)
        self.assertFalse(r["format_compliant"])
        self.assertTrue(score_response("Reasoning.\n#### -2.5\n", "-2.5")["format_compliant"])
        self.assertEqual(score_response("Answer 9", "9")["failure_type"], "missing_designated_final")

    def test_released_diagnostics(self):
        cfg = load_yaml("configs/feedback.yaml")
        groups = diagnostic_groups(cfg)
        self.assertEqual(len(groups), 20)
        for group in groups.values():
            self.assertEqual(exact_reward(group["corrupt_reasoning_correct_final"]["response"], group["clean_correct"]["gold_final"]), 1)
            self.assertEqual(exact_reward(group["gold_distractor_wrong_final"]["response"], group["clean_correct"]["gold_final"]), 0)

    def test_resume_metadata_and_pair_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bind_run(root, {"x": 1})
            bind_run(root, {"x": 1})
            with self.assertRaises(AssertionError):
                bind_run(root, {"x": 2})
        task = {"a_sha256": "a", "b_sha256": "b", "a_response": "x", "b_response": "y"}
        check_pairs([{**public_task(task), "ai_preference": "A"}], [task])
        with self.assertRaises(AssertionError):
            check_pairs([{**public_task(task), "a_sha256": "changed", "ai_preference": "A"}], [task])

    def test_full_aggregation(self):
        sets = {name: [{"source_index": i, "question": f"Synthetic question {i}", "gold_final": "2",
                       "messages": [{"role": "user", "content": f"Synthetic prompt {i}"}]} for i in range(3)]
                for name in ["gsm", "transfer"]}
        groups = diagnostic_groups(load_yaml("configs/feedback.yaml"), 2)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for dataset, rows in sets.items():
                for name in POLICIES:
                    response = "Synthetic reasoning.\n#### " + ("1" if name == "sft" else "2")
                    generated = [{"row_index": i, "source_index": str(r["source_index"]), "dataset": dataset,
                                  "policy": name, "question": r["question"], "messages": r["messages"],
                                  "gold_final": "2", "response": response, "response_tokens": 8,
                                  "terminated_with_eos": True, "clipped_at_max": False, **score_response(response, "2")}
                                 for i, r in enumerate(rows)]
                    save_rows(root / f"generated_{dataset}_{name}.jsonl", generated)
            policy_pairs = [{**public_task(t), "ai_preference": t["verifier_preference"]} for t in pair_records(sets, root)]
            save_rows(root / "policy_pairs.jsonl", policy_pairs)
            diag = []
            for task in diagnostic_tasks(groups):
                preference = task["verifier_preference"]
                if task["variant_a"] == "clean_correct" and task["variant_b"] == "corrupt_reasoning_correct_final":
                    preference = "A"
                if task["variant_a"] == "clean_correct" and task["variant_b"] == "persuasive_filler_correct":
                    preference = "B"
                diag.append({**public_task(task), "ai_preference": preference})
            save_rows(root / "diagnostic_pairs.jsonl", diag)
            table, agreement, _, drops = policy_tables(root, sets)
            self.assertEqual(next(r for r in table if r["dataset"] == "gsm" and r["policy"] == "rlaif")["ai_pairwise_score_vs_sft"], 1)
            self.assertTrue(all(r["verifier_judge_agreement_all"] == 1 for r in agreement))
            self.assertTrue(all(r["exact_accuracy_gsm_minus_transfer"] == 0 for r in drops))
            variants, controlled, sensitivity = diagnostic_tables(root, groups)
            self.assertEqual(len(variants), 5)
            self.assertEqual(len(controlled), 8)
            ai = next(r for r in sensitivity if r["mechanism"] == "ai_pairwise")
            exact = next(r for r in sensitivity if r["mechanism"] == "exact_verifier")
            self.assertEqual(ai["S_reason"], 1)
            self.assertEqual(exact["S_reason"], 0)
            self.assertEqual(exact["reason_tie_rate"], 1)
            self.assertEqual(ai["filler_preferred_rate"], 1)
            self.assertEqual(exact["S_outcome"], 1)
            self.assertTrue(all(r["wrong_preference_rate"] is None for r in controlled if r["perturbation"] == "persuasive_filler_correct"))
            summarize(root, sets, groups)
            validate(root, sets, groups)


if __name__ == "__main__":
    unittest.main()
