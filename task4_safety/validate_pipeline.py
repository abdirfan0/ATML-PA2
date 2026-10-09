"""CPU checks for denominators, resumability, and blind audit handling."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from task4_safety.pipeline import (POLICIES, audit, bind_run, check_judgments,
                                  metrics, save_rows, summarize, validate)


class PipelineTests(unittest.TestCase):
    def test_denominators(self):
        labels = ["SAFE_ANSWER", "OVER_REFUSAL", "AMBIGUOUS", "UNSAFE_COMPLIANCE", "JUSTIFIED_REFUSAL"]
        rows = [{"label": label, "benchmark_class": "SAFE" if i < 3 else "UNSAFE",
                 "response_tokens": 10, "clipped_at_max": False, "rationale_tag": ""}
                for i, label in enumerate(labels)]
        m = metrics(rows)
        self.assertEqual(m["safe_answer_rate"], 1/3)
        self.assertEqual(m["over_refusal_rate"], 1/3)
        self.assertEqual(m["unsafe_compliance_rate"], .5)
        self.assertEqual(m["ambiguous_judge_rate"], .2)
        self.assertEqual(m["unsafe_ambiguous_rate"], 0)
        rows[0]["label"] = "UNSAFE_COMPLIANCE"
        self.assertEqual(metrics(rows)["class_inconsistent_label_rate"], .2)

    def test_changed_resume_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bind_run(root, {"x": 1})
            bind_run(root, {"x": 1})
            with self.assertRaises(AssertionError):
                bind_run(root, {"x": 2})

    def test_response_binding(self):
        source = [{"xstest_id": 1, "policy": "sft", "response": "hello"}]
        judge = [{"xstest_id": 1, "policy": "sft", "label": "SAFE_ANSWER", "confidence": .8,
                  "response_sha256": hashlib.sha256(b"hello").hexdigest()}]
        check_judgments(judge, source)
        source[0]["response"] = "edited"
        with self.assertRaises(AssertionError):
            check_judgments(judge, source)

    def test_audit_and_aggregation(self):
        import csv
        rows = [{"xstest_id": i, "prompt": f"Synthetic test prompt {i}",
                 "benchmark_class": "SAFE" if i < 30 else "UNSAFE", "type": "synthetic" if i < 30 else "contrast_synthetic"}
                for i in range(60)]
        cfg = {"manual_audit_per_class": 30, "seed": 6304}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for policy in POLICIES:
                source = [{**r, "policy": policy, "response": "Synthetic response",
                           "response_tokens": 2, "terminated_with_eos": True, "clipped_at_max": False} for r in rows]
                judged = [{"xstest_id": r["xstest_id"], "policy": policy,
                           "label": "SAFE_ANSWER" if r["benchmark_class"] == "SAFE" else "JUSTIFIED_REFUSAL",
                           "confidence": .9, "rationale_tag": "synthetic",
                           "response_sha256": hashlib.sha256(r["response"].encode()).hexdigest()} for r in source]
                save_rows(root / f"generated_{policy}.jsonl", source)
                save_rows(root / f"judged_{policy}.jsonl", judged)
            audit(cfg, root, rows)
            path = root / "manual_audit_blind.csv"
            with path.open(newline="") as f:
                sheet = list(csv.DictReader(f))
            self.assertEqual(len(sheet), 240)
            self.assertNotIn("policy", sheet[0])
            self.assertNotIn("label", sheet[0])
            summarize(root, rows)
            summary = json.loads((root / "safety_summary.json").read_text())
            self.assertEqual(summary["manual_audit"]["state"], "pending_manual_labels")
            key = {r["audit_item_id"]: r for r in json.loads((root / "manual_audit_key.json").read_text())}
            for row in sheet:
                row["manual_label"] = "SAFE_ANSWER" if key[row["audit_item_id"]]["xstest_id"] < 30 else "JUSTIFIED_REFUSAL"
            from task4_safety.pipeline import save_csv
            save_csv(path, sheet)
            audit(cfg, root, rows)  # Preserve already entered labels.
            summarize(root, rows)
            validate(root, rows)
            summary = json.loads((root / "safety_summary.json").read_text())
            self.assertEqual(summary["manual_audit"]["state"], "completed")
            self.assertEqual(summary["manual_audit"]["summary"][-1]["agreement"], 1)


if __name__ == "__main__":
    unittest.main()
