"""Published statistics must remain traceable to the recorded question runs."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from statistics import median

from evaluation.build_site_report import build_report


BASE = Path(__file__).resolve().parents[1]


class EvaluationReportTests(unittest.TestCase):
    def test_aggregates_use_recorded_runs_and_labelled_pages(self):
        report = build_report()
        questions = {q["id"]: q for q in report["questions"]}
        for method in report["methods"]:
            runs = [r for r in report["runs"] if r["mode"] == method["id"]]
            expected_hits = sum(any(s["paper_id"] == questions[r["id"]]["gold_paper"]
                                    and s["page"] in questions[r["id"]]["gold_pages"]
                                    for s in r["top_sources"]) for r in runs)
            self.assertEqual(method["hits"], expected_hits)
            self.assertEqual(method["median_ms"], round(median(r["timing_ms"] for r in runs), 1))
        budget = report["budget_experiment"]
        self.assertEqual([b["hits"] for b in budget["budgets"]], [9, 9])
        self.assertAlmostEqual(budget["budgets"][0]["median_ms"], 1164.5)
        self.assertAlmostEqual(budget["budgets"][1]["median_ms"], 2903.25)
        self.assertEqual(budget["median_reduction_percent"], 59.9)

    def changed_input(self, filename, change):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        shutil.copytree(BASE / "evaluation", root / "evaluation", ignore=shutil.ignore_patterns("__pycache__"))
        (root / "data").mkdir()
        shutil.copyfile(BASE / "data/papers.json", root / "data/papers.json")
        path = root / filename
        data = json.loads(path.read_text())
        change(data)
        path.write_text(json.dumps(data))
        return root / "evaluation"

    def test_rejects_a_hit_flag_unsupported_by_returned_pages(self):
        base = self.changed_input("evaluation/results.json", lambda d: d["runs"][0].update(page_hit_at_5=False))
        with self.assertRaisesRegex(ValueError, "page-hit flag"):
            build_report(base)

    def test_rejects_statistics_for_a_different_pdf_version(self):
        base = self.changed_input("data/papers.json", lambda d: d[0].update(sha256="different-version"))
        with self.assertRaisesRegex(ValueError, "PDF versions"):
            build_report(base)

    def test_rejects_unpaired_candidate_budget_results(self):
        base = self.changed_input("evaluation/free_tier_results.json", lambda d: d["runs"].pop())
        with self.assertRaisesRegex(ValueError, "same question set"):
            build_report(base)

    def test_shipped_snapshot_matches_current_inputs(self):
        shipped = json.loads((BASE / "src/mathpaper/web/evaluation.json").read_text())
        self.assertEqual(shipped, build_report())


if __name__ == "__main__":
    unittest.main()
