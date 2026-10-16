"""Deterministic safeguards. No model downloads or live LLM calls required."""
import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz
import numpy as np

from mathpaper.agent import check_claims, conversation, valid_citations
from mathpaper.ingest import detect_label, ingest_pdf
from mathpaper.retrieval import ResearchIndex, bm25_scores, fuse_rankings, referenced_labels
from mathpaper.store import chunks, connect, source_record


class ReplyModel:
    """A test fixture, never a substitute for the production ChatModel."""
    def __init__(self, verdicts):
        self.verdicts = verdicts
        self.calls = 0

    async def complete(self, *args, **kwargs):
        self.calls += 1
        return {"content": json.dumps({"verdicts": self.verdicts})}


class RetrievalTests(unittest.TestCase):
    def test_rrf_rewards_complementary_hits_without_duplicate_votes(self):
        result = fuse_rankings([["a", "a", "b"], ["b", "c"]])
        self.assertEqual(result[0][0], "b")
        self.assertAlmostEqual(dict(result)["a"], 1/61)
        self.assertAlmostEqual(dict(result)["b"], 1/62+1/61)

    def test_bm25_exact_term_and_empty_corpus(self):
        result = bm25_scores("hypercube", ["Gaussian measures", "hypercube Markov chain"])
        self.assertGreater(result[1], result[0])
        self.assertEqual(bm25_scores("anything", []), [])

    def test_labels_and_mentions_are_distinct(self):
        self.assertEqual(detect_label("Theorem 2 (Retrieval Guarantee). Let X..."), "Theorem 2")
        self.assertEqual(detect_label("Theorem\n2\n(Retrieval Guarantee)."), "Theorem 2")
        for text in ["Theorem 2 guarantees convergence.", "Lemma 16 to conclude", "Theorem 2, proved in Appendix A"]:
            self.assertIsNone(detect_label(text))

    def test_plural_assumption_references(self):
        self.assertEqual(referenced_labels("Theorem 10 uses Assumptions 4, 5 and 6 and Lemma 9."), ["Assumption 4", "Assumption 5", "Assumption 6", "Lemma 9", "Theorem 10"])

    def test_ingestion_is_content_addressed_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            pdf = base / "sample.pdf"
            with fitz.open() as document:
                document.new_page().insert_text((72, 72), "Theorem 1. A sample statement.")
                document.save(pdf)
            root = base / "data"
            first = ingest_pdf(pdf, title="Sample", root=root)
            again = ingest_pdf(pdf, root=root)
            self.assertEqual(first["id"], hashlib.sha256(pdf.read_bytes()).hexdigest()[:16])
            self.assertTrue(again["already_ingested"])
            self.assertEqual((root / "papers" / (first["id"]+".pdf")).read_bytes(), pdf.read_bytes())
            with connect(root) as db:
                block = dict(db.execute("SELECT * FROM blocks").fetchone())
                self.assertEqual(block["page"], 1)
                self.assertEqual(block["label"], "Theorem 1")

    def _corpus(self, root):
        with connect(root) as db:
            for key in ("a", "b"):
                db.execute("INSERT INTO papers VALUES(?,?,?,?,?,?)", (key, key, key+".pdf", key*64, 1, ""))
                # More than four formula blocks catches accidental fixed-size expansion.
                for order in range(8):
                    label = "Theorem 1" if order == 0 else "Lemma 2" if order == 7 else None
                    text = f"{label or 'formula'} hypercube {key} fragment {order}"
                    db.execute("INSERT INTO blocks VALUES(?,?,?,?,?,?,?)", (f"{key}:p1:b{order}", key, 1, order, label, text, "[]"))
                db.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?)", (key+"chunk", key, 1, 0, "Theorem 1", f"hypercube chain {key}", json.dumps([f"{key}:p1:b0", f"{key}:p1:b5"]), 5))

    def test_theorem_after_formula_has_its_own_statement_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            pdf = base / "merged-heading.pdf"
            with fitz.open() as document:
                page = document.new_page()
                page.insert_text((72, 72), "Assumption 1. The parameter is positive.\n"
                                 "2 log(4 pi lambda) + d .\n"
                                 "Theorem 1. The chain converges.\n"
                                 "Definition 2. A distinct statement.")
                document.save(pdf)
            root = base / "data"
            paper = ingest_pdf(pdf, root=root)
            index = ResearchIndex(root)
            theorem = index.get_statement(paper["id"], "Theorem 1")
            text = "\n".join(s["text"] for s in theorem["sources"])
            self.assertIn("The chain converges.", text)
            self.assertNotIn("2 log", text)
            self.assertNotIn("Definition 2", text)
            self.assertNotIn("Theorem 1", "\n".join(s["text"] for s in index.get_statement(paper["id"], "Assumption 1")["sources"]))

    def test_paper_filter_and_statement_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._corpus(root)
            index = ResearchIndex(root)
            result = index.search("hypercube", paper_ids=["b"], mode="bm25", top_k=1)
            self.assertEqual({s["paper_id"] for s in result["sources"]}, {"b"})
            statement = index.get_statement("a", "Theorem 1")
            self.assertEqual(len(statement["sources"]), 7)
            self.assertFalse(statement["truncated"])
            self.assertEqual({s["paper_id"] for s in statement["sources"]}, {"a"})
            context = index.get_context("achunk")
            self.assertIn("a:p1:b5", {s["source_id"] for s in context["sources"]})

    def test_stale_vectors_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._corpus(root)
            np.savez(ResearchIndex(root).vector_path, ids=np.array(["achunk", "bchunk"]), vectors=np.ones((2, 3)), signature=np.array("stale"))
            with self.assertRaisesRegex(ValueError, "stale"):
                ResearchIndex(root)._load_vectors(chunks(root))

    def test_int8_cache_does_not_reuse_full_precision_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict('os.environ', {'MATHPAPER_MODEL_PRECISION': 'int8'}):
                index = ResearchIndex(root)
                self.assertEqual(index.vector_path.name, 'vectors-int8.npz')
                self.assertEqual(index.models.directory.name, 'models-int8')
            with patch.dict('os.environ', {'MATHPAPER_MODEL_PRECISION': 'fp32'}):
                self.assertEqual(ResearchIndex(root).vector_path.name, 'vectors.npz')

    def test_pdf_link_uses_physical_page(self):
        source = source_record({"id": "source", "paper_id": "a", "title": "A", "page": 12, "text": "text", "sha256": "x"})
        self.assertEqual(source["url"], "/api/papers/a/pdf#page=12")


class CitationGateTests(unittest.TestCase):
    def setUp(self):
        self.evidence = {"real": {"text": "The chain has a cutoff."}}
        self.claim = {"text": "The chain has a cutoff.", "source_ids": ["real"]}

    def test_invented_citation_never_reaches_model(self):
        model = ReplyModel([{"index": 0, "status": "supported"}])
        fake = {"text": "An assertion", "source_ids": ["invented"]}
        self.assertFalse(valid_citations(fake, self.evidence))
        accepted, rejected = asyncio.run(check_claims([fake], self.evidence, model))
        self.assertEqual((accepted, rejected, model.calls), ([], [0], 0))

    def test_missing_duplicate_and_negative_verdicts_fail_closed(self):
        cases = [[], [{"index": 0, "status": "insufficient"}], [{"index": 0, "status": "contradicted"}], [{"index": 0, "status": "supported"}, {"index": 0, "status": "supported"}]]
        for verdicts in cases:
            with self.subTest(verdicts=verdicts):
                accepted, rejected = asyncio.run(check_claims([self.claim], self.evidence, ReplyModel(verdicts)))
                self.assertEqual((accepted, rejected), ([], [0]))

    def test_only_supported_claim_is_published(self):
        other = {"text": "A stronger unsupported claim.", "source_ids": ["real"]}
        model = ReplyModel([{"index": 0, "status": "supported"}, {"index": 1, "status": "insufficient"}])
        self.assertEqual(asyncio.run(check_claims([self.claim, other], self.evidence, model)), ([self.claim], [1]))

    def test_history_is_separate_per_thread_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conversation(root, "a", [{"role": "user", "content": str(i)} for i in range(20)])
            self.assertEqual(len(conversation(root, "a")), 12)
            self.assertEqual(conversation(root, "b"), [])


if __name__ == "__main__":
    unittest.main()
