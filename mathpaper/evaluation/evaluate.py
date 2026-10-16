"""Small page-hit smoke evaluation, not a held-out retrieval or answer benchmark.

Run from the project directory: python evaluation/evaluate.py
"""
import json
import platform
from pathlib import Path
from statistics import median

from mathpaper.retrieval import ResearchIndex
from mathpaper.store import chunks, papers


def main():
    base = Path(__file__).resolve().parent
    questions = json.loads((base / "questions.json").read_text())
    index = ResearchIndex()
    corpus = {r["id"]: r for r in chunks(index.root)}
    report = {
        "scope": "10 authoring-time seed questions. Page-hit checks only, not answer correctness, recall of all relevant passages, or held-out validation.",
        "python": platform.python_version(), "platform": platform.platform(),
        "candidates": 50, "top_k": 5, "max_pair_tokens": 512,
        "model_precision": index.models.precision,
        "corpus": [{k: p[k] for k in ("id", "title", "sha256", "pages")} for p in papers(index.root)],
        "model_lock": json.loads((Path(__file__).parents[1]/"src/mathpaper"/("model-lock-int8.json" if index.models.precision == "int8" else "model-lock.json")).read_text()),
        "runs": [], "summary": [],
    }

    def relevant(source, question):
        return source["paper_id"] == question["gold_paper"] and source["page"] in question["gold_pages"]

    for mode in ["bm25", "dense", "hybrid", "hybrid-rerank"]:
        runs = []
        for question in questions:
            result = index.search(question["question"], mode=mode, candidates=50, top_k=5)
            run = {"id": question["id"], "mode": mode, "page_hit_at_5": any(relevant(s, question) for s in result["sources"]),
                   "candidate_page_hit_at_50": any(relevant(corpus[k], question) for k in result["candidate_ids"]),
                   "timing_ms": result["timing_ms"],
                   "top_sources": [{k:s[k] for k in ("source_id", "paper_id", "page", "label", "score")} for s in result["sources"]]}
            runs.append(run)
            print(mode, question["id"], "HIT" if run["page_hit_at_5"] else "MISS", result["timing_ms"], flush=True)
        report["runs"].extend(runs)
        report["summary"].append({"mode": mode, "page_hits_at_5": sum(r["page_hit_at_5"] for r in runs),
                                  "candidate_page_hits_at_50": sum(r["candidate_page_hit_at_50"] for r in runs),
                                  "questions": len(runs), "median_ms": round(median(r["timing_ms"] for r in runs), 1)})
    (base / "results.json").write_text(json.dumps(report, indent=2))
    lines = ["# Retrieval smoke check", "", report["scope"], "",
             "The gold labels identify physical PDF pages containing useful evidence. A hit means at least one such page appears among the top five passages. It does not establish that the complete statement or all assumptions were retrieved.", "",
             "All questions search all three papers. N=50 candidates; k=5 retained; reranker pair budget=512 tokens. The CPU inference path is real ONNX inference. Timings include lazy startup on the first relevant query and are illustrative measurements from this environment.", "",
             "| Method | Page hits in top 5 | Page hits among 50 candidates | Median query time |",
             "|---|---:|---:|---:|"]
    for row in report["summary"]:
        lines.append(f"| {row['mode']} | {row['page_hits_at_5']}/10 | {row['candidate_page_hits_at_50']}/10 | {row['median_ms']} ms |")
    lines += ["", "## Queries missed by hybrid + reranking", ""]
    missed = [r["id"] for r in report["runs"] if r["mode"] == "hybrid-rerank" and not r["page_hit_at_5"]]
    lines += [f"- {key}" for key in missed] or ["None in this small seed set."]
    lines += ["", "These questions were created during development. Do not report the results as a general RAG benchmark. Next, label additional unseen questions at passage level, include unanswerable and multi-paper questions, and separately evaluate support, citation correctness, assumptions, and latency percentiles. No live generative model was used in this retrieval check.", ""]
    (base / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
