"""Build the public evaluation snapshot from recorded runs; no model calls.

Run from mathpaper/: python evaluation/build_site_report.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import median


METHODS = {
    "bm25": ("BM25", "Keyword matching"),
    "dense": ("Dense", "MiniLM embeddings"),
    "hybrid": ("Hybrid", "BM25 + dense, reciprocal rank fusion"),
    "hybrid-rerank": ("Hybrid + reranking", "Fusion + MiniLM cross-encoder"),
}
REPOSITORY = "https://github.com/chandantankala/Applied-Probability-RAG"
BRANCH = "mathpaper-free-web-demo"


def build_report(base: Path | None = None) -> dict:
    base = base or Path(__file__).resolve().parent
    filenames = ["questions.json", "results.json", "free_tier_results.json",
                 "chat_connection_validation.json", "validation.json"]
    inputs = {name: json.loads((base / name).read_text()) for name in filenames}
    questions, results, free, live, validation = (inputs[name] for name in filenames)
    manifest_path = base.parent / "data" / "papers.json"
    manifest = json.loads(manifest_path.read_text())
    papers = {paper["id"]: paper for paper in manifest}
    if {p["id"] for p in results["corpus"]} != set(papers):
        raise ValueError("Evaluation and deployed corpus differ.")
    for paper in results["corpus"]:
        if any(paper[key] != papers[paper["id"]][key] for key in ("title", "sha256", "pages")):
            raise ValueError("Evaluation and deployed PDF versions differ.")
    corpus = validation["corpus"]
    if (set(papers) != set(corpus["source_sha256"])
            or any(p["sha256"] != corpus["source_sha256"][p["id"]] for p in manifest)
            or corpus["papers"] != len(papers)
            or corpus["pages"] != sum(p["pages"] for p in manifest)):
        raise ValueError("Corpus validation and manifest differ.")
    gold = {q["id"]: q for q in questions}
    if len(gold) != len(questions):
        raise ValueError("Duplicate question IDs.")
    for q in questions:
        if not q["gold_pages"] or any(not 1 <= p <= papers[q["gold_paper"]]["pages"] for p in q["gold_pages"]):
            raise ValueError("Invalid reference page.")

    runs = []
    methods = []
    for mode, (label, description) in METHODS.items():
        selected = [r for r in results["runs"] if r["mode"] == mode]
        if len(selected) != len(gold) or {r["id"] for r in selected} != set(gold):
            raise ValueError("Each method needs exactly one run per question.")
        for run in selected:
            question = gold[run["id"]]
            if len(run["top_sources"]) != results["top_k"]:
                raise ValueError("Missing retained passages.")
            for source in run["top_sources"]:
                if not 1 <= source["page"] <= papers[source["paper_id"]]["pages"]:
                    raise ValueError("Invalid returned page.")
            hit = any(source["paper_id"] == question["gold_paper"]
                      and source["page"] in question["gold_pages"]
                      for source in run["top_sources"])
            if hit != run["page_hit_at_5"]:
                raise ValueError("Recorded page-hit flag disagrees with the returned pages.")
            runs.append(run)
        methods.append({"id": mode, "label": label, "description": description,
                        "hits": sum(r["page_hit_at_5"] for r in selected),
                        "candidate_hits": sum(r["candidate_page_hit_at_50"] for r in selected),
                        "questions": len(selected), "median_ms": round(median(r["timing_ms"] for r in selected), 1)})

    budgets = []
    for candidates in (20, 50):
        selected = [r for r in free["runs"] if r["candidates"] == candidates]
        if len(selected) != len(gold) or {r["id"] for r in selected} != set(gold):
            raise ValueError("Each candidate budget needs the same question set.")
        budgets.append({"candidates": candidates, "hits": sum(r["page_hit_at_5"] for r in selected),
                        "questions": len(selected), "median_ms": median(r["timing_ms"] for r in selected)})
    reduction = 100 * (1 - budgets[0]["median_ms"] / budgets[1]["median_ms"])
    live_checks = [{"kind": c["kind"], "question": c["question"], "answer": c["answer"],
                    "citations": c["citations"], "citation_pages": c["citation_pages"],
                    "tool_calls": c["tool_calls"], "model_calls": c["trace"]["model_calls"],
                    "timing_seconds": c["timing_seconds"]} for c in live["verified_live_checks"]]
    sources = [{"file": name, "sha256": hashlib.sha256((base / name).read_bytes()).hexdigest(),
                "url": f"{REPOSITORY}/blob/{BRANCH}/mathpaper/evaluation/{name}"} for name in filenames]
    sources.append({"file": "data/papers.json", "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                    "url": f"{REPOSITORY}/blob/{BRANCH}/mathpaper/data/papers.json"})
    return {
        "schema_version": 1,
        "scope": results["scope"],
        "metric": "Evidence-page hit@5: at least one labelled physical PDF page from the expected paper appears among the first five returned passages.",
        "repository": f"{REPOSITORY}/tree/{BRANCH}",
        "corpus": {k: corpus[k] for k in ("papers", "pages", "passages")},
        "papers": manifest, "questions": questions, "methods": methods, "runs": runs,
        "retrieval_config": {k: results[k] for k in ("candidates", "top_k", "max_pair_tokens", "model_precision", "python", "platform", "model_lock")},
        "budget_experiment": {"config": {k: free[k] for k in ("precision", "batch_size", "ort_threads", "top_k")},
                              "budgets": budgets, "runs": free["runs"], "median_reduction_percent": round(reduction, 1)},
        "live": {"checked_at_utc": live["checked_at_utc"], "model": live["model"],
                 "code_commit": live["deployed_code_commit"], "checks": live_checks},
        "sources": sources,
    }


def main():
    target = Path(__file__).resolve().parents[1] / "src/mathpaper/web/evaluation.json"
    target.write_text(json.dumps(build_report(), indent=2, ensure_ascii=False) + "\n")
    print(f"Built {target.name} from recorded evaluation inputs.")


if __name__ == "__main__":
    main()
