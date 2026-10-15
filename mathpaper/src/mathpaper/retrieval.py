from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .models import NeuralModels
from .store import chunks, connect, corpus_fingerprint, data_path, records, source_record


def terms(text: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:[.-][^\W_]+)*", text.casefold(), flags=re.UNICODE)


def bm25_scores(query: str, documents: list[str], k1=1.5, b=.75) -> list[float]:
    counts = [Counter(terms(doc)) for doc in documents]
    if not counts:
        return []
    lengths = [sum(c.values()) for c in counts]
    avg = sum(lengths)/len(lengths) or 1
    scores = [0.] * len(counts)
    for term in set(terms(query)):
        df = sum(term in counter for counter in counts)
        idf = math.log(1+(len(counts)-df+.5)/(df+.5))
        for i, counter in enumerate(counts):
            tf = counter[term]
            scores[i] += idf * tf * (k1+1) / (tf+k1*(1-b+b*lengths[i]/avg))
    return scores


def fuse_rankings(rankings: list[list[str]], constant=60) -> list[tuple[str, float]]:
    scores = Counter()
    for ranking in rankings:
        for rank, key in enumerate(dict.fromkeys(ranking), 1):
            scores[key] += 1/(constant+rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


class ResearchIndex:
    def __init__(self, root: Path | None = None):
        self.root = root or data_path()
        self.models = NeuralModels(self.root)
        self.vector_path = self.root / ("vectors-int8.npz" if self.models.precision == "int8" else "vectors.npz")
        self._lock = threading.RLock()
        self._signature = None
        self._vectors = None

    def build_vectors(self):
        rows = chunks(self.root)
        if not rows:
            raise ValueError("Ingest papers and build chunks first.")
        signature = corpus_fingerprint(rows)+":"+self.models.signature("embedding")
        vectors = self.models.embed([r["text"] for r in rows])
        with self.vector_path.open("wb") as handle:
            np.savez_compressed(handle, ids=np.array([r["id"] for r in rows]), vectors=vectors, signature=np.array(signature))
        self._signature, self._vectors = signature, vectors
        return len(rows)

    def _load_vectors(self, rows):
        signature = corpus_fingerprint(rows)+":"+self.models.signature("embedding")
        if signature == self._signature:
            return self._vectors
        path = self.vector_path
        if not path.exists():
            raise ValueError("The embedding index is missing. Run mathpaper index.")
        with np.load(path, allow_pickle=False) as saved:
            if saved["signature"].item() != signature or saved["ids"].tolist() != [r["id"] for r in rows]:
                raise ValueError("The embedding index is stale. Run mathpaper index.")
            self._vectors = saved["vectors"].copy()
        self._signature = signature
        return self._vectors

    def search(self, query: str, paper_ids=None, candidates=50, top_k=5, mode="hybrid-rerank", max_length=512):
        query = query.strip()
        if not query:
            raise ValueError("Enter a research question.")
        if not 1 <= top_k <= candidates <= 150:
            raise ValueError("Require 1 <= top_k <= candidates <= 150.")
        limit = min(150, max(1, int(os.getenv("MATHPAPER_MAX_CANDIDATES", "150"))))
        if candidates > limit:
            raise ValueError(f"This deployment supports at most {limit} candidates per request.")
        if mode not in {"bm25", "dense", "hybrid", "hybrid-rerank"}:
            raise ValueError("Unknown retrieval mode.")
        started = time.perf_counter()
        with self._lock:
            all_rows = chunks(self.root)
            rows = [r for r in all_rows if not paper_ids or r["paper_id"] in paper_ids]
            if not rows:
                return {"sources": [], "timing_ms": 0, "mode": mode, "candidates_scored": 0}
            lookup = {r["id"]: r for r in rows}
            lexical = bm25_scores(query, [r["text"] for r in rows])
            lexical_order = [rows[i]["id"] for i in sorted(range(len(rows)), key=lambda i: (-lexical[i], rows[i]["id"])) if lexical[i] > 0]
            if mode == "bm25":
                candidates_list = lexical_order[:candidates]
                fused = {key: lexical[[r["id"] for r in rows].index(key)] for key in candidates_list}
            else:
                vectors = self._load_vectors(all_rows)
                mask = [i for i, r in enumerate(all_rows) if r["id"] in lookup]
                query_vector = self.models.embed([query])[0]
                cosine = vectors[mask] @ query_vector
                dense_order = [rows[i]["id"] for i in np.argsort(-cosine, kind="stable")[:candidates]]
                if mode == "dense":
                    candidates_list = dense_order
                    fused = {rows[i]["id"]: float(cosine[i]) for i in range(len(rows))}
                else:
                    fusion = fuse_rankings([lexical_order[:candidates], dense_order])[:candidates]
                    candidates_list, fused = [key for key, _ in fusion], dict(fusion)
            scored = [(key, fused[key]) for key in candidates_list]
            pre_rerank = [key for key, _ in scored]
            if mode == "hybrid-rerank" and scored:
                values = self.models.rerank(query, [lookup[key]["text"] for key in candidates_list], max_length=max_length)
                scored = sorted(zip(candidates_list, values), key=lambda pair: (-pair[1], pair[0]))
            sources = []
            for key, score in scored[:top_k]:
                result = source_record(lookup[key])
                result.update({"score": score, "fusion_score": fused[key]})
                sources.append(result)
            return {
                "sources": sources, "mode": mode,
                "candidates_scored": len(scored) if mode == "hybrid-rerank" else 0,
                "candidate_ids": pre_rerank,
                "timing_ms": round((time.perf_counter()-started)*1000, 1),
                "score_note": "Reranker logits and retrieval scores measure relevance, not factual confidence.",
            }

    def get_statement(self, paper_id: str, label: str) -> dict:
        hits = records("SELECT b.*, p.title, p.sha256 FROM blocks b JOIN papers p ON p.id=b.paper_id WHERE b.paper_id=? AND lower(b.label)=lower(?) ORDER BY b.ordinal", (paper_id, label.strip()), self.root)
        if not hits:
            return {"sources": [], "missing": f"No indexed block labelled {label!r} in this uploaded PDF. Try search_passages or inspect its page."}
        expanded = []
        truncated = False
        for hit in hits[:5]:
            # Formula fragments may span many blocks and continue on the next page.
            neighbors = records("SELECT b.*,p.title,p.sha256 FROM blocks b JOIN papers p ON p.id=b.paper_id WHERE b.paper_id=? AND b.ordinal>=? ORDER BY b.ordinal LIMIT 81", (paper_id, hit["ordinal"]), self.root)
            characters = 0
            for position, row in enumerate(neighbors):
                if position and (row["label"] or re.match(r"^Proof\b", row["text"])):
                    break
                if position >= 80 or characters+len(row["text"]) > 16000:
                    truncated = True
                    break
                expanded.append(source_record(row))
                characters += len(row["text"])
        return {"sources": list({r["source_id"]: r for r in expanded}.values()), "truncated": truncated, "note": "Label-to-boundary context, not a certified statement parse. It may include surrounding prose or figures. Check original pages for formula layout and completeness."}

    def get_context(self, source_id: str) -> dict:
        rows = records("SELECT c.*,p.title,p.sha256 FROM chunks c JOIN papers p ON p.id=c.paper_id WHERE c.id=?", (source_id,), self.root)
        if rows:
            item = rows[0]
            blocks = json.loads(item["block_ids"])
            anchor = records("SELECT * FROM blocks WHERE id=?", (blocks[0],), self.root)[0]
            last = records("SELECT * FROM blocks WHERE id=?", (blocks[-1],), self.root)[0]
        else:
            anchors = records("SELECT * FROM blocks WHERE id=?", (source_id,), self.root)
            if not anchors:
                return {"sources": [], "missing": "Unknown source ID."}
            anchor = anchors[0]
            last = anchor
        nearby = records("SELECT b.*,p.title,p.sha256 FROM blocks b JOIN papers p ON p.id=b.paper_id WHERE b.paper_id=? AND b.ordinal BETWEEN ? AND ? ORDER BY b.ordinal LIMIT 100", (anchor["paper_id"], max(0,anchor["ordinal"]-1), last["ordinal"]+4), self.root)
        refs = referenced_labels(" ".join(r["text"] for r in nearby))
        return {"sources": [source_record(r) for r in nearby], "referenced_labels": refs, "note": "References are extracted from text and must be resolved explicitly with get_statement."}


def referenced_labels(text: str) -> list[str]:
    number = r"(?:[A-Z]\.)?\d+(?:\.\d+)*"
    pattern = rf"\b(Assumptions?|Definitions?|Lemmas?|Theorems?|Propositions?)\s+({number}(?:\s*(?:,\s*(?:and\s+)?|and\s+){number})*)"
    labels = set()
    for match in re.finditer(pattern, text):
        kind = match.group(1).removesuffix("s")
        labels.update(f"{kind} {n}" for n in re.findall(number, match.group(2)))
    return sorted(labels)
