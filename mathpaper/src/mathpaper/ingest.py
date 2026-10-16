from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import fitz

from .store import connect, data_path

LABEL = re.compile(r"^(Theorem|Lemma|Proposition|Corollary|Definition|Assumption|Remark)\s+((?:[A-Z]\.)?\d+(?:\.\d+)*|[A-Z])\b", re.I)
DECLARATION = re.compile(
    r"^(?:Theorem|Lemma|Proposition|Corollary|Definition|Assumption|Remark)\s+"
    r"(?:(?:[A-Z]\.)?\d+(?:\.\d+)*|[A-Z])(?=\s*[.(])",
    re.I | re.M,
)


def split_declarations(text: str) -> list[str]:
    """Separate headings that the PDF grouped with a preceding equation.

    Require declaration punctuation after the number so ordinary references
    such as 'Theorem 1 guarantees ...' do not create statement boundaries.
    """
    starts = [0] + [m.start() for m in DECLARATION.finditer(text) if m.start()]
    ends = starts[1:] + [len(text)]
    return [text[start:end].strip() for start, end in zip(starts, ends) if text[start:end].strip()]


def detect_label(text: str) -> str | None:
    match = LABEL.match(text)
    if not match:
        return None
    remainder = text[match.end():].lstrip()
    if remainder.startswith(",") or re.match(r"(?:to|with|guarantees|extends|proves|is|establishes)\b", remainder, re.I):
        return None
    return f"{match.group(1).title()} {match.group(2)}"


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\ufffd", " ")
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def ingest_pdf(path: Path, title: str | None = None, notes: str = "", root: Path | None = None) -> dict:
    """Preserve exact PDF bytes and extract page-linked text blocks.

    This is a layout heuristic, not mathematical OCR. Original pages remain
    authoritative. Formula glyphs/ordering need manual inspection.
    """
    root = root or data_path()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    paper_id = digest[:16]
    with connect(root) as db:
        old = db.execute("SELECT * FROM papers WHERE sha256=?", (digest,)).fetchone()
        if old:
            return {**dict(old), "already_ingested": True}
    extracted = []
    with fitz.open(path) as document:
        if document.needs_pass:
            raise ValueError("The PDF is encrypted; provide an unlocked copy.")
        if not len(document):
            raise ValueError("The PDF has no pages.")
        paper_title = title or document.metadata.get("title") or path.stem
        raw_pages = [page.get_text("blocks", sort=False) for page in document]
        repeated_headers = Counter()
        for page, blocks in zip(document, raw_pages):
            for block in blocks:
                if block[6] == 0 and block[1] < 62:
                    text = clean_text(block[4])
                    if len(text) > 15:
                        repeated_headers[text] += 1
        for page_no, (page, blocks) in enumerate(zip(document, raw_pages), 1):
            kept = []
            for block in blocks:
                x0, y0, x1, y1, text, _, kind = block[:7]
                if kind != 0:
                    continue
                text = clean_text(text)
                if not text:
                    continue
                # Review line-number columns, not isolated numbers in mathematics.
                if x1 < page.rect.width * .085 and all(re.fullmatch(r"\d{2,4}", v.strip()) for v in text.splitlines() if v.strip()):
                    continue
                if y0 > page.rect.height * .91 and re.fullmatch(r"\d+", text):
                    continue
                if y0 < 62 and repeated_headers[text] >= max(3, len(document) // 3):
                    continue
                kept.append((x0, y0, x1, y1, text))
            # A right-column paragraph is distinguished from centered equations.
            two_columns = sum(1 for b in kept if b[0] > page.rect.width * .47 and b[2]-b[0] > page.rect.width * .23 and len(b[4]) > 100) >= 2
            if two_columns:
                full = [b for b in kept if b[2]-b[0] > page.rect.width * .65]
                top_end = min((b[1] for b in kept if b not in full), default=100)
                top = [b for b in full if b[1] < top_end]
                remaining = [b for b in kept if b not in top]
                # Keep full-width captions separate; preserve left-then-right flow.
                left = [b for b in remaining if b[0] < page.rect.width * .49]
                right = [b for b in remaining if b not in left]
                ordered = sorted(top, key=lambda b: b[1]) + sorted(left, key=lambda b: (b[1], b[0])) + sorted(right, key=lambda b: (b[1], b[0]))
            else:
                ordered = sorted(kept, key=lambda b: (b[1], b[0]))
            ordered = [(*block[:4], part) for block in ordered for part in split_declarations(block[4])]
            for order, block in enumerate(ordered):
                label = detect_label(block[4])
                extracted.append((f"{paper_id}:p{page_no}:b{order}", paper_id, page_no, len(extracted), label, block[4], json.dumps(block[:4])))
        page_count = len(document)
    if not extracted:
        raise ValueError("No readable text was found. OCR or a different PDF is needed.")
    destination = root / "papers" / f"{paper_id}.pdf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if path.resolve() != destination.resolve():
        shutil.copyfile(path, destination)
    with connect(root) as db:
        db.execute("INSERT INTO papers VALUES(?,?,?,?,?,?)", (paper_id, paper_title, path.name, digest, page_count, notes))
        db.executemany("INSERT INTO blocks VALUES(?,?,?,?,?,?,?)", extracted)
    return {"id": paper_id, "title": paper_title, "pages": page_count, "blocks": len(extracted), "sha256": digest, "already_ingested": False}


def build_chunks(tokenizer, root: Path | None = None, budget: int = 220, overlap: int = 24) -> int:
    """Create page-contained, label-aware chunks using the model's tokenizer.

    Long mathematical blocks are split and linked to their original block.
    The complete original block remains available through get_statement.
    """
    if budget < 32 or overlap < 0 or overlap >= budget:
        raise ValueError("Choose a budget >=32 and 0 <= overlap < budget.")
    root = root or data_path()
    tokenizer.no_truncation()
    tokenizer.no_padding()
    with connect(root) as db:
        blocks = [dict(r) for r in db.execute("SELECT * FROM blocks ORDER BY paper_id,ordinal")]
        output = []
        pending, pending_ids, pending_page, pending_paper, pending_label = [], [], None, None, None

        def emit():
            if not pending:
                return
            text = "\n\n".join(pending)
            count = len(tokenizer.encode(text, add_special_tokens=False).ids)
            key = hashlib.sha256((pending_paper + str(pending_page) + json.dumps(pending_ids) + text).encode()).hexdigest()[:20]
            output.append((key, pending_paper, pending_page, len(output), pending_label, text, json.dumps(pending_ids), count))

        for b in blocks:
            text = b["text"]
            encoded = tokenizer.encode(text, add_special_tokens=False)
            # Token offsets preserve original text and mathematical symbols.
            windows = []
            for start in range(0, len(encoded.ids), budget-overlap):
                end = min(start+budget, len(encoded.ids))
                lo = encoded.offsets[start][0]
                hi = encoded.offsets[end-1][1]
                windows.append(text[lo:hi])
                if end == len(encoded.ids):
                    break
            for segment in windows:
                candidate = "\n\n".join(pending+[segment])
                too_long = len(tokenizer.encode(candidate, add_special_tokens=False).ids) > budget
                new_boundary = b["paper_id"] != pending_paper or b["page"] != pending_page or (b["label"] and pending)
                if pending and (too_long or new_boundary):
                    emit()
                    pending, pending_ids, pending_label = [], [], None
                pending_paper, pending_page = b["paper_id"], b["page"]
                pending.append(segment)
                pending_ids.append(b["id"])
                pending_label = pending_label or b["label"]
        emit()
        db.execute("DELETE FROM chunks")
        db.executemany("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?)", output)
    for filename in ("vectors.npz", "vectors-int8.npz"):
        (root / filename).unlink(missing_ok=True)
    return len(output)
