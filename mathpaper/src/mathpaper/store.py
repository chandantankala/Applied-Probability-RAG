from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path


def data_path() -> Path:
    return Path(os.environ.get("MATHPAPER_DATA", "data")).expanduser().resolve()


def connect(root: Path | None = None) -> sqlite3.Connection:
    root = root or data_path()
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / "corpus.sqlite", timeout=30)
    db.row_factory = sqlite3.Row
    db.executescript("""
        PRAGMA foreign_keys=ON;
        CREATE TABLE IF NOT EXISTS papers(
            id TEXT PRIMARY KEY, title TEXT NOT NULL, filename TEXT NOT NULL,
            sha256 TEXT UNIQUE NOT NULL, pages INTEGER NOT NULL, notes TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS blocks(
            id TEXT PRIMARY KEY, paper_id TEXT NOT NULL REFERENCES papers(id),
            page INTEGER NOT NULL, ordinal INTEGER NOT NULL, label TEXT,
            text TEXT NOT NULL, bbox TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS chunks(
            id TEXT PRIMARY KEY, paper_id TEXT NOT NULL REFERENCES papers(id),
            page INTEGER NOT NULL, ordinal INTEGER NOT NULL, label TEXT,
            text TEXT NOT NULL, block_ids TEXT NOT NULL, tokens INTEGER NOT NULL);
    """)
    return db


def records(sql: str, params: tuple = (), root: Path | None = None) -> list[dict]:
    with connect(root) as db:
        return [dict(row) for row in db.execute(sql, params)]


def papers(root: Path | None = None) -> list[dict]:
    return records("SELECT p.*, (SELECT count(*) FROM chunks c WHERE c.paper_id=p.id) chunks FROM papers p ORDER BY title", root=root)


def chunks(root: Path | None = None) -> list[dict]:
    return records("SELECT c.*, p.title, p.sha256 FROM chunks c JOIN papers p ON p.id=c.paper_id ORDER BY c.paper_id,c.ordinal", root=root)


def corpus_fingerprint(rows: list[dict]) -> str:
    payload = [(r["id"], r["text"], r.get("title", "")) for r in rows]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def source_record(row: dict) -> dict:
    return {
        "source_id": row["id"], "paper_id": row["paper_id"],
        "title": row["title"], "page": row["page"],
        "label": row.get("label"), "text": row["text"],
        "source_sha256": row["sha256"],
        "url": f"/api/papers/{row['paper_id']}/pdf#page={row['page']}",
    }
