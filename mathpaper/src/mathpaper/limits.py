"""Small shared quota for the single-instance public demo; no visitor identifiers stored."""
from __future__ import annotations

import os
import sqlite3
import time
import uuid
from pathlib import Path

from .store import data_path


class ChatLimitError(ValueError):
    def __init__(self, message: str, retry_after: int):
        super().__init__(message)
        self.retry_after = retry_after


def daily_limit() -> int:
    return min(100, max(1, int(os.getenv("MATHPAPER_CHAT_DAILY_LIMIT", "20"))))


class PublicChatGuard:
    def __init__(self, root: Path | None = None):
        self.root = root or data_path()

    def reserve(self, now: float | None = None) -> str:
        now = time.time() if now is None else now
        self.root.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.root / "chat-usage.sqlite", timeout=5) as db:
            db.execute("CREATE TABLE IF NOT EXISTS requests(token TEXT PRIMARY KEY,started REAL NOT NULL,lease_until REAL NOT NULL)")
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM requests WHERE started<=?", (now-86400,))
            count, first, last, active = db.execute("SELECT count(*),min(started),max(started),max(lease_until) FROM requests").fetchone()
            if active is not None and active > now:
                raise ChatLimitError("The demo is answering another question. Please try again shortly; Find evidence is available.", 15)
            if count >= daily_limit():
                retry = max(1, int(first+86400-now)+1)
                raise ChatLimitError("This public demo has reached its shared daily answer limit. Please try again later or use Find evidence.", retry)
            cooldown = min(120, max(0, int(os.getenv("MATHPAPER_CHAT_COOLDOWN", "20"))))
            if last is not None and now-last < cooldown:
                raise ChatLimitError("Please wait a few seconds before asking another question. Find evidence remains available.", max(1, int(last+cooldown-now)+1))
            token = uuid.uuid4().hex
            db.execute("INSERT INTO requests VALUES(?,?,?)", (token, now, now+210))
            return token

    def release(self, token: str):
        with sqlite3.connect(self.root / "chat-usage.sqlite", timeout=5) as db:
            db.execute("UPDATE requests SET lease_until=0 WHERE token=?", (token,))
