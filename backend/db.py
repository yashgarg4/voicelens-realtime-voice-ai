"""SQLite session storage (stdlib sqlite3 — no extra dependency).

One row per interview attempt: the question asked, timestamps, and the parsed
Content/Depth/Structure scores. Persists across restarts so the dashboard's
ScoreHistory survives a reload. All functions are synchronous; the FastAPI
routes call them via `asyncio.to_thread` so they don't block the event loop.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent / "voicelens.db"

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def _connect() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
    return _conn


def init_db() -> None:
    with _lock:
        conn = _connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                question_id INTEGER,
                category    TEXT,
                question    TEXT,
                started_at  TEXT,
                ended_at    TEXT,
                content     INTEGER,
                depth       INTEGER,
                structure   INTEGER,
                overall     REAL,
                summary     TEXT
            )
            """
        )
        conn.commit()


def create_session(
    session_id: str,
    question_id: int,
    category: str,
    question: str,
    started_at: str,
) -> None:
    with _lock:
        conn = _connect()
        conn.execute(
            "INSERT OR REPLACE INTO sessions "
            "(id, question_id, category, question, started_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, question_id, category, question, started_at),
        )
        conn.commit()


def update_scores(session_id: str, scores: dict) -> None:
    with _lock:
        conn = _connect()
        conn.execute(
            "UPDATE sessions SET content=?, depth=?, structure=?, overall=?, "
            "summary=? WHERE id=?",
            (
                scores["content"],
                scores["depth"],
                scores["structure"],
                scores["overall"],
                scores["summary"],
                session_id,
            ),
        )
        conn.commit()


def end_session(session_id: str, ended_at: str) -> None:
    with _lock:
        conn = _connect()
        conn.execute(
            "UPDATE sessions SET ended_at=? WHERE id=?", (ended_at, session_id)
        )
        conn.commit()


def list_sessions(limit: int = 100) -> list[dict]:
    """Most-recent-first sessions that have at least started."""
    with _lock:
        conn = _connect()
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
