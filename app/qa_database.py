"""SQLite-based Q&A persistence, independent of Neo4j/Milvus."""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "qa_history.db"


class QADatabase:
    def __init__(self, db_path: str = ""):
        self.db_path = db_path or str(DB_PATH)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    closed_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS qa_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    strategy TEXT,
                    complexity REAL,
                    routing_reasoning TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_qa_session ON qa_records(session_id)
            """)

    def create_session(self, session_id: str):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, created_at) VALUES (?, ?)",
                (session_id, time.strftime("%Y-%m-%dT%H:%M:%S")),
            )
        logger.info(f"Session created in SQLite: {session_id}")

    def close_session(self, session_id: str):
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET closed_at = ? WHERE id = ?",
                (time.strftime("%Y-%m-%dT%H:%M:%S"), session_id),
            )

    def record_qa(
        self,
        session_id: str,
        question: str,
        answer: str,
        strategy: str = "unknown",
        complexity: float = 0.0,
        routing_reasoning: str = "",
    ):
        with self._connect() as conn:
            max_turn = conn.execute(
                "SELECT COALESCE(MAX(turn_index), -1) FROM qa_records WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO qa_records (session_id, turn_index, question, answer, strategy, complexity, routing_reasoning, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, max_turn + 1, question, answer, strategy, complexity, routing_reasoning, time.strftime("%Y-%m-%dT%H:%M:%S")),
            )
        logger.info(f"QA recorded in SQLite: {session_id} turn {max_turn + 1}")

    def list_sessions(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT s.id, s.created_at, s.closed_at, COUNT(q.id) AS turn_count "
                "FROM sessions s LEFT JOIN qa_records q ON s.id = q.session_id "
                "GROUP BY s.id ORDER BY s.created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [{"id": r["id"], "start_time": r["created_at"], "end_time": r["closed_at"] or "", "turn_count": r["turn_count"]} for r in rows]

    def get_session_detail(self, session_id: str) -> Optional[dict]:
        with self._connect() as conn:
            ses = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if not ses:
                return None
            turns = conn.execute(
                "SELECT question, answer, strategy, complexity, routing_reasoning, created_at "
                "FROM qa_records WHERE session_id = ? ORDER BY turn_index",
                (session_id,),
            ).fetchall()
            return {
                "id": ses["id"],
                "start_time": ses["created_at"],
                "end_time": ses["closed_at"] or "",
                "turns": [{
                    "question": t["question"],
                    "answer": t["answer"],
                    "strategy": t["strategy"] or "",
                    "complexity": t["complexity"] or 0,
                    "timestamp": t["created_at"],
                } for t in turns],
            }

    def delete_session(self, session_id: str):
        with self._connect() as conn:
            conn.execute("DELETE FROM qa_records WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        logger.info(f"Session deleted from SQLite: {session_id}")
