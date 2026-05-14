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
                CREATE TABLE IF NOT EXISTS patients (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    access_code TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    closed_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS qa_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    client_id TEXT NOT NULL DEFAULT '',
                    turn_index INTEGER NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    strategy TEXT,
                    complexity REAL,
                    routing_reasoning TEXT,
                    retrieved_docs TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_qa_session ON qa_records(session_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_qa_client ON qa_records(client_id)
            """)
            # Migrations
            for col, tbl in [("retrieved_docs", "qa_records"), ("client_id", "sessions"), ("client_id", "qa_records")]:
                try:
                    conn.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} TEXT DEFAULT ''")
                except sqlite3.OperationalError:
                    pass

    # ── Patient management ──────────────────────────────────────────

    def register_patient(self, name: str, access_code: str = "") -> dict:
        import hashlib
        pid = "p_" + hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()[:8]
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO patients (id, name, access_code, created_at) VALUES (?, ?, ?, ?)",
                (pid, name, access_code, time.strftime("%Y-%m-%dT%H:%M:%S")),
            )
        logger.info(f"Patient registered: {pid} ({name})")
        return {"patient_id": pid, "name": name}

    def list_patients(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, name, created_at FROM patients ORDER BY created_at DESC LIMIT 50"
            ).fetchall()
        return [{"patient_id": r["id"], "name": r["name"], "created_at": r["created_at"]} for r in rows]

    def verify_patient(self, patient_id: str, access_code: str = "") -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, name FROM patients WHERE id = ? AND access_code = ?",
                (patient_id, access_code),
            ).fetchone()
        if row:
            return {"patient_id": row["id"], "name": row["name"]}
        return None

    # ── Session management ──────────────────────────────────────────

    def create_session(self, session_id: str, client_id: str = ""):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, client_id, created_at) VALUES (?, ?, ?)",
                (session_id, client_id, time.strftime("%Y-%m-%dT%H:%M:%S")),
            )
        logger.info(f"Session created in SQLite: {session_id} (client={client_id})")

    def ensure_session_client(self, session_id: str, client_id: str):
        """Bind an existing session to a client_id if not already bound."""
        if not client_id:
            return
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET client_id = ? WHERE id = ? AND client_id = ''",
                (client_id, session_id),
            )

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
        retrieved_docs: str = "",
        client_id: str = "",
    ):
        with self._connect() as conn:
            max_turn = conn.execute(
                "SELECT COALESCE(MAX(turn_index), -1) FROM qa_records WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO qa_records (session_id, client_id, turn_index, question, answer, strategy, complexity, routing_reasoning, retrieved_docs, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, client_id, max_turn + 1, question, answer, strategy, complexity, routing_reasoning, retrieved_docs, time.strftime("%Y-%m-%dT%H:%M:%S")),
            )
        logger.info(f"QA recorded in SQLite: {session_id} turn {max_turn + 1} (client={client_id})")

    def list_sessions(self, client_id: str = "", limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT s.id, s.created_at, s.closed_at, COUNT(q.id) AS turn_count "
                "FROM sessions s LEFT JOIN qa_records q ON s.id = q.session_id "
                "WHERE s.client_id = ? "
                "GROUP BY s.id ORDER BY s.created_at DESC LIMIT ?",
                (client_id, limit),
            ).fetchall()
            return [{"id": r["id"], "start_time": r["created_at"], "end_time": r["closed_at"] or "", "turn_count": r["turn_count"]} for r in rows]

    def get_session_detail(self, session_id: str, client_id: str = "") -> Optional[dict]:
        with self._connect() as conn:
            ses = conn.execute(
                "SELECT * FROM sessions WHERE id = ? AND client_id = ?",
                (session_id, client_id),
            ).fetchone()
            if not ses:
                return None
            turns = conn.execute(
                "SELECT question, answer, strategy, complexity, routing_reasoning, retrieved_docs, created_at "
                "FROM qa_records WHERE session_id = ? AND client_id = ? ORDER BY turn_index",
                (session_id, client_id),
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
                    "retrieved_docs": t["retrieved_docs"] or "[]",
                } for t in turns],
            }

    def delete_session(self, session_id: str, client_id: str = ""):
        with self._connect() as conn:
            conn.execute("DELETE FROM qa_records WHERE session_id = ? AND client_id = ?", (session_id, client_id))
            conn.execute("DELETE FROM sessions WHERE id = ? AND client_id = ?", (session_id, client_id))
        logger.info(f"Session deleted from SQLite: {session_id} (client={client_id})")
