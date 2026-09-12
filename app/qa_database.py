"""SQLite-based Q&A persistence, independent of Neo4j/Milvus."""

import hashlib
import hmac
import json
import logging
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "qa_history.db"

# 访问码存储参数
PBKDF2_ROUNDS = 100_000
ACCESS_CODE_DIGITS = 6


def hash_access_code(code: str, salt: str) -> str:
    """PBKDF2-HMAC-SHA256 派生访问码哈希，返回十六进制字符串。"""
    return hashlib.pbkdf2_hmac(
        "sha256", code.encode("utf-8"), salt.encode("utf-8"), PBKDF2_ROUNDS
    ).hex()


def generate_access_code() -> str:
    return f"{secrets.randbelow(10 ** ACCESS_CODE_DIGITS):0{ACCESS_CODE_DIGITS}d}"


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

            # 访问码改哈希存储（access_code 列保留但不再写入明文）
            for col in ("access_salt", "access_hash"):
                try:
                    conn.execute(f"ALTER TABLE patients ADD COLUMN {col} TEXT DEFAULT ''")
                except sqlite3.OperationalError:
                    pass

            # 一次性清理：哈希机制上线前的旧患者记录 access_hash 为空，而它们的
            # access_code 是明文且多为空串——空码对空码即通过校验，等同于后门。
            # 这类记录无法安全迁移（原文可能为空），连同其会话与问答一并删除，
            # 需要用户重新注册。新注册的记录 access_hash 非空，故本清理幂等。
            legacy_ids = [
                r["id"] for r in conn.execute(
                    "SELECT id FROM patients WHERE COALESCE(access_hash, '') = ''"
                ).fetchall()
            ]
            for pid in legacy_ids:
                conn.execute("DELETE FROM qa_records WHERE client_id = ?", (pid,))
                conn.execute("DELETE FROM sessions WHERE client_id = ?", (pid,))
                conn.execute("DELETE FROM patients WHERE id = ?", (pid,))
            if legacy_ids:
                logger.warning(
                    f"已清理 {len(legacy_ids)} 条无访问码哈希的旧患者记录及其会话数据，"
                    f"这些身份需要重新注册"
                )

    # ── Patient management ──────────────────────────────────────────

    def register_patient(self, name: str, access_code: str = "") -> dict:
        """注册患者。未提供访问码时由服务端生成一个，并只在本次响应中明文返回。

        库中仅保存 PBKDF2 哈希与随机 salt，明文不落盘。
        """
        pid = "p_" + hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()[:8]
        code = access_code.strip() or generate_access_code()
        salt = secrets.token_hex(16)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO patients (id, name, access_code, access_salt, access_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (pid, name, "", salt, hash_access_code(code, salt),
                 time.strftime("%Y-%m-%dT%H:%M:%S")),
            )
        logger.info(f"Patient registered: {pid} ({name})")
        # access_code 是唯一一次可读到明文的时机，调用方需提示用户保存
        return {"patient_id": pid, "name": name, "access_code": code}

    def list_patients(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, name, created_at FROM patients ORDER BY created_at DESC LIMIT 50"
            ).fetchall()
        return [{"patient_id": r["id"], "name": r["name"], "created_at": r["created_at"]} for r in rows]

    def verify_patient(self, patient_id: str, access_code: str = "") -> Optional[dict]:
        """校验患者身份。访问码为空一律拒绝，不做哈希比对。

        此前是 ``WHERE id = ? AND access_code = ?`` 的明文比对，注册时若留空，
        空码对空码即通过，配合公开的患者列表等于任意接管。现在强制非空，
        且用 compare_digest 定时安全比较。
        """
        if not access_code.strip():
            return None

        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, name, access_salt, access_hash FROM patients WHERE id = ?",
                (patient_id,),
            ).fetchone()

        if not row or not row["access_hash"]:
            return None

        expected = hash_access_code(access_code.strip(), row["access_salt"])
        if not hmac.compare_digest(expected, row["access_hash"]):
            return None
        return {"patient_id": row["id"], "name": row["name"]}

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
