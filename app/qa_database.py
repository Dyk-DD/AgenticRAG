"""SQLite-based Q&A persistence, independent of Neo4j/Milvus."""

import hashlib
import hmac
import json
import logging
import re
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "qa_history.db"

# 凭据存储参数
PBKDF2_ROUNDS = 100_000
# 改名、自动标题、启动回填共用一个上限，避免三处各截各的
TITLE_MAX_LEN = 40
MIN_PASSWORD_LEN = 8
# PBKDF2 的开销与输入长度成正比，不限长等于送一个廉价的 CPU DoS
MAX_PASSWORD_LEN = 128
MAX_EMAIL_LEN = 254

# 标题里要剥掉的控制字符。不能用 str.isprintable() 代劳：它会把 U+200D
# （零宽连接符）一并剥掉，从而把 emoji 组合序列拆成两个孤立字符。
_TITLE_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def hash_secret(secret: str, salt: str) -> str:
    """PBKDF2-HMAC-SHA256 派生凭据哈希，返回十六进制字符串。

    账号密码与老的访问码共用。PBKDF2_ROUNDS 不可下调：库里已有的
    access_hash 是按当前轮数算出来的，改了它们就再也验不过。
    """
    return hashlib.pbkdf2_hmac(
        "sha256", secret.encode("utf-8"), salt.encode("utf-8"), PBKDF2_ROUNDS
    ).hex()


def clean_title(raw: str) -> str:
    """清洗会话标题：去控制字符、折叠空白、截断到 TITLE_MAX_LEN。"""
    text = _TITLE_CTRL_RE.sub("", raw or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:TITLE_MAX_LEN].strip()


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
            for col, tbl in [
                ("retrieved_docs", "qa_records"), ("client_id", "sessions"), ("client_id", "qa_records"),
                ("title", "sessions"),
                ("email", "patients"), ("password_salt", "patients"), ("password_hash", "patients"),
            ]:
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

            # 邮箱唯一。必须是部分索引：老记录 email 全是空串，普通唯一索引会冲突。
            # 用 lower(email) 让 X@y.com 与 x@y.com 判为同一个。
            #
            # 单独一个 try 且连 IntegrityError 一起接：唯一索引撞上已有重复数据时
            # 抛的是 IntegrityError 而不是 OperationalError，漏掉它会一路穿过
            # _init_schema → QADatabase() → get_qa_db()（那里没有 try），变成每个
            # 请求 500，配合 restart: unless-stopped 就是崩溃循环。
            try:
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_patients_email "
                    "ON patients(lower(email)) WHERE email != ''"
                )
            except (sqlite3.OperationalError, sqlite3.IntegrityError):
                logger.exception("创建 email 唯一索引失败，账号去重退化为仅应用层检查")

            # 一次性清理：哈希机制上线前的旧患者记录两种凭据都没有，而它们的
            # access_code 是明文且多为空串——空码对空码即通过校验，等同于后门。
            # 这类记录无法安全迁移（原文可能为空），连同其会话与问答一并删除，
            # 需要用户重新注册。
            #
            # 两个条件缺一不可：邮箱账号只写 password_hash 不写 access_hash，
            # 只看 access_hash 会在下一次启动时把所有账号连同数据一起删掉。
            legacy_ids = [
                r["id"] for r in conn.execute(
                    "SELECT id FROM patients "
                    "WHERE COALESCE(access_hash, '') = '' AND COALESCE(password_hash, '') = ''"
                ).fetchall()
            ]
            for pid in legacy_ids:
                conn.execute("DELETE FROM qa_records WHERE client_id = ?", (pid,))
                conn.execute("DELETE FROM sessions WHERE client_id = ?", (pid,))
                conn.execute("DELETE FROM patients WHERE id = ?", (pid,))
            if legacy_ids:
                logger.warning(
                    f"已清理 {len(legacy_ids)} 条无任何凭据的旧患者记录及其会话数据，"
                    f"这些身份需要重新注册"
                )

            # 回填会话标题：老会话没有标题列，用它的第一条提问补上。
            #
            # 幂等性靠 question != '' 而不只是 EXISTS：substr(NULL,1,40) 是 NULL，
            # 而 COALESCE(NULL,'') = '' 为真，所以没有这个条件时，无提问的会话会
            # 在每次启动时被反复重写（写入 NULL，下次又匹配）。
            # ORDER BY turn_index, id 是为了确定性：record_qa 在并发下可能产生两条
            # turn_index = 0，靠 id 兜底。
            try:
                conn.execute(
                    """
                    UPDATE sessions SET title = (
                        SELECT substr(q.question, 1, :maxlen)
                        FROM qa_records q
                        WHERE q.session_id = sessions.id AND q.question != ''
                        ORDER BY q.turn_index, q.id
                        LIMIT 1
                    )
                    WHERE COALESCE(title, '') = ''
                      AND EXISTS (
                          SELECT 1 FROM qa_records q
                          WHERE q.session_id = sessions.id AND q.question != ''
                      )
                    """,
                    {"maxlen": TITLE_MAX_LEN},
                )
            except sqlite3.Error:
                # 数据异常时退化成「没有标题」而不是「服务起不来」
                logger.exception("会话标题回填失败，已跳过（不影响服务启动）")

    # ── Account management ──────────────────────────────────────────
    #
    # 表名与列名仍是 patients / patient_id / client_id：sessions 与 qa_records
    # 的归属列都按这个 id 关联，改名要连带动两张表，风险大而无功能收益。
    # 「账号」是它在界面上的叫法。

    def register_account(self, email: str, name: str, password: str) -> dict:
        """创建账号。email 必须已由调用方规范化（strip + lower）。

        邮箱重复不在这里预查，交给 idx_patients_email 唯一索引兜底，冲突时
        抛 sqlite3.IntegrityError 由端点转成 409 —— FastAPI 会把同步端点丢进
        线程池，先查后插存在真实的竞态窗口。

        id 用 "u_" + 随机十六进制，绝不能用邮箱本身：client_id 会被拼进 Milvus
        过滤表达式，而 _SAFE_FILTER_RE 只允许 [A-Za-z0-9_-]，出现 @ 或 . 会让
        语义检索静默拒答（写入正常、读取永远为空）。
        """
        pid = "u_" + secrets.token_hex(6)
        salt = secrets.token_hex(16)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO patients (id, name, access_code, access_salt, access_hash, "
                "email, password_salt, password_hash, created_at) "
                "VALUES (?, ?, '', '', '', ?, ?, ?, ?)",
                (pid, name, email, salt, hash_secret(password, salt),
                 time.strftime("%Y-%m-%dT%H:%M:%S")),
            )
        logger.info(f"Account registered: {pid} ({email})")
        return {"patient_id": pid, "name": name, "email": email}

    def verify_account(self, email: str, password: str) -> Optional[dict]:
        """校验账号密码。邮箱不存在或密码不符一律返回 None，不区分二者。

        邮箱不存在时也跑一次等价开销的哈希：否则响应时间会暴露某个邮箱是否
        已注册，可用来枚举用户。
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, name, email, password_salt, password_hash FROM patients "
                "WHERE lower(email) = ? AND email != ''",
                (email,),
            ).fetchone()

        if not row or not row["password_hash"]:
            hash_secret(password, "0" * 32)
            return None

        expected = hash_secret(password, row["password_salt"])
        if not hmac.compare_digest(expected, row["password_hash"]):
            return None
        return {"patient_id": row["id"], "name": row["name"], "email": row["email"]}

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
            # 首条提问顺带定标题。判断是 max_turn == -1（本次写入的下标是
            # max_turn + 1），写成 == 0 会把标题安到第二条提问上。
            # COALESCE(title,'') = '' 保证用户改过的名字不会被覆盖；WHERE 里再带
            # client_id 是纵深防御。同处一个 with 块，与上面的 INSERT 同一事务。
            if max_turn == -1:
                title = clean_title(question)
                if title:
                    conn.execute(
                        "UPDATE sessions SET title = ? "
                        "WHERE id = ? AND client_id = ? AND COALESCE(title, '') = ''",
                        (title, session_id, client_id),
                    )
        logger.info(f"QA recorded in SQLite: {session_id} turn {max_turn + 1} (client={client_id})")

    def list_sessions(self, client_id: str = "", limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT s.id, s.created_at, s.closed_at, COALESCE(s.title, '') AS title, "
                "COUNT(q.id) AS turn_count "
                "FROM sessions s LEFT JOIN qa_records q ON s.id = q.session_id "
                "WHERE s.client_id = ? "
                # created_at 只精确到秒，同一秒建的两条会话顺序不定，配合下面的
                # LIMIT 会让会话在两次调用之间换位甚至被挤掉，所以用 id 兜底
                "GROUP BY s.id ORDER BY s.created_at DESC, s.id DESC LIMIT ?",
                (client_id, limit),
            ).fetchall()
            return [{"id": r["id"], "title": r["title"], "start_time": r["created_at"], "end_time": r["closed_at"] or "", "turn_count": r["turn_count"]} for r in rows]

    def set_session_title(self, session_id: str, title: str, client_id: str = "") -> bool:
        """改写会话标题。client_id 参与 WHERE，返回是否真的改到了行。"""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ? AND client_id = ?",
                (title, session_id, client_id),
            )
            return cur.rowcount > 0

    def get_session_detail(self, session_id: str, client_id: str = "") -> Optional[dict]:
        with self._connect() as conn:
            ses = conn.execute(
                "SELECT * FROM sessions WHERE id = ? AND client_id = ?",
                (session_id, client_id),
            ).fetchone()
            if not ses:
                return None
            # 带 id 排序的原因同 list_sessions：并发下可能出现两条 turn_index = 0
            turns = conn.execute(
                "SELECT question, answer, strategy, complexity, routing_reasoning, retrieved_docs, created_at "
                "FROM qa_records WHERE session_id = ? AND client_id = ? ORDER BY turn_index, id",
                (session_id, client_id),
            ).fetchall()
            return {
                "id": ses["id"],
                "title": ses["title"] or "",
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
