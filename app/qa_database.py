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

# 邮箱验证码。真正的防猜测约束是「TTL × 尝试上限 × 每邮箱发信上限」这三者的
# 乘积，不是码空间本身（6 位只有 10^6 种）。详见 consume_email_code。
EMAIL_CODE_TTL_SECONDS = 600
EMAIL_CODE_MAX_ATTEMPTS = 5
# 过期的码再多留一天便于排查，之后删掉
CODE_RETENTION_SECONDS = 24 * 3600

# 发信/验码端点的 purpose 白名单。'login'（免密登录）刻意不在其中 —— 本版
# 没做那条路径，留个口子只会让人以为它存在。
EMAIL_CODE_PURPOSES = ("register", "reset")

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


def generate_email_code() -> str:
    """生成 6 位数字验证码。

    用 secrets.randbelow 而不是 secrets.token_hex(...) % 10**6：10**6 不是
    256 的幂，取模会让低位产生可测的偏斜。randbelow 无偏。
    ":06d" 保留前导零 —— 前端输入框是字符串，不是数字。
    """
    return f"{secrets.randbelow(10**6):06d}"


def _prune_email_codes(conn: sqlite3.Connection) -> None:
    """删掉过期超过 CODE_RETENTION_SECONDS 的验证码行。

    ⚠️ 本函数只含一条 DELETE，只针对一个硬编码的表名。**不要**把它泛化成
    「遍历一批表名」或「接受表名参数」—— scripts/clear_test_data.py 的 TABLES
    列表正是那种形状，照抄过来等于重新制造「一次调用删掉多张表」的风险。
    本模块已经有过一次因为清理谓词写错而静默毁数据的先例。
    """
    conn.execute(
        "DELETE FROM email_codes WHERE expires_at < ?",
        (int(time.time()) - CODE_RETENTION_SECONDS,),
    )


def _mark_consumed(conn: sqlite3.Connection, code_id: int, now: int) -> None:
    """把一个码标记为已消费（作废）。幂等：已消费的不再改 consumed_at。"""
    conn.execute(
        "UPDATE email_codes SET consumed_at = ? WHERE id = ? AND consumed_at IS NULL",
        (now, code_id),
    )


# 上次清理过期验证码的时刻（unix 秒）。节流用：发信路径每次跑一条 DELETE 没必要。
_last_code_prune = 0.0


def _maybe_prune_email_codes(conn: sqlite3.Connection, now: int) -> None:
    """节流版清理，最多每小时真跑一次。失败只记日志，绝不影响发信。"""
    global _last_code_prune
    if now - _last_code_prune < 3600:
        return
    _last_code_prune = now
    try:
        _prune_email_codes(conn)
    except sqlite3.Error:
        logger.exception("清理过期验证码失败，已跳过")


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

            # 邮箱验证码。**刻意不放进 patients 表**：那张表下面就有启动清理，
            # 任何「还没有凭据」的行都会被它连同会话与问答一起删掉。账号行由
            # register_account() 在验证码通过之后才创建，所以验证期间库里根本
            # 没有半成品账号可删 —— 这是本功能不会重蹈上次毁数据覆辙的结构性
            # 原因，不只是「小心一点」。
            #
            # 时间戳用 unix 整数，与本文件其他地方的年月日文本不同。这是刻意
            # 偏离：TTL 比较若用文本就得靠字符串比较加 datetime() 转换，而模块
            # 里没有秒级 ISO 助手；本表不与任何表 join，不一致的代价是零。
            conn.execute("""
                CREATE TABLE IF NOT EXISTS email_codes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    code_salt TEXT NOT NULL,
                    code_hash TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    consumed_at INTEGER
                )
            """)
            # (email, purpose) 活跃时唯一：把「重发即作废上一个码」从应用层
            # 约定变成数据库不变量，并发重发不可能留下两个活码。
            # CREATE TABLE IF NOT EXISTS 上面没包 try，是因为它没有数据相关的
            # 失败模式；索引有（手工测试插过重复行），所以照 idx_patients_email
            # 的惯例两个异常都接。
            try:
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_email_codes_live "
                    "ON email_codes(email, purpose) WHERE consumed_at IS NULL"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_email_codes_expires "
                    "ON email_codes(expires_at)"
                )
            except (sqlite3.OperationalError, sqlite3.IntegrityError):
                logger.exception("创建验证码索引失败，重发将退化为仅应用层作废旧码")

            try:
                _prune_email_codes(conn)
            except sqlite3.Error:
                # 清理失败不能拦住服务启动 —— 与标题回填同样的取舍
                logger.exception("清理过期验证码失败，已跳过（不影响服务启动）")

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

    def account_exists(self, email: str) -> bool:
        """该邮箱是否已注册。email 必须已由调用方规范化。

        只用于发信端点的快速失败（注册用的码没必要发给一个已注册的地址），
        不是注册的唯一防线 —— 那条是 idx_patients_email 唯一索引。
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM patients WHERE lower(email) = ? AND email != ''",
                (email,),
            ).fetchone()
        return row is not None

    def set_account_password(self, email: str, password: str) -> bool:
        """重置账号密码，返回是否真的改到了行。email 必须已规范化。

        换新盐而不是复用旧盐：盐的作用就是「同一个密码在不同账号/不同次设置后
        哈希不同」，复用会让旧哈希的彩虹表重新变得有用。

        ⚠️ 这**不能**吊销已经签发的 patient_token：令牌是无状态 HMAC，30 天
        有效，而本模块与 security.py 都没有 denylist。改了密码，一个已经偷到
        令牌的人仍然进得来。真要修得给 patients 加 token_epoch 并在 _get_patient
        里校验 —— 那是独立的一次改动，别顺手塞进这里。
        """
        salt = secrets.token_hex(16)
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE patients SET password_salt = ?, password_hash = ? "
                "WHERE lower(email) = ? AND email != ''",
                (salt, hash_secret(password, salt), email),
            )
            return cur.rowcount > 0

    # ── Email verification codes ────────────────────────────────────

    def store_email_code(self, email: str, purpose: str, code: str, ttl_seconds: int) -> None:
        """落库一个已经生成好的验证码。

        生成（generate_email_code）与落库分开，是因为调用方必须**先把信发出去、
        成功了再来调本方法**。顺序反过来的话，发信失败会在库里留下一个活码，
        于是用户被冷却卡住、手里却没有任何可用的码。

        并发：同一个 (email, purpose) 同时进来两次，靠 idx_email_codes_live 兜底
        —— 后者会撞唯一索引而不是留下两个活码。发信端点上还有一把进程级的锁，
        所以正常路径不会走到那个冲突。--workers 1 意味着只有一个进程，
        「同一进程内的锁」就是全部所需的互斥。
        """
        salt = secrets.token_hex(16)
        now = int(time.time())
        with self._connect() as conn:
            _maybe_prune_email_codes(conn, now)
            # 重发作废上一个活码
            conn.execute(
                "DELETE FROM email_codes WHERE email = ? AND purpose = ? "
                "AND consumed_at IS NULL",
                (email, purpose),
            )
            conn.execute(
                "INSERT INTO email_codes "
                "(email, purpose, code_salt, code_hash, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (email, purpose, salt, hash_secret(code, salt), now, now + ttl_seconds),
            )

    def consume_email_code(self, email: str, purpose: str, code: str) -> str:
        """校验并消费验证码。

        返回 'ok' | 'wrong' | 'missing' | 'expired' | 'too_many'。
        'missing' 同时表示「从没申请过」与「已被消费掉」—— 端点对二者也只回
        同一个 403，不给出「这个码存在过」的信息。

        哈希用的是与密码同一个 hash_secret（PBKDF2 10 万轮）。**但不要说它是
        本机制的保护**：6 位码只有 10^6 种，拿到库的人离线爆破大约单核 22 小时、
        64 核 20 分钟。真正的防线是 TTL、单次使用、尝试上限，以及发信端点的
        每邮箱每日配额 —— 要调准的是那个配额，不是哈希轮数。
        """
        now = int(time.time())
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, code_salt, code_hash, attempts, expires_at FROM email_codes "
                "WHERE email = ? AND purpose = ? AND consumed_at IS NULL "
                "ORDER BY id DESC LIMIT 1",
                (email, purpose),
            ).fetchone()
            if row is None:
                return "missing"
            if row["expires_at"] < now:
                _mark_consumed(conn, row["id"], now)
                return "expired"
            if row["attempts"] >= EMAIL_CODE_MAX_ATTEMPTS:
                _mark_consumed(conn, row["id"], now)
                return "too_many"
            if not hmac.compare_digest(hash_secret(code, row["code_salt"]), row["code_hash"]):
                # 计数也要 CAS：带 consumed_at IS NULL 才自增，免得给一个已被
                # 别人消费掉的码继续记账（那会让人以为码还有效）
                conn.execute(
                    "UPDATE email_codes SET attempts = attempts + 1 "
                    "WHERE id = ? AND consumed_at IS NULL",
                    (row["id"],),
                )
                return "wrong"
            # 单次使用的全部保证就在下面这个 rowcount 上：并发的第二个请求会
            # 发现行已被消费（rowcount 0）而不是也拿到 ok。**不得**换成不带
            # 检查的 UPDATE —— FastAPI 把同步端点丢进线程池，并发是真会发生的，
            # --workers 1 挡不住。
            cur = conn.execute(
                "UPDATE email_codes SET consumed_at = ? "
                "WHERE id = ? AND consumed_at IS NULL AND attempts < ?",
                (now, row["id"], EMAIL_CODE_MAX_ATTEMPTS),
            )
            return "ok" if cur.rowcount == 1 else "missing"

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
