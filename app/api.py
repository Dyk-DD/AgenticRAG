"""
FastAPI layer for Agentic-RAG clinical decision system.
Provides REST + SSE endpoints for the React frontend.
"""

import asyncio
import collections
import hmac
import json
import logging
import os
import re
import sqlite3
import sys
import threading
import time
from typing import Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import mailer
from main import ClinicalDecisionSystem
from qa_database import (
    EMAIL_CODE_PURPOSES,
    EMAIL_CODE_TTL_SECONDS,
    MAX_EMAIL_LEN,
    MAX_PASSWORD_LEN,
    MIN_PASSWORD_LEN,
    QADatabase,
    SHARE_MAX_TURNS,
    SHARE_TURN_FIELDS,
    clean_title,
    generate_email_code,
)
from security import (
    ACCESS_SUBJECT,
    ACCESS_TTL,
    PATIENT_TTL,
    issue_token,
    verify_token,
)

# 加载 .env：优先找 app/.env，没找到则找项目根目录的 .env
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if not os.path.exists(_env_path):
    _env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(_env_path, override=True)
logger = logging.getLogger(__name__)

ACCESS_PASSWORD = os.getenv("ACCESS_PASSWORD", "")
# 未配置密码时默认拒绝服务（fail closed）。仅本机开发可显式开闸。
ALLOW_NO_AUTH = os.getenv("ALLOW_NO_AUTH", "") in ("1", "true", "True")


def _env(name: str, default: str) -> str:
    """读环境变量，未设置或为空时回落到默认值。"""
    value = os.getenv(name)
    return value if value else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except ValueError:
        return default


# ── Rate limiter (in-memory sliding window) ──────────────────────────

RATE_LIMIT_WINDOW = 60  # seconds
# 上限可配。一次对话会打若干个请求（chat/stream + sessions + stats），
# 而静态资源已在中间件里提前放行，不占用这里的额度。
RATE_LIMIT_MAX = _env_int("RATE_LIMIT_MAX", 60)


class RateLimiter:
    """进程内滑动窗口限流器。

    窗口与上限由构造参数给：验证码发信要表达「每天 N 封」，那是全局 60 秒窗口
    表达不了的。

    ⚠️ 进程内且易失：容器重启即归零，多 worker 时每个 worker 各算一份
    （Dockerfile 钉死 --workers 1，所以眼下就是全局的）。用于发信配额时要清楚
    这一点 —— 重启丢的是**额度记账**，不是防猜测的硬约束，后者在 SQLite 里。
    """

    def __init__(self, window_seconds: int, max_events: int, max_keys: int = 10_000):
        self._window = window_seconds
        self._max = max_events
        self._max_keys = max_keys
        self._windows: dict[str, collections.deque] = {}

    def check(self, key: str) -> bool:
        now = time.time()
        window = self._windows.get(key)
        if window is None:
            # 回收：不加这一步，key 空间就由调用方决定 —— 旧版 key 只有客户端
            # IP（缓慢泄漏），换成邮箱之后攻击者每请求换一个地址就能让
            # _windows 无限增长，等于一个可控的内存放大面。
            if len(self._windows) >= self._max_keys:
                self._evict()
            self._windows[key] = collections.deque([now])
            return True
        # 滑动窗口：移除超出时间窗口的记录
        while window and window[0] < now - self._window:
            window.popleft()
        if len(window) >= self._max:
            return False
        window.append(now)
        return True

    def _evict(self) -> None:
        """先丢已空的窗口，再按插入序丢最旧的，直到降到上限以下。

        dict 保序，所以 next(iter(...)) 就是最早插入的那个。被丢掉的 key 会
        丢失它的计数（等于提前放行），这是刻意的取舍：在内存被打爆和个别 key
        的额度被重置之间，选后者。
        """
        for k in [k for k, w in self._windows.items() if not w]:
            del self._windows[k]
        while len(self._windows) >= self._max_keys:
            self._windows.pop(next(iter(self._windows)))


_rate_limiter = RateLimiter(RATE_LIMIT_WINDOW, RATE_LIMIT_MAX)

# ── 验证码的发信 / 校验配额 ──────────────────────────────────────────
#
# 每邮箱的配额**不能**放中间件里：中间件只看得到 request.url.path，拿不到邮箱
# （要拿就得 await request.body() 再把流塞回去，既脆又会让全站每个请求都被迫
# 缓冲）。中间件那层保持 IP 键、全局；邮箱键的配额放在端点体内。
#
# 发信上限必须压在邮件服务商自己的每日上限之下，否则用户看到的是看似代码 bug
# 的神秘 502。个人 QQ 邮箱的每日发信上限常远低于 100。
EMAIL_CODE_RESEND_SECONDS = _env_int("EMAIL_CODE_RESEND_SECONDS", 60)
EMAIL_SEND_MAX_PER_EMAIL_HOUR = _env_int("EMAIL_SEND_MAX_PER_EMAIL_HOUR", 3)
EMAIL_SEND_MAX_PER_EMAIL_DAY = _env_int("EMAIL_SEND_MAX_PER_EMAIL_DAY", 10)
EMAIL_SEND_MAX_PER_IP_DAY = _env_int("EMAIL_SEND_MAX_PER_IP_DAY", 30)
EMAIL_VERIFY_MAX_PER_IP_HOUR = _env_int("EMAIL_VERIFY_MAX_PER_IP_HOUR", 30)

_email_cooldown = RateLimiter(EMAIL_CODE_RESEND_SECONDS, 1)
_email_hourly = RateLimiter(3600, EMAIL_SEND_MAX_PER_EMAIL_HOUR)
_email_daily = RateLimiter(86400, EMAIL_SEND_MAX_PER_EMAIL_DAY)
_ip_send_daily = RateLimiter(86400, EMAIL_SEND_MAX_PER_IP_DAY)
_ip_verify_hourly = RateLimiter(3600, EMAIL_VERIFY_MAX_PER_IP_HOUR)

# 发信这一段（查配额 → 发信 → 落库）串行化，双击不会产生两封信加两次竞态插入。
# 代价是发送被串行（每次最多 SMTP_TIMEOUT_SECONDS），在本站并发下无所谓。
# 若将来并发上来了，替代方案是按邮箱加锁，但那需要自己的回收（同限流器的
# 无界 key 问题）。
_EMAIL_CODE_LOCK = threading.Lock()


def _client_ip(request: Request) -> str:
    """取真实客户端 IP。

    套上 Cloudflare 隧道之后 request.client.host 是隧道出口 IP，
    全站共用一个 key，限流会退化成"所有人一起被限"。因此优先读代理头。
    注意：仅当确实部署在可信代理之后时这些头才可信。
    """
    forwarded = request.headers.get("cf-connecting-ip") or ""
    if not forwarded:
        xff = request.headers.get("x-forwarded-for", "")
        forwarded = xff.split(",")[0].strip() if xff else ""
    if forwarded:
        return forwarded
    return request.client.host if request.client else "unknown"

# ── App ──────────────────────────────────────────────────────────────

app = FastAPI(title="Agentic-RAG API", version="1.0")

# 生产部署下前端与 API 同源，CORS 不会被触发；此处保留列表是为开发模式
# （vite dev server 在 5173）以及将前端另行部署到别的源的情况。逗号分隔。
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        o.strip()
        for o in _env(
            "CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,https://dyk-dd.github.io",
        ).split(",")
        if o.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth middleware ───────────────────────────────────────────────────

PUBLIC_PATHS = {"/api/health", "/api/auth"}

# 公开的路径前缀。目前只有会话分享的读取接口：拿到链接的人不需要访问密码。
# ⚠️ 前缀必须与 HTTP 方法**绑在一起**判断（见下方中间件），不能只按前缀放行 ——
# 只按前缀的话对所有方法都生效，而撤销端点是 DELETE /api/shares/{token}
# （多个 s），今天靠拼写差异侥幸躲过去了，但那是靠拼写而不是靠语义建立的边界。
# 哪天有人把撤销端点改回 DELETE /api/share/{token}，中间件就会把一个未认证的
# **状态变更**请求放进业务代码里。
# 注意这条与 PROTECTED_PREFIXES 是并列的：/api/share/... 仍在 ("/api",) 覆盖
# 范围内，所以它照样经过限流 —— 「公开 ≠ 无限额」，分享链接是一条零成本读通道。
PUBLIC_GET_PREFIXES = ("/api/share/",)

# 需要鉴权的路径前缀，其余一切（SPA 路由、/assets/*.js、favicon）都公开。
#
# 这条分界必须画在中间件里，不能靠"把静态路由注册在中间件之后"来实现：
# Starlette 的 add_middleware 会把中间件包在 router 之外，而 Mount 也属于
# router，所以任何路由都逃不过中间件——按路由注册位置来区分是做不到的。
PROTECTED_PREFIXES = ("/api", "/docs", "/redoc", "/openapi.json")


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # OPTIONS 预检不带凭据，始终放行
    if request.method == "OPTIONS":
        return await call_next(request)

    path = request.url.path
    # 静态资源与 SPA 路由公开，且不占用 API 的限流额度
    if not path.startswith(PROTECTED_PREFIXES):
        return await call_next(request)

    # 限流必须在公开路径判断之前：/api/auth 本身是公开的，
    # 若先 return 就等于给密码暴破留了一条无限次尝试的通道。
    if not _rate_limiter.check(_client_ip(request)):
        return JSONResponse(
            status_code=429,
            content={"detail": "请求过于频繁，请稍后再试"}
        )

    # 公开分享读取。必须在限流之后（见上），且方法必须一起判 —— 理由见
    # PUBLIC_GET_PREFIXES 的注释。
    if path in PUBLIC_PATHS or (
        request.method in ("GET", "HEAD") and path.startswith(PUBLIC_GET_PREFIXES)
    ):
        return await call_next(request)

    # 未配置访问密码时拒绝服务，而不是静默放行全部接口
    if not ACCESS_PASSWORD:
        if ALLOW_NO_AUTH:
            return await call_next(request)
        return JSONResponse(
            status_code=503,
            content={"detail": "服务未配置 ACCESS_PASSWORD，已拒绝访问"}
        )

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"detail": "未提供访问令牌"})

    # 验签而非明文比对：令牌不可伪造、可过期，且不再是密码本身
    claims = verify_token(auth_header[7:])
    if not claims or claims.get("sub") != ACCESS_SUBJECT:
        return JSONResponse(status_code=401, content={"detail": "访问令牌无效或已过期"})

    return await call_next(request)


@app.post("/api/auth")
async def auth_login(req: Request):
    """校验访问密码，签发带有效期的令牌（不再回传密码本身）。"""
    body = await req.json()
    password = body.get("password", "")
    if not ACCESS_PASSWORD:
        return JSONResponse(status_code=503, content={"detail": "服务未配置 ACCESS_PASSWORD"})
    if not hmac.compare_digest(password, ACCESS_PASSWORD):
        return JSONResponse(status_code=403, content={"detail": "密码错误"})
    return {"token": issue_token(ACCESS_SUBJECT, ACCESS_TTL)}

_rag: Optional[ClinicalDecisionSystem] = None
_qa_db: Optional[QADatabase] = None
_init_lock = threading.Lock()


def get_rag() -> ClinicalDecisionSystem:
    global _rag
    if _rag is None:
        with _init_lock:
            if _rag is None:  # double-check
                logger.info("Initializing ClinicalDecisionSystem...")
                try:
                    rag = ClinicalDecisionSystem()
                    rag.initialize_system()
                    rag.build_knowledge_base()
                    _rag = rag
                    # 不再为全局会话同步 SQLite：REST 侧会话按患者各自创建，
                    # 全局字段只服务 CLI，落库会产生一条无归属的孤儿会话
                    logger.info("ClinicalDecisionSystem ready")
                except Exception as e:
                    logger.error(f"System initialization failed: {e}", exc_info=True)
                    raise HTTPException(status_code=503, detail=f"系统初始化失败: {e}")
    return _rag


def get_qa_db() -> QADatabase:
    global _qa_db
    if _qa_db is None:
        _qa_db = QADatabase()
    return _qa_db


from fastapi import Request as FastAPIRequest

def _get_patient(req: FastAPIRequest) -> str:
    """从患者身份令牌解出 patient_id，验签失败一律 401。

    旧实现直接返回请求头 x-patient-id 的值，等于把数据隔离的决定权交给
    客户端——改一个 header 就能读写他人病历。现在身份只能来自服务端签发的
    令牌，且令牌载荷带签名，改 patient_id 会导致验签失败。
    """
    claims = verify_token(req.headers.get("x-patient-token", ""))
    subject = (claims or {}).get("sub", "")
    # 访问令牌（sub="access"）不能用来冒充患者身份
    if not subject or subject == ACCESS_SUBJECT:
        raise HTTPException(status_code=401, detail="患者身份无效或已过期，请重新登录")
    return subject

# ── Account endpoints ─────────────────────────────────────────────────
#
# 账号叠在全局访问密码之后：中间件先验 Bearer 访问令牌，这里再验账号密码。
# 保留全局门是有意的 —— 它仍是一层挡住陌生人烧 DeepSeek 额度的屏障。
#
# 发信端点**同样留在全局门之后**，不进 PUBLIC_PATHS。一旦服务器能发信，多出来
# 的第二个可被烧的资源是发信域名的声誉与服务商额度；让未认证方使本站域名向
# 任意地址发信，是比烧额度更糟的垃圾邮件/黑名单向量。而且它买不到什么：
# ChatGate 已经强制先过 LoginPage 再到 AccountGate，用户能走到发码表单时手上
# 必然有 Bearer 令牌。
#
# 返回体沿用 patient_token 这个名字，令牌载荷也仍然是 issue_token(patient_id)，
# 这样 _get_patient 与全部 client_id 数据隔离逻辑一行都不用改。
#
# 验证码本身**不签发任何令牌**：那会绕不开 _get_patient 只拒绝 sub == "access"
# 的问题（任何其它已签名的 sub 都会被当成 patient_id）。不签发就没有这个面。

# 不做 RFC 5322，也不查 MX —— 但它**不再**只是拦手误：这个地址会被塞进邮件头
# 的 To:，而换行就是头部注入原语。所以必须是 fullmatch 而不是 match：re.match
# 配 $ 锚会放过结尾的 \n（实测 "a@b.co\n" 能通过 match），今天只是因为
# _normalize_email 恰好先 strip 了才没事 —— 那是调用顺序上的巧合，不是保证。
_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")

# 验证码校验失败的响应文案。'missing'（从没申请过 / 已被消费）与 'wrong'
# 合并成同一句：二者都不该告诉调用方「这个码存在过」。
_CODE_ERRORS = {
    "missing": "验证码错误",
    "wrong": "验证码错误",
    "expired": "验证码已过期，请重新获取",
    "too_many": "验证码错误次数过多，请重新获取",
}

class AccountEmailCode(BaseModel):
    email: str
    purpose: str = "register"

class AccountRegister(BaseModel):
    email: str
    password: str
    name: str = ""
    # 默认空串而不是必填：必填缺字段时 FastAPI 抛的是 422，而它的 detail 是
    # 一个数组，前端的 `detail || HTTP xxx` 会渲染成 [object Object]。
    # 给默认值再在端点里校验，用户看到的就是一句干净的 400。
    code: str = ""

class AccountLogin(BaseModel):
    email: str
    password: str

class AccountPasswordReset(BaseModel):
    email: str
    code: str = ""
    new_password: str = ""

def _normalize_email(raw: str) -> str:
    return (raw or "").strip().lower()


def _validate_email_or_400(email: str) -> None:
    if len(email) > MAX_EMAIL_LEN or not _EMAIL_RE.fullmatch(email):
        raise HTTPException(status_code=400, detail="邮箱格式不正确")


def _validate_code_or_400(code: str) -> str:
    """验证码必须是恰好 6 位 ASCII 数字。返回去空白后的码。"""
    code = (code or "").strip()
    # isascii 不能省：'１２３４５６'（全角）的 isdigit() 是 True，而它永远
    # 不可能是我们发出去的码，放进去只会白烧一个 attempts 名额。
    if len(code) != 6 or not code.isascii() or not code.isdigit():
        raise HTTPException(status_code=400, detail="验证码格式不正确")
    return code


@app.post("/api/accounts/email-code")
def send_email_code(body: AccountEmailCode, request: Request):
    """发一封验证码邮件。成功返回 200，客户端永远拿不到验证码本身。"""
    email = _normalize_email(body.email)
    purpose = (body.purpose or "register").strip().lower()
    _validate_email_or_400(email)
    if purpose not in EMAIL_CODE_PURPOSES:
        raise HTTPException(status_code=400, detail="验证方式不正确")

    db = get_qa_db()
    exists = db.account_exists(email)
    if purpose == "register" and exists:
        # 与 POST /api/accounts/register 的 409 同一个文案。这里泄露「邮箱是否
        # 已注册」是**刻意**的：注册接口本来就泄露同样的事实，堵这里买不到任何
        # 机密性，却要为每个已注册地址白花一次真实发信。
        raise HTTPException(status_code=409, detail="该邮箱已注册")
    if purpose == "reset" and not exists:
        # 不告诉调用方这个邮箱存不存在，也不白花一次发信。
        #
        # ⚠️ 残留的旁路：这条分支立即返回，而真发信要花几百毫秒到几秒，所以
        # 响应时间仍然能区分「已注册」。它不构成新的泄露（register 的 409 已经
        # 明说了），要靠 dummy sleep 去抹平则是拿一个每秒一次的延迟换一个已知
        # 的秘密，不划算。如实记在这里，不要假装堵上了。
        logger.info("验证码邮件跳过：邮箱未注册 purpose=%s", purpose)
        return _code_sent_payload()

    # 三层分开是为了给出不同文案：冷却是等一等就有，每日上限是今天没戏了。
    if not _email_cooldown.check(email):
        raise HTTPException(
            status_code=429,
            detail="发送过于频繁，请稍后再试",
            headers={"Retry-After": str(EMAIL_CODE_RESEND_SECONDS)},
        )
    if not (_email_hourly.check(email) and _email_daily.check(email)
            and _ip_send_daily.check(_client_ip(request))):
        raise HTTPException(status_code=429, detail="今日验证码发送次数已达上限")

    if not mailer.is_configured():
        raise HTTPException(status_code=503, detail="未配置邮件服务，无法发送验证码")

    with _EMAIL_CODE_LOCK:
        code = generate_email_code()
        try:
            mailer.send_code_email(email, code, purpose, EMAIL_CODE_TTL_SECONDS)
        except mailer.MailNotConfigured:
            raise HTTPException(status_code=503, detail="未配置邮件服务，无法发送验证码")
        except mailer.MailSendError as exc:
            # 502 而不是 503：503 在中间件里已经表示「这个部署缺 ACCESS_PASSWORD」，
            # 两者必须在日志和监控里可区分。只记收件人与异常类型 —— 验证码本身
            # 是一次性凭据，绝不能进日志（这里也绝不回显给客户端）。
            logger.error(
                "验证码邮件发送失败 recipient=%s purpose=%s type=%s",
                email, purpose, type(exc).__name__, exc_info=True,
            )
            raise HTTPException(status_code=502, detail="验证码发送失败，请稍后再试")
        # 只有发送成功才落库：发失败却先落库的话，用户被冷却卡住、手里却没码。
        db.store_email_code(email, purpose, code, EMAIL_CODE_TTL_SECONDS)

    logger.info("验证码已发送 recipient=%s purpose=%s", email, purpose)
    return _code_sent_payload()


def _code_sent_payload() -> dict:
    """发信成功的响应。resend_after 由服务端下发，前端的倒计时就不会与服务端的
    冷却各说各话。"""
    return {
        "sent": True,
        "expires_in": EMAIL_CODE_TTL_SECONDS,
        "resend_after": EMAIL_CODE_RESEND_SECONDS,
    }


@app.post("/api/accounts/register", status_code=201)
def register_account(body: AccountRegister, request: Request):
    email = _normalize_email(body.email)
    _validate_email_or_400(email)
    if not (MIN_PASSWORD_LEN <= len(body.password) <= MAX_PASSWORD_LEN):
        raise HTTPException(
            status_code=400,
            detail=f"密码长度需在 {MIN_PASSWORD_LEN}-{MAX_PASSWORD_LEN} 个字符之间",
        )
    code = _validate_code_or_400(body.code)
    if not _ip_verify_hourly.check(f"{_client_ip(request)}:register"):
        raise HTTPException(status_code=429, detail="尝试过于频繁，请稍后再试")

    # 先验码并消费，再建账号。反过来的话会开一个重放窗口：部分失败会留下
    # 「有效码 + 已建账号」。代价是随后的 INSERT 若失败（磁盘错、或输给
    # idx_patients_email 的竞争），码已烧掉，用户得重新获取 —— 可接受。
    result = get_qa_db().consume_email_code(email, "register", code)
    if result != "ok":
        # 403 而不是 401：客户端的 ensureAccountOk 只在 401 时清本地身份并整页
        # 重载，用 401 会把用户填好的表单连同全局访问令牌一起丢掉。
        raise HTTPException(status_code=403, detail=_CODE_ERRORS.get(result, "验证码错误"))

    name = (body.name or "").strip() or email.split("@", 1)[0]
    try:
        account = get_qa_db().register_account(email, name[:40], body.password)
    except sqlite3.IntegrityError:
        # 唯一索引是防重复注册的唯一防线（先查后插有竞态窗口）
        raise HTTPException(status_code=409, detail="该邮箱已注册")
    account["patient_token"] = issue_token(account["patient_id"], PATIENT_TTL)
    return account


@app.post("/api/accounts/password-reset")
def password_reset(body: AccountPasswordReset, request: Request):
    """验证码通过后重设密码。

    ⚠️ **不能吊销已经签发的 patient_token**：令牌是无状态 HMAC、30 天有效，
    而 security.py 与本模块都没有 denylist。所以改密码挡不住一个已经偷到令牌
    的人。真正的修法是给 patients 加 token_epoch 并在 _get_patient 里校验 ——
    那是独立的一次改动。
    """
    email = _normalize_email(body.email)
    _validate_email_or_400(email)
    if not (MIN_PASSWORD_LEN <= len(body.new_password) <= MAX_PASSWORD_LEN):
        raise HTTPException(
            status_code=400,
            detail=f"密码长度需在 {MIN_PASSWORD_LEN}-{MAX_PASSWORD_LEN} 个字符之间",
        )
    code = _validate_code_or_400(body.code)
    # 每 IP 的校验配额，与每码的 attempts 正交：防的是 attempts 计数器本身
    # 失效（实现 bug）时无人兜底。它计每一次请求，不论是否存在活码。
    if not _ip_verify_hourly.check(f"{_client_ip(request)}:reset"):
        raise HTTPException(status_code=429, detail="尝试过于频繁，请稍后再试")

    result = get_qa_db().consume_email_code(email, "reset", code)
    if result != "ok":
        raise HTTPException(status_code=403, detail=_CODE_ERRORS.get(result, "验证码错误"))
    if not get_qa_db().set_account_password(email, body.new_password):
        # 码有效但账号不见了 —— 只可能是发码之后被删掉的（比如管理员清了测试
        # 数据）。如实说，别假报成功。
        raise HTTPException(status_code=404, detail="该邮箱未注册")
    logger.info("密码已重置 recipient=%s", email)
    # 刻意不签发令牌：让所有签发令牌的端点维持同一种响应形状，前端本来就有
    # 模式切换，多一次点击不值得多一条写 localStorage 的代码路径。
    return {"reset": True}

@app.post("/api/accounts/login")
def login_account(body: AccountLogin):
    # 邮箱与密码错误返回同一个 403，不区分二者，避免枚举已注册邮箱
    account = get_qa_db().verify_account(_normalize_email(body.email), body.password)
    if not account:
        raise HTTPException(status_code=403, detail="邮箱或密码错误")
    account["patient_token"] = issue_token(account["patient_id"], PATIENT_TTL)
    return account


def build_retrieved_docs_json(documents) -> str:
    """Extract source metadata from retrieved documents for traceability."""
    entries = []
    for i, doc in enumerate(documents):
        meta = doc.metadata
        entries.append({
            "index": i,
            "chunk_id": meta.get("chunk_id", ""),
            "node_id": meta.get("node_id", ""),
            "title": meta.get("title", ""),
            "department": meta.get("department", ""),
            "search_source": meta.get("search_source", meta.get("search_method", "")),
            "route_strategy": meta.get("route_strategy", ""),
            "relevance_score": meta.get("final_score", meta.get("relevance_score", 0)),
        })
    return json.dumps(entries, ensure_ascii=False)


# ── Request/Response models ──────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    stream: bool = True


class ChatResponse(BaseModel):
    answer: str
    routing: Optional[dict] = None
    session_id: str
    error: Optional[str] = None


# ── Chat endpoints ───────────────────────────────────────────────────

@app.post("/api/chat")
def chat(req: ChatRequest, request: FastAPIRequest) -> ChatResponse:
    rag = get_rag()
    db = get_qa_db()
    cid = _get_patient(request)
    sid = _ensure_session(rag, db, cid)
    try:
        # 显式传入会话 id：绝不能依赖 rag.current_session_id，否则并发用户串会话
        result, analysis = rag.ask_question_with_routing(
            req.question, stream=False, session_id=sid
        )
        routing = None
        retrieved_json = "[]"
        if analysis:
            routing = {
                "strategy": analysis.recommended_strategy.value,
                "complexity": analysis.query_complexity,
                "intensity": analysis.relationship_intensity,
                "reasoning": analysis.reasoning,
            }
            router = getattr(rag, 'router', None)
            if router:
                docs = router.last_retrieved_docs if hasattr(router, 'last_retrieved_docs') else []
                if docs:
                    retrieved_json = build_retrieved_docs_json(docs)
                    logger.info(f"[检索来源] 共 {len(docs)} 条:\n{retrieved_json}")

        if sid:
            db.ensure_session_client(sid, cid)
            db.record_qa(
                session_id=sid,
                client_id=cid,
                question=req.question,
                answer=str(result),
                strategy=analysis.recommended_strategy.value if analysis else "unknown",
                complexity=analysis.query_complexity if analysis else 0.0,
                routing_reasoning=analysis.reasoning if analysis else "",
                retrieved_docs=retrieved_json,
            )

        return ChatResponse(
            answer=str(result),
            routing=routing,
            session_id=sid or "",
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ChatResponse(
            answer="",
            session_id=sid or "",
            error=str(e),
        )


# ── 会话隔离 ─────────────────────────────────────────────────────────
# _rag 是进程级单例，ClinicalDecisionSystem.current_session_id 只是它的一个
# 实例字段。若各接口直接读写该字段，所有患者会共用同一个会话：A 的提问会被
# 记进 B 的对话历史，且 B 之后再也读不到自己被"串"走的那一轮。
# 因此按 client_id 各自维护当前会话，实例字段仅保留给 CLI 单用户场景。

_client_sessions: dict[str, str] = {}
_client_sessions_lock = threading.Lock()


def _ensure_session(rag, db, cid: str) -> str:
    """返回该患者当前的会话 id，首次访问时创建并落库。"""
    with _client_sessions_lock:
        existing = _client_sessions.get(cid)
        if existing:
            return existing
        if not rag.memory_module:
            return ""
        sid = rag.memory_module.create_session()
        _client_sessions[cid] = sid
    # 在锁外写 SQLite，避免持锁做 I/O
    db.create_session(sid, client_id=cid)
    return sid


def _set_session(cid: str, sid: str) -> None:
    with _client_sessions_lock:
        _client_sessions[cid] = sid


def _current_session(cid: str) -> str:
    with _client_sessions_lock:
        return _client_sessions.get(cid, "")


def _drop_session(cid: str) -> str:
    """移除并返回该患者的当前会话 id（用于删除后置空）。"""
    with _client_sessions_lock:
        return _client_sessions.pop(cid, "")


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, request: FastAPIRequest):
    rag = get_rag()
    db = get_qa_db()
    cid = _get_patient(request)

    # 确保有可用会话（首次提问或删除当前会话后自动创建）
    # 按患者解析，不能用进程级的 rag.current_session_id
    sid = _ensure_session(rag, db, cid)

    async def event_stream():
        # 立刻推一个注释帧，把响应体发出去。首次 yield 之前要跑记忆检索和
        # 一次 DeepSeek 路由调用，若超过隧道/边缘的空闲超时就会先收到 524。
        # 注释帧符合 SSE 规范，前端的手写解析器会直接跳过。
        yield ": open\n\n"
        try:
            # 1. Retrieve memory context
            memory_context = ""
            if rag.memory_module and sid:
                memory_context = rag.memory_module.retrieve_memory_context(
                    req.question, sid, cid
                )

            # 2. Route query (safe access pattern matching main.py)
            router = getattr(rag, 'router', getattr(rag, 'query_router', None))
            if not router:
                yield f"event: error\ndata: {json.dumps({'message': '系统路由模块未就绪'}, ensure_ascii=False)}\n\n"
                return

            relevant_docs, analysis = router.route_query(req.question, rag.config.top_k)

            # 3. Send routing event
            routing_data = None
            if analysis:
                routing_data = {
                    "strategy": analysis.recommended_strategy.value,
                    "complexity": analysis.query_complexity,
                    "intensity": analysis.relationship_intensity,
                    "reasoning": analysis.reasoning,
                }
                yield f"event: routing\ndata: {json.dumps(routing_data, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)

            if not relevant_docs:
                yield f"event: error\ndata: {json.dumps({'message': '未找到相关临床参考信息'}, ensure_ascii=False)}\n\n"
                return

            # 4. Stream tokens — yield to event loop after each to flush
            full_answer = ""
            token_count = 0
            for chunk in rag.generation_module.generate_adaptive_answer_stream(
                req.question, relevant_docs, memory_context
            ):
                if isinstance(chunk, str):
                    full_answer += chunk
                    yield f"event: token\ndata: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
                    token_count += 1
                    if token_count % 3 == 0:
                        await asyncio.sleep(0)

            await asyncio.sleep(0)

            # 5. Persist to SQLite
            if sid:
                db.ensure_session_client(sid, cid)
                strategy_val = analysis.recommended_strategy.value if analysis else "unknown"
                complexity_val = analysis.query_complexity if analysis else 0.0
                reasoning_val = analysis.reasoning if analysis else ""

                # Build retrieved docs JSON and log
                retrieved_json = build_retrieved_docs_json(relevant_docs)
                logger.info(f"[检索来源] 共 {len(relevant_docs)} 条:\n{retrieved_json}")

                try:
                    db.record_qa(
                        session_id=sid,
                        client_id=cid,
                        question=req.question,
                        answer=full_answer,
                        strategy=strategy_val,
                        complexity=complexity_val,
                        routing_reasoning=reasoning_val,
                        retrieved_docs=retrieved_json,
                    )
                except Exception:
                    logger.warning("Failed to persist to SQLite", exc_info=True)

                # Also record to Neo4j memory
                if rag.memory_module:
                    try:
                        extracted = rag.memory_module.extract_referenced_entities(req.question, full_answer)
                        rag.memory_module.record_turn(
                            session_id=sid,
                            question=req.question,
                            answer=full_answer,
                            strategy=strategy_val,
                            complexity=complexity_val,
                            extracted_entities=extracted,
                            # 记忆必须带患者归属：没有它，语义召回无法隔离，
                            # 删除会话时也无法把这条记忆一并清掉
                            client_id=cid,
                        )
                    except Exception:
                        logger.warning("Failed to record turn to memory", exc_info=True)

            # 6. Done event
            yield f"event: done\ndata: {json.dumps({'session_id': sid, 'answer_length': len(full_answer)}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Session endpoints ─────────────────────────────────────────────────

@app.get("/api/sessions")
def list_sessions(request: FastAPIRequest):
    cid = _get_patient(request)
    db = get_qa_db()
    return {"sessions": db.list_sessions(client_id=cid)}


@app.get("/api/sessions/{session_id}")
def get_session_detail(session_id: str, request: FastAPIRequest):
    cid = _get_patient(request)
    db = get_qa_db()
    detail = db.get_session_detail(session_id, client_id=cid)
    if detail is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return detail


class SessionRename(BaseModel):
    title: str


@app.put("/api/sessions/{session_id}")
def rename_session(session_id: str, body: SessionRename, request: FastAPIRequest):
    cid = _get_patient(request)
    db = get_qa_db()

    # 归属校验必须前置，与下面的 DELETE 同一惯例。改名不碰记忆存储，所以这道
    # 预检就是全部防线——漏了就是「任何登录用户可改任何会话名」，而 session_id
    # 是可猜的（sess_<unix秒>_<6位十六进制>）。
    if db.get_session_detail(session_id, client_id=cid) is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    title = clean_title(body.title)
    if not title:
        raise HTTPException(status_code=400, detail="标题不能为空")

    if not db.set_session_title(session_id, title, client_id=cid):
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"session_id": session_id, "title": title}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, request: FastAPIRequest):
    cid = _get_patient(request)
    rag = get_rag()
    db = get_qa_db()

    # 归属校验必须前置：下面删 Neo4j 的语句只按 session_id 匹配，不校验归属，
    # 少了这一步任何登录用户传别人的 session_id 就能清空对方的图谱记忆。
    if db.get_session_detail(session_id, client_id=cid) is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 顺序很重要：先 close_session（它会把 _pending_flush 里的轮次 flush 到
    # Neo4j 和 Milvus），再删三个存储。旧实现是"先删 Neo4j 再 close"，于是
    # 刷新回来的轮次反而被写进 Milvus 并永久留下 —— 删一次多一份。
    memory_result = {"milvus": True, "neo4j": True}
    if rag.memory_module:
        try:
            rag.memory_module.close_session(session_id)
        except Exception as e:
            logger.warning(f"关闭会话记忆失败: {e}")
        memory_result = rag.memory_module.delete_session_memory(session_id)

    db.delete_session(session_id, client_id=cid)

    # 删的是当前会话则移除映射，下次提问时自动创建新会话
    if _current_session(cid) == session_id:
        _drop_session(cid)

    # 记忆存储没删干净要显式告诉调用方：否则用户看到"删除成功"，
    # 而副本仍在，且仍可能被召回。
    if not all(memory_result.values()):
        logger.error(f"会话记忆未完全删除: {session_id} -> {memory_result}")

    return {
        "status": "deleted",
        "session_id": session_id,
        "memory_deleted": memory_result,
        "fully_deleted": all(memory_result.values()),
    }


@app.put("/api/sessions/{session_id}/pin")
def pin_session(session_id: str, request: FastAPIRequest):
    """置顶会话。**刻意不带请求体**（见下），所以 PUT 与 DELETE 是一对纯语义开关。"""
    cid = _get_patient(request)
    db = get_qa_db()

    # 归属校验必须前置，与改名/删除同一惯例：session_id 是可猜的
    # （sess_<unix秒>_<6位十六进制>），漏了就是任何登录用户可置顶任何人的会话。
    if db.get_session_detail(session_id, client_id=cid) is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 为什么用无请求体的 PUT/DELETE 而不是 {"pinned": true}：
    # 本文件给所有 Pydantic 字段都写了默认值（见 ShareCreate 附近的说明），
    # 于是 pinned: bool = False 会让 {"pinned_": true} 这种拼写错误**返回 200
    # 并且取消置顶** —— 一次静默的反向操作。意图写进 HTTP 方法里就没有可拼错的
    # 字段。这也与已有的无体 PUT /api/sessions/{id}/activate 同一风格。
    if not db.set_session_pinned(session_id, True, client_id=cid):
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"session_id": session_id, "pinned": True}


@app.delete("/api/sessions/{session_id}/pin")
def unpin_session(session_id: str, request: FastAPIRequest):
    cid = _get_patient(request)
    db = get_qa_db()

    if db.get_session_detail(session_id, client_id=cid) is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    if not db.set_session_pinned(session_id, False, client_id=cid):
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"session_id": session_id, "pinned": False}


@app.put("/api/sessions/{session_id}/activate")
def activate_session(session_id: str, request: FastAPIRequest):
    """切换到指定历史会话，后续提问将延续该会话"""
    cid = _get_patient(request)
    rag = get_rag()
    db = get_qa_db()

    detail = db.get_session_detail(session_id, client_id=cid)
    if detail is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 关闭当前会话（flush 未持久化的轮次）
    previous = _current_session(cid)
    if rag.memory_module and previous:
        rag.memory_module.close_session(previous)

    # 只切换该患者自己的当前会话，不动全局字段
    _set_session(cid, session_id)
    logger.info(f"患者 {cid} 已切换到会话: {session_id}")
    return {"session_id": session_id}


@app.post("/api/sessions")
def create_session(request: FastAPIRequest):
    cid = _get_patient(request)
    rag = get_rag()
    db = get_qa_db()

    previous = _current_session(cid)
    if rag.memory_module and previous:
        rag.memory_module.close_session(previous)

    if not rag.memory_module:
        return {"session_id": ""}

    new_sid = rag.memory_module.create_session()
    _set_session(cid, new_sid)
    db.create_session(new_sid, client_id=cid)

    return {"session_id": new_sid}


# ── Share endpoints ───────────────────────────────────────────────────

# 字段一律给默认值。required 字段触发的是 422，而 422 的 detail 是一个数组，
# 前端把它当字符串渲染出来就是 [object Object]（见本文件其他模型同样的说明）。
class ShareCreate(BaseModel):
    turns: list[int] = []


@app.post("/api/sessions/{session_id}/share")
def create_share(session_id: str, body: ShareCreate, request: FastAPIRequest):
    """把选中的若干轮问答固化成一条公开链接。"""
    cid = _get_patient(request)
    db = get_qa_db()

    # 归属校验必须前置，与改名/删除同一惯例：session_id 是可猜的
    # （sess_<unix秒>_<6位十六进制>），漏了就是任何登录用户可分享任何人的会话，
    # 而且是**公开**分享。
    detail = db.get_session_detail(session_id, client_id=cid)
    if detail is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 请求里传的是真实 turn_index，不是数组下标 —— 下标是隐式契约，任何一次
    # 「过滤掉空答案」这类前端改动都会让它静默错位，后果是把 A 轮的答案公开挂在
    # B 轮的问题下面。注意 turn_index 不保证连续（并发下会有两条 = 0）。
    wanted = body.turns
    if not wanted:
        raise HTTPException(status_code=400, detail="请至少选择一轮对话")
    if len(wanted) > SHARE_MAX_TURNS:
        raise HTTPException(status_code=400, detail=f"一次最多分享 {SHARE_MAX_TURNS} 轮对话")

    by_index = {t["turn_index"]: t for t in detail["turns"]}
    picked: list[dict] = []
    seen: set[int] = set()
    for idx in wanted:
        if idx in seen:
            continue
        turn = by_index.get(idx)
        # 不存在的轮次直接拒绝整个请求，而不是跳过：静默少几轮会让用户以为
        # 自己漏勾了，而链接已经发出去了。
        if turn is None:
            raise HTTPException(status_code=400, detail=f"第 {idx} 轮对话不存在")
        seen.add(idx)
        picked.append(turn)

    # 按会话原本的顺序输出，而不是用户勾选的先后 —— 后者会让分享页上的问答
    # 顺序与用户记忆中的对话顺序不一致。
    picked.sort(key=lambda t: t["turn_index"])

    # 白名单裁剪，理由见 qa_database.SHARE_TURN_FIELDS
    snapshot = [{k: t.get(k, "") for k in SHARE_TURN_FIELDS} for t in picked]
    data = db.create_share(session_id, cid, detail["title"], snapshot)
    # path 由后端给出，前端不自己拼 —— 路由前缀属于服务端的事
    return {**data, "path": f"/s/{data['token']}"}


@app.get("/api/share/{token}")
def get_share(token: str):
    """公开读取。**无需任何鉴权** —— token 本身就是凭据。

    失效（不存在 / 已撤销 / 行损坏）一律 404，不区分原因：区分了就等于给
    「这个 token 曾经存在过」提供了一条探测通道。
    """
    data = get_qa_db().get_share(token)
    if data is None:
        raise HTTPException(status_code=404, detail="分享不存在或已失效")
    return data


@app.delete("/api/shares/{token}")
def revoke_share(token: str, request: FastAPIRequest):
    """撤销分享。注意路径是 /api/shares/（复数），与公开的 /api/share/ 不同 ——
    中间件的公开前缀是方法绑定的，这个 DELETE 走不到那条豁免。"""
    cid = _get_patient(request)
    db = get_qa_db()

    # client_id 参与 WHERE（见 revoke_share），所以别人的 token 撤不掉
    if not db.revoke_share(token, client_id=cid):
        raise HTTPException(status_code=404, detail="分享不存在或已撤销")
    return {"status": "revoked"}


# ── Stats endpoint ────────────────────────────────────────────────────

@app.get("/api/stats")
def get_stats():
    rag = get_rag()
    stats = {"qa_pairs": 0, "departments": 0, "milvus_rows": 0, "total_queries": 0}

    if rag.data_module:
        ds = rag.data_module.get_statistics()
        stats["qa_pairs"] = ds.get("total_qa_pairs", 0)
        stats["departments"] = ds.get("total_departments", 0)

    if rag.index_module:
        ms = rag.index_module.get_collection_stats()
        stats["milvus_rows"] = ms.get("row_count", 0)

    if rag.router:
        rs = rag.router.get_route_statistics()
        stats["total_queries"] = rs.get("total_queries", 0)
        stats["route_distribution"] = {
            "traditional": rs.get("traditional_count", 0),
            "graph_rag": rs.get("graph_rag_count", 0),
            "combined": rs.get("combined_count", 0),
        }

    return stats


# ── Startup event ──────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """服务器启动时初始化 RAG 系统，初始化完成后再接受请求"""
    logger.info("正在初始化 ClinicalDecisionSystem，请稍候...")
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, get_rag)
        logger.info("ClinicalDecisionSystem 初始化完毕，服务器就绪")
    except Exception as e:
        logger.error(f"ClinicalDecisionSystem 初始化失败: {e}")
        logger.error("请检查: 1) Neo4j 是否运行  2) Milvus 是否运行  3) .env 配置是否正确")


# ── Health check ──────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    """公开的健康探针。

    不返回 session_id：该接口在 PUBLIC_PATHS 中无需鉴权，回传会话 id
    等于向任何未认证的访问者泄漏他人当前正在使用的会话。
    """
    rag = _rag
    if rag is None:
        return {"status": "initializing"}
    return {"status": "ok" if rag.system_ready else "initializing"}


# ── 静态资源：SPA 构建产物 ─────────────────────────────────────────────
#
# 生产部署下前端与 API 同源（Cloudflare 隧道指向本容器），页面和接口共用
# 一个域名，因此既没有 CORS 也没有 mixed content。开发模式由 vite dev
# server 自己服务页面，这段只在 STATIC_DIR 存在时才有意义。
#
# 容器内挂载点为 /app/static（见 docker-compose.yml）；原生运行 uvicorn 时
# 默认回落到 <repo>/frontend/dist，方便不装 Docker 时验证。

STATIC_DIR = _env(
    "STATIC_DIR",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "frontend", "dist",
    ),
)
_ASSETS_DIR = os.path.join(STATIC_DIR, "assets")
_INDEX_FILE = os.path.join(STATIC_DIR, "index.html")

# 内容哈希命名的静态资源。必须注册在下面的 catch-all 之前：Starlette 按注册
# 顺序匹配、先匹配者生效，否则 /assets/*.js 会被 catch-all 吞掉。
if os.path.isdir(_ASSETS_DIR):
    app.mount("/assets", StaticFiles(directory=_ASSETS_DIR), name="assets")


@app.api_route("/{full_path:path}", methods=["GET", "HEAD"], include_in_schema=False)
async def spa_fallback(full_path: str):
    """把未匹配的路径交给 SPA 的客户端路由。

    没有这一段的话，在 /history 上刷新页面会拿到 404 而不是 index.html。
    """
    # 未知的 /api/* 必须保持 404，不能回落到 index.html
    if full_path.split("/", 1)[0] in {"api", "docs", "redoc", "openapi.json"}:
        raise HTTPException(status_code=404, detail="Not Found")

    if not os.path.isfile(_INDEX_FILE):
        raise HTTPException(
            status_code=404,
            detail="前端静态资源未构建，请先执行 cd frontend && npm run build",
        )

    # dist 根目录下的散装文件（favicon.svg、icons.svg）。
    # realpath + 前缀校验用于挡住 %2e%2e%2f 路径穿越——{full_path:path}
    # 会把 "../" 原样收下，光靠 isfile 拦不住。
    if full_path:
        candidate = os.path.realpath(os.path.join(STATIC_DIR, full_path))
        root = os.path.realpath(STATIC_DIR)
        if candidate.startswith(root + os.sep) and os.path.isfile(candidate):
            return FileResponse(candidate)

    # index.html 不缓存：产物带内容哈希，重新构建后要能立刻生效
    return FileResponse(_INDEX_FILE, headers={"Cache-Control": "no-cache"})
