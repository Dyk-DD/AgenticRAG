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

from main import ClinicalDecisionSystem
from qa_database import (
    MAX_EMAIL_LEN,
    MAX_PASSWORD_LEN,
    MIN_PASSWORD_LEN,
    QADatabase,
    clean_title,
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
    def __init__(self):
        self._windows: dict[str, collections.deque] = {}

    def check(self, key: str) -> bool:
        now = time.time()
        window = self._windows.get(key)
        if window is None:
            self._windows[key] = collections.deque([now])
            return True
        # 滑动窗口：移除超出时间窗口的记录
        while window and window[0] < now - RATE_LIMIT_WINDOW:
            window.popleft()
        if len(window) >= RATE_LIMIT_MAX:
            return False
        window.append(now)
        return True


_rate_limiter = RateLimiter()


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

    if path in PUBLIC_PATHS:
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
# 保留全局门是有意的 —— 它是不发验证邮件的前提下唯一能挡住陌生人烧
# DeepSeek 额度的一层。
#
# 返回体沿用 patient_token 这个名字，令牌载荷也仍然是 issue_token(patient_id)，
# 这样 _get_patient 与全部 client_id 数据隔离逻辑一行都不用改。
#
# 不发邮件：没有验证码、没有密码重置。忘记密码只能靠管理员跑
# scripts/migrate_sessions_to_account.py 改挂到新账号，前端要把这点说清。

# 够用即可：不发信，所以只用来拦手误，不做 RFC 5322，也不查 MX
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

class AccountRegister(BaseModel):
    email: str
    password: str
    name: str = ""

class AccountLogin(BaseModel):
    email: str
    password: str

def _normalize_email(raw: str) -> str:
    return (raw or "").strip().lower()

@app.post("/api/accounts/register", status_code=201)
def register_account(body: AccountRegister):
    email = _normalize_email(body.email)
    if len(email) > MAX_EMAIL_LEN or not _EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="邮箱格式不正确")
    if not (MIN_PASSWORD_LEN <= len(body.password) <= MAX_PASSWORD_LEN):
        raise HTTPException(
            status_code=400,
            detail=f"密码长度需在 {MIN_PASSWORD_LEN}-{MAX_PASSWORD_LEN} 个字符之间",
        )
    name = (body.name or "").strip() or email.split("@", 1)[0]
    try:
        account = get_qa_db().register_account(email, name[:40], body.password)
    except sqlite3.IntegrityError:
        # 唯一索引是防重复注册的唯一防线（先查后插有竞态窗口）
        raise HTTPException(status_code=409, detail="该邮箱已注册")
    account["patient_token"] = issue_token(account["patient_id"], PATIENT_TTL)
    return account

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
