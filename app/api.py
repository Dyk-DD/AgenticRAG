"""
FastAPI layer for Agentic-RAG clinical decision system.
Provides REST + SSE endpoints for the React frontend.
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from main import ClinicalDecisionSystem
from qa_database import QADatabase

load_dotenv(override=True)
logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic-RAG API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://dyk-dd.github.io",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_rag: Optional[ClinicalDecisionSystem] = None
_qa_db: Optional[QADatabase] = None


def get_rag() -> ClinicalDecisionSystem:
    global _rag
    if _rag is None:
        logger.info("Initializing ClinicalDecisionSystem...")
        try:
            rag = ClinicalDecisionSystem()
            rag.initialize_system()
            rag.build_knowledge_base()
            _rag = rag
            # Sync SQLite session
            if _rag.current_session_id:
                get_qa_db().create_session(_rag.current_session_id)
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
def chat(req: ChatRequest) -> ChatResponse:
    rag = get_rag()
    db = get_qa_db()
    try:
        result, analysis = rag.ask_question_with_routing(req.question, stream=False)
        routing = None
        retrieved_json = "[]"
        if analysis:
            routing = {
                "strategy": analysis.recommended_strategy.value,
                "complexity": analysis.query_complexity,
                "intensity": analysis.relationship_intensity,
                "reasoning": analysis.reasoning,
            }
            # Log and capture retrieved document sources
            router = getattr(rag, 'router', None)
            if router:
                docs = router.last_retrieved_docs if hasattr(router, 'last_retrieved_docs') else []
                if docs:
                    retrieved_json = build_retrieved_docs_json(docs)
                    logger.info(f"[检索来源] 共 {len(docs)} 条:\n{retrieved_json}")

        # Persist to SQLite
        if rag.current_session_id:
            sid = rag.current_session_id
            db.record_qa(
                session_id=sid,
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
            session_id=rag.current_session_id or "",
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ChatResponse(
            answer="",
            session_id=rag.current_session_id or "",
            error=str(e),
        )


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    rag = get_rag()
    db = get_qa_db()

    async def event_stream():
        try:
            # 1. Retrieve memory context
            memory_context = ""
            if rag.memory_module and rag.current_session_id:
                memory_context = rag.memory_module.retrieve_memory_context(
                    req.question, rag.current_session_id
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
            if rag.current_session_id:
                sid = rag.current_session_id
                strategy_val = analysis.recommended_strategy.value if analysis else "unknown"
                complexity_val = analysis.query_complexity if analysis else 0.0
                reasoning_val = analysis.reasoning if analysis else ""

                # Build retrieved docs JSON and log
                retrieved_json = build_retrieved_docs_json(relevant_docs)
                logger.info(f"[检索来源] 共 {len(relevant_docs)} 条:\n{retrieved_json}")

                try:
                    db.record_qa(
                        session_id=sid,
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
                        )
                    except Exception:
                        logger.warning("Failed to record turn to memory", exc_info=True)

            # 6. Done event
            yield f"event: done\ndata: {json.dumps({'session_id': rag.current_session_id, 'answer_length': len(full_answer)}, ensure_ascii=False)}\n\n"

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
def list_sessions():
    db = get_qa_db()
    return {"sessions": db.list_sessions()}


@app.get("/api/sessions/{session_id}")
def get_session_detail(session_id: str):
    db = get_qa_db()
    detail = db.get_session_detail(session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return detail


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    rag = get_rag()
    db = get_qa_db()

    db.delete_session(session_id)

    if rag.memory_module:
        try:
            rag.memory_module.close_session(session_id)
            if rag.memory_module.driver:
                with rag.memory_module.driver.session() as s:
                    s.run(
                        "MATCH (ses:Session {session_id: $sid}) "
                        "OPTIONAL MATCH (ses)-[:CONTAINS]->(t:Turn) "
                        "DETACH DELETE t, ses",
                        sid=session_id,
                    )
        except Exception as e:
            logger.warning(f"Failed to delete Neo4j session: {e}")

    if rag.current_session_id == session_id:
        rag.current_session_id = rag.memory_module.create_session()
        db.create_session(rag.current_session_id)

    return {"status": "deleted", "session_id": session_id}


@app.post("/api/sessions")
def create_session():
    rag = get_rag()
    db = get_qa_db()

    if rag.memory_module and rag.current_session_id:
        rag.memory_module.close_session(rag.current_session_id)
    if rag.memory_module:
        rag.current_session_id = rag.memory_module.create_session()
        db.create_session(rag.current_session_id)

    return {"session_id": rag.current_session_id}


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


# ── Health check ──────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    rag = get_rag()
    return {
        "status": "ok" if rag.system_ready else "initializing",
        "session_id": rag.current_session_id,
    }
