"""
在线数据集加载器
从 QADatabase (SQLite) 拉取历史问答记录作为测试查询
"""

import json
import logging
from typing import List, Optional

from langchain_core.documents import Document

from ..models import TestQuery

logger = logging.getLogger(__name__)


class OnlineDataset:
    """
    从 QADatabase 加载在线查询

    使用 stored retrieved_docs JSON 重建 (question, docs) 对，
    只对已有记录进行重新评判，不重新检索。
    """

    def __init__(self, qa_db):
        self.qa_db = qa_db

    # ------------------------------------------------------------------
    #  Core
    # ------------------------------------------------------------------

    def load_from_db(
        self,
        session_ids: Optional[List[str]] = None,
        limit: int = 100,
        min_turns: int = 0,
    ) -> List[TestQuery]:
        """
        从数据库加载历史 QA 记录作为测试查询

        Args:
            session_ids: 限定会话列表；None 表示所有会话
            limit: 最多加载条数
            min_turns: 会话最少轮次过滤

        Returns:
            List[TestQuery]
        """
        all_sessions = self.qa_db.list_sessions(limit=limit)

        queries = []
        for ses in all_sessions:
            sid = ses["id"]
            if session_ids and sid not in session_ids:
                continue
            if ses.get("turn_count", 0) < min_turns:
                continue

            detail = self.qa_db.get_session_detail(sid)
            if not detail:
                continue

            for turn in detail.get("turns", []):
                retrieved_docs_str = turn.get("retrieved_docs", "[]")
                try:
                    docs_meta = json.loads(retrieved_docs_str)
                except (json.JSONDecodeError, TypeError):
                    docs_meta = []

                # 从 metadata 重建 Document 对象
                retrieved_docs = []
                for entry in docs_meta:
                    meta = {
                        "chunk_id": entry.get("chunk_id", ""),
                        "node_id": entry.get("node_id", ""),
                        "title": entry.get("title", ""),
                        "department": entry.get("department", ""),
                        "search_source": entry.get("search_source", ""),
                        "route_strategy": entry.get("route_strategy", ""),
                        "relevance_score": entry.get("relevance_score", 0),
                    }
                    doc = Document(
                        page_content=entry.get("content", meta.get("title", "")),
                        metadata=meta,
                    )
                    retrieved_docs.append(doc)

                queries.append(
                    TestQuery(
                        query_id=f"online_{len(queries)}",
                        query_text=turn.get("question", ""),
                        ground_truth_node_ids=[],
                        department="",
                        metadata={
                            "session_id": sid,
                            "timestamp": turn.get("timestamp", ""),
                            "strategy": turn.get("strategy", ""),
                            "complexity": turn.get("complexity", 0),
                            "stored_answer": turn.get("answer", ""),
                            "stored_docs_count": len(retrieved_docs),
                        },
                    )
                )

                if len(queries) >= limit:
                    break
            if len(queries) >= limit:
                break

        logger.info(f"从 DB 加载了 {len(queries)} 条在线查询")
        return queries

    # ------------------------------------------------------------------
    #  Live capture
    # ------------------------------------------------------------------

    @staticmethod
    def capture_live(
        question: str,
        retrieved_docs: List[Document],
        answer: str,
        strategy: str = "",
        department: str = "",
        session_id: str = "",
    ) -> TestQuery:
        """
        将实时查询转换为 TestQuery（供即时评估用）
        """
        return TestQuery(
            query_id=f"live_{hash(question) % 10**8}",
            query_text=question,
            department=department,
            metadata={
                "session_id": session_id,
                "strategy": strategy,
                "stored_answer": answer,
                "stored_docs_count": len(retrieved_docs),
            },
        )
