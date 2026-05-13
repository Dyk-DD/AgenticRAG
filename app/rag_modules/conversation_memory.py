"""
临床决策辅助系统 - 对话记忆模块
三层混合记忆架构：
1. 短期缓冲 (deque) — 当前会话最近 N 轮对话，零延迟
2. 图谱记忆 (Neo4j) — 结构化诊断推理链，链接到疾病/科室实体
3. 语义记忆 (Milvus) — 跨会话相似病例召回
"""

import logging
import os
import time
import hashlib
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from pymilvus import DataType, CollectionSchema, FieldSchema

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    turn_id: str
    session_id: str
    turn_index: int
    question: str
    answer: str
    strategy: str
    timestamp: str
    complexity_score: float
    disease_entities: List[str] = field(default_factory=list)
    department: str = ""


@dataclass
class MemoryContext:
    recent_turns: str
    similar_past_turns: str
    session_summary: str
    graph_context: str


class ConversationMemory:
    """临床对话记忆模块"""

    def __init__(self, config, llm_client, data_module, milvus_module):
        self.config = config
        self.llm_client = llm_client
        self.data_module = data_module
        self.milvus_module = milvus_module

        self.driver = None
        self.buffer: deque[ConversationTurn] = deque(maxlen=config.memory_buffer_size)
        self._collection_ready = False
        self._neo4j_ready = False
        self._pending_flush: List[ConversationTurn] = []

    def initialize(self):
        """初始化记忆模块：连接 Neo4j，创建 Milvus 集合"""
        if not self.config.memory_enabled:
            logger.info("记忆模块未启用")
            return

        self._connect_neo4j()
        self._setup_milvus_collection()

    def _connect_neo4j(self):
        try:
            self.driver = self.data_module.driver
            if self.driver:
                with self.driver.session() as session:
                    session.run("RETURN 1")
                    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Session) REQUIRE s.session_id IS UNIQUE")
                    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Turn) REQUIRE t.turn_id IS UNIQUE")
                self._neo4j_ready = True
                logger.info("记忆模块 Neo4j 连接就绪")
        except Exception as e:
            logger.warning(f"记忆模块 Neo4j 连接失败，图记忆暂不可用: {e}")

    def _setup_milvus_collection(self):
        try:
            if not self.milvus_module.client:
                logger.warning("Milvus 客户端不可用，语义记忆暂不可用")
                return

            collection_name = self.config.memory_milvus_collection
            if self.milvus_module.client.has_collection(collection_name):
                self.milvus_module.client.load_collection(collection_name)
                self._collection_ready = True
                logger.info(f"Milvus 记忆集合 {collection_name} 已加载")
                return

            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=150, is_primary=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.config.milvus_dimension),
                FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=5000),
                FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=8000),
                FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="strategy", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="department", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="disease_entities", dtype=DataType.VARCHAR, max_length=2000),
            ]
            schema = CollectionSchema(fields=fields, description="临床对话记忆向量集合")

            self.milvus_module.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                metric_type="COSINE",
                consistency_level="Strong",
            )

            index_params = self.milvus_module.client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 200},
            )
            self.milvus_module.client.create_index(
                collection_name=collection_name, index_params=index_params
            )
            self.milvus_module.client.load_collection(collection_name)
            self._collection_ready = True
            logger.info(f"Milvus 记忆集合 {collection_name} 创建完成")

        except Exception as e:
            logger.warning(f"Milvus 记忆集合初始化失败: {e}")

    # ==================== Session Lifecycle ====================

    def create_session(self) -> str:
        session_id = f"sess_{int(time.time())}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:6]}"
        self.buffer.clear()
        self._pending_flush.clear()

        if self._neo4j_ready and self.driver:
            try:
                with self.driver.session() as s:
                    s.run(
                        "MERGE (ses:Session {session_id: $sid}) SET ses.start_time = datetime(), ses.turn_count = 0",
                        sid=session_id,
                    )
            except Exception as e:
                logger.warning(f"创建 Neo4j 会话节点失败: {e}")

        logger.info(f"新临床会话已创建: {session_id}")
        return session_id

    def close_session(self, session_id: str):
        if self._pending_flush:
            self._flush_to_neo4j(self._pending_flush)
            self._flush_to_milvus(self._pending_flush)
            self._pending_flush.clear()

        if self._neo4j_ready and self.driver:
            try:
                with self.driver.session() as s:
                    s.run(
                        "MATCH (ses:Session {session_id: $sid}) SET ses.end_time = datetime()",
                        sid=session_id,
                    )
            except Exception as e:
                logger.warning(f"关闭 Neo4j 会话失败: {e}")

        logger.info(f"临床会话已关闭: {session_id}")

    # ==================== Memory Recording ====================

    def record_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        strategy: str = "unknown",
        complexity: float = 0.0,
        extracted_entities: Optional[Dict[str, Any]] = None,
    ):
        if not self.config.memory_enabled:
            return

        turn = ConversationTurn(
            turn_id=f"turn_{session_id}_{len(self.buffer)}",
            session_id=session_id,
            turn_index=len(self.buffer),
            question=question,
            answer=answer[:2000],
            strategy=strategy,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            complexity_score=complexity,
            disease_entities=extracted_entities.get("diseases", []) if extracted_entities else [],
            department=extracted_entities.get("department", "") if extracted_entities else "",
        )

        self.buffer.append(turn)
        self._pending_flush.append(turn)

        if len(self._pending_flush) >= self.config.memory_batch_flush_size:
            self._flush_to_neo4j(self._pending_flush)
            self._flush_to_milvus(self._pending_flush)
            self._pending_flush.clear()

        if len(self.buffer) >= self.config.memory_summary_threshold:
            self._summarize_buffer(session_id)

    def _flush_to_neo4j(self, turns: List[ConversationTurn]):
        if not self._neo4j_ready or not self.driver:
            return
        if not turns:
            return

        try:
            with self.driver.session() as s:
                batch = [
                    {
                        "turn_id": t.turn_id,
                        "session_id": t.session_id,
                        "turn_index": t.turn_index,
                        "question": t.question,
                        "answer": t.answer,
                        "strategy": t.strategy,
                        "timestamp": t.timestamp,
                        "complexity": t.complexity_score,
                        "diseases": t.disease_entities,
                        "department": t.department,
                    }
                    for t in turns
                ]

                s.run(
                    """
                    UNWIND $batch AS data
                    MATCH (ses:Session {session_id: data.session_id})
                    CREATE (t:Turn {
                        turn_id: data.turn_id,
                        turn_index: data.turn_index,
                        question: data.question,
                        answer: data.answer,
                        strategy: data.strategy,
                        timestamp: datetime(data.timestamp),
                        complexity_score: data.complexity
                    })
                    CREATE (ses)-[:CONTAINS]->(t)
                    SET ses.turn_count = coalesce(ses.turn_count, 0) + 1
                    """,
                    batch=batch,
                )

                for t in turns:
                    if t.turn_index > 0:
                        prev_turn_id = f"turn_{t.session_id}_{t.turn_index - 1}"
                        s.run(
                            "MATCH (prev:Turn {turn_id: $prev_id}) "
                            "MATCH (curr:Turn {turn_id: $curr_id}) "
                            "MERGE (prev)-[:NEXT]->(curr)",
                            prev_id=prev_turn_id,
                            curr_id=t.turn_id,
                        )

                for t in turns:
                    for disease_name in t.disease_entities:
                        s.run(
                            "MATCH (t:Turn {turn_id: $tid}) "
                            "MERGE (d:Disease {name: $dname}) "
                            "MERGE (t)-[:REFERENCES_DISEASE]->(d)",
                            tid=t.turn_id,
                            dname=disease_name,
                        )
                    if t.department:
                        s.run(
                            "MATCH (t:Turn {turn_id: $tid}) "
                            "MERGE (d:Department {name: $dname}) "
                            "MERGE (t)-[:REFERENCES_DEPT]->(d)",
                            tid=t.turn_id,
                            dname=t.department,
                        )

            logger.info(f"已写入 {len(turns)} 轮到 Neo4j")
        except Exception as e:
            logger.warning(f"Neo4j 记忆写入失败，将在下次重试: {e}")

    def _flush_to_milvus(self, turns: List[ConversationTurn]):
        if not self._collection_ready:
            return
        if not turns:
            return

        try:
            texts = [t.question[:800] for t in turns]
            vectors = self.milvus_module.embeddings.embed_documents(texts)
            collection_name = self.config.memory_milvus_collection

            entities = []
            for t, vector in zip(turns, vectors):
                entities.append(
                    {
                        "id": t.turn_id,
                        "vector": vector,
                        "question": t.question[:5000],
                        "answer": t.answer[:8000],
                        "session_id": t.session_id,
                        "strategy": t.strategy,
                        "department": t.department,
                        "disease_entities": ", ".join(t.disease_entities)[:2000],
                    }
                )

            self.milvus_module.client.insert(collection_name=collection_name, data=entities)
            logger.info(f"已写入 {len(turns)} 轮到 Milvus")

        except Exception as e:
            logger.warning(f"Milvus 记忆写入失败，将在下次重试: {e}")

    def _summarize_buffer(self, session_id: str):
        if not self.llm_client or len(self.buffer) < 3:
            return

        dialogue = "\n".join(
            f"问: {t.question}\n答: {t.answer[:300]}" for t in self.buffer
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是临床会话摘要助手。用2-3句话总结以下医患对话涉及的主要症状、疑似疾病、讨论的药物和关键建议。只输出摘要，不要额外解释。",
                    },
                    {"role": "user", "content": dialogue},
                ],
                temperature=0.1,
                max_tokens=300,
            )
            summary = response.choices[0].message.content.strip()

            if self._neo4j_ready and self.driver:
                with self.driver.session() as s:
                    s.run(
                        "MATCH (ses:Session {session_id: $sid}) SET ses.patient_summary = $summary",
                        sid=session_id,
                        summary=summary,
                    )
            logger.info(f"会话摘要已更新: {summary[:80]}...")
        except Exception as e:
            logger.warning(f"LLM 摘要生成失败: {e}")

    # ==================== Memory Retrieval ====================

    def retrieve_memory_context(self, question: str, session_id: str) -> str:
        if not self.config.memory_enabled:
            return ""

        parts = []

        recent = self._format_buffer_context()
        if recent:
            parts.append(recent)

        similar = self._semantic_recall(question, session_id)
        if similar:
            parts.append(similar)

        graph_ctx = self._graph_recall(session_id)
        if graph_ctx:
            parts.append(graph_ctx)

        full_context = "\n\n".join(parts)
        max_chars = self.config.memory_max_context_chars
        if len(full_context) > max_chars:
            full_context = full_context[:max_chars] + "\n[...记忆上下文已截断]"

        return full_context

    def _format_buffer_context(self) -> str:
        if not self.buffer:
            return ""

        lines = ["【当前会话近期对话】"]
        for i, t in enumerate(self.buffer):
            lines.append(f"第{i + 1}轮 — 问: {t.question}")
            lines.append(f"第{i + 1}轮 — 答: {t.answer[:300]}")
            if t.disease_entities:
                lines.append(f"第{i + 1}轮 — 涉及疾病: {', '.join(t.disease_entities)}")
            lines.append("")

        return "\n".join(lines)

    def _semantic_recall(self, question: str, session_id: str) -> str:
        if not self._collection_ready:
            return ""

        try:
            query_vector = self.milvus_module.embeddings.embed_query(question)
            collection_name = self.config.memory_milvus_collection

            results = self.milvus_module.client.search(
                collection_name=collection_name,
                data=[query_vector],
                anns_field="vector",
                limit=self.config.memory_top_k * 2,
                output_fields=["question", "answer", "session_id", "disease_entities", "strategy"],
                search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            )

            if not results or not results[0]:
                return ""

            lines = ["【历史相似病例回忆】"]
            seen = set()
            count = 0
            for hit in results[0]:
                if hit["distance"] < 0.5:
                    continue
                q = hit["entity"].get("question", "")
                if q in seen:
                    continue
                seen.add(q)
                ans = hit["entity"].get("answer", "")[:200]
                diseases = hit["entity"].get("disease_entities", "")
                lines.append(f"- 曾提问: {q[:200]}")
                lines.append(f"  当时回答: {ans}")
                if diseases:
                    lines.append(f"  关联疾病: {diseases}")
                lines.append("")
                count += 1
                if count >= self.config.memory_top_k:
                    break

            if count == 0:
                return ""

            return "\n".join(lines)

        except Exception as e:
            logger.warning(f"语义记忆召回失败: {e}")
            return ""

    def _graph_recall(self, session_id: str) -> str:
        if not self._neo4j_ready or not self.driver:
            return ""

        try:
            with self.driver.session() as s:
                result = s.run(
                    """
                    MATCH (ses:Session {session_id: $sid})-[:CONTAINS]->(t:Turn)
                    OPTIONAL MATCH (t)-[:REFERENCES_DISEASE]->(d:Disease)
                    WITH t, collect(DISTINCT d.name) AS diseases
                    RETURN t.question AS question, t.answer AS answer,
                           t.strategy AS strategy, diseases
                    ORDER BY t.turn_index
                    """,
                    sid=session_id,
                )

                records = list(result)
                if not records:
                    return ""

                all_diseases = set()
                for r in records:
                    for d in r.get("diseases", []) or []:
                        all_diseases.add(d)

                strategies = set(r.get("strategy", "") for r in records if r.get("strategy"))

                lines = [
                    "【会话图谱摘要】",
                    f"本轮会话已讨论 {len(records)} 个问题",
                    f"涉及疾病: {', '.join(all_diseases) if all_diseases else '暂无'}",
                    f"使用检索策略: {', '.join(strategies) if strategies else '暂无'}",
                ]
                return "\n".join(lines)

        except Exception as e:
            logger.warning(f"图记忆召回失败: {e}")
            return ""

    # ==================== Entity Extraction ====================

    def extract_referenced_entities(self, question: str, answer: str) -> Dict[str, Any]:
        """从问答中提取医学实体，用于建立图关系"""
        entities: Dict[str, Any] = {"diseases": [], "department": ""}
        if not self._neo4j_ready or not self.driver:
            return entities

        combined = f"{question} {answer[:500]}"

        try:
            with self.driver.session() as s:
                result = s.run(
                    "MATCH (d:Disease) WHERE size(d.name) >= 2 RETURN d.name AS name"
                )
                disease_names = [r["name"] for r in result]
        except Exception:
            disease_names = []

        for name in disease_names:
            if name and len(name) >= 2 and name in combined:
                if name not in entities["diseases"]:
                    entities["diseases"].append(name)

        dept_keywords = ["内科", "外科", "儿科", "妇产科", "男科", "肿瘤科", "骨科", "眼科", "耳鼻喉科",
                         "皮肤科", "神经科", "心血管", "消化科", "呼吸科", "内分泌", "泌尿", "血液科",
                         "急诊", "ICU", "康复", "中医", "心理"]
        for kw in dept_keywords:
            if kw in combined:
                entities["department"] = kw
                break

        return entities

    # ==================== Helpers ====================

    def get_session_summary(self, session_id: str) -> str:
        if not self._neo4j_ready or not self.driver:
            return ""
        try:
            with self.driver.session() as s:
                result = s.run(
                    "MATCH (ses:Session {session_id: $sid}) RETURN ses.patient_summary AS summary",
                    sid=session_id,
                )
                record = result.single()
                return record["summary"] if record and record["summary"] else ""
        except Exception:
            return ""

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "buffer_size": len(self.buffer),
            "pending_flush": len(self._pending_flush),
            "neo4j_ready": self._neo4j_ready,
            "milvus_ready": self._collection_ready,
            "memory_enabled": self.config.memory_enabled,
        }

    def close(self):
        if self._pending_flush:
            self._flush_to_neo4j(self._pending_flush)
            self._flush_to_milvus(self._pending_flush)
            self._pending_flush.clear()
        logger.info("记忆模块已关闭")
