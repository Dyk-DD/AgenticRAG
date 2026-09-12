"""
临床决策辅助系统 - 对话记忆模块
三层混合记忆架构：
1. 短期缓冲 (deque) — 当前会话最近 N 轮对话，零延迟
2. 图谱记忆 (Neo4j) — 结构化诊断推理链，链接到疾病/科室实体
3. 语义记忆 (Milvus) — 跨会话相似病例召回
"""

import logging
import os
import re
import time
import hashlib
import json
import tempfile
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from pymilvus import DataType, CollectionSchema, FieldSchema

logger = logging.getLogger(__name__)

# 会被拼进 Milvus 过滤表达式的值（session_id / client_id）必须只含这些字符。
# 当前两者都由服务端生成（sess_<ts>_<hex> / p_<hex>），本来就不可伪造；
# 这道校验是防止将来 ID 生成规则改用用户输入后，字符串拼接变成注入点。
_SAFE_FILTER_RE = re.compile(r"[A-Za-z0-9_\-]{1,100}")


def _is_safe_filter_value(value: str) -> bool:
    return bool(value) and bool(_SAFE_FILTER_RE.fullmatch(value))


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
    # 归属患者。语义记忆靠它做隔离——没有它就等于跨患者召回，
    # 这是"病历串号"级别的缺陷，所以默认空串时必须拒绝召回（fail-closed）。
    client_id: str = ""


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
        self._summary_done = False

        # WAL（Write-Ahead Log）：崩溃恢复
        # 必须是三层 dirname。本文件在 app/rag_modules/ 下，比 app/qa_database.py
        # 深一层，所以要多退一级才能回到仓库根（容器里是 /app），再拼 data 才
        # 对得上 rag_state 卷挂载的 /app/data。
        # 只退两级会得到 /app/app/data：容器里该目录不存在，而 /app/app 属 root，
        # 非 root 的 appuser 建不出来，WAL 会静默失效 —— 只在日志里留一行
        # WARNING，功能表面看却正常，崩溃时才会发现未刷新的记忆丢了。
        _root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        self._wal_path = os.path.join(_root, "data", ".memory_wal.jsonl")
        self._recover_from_wal()

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
                # 老版本的集合没有 client_id 字段，语义召回就无法按患者隔离
                # （等于跨患者召回病历）。这种集合直接删掉重建：里面只是派生
                # 缓存，丢掉不影响知识库，而留着就是一个持续泄漏源。
                if not self._has_client_id_field(collection_name):
                    logger.warning(
                        f"Milvus 记忆集合 {collection_name} 缺少 client_id 字段"
                        f"（无法隔离患者），删除重建"
                    )
                    self.milvus_module.client.drop_collection(collection_name)
                else:
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
                # 患者归属。语义召回和删除都按它过滤，是记忆隔离的唯一依据。
                FieldSchema(name="client_id", dtype=DataType.VARCHAR, max_length=100),
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
            # 标量索引不是可选项：召回要按 client_id 过滤、删除要按 session_id
            # 过滤，没有索引就是每次全表扫描。INVERTED 需要 Milvus >= 2.4。
            for scalar_field in ("client_id", "session_id"):
                index_params.add_index(
                    field_name=scalar_field, index_type="INVERTED"
                )
            self.milvus_module.client.create_index(
                collection_name=collection_name, index_params=index_params
            )
            self.milvus_module.client.load_collection(collection_name)
            self._collection_ready = True
            logger.info(f"Milvus 记忆集合 {collection_name} 创建完成")

        except Exception as e:
            logger.warning(f"Milvus 记忆集合初始化失败: {e}")

    def _has_client_id_field(self, collection_name: str) -> bool:
        """判断集合是否已经带 client_id（用于识别需要重建的老集合）"""
        try:
            desc = self.milvus_module.client.describe_collection(collection_name)
            return any(f.get("name") == "client_id" for f in desc.get("fields", []))
        except Exception as e:
            # 判不出来时保守地当作"没有"→ 重建。宁可重建一个空集合，
            # 也不要因为探测失败而继续用一个无法隔离患者的集合。
            logger.warning(f"探测集合 {collection_name} 字段失败，按需重建处理: {e}")
            return False

    # ==================== Session Lifecycle ====================

    def create_session(self) -> str:
        session_id = f"sess_{int(time.time())}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:6]}"
        self.buffer.clear()
        self._pending_flush.clear()
        self._summary_done = False

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
            self._wal_clear()

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

    def delete_session_memory(self, session_id: str) -> Dict[str, bool]:
        """把某会话从所有记忆存储里抹掉：Milvus 语义记忆 + Neo4j 图谱记忆。

        **必须在 close_session() 之后调用。** close_session 会把 _pending_flush
        里的轮次写进 Milvus 和 Neo4j，先删再 flush 等于白删 —— 这正是原来的
        顺序问题（旧代码先删 Neo4j 再 close，Milvus 从来没被删过，而且刷新回来
        的轮次会永久留在里面）。

        返回每个存储的删除是否成功。调用方必须检查：某一个失败就意味着
        "用户以为删干净了，其实还有副本"。
        """
        result = {"milvus": False, "neo4j": False}
        if not session_id or not _is_safe_filter_value(session_id):
            logger.warning(f"session_id 非法，拒绝删除记忆: {session_id!r}")
            return result

        # 1. Milvus 语义记忆
        if self._collection_ready and self.milvus_module.client:
            try:
                self.milvus_module.client.delete(
                    collection_name=self.config.memory_milvus_collection,
                    filter=f'session_id == "{session_id}"',
                )
                result["milvus"] = True
                logger.info(f"已删除 Milvus 中的会话记忆: {session_id}")
            except Exception as e:
                logger.warning(f"删除 Milvus 会话记忆失败: {e}")
        else:
            # 集合没就绪说明本来就没写过，视为已达成
            result["milvus"] = True

        # 2. Neo4j 图谱记忆
        if self._neo4j_ready and self.driver:
            try:
                with self.driver.session() as s:
                    s.run(
                        "MATCH (ses:Session {session_id: $sid}) "
                        "OPTIONAL MATCH (ses)-[:CONTAINS]->(t:Turn) "
                        "DETACH DELETE t, ses",
                        sid=session_id,
                    )
                result["neo4j"] = True
                logger.info(f"已删除 Neo4j 中的会话记忆: {session_id}")
            except Exception as e:
                logger.warning(f"删除 Neo4j 会话记忆失败: {e}")
        else:
            result["neo4j"] = True

        return result

    # ==================== Write-Ahead Log (崩溃恢复) ====================

    def _wal_append(self, turn: ConversationTurn):
        """将未刷新的轮次写入 WAL 文件，系统崩溃后可恢复"""
        try:
            os.makedirs(os.path.dirname(self._wal_path), exist_ok=True)
            with open(self._wal_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    "turn_id": turn.turn_id,
                    "session_id": turn.session_id,
                    "turn_index": turn.turn_index,
                    "question": turn.question,
                    "answer": turn.answer,
                    "strategy": turn.strategy,
                    "timestamp": turn.timestamp,
                    "complexity_score": turn.complexity_score,
                    "disease_entities": turn.disease_entities,
                    "department": turn.department,
                    # 落盘必须带上归属：恢复时不带 client_id 的话，这些记忆就
                    # 变成了无主数据，既不能按患者召回也不能按患者删除。
                    "client_id": turn.client_id,
                }, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.warning(f"WAL 写入失败: {e}")

    def _wal_clear(self):
        """成功刷新后清空 WAL"""
        try:
            if os.path.exists(self._wal_path):
                os.remove(self._wal_path)
        except Exception as e:
            logger.warning(f"WAL 清理失败: {e}")

    def _recover_from_wal(self):
        """启动时检查 WAL，恢复未刷新的数据"""
        if not os.path.exists(self._wal_path):
            return
        try:
            recovered = []
            with open(self._wal_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    turn = ConversationTurn(
                        turn_id=data["turn_id"],
                        session_id=data["session_id"],
                        turn_index=data["turn_index"],
                        question=data["question"],
                        answer=data["answer"],
                        strategy=data["strategy"],
                        timestamp=data["timestamp"],
                        complexity_score=data.get("complexity_score", 0.0),
                        disease_entities=data.get("disease_entities", []),
                        department=data.get("department", ""),
                        # 老 WAL 没有这个键（升级前写入的），取不到就是空串；
                        # 这些记忆因此不会被任何患者召回 —— fail-closed，
                        # 宁可丢掉一条无主的记忆，也不要让它跨患者出现。
                        client_id=data.get("client_id", ""),
                    )
                    recovered.append(turn)

            if recovered:
                logger.info(f"WAL 中发现 {len(recovered)} 条未刷新的记忆，正在恢复...")
                self._flush_to_neo4j(recovered)
                self._flush_to_milvus(recovered)
                self._wal_clear()
                logger.info(f"WAL 恢复完成")
        except Exception as e:
            logger.warning(f"WAL 恢复失败: {e}")

    # ==================== Memory Recording ====================

    def record_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        strategy: str = "unknown",
        complexity: float = 0.0,
        extracted_entities: Optional[Dict[str, Any]] = None,
        client_id: str = "",
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
            # 必须随每条记忆一起落库：没有归属的记忆召回时无法隔离，
            # 也无法在患者删除会话时被一并清除。
            client_id=client_id,
        )

        self.buffer.append(turn)
        self._pending_flush.append(turn)

        # 写入 WAL（崩溃后可恢复）
        self._wal_append(turn)

        if len(self._pending_flush) >= self.config.memory_batch_flush_size:
            self._flush_to_neo4j(self._pending_flush)
            self._flush_to_milvus(self._pending_flush)
            self._pending_flush.clear()
            self._wal_clear()

        if not self._summary_done and len(self.buffer) >= self.config.memory_summary_threshold:
            self._summarize_buffer(session_id)
            self._summary_done = True

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
                        "client_id": t.client_id,
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

    def retrieve_memory_context(self, question: str, session_id: str, client_id: str = "") -> str:
        """
        多源记忆融合检索（修复版）

        修复内容：
        1. 每个来源带优先级标注和时间戳，帮助 LLM 区分可信度
        2. 按重要性顺序拼接（短期缓冲 > 图谱摘要 > 语义回忆），重要内容不被截断
        3. 截断在完整句子/实体边界处，避免切碎关键信息
        """
        if not self.config.memory_enabled:
            return ""

        parts = []

        # 按可靠性降序排列：缓冲 > 图谱 > 语义
        recent = self._format_buffer_context()
        if recent:
            parts.append(("buffer", recent))

        graph_ctx = self._graph_recall(session_id)
        if graph_ctx:
            parts.append(("graph", graph_ctx))

        similar = self._semantic_recall(question, session_id, client_id)
        if similar:
            parts.append(("semantic", similar))

        if not parts:
            return ""

        # 按优先级拼接（缓冲最重要，优先完整保留）
        full_context = ""
        max_chars = self.config.memory_max_context_chars
        remaining = max_chars

        for source_type, content in parts:
            header = f"[来源: {source_type}]"
            if source_type == "buffer":
                source_note = "（当前会话记录，高可靠性）"
            elif source_type == "graph":
                source_note = "（当前会话图谱摘要）"
            else:
                source_note = "（历史相似病例，跨会话参考，可靠性较低）"

            section = f"{header}{source_note}\n{content}\n\n"

            if len(section) <= remaining:
                full_context += section
                remaining -= len(section)
            else:
                # 剩余空间不足时，智能截断（保留完整句子）
                available = remaining - len(header) - len(source_note) - 4  # -\n\n
                if available > 50:
                    truncated = self._smart_truncate(content, available)
                    section = f"{header}{source_note}\n{truncated}\n\n[该部分因长度限制被截断]\n\n"
                    full_context += section
                remaining = 0
                break

        return full_context

    def _smart_truncate(self, text: str, max_chars: int) -> str:
        """在完整句子边界截断，避免切碎关键信息"""
        if len(text) <= max_chars:
            return text

        # 尝试在句号、问号、感叹号处截断
        truncated = text[:max_chars]
        # 从后往前找句子结束符
        for sep in ['。', '？', '！', '\n']:
            last_sep = truncated.rfind(sep)
            if last_sep > max_chars * 0.6:  # 确保截断点不会太靠前
                return truncated[:last_sep + 1] + "\n[...截断]"

        # 如果没有完整句子边界，在完整单词/词语边界截断
        for sep in [' ', '，', '；']:
            last_sep = truncated.rfind(sep)
            if last_sep > max_chars * 0.7:
                return truncated[:last_sep] + "\n[...截断]"

        # 最后兜底：在字符边界截断
        return truncated.rstrip() + "\n[...截断]"

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

    def _semantic_recall(self, question: str, session_id: str, client_id: str = "") -> str:
        if not self._collection_ready:
            return ""

        # fail-closed：没有归属就召回不了任何东西，直接返回空。
        # 曾经的实现不带任何过滤条件，搜的是整个集合 —— 于是 A 患者的问答
        # 会作为「历史相似病例」出现在 B 患者的回答里（病历串号），
        # 而且患者删掉的历史也照样被召回。宁可少一层召回，也不能跨患者。
        if not client_id:
            logger.warning("语义记忆召回缺少 client_id，跳过（避免跨患者召回）")
            return ""
        if not _is_safe_filter_value(client_id):
            logger.warning("client_id 含非法字符，拒绝语义记忆召回")
            return ""

        try:
            query_vector = self.milvus_module.embeddings.embed_query(question)
            collection_name = self.config.memory_milvus_collection

            results = self.milvus_module.client.search(
                collection_name=collection_name,
                data=[query_vector],
                anns_field="vector",
                # 只召回本患者自己的历史。用 client_id 而不是 session_id：
                # 这层的价值就是"跨会话"回忆同一个患者的既往提问，按 session
                # 过滤会退化成与短期缓冲重复。
                filter=f'client_id == "{client_id}"',
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
        """
        图记忆召回（修复版）

        原实现只返回聚合计数和疾病名列表，不包含任何图谱结构信息。
        修复后：
        - 返回当前会话中的实际图谱路径和关系
        - 包含疾病-科室关联、疾病-疾病共现路径
        - 对最近一轮对话建立实体关联推理链
        """
        if not self._neo4j_ready or not self.driver:
            return ""

        try:
            lines = ["【当前会话图谱结构】"]
            with self.driver.session() as s:
                # 1. 获取本轮对话涉及的实体关系和路径
                #    从 Turn 节点出发，通过 REFERENCES_DISEASE / REFERENCES_DEPT 找到实体
                path_result = s.run(
                    """
                    MATCH (ses:Session {session_id: $sid})-[:CONTAINS]->(t:Turn)
                    OPTIONAL MATCH (t)-[:REFERENCES_DISEASE]->(d:Disease)
                    OPTIONAL MATCH (t)-[:REFERENCES_DEPT]->(dept:Department)
                    WITH t, collect(DISTINCT d.name) AS diseases, collect(DISTINCT dept.name) AS departments
                    RETURN t.turn_index AS idx, t.question AS question,
                           t.strategy AS strategy, diseases, departments,
                           t.complexity_score AS complexity
                    ORDER BY t.turn_index
                    LIMIT 10
                    """,
                    sid=session_id,
                )
                records = list(path_result)
                if not records:
                    return ""

                # 2. 构建轮次间的推理路径
                all_diseases = set()
                prev_diseases = set()
                for r in records:
                    diseases = set(r.get("diseases", []) or [])
                    depts = set(r.get("departments", []) or [])
                    idx = r["idx"]
                    question = r["question"][:80]

                    reasoning_link = ""
                    # 检测与前一轮的疾病关系
                    new_diseases = diseases - prev_diseases
                    continued_diseases = diseases & prev_diseases
                    if continued_diseases:
                        reasoning_link = f"（延续前轮疾病: {', '.join(continued_diseases)}）"
                    if new_diseases:
                        reasoning_link += f"（新增: {', '.join(new_diseases)}）"

                    dept_str = f" | 科室: {', '.join(depts)}" if depts else ""
                    lines.append(f"  - 第{idx + 1}轮: {question}{dept_str}{reasoning_link}")
                    all_diseases.update(diseases)
                    prev_diseases = diseases

                # 3. 构建疾病共现路径（从本轮会话中挖掘）
                if len(all_diseases) >= 2:
                    # 查找疾病之间通过 Turn 节点的共现路径
                    path_rels = s.run(
                        """
                        MATCH (ses:Session {session_id: $sid})-[:CONTAINS]->(t:Turn)
                        MATCH (t)-[:REFERENCES_DISEASE]->(d1:Disease)
                        MATCH (t)-[:REFERENCES_DISEASE]->(d2:Disease)
                        WHERE d1.name < d2.name
                        WITH d1, d2, collect(t.turn_index) AS co_turns
                        RETURN d1.name AS disease_a, d2.name AS disease_b,
                               size(co_turns) AS co_occurrence_count
                        ORDER BY co_occurrence_count DESC
                        LIMIT 5
                        """
                    )
                    path_rels_list = list(path_rels)
                    if path_rels_list:
                        lines.append("  📎 疾病共现关系（同一轮讨论的疾病对，提示潜在并发症/关联）:")
                        for pr in path_rels_list:
                            lines.append(f"    - {pr['disease_a']} ↔ {pr['disease_b']} "
                                         f"（共现 {pr['co_occurrence_count']} 次）")

                # 4. 汇总
                lines.insert(1, f"  共 {len(records)} 轮对话 | 涉及 {len(all_diseases)} 种疾病")

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
            self._wal_clear()
        logger.info("记忆模块已关闭")
