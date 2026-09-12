"""
临床辅助系统 - 真正的图RAG检索模块
基于图结构的医学知识推理和检索，而非简单的关键词匹配
"""

import json
import logging
import math
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import re

from langchain_core.documents import Document
from neo4j import GraphDatabase

from .utils import extract_json_from_llm

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """查询类型枚举"""
    ENTITY_RELATION = "entity_relation"  # 实体关系查询：A和B有什么关系？
    MULTI_HOP = "multi_hop"  # 多跳查询：A通过什么连接到C？
    SUBGRAPH = "subgraph"  # 子图查询：A相关的所有信息
    PATH_FINDING = "path_finding"  # 路径查找：从A到B的最佳路径
    CLUSTERING = "clustering"  # 聚类查询：和A相似的都有什么？

@dataclass
class GraphQuery:
    """图查询结构"""
    query_type: QueryType
    source_entities: List[str]
    target_entities: List[str] = None
    relation_types: List[str] = None
    max_depth: int = 2
    max_nodes: int = 50
    constraints: Dict[str, Any] = None

@dataclass
class GraphPath:
    """图路径结构"""
    nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    path_length: int
    relevance_score: float
    path_type: str

@dataclass
class KnowledgeSubgraph:
    """知识子图结构"""
    central_nodes: List[Dict[str, Any]]
    connected_nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    graph_metrics: Dict[str, float]
    reasoning_chains: List[List[str]]

class GraphRAGRetrieval:
    """
        临床决策图RAG检索系统
        核心特点：
        1. 临床意图理解：识别医学图查询模式
        2. 多跳临床推理：药物-症状-科室深度探索
        3. 临床子图提取：相关疾病与问答知识网络
    """
    
    def __init__(self, config, llm_client, milvus_module=None):
        self.config = config
        self.llm_client = llm_client
        self.milvus_module = milvus_module
        self.driver = None

        # 图结构缓存
        self.entity_cache = {}  # 实体缓存
        self.relation_cache = {}    # 关系缓存
        self.subgraph_cache = {}    # 子图缓存

        # 语义重排序候选数上限：BGE 模型推理较慢，控制传入候选数平衡速度与召回
        self.max_rerank_candidates = 150
        
    def initialize(self):
        """初始化图RAG检索系统"""
        logger.info("初始化图RAG检索系统...")
        
        # 连接Neo4j
        try:
            # 创建数据库驱动实例（带超时保护）
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password),
                connection_timeout=10,
            )
            # 测试连接
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Neo4j连接成功")
        except Exception as e:
            logger.error(f"Neo4j连接失败: {e}")
            return
        
        # 预热：构建实体和关系索引
        self._build_graph_index()

    def _build_graph_index(self):
        """构建图索引以加速查询"""
        logger.info("构建临床图结构索引...")

        try:
            with self.driver.session() as session:
                # 移除 n.category，改为可能存在的属性或直接删除
                entity_query = """
                MATCH (n)
                WHERE n.nodeId IS NOT NULL OR n.name IS NOT NULL
                WITH n, COUNT { (n)--() } as degree
                RETURN labels(n) as node_labels, n.nodeId as node_id,
                       n.name as name, n.department as department, degree
                ORDER BY degree DESC
                LIMIT 10000
                """

                result = session.run(entity_query)
                for record in result:
                    node_id = record["node_id"]
                    name = record["name"]
                    # Disease 节点无 nodeId，用 name 兜底作为 key
                    key = node_id or f"name:{name}"
                    self.entity_cache[key] = {
                        "labels": record["node_labels"],
                        "name": name,
                        "department": record.get("department"),
                        "degree": record["degree"]
                    }
                
                # 构建关系类型索引
                relation_query = """
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(r) as frequency
                ORDER BY frequency DESC
                """
                
                result = session.run(relation_query)
                for record in result:
                    rel_type = record["rel_type"]
                    self.relation_cache[rel_type] = record["frequency"]
                    
                logger.info(f"索引构建完成: {len(self.entity_cache)}个实体, {len(self.relation_cache)}个关系类型")
                
        except Exception as e:
            logger.error(f"构建图索引失败: {e}")

    def understand_graph_query(self, query: str) -> GraphQuery:
        """
        理解查询的图结构意图
        核心重构：从自然语言到临床医学图查询的转换
        """
        prompt = f"""你是一个医学图谱查询专家。请将用户的问题映射到医学知识图谱上。

已知图中的节点类型：
- **Disease/Symptom**：疾病名（如"高血压"、"胃炎"、"脂肪肝"）
- **Drug**：药物名（如"党参"、"雷贝拉唑"）
- **Department**：科室名（如"心血管科"、"消化内科"）
- **Consultation**：病例/问答节点（含 title、ask、answer 属性）

主要关系：
- (Consultation)-[:BELONGS_TO_DEPT]->(Department)
- (Consultation)-[:MENTIONS_DISEASE]->(Disease)
- (Drug)-[:TREATS]->(Disease) / (Consultation)-[:MENTIONS_DRUG]->(Drug)

查询：{query}

请分析以下字段，严格按规则执行：

1. query_type：
   - entity_relation：实体间直接关系（高血压和党参有禁忌吗？）
   - multi_hop：多跳推理（高血压患者常去哪个科室？→ 高血压→病例→科室）
   - subgraph：完整子图（心血管科有哪些常见病例？）
   - path_finding：路径查找（从症状到确诊用药的路径）
   - clustering：聚类相似性（和这个病例类似的症状有哪些？）

2. source_entities：**这是最关键的一步，请严格遵循以下规则：**
   - ⛔ **禁止**提取症状描述词（如"肚子疼"、"发烧"、"咳嗽"、"拉肚子"、"头晕"、"恶心"、"乏力"等）。这些是症状描述，不是图中的疾病实体。
   - ✅ 只提取在图中**真实存在**的实体名：
     * 疾病名（如"高血压"、"胃炎"、"脂肪肝"、"糖尿病"）
     * 药物名（如"党参"、"阿莫西林"）
     * 科室名（如"心血管科"、"消化内科"）
   - 如果查询**只描述了症状**（如"肚子疼、拉肚子、发烧"），但没有明确疾病名 → 尝试推断最可能的疾病名（如"肠胃炎"），如果无法确定则返回空列表 []
   - 不要把"怎么办"、"能不能吃"、"推荐"等抽象词放进来

3. target_entities：只在需限制路径终点时填写，同样只填疾病/药物/科室名，不确定则留空 []

4. relation_types：优先考虑的关系类型，如 ["MENTIONS_DISEASE", "BELONGS_TO_DEPT", "TREATS"]

5. max_depth：1-3 的整数

示例：

查询："高血压患者能吃党参吗？"
{{
  "query_type": "entity_relation",
  "source_entities": ["高血压", "党参"],
  "target_entities": [],
  "relation_types": ["MENTIONS_DISEASE", "MENTIONS_DRUG", "TREATS"],
  "max_depth": 2
}}

查询："我这两天肠胃不舒服，恶心，肚子疼，拉肚子"
{{
  "query_type": "multi_hop",
  "source_entities": ["肠胃炎"],
  "target_entities": [],
  "relation_types": [],
  "max_depth": 2
}}

查询："宝宝咳嗽发烧两天了，怎么办？"
{{
  "query_type": "multi_hop",
  "source_entities": ["上呼吸道感染"],
  "target_entities": [],
  "relation_types": [],
  "max_depth": 2
}}

查询："消化内科有哪些常见病？"
{{
  "query_type": "subgraph",
  "source_entities": ["消化内科"],
  "target_entities": [],
  "relation_types": ["BELONGS_TO_DEPT", "MENTIONS_DISEASE"],
  "max_depth": 2
}}

⚠️ **重要**：如果查询只有症状描述且无法推断具体疾病名，source_entities 返回空列表 []。
请严格返回 JSON 对象，不要多余文字。
"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )

            raw_content = response.choices[0].message.content.strip()
            result = extract_json_from_llm(raw_content)
            if result is None:
                raise ValueError("无法从 LLM 输出中提取 JSON")

            return GraphQuery(
                query_type=QueryType(result.get("query_type", "subgraph")),
                source_entities=result.get("source_entities", []),
                target_entities=result.get("target_entities", []),
                relation_types=result.get("relation_types", []),
                max_depth=result.get("max_depth", 2),
                max_nodes=50,
                constraints=result.get("constraints", {})
            )

        except Exception as e:
            logger.error(f"医学查询意图理解失败: {e}")
            return GraphQuery(
                query_type=QueryType.SUBGRAPH,
                source_entities=[query],
                max_depth=2
            )

    def _tokenize_query(self, query: str) -> List[str]:
        """将查询文本分词为查询 Token 列表，用于 Cypher 关键词匹配"""
        if not query or not query.strip():
            return []

        try:
            import jieba
            STOP_WORDS = {
                "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都",
                "一", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
                "没有", "看", "好", "自己", "这", "那", "什么", "怎么", "如何",
                "为什么", "哪个", "能", "来", "吧", "吗", "嗯", "啊", "哦", "呢",
                "呀", "嘛", "得", "地", "把", "被", "让", "从", "向", "跟", "为",
                "因", "所以", "但是", "不过", "因为", "所以", "然后", "之后",
                "已经", "可以", "应该", "需要", "可能", "想要", "比较", "非常",
                "特别", "一直", "有点", "一些", "这个", "那个", "这些", "那些",
                "还是", "请问", "您好", "你好", "谢谢", "感谢", "多久", "多少",
                "怎么办", "怎么回事", "是否", "是不是", "能不能", "有没有",
                "会不会", "要不要", "该不该", "做", "进行", "给予", "引起",
                "导致", "造成", "出现", "产生", "表现", "症状", "情况", "时候",
                "时间", "方法", "方式", "结果", "效果", "影响", "作用", "问题",
                "原因", "目的", "意义", "特点", "性质", "什么", "多久",
            }
            tokens = jieba.lcut(query)
            result = [t.strip() for t in tokens
                      if len(t.strip()) >= 2 and t.strip() not in STOP_WORDS]
            # 去重保持顺序
            seen = set()
            return [t for t in result if not (t in seen or seen.add(t))]
        except Exception:
            # jieba 不可用时退化为按常见分隔符拆分
            tokens = re.split(r'[\s,，。；;：:！!？?、/\\()（）【】\[\]{}]', query)
            return [t.strip() for t in tokens if len(t.strip()) >= 2]

    def multi_hop_traversal(self, graph_query: GraphQuery, top_k: int = 10, query: str = "") -> List[GraphPath]:
        """临床多跳图遍历"""
        # logger.info(f"执行多跳遍历: {graph_query.source_entities} -> {graph_query.target_entities}")
        logger.info(f"执行多跳遍历 | source={graph_query.source_entities}, target={graph_query.target_entities}")
        paths = []
        if not self.driver:
            return paths

        try:
            with self.driver.session() as session:
                source_entities = graph_query.source_entities
                target_keywords = graph_query.target_entities or []
                max_depth = graph_query.max_depth

                # 关键词匹配 token：从查询文本中提取
                query_tokens = self._tokenize_query(query) if query else []
                if not query_tokens:
                    # 无查询文本时，用 source_entities 作为 token
                    query_tokens = source_entities[:]

                # 兼容医疗节点的 target 过滤（全部参数化，无字符串拼接）
                # 兼容医疗节点的 source 匹配 (支持 name 和 title)
                # 注意: 路径长度使用字面量（Neo4j 5+ 不支持参数化路径长度）
                cypher_query = f"""
                UNWIND $source_entities as source_name
                MATCH (source)
                WHERE (source.name IS NOT NULL AND source.name CONTAINS source_name)
                   OR source.nodeId = source_name

                MATCH path = (source)-[*1..{max_depth}]-(target)
                WHERE NOT source = target
                """ + (
                    """
                    AND ANY(kw IN $target_keywords WHERE
                        (target.name IS NOT NULL AND toString(target.name) CONTAINS kw) OR
                        (target.title IS NOT NULL AND toString(target.title) CONTAINS kw) OR
                        (target.department IS NOT NULL AND toString(target.department) CONTAINS kw)
                    )
                    """ if target_keywords else ""
                ) + """
                WITH path, source, target, length(path) as path_len,
                     relationships(path) as rels, nodes(path) as path_nodes
                WITH path, source, target, path_len, rels, path_nodes,
                     /* 新评分公式：路径长度权重减半 + 关键词重叠匹配 */
                     (1.0 / path_len) * 0.4 +
                     (CASE WHEN SIZE($query_tokens) > 0 THEN
                         SIZE([kw IN $query_tokens WHERE
                             (target.title IS NOT NULL AND target.title CONTAINS kw) OR
                             (target.ask IS NOT NULL AND target.ask CONTAINS kw) OR
                             (target.name IS NOT NULL AND target.name CONTAINS kw)
                         ]) * 0.4 / SIZE($query_tokens)
                      ELSE 0.0 END) +
                     (CASE WHEN ANY(r IN rels WHERE type(r) IN $relation_types) THEN 0.2 ELSE 0.0 END) as relevance

                ORDER BY relevance DESC
                LIMIT 500
                RETURN path, source, target, path_len, rels, path_nodes, relevance
                """

                params = {
                    "source_entities": source_entities,
                    "query_tokens": query_tokens,
                    "relation_types": graph_query.relation_types or [],
                    "max_depth": max_depth,
                }
                if target_keywords:
                    params["target_keywords"] = target_keywords

                result = session.run(cypher_query, params)
                all_paths = []
                for record in result:
                    path_data = self._parse_neo4j_path(record)
                    if path_data:
                        all_paths.append(path_data)

                # 后过滤：优先保留含 Consultation 节点（有 qa_xxx nodeId）的路径
                # 扩大候选集：保证语义重排序有足够多的候选文档
                # 注意：候选集过小会导致相关度排名靠后但仍相关的路径被截断，
                #       尤其是当某个疾病关联的 QA 非常多时（如 脂肪肝 有 387+ 条），
                #       目标 ground truth 节点可能在排名 #200+。需要更大的候选池
                #       让语义重排序有机会将相关节点重新排到前面。
                consult_paths = [p for p in all_paths if any(
                    n.get("id", "").startswith("qa_") for n in p.nodes
                )]
                other_paths = [p for p in all_paths if p not in consult_paths]
                # 至少保留 top_k*50 个候选（至少 500），保证大关联量场景下 ground truth 不被截断
                max_candidates = min(len(consult_paths), max(top_k * 50, 500)) if consult_paths else 50
                if len(consult_paths) >= 5:
                    paths = consult_paths[:max_candidates]
                else:
                    paths = consult_paths + other_paths[:max(5, max_candidates - len(consult_paths))]
                paths = paths[:max_candidates]
        except Exception as e:
            logger.error(f"临床多跳遍历失败: {e}")

        return paths

    def extract_knowledge_subgraph(self, graph_query: GraphQuery) -> KnowledgeSubgraph:
        """提取临床知识子图"""
        logger.info(f"提取临床知识子图: {graph_query.source_entities}")
        if not self.driver:
            return self._fallback_subgraph_extraction(graph_query)

        try:
            with self.driver.session() as session:
                cypher_query = f"""
                UNWIND $source_entities as entity_name
                MATCH (source)
                WHERE (source.name IS NOT NULL AND source.name CONTAINS entity_name) 
                   OR source.nodeId = entity_name

                MATCH (source)-[r*1..{graph_query.max_depth}]-(neighbor)
                WITH source, collect(DISTINCT neighbor) as neighbors, collect(DISTINCT r) as relationships
                WHERE size(neighbors) <= $max_nodes

                WITH source, neighbors, relationships, size(neighbors) as node_count, size(relationships) as rel_count
                RETURN source,
                       neighbors[0..{graph_query.max_nodes}] as nodes,
                       relationships[0..{graph_query.max_nodes}] as rels,
                       {{
                           node_count: node_count,
                           relationship_count: rel_count,
                           density: CASE WHEN node_count > 1 THEN toFloat(rel_count) / (node_count * (node_count - 1) / 2) ELSE 0.0 END
                       }} as metrics
                """
                result = session.run(cypher_query, {"source_entities": graph_query.source_entities,
                                                    "max_nodes": graph_query.max_nodes})
                record = result.single()
                if record:
                    return self._build_knowledge_subgraph(record)
        except Exception as e:
            logger.error(f"医学子图提取失败: {e}")

        return self._fallback_subgraph_extraction(graph_query)

    def graph_structure_reasoning(self, subgraph: KnowledgeSubgraph, query: str) -> List[str]:
        """
        基于临床图谱的结构推理：
        模拟医生的临床思维，通过图谱关系进行诊断推理或用药推断
        """
        reasoning_chains = []

        try:
            # 1. 识别临床推理模式
            reasoning_patterns = self._identify_reasoning_patterns(subgraph)

            # 2. 构建临床推理链 (如: 症状A -> 疑似疾病B -> 推荐药物C)
            for pattern in reasoning_patterns:
                chain = self._build_reasoning_chain(pattern, subgraph)
                if chain:
                    reasoning_chains.append(chain)

            # 3. 验证推理链的医学可信度/安全性
            validated_chains = self._validate_reasoning_chains(reasoning_chains, query)

            logger.info(f"临床图结构推理完成，生成 {len(validated_chains)} 条医学推理链")
            return validated_chains

        except Exception as e:
            logger.error(f"临床图结构推理失败: {e}")
            return []

    def adaptive_query_planning(self, query: str) -> List[GraphQuery]:
        """
        自适应临床查询规划：根据病情查询的复杂度动态调整检索策略
        """
        # 分析患者/医生查询的临床复杂度
        complexity_score = self._analyze_query_complexity(query)

        query_plans = []

        if complexity_score < 0.3:
            # 简单查询：直接实体关系查询（如：挂什么科？吃什么药？）
            plan = GraphQuery(
                query_type=QueryType.ENTITY_RELATION,
                source_entities=[query],
                max_depth=1,
                max_nodes=20
            )
            query_plans.append(plan)

        elif complexity_score < 0.7:
            # 中等复杂度：多跳临床路径查询（如：由某种症状推导科室及常见用药）
            plan = GraphQuery(
                query_type=QueryType.MULTI_HOP,
                source_entities=[query],
                max_depth=2,
                max_nodes=50
            )
            query_plans.append(plan)

        else:
            # 复杂查询：医学子图提取 + 深度路径寻找（如：带有基础病的复杂症状鉴别诊断）
            plan1 = GraphQuery(
                query_type=QueryType.SUBGRAPH,
                source_entities=[query],
                max_depth=3,  # 提取更深层级的并发症或关联风险
                max_nodes=100
            )
            plan2 = GraphQuery(
                query_type=QueryType.PATH_FINDING,  # 查找从当前症状到治疗方案的完整临床路径
                source_entities=[query],
                max_depth=3,
                max_nodes=50
            )
            query_plans.extend([plan1, plan2])

        return query_plans
    
    def _validate_graph_entities(self, entities: List[str]) -> List[str]:
        """
        验证实体是否在 entity_cache 中有匹配。
        返回经过去重和验证的有效实体列表（空列表 = 全部无效）。
        """
        if not entities or not self.entity_cache:
            return []

        valid = []
        # 构建 entity_cache 中的文本索引
        entity_names = []
        for cid, cached in self.entity_cache.items():
            name = cached.get("name", "")
            if name and len(name) >= 2:
                entity_names.append(name.lower().strip())

        for entity in entities:
            e_lower = entity.lower().strip()
            if not e_lower or len(e_lower) < 2:
                continue
            # 检查实体名是否在 cache 中存在（子串匹配 | 完全匹配）
            for cached_name in entity_names:
                if e_lower in cached_name or cached_name in e_lower:
                    valid.append(entity)
                    break

        return list(set(valid))  # 去重

    def _fallback_extract_entities(self, query: str) -> List[str]:
        """
        兜底实体提取 — 当 LLM 提取的实体在图中找不到时，
        直接用 entity_cache 中的实体名对查询文本做子串匹配。
        """
        if not self.entity_cache:
            return []

        query_lower = query.lower().strip()
        matched = []

        for cid, cached in self.entity_cache.items():
            name = cached.get("name", "")
            if not name or len(name) < 2:
                continue
            name_lower = name.lower().strip()
            if name_lower in query_lower:
                matched.append(name)

        # 按长度降序，优先最长匹配（最精确的实体）
        matched.sort(key=lambda x: (len(x), x), reverse=True)
        return matched[:5]

    def _vector_as_supplement(self, query: str, top_k: int) -> List[Document]:
        """
        向量补充搜索：从 Milvus 获取语义相似结果，补充图结构检索的盲区。

        当目标 QA 在 Neo4j 中不存在或未被正确关联到疾病节点时，
        图遍历无法找到它，但向量搜索仍然能通过语义匹配召回。
        """
        if not self.milvus_module:
            return []
        try:
            raw = self.milvus_module.similarity_search(query, k=top_k)
            docs = []
            seen_ids = set()
            for r in raw:
                nid = r.get("metadata", {}).get("node_id", "")
                if not nid or nid in seen_ids:
                    continue
                seen_ids.add(nid)
                content = r.get("text", "")
                metadata = r["metadata"]
                metadata["search_type"] = "graph_vector_supplement"
                metadata["vector_score"] = r.get("score", 0.0)
                docs.append(Document(page_content=content, metadata=metadata))
            logger.info(f"向量补充搜索召回 {len(docs)} 个文档")
            return docs
        except Exception as e:
            logger.warning(f"向量补充搜索失败: {e}")
            return []

    def _fuse_by_node_id(
        self,
        graph_docs: List[Document],
        vector_docs: List[Document],
        query: str,
        top_k: int,
    ) -> List[Document]:
        """
        按 node_id 去重 + 分数融合。

        融合策略（与 _semantic_rerank 一致的权重）:
          - graph doc: fused = 0.4 * norm_graph + 0.6 * semantic (计算的)
          - vector doc: fused = 0.3 * 0.5 + 0.7 * vector_score
          - 同一 node_id 同时出现在 graph + vector 时取 graph 版（保留图结构元数据）

        Returns top_k 个文档。
        """
        # 1) 构建 graph 索引
        graph_map = {}  # node_id -> doc
        for d in graph_docs:
            nid = (d.metadata.get("node_id") or "") or d.metadata.get("chunk_id", "")
            if nid:
                graph_map[nid] = d

        # 2) 计算 graph docs 的融合分 (复用 _semantic_rerank 的余弦相似度逻辑)
        if graph_docs:
            graph_docs = self._semantic_rerank(graph_docs, query)
            for d in graph_docs:
                fused = d.metadata.get("fused_score")
                d.metadata["_fused"] = fused if fused else d.metadata.get("relevance_score", 0.0)

        # 3) 合并 vector docs
        seen = set()
        fused_list = []

        # 先处理 graph docs（保留原有评分和融合分）
        for d in graph_docs:
            nid = (d.metadata.get("node_id") or "") or d.metadata.get("chunk_id", "")
            if nid in seen:
                continue
            seen.add(nid)
            fused_list.append((d.metadata.get("_fused", 0.0), d))

        # 再处理 vector docs（只补充 graph 中没有的）
        if not self.milvus_module:
            # 没有 embeddings 做余弦打分 → 直接用 vector_score
            for d in vector_docs:
                nid = (d.metadata.get("node_id") or "") or d.metadata.get("chunk_id", "")
                if nid in seen:
                    continue
                seen.add(nid)
                vs = d.metadata.get("vector_score", 0.0)
                fused = 0.3 * 0.5 + 0.7 * vs  # graph 部分给默认 0.5
                d.metadata["_fused"] = fused
                fused_list.append((fused, d))
        else:
            # 有 embeddings → 对 vector docs 也做语义重排序
            vector_reranked = self._semantic_rerank(vector_docs, query)
            for d in vector_reranked:
                nid = (d.metadata.get("node_id") or "") or d.metadata.get("chunk_id", "")
                if nid in seen:
                    continue
                seen.add(nid)
                fused = d.metadata.get("fused_score", 0.0)
                if not fused:
                    fused = 0.3 * 0.5 + 0.7 * d.metadata.get("vector_score", 0.0)
                d.metadata["_fused"] = fused
                fused_list.append((fused, d))

        # 4) 按融合分降序 → top_k
        fused_list.sort(key=lambda x: x[0], reverse=True)
        final = [doc for _, doc in fused_list[:top_k]]

        logger.info(
            f"图+向量融合完成: graph={len(graph_docs)}, vector={len(vector_docs)}, "
            f"融合后={len(final)}"
        )
        return final

    def graph_rag_search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        图RAG主搜索接口：纯图结构遍历检索

        流程:
        1. 查询意图理解 → 实体识别
        2. 多跳图遍历 → 路径提取（大候选池：LIMIT 2000 → 取 top_k*10）
        3. 路径转文档 → 图结构相关性排序 → 语义重排序
        4. 返回 top_k 个结果
        """
        logger.info(f"开始图RAG检索: {query}")

        if not self.driver:
            logger.warning("Neo4j连接未建立，返回空结果")
            return []

        # 1. 实体提取：先字典匹配（最快、最准），无效则 LLM 兜底
        dict_entities = self._fallback_extract_entities(query)
        if dict_entities:
            logger.info(f"✅ 字典匹配提取实体成功: {dict_entities}")
            graph_query = GraphQuery(
                query_type=QueryType.ENTITY_RELATION,
                source_entities=dict_entities,
                target_entities=[],
                relation_types=[],
                max_depth=2,
            )
        else:
            # LLM 提取（最灵活，适合复杂查询）
            graph_query = self.understand_graph_query(query)
            logger.info(f"查询类型: {graph_query.query_type.value}")

            # 验证 LLM 提取的实体在图中是否真实存在
            if graph_query.source_entities:
                validated = self._validate_graph_entities(graph_query.source_entities)
                if validated:
                    graph_query.source_entities = validated
                    logger.info(f"LLM 实体验证通过: {validated}")
                else:
                    logger.warning(
                        f"LLM 实体 {graph_query.source_entities} 未在图中找到，"
                        f"尝试直接用原始实体做图遍历"
                    )

        graph_results = []

        try:
            # 2. 图结构检索（纯图遍历，不依赖向量搜索）
            if graph_query.query_type in [QueryType.MULTI_HOP, QueryType.PATH_FINDING]:
                paths = self.multi_hop_traversal(graph_query, top_k=top_k, query=query)
                graph_results.extend(self._paths_to_documents(paths, query))

            elif graph_query.query_type in [QueryType.SUBGRAPH, QueryType.CLUSTERING]:
                subgraph = self.extract_knowledge_subgraph(graph_query)
                reasoning_chains = self.graph_structure_reasoning(subgraph, query)
                graph_results.extend(self._subgraph_to_documents(subgraph, reasoning_chains, query))

            elif graph_query.query_type == QueryType.ENTITY_RELATION:
                paths = self.multi_hop_traversal(graph_query, top_k=top_k, query=query)
                graph_results.extend(self._paths_to_documents(paths, query))

            # 3. 图结构相关性排序
            graph_results = self._rank_by_graph_relevance(graph_results, query)

            # 4. 语义重排序候选截断：只对 top-N 做 BGE 嵌入（避免全量推理太慢）
            #    新关键词评分公式已将目标节点排到前列，截断后仍能覆盖
            rerank_pool = graph_results[:self.max_rerank_candidates]
            logger.info(f"语义重排序候选: {len(rerank_pool)}/{len(graph_results)} 个文档")

            # 5. 语义重排序（在图结构排序基础上，用嵌入模型余弦相似度微调排序）
            reranked = self._semantic_rerank(rerank_pool, query)

            # 6. 合并结果：重排序的 top 在前，其余候选按原序接在后面
            graph_results = reranked + graph_results[self.max_rerank_candidates:]

            # 5. 输出结果
            returned_ids = [
                d.metadata.get("node_id", "?")[:20] for d in graph_results[:top_k]
            ]
            logger.info(f"图RAG检索完成，返回 {len(graph_results[:top_k])} 个结果")
            logger.debug(f"返回 node_id 列表: {returned_ids}")
            return graph_results[:top_k]

        except Exception as e:
            logger.error(f"图RAG检索失败: {e}")
            return []
    
    # ========== 辅助方法 ==========
    
    def _parse_neo4j_path(self, record) -> Optional[GraphPath]:
        """解析Neo4j路径记录"""
        try:
            path_nodes = []
            for node in record["path_nodes"]:
                # 优先取 name，没有就取 title，再没有就取 department，否则返回 未知实体
                node_name = node.get("name") or node.get("title") or node.get("department") or "未知实体"
                path_nodes.append({
                    "id": node.get("nodeId", ""),
                    "name": node_name,  # 修改这里
                    "labels": list(node.labels),
                    "properties": dict(node)
                })
            
            relationships = []
            for rel in record["rels"]:
                relationships.append({
                    "type": type(rel).__name__,
                    "properties": dict(rel)
                })
            
            return GraphPath(
                nodes=path_nodes,
                relationships=relationships,
                path_length=record["path_len"],
                relevance_score=record["relevance"],
                path_type="multi_hop"
            )
            
        except Exception as e:
            logger.error(f"路径解析失败: {e}")
            return None
    
    def _build_knowledge_subgraph(self, record) -> KnowledgeSubgraph:
        """构建知识子图对象"""
        try:
            central_nodes = [dict(record["source"])]
            connected_nodes = [dict(node) for node in record["nodes"]]
            relationships = [dict(rel) for rel in record["rels"]]
            
            return KnowledgeSubgraph(
                central_nodes=central_nodes,
                connected_nodes=connected_nodes,
                relationships=relationships,
                graph_metrics=record["metrics"],
                reasoning_chains=[]
            )
        except Exception as e:
            logger.error(f"构建知识子图失败: {e}")
            return KnowledgeSubgraph(
                central_nodes=[],
                connected_nodes=[],
                relationships=[],
                graph_metrics={},
                reasoning_chains=[]
            )

    def _fetch_consultations(self, node_ids: List[str]) -> Dict[str, Dict]:
        """批量从 Neo4j 获取 Consultation 完整内容（title/ask/answer/department）"""
        if not self.driver or not node_ids:
            return {}

        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    UNWIND $node_ids AS nid
                    MATCH (c:Consultation {nodeId: nid})
                    RETURN c.nodeId AS node_id, c.title AS title,
                           c.ask AS ask, c.answer AS answer,
                           c.department AS department
                    """,
                    {"node_ids": node_ids},
                )
                consultations = {}
                for record in result:
                    consultations[record["node_id"]] = {
                        "title": record.get("title", "") or "",
                        "ask": record.get("ask", "") or "",
                        "answer": record.get("answer", "") or "",
                        "department": record.get("department", "") or "",
                    }
                return consultations
        except Exception as e:
            logger.error(f"批量获取 Consultation 失败: {e}")
            return {}

    def _paths_to_documents(self, paths: List[GraphPath], query: str) -> List[Document]:
        """
        将图路径转换为 Document 对象
        优先提取每条路径中包含的 Consultation 节点的完整 QA 内容；
        仅当路径不含 Consultation 时才回退到路径描述。
        """
        if not paths:
            return []

        # === 1. 收集所有路径中的 Consultation node_id ===
        consultation_ids = set()
        for path in paths:
            for node in path.nodes:
                nid = node.get("id") or node.get("properties", {}).get("nodeId", "")
                if nid and nid.startswith("qa_"):
                    consultation_ids.add(nid)

        # === 2. 从 Neo4j 批量获取 Consultation 详情 ===
        consultation_map = self._fetch_consultations(list(consultation_ids))

        # === 3. 构建 Document 列表 ===
        documents = []
        seen_consultation_ids = set()
        seen_path_descs = set()

        for path in paths:
            # 提取路径中的疾病/科室名用于标题
            disease_names = []
            dept_name = ""
            for node in path.nodes:
                labels = node.get("labels", [])
                if any("Disease" in str(l) for l in labels) and node.get("name"):
                    disease_names.append(node["name"])
                if any("Department" in str(l) for l in labels) and node.get("name"):
                    dept_name = node["name"]

            # --- 优先用路径中的 Consultation 节点生成 Document ---
            found_consultation = False
            for node in path.nodes:
                nid = node.get("id") or node.get("properties", {}).get("nodeId", "")
                if nid and nid.startswith("qa_") and nid not in seen_consultation_ids:
                    seen_consultation_ids.add(nid)
                    consult = consultation_map.get(nid, {})
                    title = consult.get("title") or node.get("name") or node.get("title") or "医学问答"
                    ask = consult.get("ask", "")
                    answer = consult.get("answer", "")

                    if ask and answer:
                        page_content = f"【{title}】\n问：{ask}\n答：{answer}"
                    else:
                        page_content = self._build_path_description(path)

                    metadata = {
                        "search_type": "medical_graph_consultation",
                        "node_id": nid,
                        "chunk_id": f"{nid}_full",
                        "title": title,
                        "department": consult.get("department") or dept_name,
                        "relevance_score": path.relevance_score,
                        "path_length": path.path_length,
                        "diseases": disease_names[:5],
                    }
                    documents.append(Document(page_content=page_content, metadata=metadata))
                    found_consultation = True
                    break  # 每个 path 只取第一个未见的 Consultation

            # --- 回退：该路径没有 Consultation 节点 → 用路径描述 ---
            if not found_consultation:
                path_desc = self._build_path_description(path)
                if path_desc not in seen_path_descs:
                    seen_path_descs.add(path_desc)
                    medical_name = (
                        path.nodes[0].get("name", path.nodes[0].get("title", "临床图谱结果"))
                        if path.nodes else "临床图谱结果"
                    )
                    # ä¸º fallback è·¯å¾æå node_id
                    node_id = next((n.get("id", "") for n in path.nodes if n.get("id", "").startswith("qa_")), "")
                    metadata = {
                        "search_type": "medical_graph_path",
                        "path_length": path.path_length,
                        "relevance_score": path.relevance_score,
                        "title": medical_name,
                        "diseases": disease_names[:5],
                        "node_id": node_id,
                        "chunk_id": f"{node_id}_full" if node_id else "",
                    }
                    documents.append(Document(page_content=path_desc, metadata=metadata))

        return documents
    
    def _subgraph_to_documents(self, subgraph: KnowledgeSubgraph,
                              reasoning_chains: List[str], query: str) -> List[Document]:
        """
        将知识子图转换为 Document 列表。

        优先提取子图中所有 Consultation 节点，为每个生成独立的 QA Document；
        并补充一张子图描述文档。这样既保留了完整的内容覆盖，也确保每个
        Consultation 的 node_id 能参与 ground truth 匹配。
        """
        documents = []
        all_nodes = subgraph.central_nodes + subgraph.connected_nodes

        # 1. 收集子图中所有 Consultation node_id
        consultation_ids = []
        for node in all_nodes:
            if isinstance(node, dict):
                nid = node.get("nodeId", "")
                if nid and nid.startswith("qa_"):
                    consultation_ids.append(nid)

        # 2. 批量获取 Consultation 完整内容 → 每个生成独立 Document
        if consultation_ids:
            consultation_map = self._fetch_consultations(list(set(consultation_ids)))
            seen_ids = set()
            for nid in consultation_ids:
                if nid in seen_ids:
                    continue
                seen_ids.add(nid)
                consult = consultation_map.get(nid, {})
                title = consult.get("title") or "医学问答"
                ask = consult.get("ask", "")
                answer = consult.get("answer", "")

                if ask and answer:
                    page_content = f"【{title}】\n问：{ask}\n答：{answer}"
                else:
                    page_content = title

                # 从子图节点中提取疾病/科室信息（用于元数据）
                disease_names = []
                for node in all_nodes:
                    labels = node.get("labels", [])
                    if any("Disease" in str(l) for l in labels) and node.get("name"):
                        disease_names.append(node["name"])

                metadata = {
                    "search_type": "medical_subgraph_consultation",
                    "node_id": nid,
                    "chunk_id": f"{nid}_full",
                    "title": title,
                    "department": consult.get("department", ""),
                    "diseases": disease_names[:5],
                    "relevance_score": 1.0,
                    "reasoning_chains": reasoning_chains,
                }
                documents.append(Document(page_content=page_content, metadata=metadata))

        # 3. 补充子图整体描述（如没有 Consultation 节点则作为兜底）
        subgraph_desc = self._build_subgraph_description(subgraph)
        medical_name = subgraph.central_nodes[0].get("name", subgraph.central_nodes[0].get("title",
                                                                                           "医学子图")) if subgraph.central_nodes else "医学子图"

        meta_subgraph = {
            "search_type": "medical_subgraph",
            "node_count": len(subgraph.connected_nodes),
            "relationship_count": len(subgraph.relationships),
            "graph_density": subgraph.graph_metrics.get("density", 0.0),
            "reasoning_chains": reasoning_chains,
            "title": medical_name,
        }
        if consultation_ids:
            meta_subgraph["node_id"] = consultation_ids[0]
            meta_subgraph["chunk_id"] = f"{consultation_ids[0]}_full"
        documents.append(Document(page_content=subgraph_desc, metadata=meta_subgraph))

        return documents
    
    def _build_path_description(self, path: GraphPath) -> str:
        """构建路径的自然语言描述"""
        if not path.nodes:
            return "空路径"
            
        desc_parts = []
        for i, node in enumerate(path.nodes):
            desc_parts.append(node.get("name", f"节点{i}"))
            if i < len(path.relationships):
                rel_type = path.relationships[i].get("type", "相关")
                desc_parts.append(f" --{rel_type}--> ")
        
        return "".join(desc_parts)
    
    def _build_subgraph_description(self, subgraph: KnowledgeSubgraph) -> str:
        central_names = [str(node.get("name") or node.get("title", "未知")) for node in subgraph.central_nodes]
        return f"关于【{', '.join(central_names)}】的临床知识网络，包含 {len(subgraph.connected_nodes)} 个相关临床概念和 {len(subgraph.relationships)} 个医学关系。"
    
    def _rank_by_graph_relevance(self, documents: List[Document], query: str) -> List[Document]:
        """基于图结构相关性排序"""
        return sorted(documents,
                     key=lambda x: x.metadata.get("relevance_score", 0.0),
                     reverse=True)

    def _semantic_rerank(self, documents: List[Document], query: str) -> List[Document]:
        """
        语义重排序：使用嵌入模型对候选文档做向量相似度重排。

        在图结构排序的基础上，引入语义维度——确保与查询文本最相关
        的 Consultation 节点排到前列，而非仅靠节点度数/路径长度。

        使用融合评分: final = 0.4 * graph_relevance + 0.6 * semantic_similarity
        """
        if not documents:
            return documents

        # 没有嵌入模型时降级为原顺序
        if not self.milvus_module or not hasattr(self.milvus_module, 'embeddings'):
            logger.info("语义重排序不可用（无嵌入模型），使用图结构排序结果")
            return documents

        try:
            embeddings = self.milvus_module.embeddings

            # 获取查询向量
            query_vec = embeddings.embed_query(query)

            # 批量获取文档向量
            doc_texts = []
            for doc in documents:
                text = doc.page_content
                if not text or len(text.strip()) < 5:
                    text = doc.metadata.get("title", "") or "医学参考"
                doc_texts.append(text[:512])  # 截断以免超长

            doc_vecs = embeddings.embed_documents(doc_texts)

            # 计算余弦相似度并融合评分
            def _cosine_sim(a, b):
                dot = sum(x * y for x, y in zip(a, b))
                na = math.sqrt(sum(x * x for x in a))
                nb = math.sqrt(sum(x * x for x in b))
                return dot / (na * nb) if na > 0 and nb > 0 else 0.0

            scored = []
            for i, doc in enumerate(documents):
                semantic_score = _cosine_sim(query_vec, doc_vecs[i])
                graph_score = doc.metadata.get("relevance_score", 0.0)
                # 归一化 graph_score 到 [0, 1] 范围（原始分通常 0.5~2.5）
                norm_graph = min(graph_score / 3.0, 1.0)
                # 融合评分：0.1 图结构 + 0.9 语义（语义主导排序）
                fused = 0.1 * norm_graph + 0.9 * semantic_score
                scored.append((fused, semantic_score, doc))

            # 按融合评分降序
            scored.sort(key=lambda x: x[0], reverse=True)

            re_ranked = []
            for fused, sem_score, doc in scored:
                doc.metadata["semantic_score"] = round(sem_score, 4)
                doc.metadata["fused_score"] = round(fused, 4)
                re_ranked.append(doc)

            logger.info(
                f"语义重排序完成: 最高语义分={scored[0][1]:.4f}, "
                f"最低={scored[-1][1]:.4f}"
            )
            return re_ranked

        except Exception as e:
            logger.warning(f"语义重排序失败，使用图结构排序结果: {e}")
            return documents

    def _analyze_query_complexity(self, query: str) -> float:
        """
        分析医学查询复杂度
        引入临床场景下的高频复杂词汇来判断查询深度
        """
        # 将原有的通用词替换为医学诊断与治疗的特征词汇
        complexity_indicators = [
            "并发症", "副作用", "禁忌", "相互作用", "机理",
            "鉴别诊断", "预后", "风险", "为什么", "方案",
            "区别", "严重", "怎么治"
        ]

        # 计算特征词命中率
        score = sum(1 for indicator in complexity_indicators if indicator in query)

        # 医疗查询通常较短但意图复杂，适当放大得分权重
        return min((score / len(complexity_indicators)) * 2.5, 1.0)
    
    def _identify_reasoning_patterns(self, subgraph: KnowledgeSubgraph) -> List[Dict]:
        """
        基于实际子图数据识别临床推理模式
        分析子图中的节点类型和关系，识别症状→疾病、疾病→药物、药物禁忌等模式
        """
        patterns = []

        node_labels = set()
        node_names = set()
        rel_types = set()

        for node in subgraph.central_nodes + subgraph.connected_nodes:
            if isinstance(node, dict):
                labels = node.get('labels', []) or node.get('_labels', [])
                if isinstance(labels, (list, tuple)):
                    for label in labels:
                        node_labels.add(label)
                name = node.get('name', '') or node.get('title', '')
                if name:
                    node_names.add(name)

        for rel in subgraph.relationships:
            if isinstance(rel, dict):
                rel_types.add(rel.get('type', '') or rel.get('_type', ''))

        # 模式1: 疾病-科室关联路径
        if 'Disease' in node_labels and 'Department' in node_labels:
            patterns.append({
                "type": "disease_dept_pathway",
                "description": "疾病-科室关联路径",
                "nodes": list(node_names),
                "relevance": "high"
            })

        # 模式2: 多疾病共现（提示并发症关联）
        disease_nodes = [
            n for n in subgraph.central_nodes + subgraph.connected_nodes
            if isinstance(n, dict) and (
                'Disease' in (n.get('labels', []) or n.get('_labels', []))
            )
        ]
        if len(disease_nodes) >= 2:
            disease_names = [
                n.get('name', '') for n in disease_nodes if n.get('name', '')
            ]
            patterns.append({
                "type": "disease_comorbidity",
                "description": f"多疾病共现（潜在并发症关联）: {', '.join(disease_names[:5])}",
                "nodes": disease_names,
                "relevance": "high"
            })

        # 模式3: 药物相关推理
        if 'Drug' in node_labels:
            drug_nodes = [
                n for n in subgraph.central_nodes + subgraph.connected_nodes
                if isinstance(n, dict) and 'Drug' in (n.get('labels', []) or n.get('_labels', []))
            ]
            drug_names = [n.get('name', '') for n in drug_nodes if n.get('name', '')]
            patterns.append({
                "type": "drug_related",
                "description": f"涉及药物: {', '.join(drug_names[:3])}",
                "nodes": drug_names,
                "relevance": "high"
            })

        # 模式4: 关联密集度提示综合推理
        node_count = len(subgraph.central_nodes) + len(subgraph.connected_nodes)
        rel_count = len(subgraph.relationships)
        if node_count > 0 and rel_count / node_count > 0.5:
            patterns.append({
                "type": "dense_connection",
                "description": f"高密度关联网络（{node_count}节点, {rel_count}关系），需要综合分析",
                "nodes": list(node_names),
                "relevance": "medium"
            })

        if not patterns:
            patterns.append({
                "type": "basic_lookup",
                "description": "基础信息查询",
                "nodes": list(node_names),
                "relevance": "low"
            })

        return patterns

    def _build_reasoning_chain(self, pattern: Dict, subgraph: KnowledgeSubgraph) -> Optional[str]:
        """
        基于实际图数据构建推理链
        从子图中提取具体的节点-关系-节点三元组，组成可读的推理路径
        """
        if pattern["type"] == "disease_dept_pathway":
            chains = []
            for node in subgraph.central_nodes + subgraph.connected_nodes:
                if isinstance(node, dict) and 'Disease' in (node.get('labels', []) or node.get('_labels', [])):
                    disease_name = node.get('name', '')
                    for rel in subgraph.relationships:
                        if not isinstance(rel, dict):
                            continue
                        if rel.get('type') == 'BELONGS_TO_DEPT' or rel.get('_type') == 'BELONGS_TO_DEPT':
                            chains.append(f"{disease_name} -> 对应科室")
            if chains:
                return "; ".join(chains[:3])
            return f"疾病-科室关联: {pattern.get('description', '')}"

        elif pattern["type"] == "disease_comorbidity":
            nodes = pattern.get("nodes", [])
            if len(nodes) >= 2:
                pairs = []
                for i in range(min(len(nodes), 4)):
                    for j in range(i + 1, min(len(nodes), 4)):
                        pairs.append(f"{nodes[i]} <-> {nodes[j]}（同一病例共现，需关注潜在关联）")
                return " | ".join(pairs[:3])
            return pattern.get("description", "")

        elif pattern["type"] == "drug_related":
            disease_names = set()
            for node in subgraph.central_nodes + subgraph.connected_nodes:
                if isinstance(node, dict) and 'Disease' in (node.get('labels', []) or node.get('_labels', [])):
                    name = node.get('name', '')
                    if name:
                        disease_names.add(name)
            drug_names = pattern.get("nodes", [])
            if disease_names and drug_names:
                inferences = []
                for d in list(disease_names)[:3]:
                    for dr in drug_names[:3]:
                        inferences.append(f"{dr} <-> {d}（药物与疾病存在关联，需核查禁忌/适应症）")
                return " | ".join(inferences[:3])
            return pattern.get("description", "")

        elif pattern["type"] == "dense_connection":
            rel_chains = []
            for i, rel in enumerate(subgraph.relationships[:5]):
                if isinstance(rel, dict):
                    rel_chains.append(f"关系链{i + 1}: {rel.get('type', rel.get('_type', '未知'))}")
            if rel_chains:
                return "; ".join(rel_chains)
            return pattern.get("description", "")

        return pattern.get("description", "")

    def _validate_reasoning_chains(self, chains: List[str], query: str) -> List[str]:
        """
        验证推理链的医学可信度
        非空过滤、去重、按查询相关性排序
        """
        if not chains:
            return []
        seen = set()
        unique = []
        for chain in chains:
            if chain and chain not in seen:
                seen.add(chain)
                unique.append(chain)

        query_terms = set(query.lower())
        scored = []
        for chain in unique:
            chain_lower = chain.lower()
            overlap = sum(1 for term in query_terms if term in chain_lower and term.strip())
            scored.append((chain, overlap))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [chain for chain, _ in scored[:5]]
    
    def _find_entity_relations(self, graph_query: GraphQuery, session) -> List[GraphPath]:
        """查找实体间关系"""
        return []
    
    def _find_shortest_paths(self, graph_query: GraphQuery, session) -> List[GraphPath]:
        """查找最短路径"""
        return []
    
    def _fallback_subgraph_extraction(self, graph_query: GraphQuery) -> KnowledgeSubgraph:
        """降级子图提取"""
        return KnowledgeSubgraph(
            central_nodes=[],
            connected_nodes=[],
            relationships=[],
            graph_metrics={},
            reasoning_chains=[]
        )
    
    def close(self):
        """关闭资源连接"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
            logger.info("临床图RAG检索系统已关闭")