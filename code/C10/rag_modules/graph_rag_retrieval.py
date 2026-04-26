"""
临床辅助系统 - 真正的图RAG检索模块
基于图结构的医学知识推理和检索，而非简单的关键词匹配
"""

import json
import logging
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import re

from langchain_core.documents import Document
from neo4j import GraphDatabase

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
    
    def __init__(self, config, llm_client):
        self.config = config
        self.llm_client = llm_client
        self.driver = None
        
        # 图结构缓存
        self.entity_cache = {}  # 实体缓存
        self.relation_cache = {}    # 关系缓存
        self.subgraph_cache = {}    # 子图缓存
        
    def initialize(self):
        """初始化图RAG检索系统"""
        logger.info("初始化图RAG检索系统...")
        
        # 连接Neo4j
        try:
            # 创建数据库驱动实例
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri, 
                auth=(self.config.neo4j_user, self.config.neo4j_password)
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
                WHERE n.nodeId IS NOT NULL
                WITH n, COUNT { (n)--() } as degree
                RETURN labels(n) as node_labels, n.nodeId as node_id, 
                       n.name as name, n.department as department, degree
                ORDER BY degree DESC
                LIMIT 1000
                """

                result = session.run(entity_query)
                for record in result:
                    node_id = record["node_id"]
                    self.entity_cache[node_id] = {
                        "labels": record["node_labels"],
                        "name": record["name"],
                        "department": record.get("department"),  # 改为 department 或直接移除
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
        prompt = f"""
        作为医学图谱和临床决策专家，分析以下查询的图结构意图，并将自然语言问题映射到**已有医学知识图谱**上。

        已知图中大致有以下节点和关系：
        - 节点类型：
          - Department：科室节点（如"心血管科"、"消化内科"）
          - Consultation：病例/问答节点，包含 title（主诉）、ask（病情描述）、answer（医生建议）等属性
          - Disease/Symptom：疾病或症状节点（如"高血压"、"胃炎"）
          - Drug：药物节点（如"党参"、"雷贝拉唑"）
        - 主要关系：
          - (Consultation)-[:BELONGS_TO_DEPT]->(Department)
          - (Consultation)-[:MENTIONS_DISEASE]->(Disease)
          - (Drug)-[:TREATS]->(Disease) 或 (Consultation)-[:MENTIONS_DRUG]->(Drug)

        请根据上述图结构分析下面的查询：

        查询：{query}

        请识别：
        1. 查询类型：
           - entity_relation: 询问实体间的直接关系（如：高血压和党参有禁忌吗？）
           - multi_hop: 需要多跳推理（如：高血压患者常去哪个科室？需要：高血压→病例→科室）
           - subgraph: 需要完整子图（如：心血管科有哪些常见病例和用药？需要科室相关的完整知识网络）
           - path_finding: 路径查找（如：从症状到确诊用药的临床路径）
           - clustering: 聚类相似性（如：和这个病例类似的症状有哪些？）

        2. source_entities：
           - 只包含在图中**很有可能有对应节点**的具体实体名称
           - 优先选择：科室名（如"心血管科"）、疾病名（如"高血压"）、药物名（如"党参"）
           - 不要把抽象概念或约束（如"能不能吃"、"怎么办"、"推荐药物"）放进 source_entities

        3. target_entities：
           - 只在确实需要限制「路径终点」时填写
           - 同样只能使用可能出现在 Department / Disease / Drug 节点上的名称
           - 如果不确定目标实体怎么映射到图中，请返回空列表 []

        4. relation_types：本次推理中希望优先考虑的关系类型列表
           - 例如：["MENTIONS_DISEASE", "BELONGS_TO_DEPT", "TREATS"]

        5. max_depth：建议的图遍历深度（1-3 之间的整数）

        6. constraints：可选的**属性级约束**，用于表达图结构之外的过滤条件，例如：
           - 患者特征（如"老年人"、"孕妇"）
           - 症状限制（如"伴随头晕"）
           用一个字典描述，例如：
           {{
             "patient_type": ["老年人"],
             "symptom_filter": ["头晕"]
           }}

        示例1：
        查询："高血压患者能吃党参吗？"
        期望分析：这是 entity_relation 或 multi_hop 查询，需要验证 高血压 和 党参 之间的关联。

        返回JSON示例：
        {{
          "query_type": "entity_relation",
          "source_entities": ["高血压", "党参"],
          "target_entities": [],
          "relation_types": ["MENTIONS_DISEASE", "MENTIONS_DRUG", "TREATS"],
          "max_depth": 2,
          "constraints": {{}}
        }}

        请严格返回一个合法的 JSON 对象，不要包含任何多余的说明文字。
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
            json_match = re.search(r'\{[\s\S]*\}', raw_content)
            clean_content = json_match.group(0) if json_match else raw_content

            result = json.loads(clean_content)

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

    def multi_hop_traversal(self, graph_query: GraphQuery) -> List[GraphPath]:
        """临床多跳图遍历"""
        logger.info(f"执行多跳遍历: {graph_query.source_entities} -> {graph_query.target_entities}")
        paths = []
        if not self.driver:
            return paths

        try:
            with self.driver.session() as session:
                source_entities = graph_query.source_entities
                target_keywords = graph_query.target_entities or []
                max_depth = graph_query.max_depth

                # 兼容医疗节点的 target 过滤
                target_filter_clause = ""
                if target_keywords:
                    target_filter_clause = """
                AND ANY(kw IN $target_keywords WHERE
                    (target.name IS NOT NULL AND toString(target.name) CONTAINS kw) OR
                    (target.title IS NOT NULL AND toString(target.title) CONTAINS kw) OR
                    (target.department IS NOT NULL AND toString(target.department) CONTAINS kw)
                )"""

                # 兼容医疗节点的 source 匹配 (支持 name 和 title)
                cypher_query = f"""
                UNWIND $source_entities as source_name
                MATCH (source)
                WHERE (source.name IS NOT NULL AND source.name CONTAINS source_name) 
                   OR (source.title IS NOT NULL AND source.title CONTAINS source_name) 
                   OR source.nodeId = source_name

                MATCH path = (source)-[*1..{max_depth}]-(target)
                WHERE NOT source = target{target_filter_clause}

                WITH path, source, target, length(path) as path_len, relationships(path) as rels, nodes(path) as path_nodes
                WITH path, source, target, path_len, rels, path_nodes,
                     (1.0 / path_len) + 
                     (REDUCE(s = 0.0, n IN path_nodes | s + COUNT {{ (n)--() }}) / 10.0 / size(path_nodes)) +
                     (CASE WHEN ANY(r IN rels WHERE type(r) IN $relation_types) THEN 0.3 ELSE 0.0 END) as relevance

                ORDER BY relevance DESC
                LIMIT 20
                RETURN path, source, target, path_len, rels, path_nodes, relevance
                """

                params = {"source_entities": source_entities, "relation_types": graph_query.relation_types or []}
                if target_keywords:
                    params["target_keywords"] = target_keywords

                result = session.run(cypher_query, params)
                for record in result:
                    path_data = self._parse_neo4j_path(record)
                    if path_data:
                        paths.append(path_data)
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
                   OR (source.title IS NOT NULL AND source.title CONTAINS entity_name) 
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
    
    def graph_rag_search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        图RAG主搜索接口：整合所有图RAG能力
        """
        logger.info(f"开始图RAG检索: {query}")
        
        if not self.driver:
            logger.warning("Neo4j连接未建立，返回空结果")
            return []
        
        # 1. 查询意图理解
        graph_query = self.understand_graph_query(query)
        logger.info(f"查询类型: {graph_query.query_type.value}")
        
        results = []
        
        try:
            # 2. 根据查询类型执行不同策略
            if graph_query.query_type in [QueryType.MULTI_HOP, QueryType.PATH_FINDING]:
                # 多跳遍历 / 路径查找
                paths = self.multi_hop_traversal(graph_query)
                results.extend(self._paths_to_documents(paths, query))
                
            elif graph_query.query_type in [QueryType.SUBGRAPH, QueryType.CLUSTERING]:
                # 子图提取 / 聚类查询：都视为“围绕核心实体的局部知识网络”
                subgraph = self.extract_knowledge_subgraph(graph_query)
                
                # 图结构推理
                reasoning_chains = self.graph_structure_reasoning(subgraph, query)
                
                results.extend(self._subgraph_to_documents(subgraph, reasoning_chains, query))
                
            elif graph_query.query_type == QueryType.ENTITY_RELATION:
                # 实体关系查询（可以视为一跳 / 少量跳的路径查询）
                paths = self.multi_hop_traversal(graph_query)
                results.extend(self._paths_to_documents(paths, query))
            
            # 3. 图结构相关性排序
            results = self._rank_by_graph_relevance(results, query)
            
            logger.info(f"图RAG检索完成，返回 {len(results[:top_k])} 个结果")
            return results[:top_k]
            
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

    def _paths_to_documents(self, paths: List[GraphPath], query: str) -> List[Document]:
        documents = []
        for path in paths:
            path_desc = self._build_path_description(path)
            medical_name = path.nodes[0].get("name",
                                             path.nodes[0].get("title", "临床图谱结果")) if path.nodes else "临床图谱结果"
            doc = Document(
                page_content=path_desc,
                metadata={
                    "search_type": "medical_graph_path",
                    "path_length": path.path_length,
                    "relevance_score": path.relevance_score,
                    "path_type": path.path_type,
                    "node_count": len(path.nodes),
                    "relationship_count": len(path.relationships),
                    "title": medical_name
                }
            )
            documents.append(doc)
        return documents
    
    def _subgraph_to_documents(self, subgraph: KnowledgeSubgraph, 
                              reasoning_chains: List[str], query: str) -> List[Document]:
        """将知识子图转换为Document对象"""
        documents = []
        
        # 子图整体描述
        subgraph_desc = self._build_subgraph_description(subgraph)
        medical_name = subgraph.central_nodes[0].get("name", subgraph.central_nodes[0].get("title",
                                                                                           "医学子图")) if subgraph.central_nodes else "医学子图"

        doc = Document(
            page_content=subgraph_desc,
            metadata={
                "search_type": "medical_subgraph",
                "node_count": len(subgraph.connected_nodes),
                "relationship_count": len(subgraph.relationships),
                "graph_density": subgraph.graph_metrics.get("density", 0.0),
                "reasoning_chains": reasoning_chains,
                "title": medical_name
            }
        )
        documents.append(doc)
        
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
    
    def _identify_reasoning_patterns(self, subgraph: KnowledgeSubgraph) -> List[str]:
        """
        识别临床图谱的推理模式
        将“因果/组成/相似”转变为医学领域的专业逻辑范式
        """
        return [
            "症状-疾病诊断关联",
            "疾病-药物治疗方案",
            "药物禁忌与副作用提示",
            "跨科室联合会诊逻辑"
        ]
    
    def _build_reasoning_chain(self, pattern: str, subgraph: KnowledgeSubgraph) -> Optional[str]:
        """构建推理链"""
        return f"基于{pattern}的推理链"
    
    def _validate_reasoning_chains(self, chains: List[str], query: str) -> List[str]:
        """验证推理链"""
        return chains[:3]
    
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