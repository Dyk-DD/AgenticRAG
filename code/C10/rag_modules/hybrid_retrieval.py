"""
临床辅助系统 - 混合检索模块
基于双层检索范式：实体级（疾病/药物/科室） + 主题级（症状/病理/治疗方案）检索
结合图结构检索和向量检索，使用Round-robin轮询策略
"""

import json
import logging
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from neo4j import GraphDatabase
import torch
from transformers import AutoModelForSequenceClassification

# 引入重构后的医学图索引模块
from .graph_indexing import MedicalGraphIndexingModule

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """检索结果数据结构"""
    content: str
    node_id: str
    node_type: str
    relevance_score: float
    retrieval_level: str  # 'low' or 'high'
    metadata: Dict[str, Any]

class HybridRetrievalModule:
    """
    混合检索模块
    核心特点：
    1. 双层检索范式（实体级 + 主题级）
    2. 关键词提取和匹配
    3. 图结构+向量检索结合
    4. 一跳邻居扩展
    5. Round-robin轮询合并策略
    """
    
    def __init__(self, config, milvus_module, data_module, llm_client):
        self.config = config
        self.milvus_module = milvus_module
        self.data_module = data_module
        self.llm_client = llm_client
        self.driver = None
        self.bm25_retriever = None
        
        # 图索引模块
        self.graph_indexing = MedicalGraphIndexingModule(config, llm_client)
        self.graph_indexed = False
        
    def initialize(self, chunks: List[Document]):
        """初始化检索系统"""
        logger.info("初始化临床混合检索模块...")
        
        # 连接Neo4j
        self.driver = GraphDatabase.driver(
            self.config.neo4j_uri, 
            auth=(self.config.neo4j_user, self.config.neo4j_password)
        )
        
        # 初始化BM25检索器
        if chunks:
            self.bm25_retriever = BM25Retriever.from_documents(chunks)
            logger.info(f"BM25检索器初始化完成，文档数量: {len(chunks)}")
        
        # 初始化图索引
        self._build_graph_index()

    def _build_graph_index(self):
        """构建图索引"""
        if self.graph_indexed:
            return

        logger.info("开始构建临床图索引...")

        try:
            # 1. 从 data_module 获取医学问答对
            qa_pairs = getattr(self.data_module, 'qa_pairs', [])

            unique_departments = {}
            consultations = []

            for i, qa in enumerate(qa_pairs):
                dept_name = qa.get('department', '未知科室')
                if dept_name not in unique_departments:
                    unique_departments[dept_name] = {"node_id": f"dept_{dept_name}", "name": dept_name}

                consultations.append({
                    "node_id": f"qa_{i}",
                    "title": qa.get('title', ''),
                    "ask": qa.get('ask', ''),
                    "answer": qa.get('answer', ''),
                    "department": dept_name
                })

            departments_list = list(unique_departments.values())

            # 2. 新增：从 Neo4j 中动态拉取已构建的疾病实体
            diseases_list = []
            if self.driver:
                with self.driver.session() as session:
                    result = session.run("MATCH (d:Disease) RETURN d.name as name")
                    for record in result:
                        diseases_list.append({
                            "node_id": f"disease_{record['name']}",
                            "name": record["name"]
                        })

            # 3. 创建实体键值对 (传入科室、病例、疾病)
            self.graph_indexing.create_entity_key_values(
                departments=departments_list,
                consultations=consultations,
                diseases=diseases_list  # <--- 动态传入疾病列表
            )

            # 4. 创建关系键值对
            relationships = self._extract_relationships_from_graph()
            self.graph_indexing.create_relation_key_values(relationships)

            # 5. 去重优化
            self.graph_indexing.deduplicate_entities_and_relations()

            self.graph_indexed = True
            stats = self.graph_indexing.get_statistics()
            logger.info(f"临床图索引构建完成: {stats}")

        except Exception as e:
            logger.error(f"构建临床图索引失败: {e}")

    def _extract_relationships_from_graph(self) -> List[Tuple[str, str, str]]:
        """从Neo4j医学图中提取关系，修复ID匹配问题"""
        relationships = []
        try:
            with self.driver.session() as session:
                # 重构 Cypher: 根据节点类型强制拼装出与 Python 内存相同的自定义 ID
                query = """
                MATCH (source)-[r]->(target)
                RETURN 
                    CASE 
                        WHEN 'Consultation' IN labels(source) THEN source.nodeId 
                        WHEN 'Department' IN labels(source) THEN 'dept_' + source.name
                        WHEN 'Disease' IN labels(source) THEN 'disease_' + source.name
                        ELSE coalesce(source.nodeId, elementId(source)) 
                    END as source_id,
                    type(r) as relation_type,
                    CASE 
                        WHEN 'Consultation' IN labels(target) THEN target.nodeId 
                        WHEN 'Department' IN labels(target) THEN 'dept_' + target.name
                        WHEN 'Disease' IN labels(target) THEN 'disease_' + target.name
                        ELSE coalesce(target.nodeId, elementId(target)) 
                    END as target_id
                """
                result = session.run(query)
                for record in result:
                    relationships.append((
                        str(record["source_id"]),
                        record["relation_type"],
                        str(record["target_id"])
                    ))
        except Exception as e:
            logger.error(f"提取医学图关系失败: {e}")
        return relationships

    # 意图识别模块
    def extract_query_keywords(self, query: str) -> Tuple[List[str], List[str]]:
        """
        提取临床查询关键词：实体级 + 主题级
        """
        prompt = f"""
        作为一名专业的临床医学助手，请分析以下患者或医生的查询，并提取关键词，分为两个层次：

        查询：{query}

        提取规则：
        1. 实体级关键词：具体的疾病名称、症状名词、药物名称、科室名称、检查项目等具体医学实体。
           - 例如：高血压、党参、心血管科、心电图、雷贝拉唑、胃炎
           - 尽量提取具有明确指向性的名词。

        2. 主题级关键词：抽象的病理概念、治疗主题、注意事项、饮食禁忌、预后等。
           - 例如：降血压、副作用、饮食禁忌、备孕期可用、物理降温
           - 排除一般性动作词，如：怎么办、能吃吗、推荐、介绍。

        示例：
        查询："高血压患者能吃党参吗？" 
        {{
            "entity_keywords": ["高血压", "党参"],
            "topic_keywords": ["饮食禁忌", "降血压", "药物相互作用"]
        }}

        查询："查出来胆汁反流性胃炎怎么办，会影响生理期吗？"
        {{
            "entity_keywords": ["胆汁反流性胃炎", "生理期", "消化科"],
            "topic_keywords": ["治疗方案", "月经推迟", "内分泌影响"]
        }}

        请严格按照JSON格式返回，不要包含多余的文字：
        {{
            "entity_keywords": ["实体1", "实体2", ...],
            "topic_keywords": ["主题1", "主题2", ...]
        }}
        """

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"}
            )

            import re
            raw_content = response.choices[0].message.content.strip()
            json_match = re.search(r'\{[\s\S]*\}', raw_content)
            clean_content = json_match.group(0) if json_match else raw_content

            result = json.loads(clean_content)
            entity_keywords = result.get("entity_keywords", [])
            topic_keywords = result.get("topic_keywords", [])

            logger.info(f"临床关键词提取完成 - 实体级: {entity_keywords}, 主题级: {topic_keywords}")
            return entity_keywords, topic_keywords

        except Exception as e:
            logger.error(f"医学关键词提取失败: {e}")
            keywords = query.split()
            return keywords[:3], keywords[3:6] if len(keywords) > 3 else keywords

    # 图结构双层检索模块
    def entity_level_retrieval(self, entity_keywords: List[str], top_k: int = 5) -> List[RetrievalResult]:
        """实体级检索：专注于具体疾病、药物和科室实体"""
        results = []

        # 1. 使用图索引进行实体检索
        for keyword in entity_keywords:
            # 检索匹配的实体
            entities = self.graph_indexing.get_entities_by_key(keyword)

            for entity in entities:
                # 获取邻居信息
                neighbors = self._get_node_neighbors(entity.metadata["node_id"], max_neighbors=2)

                # 构建增强内容
                enhanced_content = entity.value_content
                if neighbors:
                    enhanced_content += f"\n相关临床信息: {', '.join(neighbors)}"

                results.append(RetrievalResult(
                    content=enhanced_content,
                    node_id=entity.metadata["node_id"],
                    node_type=entity.entity_type,
                    relevance_score=0.9,  # 精确匹配得分较高
                    retrieval_level="entity",
                    metadata={
                        "entity_name": entity.entity_name,
                        "entity_type": entity.entity_type,
                        "index_keys": entity.index_keys,
                        "matched_keyword": keyword
                    }
                ))
        
        # 2. 如果图索引结果不足，使用Neo4j进行补充检索
        if len(results) < top_k:
            neo4j_results = self._neo4j_entity_level_search(entity_keywords, top_k - len(results))
            results.extend(neo4j_results)
            
        # 3. 按相关性排序并返回
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"实体级检索完成，返回 {len(results)} 个结果")
        return results[:top_k]

    def _neo4j_entity_level_search(self, keywords: List[str], limit: int) -> List[RetrievalResult]:
        """
        Neo4j医学实体补充检索
        不仅匹配文本内容，还匹配科室等核心医疗实体属性
        """
        results = []

        try:
            with self.driver.session() as session:
                # 增强的 Cypher 查询：
                # 1. 增加了对 department (科室实体) 的直接匹配
                # 2. 使用 coalesce 兼容内部 ID，防止 nodeId 缺失导致报错
                cypher_query = """
                UNWIND $keywords as keyword
                MATCH (node:Consultation)
                WHERE node.department CONTAINS keyword
                   OR node.title CONTAINS keyword 
                   OR node.ask CONTAINS keyword 
                   OR node.answer CONTAINS keyword
                RETURN 
                    coalesce(node.nodeId, elementId(node)) as node_id,
                    node.title as title,
                    node.department as department,
                    node.ask as ask,
                    node.answer as answer,
                    keyword as matched_keyword
                LIMIT $limit
                """

                result = session.run(cypher_query, {"keywords": keywords, "limit": limit})

                for record in result:
                    # 重新组织更严谨的临床上下文结构
                    # 在 _neo4j_entity_level_search 中：
                    content_parts = []
                    if record["department"]:
                        content_parts.append(f"【历史就诊科室】: {record['department']}")
                    if record["title"]:
                        content_parts.append(f"【历史病例主诉】: {record['title']}")
                    if record["ask"]:
                        content_parts.append(f"【历史患者提问/描述】: {record['ask']}")
                    if record["answer"]:
                        content_parts.append(f"【当时医生诊断】: {record['answer']}")

                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        node_id=str(record["node_id"]),
                        node_type="Consultation",
                        relevance_score=0.75,  # 提高基础权重，因为实体匹配的精准度通常较高
                        retrieval_level="entity",
                        metadata={
                            "title": record["title"],
                            "department": record["department"],
                            "matched_keyword": record["matched_keyword"],
                            "source": "neo4j_entity_fallback"
                        }
                    ))
        except Exception as e:
            logger.error(f"Neo4j医学实体补充检索失败: {e}")

        return results

    def topic_level_retrieval(self, topic_keywords: List[str], top_k: int = 5) -> List[RetrievalResult]:
        """主题级检索：专注于临床症状、副作用、病理机制等"""
        results = []

        # 1. 使用图索引进行关系/主题检索
        for keyword in topic_keywords:
            # 检索匹配的关系
            relations = self.graph_indexing.get_relations_by_key(keyword)

            for relation in relations:
                # 获取相关实体信息
                source_entity = self.graph_indexing.entity_kv_store.get(relation.source_entity)
                target_entity = self.graph_indexing.entity_kv_store.get(relation.target_entity)

                if source_entity and target_entity:
                    content_parts = [
                        f"临床主题: {keyword}",
                        relation.value_content,
                        f"相关主体: {source_entity.entity_name}",
                        f"关联信息: {target_entity.entity_name}"
                    ]

                    # 添加源实体的详细信息（适配病例记录 Consultation）
                    if source_entity.entity_type == "Consultation":
                        newline = '\n'
                        content_parts.append(f"病例详情: {source_entity.value_content.split(newline)[0]}")

                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        node_id=relation.source_entity,  # 以主要实体为ID
                        node_type=source_entity.entity_type,
                        relevance_score=0.95,  # 主题匹配得分
                        retrieval_level="topic",
                        metadata={
                            "relation_id": relation.relation_id,
                            "relation_type": relation.relation_type,
                            "source_name": source_entity.entity_name,
                            "target_name": target_entity.entity_name,
                            "matched_keyword": keyword,
                            "index_keys": relation.index_keys
                        }
                    ))

        # 2. 使用实体的分类信息进行主题检索（适配病例和科室）
        for keyword in topic_keywords:
            entities = self.graph_indexing.get_entities_by_key(keyword)
            for entity in entities:
                # 匹配病例和科室实体
                if entity.entity_type in ["Consultation", "Department"]:
                    content_parts = [
                        f"临床分类: {keyword}",
                        entity.value_content
                    ]

                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        node_id=entity.metadata["node_id"],
                        node_type=entity.entity_type,
                        relevance_score=0.85,  # 分类匹配得分
                        retrieval_level="topic",
                        metadata={
                            "entity_name": entity.entity_name,
                            "entity_type": entity.entity_type,
                            "matched_keyword": keyword,
                            "source": "category_match"
                        }
                    ))

        # 3. 如果结果不足，使用Neo4j进行补充检索
        if len(results) < top_k:
            neo4j_results = self._neo4j_topic_level_search(topic_keywords, top_k - len(results))
            results.extend(neo4j_results)

        # 4. 按相关性排序并返回
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        logger.info(f"临床主题级检索完成，返回 {len(results)} 个结果")
        return results[:top_k]

    def _neo4j_topic_level_search(self, keywords: List[str], limit: int) -> List[RetrievalResult]:
        """Neo4j临床主题级检索补充"""
        results = []

        try:
            with self.driver.session() as session:
                # 完全重构Cypher，查询医疗问答节点(Consultation)
                cypher_query = """
                UNWIND $keywords as keyword
                MATCH (c:Consultation)
                WHERE c.department CONTAINS keyword 
                   OR c.title CONTAINS keyword
                   OR c.ask CONTAINS keyword
                   OR c.answer CONTAINS keyword
                RETURN 
                    c.nodeId as node_id,
                    c.title as title,
                    c.department as department,
                    c.ask as ask,
                    c.answer as answer,
                    keyword as matched_keyword
                LIMIT $limit
                """

                result = session.run(cypher_query, {
                    "keywords": keywords,
                    "limit": limit
                })

                for record in result:
                    # 在 _neo4j_topic_level_search 中：
                    content_parts = [
                        f"【历史病例主诉】: {record['title']}",
                        f"【历史所属科室】: {record['department']}",
                        f"【历史患者提问】: {record['ask']}",
                        f"【当时医生诊断】: {record['answer']}"
                    ]

                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        node_id=record["node_id"],
                        node_type="Consultation",
                        relevance_score=0.75,  # 补充检索得分
                        retrieval_level="topic",
                        metadata={
                            "title": record["title"],
                            "department": record["department"],
                            "matched_keyword": record["matched_keyword"],
                            "source": "neo4j_fallback"
                        }
                    ))

        except Exception as e:
            logger.error(f"Neo4j医学主题检索失败: {e}")

        return results

    def dual_level_retrieval(self, query: str, top_k: int = 5) -> List[Document]:
        """双层检索：结合实体级和临床主题级检索"""
        logger.info(f"开始医学双层检索: {query}")

        entity_keywords, topic_keywords = self.extract_query_keywords(query)
        entity_results = self.entity_level_retrieval(entity_keywords, top_k)
        topic_results = self.topic_level_retrieval(topic_keywords, top_k)

        all_results = entity_results + topic_results

        seen_nodes = set()
        unique_results = []

        for result in sorted(all_results, key=lambda x: x.relevance_score, reverse=True):
            if result.node_id not in seen_nodes:
                seen_nodes.add(result.node_id)
                unique_results.append(result)

        documents = []
        for result in unique_results[:top_k]:
            # 兼容医学属性 title 替代 recipe_name
            title_name = result.metadata.get("title") or result.metadata.get("entity_name", "临床参考资料")

            doc = Document(
                page_content=result.content,
                metadata={
                    "node_id": result.node_id,
                    "node_type": result.node_type,
                    "retrieval_level": result.retrieval_level,
                    "relevance_score": result.relevance_score,
                    "title": title_name,  # 确保向外输出时使用 title 字段
                    "search_type": "dual_level",
                    **result.metadata
                }
            )
            documents.append(doc)

        logger.info(f"医学双层检索完成，返回 {len(documents)} 个文档")
        return documents

    def vector_search_enhanced(self, query: str, top_k: int = 5) -> List[Document]:
        """增强的向量检索：结合临床图信息"""
        try:
            vector_docs = self.milvus_module.similarity_search(query, k=top_k * 2)

            enhanced_docs = []
            for result in vector_docs:
                content = result.get("text", "")
                metadata = result.get("metadata", {})
                node_id = metadata.get("node_id")

                if node_id:
                    neighbors = self._get_node_neighbors(node_id)
                    if neighbors:
                        # 补充临床关联词
                        neighbor_info = f"\n临床关联信息: {', '.join(neighbors[:3])}"
                        content += neighbor_info

                # 兼容医学属性 title
                title_name = metadata.get("title", "临床参考资料")
                vector_score = result.get("score", 0.0)
                logger.debug(f"医学向量检索得分: {title_name} = {vector_score}")

                doc = Document(
                    page_content=content,
                    metadata={
                        **metadata,
                        "title": title_name,
                        "score": vector_score,
                        "search_type": "vector_enhanced"
                    }
                )
                enhanced_docs.append(doc)

            return enhanced_docs[:top_k]

        except Exception as e:
            logger.error(f"临床增强向量检索失败: {e}")
            return []

    def _get_node_neighbors(self, node_id: str, max_neighbors: int = 3) -> List[str]:
        """获取节点的临床邻居信息"""
        try:
            with self.driver.session() as session:
                # 兼容科室/疾病名称 或 问答标题属性
                query = """
                MATCH (n {nodeId: $node_id})-[r]-(neighbor)
                RETURN coalesce(neighbor.name, neighbor.title, neighbor.department, '关联实体') as name
                LIMIT $limit
                """
                result = session.run(query, {"node_id": node_id, "limit": max_neighbors})
                return [record["name"] for record in result if record["name"]]
        except Exception as e:
            logger.error(f"获取医学邻居节点失败: {e}")
            return []

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Document]:
        """混合检索：使用Round-robin轮询合并策略"""
        logger.info(f"开始医学混合检索: {query}")

        dual_docs = self.dual_level_retrieval(query, top_k)
        vector_docs = self.vector_search_enhanced(query, top_k)

        merged_docs = []
        seen_doc_ids = set()
        max_len = max(len(dual_docs), len(vector_docs))
        origin_len = len(dual_docs) + len(vector_docs)

        for i in range(max_len):
            if i < len(dual_docs):
                doc = dual_docs[i]
                doc_id = doc.metadata.get("node_id", hash(doc.page_content))
                if doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    doc.metadata["search_method"] = "dual_level"
                    doc.metadata["round_robin_order"] = len(merged_docs)
                    doc.metadata["final_score"] = doc.metadata.get("relevance_score", 0.0)
                    merged_docs.append(doc)

            if i < len(vector_docs):
                doc = vector_docs[i]
                doc_id = doc.metadata.get("node_id", hash(doc.page_content))
                if doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    doc.metadata["search_method"] = "vector_enhanced"
                    doc.metadata["round_robin_order"] = len(merged_docs)

                    vector_score = doc.metadata.get("score", 0.0)
                    similarity_score = max(0.0, 1.0 - vector_score) if vector_score <= 1.0 else 0.0
                    doc.metadata["final_score"] = similarity_score
                    merged_docs.append(doc)

        final_docs = merged_docs[:top_k]

        logger.info(f"临床Round-robin合并：从总共{origin_len}个结果合并为{len(final_docs)}个参考文档")
        return final_docs
        
    def close(self):
        """关闭资源连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭") 