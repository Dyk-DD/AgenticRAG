"""
临床决策辅助系统 - 智能查询路由器
根据医疗查询特点自动选择最适合的检索策略：
- 传统混合检索：适合简单的医疗信息查找（如：科室指南、基础病症查询）
- 图RAG检索：适合复杂的临床关系推理和知识发现（如：用药禁忌、并发症推演、鉴别诊断）
"""

import json
import logging
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re

from langchain_core.documents import Document

from .utils import extract_json_from_llm

logger = logging.getLogger(__name__)

class SearchStrategy(Enum):
    """搜索策略枚举"""
    HYBRID_TRADITIONAL = "hybrid_traditional"  # 传统混合检索
    GRAPH_RAG = "graph_rag"  # 图RAG检索
    COMBINED = "combined"  # 组合策略
    
@dataclass
class QueryAnalysis:
    """查询分析结果"""
    query_complexity: float  # 查询复杂度 (0-1)
    relationship_intensity: float  # 关系密集度 (0-1)
    reasoning_required: bool  # 是否需要临床推理
    entity_count: int  # 医学实体数量
    recommended_strategy: SearchStrategy
    confidence: float  # 推荐置信度
    reasoning: str  # 推荐理由

class IntelligentQueryRouter:
    """
    智能查询路由器
    核心能力：
    1. 临床复杂度分析：识别简单查找 vs 复杂鉴别诊断
    2. 医学关系密集度评估：判断是否需要图结构优势（如药物相互作用）
    3. 策略自动选择：路由到最适合的检索引擎
    """

    # 与关联性深度推理相关的关键词
    COMPLEXITY_KEYWORDS = ["并发症", "鉴别诊断", "禁忌", "副作用", "机理", "风险", "为什么", "方案", "预后", "严重"]
    RELATION_KEYWORDS = ["配伍", "同服", "相互作用", "导致", "引发", "引起", "合并", "伴随"]

    def __init__(self,
                 traditional_retrieval,  # 传统混合检索模块
                 graph_rag_retrieval,  # 图RAG检索模块
                 llm_client,
                 config):
        self.traditional_retrieval = traditional_retrieval
        self.graph_rag_retrieval = graph_rag_retrieval
        self.llm_client = llm_client
        self.config = config

        # 路由统计
        self.route_stats = {
            "traditional_count": 0,
            "graph_rag_count": 0,
            "combined_count": 0,
            "total_queries": 0
        }

        # 疾病/药物实体词典缓存（用于规则降级）
        self._disease_entity_cache: List[str] = []
        self._drug_entity_cache: List[str] = []
        self._entity_patterns = None

    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        深度分析临床查询特征，决定最佳检索策略
        """
        logger.info(f"分析临床查询特征: {query}")

        # 使用LLM进行医学智能分析
        analysis_prompt = f"""
        作为临床决策辅助系统（RAG）的查询分析专家，请深度分析以下医疗查询的特征：

        患者/医生查询：{query}

        请从以下维度分析：

        1. 临床查询复杂度 (0-1)：
           - 0.0-0.3: 简单医疗信息查找（如：感冒挂什么科？正常血压范围是多少？）
           - 0.4-0.7: 中等复杂度（如：心血管科有哪些常见病例和治疗方案？胃炎的注意事项？）
           - 0.8-1.0: 高复杂度临床推理（如：高血压合并糖尿病患者，出现头晕症状应如何进行鉴别诊断与用药？）

        2. 医学关系密集度 (0-1)：
           - 0.0-0.3: 单一医学实体信息（如：雷贝拉唑的副作用是什么？）
           - 0.4-0.7: 实体间简单关联（如：高血压患者能吃党参吗？某种药对应的适应症？）
           - 0.8-1.0: 复杂临床关系网络（如：休克患者的预后评估路径、并发症网络及跨科室治疗逻辑）

        3. 临床推理需求：
           - 是否需要多跳推理（如：症状 -> 疾病 -> 禁忌药物）？
           - 是否需要因果分析（如：不良反应归因）？
           - 是否需要鉴别诊断对比分析？

        4. 医学实体识别：
           - 查询中包含多少个明确的医学实体（疾病、症状、药物、科室等）？

        基于分析推荐检索策略：
        - hybrid_traditional: 适合简单直接的医疗信息查找
        - graph_rag: 适合复杂的临床关系推理、药物禁忌检查和路径发现
        - combined: 疑难杂症，需要两种策略结合

        返回JSON格式（必须是合法的JSON对象）：
        {{
            "query_complexity": 0.6,
            "relationship_intensity": 0.8,
            "reasoning_required": true,
            "entity_count": 2,
            "recommended_strategy": "graph_rag",
            "confidence": 0.85,
            "reasoning": "该查询涉及高血压与党参（疾病与药物）之间的关系验证，需要利用知识图谱进行严格的禁忌推理"
        }}
        """

        try:
            # 1. 在 API 请求中增加 response_format 强制约束 (如果模型 API 支持)
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=800,
                response_format={"type": "json_object"}  # 强烈建议加上这行，强制大模型输出 JSON
            )

            # 2. 获取原始返回文本
            raw_content = response.choices[0].message.content.strip()

            # 3. 使用稳健的 JSON 提取
            result = extract_json_from_llm(raw_content)

            if result is None:
                logger.warning("LLM 返回中未能提取有效 JSON，降级为规则匹配")
                return self._rule_based_analysis(query)

            analysis = QueryAnalysis(
                query_complexity=float(result.get("query_complexity", 0.5)),
                relationship_intensity=float(result.get("relationship_intensity", 0.5)),
                reasoning_required=bool(result.get("reasoning_required", False)),
                entity_count=int(result.get("entity_count", 1)),
                recommended_strategy=SearchStrategy(result.get("recommended_strategy", "hybrid_traditional")),
                confidence=float(result.get("confidence", 0.5)),
                reasoning=str(result.get("reasoning", "默认分析"))
            )

            logger.info(f"临床查询分析完成: {analysis.recommended_strategy.value} (置信度: {analysis.confidence:.2f})")
            return analysis

        except Exception as e:
            logger.error(f"临床查询分析失败，降级为规则匹配: {e}")
            return self._rule_based_analysis(query)

    def load_entity_cache(self):
        """从 Neo4j 加载疾病和药物实体，增强规则降级覆盖率"""
        try:
            driver = getattr(self.graph_rag_retrieval, 'driver', None)
            if driver:
                with driver.session() as s:
                    diseases = s.run("MATCH (d:Disease) RETURN d.name AS name").data()
                    self._disease_entity_cache = [r['name'] for r in diseases if r.get('name')]
                    try:
                        drugs = s.run("MATCH (d:Drug) RETURN d.name AS name").data()
                        self._drug_entity_cache = [r['name'] for r in drugs if r.get('name')]
                    except Exception:
                        self._drug_entity_cache = []
        except Exception:
            # 如果图还没有疾病/药物，使用空列表
            self._disease_entity_cache = []
            self._drug_entity_cache = []

        # 编译实体模式
        all_entities = self._disease_entity_cache + self._drug_entity_cache
        if all_entities:
            escaped = [re.escape(e) for e in sorted(all_entities, key=len, reverse=True)]
            self._entity_patterns = re.compile('|'.join(escaped))
        else:
            self._entity_patterns = None

        logger.info(f"路由器实体缓存已加载: {len(self._disease_entity_cache)} 疾病, {len(self._drug_entity_cache)} 药物")

    def _rule_based_analysis(self, query: str) -> QueryAnalysis:
        """基于医学特征规则的降级分析（增强版：关键词 + 实体匹配）"""
        # 如果实体缓存未加载，尝试加载
        if not self._disease_entity_cache and not self._drug_entity_cache:
            self.load_entity_cache()

        # 1. 关键词匹配（原有逻辑）
        complexity = sum(1 for kw in self.COMPLEXITY_KEYWORDS if kw in query) / max(1, len(self.COMPLEXITY_KEYWORDS))
        relation_intensity = sum(1 for kw in self.RELATION_KEYWORDS if kw in query) / max(1, len(self.RELATION_KEYWORDS))

        # 2. 实体匹配 —— 检查是否命中疾病/药物实体
        entity_hits = 0
        if self._entity_patterns:
            matches = self._entity_patterns.findall(query)
            entity_hits = len(matches)

        # 3. 多实体出现在同一查询中往往需要图推理（如药物-疾病相互作用）
        multi_entity_graph_needed = entity_hits >= 2

        # 判断是否需要图检索
        needs_graph = (complexity > 0.0 or relation_intensity > 0.0 or multi_entity_graph_needed)

        if needs_graph:
            strategy = SearchStrategy.GRAPH_RAG
        else:
            strategy = SearchStrategy.HYBRID_TRADITIONAL

        # 计算综合复杂度
        combined_complexity = min(
            complexity * 3 + (entity_hits * 0.15),
            1.0
        )

        return QueryAnalysis(
            query_complexity=combined_complexity,
            relationship_intensity=min(relation_intensity * 3 + (0.2 if multi_entity_graph_needed else 0.0), 1.0),
            reasoning_required=(complexity > 0.0 or multi_entity_graph_needed),
            entity_count=max(entity_hits, len(query.split()) // 2),
            recommended_strategy=strategy,
            confidence=0.65 if multi_entity_graph_needed else 0.6,
            reasoning="基于临床术语特征和实体匹配触发的规则分析"
        )

    def route_query(self, query: str, top_k: int = 5) -> Tuple[List[Document], QueryAnalysis]:
        """
        智能路由查询到最适合的检索引擎
        """
        logger.info(f"开始临床智能路由: {query}")

        # 1. 分析查询特征
        analysis = self.analyze_query(query)

        # 2. 更新统计
        self._update_route_stats(analysis.recommended_strategy)

        # 3. 根据策略执行检索
        documents = []

        try:
            if analysis.recommended_strategy == SearchStrategy.HYBRID_TRADITIONAL:
                logger.info("🩺 使用传统混合检索 (适合基础医疗信息)")
                documents = self.traditional_retrieval.hybrid_search(query, top_k)

            elif analysis.recommended_strategy == SearchStrategy.GRAPH_RAG:
                logger.info("🧬 使用图RAG检索 (适合临床逻辑推理与路径发现)")
                documents = self.graph_rag_retrieval.graph_rag_search(query, top_k)

            elif analysis.recommended_strategy == SearchStrategy.COMBINED:
                logger.info("🏥 使用临床组合检索策略 (应对复杂疑难杂症)")
                documents = self._combined_search(query, top_k)

            # 4. 结果后处理
            documents = self._post_process_results(documents, analysis)

            # Cache for traceability
            self.last_retrieved_docs = documents

            logger.info(f"路由完成，返回 {len(documents)} 份医学参考资料")
            return documents, analysis

        except Exception as e:
            logger.error(f"临床查询路由失败: {e}")
            # 降级到传统检索
            documents = self.traditional_retrieval.hybrid_search(query, top_k)
            self.last_retrieved_docs = documents
            return documents, analysis

    def _combined_search(self, query: str, top_k: int) -> List[Document]:
        """
        组合搜索策略：RRF 融合传统检索和图 RAG 结果，替代 round-robin
        """
        # 多取一些保证 RRF 排名充分
        traditional_docs = self.traditional_retrieval.hybrid_search(query, top_k * 2)
        graph_docs = self.graph_rag_retrieval.graph_rag_search(query, top_k * 2)

        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        K = 60  # RRF 常数

        def index_docs(docs, source):
            for rank, doc in enumerate(docs):
                doc_id = doc.metadata.get("node_id", str(hash(doc.page_content[:200])))
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (K + rank)
                if doc_id not in doc_map:
                    doc.metadata["search_source"] = source
                    doc_map[doc_id] = doc

        index_docs(traditional_docs, "medical_traditional")
        index_docs(graph_docs, "medical_graph_rag")

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        result = [doc_map[doc_id] for doc_id, _ in ranked[:top_k]]
        return result
    
    def _post_process_results(self, documents: List[Document], analysis: QueryAnalysis) -> List[Document]:
        """
        结果后处理：根据查询分析优化结果
        """
        for doc in documents:
            # 添加路由信息到元数据
            doc.metadata.update({
                "route_strategy": analysis.recommended_strategy.value,
                "clinical_complexity": analysis.query_complexity,
                "route_confidence": analysis.confidence
            })
        
        return documents
    
    def _update_route_stats(self, strategy: SearchStrategy):
        self.route_stats["total_queries"] += 1
        if strategy == SearchStrategy.HYBRID_TRADITIONAL:
            self.route_stats["traditional_count"] += 1
        elif strategy == SearchStrategy.GRAPH_RAG:
            self.route_stats["graph_rag_count"] += 1
        elif strategy == SearchStrategy.COMBINED:
            self.route_stats["combined_count"] += 1
    
    def get_route_statistics(self) -> Dict[str, Any]:
        """获取路由统计信息"""
        total = self.route_stats["total_queries"]
        if total == 0:
            return self.route_stats
        
        return {
            **self.route_stats,
            "traditional_ratio": self.route_stats["traditional_count"] / total,
            "graph_rag_ratio": self.route_stats["graph_rag_count"] / total,
            "combined_ratio": self.route_stats["combined_count"] / total
        }

    def explain_routing_decision(self, query: str) -> str:
        """解释路由决策过程"""
        analysis = self.analyze_query(query)

        explanation = f"""
        ==== 临床查询路由分析报告 ====

        【问诊查询】：{query}

        【医学特征分析】：
        - 临床复杂度评估：{analysis.query_complexity:.2f} ({'基础问询' if analysis.query_complexity < 0.4 else '中度疑难' if analysis.query_complexity < 0.8 else '高度复杂/鉴别诊断'})
        - 图谱关系密集度：{analysis.relationship_intensity:.2f} ({'单一实体' if analysis.relationship_intensity < 0.4 else '实体强关联' if analysis.relationship_intensity < 0.8 else '复杂临床网络'})
        - 逻辑推理需求：{'是 (需进行多跳医学推断)' if analysis.reasoning_required else '否 (直接查询即可)'}
        - 提取医疗实体：{analysis.entity_count} 个

        【系统调度结论】：
        - 推荐检索策略：{analysis.recommended_strategy.value}
        - 决策置信度：{analysis.confidence:.2f}
        - 调度理由：{analysis.reasoning}
        ==============================
        """
        return explanation