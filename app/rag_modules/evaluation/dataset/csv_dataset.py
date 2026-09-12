"""
CSV 测试数据集加载器
从 CSV 文件加载测试查询并构建 Ground Truth
"""

import hashlib
import logging
import os
import random
from typing import List, Optional

import pandas as pd

from ..models import TestQuery

logger = logging.getLogger(__name__)


class CSVTestDataset:
    """从 CSV 文件加载测试查询集"""

    def __init__(self, csv_path: str, llm_client=None):
        self.csv_path = csv_path
        self.llm_client = llm_client

    # ------------------------------------------------------------------
    #  Core
    # ------------------------------------------------------------------

    def load(
        self,
        sample_size: Optional[int] = None,
        seed: int = 42,
    ) -> List[TestQuery]:
        """
        加载 CSV 并构建 TestQuery 列表

        CSV 期望列: department, title, ask, answer
        每行的 ask 作为 query, chunk_id 作为 ground truth
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"测试CSV不存在: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        # 标准化列名
        df.columns = [c.strip().lower() for c in df.columns]
        logger.info(f"加载测试CSV: {self.csv_path} ({len(df)} 条)")

        # 检测是否存在 original_ask 列（LLM 改写后的测试集）
        has_original_ask = "original_ask" in df.columns
        if has_original_ask:
            logger.info("检测到 original_ask 列，将使用 original_ask 计算 Ground Truth，"
                        "使用 ask 作为查询文本")

        # 检测是否存在 expected_routing_strategy 列（标注过路由策略）
        has_expected_routing = "expected_routing_strategy" in df.columns
        if has_expected_routing:
            logger.info("检测到 expected_routing_strategy 列，将使用已有的路由标签")

        queries = []
        for idx, row in df.iterrows():
            dept = str(row.get("department", "未知科室")).strip()
            title = str(row.get("title", "")).strip()
            ask = str(row.get("ask", "")).strip()
            answer = str(row.get("answer", "")).strip()

            if not ask:
                continue

            # 读取 LLM 标注的路由策略（如果 CSV 中已有）
            expected_routing = None
            if has_expected_routing:
                val = row.get("expected_routing_strategy", "")
                if val and str(val).strip():
                    expected_routing = str(val).strip()

            # 改写后的测试集：用 original_ask 计算 node_id（匹配数据库中的记录），
            # 用改写后的 ask 作为实际查询文本
            if has_original_ask:
                original_ask = str(row.get("original_ask", "")).strip()
                if original_ask:
                    node_id = self._compute_node_id(dept, title, original_ask, answer)
                else:
                    node_id = self._compute_node_id(dept, title, ask, answer)
            else:
                node_id = self._compute_node_id(dept, title, ask, answer)

            queries.append(
                TestQuery(
                    query_id=f"q_{idx}",
                    query_text=ask,
                    ground_truth_node_ids=[node_id],
                    ground_truth_chunk_ids=[f"{node_id}_full"],
                    expected_routing_strategy=expected_routing,
                    department=dept,
                    metadata={
                        "title": title,
                        "answer": answer,
                        "csv_row": idx,
                    },
                )
            )

        # 采样
        if sample_size and sample_size < len(queries):
            random.seed(seed)
            queries = random.sample(queries, sample_size)
            logger.info(f"随机采样 {sample_size} 条查询 (seed={seed})")

        logger.info(f"测试集构建完成: {len(queries)} 条有效查询")
        return queries

    # ------------------------------------------------------------------
    #  Paraphrase
    # ------------------------------------------------------------------

    def paraphrase_queries(
        self,
        queries: List[TestQuery],
        model: str = "deepseek-v4-flash",
    ) -> List[TestQuery]:
        """
        用 LLM 改写查询文本，防止 query 原文在索引中导致数据泄露
        （防止 query 原文在索引中导致数据泄露）
        """
        if not self.llm_client:
            logger.warning("未提供 llm_client，跳过 paraphrase")
            return queries

        logger.info(f"开始 LLM 改写 {len(queries)} 条查询...")
        cache: dict = {}
        paraphrased = []

        for item in queries:
            if item.query_text in cache:
                new_text = cache[item.query_text]
            else:
                prompt = (
                    f"请将以下患者的提问改写成另一种说法，保持医学含义完全不变，"
                    f"但使用不同的措辞和句式。只需输出改写后的文本，不要任何解释：\n\n"
                    f"{item.query_text}"
                )
                try:
                    resp = self.llm_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=256,
                    )
                    new_text = resp.choices[0].message.content.strip()
                    cache[item.query_text] = new_text
                except Exception as e:
                    logger.warning(f"改写失败 ({item.query_id}): {e}")
                    new_text = item.query_text

            paraphrased.append(
                TestQuery(
                    query_id=item.query_id,
                    query_text=new_text,
                    original_query=item.query_text,
                    ground_truth_node_ids=item.ground_truth_node_ids,
                    ground_truth_chunk_ids=item.ground_truth_chunk_ids,
                    department=item.department,
                    metadata=item.metadata,
                )
            )

        logger.info(f"改写完成，共 {len(paraphrased)} 条")
        return paraphrased

    # ------------------------------------------------------------------
    #  Routing Ground Truth
    # ------------------------------------------------------------------

    @staticmethod
    def infer_expected_routing(
        queries: List[TestQuery],
        complexity_keywords: Optional[List[str]] = None,
        relation_keywords: Optional[List[str]] = None,
    ) -> List[TestQuery]:
        """
        通过规则推断预期的路由策略（用于没有手工标注的测试集）
        - 包含复杂度关键词 OR 关系关键词 → graph_rag
        - 否则 → hybrid_traditional
        """
        ck = complexity_keywords or [
            "并发症", "鉴别诊断", "禁忌", "副作用", "机理",
            "风险", "为什么", "方案", "预后", "严重",
        ]
        rk = relation_keywords or [
            "配伍", "同服", "相互作用", "导致", "引发",
            "引起", "合并", "伴随",
        ]

        for q in queries:
            # 如果 CSV 中已有 LLM 标注（或手工标注），保留不覆盖
            if q.expected_routing_strategy:
                continue
            text = q.query_text
            has_complex = any(kw in text for kw in ck)
            has_relation = any(kw in text for kw in rk)
            if has_complex or has_relation:
                q.expected_routing_strategy = "graph_rag"
            else:
                q.expected_routing_strategy = "hybrid_traditional"

        return queries

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_node_id(department: str, title: str, ask: str, answer: str) -> str:
        """与 build_chunks 保持一致的 node_id 计算方法"""
        unique_str = f"{department}_{title}_{ask}_{answer}"
        return "qa_" + hashlib.md5(unique_str.encode("utf-8")).hexdigest()[:12]
