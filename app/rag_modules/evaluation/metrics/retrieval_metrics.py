"""
检索质量指标计算
与 scripts/evaluate.py 的指标公式保持一致
"""

import math
from typing import Dict, List, Any

from ..models import EvalSample


def compute_retrieval_metrics(
    samples: List[EvalSample],
    top_k_values: List[int],
) -> Dict[str, Any]:
    """
    计算检索指标

    指标:
        Recall@k:   前k个结果中有相关文档的查询比例
        Precision@k:前k个结果中相关文档的比例均值
        F1@k:       调和平均
        MRR:        第一个相关文档排名倒数的均值
        HitRate:    至少召回一个相关文档的查询比例
        NDCG@k:     归一化折损累计增益
        AvgFirstRelevantRank: 首次相关文档的平均排名
    """
    total = len(samples)
    if total == 0:
        return {"total_queries": 0, "error_count": 0}

    metrics = {}

    for k in top_k_values:
        # Recall@k
        recall = sum(
            1 for s in samples if _has_relevant_at_k(s, k)
        ) / total

        # Precision@k
        precision = sum(
            _precision_at_k(s, k) for s in samples
        ) / total

        f1 = _safe_f1(precision, recall)

        metrics[f"Recall@{k}"] = round(recall, 4)
        metrics[f"Precision@{k}"] = round(precision, 4)
        metrics[f"F1@{k}"] = round(f1, 4)

        # NDCG@k
        ndcg = sum(_ndcg_at_k(s, k) for s in samples) / total
        metrics[f"NDCG@{k}"] = round(ndcg, 4)

    # MRR (fix: parenthesize correctly + guard empty)
    mrr = sum(
        1.0 / (s.doc_relevance_flags.index(True) + 1)
        if True in s.doc_relevance_flags
        else 0.0
        for s in samples
        if s.doc_relevance_flags is not None
    ) / total
    metrics["MRR"] = round(mrr, 4)

    # HitRate
    hit = sum(
        1 for s in samples if True in s.doc_relevance_flags
    ) / total
    metrics["HitRate"] = round(hit, 4)

    # AvgFirstRelevantRank
    first_ranks = [
        s.doc_relevance_flags.index(True) + 1
        for s in samples
        if True in s.doc_relevance_flags
    ]
    metrics["AvgFirstRelevantRank"] = (
        round(sum(first_ranks) / len(first_ranks), 2) if first_ranks else None
    )

    metrics["total_queries"] = total
    metrics["error_count"] = sum(1 for s in samples if s.error)

    return metrics


# ------------------------------------------------------------------
#  Internal helpers
# ------------------------------------------------------------------


def _has_relevant_at_k(sample: EvalSample, k: int) -> bool:
    """前 k 个位置是否有相关文档"""
    flags = sample.doc_relevance_flags[:k]
    return any(flags)


def _precision_at_k(sample: EvalSample, k: int) -> float:
    """前 k 个中相关文档占比 (匹配 evaluate_recall.py 公式)"""
    flags = sample.doc_relevance_flags[:k]
    relevant = sum(1 for f in flags if f)
    # 与 evaluate_recall.py 一致：min(k, num_retrieved)
    num_retrieved = len(sample.retrieved_docs) or len(sample.doc_relevance_flags) or k
    count = min(k, num_retrieved)
    return relevant / count if count > 0 else 0.0


def _safe_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _dcg_at_k(relevance: List[float], k: int) -> float:
    """折损累计增益"""
    dcg = 0.0
    for i in range(min(k, len(relevance))):
        rel = relevance[i]
        # DCG formula: (2^rel - 1) / log2(i+2)
        dcg += (2**rel - 1) / math.log2(i + 2)
    return dcg


def _ndcg_at_k(sample: EvalSample, k: int) -> float:
    """
    NDCG@k: 归一化折损累计增益
    relevance = 1 if doc_relevance_flags[i] else 0
    """
    relevance = [1.0 if f else 0.0 for f in sample.doc_relevance_flags]
    dcg = _dcg_at_k(relevance, k)

    # IDCG: 理想排序（最多 min(k, total_docs) 个 1）
    ideal = sorted(relevance, reverse=True)
    idcg = _dcg_at_k(ideal, k)

    return dcg / idcg if idcg > 0 else 0.0
