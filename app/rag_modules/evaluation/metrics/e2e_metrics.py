"""
端到端质量指标计算
评估整体响应延迟、错误率等
"""

import statistics
from typing import Dict, List, Any

from ..models import EvalSample


def compute_e2e_metrics(
    samples: List[EvalSample],
) -> Dict[str, Any]:
    """
    计算端到端指标

    指标:
        TotalLatencyMs:  avg(端到端耗时)
        P95LatencyMs:    P95 分位耗时
        RetrievalLatencyMs: avg(检索耗时)
        ErrorRate:       异常查询比例
    """
    total = len(samples)
    if total == 0:
        return {}

    metrics = {}

    # 延迟
    latencies = [s.total_latency_ms for s in samples if s.total_latency_ms > 0]
    if latencies:
        metrics["TotalLatencyMs"] = round(sum(latencies) / len(latencies), 2)
        p95 = (
            statistics.quantiles(latencies, n=20)[-1]
            if len(latencies) >= 20
            else max(latencies)
        )
        metrics["P95LatencyMs"] = round(p95, 2)
    else:
        metrics["TotalLatencyMs"] = 0
        metrics["P95LatencyMs"] = 0

    # 检索延迟
    retrieval_latencies = [
        s.retrieval_latency_ms for s in samples if s.retrieval_latency_ms > 0
    ]
    metrics["RetrievalLatencyMs"] = (
        round(sum(retrieval_latencies) / len(retrieval_latencies), 2)
        if retrieval_latencies else 0
    )

    # 错误率
    error_count = sum(1 for s in samples if s.error)
    metrics["ErrorRate"] = round(error_count / total, 4)
    metrics["total_queries"] = total
    metrics["error_count"] = error_count

    return metrics
