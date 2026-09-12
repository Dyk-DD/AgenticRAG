"""
路由质量指标计算
评估查询路由的准确率、各策略精度、置信度校准等
"""

from typing import Dict, List, Any

from ..models import EvalSample


def compute_routing_metrics(
    samples: List[EvalSample],
) -> Dict[str, Any]:
    """
    计算路由指标

    前提: sample.test_query.expected_routing_strategy 已设置
    指标:
        RoutingAccuracy:   路由策略与 ground truth 的匹配率
        HybridPrecision:   路由到 hybrid 且正确的比例
        GraphRAGPrecision: 路由到 graph_rag 且正确的比例
        ConfidenceCalibration: 置信度校准误差 (Brier score)
        RoutingLatencyMs:  avg(路由耗时)
        routing_distribution: {策略: 次数} 分布
    """
    total = len(samples)
    if total == 0:
        return {}

    # 过滤出有 ground truth 的样本（只统计路由器自然决策，排除强制覆盖）
    labeled = [s for s in samples if s.test_query.expected_routing_strategy and s.is_natural_route]
    if not labeled:
        return {"note": "没有标注 ground truth 路由策略，跳过路由评估"}

    correct = 0
    hybrid_total = 0
    hybrid_correct = 0
    graphrag_total = 0
    graphrag_correct = 0
    combined_total = 0
    combined_correct = 0
    total_confidence = 0.0
    total_calibration_error = 0.0
    routing_distribution: Dict[str, int] = {}
    latency_sum = 0.0
    latency_count = 0

    for s in labeled:
        expected = s.test_query.expected_routing_strategy
        actual = s.actual_routing or s.strategy_used

        # 更新分布
        routing_distribution[actual] = routing_distribution.get(actual, 0) + 1

        # 是否匹配
        is_match = (expected == actual)
        if is_match:
            correct += 1

        # 各策略精度统计
        if actual == "hybrid_traditional" or actual == "hybrid":
            hybrid_total += 1
            if is_match:
                hybrid_correct += 1
        elif actual in ("graph_rag", "graph_rag_retrieval"):
            graphrag_total += 1
            if is_match:
                graphrag_correct += 1
        elif actual == "combined":
            combined_total += 1
            if is_match:
                combined_correct += 1

        # 置信度校准
        confidence = 0.0
        if s.query_analysis:
            confidence = getattr(s.query_analysis, "confidence", 0.0)
        total_confidence += confidence
        total_calibration_error += (confidence - (1.0 if is_match else 0.0)) ** 2

        # 延迟
        if s.retrieval_latency_ms > 0:
            latency_sum += s.retrieval_latency_ms
            latency_count += 1

    metrics = {
        "RoutingAccuracy": round(correct / len(labeled), 4),
        "HybridPrecision": round(hybrid_correct / hybrid_total, 4) if hybrid_total > 0 else None,
        "GraphRAGPrecision": round(graphrag_correct / graphrag_total, 4) if graphrag_total > 0 else None,
        "CombinedPrecision": round(combined_correct / combined_total, 4) if combined_total > 0 else None,
        "ConfidenceCalibration": round(total_calibration_error / len(labeled), 4),
        "AvgConfidence": round(total_confidence / len(labeled), 4),
        "RoutingLatencyMs": round(latency_sum / latency_count, 2) if latency_count > 0 else 0,
        "routing_distribution": routing_distribution,
        "labeled_count": len(labeled),
    }

    return metrics
