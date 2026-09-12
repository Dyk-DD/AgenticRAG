"""
多样性指标计算
评估检索结果的来源多样性、节点类型覆盖度等
"""

from typing import Dict, List, Any

from ..models import EvalSample


def compute_diversity_metrics(
    samples: List[EvalSample],
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    计算多样性指标

    指标:
        IntraListDiversity@k:  前k个结果中唯一检索来源的比例
        TopicCoverage:         节点类型覆盖度（entity/topic/graph relation 等）
        RRFContributionSpread: RRF 贡献源的分布广度
        SourceTypeEntropy:     检索来源类型的香农熵
    """
    total = len(samples)
    if total == 0:
        return {}

    metrics = {}

    # Intra-List Diversity
    diversity_scores = []
    for s in samples:
        doc_types = [
            d.metadata.get("search_type", d.metadata.get("search_method", "unknown"))
            for d in s.retrieved_docs[:top_k]
            if d.metadata
        ]
        unique_types = len(set(doc_types))
        diversity_scores.append(unique_types / max(top_k, 1))

    metrics["IntraListDiversity@k"] = round(
        sum(diversity_scores) / total, 4
    )

    # Topic Coverage: 节点类型覆盖度
    all_node_types = {
        "entity", "topic", "dual_level", "vector_enhanced",
        "medical_graph_path", "medical_subgraph",
    }
    coverage_scores = []
    for s in samples:
        retrieved_types = {
            d.metadata.get("retrieval_level", d.metadata.get("search_type", ""))
            for d in s.retrieved_docs[:top_k]
            if d.metadata
        }
        coverage = len(retrieved_types & all_node_types) / max(len(all_node_types), 1)
        coverage_scores.append(coverage)

    metrics["TopicCoverage"] = round(sum(coverage_scores) / total, 4)

    # RRF Contribution Spread
    spread_scores = []
    for s in samples:
        sources = set()
        for d in s.retrieved_docs[:top_k]:
            if d.metadata:
                contribs = d.metadata.get("rrf_contributions", [])
                for c in contribs:
                    sources.add(c.get("source", ""))
        spread = len(sources) / max(top_k, 1)
        spread_scores.append(spread)

    metrics["RRFContributionSpread"] = round(
        sum(spread_scores) / total, 4
    )

    return metrics
