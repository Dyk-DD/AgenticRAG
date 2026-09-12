"""
生成质量指标计算
评估 AI 回答的忠实度、幻觉率、完整性和合规性
"""

from typing import Dict, List, Any

from ..models import EvalSample

# 判定项 -> 指标名 / 样本字段名
_LABEL_TO_METRIC = {
    "faithfulness": "Faithfulness",
    "hallucination": "HallucinationRate",
    "completeness": "Completeness",
    "answer_relevance": "AnswerRelevance",
}
_LABEL_TO_FIELD = {
    "faithfulness": "faithfulness_score",
    "hallucination": "hallucination_score",
    "completeness": "completeness_score",
    "answer_relevance": "answer_relevance_score",
}


def _avg_valid(samples: List[EvalSample], label: str, attr: str):
    """只对拿到真实裁决的样本求均值

    Judge 回退时样本里存的是默认值（0.5 / False），把它算进均值等于凭空
    造了一个数字。样本全部回退时返回 None —— 没有可报的数，就不要报数。

    Returns:
        (均值或 None, 有效样本数, 回退样本数)
    """
    valid = [s for s in samples if label not in s.judge_failures]
    fallbacks = len(samples) - len(valid)
    if not valid:
        return None, 0, fallbacks
    return round(sum(getattr(s, attr) for s in valid) / len(valid), 4), len(valid), fallbacks


def compute_generation_metrics(
    samples: List[EvalSample],
) -> Dict[str, Any]:
    """
    计算生成指标

    指标:
        Faithfulness:        avg(faithfulness_score) 回答中的声明被文档支持的比率
        HallucinationRate:   avg(hallucination_score) 编造医学事实的比率 (越低越好)
        Completeness:        avg(completeness_score) 回答覆盖用户问题所有方面的比率
        AnswerRelevance:     avg(answer_relevance_score) 回答与问题的整体相关性
        HasDisclaimer:       包含医疗免责声明的回答比例
        UnhelpfulRate:       含"抱歉/无法回答"的回答比例
        GenerationLatencyMs: avg(生成耗时)
    """
    total = len(samples)
    if total == 0:
        return {}

    # 过滤出有生成的样本
    with_generation = [s for s in samples if s.generated_answer]
    if not with_generation:
        # 用独立的 skipped 标记，不要复用 note —— note 现在也用来提示
        # "有 Judge 回退"，那是"指标仍然有意义"的情况，不该触发报告的提前返回
        return {"skipped": True, "note": "没有生成回答的样本，跳过生成评估"}

    n = len(with_generation)

    metrics: Dict[str, Any] = {}

    # 四个 Judge 指标：排除回退样本后再平均，并同时报出有效/回退样本数。
    # 回退样本的分数是默认值，混进来就是编造指标。
    for label, key in _LABEL_TO_METRIC.items():
        avg, n_valid, n_fb = _avg_valid(with_generation, label, _LABEL_TO_FIELD[label])
        metrics[key] = avg                       # None = 无有效裁决，报告应显示 N/A
        metrics[f"{key}_N"] = n_valid
        metrics[f"{key}_Fallbacks"] = n_fb

    total_fallbacks = sum(
        metrics[f"{key}_Fallbacks"] for key in _LABEL_TO_METRIC.values()
    )
    metrics["JudgeFallbacks"] = total_fallbacks
    if total_fallbacks:
        metrics["note"] = (
            f"{total_fallbacks} 次 Judge 回退（默认值）已被排除在均值之外，"
            f"相关指标基于更少的样本；请检查 LLM_MODEL 与 max_tokens"
        )

    metrics["GenerationLatencyMs"] = round(
        sum(s.generation_latency_ms for s in with_generation) / n, 2
    )

    # 免责声明检测
    disclaimer_count = sum(
        1 for s in with_generation if "免责声明" in s.generated_answer
    )
    metrics["HasDisclaimer"] = round(disclaimer_count / n, 4)

    # 无帮助回答率
    unhelpful_count = sum(
        1 for s in with_generation
        if "抱歉" in s.generated_answer or "无法回答" in s.generated_answer
    )
    metrics["UnhelpfulRate"] = round(unhelpful_count / n, 4)

    metrics["total_generated"] = n
    return metrics
