"""
JSON 报告生成器
"""

import json
import logging
import os
from typing import Dict, Any, Optional

from ..models import EvalReport, EvalSample

logger = logging.getLogger(__name__)


class JSONReporter:
    """生成结构化 JSON 评估报告"""

    @staticmethod
    def generate(report: EvalReport) -> Dict[str, Any]:
        """将 EvalReport 转换为可序列化的字典"""
        return {
            "meta": {
                "timestamp": report.timestamp,
                "config": {
                    "mode": report.config.mode.value,
                    "stages": [s.value for s in report.config.stages],
                    "judge_model": report.config.judge_model,
                    "top_k_values": report.config.top_k_values,
                    "strategies": report.config.strategies,
                    "paraphrase": report.config.paraphrase,
                    "sample_size": report.config.sample_size,
                },
                "system_info": report.system_info,
            },
            "retrieval": report.retrieval_metrics,
            "routing": report.routing_metrics,
            "generation": report.generation_metrics,
            "end_to_end": report.e2e_metrics,
            # degraded=True 时说明有 Judge 判定回退到默认值，相关指标不可信
            "judge_health": report.judge_health,
            "strategy_breakdown": report.strategy_breakdown,
            "department_breakdown": report.department_breakdown,
            "samples": [JSONReporter._serialize_sample(s) for s in report.samples],
        }

    @staticmethod
    def save(report: EvalReport, path: str):
        """保存 JSON 报告到文件"""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = JSONReporter.generate(report)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"评估报告已保存: {path}")

    @staticmethod
    def _serialize_sample(sample: EvalSample) -> Dict[str, Any]:
        """序列化单个评估样本"""
        return {
            "query_id": sample.test_query.query_id,
            "query": sample.test_query.query_text[:80],
            "department": sample.test_query.department,
            "strategy_used": sample.strategy_used,
            "error": sample.error,
            "retrieval_latency_ms": round(sample.retrieval_latency_ms, 2),
            "generation_latency_ms": round(sample.generation_latency_ms, 2),
            "total_latency_ms": round(sample.total_latency_ms, 2),
            "doc_count": len(sample.retrieved_docs),
            "relevant_count": sum(1 for f in sample.doc_relevance_flags if f),
            "doc_relevance_flags": sample.doc_relevance_flags,
            "faithfulness_score": round(sample.faithfulness_score, 4),
            "hallucination_score": round(sample.hallucination_score, 4),
            "completeness_score": round(sample.completeness_score, 4),
            "answer_relevance_score": round(sample.answer_relevance_score, 4),
            # 非空表示上面某个分数是默认值而非真实裁决
            "judge_failures": sample.judge_failures,
            "judge_failure_reasons": sample.judge_failure_reasons,
        }
