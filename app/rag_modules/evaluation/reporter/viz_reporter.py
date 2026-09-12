"""
可视化数据生成器
生成可供图表库（matplotlib / plotly / 前端 ECharts）使用的数据结构
"""

from typing import Dict, List, Any
from collections import defaultdict

from ..models import EvalReport


class VizReporter:
    """
    生成可视化就绪数据

    使用方式:
        VizReporter.generate_radar_data(report)  → 雷达图
        VizReporter.generate_comparison_data(report) → 对比柱状图
        VizReporter.generate_timeseries_data(reports) → 时间序列
    """

    @staticmethod
    def generate_radar_data(report: EvalReport) -> Dict[str, Any]:
        """
        生成五维雷达图数据

        维度: Recall@5, Faithfulness, AnswerRelevance, Diversity, Latency(反向)
        """
        r = report.retrieval_metrics
        g = report.generation_metrics
        e = report.e2e_metrics

        # 延迟反归一化 (越小越好 → 越大越好)
        latency_raw = e.get("TotalLatencyMs", 5000)
        latency_score = max(0, 1.0 - latency_raw / 10000)

        # Judge 回退时对应指标是 None（无有效裁决）。这时既不能 *100（TypeError），
        # 也不能画成 0（那是在图里编造一个"很差的分数"）——给 None 并在
        # 顶层标记 degraded，让前端把该轴置灰。
        health = report.judge_health or {}

        def _pct(value):
            return None if value is None else value * 100

        return {
            "dimensions": ["检索准确率", "生成忠实度", "回答相关性", "结果多样性", "响应速度"],
            "metrics": {
                "value": [
                    _pct(r.get("Recall@5", 0)),
                    _pct(g.get("Faithfulness")),
                    _pct(g.get("AnswerRelevance")),
                    _pct(r.get("IntraListDiversity@k", 0)),
                    latency_score * 100,
                ],
            },
            "unit": "%",
            # 为真时上面 value 里可能有 None，且相关维度不可信
            "judge_degraded": bool(health.get("degraded")),
            "warning": health.get("summary"),
        }

    @staticmethod
    def generate_comparison_data(
        report: EvalReport,
    ) -> Dict[str, Any]:
        """
        生成多策略对比柱状图数据

        比较: hybrid / graph_rag / combined 的 hit_rate + latency
        """
        breakdown = report.strategy_breakdown
        if not breakdown:
            return {}

        strategies = list(breakdown.keys())
        return {
            "strategies": strategies,
            "metrics": {
                "hit_rate": [
                    breakdown[s].get("hit_rate", 0) * 100 for s in strategies
                ],
                "avg_latency_ms": [
                    breakdown[s].get("avg_latency_ms", 0) for s in strategies
                ],
                "count": [breakdown[s].get("count", 0) for s in strategies],
            },
        }

    @staticmethod
    def generate_timeseries_data(
        reports: List[EvalReport],
    ) -> Dict[str, Any]:
        """
        生成时间序列折线图数据

        输入: 多次评估的 EvalReport 列表
        输出: 每个指标的时序数据
        """
        timestamps = [r.timestamp[:19] for r in reports]

        series = defaultdict(list)
        for r in reports:
            # 提取关键指标
            rm = r.retrieval_metrics
            series["Recall@5"].append(rm.get("Recall@5", 0))
            series["MRR"].append(rm.get("MRR", 0))
            series["HitRate"].append(rm.get("HitRate", 0))

            gm = r.generation_metrics
            series["Faithfulness"].append(gm.get("Faithfulness", 0))
            series["HallucinationRate"].append(gm.get("HallucinationRate", 0))

            e2e = r.e2e_metrics
            series["LatencyMs"].append(e2e.get("TotalLatencyMs", 0))

        return {
            "timestamps": timestamps,
            "series": dict(series),
        }

    @staticmethod
    def generate_department_chart(
        report: EvalReport,
    ) -> Dict[str, Any]:
        """
        生成科室维度横向柱状图数据
        """
        dept = report.department_breakdown
        if not dept:
            return {}

        sorted_depts = sorted(dept.items(), key=lambda x: x[1].get("Recall@5", 0), reverse=True)

        return {
            "departments": [d[0] for d in sorted_depts],
            "recall_at_5": [d[1].get("Recall@5", 0) * 100 for d in sorted_depts],
            "count": [d[1].get("count", 0) for d in sorted_depts],
        }

    @staticmethod
    def generate_routing_pie(report: EvalReport) -> Dict[str, Any]:
        """生成路由分布饼图数据"""
        rm = report.routing_metrics
        dist = rm.get("routing_distribution", {})
        if not dist:
            sb = report.strategy_breakdown
            dist = {k: v["count"] for k, v in sb.items()}

        return {
            "labels": list(dist.keys()),
            "values": list(dist.values()),
        }
