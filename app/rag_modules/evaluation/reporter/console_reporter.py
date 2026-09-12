"""
终端格式化报告输出
终端格式化报告输出，含 ASCII 柱状图
"""

import logging
from typing import Dict, Any, Optional

from ..models import EvalReport

logger = logging.getLogger(__name__)


class ConsoleReporter:
    """终端格式化报告输出"""

    @staticmethod
    def print_report(report: EvalReport, detailed: bool = False):
        """
        打印完整的格式化评估报告

        Args:
            report: EvalReport 实例
            detailed: 是否打印详细样本信息
        """
        print()
        print("=" * 62)
        print(f"  评估报告")
        print(f"  模式: {report.config.mode.value}  |  时间: {report.timestamp[:19]}")
        print("=" * 62)

        # Judge 健康度告警必须排在任何指标之前 —— 有回退时下面的数字不可信，
        # 先看到指标再看告警，人就已经把假数字记住了
        if (report.judge_health or {}).get("degraded"):
            ConsoleReporter._print_judge_health_warning(report.judge_health)

        # 检索指标
        if report.retrieval_metrics:
            ConsoleReporter._print_retrieval_section(report.retrieval_metrics)

        # 路由指标
        if report.routing_metrics:
            ConsoleReporter._print_routing_section(report.routing_metrics)

        # 生成指标
        if report.generation_metrics:
            ConsoleReporter._print_generation_section(report.generation_metrics)

        # 端到端指标
        if report.e2e_metrics:
            ConsoleReporter._print_e2e_section(report.e2e_metrics)

        # 策略分解
        if report.strategy_breakdown:
            ConsoleReporter._print_strategy_breakdown(report.strategy_breakdown)

        # 详细样本（可选）
        if detailed and report.samples:
            ConsoleReporter._print_samples(report.samples)

        print()
        print("=" * 62)
        print()

    # ------------------------------------------------------------------
    #  各节输出
    # ------------------------------------------------------------------

    @staticmethod
    def _print_judge_health_warning(health: Dict[str, Any]):
        """Judge 回退告警 —— 必须显眼

        Judge 拿不到真实裁决时会返回默认值（0.5 / False），与真实评分无法区分。
        不显式提示的话，报告会让人念出编造的指标。
        """
        print()
        print("  " + "!" * 56)
        print(f"  ⚠️  Judge 健康度告警: {health.get('summary')}")
        for label, count in (health.get("failed_labels") or {}).items():
            print(f"      - {label:<16} 回退 {count} 次")
        print("      → 上述维度指标不可信（请检查 LLM_MODEL / max_tokens）")
        print("  " + "!" * 56)

    @staticmethod
    def _print_retrieval_section(metrics: Dict[str, Any]):
        """打印检索指标"""
        total = metrics.get("total_queries", "?")
        print(f"\n  📊 检索指标 ({total} 条查询):")
        print(f"  " + "-" * 42)
        for k in [1, 3, 5, 10]:
            r = metrics.get(f"Recall@{k}")
            p = metrics.get(f"Precision@{k}")
            f = metrics.get(f"F1@{k}")
            if r is not None:
                print(f"    Recall@{k}    = {r:.2%}")
                print(f"    Precision@{k} = {p:.2%}")
                print(f"    F1@{k}        = {f:.2%}")
                print()
        if "MRR" in metrics:
            print(f"    MRR            = {metrics['MRR']}")
        if "HitRate" in metrics:
            print(f"    HitRate        = {metrics['HitRate']:.2%}")
        if "NDCG@5" in metrics:
            print(f"    NDCG@5         = {metrics['NDCG@5']}")
        if "AvgFirstRelevantRank" in metrics and metrics["AvgFirstRelevantRank"] is not None:
            print(f"    Avg首相关排名   = {metrics['AvgFirstRelevantRank']}")
        if metrics.get("error_count", 0) > 0:
            print(f"    ⚠️  异常数: {metrics['error_count']}")

        # 多样性指标
        for div_key in ["IntraListDiversity@k", "TopicCoverage", "RRFContributionSpread"]:
            if div_key in metrics:
                print(f"    {div_key:<15} = {metrics[div_key]:.2%}")

    @staticmethod
    def _print_routing_section(metrics: Dict[str, Any]):
        """打印路由指标"""
        print(f"\n  🧭 路由指标:")
        print(f"  " + "-" * 42)
        if "note" in metrics:
            print(f"    {metrics['note']}")
            return
        print(f"    RoutingAccuracy    = {metrics.get('RoutingAccuracy', 'N/A'):.2%}")
        print(f"    HybridPrecision    = {metrics.get('HybridPrecision', 'N/A')}")
        print(f"    GraphRAGPrecision  = {metrics.get('GraphRAGPrecision', 'N/A')}")
        print(f"    CombinedPrecision  = {metrics.get('CombinedPrecision', 'N/A')}")
        print(f"    ConfidenceCalib.   = {metrics.get('ConfidenceCalibration', 'N/A')}")
        print(f"    RoutingLatencyMs   = {metrics.get('RoutingLatencyMs', 'N/A')}")
        dist = metrics.get("routing_distribution", {})
        if dist:
            print(f"    分布: {dict(dist)}")

    @staticmethod
    def _print_generation_section(metrics: Dict[str, Any]):
        """打印生成指标"""
        print(f"\n  ✍️  生成指标:")
        print(f"  " + "-" * 42)
        if metrics.get("skipped"):
            print(f"    {metrics.get('note', '跳过生成评估')}")
            return

        # 有 Judge 回退时对应指标是 None（没有真实裁决可报），
        # 不能直接 :.2% —— 那会 TypeError 把整个报告打挂
        for key in ("Faithfulness", "HallucinationRate", "Completeness", "AnswerRelevance"):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                line = f"    {key:<18} = {value:.2%}"
            else:
                line = f"    {key:<18} = N/A（无有效裁决）"
            fb = metrics.get(f"{key}_Fallbacks", 0)
            n_valid = metrics.get(f"{key}_N")
            if fb:
                line += f"   [回退 {fb}，基于 {n_valid} 个有效样本]"
            print(line)

        if metrics.get("note"):
            print(f"    ⚠️  {metrics['note']}")
        if "HasDisclaimer" in metrics:
            print(f"    HasDisclaimer      = {metrics['HasDisclaimer']:.2%}")
        if "UnhelpfulRate" in metrics:
            print(f"    UnhelpfulRate      = {metrics['UnhelpfulRate']:.2%}")

    @staticmethod
    def _print_e2e_section(metrics: Dict[str, Any]):
        """打印端到端指标"""
        print(f"\n  ⏱️  端到端指标:")
        print(f"  " + "-" * 42)
        print(f"    TotalLatencyMs     = {metrics.get('TotalLatencyMs', 'N/A')}")
        print(f"    P95LatencyMs       = {metrics.get('P95LatencyMs', 'N/A')}")
        print(f"    RetrievalLatencyMs = {metrics.get('RetrievalLatencyMs', 'N/A')}")
        print(f"    ErrorRate          = {metrics.get('ErrorRate', 'N/A'):.2%}")

    @staticmethod
    def _print_strategy_breakdown(breakdown: Dict[str, Dict[str, Any]]):
        """打印策略对比 — 每个策略展示完整检索指标"""
        print(f"\n  📈 策略对比:")
        print(f"  " + "-" * 52)
        for strategy in sorted(breakdown.keys()):
            s = breakdown.get(strategy, {})
            ret = s.get("retrieval_metrics", {})
            print(f"\n  【{strategy}】")
            print(f"  {'查询数':<10} {'命中率':<12} {'延迟(ms)':<12}")
            print(f"  {s.get('count', 0):<10} {s.get('hit_rate', 0):<12.2%} {s.get('avg_latency_ms', 0):<12.2f}")
            if ret:
                for k in sorted(ret.keys()):
                    val = ret[k]
                    if isinstance(val, float):
                        if "NDCG" in k or "MRR" in k:
                            print(f"    {k:<18} = {val:.4f}")
                        else:
                            print(f"    {k:<18} = {val:.2%}")
                    else:
                        print(f"    {k:<18} = {val}")

    @staticmethod
    def _print_samples(samples):
        """打印详细样本信息"""
        print(f"\n  📝 样本详情 (前10条):")
        print(f"  " + "-" * 62)
        for s in samples[:10]:
            rel = sum(1 for f in s.doc_relevance_flags if f) if s.doc_relevance_flags else 0
            total = len(s.doc_relevance_flags)
            print(f"    [{s.test_query.query_id}] {s.test_query.query_text[:40]:<40}")
            print(f"    策略: {s.strategy_used:<20} 相关: {rel}/{total}  延迟: {s.retrieval_latency_ms:.0f}ms")
            if s.error:
                print(f"    ⚠️  {s.error}")
