#!/usr/bin/env python
"""
三种检索策略对比测试
========================
使用 sample_testset.py 预先采样的测试集，
分别用 hybrid / graph_rag / combined 三种策略检索，
对比召回率、精确率、MRR、HitRate 等指标，结果保存到本目录。

使用方法:
    1. 先采样（仅执行一次）
       conda activate all-in-rag
       cd scripts/strategy_comparison
       python sample_testset.py

    2. 再评估
       python run_comparison.py
"""

import os
import sys
import json
import time
import logging
from contextlib import redirect_stdout
from io import StringIO

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
for lib in ["neo4j", "httpx", "httpcore", "urllib3", "pymilvus", "jieba"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

logger = logging.getLogger("strategy_comparison")

# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    # 输出/脚本目录
    output_dir = os.path.dirname(os.path.abspath(__file__))
    # 预采样的测试集（使用 LLM 语义改写后的版本，更贴近真实用户提问）
    sampled_csv = os.path.join(output_dir, "sampled_100_qa_rewritten_labeled.csv")

    if not os.path.exists(sampled_csv):
        print(f"❌ 未找到采样文件: {sampled_csv}")
        print(f"   请先运行: python sample_testset.py")
        sys.exit(1)

    print("=" * 62)
    print("  三种检索策略对比测试")
    print(f"  测试集: {sampled_csv}")
    print(f"  输出目录: {output_dir}")
    print("=" * 62)

    # ------------------------------------------------------------------
    #  1. 初始化系统
    # ------------------------------------------------------------------
    print("\n[1/5] 正在初始化 ClinicalDecisionSystem...")
    from app.main import ClinicalDecisionSystem
    system = ClinicalDecisionSystem()
    system.initialize_system()
    print("  ✅ 系统模块初始化完成")

    # 构建/加载知识库（建立 Neo4j 连接、Milvus 向量索引、BM25 索引）
    # 首次运行会从 CSV 重建索引，之后检测已有数据则跳过
    print("\n  ⟳ 构建/加载知识库（首次会重建 Neo4j + Milvus 索引，可能需要数分钟）...")
    system.build_knowledge_base()
    print("  ✅ 知识库就绪")

    # ------------------------------------------------------------------
    #  2. 配置评估
    # ------------------------------------------------------------------
    print("\n[2/5] 配置评估参数...")
    from app.rag_modules.evaluation import (
        EvalConfig, EvalMode, Evaluator,
    )
    from app.rag_modules.evaluation.reporter.console_reporter import ConsoleReporter
    from app.rag_modules.evaluation.reporter.json_reporter import JSONReporter

    config = EvalConfig(
        mode=EvalMode.OFFLINE,
        test_csv_path=sampled_csv,
        strategies=["hybrid", "graph_rag", "combined"],
        top_k_values=[3, 5, 10],
        stages=[],
        llm_judge=False,
        measure_diversity=False,
    )
    print(f"  策略: {config.strategies}")
    print(f"  Top-K: {config.top_k_values}")

    # ------------------------------------------------------------------
    #  3. 执行评估
    # ------------------------------------------------------------------
    print("\n[3/5] 开始评估（逐策略逐条检索，请耐心等待）...")
    t_start = time.perf_counter()

    evaluator = Evaluator(config=config, system=system)
    report = evaluator.evaluate()

    elapsed = time.perf_counter() - t_start
    print(f"\n  ✅ 评估完成! 耗时: {elapsed:.1f}s")

    # ------------------------------------------------------------------
    #  4. 路由准确率评估（独立于 Evaluator，使用 IntelligentQueryRouter）
    # ------------------------------------------------------------------
    print("\n[4/5] 路由准确率评估...")
    routing_metrics = _evaluate_routing_accuracy(system, sampled_csv)

    # ------------------------------------------------------------------
    #  5. 保存报告
    # ------------------------------------------------------------------
    print("\n[5/5] 保存报告...")

    # 4a. TXT 完整报告
    txt_path = os.path.join(output_dir, "strategy_comparison_report.txt")
    buf = StringIO()
    with redirect_stdout(buf):
        ConsoleReporter.print_report(report, detailed=True)
    console_output = buf.getvalue()

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 62 + "\n")
        f.write("  Agentic-RAG 三种检索策略对比测试报告\n")
        f.write(f"  生成时间: {report.timestamp[:19]}\n")
        f.write(f"  测试集: sampled_100_qa_rewritten.csv (LLM 改写版)\n")
        f.write(f"  评估耗时: {elapsed:.1f}s\n")
        f.write("=" * 62 + "\n")
        f.write(console_output)
        # 追加路由准确率
        if routing_metrics:
            f.write("\n" + "=" * 62 + "\n")
            f.write("  路由准确率评估（IntelligentQueryRouter × LLM 标签）\n")
            f.write("=" * 62 + "\n")
            f.write(f"  总查询数:   {routing_metrics.get('total_queries', 0)}\n")
            f.write(f"  正确数:     {routing_metrics.get('correct_count', 0)}\n")
            f.write(f"  路由准确率: {routing_metrics.get('overall_accuracy', 0):.2%}\n")
            f.write("-" * 62 + "\n")
            f.write(f"  {'#':<3} {'科室':<6} {'标题':<24} {'真实策略':<10} {'预测策略':<10} {'✓/✗'}\n")
            f.write("-" * 62 + "\n")
            for d in routing_metrics.get("details", []):
                mark = "✓" if d["correct"] else "✗"
                f.write(f"  {d['index']:<3} {d['dept']:<6} {d['title']:<24} {d['expected']:<10} {d['actual']:<10} {mark}\n")
            f.write("=" * 62 + "\n")

    # 4b. JSON 报告
    json_path = os.path.join(output_dir, "strategy_comparison_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        data = JSONReporter.generate(report)
        data["routing_accuracy"] = routing_metrics
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 4c. 对比摘要
    summary_path = os.path.join(output_dir, "comparison_summary.txt")
    _write_summary(summary_path, report, elapsed, routing_metrics)

    print(f"\n  ✅ 报告已保存至: {output_dir}")
    print(f"     - strategy_comparison_report.txt  (完整报告)")
    print(f"     - strategy_comparison_report.json (结构化数据)")
    print(f"     - comparison_summary.txt          (对比摘要)")

    # 终端输出
    ConsoleReporter.print_report(report)

    # 路由准确率终端输出
    if routing_metrics:
        print("\n" + "=" * 62)
        print("  路由准确率评估（IntelligentQueryRouter × LLM 标签）")
        print("=" * 62)
        print(f"  总查询数:   {routing_metrics.get('total_queries', 0)}")
        print(f"  正确数:     {routing_metrics.get('correct_count', 0)}")
        print(f"  路由准确率: {routing_metrics.get('overall_accuracy', 0):.2%}")
        print("=" * 62)


def _write_summary(filepath, report, elapsed, routing_metrics=None):
    """写入简洁的对比摘要"""
    sb = report.strategy_breakdown or {}
    e2e = report.e2e_metrics or {}

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 62 + "\n")
        f.write("  三种检索策略对比摘要\n")
        f.write(f"  时间: {report.timestamp[:19]}\n")
        f.write("=" * 62 + "\n\n")

        # 按策略展示完整检索指标
        f.write("【各策略检索指标】\n")
        for strategy in sorted(sb.keys()):
            s = sb.get(strategy, {})
            if not s:
                continue
            ret = s.get("retrieval_metrics", {})
            f.write(f"\n  【{strategy}】")
            f.write(f"  查询数={s.get('count', 0)}, 命中率={s.get('hit_rate', 0):.2%}, "
                    f"延迟={s.get('avg_latency_ms', 0):.0f}ms\n")
            if ret:
                for k in sorted(ret.keys()):
                    val = ret[k]
                    if isinstance(val, float):
                        if "NDCG" in k or "MRR" in k:
                            f.write(f"    {k:<18} = {val:.4f}\n")
                        else:
                            f.write(f"    {k:<18} = {val:.2%}\n")
                    else:
                        f.write(f"    {k:<18} = {val}\n")
        f.write("\n")

        # 路由准确率（独立评估）
        if routing_metrics:
            f.write("【路由准确率（IntelligentQueryRouter × LLM 标签）】\n")
            f.write(f"  总查询数:   {routing_metrics.get('total_queries', 0)}\n")
            f.write(f"  正确数:     {routing_metrics.get('correct_count', 0)}\n")
            acc = routing_metrics.get('overall_accuracy', 0)
            f.write(f"  路由准确率: {acc:.2%}\n")
            f.write(f"  {'#':<3} {'科室':<6} {'标题':<24} {'真实策略':<10} {'预测策略':<10} {'✓/✗'}\n")
            f.write("  " + "-" * 58 + "\n")
            for d in routing_metrics.get("details", []):
                mark = "✓" if d["correct"] else "✗"
                f.write(f"  {d['index']:<3} {d['dept']:<6} {d['title']:<24} {d['expected']:<10} {d['actual']:<10} {mark}\n")
            f.write("\n")

        # 性能
        f.write("【性能】\n")
        if e2e.get("TotalLatencyMs"):
            f.write(f"  端到端耗时: {e2e['TotalLatencyMs']:.0f}ms\n")
        if e2e.get("P95LatencyMs"):
            f.write(f"  P95 延迟:   {e2e['P95LatencyMs']:.0f}ms\n")
        f.write(f"  评估总耗时: {elapsed:.1f}s\n")
        if e2e.get("ErrorRate"):
            f.write(f"  错误率:     {e2e['ErrorRate']:.2%}\n")

        f.write("\n" + "=" * 62 + "\n")


def _evaluate_routing_accuracy(system, labeled_csv_path) -> dict:
    """
    独立路由准确率评估
    使用 system.router.analyze_query() 分析查询路由策略（不触发检索），
    与 CSV 中已标注的 expected_routing_strategy 对比，
    并将预测结果追加到 CSV。

    Args:
        system: ClinicalDecisionSystem 实例（含 router）
        labeled_csv_path: 已标注路由策略的 CSV 路径（含 expected_routing_strategy 列）

    Returns:
        dict: 包含 accuracy、total_queries、correct_count、details
    """
    import pandas as pd

    if not os.path.exists(labeled_csv_path):
        print(f"  ⚠️  标注文件不存在: {labeled_csv_path}，跳过路由评估")
        return {}

    df = pd.read_csv(labeled_csv_path)
    if "expected_routing_strategy" not in df.columns:
        print(f"  ⚠️  CSV 缺少 expected_routing_strategy 列，跳过路由评估")
        return {}

    print("\n[路由准确率评估]")
    print(f"  加载标注文件: {labeled_csv_path} ({len(df)} 条)")
    print(f"  使用 IntelligentQueryRouter.analyze_query() 逐条路由分析...")

    # 统计
    correct = 0
    total = 0
    details = []
    predicted_col = []

    for idx, row in df.iterrows():
        expected = str(row["expected_routing_strategy"]).strip()
        if not expected:
            predicted_col.append("")
            continue

        # 构建查询文本（与 label_routing_ground_truth.py 一致）
        dept = str(row.get("department", ""))
        title = str(row.get("title", ""))
        ask = str(row.get("ask", ""))
        query_text = f"【科室】{dept}\n【标题】{title}\n【描述】{ask}"

        try:
            analysis = system.router.analyze_query(query_text)
            actual = analysis.recommended_strategy.value
        except Exception as e:
            print(f"\n  ❌ 第 {idx+1} 条路由失败: {e}")
            actual = "error"

        predicted_col.append(actual)

        is_correct = actual == expected
        if is_correct:
            correct += 1
        total += 1

        details.append({
            "index": idx + 1,
            "dept": dept,
            "title": title[:30],
            "expected": expected,
            "actual": actual,
            "correct": is_correct,
        })

        # 进度显示
        if (idx + 1) % 5 == 0 or idx == len(df) - 1:
            print(f"  ⟳ [{idx+1}/{len(df)}] 当前准确率: {correct/max(1,total):.1%} ({correct}/{total})")

    # ---- 计算总准确率 ----
    overall_accuracy = correct / total if total > 0 else 0

    # ---- 终端输出 ----
    print(f"\n  ✅ 路由评估完成!")
    print(f"  {'='*40}")
    print(f"  总查询数:   {total}")
    print(f"  正确数:     {correct}")
    print(f"  路由准确率: {overall_accuracy:.2%}")
    print(f"  {'='*40}")

    return {
        "total_queries": total,
        "correct_count": correct,
        "overall_accuracy": overall_accuracy,
        "details": details,
    }


if __name__ == "__main__":
    main()
