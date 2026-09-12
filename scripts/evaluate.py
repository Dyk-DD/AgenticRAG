#!/usr/bin/env python
"""
Agentic-RAG 系统统一评估 CLI
================================
全面评估检索/路由/生成/端到端各环节质量。

离线模式 (默认):  使用 CSV 测试集
在线模式:         从 SQLite 历史对话中评估

使用方法:
  # 离线: 检索评估 (默认)
  python scripts/evaluate.py --csv data/test_20.csv

  # 离线: 全阶段评估 (检索+路由+生成)
  python scripts/evaluate.py --csv data/test_20.csv --stages retrieval routing generation e2e

  # 离线: 指定策略 + LLM Judge
  python scripts/evaluate.py --strategies hybrid graph_rag combined --llm-judge

  # 在线: 从历史对话评估
  python scripts/evaluate.py --mode online --limit 50

  # 输出 JSON + 详细终端报告
  python scripts/evaluate.py --output report.json --detailed

  # 快速基线 (仅检索)
  python scripts/evaluate.py --quick
"""

import os
import sys
import json
import logging
import argparse
import time

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
# 抑制第三方库详细日志
for lib in ["neo4j", "httpx", "httpcore", "urllib3", "pymilvus", "jieba"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

logger = logging.getLogger("evaluate")

# ---------------------------------------------------------------------------
#  Argument parsing
# ---------------------------------------------------------------------------

STRATEGY_CHOICES = ["hybrid", "graph_rag", "combined", "router"]
STAGE_CHOICES = ["retrieval", "routing", "generation", "e2e"]
MODE_CHOICES = ["offline", "online"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Agentic-RAG 评估工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 模式
    parser.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        default="offline",
        help="评估模式 (默认: offline)",
    )

    # 数据源
    parser.add_argument(
        "--csv",
        default=None,
        help="测试集 CSV 路径 (离线模式)",
    )
    parser.add_argument(
        "--session-ids",
        nargs="*",
        default=None,
        help="要评估的会话 ID 列表 (在线模式)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="最大查询数 (默认: 100)",
    )

    # 评估范围
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=STRATEGY_CHOICES + ["all"],
        default=None,
        help="要测试的检索策略 (默认: 全部)",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGE_CHOICES + ["all"],
        default=None,
        help="要评估的阶段 (默认: retrieval routing)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="最大 Top-K (默认: 10)",
    )

    # LLM
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        default=False,
        help="启用 LLM-as-Judge (相关性/忠实度/幻觉评估)",
    )
    parser.add_argument(
        "--paraphrase",
        action="store_true",
        default=False,
        help="用 LLM 改写查询 (防数据泄露)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="随机采样查询数 (默认: 全部)",
    )

    # 输出
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="JSON 报告输出路径",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        default=False,
        help="打印详细样本信息",
    )

    # 快捷方式
    parser.add_argument(
        "--quick",
        action="store_true",
        default=False,
        help="快速基线模式 (仅检索，不使用 LLM Judge)",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    # 解决策略列表
    if args.strategies is None or "all" in (args.strategies or []):
        strategies = STRATEGY_CHOICES
    else:
        strategies = args.strategies

    # 解决阶段列表
    if args.stages is None or "all" in (args.stages or []):
        if args.mode == "online":
            stages = ["generation", "e2e"]
        elif args.quick:
            stages = ["retrieval"]
        else:
            stages = ["retrieval", "routing"]
    else:
        stages = args.stages

    # 快速基线 → 强制 offline + retrieval + no judge
    if args.quick:
        args.mode = "offline"
        args.llm_judge = False

    # ------------------------------------------------------------------
    #  导入 (延迟加载, 避免在 --help 时就触发)
    # ------------------------------------------------------------------
    from app.main import ClinicalDecisionSystem
    from app.rag_modules.evaluation import (
        EvalConfig, EvalMode, EvalStage, EvalReport,
        Evaluator, EvalIntegration,
    )
    from app.rag_modules.evaluation.reporter.console_reporter import ConsoleReporter
    from app.rag_modules.evaluation.reporter.json_reporter import JSONReporter

    # ------------------------------------------------------------------
    #  初始化系统
    # ------------------------------------------------------------------
    logger.info("正在初始化 ClinicalDecisionSystem...")
    system = ClinicalDecisionSystem()
    system.initialize_system()

    # 解析 LLM client (for judges)
    llm_client = None
    if args.llm_judge or args.paraphrase:
        try:
            from openai import OpenAI
            api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            if api_key:
                llm_client = OpenAI(api_key=api_key, base_url=base_url)
                logger.info(f"LLM 客户端已初始化 ({base_url})")
            else:
                logger.warning("未设置 DEEPSEEK_API_KEY, LLM Judge 功能不可用")
        except Exception as e:
            logger.warning(f"LLM 客户端初始化失败: {e}")

    # ------------------------------------------------------------------
    #  构建 EvalConfig
    # ------------------------------------------------------------------
    config = EvalConfig(
        mode=EvalMode.OFFLINE if args.mode == "offline" else EvalMode.ONLINE,
        stages=[EvalStage(s) for s in stages],
        top_k_values=[1, 3, 5, args.top_k],
        strategies=strategies,
        test_csv_path=args.csv,
        llm_judge=args.llm_judge,
        paraphrase=args.paraphrase,
        sample_size=args.sample,
    )
    logger.info(
        f"评估配置: mode={config.mode.value} "
        f"stages={[s.value for s in config.stages]} "
        f"strategies={strategies}"
    )

    # ------------------------------------------------------------------
    #  执行评估
    # ------------------------------------------------------------------
    qa_db = getattr(system, "qa_db", None)

    if args.mode == "online" and qa_db:
        logger.info(f"在线评估: 最多 {args.limit} 条历史记录")
        integration = EvalIntegration(system=system, qa_db=qa_db, llm_client=llm_client)
        report = integration.evaluate_from_qa_db(
            session_ids=args.session_ids,
            limit=args.limit,
        )
    else:
        logger.info(f"离线评估: CSV={config.test_csv_path or '自动发现'}")
        evaluator = Evaluator(
            config=config,
            system=system,
            llm_client=llm_client,
            qa_db=qa_db,
        )
        report = evaluator.evaluate()

    # ------------------------------------------------------------------
    #  输出
    # ------------------------------------------------------------------
    # 终端报告
    ConsoleReporter.print_report(report, detailed=args.detailed)

    # JSON 保存
    if args.output:
        JSONReporter.save(report, args.output)
        logger.info(f"JSON 报告已保存至: {args.output}")

    # 打印摘要行 (方便 grep)
    r = report.retrieval_metrics or {}
    g = report.generation_metrics or {}
    health = report.judge_health or {}
    print(f"\n>>> SUMMARY: "
          f"Recall@5={r.get('Recall@5', 'N/A'):.2%}  "
          f"MRR={r.get('MRR', 'N/A'):.4f}  "
          f"HitRate={r.get('HitRate', 'N/A'):.2%}  "
          f"Faithfulness={g.get('Faithfulness', 'N/A')}  "
          # 与指标同行输出：回退数不为 0 时，这一行的指标就是不可信的
          f"JudgeFallbacks={health.get('total_failures', 0)}/{health.get('total_calls', 0)}  "
          f"TotalQueries={r.get('total_queries', 0)}")


if __name__ == "__main__":
    main()
