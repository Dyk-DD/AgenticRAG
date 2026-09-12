"""
评估主编排器
协调数据集、检索管道、LLM Judges、指标计算器完成完整评估流程
"""

import os
import sys
import logging
import time
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

from ..intelligent_query_router import SearchStrategy
from .models import (
    EvalConfig,
    EvalMode,
    EvalStage,
    EvalReport,
    EvalSample,
    TestQuery,
)
from .dataset.csv_dataset import CSVTestDataset
from .dataset.online_dataset import OnlineDataset
from .metrics.retrieval_metrics import compute_retrieval_metrics
from .judges.judge_health import collect_judge_health

logger = logging.getLogger(__name__)


class Evaluator:
    """
    评估主编排器

    数据流:
    1. 加载测试数据集 (离线) 或 拉取历史记录 (在线)
    2. 对每条查询:
       a. route_query() → (docs, analysis)
       b. (可选) generate_adaptive_answer() → answer
       c. LLM Judges 评判
    3. 计算汇总指标
    4. 生成报告
    """

    def __init__(
        self,
        config: EvalConfig,
        system: Any,  # ClinicalDecisionSystem
        llm_client: Optional[Any] = None,
        qa_db: Optional[Any] = None,
    ):
        self.config = config
        self.system = system
        self.llm_client = llm_client
        self.qa_db = qa_db

        # 内部状态
        self._judges: Dict[str, Any] = {}
        self._samples: List[EvalSample] = []

    # ------------------------------------------------------------------
    #  主入口
    # ------------------------------------------------------------------

    def evaluate(self) -> EvalReport:
        """
        执行完整评估流程（离线模式）

        对每条查询运行所有配置的策略:
        - "router" 策略: 使用路由器的自然决策（每条查询一次）
        - 其他策略: 强制所有查询通过该策略
        """
        logger.info("=" * 50)
        logger.info(f"开始评估 | 模式: {self.config.mode.value}")
        logger.info(f"阶段: {[s.value for s in self.config.stages]}")
        logger.info(f"策略: {self.config.strategies}")
        logger.info("=" * 50)

        # 同一个 Evaluator 实例可能被跑多轮，健康度计数必须每轮归零，
        # 否则第二轮会把第一轮的回退数一起报出来
        self._reset_judge_health()

        # 1. 加载测试集
        queries = self._load_dataset()
        logger.info(f"测试查询集: {len(queries)} 条")

        # 2. 确定要运行的策略列表
        strategies_to_run = list(self.config.strategies or [])
        has_router = "router" in strategies_to_run
        if has_router:
            strategies_to_run.remove("router")

        # 如果没有明确指定策略，默认用 router
        if not strategies_to_run and not has_router:
            has_router = True

        # 3. 逐策略 × 逐条评估
        self._samples = []
        total_runs = len(queries) * (len(strategies_to_run) + (1 if has_router else 0))
        run_count = 0

        for i, q in enumerate(queries):
            # 进度输出（即使 logger 被静音也能看到）
            sys.stdout.write(f"\r  ⟳ 评估进度: [{i+1}/{len(queries)}] {q.query_text[:50]:<50}")
            sys.stdout.flush()
            # 3a. Router 自然策略（每条查询一次）
            if has_router:
                run_count += 1
                logger.info(
                    f"评估 [{run_count}/{total_runs}] router: {q.query_text[:30]}..."
                )
                sample = self._evaluate_single(q)
                self._samples.append(sample)

            # 3b. 强制指定策略
            for strategy in strategies_to_run:
                run_count += 1
                logger.info(
                    f"评估 [{run_count}/{total_runs}] {strategy}: {q.query_text[:30]}..."
                )
                sample = self._evaluate_single(q, strategy_override=strategy)
                self._samples.append(sample)

        # 4. 计算指标
        report = self._compute_report(self._samples)
        print()  # 进度行换行
        logger.info("评估完成！")
        return report

    def evaluate_retrieval_only(self) -> Dict[str, Any]:
        """仅评估检索（不生成回答）"""
        self.config.stages = [EvalStage.RETRIEVAL]
        if EvalStage.ROUTING in self._available_stages():
            self.config.stages.append(EvalStage.ROUTING)
        report = self.evaluate()
        return {
            **report.retrieval_metrics,
            "routing": report.routing_metrics,
            "strategy_breakdown": report.strategy_breakdown,
            "department_breakdown": report.department_breakdown,
            # 这条路径也会跑 relevance judge，别把健康度漏掉
            "judge_health": report.judge_health,
        }

    def evaluate_from_db(
        self,
        session_ids: Optional[List[str]] = None,
        limit: int = 100,
    ) -> EvalReport:
        """
        在线评估：从 QADatabase 拉取历史记录

        不重新检索，只运行 Judges
        """
        if not self.qa_db:
            raise ValueError("在线评估需要提供 qa_db 实例")

        logger.info(f"在线评估: 从 QADatabase 拉取 (limit={limit})")
        self._reset_judge_health()
        ds = OnlineDataset(self.qa_db)
        queries = ds.load_from_db(session_ids=session_ids, limit=limit)
        logger.info(f"加载了 {len(queries)} 条在线查询")

        self._samples = []
        for q in queries:
            sample = self._build_sample_from_history(q)
            if sample:
                self._samples.append(sample)

        report = self._compute_report(self._samples)
        logger.info("在线评估完成！")
        return report

    # ------------------------------------------------------------------
    #  内部: 数据集加载
    # ------------------------------------------------------------------

    def _load_dataset(self) -> List[TestQuery]:
        """加载测试数据集"""
        if self.config.mode == EvalMode.OFFLINE:
            csv_path = self.config.test_csv_path
            if csv_path and csv_path.endswith(".csv"):
                from os.path import exists
                if not exists(csv_path):
                    raise FileNotFoundError(f"测试CSV不存在: {csv_path}")

            ds = CSVTestDataset(
                csv_path=csv_path or "",
                llm_client=self.llm_client,
            )
            queries = ds.load(
                sample_size=self.config.sample_size,
                seed=self.config.seed,
            )

            # 改写（可选）
            if self.config.paraphrase:
                queries = ds.paraphrase_queries(queries, model=self._judge_model())

            # 推断路由 Ground Truth
            queries = CSVTestDataset.infer_expected_routing(queries)

            return queries

        elif self.config.mode == EvalMode.ONLINE:
            ds = OnlineDataset(self.qa_db)
            return ds.load_from_db(limit=self.config.sample_size or 100)

        return []

    # ------------------------------------------------------------------
    #  内部: 单条评估
    # ------------------------------------------------------------------

    def _evaluate_single(
        self,
        test_query: TestQuery,
        strategy_override: Optional[str] = None,
    ) -> EvalSample:
        """评估一条查询的完整管道

        Args:
            test_query: 测试查询
            strategy_override: 指定策略名称（如 "hybrid_traditional", "graph_rag", "combined"）,
                               为 None 时使用路由器的自然决策
        """
        sample = EvalSample(test_query=test_query)

        try:
            # Step 1: 路由 + 检索
            t0 = time.perf_counter()

            if strategy_override:
                docs, analysis = self._search_with_strategy(
                    test_query.query_text, strategy_override
                )
                sample.strategy_used = strategy_override
                sample.actual_routing = strategy_override
                sample.is_natural_route = False
            else:
                docs, analysis = self.system.router.route_query(
                    test_query.query_text,
                    top_k=max(self.config.top_k_values),
                )
                sample.strategy_used = (
                    analysis.recommended_strategy.value if analysis else "unknown"
                )
                sample.actual_routing = sample.strategy_used

            t1 = time.perf_counter()

            # 记录检索延迟和检索到的文档
            sample.retrieval_latency_ms = (t1 - t0) * 1000
            sample.retrieved_docs = docs

            # Step 2: 生成回答
            if (
                EvalStage.GENERATION in self.config.stages
                or EvalStage.END_TO_END in self.config.stages
            ):
                t2 = time.perf_counter()
                try:
                    answer = self.system.generation_module.generate_adaptive_answer(
                        test_query.query_text, docs
                    )
                    sample.generated_answer = str(answer)
                except Exception as gen_err:
                    logger.warning(f"生成失败: {gen_err}")
                    sample.generated_answer = ""
                t3 = time.perf_counter()
                sample.generation_latency_ms = (t3 - t2) * 1000

            # 端到端延迟
            sample.total_latency_ms = sample.retrieval_latency_ms + sample.generation_latency_ms

            # Step 3: 自动精确匹配 ("self" 模式)
            # 用 ground_truth_node_ids/chunk_ids 精确匹配 doc.metadata
            # 精确匹配 ground_truth_node_ids/chunk_ids
            sample.doc_relevance_flags = self._match_ground_truth(docs, test_query)

            # Step 4: LLM Judgments (覆盖/补充精确匹配)
            if self.config.llm_judge:
                self._run_judgments(sample)

        except Exception as e:
            logger.error(f"评估异常 [{test_query.query_text[:20]}...]: {e}")
            sample.error = str(e)

        return sample

    def _build_sample_from_history(self, test_query: TestQuery) -> Optional[EvalSample]:
        """
        从历史记录构建评估样本（在线模式）
        忽略检索步骤，直接对存储的问答进行评判
        """
        stored_answer = test_query.metadata.get("stored_answer", "")
        if not stored_answer:
            return None

        sample = EvalSample(
            test_query=test_query,
            strategy_used=test_query.metadata.get("strategy", "unknown"),
            generated_answer=stored_answer,
        )

        if self.config.llm_judge:
            self._run_judgments(sample, has_docs=False)

        return sample

    # ------------------------------------------------------------------
    #  内部: LLM Judgments
    # ------------------------------------------------------------------

    @staticmethod
    def _judge_call(sample: EvalSample, judge, method: str, label: str, *args):
        """调用 Judge 并把本次调用产生的回退记到当前样本

        不改各 Judge 的返回签名，只把"没拿到真实裁决"这件事从静默变成可见：
        回退时 Judge 返回的是默认值（0.5 / False），若不留痕，报告会把默认值
        当成真实评分展示。
        """
        before = judge.failure_count(label)
        result = getattr(judge, method)(*args)
        newly = judge.failure_count(label) - before
        if newly:
            sample.judge_failures[label] = sample.judge_failures.get(label, 0) + newly
            reason = judge.last_failure_reason(label)
            if reason:
                sample.judge_failure_reasons.append(f"{label}: {reason}")
        return result

    def _run_judgments(self, sample: EvalSample, has_docs: bool = True):
        """对样本运行所有启用的 LLM Judges"""
        query = sample.test_query.query_text
        docs = sample.retrieved_docs if has_docs else []
        answer = sample.generated_answer

        # 1. 文档相关性 (覆盖精确匹配结果)
        if EvalStage.RETRIEVAL in self.config.stages and has_docs and docs:
            judge = self._get_judge("relevance")
            llm_flags = []
            for doc in docs:
                is_rel, conf, reason = self._judge_call(
                    sample, judge, "judge", "relevance", query, doc
                )
                llm_flags.append(is_rel)
                sample.doc_relevance_scores.append(conf)
                doc.metadata["eval_relevant"] = is_rel
                doc.metadata["eval_relevance_reason"] = reason
            # 有回退就保留精确匹配结果，不用 LLM 结果覆盖。
            # 回退返回 False，一旦覆盖，Recall@k / Precision@k / MRR / HitRate /
            # NDCG@k 会被集体拉成 0 —— 连检索侧的指标都会变成编造的数字。
            if sample.judge_failures.get("relevance"):
                logger.warning(
                    f"相关性 Judge 回退 {sample.judge_failures['relevance']} 次，"
                    f"保留 ground-truth 精确匹配结果（不用默认值覆盖指标）"
                )
            else:
                sample.doc_relevance_flags = llm_flags

        # 2-4. 需要生成回答
        if not answer:
            return

        if EvalStage.GENERATION in self.config.stages or EvalStage.END_TO_END in self.config.stages:
            # 忠实度
            if has_docs:
                fj = self._get_judge("faithfulness")
                score, _, _ = self._judge_call(
                    sample, fj, "judge", "faithfulness", query, answer, docs
                )
                sample.faithfulness_score = score

                # 幻觉
                hj = self._get_judge("hallucination")
                score, _, _ = self._judge_call(
                    sample, hj, "judge", "hallucination", query, answer, docs
                )
                sample.hallucination_score = score
            else:
                # 没有检索文档就没跑这两个判定，字段会停在默认值 0.0。若不标记，
                # 在线评估会把"没跑"当成真实的 0 分平均进去 —— 同样是编造指标。
                for label in ("faithfulness", "hallucination"):
                    sample.judge_failures[label] = sample.judge_failures.get(label, 0) + 1
                    sample.judge_failure_reasons.append(
                        f"{label}: 未运行（无检索文档，在线评估路径）"
                    )

            # 完整性
            cj = self._get_judge("completeness")
            score, _ = self._judge_call(
                sample, cj, "judge", "completeness", query, answer, docs
            )
            sample.completeness_score = score

            # 回答相关性
            score, _ = self._judge_call(
                sample, cj, "judge_answer_relevance", "answer_relevance", query, answer
            )
            sample.answer_relevance_score = score

    def _search_with_strategy(
        self,
        query_text: str,
        strategy: str,
    ) -> Tuple[List[Any], Any]:
        """
        使用指定策略执行检索（不通过路由器）

        使用指定策略执行检索（不通过路由器）:
        - hybrid_traditional → traditional_retrieval.hybrid_search()
        - graph_rag         → graph_rag_retrieval.graph_rag_search()
        - combined          → router._combined_search()
        - 其他              → route_query()
        """
        top_k = max(self.config.top_k_values)
        router = self.system.router

        if strategy in ("hybrid_traditional", "hybrid"):
            docs = self.system.traditional_retrieval.hybrid_search(query_text, top_k)
            return docs, None

        if strategy == "graph_rag":
            docs = self.system.graph_rag_retrieval.graph_rag_search(query_text, top_k)
            return docs, None

        if strategy == "combined":
            docs = router._combined_search(query_text, top_k)
            return docs, None

        # Fallback: 让路由器处理
        return router.route_query(query_text, top_k=top_k)

    @staticmethod
    def _match_ground_truth(
        docs: List[Any],
        test_query: TestQuery,
    ) -> List[bool]:
        """
        精确匹配：检查每篇 doc 的 node_id/chunk_id 是否在 ground truth 中
        精确匹配：检查每篇 doc 的 node_id/chunk_id 是否在 ground truth 中
        """
        gt_node_ids = set(test_query.ground_truth_node_ids or [])
        gt_chunk_ids = set(test_query.ground_truth_chunk_ids or [])
        flags = []
        for doc in docs:
            meta = doc.metadata if hasattr(doc, "metadata") else {}
            doc_node = meta.get("node_id", "") or meta.get("chunk_id", "")
            doc_chunk = meta.get("chunk_id", "")
            is_rel = doc_node in gt_node_ids or doc_chunk in gt_chunk_ids
            flags.append(is_rel)
            # 写回 metadata 供后续 LLM Judge 参考
            meta["eval_relevant_by_id"] = is_rel
        return flags

    def _reset_judge_health(self) -> None:
        """把已加载 Judge 的回退计数归零（每轮评估开始调用）"""
        for judge in self._judges.values():
            reset = getattr(judge, "reset_health", None)
            if callable(reset):
                reset()

    def _judge_model(self) -> str:
        """Judge 使用的模型名

        优先 EvalConfig.judge_model；留空则跟随全局 LLM_MODEL。这样 `.env` 里
        改一次模型，生成与评测判定会一起切换 —— 否则会出现"生成已经换了模型、
        Judge 还在用旧的推理模型返回空响应"的错配，指标全变成默认值。
        """
        if self.config.judge_model:
            return self.config.judge_model
        system_cfg = getattr(self.system, "config", None)
        return (
            getattr(system_cfg, "llm_model", None)
            or os.getenv("LLM_MODEL")
            or "deepseek-chat"
        )

    def _get_judge(self, name: str):
        """懒加载 Judge 实例"""
        if name not in self._judges:
            model = self._judge_model()
            if name == "relevance":
                from .judges.relevance_judge import RelevanceJudge
                self._judges[name] = RelevanceJudge(self.llm_client, model)
            elif name == "faithfulness":
                from .judges.faithfulness_judge import FaithfulnessJudge
                self._judges[name] = FaithfulnessJudge(self.llm_client, model)
            elif name == "hallucination":
                from .judges.hallucination_judge import HallucinationJudge
                self._judges[name] = HallucinationJudge(self.llm_client, model)
            elif name == "completeness":
                from .judges.completeness_judge import CompletenessJudge
                self._judges[name] = CompletenessJudge(self.llm_client, model)
        return self._judges[name]

    # ------------------------------------------------------------------
    #  内部: 汇总指标
    # ------------------------------------------------------------------

    def _compute_report(self, samples: List[EvalSample]) -> EvalReport:
        """聚合所有样本计算汇总指标"""
        report = EvalReport(config=self.config)

        # 检索指标
        if EvalStage.RETRIEVAL in self.config.stages:
            from .metrics.retrieval_metrics import compute_retrieval_metrics
            report.retrieval_metrics = compute_retrieval_metrics(
                samples, self.config.top_k_values
            )

            if self.config.measure_diversity:
                from .metrics.diversity_metrics import compute_diversity_metrics
                report.retrieval_metrics.update(
                    compute_diversity_metrics(samples, top_k=max(self.config.top_k_values))
                )

        # 路由指标
        if EvalStage.ROUTING in self.config.stages:
            from .metrics.routing_metrics import compute_routing_metrics
            report.routing_metrics = compute_routing_metrics(samples)

        # 生成指标
        if EvalStage.GENERATION in self.config.stages:
            from .metrics.generation_metrics import compute_generation_metrics
            report.generation_metrics = compute_generation_metrics(samples)

        # 端到端指标
        if EvalStage.END_TO_END in self.config.stages:
            from .metrics.e2e_metrics import compute_e2e_metrics
            report.e2e_metrics = compute_e2e_metrics(samples)

        # 按策略分解
        report.strategy_breakdown = self._compute_strategy_breakdown(samples)

        # 系统信息
        report.system_info = self._collect_system_info()

        # Judge 健康度：回退次数必须与指标一起落盘、一起展示，
        # 否则那些默认值（0.5 / False）会被当成真实评分念出去
        report.judge_health = collect_judge_health(self._judges)

        report.samples = samples
        return report

    # ------------------------------------------------------------------
    #  内部: 辅助方法
    # ------------------------------------------------------------------

    def _compute_strategy_breakdown(
        self,
        samples: List[EvalSample],
    ) -> Dict[str, Dict[str, Any]]:
        """按检索策略分解指标，每个策略计算完整检索指标"""
        by_strategy = defaultdict(list)
        for s in samples:
            by_strategy[s.strategy_used].append(s)

        breakdown = {}
        for strategy, strat_samples in by_strategy.items():
            n = len(strat_samples)
            hits = sum(
                1 for s in strat_samples if True in s.doc_relevance_flags
            )
            # 完整检索指标
            ret_metrics = compute_retrieval_metrics(
                strat_samples, self.config.top_k_values
            )
            breakdown[strategy] = {
                "count": n,
                "hit_rate": round(hits / n, 4) if n > 0 else 0,
                "avg_latency_ms": round(
                    sum(s.retrieval_latency_ms for s in strat_samples) / n, 2
                )
                if n > 0
                else 0,
                "retrieval_metrics": ret_metrics,
            }
        return breakdown

    def _collect_system_info(self) -> Dict[str, Any]:
        """收集系统元信息"""
        info = {}
        try:
            if hasattr(self.system, "config") and self.system.config:
                # 必须脱敏：system_info 会被 JSON reporter 写盘，
                # 不脱敏则 neo4j_password 随每份评估报告一起外泄
                info["config"] = self.system.config.to_dict(redact_secrets=True)
        except Exception:
            pass

        try:
            if hasattr(self.system, "router") and self.system.router:
                info["route_stats"] = self.system.router.get_route_statistics()
        except Exception:
            pass

        try:
            if hasattr(self.system, "index_module") and self.system.index_module:
                info["milvus_stats"] = self.system.index_module.get_collection_stats()
        except Exception:
            pass

        try:
            if hasattr(self.system, "data_module") and self.system.data_module:
                info["kb_stats"] = self.system.data_module.get_statistics()
        except Exception:
            pass

        return info

    def _available_stages(self) -> List[EvalStage]:
        """检测当前可用的评估阶段"""
        stages = [EvalStage.RETRIEVAL, EvalStage.ROUTING]
        if hasattr(self.system, "generation_module"):
            stages.append(EvalStage.GENERATION)
            stages.append(EvalStage.END_TO_END)
        return stages
