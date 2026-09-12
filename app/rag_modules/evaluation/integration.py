"""
评估模块集成适配器
连接评估模块与 ClinicalDecisionSystem、QADatabase 等现有组件
"""

import logging
import os
from typing import List, Optional

from .models import EvalConfig, EvalMode, EvalStage, EvalReport
from .evaluator import Evaluator

logger = logging.getLogger(__name__)


class EvalIntegration:
    """
    评估集成适配器

    工厂方法: create_evaluator()
    快捷入口: evaluate_from_qa_db()
    """

    def __init__(self, system, qa_db=None, llm_client=None):
        """
        Args:
            system: ClinicalDecisionSystem 实例
            qa_db: QADatabase 实例（在线评估时需要）
            llm_client: OpenAI 客户端（可选，默认从 system 获取）
        """
        self.system = system
        self.qa_db = qa_db
        self.llm_client = llm_client or self._resolve_llm_client()

    # ------------------------------------------------------------------
    #  Factory
    # ------------------------------------------------------------------

    def create_evaluator(
        self,
        mode: EvalMode = EvalMode.OFFLINE,
        stages: Optional[List[EvalStage]] = None,
        **kwargs,
    ) -> Evaluator:
        """
        创建并返回一个配置好的 Evaluator

        Args:
            mode: 评估模式
            stages: 要评估的阶段（默认全部）
            **kwargs: 其他 EvalConfig 参数（如 top_k_values, strategies 等）
        """
        config = EvalConfig(
            mode=mode,
            stages=stages or [s for s in EvalStage],
            **kwargs,
        )

        # 如果未指定 test_csv_path，尝试自动查找
        if not config.test_csv_path:
            config.test_csv_path = self._find_test_csv()

        return Evaluator(
            config=config,
            system=self.system,
            llm_client=self.llm_client,
            qa_db=self.qa_db,
        )

    # ------------------------------------------------------------------
    #  Quick entry points
    # ------------------------------------------------------------------

    def evaluate_from_qa_db(
        self,
        session_ids: Optional[List[str]] = None,
        limit: int = 100,
    ) -> EvalReport:
        """快捷入口：从 QADatabase 拉取历史记录进行评估"""
        if not self.qa_db:
            raise ValueError("需要提供 qa_db 实例")

        evaluator = self.create_evaluator(
            mode=EvalMode.ONLINE,
            stages=[EvalStage.GENERATION, EvalStage.END_TO_END],
            sample_size=limit,
        )
        return evaluator.evaluate_from_db(session_ids=session_ids, limit=limit)

    def evaluate_current_session(self) -> EvalReport:
        """快捷入口：评估当前活动会话"""
        if not self.qa_db or not self.system.current_session_id:
            raise ValueError("没有活动会话")

        return self.evaluate_from_db(
            session_ids=[self.system.current_session_id]
        )

    def evaluate_with_csv(
        self,
        csv_path: str,
        strategies: Optional[List[str]] = None,
        top_k: int = 10,
        sample_size: Optional[int] = None,
    ) -> EvalReport:
        """快捷入口：使用 CSV 测试集进行检索评估"""
        evaluator = self.create_evaluator(
            mode=EvalMode.OFFLINE,
            stages=[EvalStage.RETRIEVAL, EvalStage.ROUTING],
            test_csv_path=csv_path,
            strategies=strategies or ["hybrid", "graph_rag", "combined", "router"],
            top_k_values=[1, 3, 5, top_k],
            sample_size=sample_size,
        )
        return evaluator.evaluate()

    # ------------------------------------------------------------------
    #  Internal
    # ------------------------------------------------------------------

    def _resolve_llm_client(self):
        """从 system 中获取或创建 LLM 客户端"""
        # 优先使用 system 自带的 client
        if hasattr(self.system, "llm_client") and self.system.llm_client:
            return self.system.llm_client

        # 否则尝试创建
        try:
            from openai import OpenAI
            import os as _os
            api_key = _os.getenv("DEEPSEEK_API_KEY")
            if api_key:
                return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        except Exception:
            pass

        logger.warning("无法解析 LLM 客户端，Judge 功能将被禁用")
        return None

    @staticmethod
    def _find_test_csv() -> Optional[str]:
        """自动查找可用的测试 CSV"""
        import glob as _glob

        # 检查默认路径
        candidates = [
            "data/test_20.csv",
            "data/processed/*.csv",
        ]

        for pattern in candidates:
            matched = _glob.glob(pattern)
            if matched:
                return matched[0]

        return None
