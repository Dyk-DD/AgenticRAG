"""
Agentic-RAG 评估模块
提供检索质量、路由质量、生成质量和端到端质量的全面评估

使用方式:
    from rag_modules.evaluation import EvalConfig, Evaluator, EvalIntegration

    # 离线评估
    evaluator = EvalIntegration(system).create_evaluator()
    report = evaluator.evaluate()

    # 在线评估（从 SQLite 历史记录）
    report = EvalIntegration(system, qa_db).evaluate_from_qa_db()
"""

from .models import EvalConfig, EvalMode, EvalStage, EvalReport, TestQuery, EvalSample
from .evaluator import Evaluator
from .integration import EvalIntegration

__all__ = [
    "EvalConfig",
    "EvalMode",
    "EvalStage",
    "EvalReport",
    "TestQuery",
    "EvalSample",
    "Evaluator",
    "EvalIntegration",
]
