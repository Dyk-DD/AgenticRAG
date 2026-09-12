"""
评估模块核心数据模型
定义评估配置、测试查询、评估样本、评估报告等数据结构
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from langchain_core.documents import Document
from datetime import datetime


class EvalMode(Enum):
    """评估模式"""
    OFFLINE = "offline"      # 离线评估：使用测试数据集
    ONLINE = "online"        # 在线评估：从 SQLite 历史记录中拉取


class EvalStage(Enum):
    """评估阶段"""
    RETRIEVAL = "retrieval"       # 检索质量
    ROUTING = "routing"           # 路由质量
    GENERATION = "generation"     # 生成质量
    END_TO_END = "end_to_end"     # 端到端质量


@dataclass
class EvalConfig:
    """评估配置"""
    mode: EvalMode = EvalMode.OFFLINE
    stages: List[EvalStage] = field(
        default_factory=lambda: [s for s in EvalStage]
    )
    test_csv_path: Optional[str] = None
    top_k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    strategies: List[str] = field(
        default_factory=lambda: ["hybrid", "graph_rag", "combined", "router"]
    )
    sample_size: Optional[int] = None
    paraphrase: bool = False               # 是否用 LLM 改写查询
    llm_judge: bool = True                 # 是否启用 LLM 判断
    # 留空表示"跟随全局 LLM_MODEL"（config.py 的 llm_model）。
    # 不要在这里写死模型名：写死会让评测 Judge 无视 LLM_MODEL 而继续用旧模型，
    # 于是生成已经切到新模型、Judge 却还在用推理模型返回空响应，指标全不可信。
    judge_model: Optional[str] = None
    seed: int = 42
    measure_latency: bool = True           # 是否测量延迟
    measure_diversity: bool = True         # 是否测量多样性
    output_dir: str = "data/eval_reports"


@dataclass
class TestQuery:
    """单条测试查询，包含 Ground Truth"""
    query_id: str
    query_text: str
    original_query: Optional[str] = None        # 改写前的原查询
    ground_truth_node_ids: List[str] = field(default_factory=list)
    ground_truth_chunk_ids: List[str] = field(default_factory=list)
    expected_routing_strategy: Optional[str] = None
    expected_answer_keywords: List[str] = field(default_factory=list)
    department: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSample:
    """单次评估样本 = 一条查询走完评估管道的结果"""
    test_query: TestQuery

    # 路由/检索阶段
    strategy_used: str = ""
    is_natural_route: bool = True              # True=路由器自然决策, False=策略强制覆盖
    query_analysis: Optional[Any] = None       # QueryAnalysis from router
    retrieved_docs: List[Document] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0

    # 路由评估
    expected_routing: Optional[str] = None
    actual_routing: Optional[str] = None

    # 生成阶段
    generated_answer: str = ""
    generation_latency_ms: float = 0.0

    # 端到端
    total_latency_ms: float = 0.0

    # LLM Judges 填充的字段
    doc_relevance_flags: List[bool] = field(default_factory=list)   # 每篇文档是否相关
    doc_relevance_scores: List[float] = field(default_factory=list) # 每篇文档的相关性分数
    faithfulness_score: float = 0.0           # 忠实度 (0-1)
    hallucination_score: float = 0.0          # 幻觉率 (0-1, 越低越好)
    completeness_score: float = 0.0           # 完整性 (0-1)
    answer_relevance_score: float = 0.0       # 回答相关性 (0-1)

    # Judge 回退记录：判定项 -> 回退次数。回退 = 没拿到真实裁决，上面的分数
    # 是默认值（0.5 / False），不能当作真实评分参与统计。
    judge_failures: Dict[str, int] = field(default_factory=dict)
    judge_failure_reasons: List[str] = field(default_factory=list)

    # 异常
    error: Optional[str] = None


@dataclass
class EvalReport:
    """完整评估报告"""
    config: EvalConfig
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # 各阶段指标
    retrieval_metrics: Dict[str, Any] = field(default_factory=dict)
    routing_metrics: Dict[str, Any] = field(default_factory=dict)
    generation_metrics: Dict[str, Any] = field(default_factory=dict)
    e2e_metrics: Dict[str, Any] = field(default_factory=dict)

    # 按策略维度的分解
    strategy_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # 按科室维度的分解
    department_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # LLM Judge 健康度（degraded=True 时说明有判定回退到默认值，
    # 受影响的指标不可信 —— 必须与指标一起展示，否则默认值会被当成真实评分）
    judge_health: Dict[str, Any] = field(default_factory=dict)

    # 所有样本详情（用于深度分析）
    samples: List[EvalSample] = field(default_factory=list)

    # 系统信息
    system_info: Dict[str, Any] = field(default_factory=dict)
