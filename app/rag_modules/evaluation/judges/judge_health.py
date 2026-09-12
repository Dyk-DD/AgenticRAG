"""
Judge 健康度统计

背景
----
4 个 LLM Judge 在"没拿到可用 JSON"时都会静默返回默认值（0.5 / False）。
不记录的话，评估报告会把默认值当成真实评分展示 —— 报告看起来合理，数字
却是编造的。对一个把评估模块当作卖点的项目，这是最坏的失败形态：它不会
报错，只会让你在面试里念出假的指标。

更隐蔽的一处：RelevanceJudge 回退返回 False，而 evaluator 会用这批结果
**覆盖** ground-truth 精确匹配标志，于是 Recall@k / Precision@k / MRR /
HitRate / NDCG@k 会被集体拉成 0 —— 连检索侧的指标都是假的。

设计
----
JudgeHealthMixin   每个 Judge 记录 调用次数 / 回退次数 / 回退原因
collect_judge_health  把各 Judge 聚合成报告级字段

故意不改各 Judge 的返回签名（仍是 Tuple[float, ...]），只加旁路计数，
这样 evaluator 的解包逻辑与既有调用方都不受影响。
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class JudgeHealthMixin:
    """为 Judge 记录调用与回退（回退 = 没拿到真实裁决）

    子类用法::

        class XxxJudge(JudgeHealthMixin):
            judge_name = "xxx"

            def __init__(self, llm_client, model):
                ...
                self._init_health()

            def judge(self, ...):
                try:
                    self._record_call()
                    ...
                    if result is None:
                        self._record_failure("JSON 解析失败")
                        return 0.5, "解析失败"     # 仍然不抛异常
                except Exception as e:
                    self._record_failure(f"异常: {e}")
                    return 0.5, str(e)

    注意 label 是"判定项"而不是"judge 对象"：CompletenessJudge 一个类要产出
    completeness 与 answer_relevance 两项独立判定，两者的回退必须分开计数。
    """

    judge_name: str = "judge"

    def _init_health(self) -> None:
        # label -> {"calls": int, "failures": int, "reasons": List[str]}
        self._health: Dict[str, Dict[str, Any]] = {}

    def _bucket(self, label: Optional[str] = None) -> Dict[str, Any]:
        return self._health.setdefault(
            label or self.judge_name, {"calls": 0, "failures": 0, "reasons": []}
        )

    def _record_call(self, label: Optional[str] = None) -> None:
        """记录一次 Judge 调用（成功或回退都算）"""
        self._bucket(label)["calls"] += 1

    def _record_failure(self, reason: str, label: Optional[str] = None) -> None:
        """记录一次回退。调用方随后照旧返回默认值，不抛异常。"""
        bucket = self._bucket(label or self.judge_name)
        bucket["failures"] += 1
        bucket["reasons"].append(reason)
        logger.warning(
            f"[{label or self.judge_name}] 回退到默认值: {reason} "
            f"(累计 {bucket['failures']}/{bucket['calls']})"
        )

    def failure_count(self, label: Optional[str] = None) -> int:
        return self._health.get(label or self.judge_name, {}).get("failures", 0)

    def last_failure_reason(self, label: Optional[str] = None) -> str:
        reasons = self._health.get(label or self.judge_name, {}).get("reasons", [])
        return reasons[-1] if reasons else ""

    def health_snapshot(self) -> Dict[str, Any]:
        labels = {
            label: {
                "calls": bucket["calls"],
                "failures": bucket["failures"],
                "failure_rate": (
                    round(bucket["failures"] / bucket["calls"], 4)
                    if bucket["calls"] else 0.0
                ),
                "sample_reasons": bucket["reasons"][:3],
            }
            for label, bucket in self._health.items()
        }
        return {
            "judge": self.judge_name,
            "labels": labels,
            "total_calls": sum(b["calls"] for b in self._health.values()),
            "total_failures": sum(b["failures"] for b in self._health.values()),
        }

    def reset_health(self) -> None:
        self._init_health()


def collect_judge_health(judges: Dict[str, Any]) -> Dict[str, Any]:
    """把 evaluator 持有的各 Judge 聚合成报告级健康度

    Returns:
        {
            "degraded": bool,          # 有任何回退即为 True
            "summary": "3/40 次 Judge 调用回退到默认值（指标不可信）",
            "total_calls": int,
            "total_failures": int,
            "failure_rate": float,
            "failed_labels": {"completeness": 3, ...},   # 只列有回退的判定项
            "per_judge": {name: health_snapshot(), ...},
        }
    """
    per_judge: Dict[str, Any] = {}
    failed_labels: Dict[str, int] = {}
    total_calls = 0
    total_failures = 0

    for name, judge in (judges or {}).items():
        # 测试里可能塞进不带 mixin 的假 Judge，别让它把整轮评估打挂
        if not isinstance(judge, JudgeHealthMixin) or not hasattr(judge, "_health"):
            continue
        snapshot = judge.health_snapshot()
        per_judge[name] = snapshot
        total_calls += snapshot["total_calls"]
        total_failures += snapshot["total_failures"]
        for label, stat in snapshot["labels"].items():
            if stat["failures"]:
                failed_labels[label] = failed_labels.get(label, 0) + stat["failures"]

    if total_calls == 0:
        return {
            "degraded": False,
            "summary": "未运行 LLM Judge",
            "total_calls": 0,
            "total_failures": 0,
            "failure_rate": 0.0,
            "failed_labels": {},
            "per_judge": per_judge,
        }

    degraded = total_failures > 0
    summary = f"{total_failures}/{total_calls} 次 Judge 调用回退到默认值（指标不可信）"
    if degraded:
        logger.warning(f"⚠️ Judge 健康度告警: {summary} 明细={failed_labels}")

    return {
        "degraded": degraded,
        "summary": summary if degraded else "所有 Judge 调用均返回真实裁决",
        "total_calls": total_calls,
        "total_failures": total_failures,
        "failure_rate": round(total_failures / total_calls, 4),
        "failed_labels": failed_labels,
        "per_judge": per_judge,
    }


def empty_content_failure_reason(raw: str, message: Any) -> str:
    """区分"空响应"与"JSON 解析失败"，并点明推理模型这一成因

    空响应是推理模型最典型的症状（思考吃光 max_tokens，正文为空），
    直接说出来能省掉下一次排查的全部时间。
    """
    if raw:
        return "JSON 解析失败"
    if getattr(message, "reasoning_content", None):
        return "空响应（疑似推理模型耗尽 max_tokens，见 config.py 的 llm_model 注释）"
    return "空响应"
