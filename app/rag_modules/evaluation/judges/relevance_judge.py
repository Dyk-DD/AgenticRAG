"""
相关性 Judge
判断检索到的文档是否与用户查询相关
使用 LLM 逐篇判断文档是否与查询相关
"""

import logging
from typing import List, Tuple, Optional

from langchain_core.documents import Document

from ..utils import extract_json_from_llm
from .judge_health import JudgeHealthMixin, empty_content_failure_reason

logger = logging.getLogger(__name__)


class RelevanceJudge(JudgeHealthMixin):
    """文档相关性 LLM Judge

    注意本 Judge 的回退返回值是 False，危害比其它三个更大：evaluator 会用这批
    结果覆盖 ground-truth 精确匹配标志，一个坏掉的 Judge 会把 Recall@k /
    Precision@k / MRR / HitRate / NDCG@k 集体变成编造的 0。所以回退必须
    被记录下来（见 JudgeHealthMixin），并由 evaluator 侧决定不回退时才覆盖。
    """

    judge_name = "relevance"

    def __init__(self, llm_client, model: str = "deepseek-chat"):
        self.client = llm_client
        self.model = model
        self._init_health()
        # 保留旧字段以免外部引用断掉；total 原先只在成功时累加，本身就是错的，
        # 新代码请用 failure_count("relevance") / health_snapshot()。
        self._stats = {"total": 0, "relevant": 0}

    def judge(
        self,
        query: str,
        doc: Document,
        context: Optional[str] = None,
    ) -> Tuple[bool, float, str]:
        """
        判断单篇文档是否与查询相关

        Returns:
            is_relevant: bool
            confidence: float (0-1)
            reason: str
        """
        doc_text = doc.page_content[:600]

        prompt = f"""你是一个医疗检索评估专家。判断以下检索到的文档是否与用户的查询相关。

【用户查询】: {query}

【检索到的文档】: {doc_text}

请逐步分析：
1. 文档内容是否直接回答了查询中的问题？
2. 文档是否包含与查询实体相关的医学知识？
3. 文档是否提供了有助于回答用户问题的上下文信息？

请严格输出JSON格式（不要包含多余的文字）：
{{
    "is_relevant": true,
    "confidence": 0.95,
    "reason": "文档包含与查询相关的医学实体和症状描述"
}}
"""
        try:
            self._record_call("relevance")
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            raw = (resp.choices[0].message.content or "").strip()
            result = extract_json_from_llm(raw)

            if result is None:
                self._record_failure(
                    empty_content_failure_reason(raw, resp.choices[0].message),
                    label="relevance",
                )
                return False, 0.0, "解析失败"

            is_relevant = bool(result.get("is_relevant", False))
            confidence = float(result.get("confidence", 0.5))
            reason = str(result.get("reason", ""))

            self._stats["total"] += 1
            if is_relevant:
                self._stats["relevant"] += 1

            return is_relevant, confidence, reason

        except Exception as e:
            logger.error(f"RelevanceJudge 判断失败: {e}")
            self._record_failure(f"异常: {e}", label="relevance")
            return False, 0.0, str(e)

    def judge_batch(
        self,
        query: str,
        docs: List[Document],
    ) -> List[Tuple[bool, float, str]]:
        """批量判断多篇文档的相关性"""
        return [self.judge(query, doc) for doc in docs]

    def get_stats(self) -> dict:
        return {**self._stats}
