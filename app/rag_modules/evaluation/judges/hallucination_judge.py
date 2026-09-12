"""
幻觉检测 Judge
检测 AI 回答中是否包含编造的医学事实、错误药物名、剂量错误等
"""

import logging
from typing import List, Tuple

from langchain_core.documents import Document

from ..utils import extract_json_from_llm
from .judge_health import JudgeHealthMixin, empty_content_failure_reason

logger = logging.getLogger(__name__)


class HallucinationJudge(JudgeHealthMixin):
    """
    幻觉检测 LLM Judge

    检测回答中是否存在：
    - 编造的医学事实
    - 错误的药物名称或剂量
    - 与常识医学知识矛盾的内容
    - 参考资料中完全没有依据的声明
    """

    judge_name = "hallucination"

    def __init__(self, llm_client, model: str = "deepseek-chat"):
        self.client = llm_client
        self.model = model
        self._init_health()

    def judge(
        self,
        question: str,
        answer: str,
        documents: List[Document],
    ) -> Tuple[float, List[str], float]:
        """
        检测回答中的幻觉

        Returns:
            hallucination_score: 0-1 (0 = 无幻觉, 越低越好)
            hallucinated_statements: 幻觉声明列表
            confidence: float
        """
        doc_summaries = []
        for i, doc in enumerate(documents[:5]):
            text = doc.page_content[:400]
            doc_summaries.append(f"[文档{i + 1}]: {text}")
        docs_text = "\n\n".join(doc_summaries)

        prompt = f"""你是一个医学事实核查专家。检测以下AI生成的临床回答中是否存在"幻觉" —— 即编造的医学事实、错误的药物名、剂量错误，或与公认医学知识矛盾的内容。

【用户提问】: {question}

【参考资料】:
{docs_text}

【AI回答】: {answer}

请：
1. 识别回答中所有可能的幻觉声明
2. 对每个幻觉声明，说明它与参考资料或医学常识的冲突
3. 计算整体幻觉分数（0 = 完全没有幻觉，1 = 严重幻觉）

输出JSON格式：
{{
    "hallucination_score": 0.1,
    "total_claims_checked": 5,
    "hallucinated_statements": [
        "声明1: 描述...（错误原因）",
        "声明2: 描述...（错误原因）"
    ],
    "confidence": 0.9,
    "analysis": "大部分内容有依据，但有少量不准确表述"
}}
"""
        try:
            self._record_call()
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            raw = (resp.choices[0].message.content or "").strip()
            result = extract_json_from_llm(raw)

            if result is None:
                self._record_failure(
                    empty_content_failure_reason(raw, resp.choices[0].message)
                )
                return 0.5, [], 0.0

            score = float(result.get("hallucination_score", 0.5))
            score = max(0.0, min(score, 1.0))
            statements = result.get("hallucinated_statements", [])
            confidence = float(result.get("confidence", 0.5))

            return score, statements, confidence

        except Exception as e:
            logger.error(f"HallucinationJudge 判断失败: {e}")
            self._record_failure(f"异常: {e}")
            return 0.5, [], 0.0
