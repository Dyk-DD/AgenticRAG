"""
完整性 Judge
判断 AI 回答是否完整覆盖了用户问题的所有关键方面
"""

import logging
from typing import List, Tuple

from langchain_core.documents import Document

from ..utils import extract_json_from_llm
from .judge_health import JudgeHealthMixin, empty_content_failure_reason

logger = logging.getLogger(__name__)


class CompletenessJudge(JudgeHealthMixin):
    """
    回答完整性 LLM Judge

    评估回答是否完整覆盖了用户问题的所有方面
    """

    judge_name = "completeness"

    def __init__(self, llm_client, model: str = "deepseek-chat"):
        self.client = llm_client
        self.model = model
        self._init_health()

    def judge(
        self,
        question: str,
        answer: str,
        documents: List[Document],
    ) -> Tuple[float, str]:
        """
        判断回答的完整性

        Returns:
            completeness_score: 0-1 (1 = 完全完整)
            missing_aspects: 遗漏方面的描述
        """
        prompt = f"""你是一个医疗问答质量评估专家。判断以下AI回答是否完整覆盖了用户问题的所有关键方面。

【用户提问】: {question}

【AI回答】: {answer}

请分析：
1. 用户问题中包含了哪些关键方面/子问题？
2. 回答是否覆盖了所有这些方面？
3. 如果存在遗漏，遗漏了什么？

输出JSON格式：
{{
    "completeness_score": 0.85,
    "total_aspects": 3,
    "covered_aspects": 2,
    "missing_aspects": "未提及药物相互作用的详细信息",
    "analysis": "回答覆盖了主要病因和治疗方法，但缺少禁忌症说明"
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
            # content 可能是 None（推理模型只吐思考不吐正文），先兜住再 strip
            raw = (resp.choices[0].message.content or "").strip()
            result = extract_json_from_llm(raw)

            if result is None:
                self._record_failure(
                    empty_content_failure_reason(raw, resp.choices[0].message)
                )
                return 0.5, "解析失败"

            score = float(result.get("completeness_score", 0.5))
            score = max(0.0, min(score, 1.0))
            missing = str(result.get("missing_aspects", ""))

            return score, missing

        except Exception as e:
            logger.error(f"CompletenessJudge 判断失败: {e}")
            self._record_failure(f"异常: {e}")
            return 0.5, str(e)

    def judge_answer_relevance(
        self,
        question: str,
        answer: str,
    ) -> Tuple[float, str]:
        """
        额外方法：判断回答与问题的整体相关性（端到端用）

        Returns:
            relevance_score: 0-1
            analysis: str
        """
        prompt = f"""你是一个医疗问答质量评估专家。判断以下AI回答与用户问题的整体相关性。

【用户提问】: {question}

【AI回答】: {answer}

评分标准：
- 0.0-0.3: 完全不相关或答非所问
- 0.4-0.6: 部分相关，但核心问题未回答
- 0.7-0.9: 回答切题，有帮助
- 1.0: 完美回答

输出JSON格式：
{{
    "relevance_score": 0.85,
    "analysis": "回答直接针对用户的问题提供了详细且有价值的信息"
}}
"""
        try:
            # 与 completeness 分开计数：同一个类产出两项独立判定
            self._record_call("answer_relevance")
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
                    label="answer_relevance",
                )
                return 0.5, ""

            score = float(result.get("relevance_score", 0.5))
            analysis = str(result.get("analysis", ""))
            return max(0.0, min(score, 1.0)), analysis

        except Exception as e:
            logger.error(f"AnswerRelevance 判断失败: {e}")
            self._record_failure(f"异常: {e}", label="answer_relevance")
            return 0.5, str(e)
