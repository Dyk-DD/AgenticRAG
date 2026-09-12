"""
忠实度 Judge
判断 AI 回答是否忠实地基于检索到的文档，没有编造事实
"""

import logging
from typing import List, Tuple

from langchain_core.documents import Document

from ..utils import extract_json_from_llm
from .judge_health import JudgeHealthMixin, empty_content_failure_reason

logger = logging.getLogger(__name__)


class FaithfulnessJudge(JudgeHealthMixin):
    """
    回答忠实度 LLM Judge

    将回答分解为独立声明，然后逐条验证是否被文档支持
    faithfulness_score = supported_claims / total_claims
    """

    judge_name = "faithfulness"

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
        判断回答的忠实度

        Returns:
            faithfulness_score: 0-1 (1 = 完全忠实)
            unsupported_claims: 不被文档支持的声明列表
            confidence: float
        """
        # 构建文档摘要（避免 token 过长）
        doc_summaries = []
        for i, doc in enumerate(documents[:5]):  # 最多 5 篇
            text = doc.page_content[:400]
            doc_summaries.append(f"[文档{i + 1}]: {text}")
        docs_text = "\n\n".join(doc_summaries)

        prompt = f"""你是一个医学事实验证专家。判断以下AI生成的回答是否忠实地基于提供的参考资料，没有编造或歪曲事实。

【用户提问】: {question}

【参考资料】:
{docs_text}

【AI回答】: {answer}

请：
1. 将AI回答分解为独立的事实性声明（每个断言为一句话）
2. 逐一检查每个声明是否在参考资料中有明确依据
3. 输出JSON格式：

{{
    "total_claims": 5,
    "supported_claims": 4,
    "unsupported_claims": ["声明1: 描述...", "声明2: 描述..."],
    "faithfulness_score": 0.8,
    "confidence": 0.9,
    "reason": "大部分声明有依据，但有一条提及的数据未在参考中找到"
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

            total = int(result.get("total_claims", 1))
            supported = int(result.get("supported_claims", 0))
            score = supported / max(total, 1)
            unsupported = result.get("unsupported_claims", [])
            confidence = float(result.get("confidence", 0.5))

            return min(score, 1.0), unsupported, confidence

        except Exception as e:
            logger.error(f"FaithfulnessJudge 判断失败: {e}")
            self._record_failure(f"异常: {e}")
            return 0.5, [], 0.0
