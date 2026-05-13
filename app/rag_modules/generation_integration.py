"""
临床决策辅助系统 - 生成集成模块
负责根据检索到的医学图谱和病例数据生成最终的临床辅助回答
"""

import logging
import os
import time
from typing import List

from openai import OpenAI
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class GenerationIntegrationModule:
    """生成集成模块 - 负责答案生成"""

    def __init__(self, model_name: str = "deepseek-chat", temperature: float = 0.1, max_tokens: int = 2048):
        """
        初始化生成集成模块
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # 初始化OpenAI客户端
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

        logger.info(f"临床生成模块初始化完成，模型: {model_name}")

    def _build_medical_prompt(self, question: str, context: str, memory_context: str ="") -> str:
        """构建医疗专属的高级 Prompt，加入严格的身份隔离墙和对话记忆"""
        memory_section = ""
        if memory_context:
            memory_section = f"""
【当前会话临床对话历史与记忆】（这是您与当前用户在同一会话中的历史对话，可用于理解指代关系和追踪诊断推理链）：
{memory_context}
"""

        return f"""
作为专业的临床决策辅助系统和权威医学专家，请基于以下提供的【历史医学病例库与临床图谱记录】来解答当前用户的真实咨询。
{memory_section}
【检索到的历史临床参考信息】（⚠️注意：这是知识库中其他患者的历史记录，绝非当前用户的情况）：
{context}

【当前用户的真实咨询问题】：{question}

【核心执行原则（非常重要）】：
1. 严格区分患者身份：检索到的参考信息中包含了其他历史患者的具体【病情描述】（如提到的亲属、具体年龄、过往病史、私人生活情况等）。**你绝对不能将这些历史患者的私人隐私和生活细节当做当前提问用户的情况写入回答中！**
2. 对话记忆参考：如果提供了【当前会话临床对话历史】，你可以利用它来理解代词指代（如"刚才提到的那个药"）、追踪诊断推理过程、避免重复提问。但这些历史信息仅用于上下文理解，核心医学依据仍需来自客观检索结果。
3. 提炼客观医学知识：你只能从参考信息中提取"客观的医学规律与知识"（例如：糖尿病的通用遗传规律、标准饮食禁忌、药物作用机制等）来直接回答当前用户的问题。
4. 严谨客观：如果检索到的信息完全没有提及当前问题需要的医学依据，请明确说明"参考信息不足"，绝不允许自行编造临床指南。

【输出结构要求】：
- 针对用户的具体问题，给出清晰、通用的病理分析和医学建议。
- 若涉及治疗/用药类，必须高亮【用药禁忌】、【相互作用】及【副作用】。
- 直接输出结论和分析，不要使用"根据上述上下文"等过渡性废话。
- 必须在末尾附上免责声明。

临床分析与建议：
"""

    def generate_adaptive_answer(self, question: str, documents: List[Document], memory_context: str = "") -> str:
        """
        智能统一医学答案生成
        自动适应病理分析、用药指导、科室推荐等不同类型的医疗查询
        """
        # 构建上下文
        context_parts = []

        for doc in documents:
            content = doc.page_content.strip()
            if content:
                # 添加检索层级和来源信息（实体级/主题级/向量/图谱）
                level = doc.metadata.get('retrieval_level', '')
                source = doc.metadata.get('search_type', '')
                prefix = f"[{level.upper()} | {source}] " if level else ""
                context_parts.append(f"{prefix}{content}")

        context = "\n\n".join(context_parts)
        prompt = self._build_medical_prompt(question, context, memory_context)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个严谨、专业的临床医学辅助决策AI。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            answer = response.choices[0].message.content.strip()
            # 强制追加免责声明
            disclaimer = "\n\n⚠️ 【医疗免责声明】：本系统的分析基于既往病例与知识图谱自动生成，仅供临床辅助参考，不能替代专业执业医师的当面诊断。具体用药及治疗方案请严格遵照医嘱。"
            if "免责声明" not in answer:
                answer += disclaimer

            return answer

        except Exception as e:
            logger.error(f"临床辅助答案生成失败: {e}")
            return f"抱歉，医学辅助系统生成回答时出现错误：{str(e)}"

    def generate_adaptive_answer_stream(self, question: str, documents: List[Document], memory_context: str = "", max_retries: int = 3):
        """
        流式医学答案生成（带网络重试机制，适合长篇临床分析报告）
        """
        # 构建医学上下文
        context_parts = []

        for doc in documents:
            content = doc.page_content.strip()
            if content:
                level = doc.metadata.get('retrieval_level', '')
                source = doc.metadata.get('search_type', '')
                prefix = f"[{level.upper()} | {source}] " if level else ""
                context_parts.append(f"{prefix}{content}")

        context = "\n\n".join(context_parts)
        prompt = self._build_medical_prompt(question, context, memory_context)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "你是一个严谨、专业的临床医学辅助决策AI。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                    timeout=60
                )

                if attempt == 0:
                    logger.info("开始流式生成临床分析报告...\n")
                else:
                    logger.info(f"第{attempt + 1}次尝试流式生成...\n")

                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        yield content

                # 检查并追加免责声明
                if "免责声明" not in full_response:
                    disclaimer = "\n\n⚠️ 【医疗免责声明】：本系统的分析基于既往病例与知识图谱自动生成，仅供临床辅助参考，不能替代专业执业医师的当面诊断。具体用药及治疗方案请严格遵照医嘱。"
                    yield disclaimer

                return

            except Exception as e:
                logger.warning(f"流式生成临床报告第{attempt + 1}次尝试失败: {e}")

                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    yield f"\n[系统提示：网络波动，{wait_time}秒后重试连接...]\n"
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"流式生成完全失败，尝试非流式后备方案")
                    yield "\n[系统提示：流式生成失败，正在切换到标准模式生成报告...]\n"

                    try:
                        fallback_response = self.generate_adaptive_answer(question, documents, memory_context)
                        yield fallback_response
                        return
                    except Exception as fallback_error:
                        logger.error(f"后备生成也失败: {fallback_error}")
                        error_msg = f"\n抱歉，生成医学参考报告时出现网络错误，请稍后重试。错误信息：{str(e)}"
                        yield error_msg
                        return
