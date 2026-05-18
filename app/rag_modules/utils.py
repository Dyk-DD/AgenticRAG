"""
临床 RAG 系统共享工具函数
"""

import json
import re
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def extract_json_from_llm(raw_content: str) -> Optional[dict]:
    """
    从 LLM 输出中稳健提取 JSON 对象。

    处理的问题：
    1. 模型在 JSON 前有思考/解释文本
    2. 模型在 JSON 外额外输出空 {} 或其他大括号
    3. 模型用 markdown code block 包裹 JSON
    4. 模型输出包含尾随逗号等非严格 JSON

    Returns:
        dict 或 None（提取失败时）
    """
    if not raw_content:
        return None

    text = raw_content.strip()

    # 1. 尝试直接解析（最常见情况）
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. 提取 markdown code block 中的 JSON（```json ... ```）
    code_block_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text)
    if code_block_match:
        candidate = code_block_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # 尝试修复常见格式问题后继续
            pass

    # 3. 从最后一个 { 到最后一个 }（跳过前导空对象 {}）
    # 策略：找到所有 {} 对，跳过最短的那些，取最长那个
    brace_positions = []
    for i, ch in enumerate(text):
        if ch == '{':
            brace_positions.append(i)
        elif ch == '}':
            if brace_positions:
                start = brace_positions.pop()
                candidate = text[start:i + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict) and len(parsed) > 0:
                        return parsed
                except json.JSONDecodeError:
                    continue

    # 4. 尝试最贪婪的 { 到最后一个 } 提取（兼容嵌套 JSON 但有大括号垃圾的情况）
    last_brace = text.rfind('}')
    first_brace = text.find('{')
    if first_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace:last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 5. 尝试修复常见 JSON 格式错误
    try:
        # 移除尾随逗号
        fixed = re.sub(r',\s*}', '}', candidate if 'candidate' in dir() else text)
        # 移除注释风格的 //
        fixed = re.sub(r'//[^\n]*', '', fixed)
        parsed = json.loads(fixed)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, UnboundLocalError):
        pass

    logger.warning(f"无法从 LLM 输出中提取 JSON: {text[:200]}...")
    return None
