#!/usr/bin/env python
"""
路由策略 Ground Truth 标注工具
=================================
读取测试集 CSV，调用 LLM 逐条分析每条查询应该分配到哪个检索策略，
新增 expected_routing_strategy 列，保存为新的 CSV 文件。

使用方法:
    conda activate all-in-rag
    cd scripts/strategy_comparison
    python label_routing_ground_truth.py                          # 默认处理 sampled_100_qa.csv
    python label_routing_ground_truth.py --input sampled_5_qa.csv --output sampled_5_qa_labeled.csv
"""

import os
import re
import sys
import csv
import time
import json
import argparse
import logging
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("label_routing")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# 路由策略说明（让 LLM 理解各策略的能力边界）
ROUTING_INSTRUCTION = """
你是一个医疗检索系统的路由决策专家。请根据用户的问题，判断应该使用哪种检索策略来获取最相关的医学信息。

可选策略：

1. hybrid_traditional —— 混合检索（向量语义 + BM25 关键词）
   适用场景: 通用医疗知识问答、就医指南、检查项目说明、药物用法等"事实性查询"。
   特点: 直接匹配语义相似或关键词相同的文档，适合查询意图清晰、不需要多步推理的问题。
   示例:
   - "高血压应该挂什么科？"
   - "急性肠胃炎如何诊治？"
   - "糖尿病吃饭能注意什么？"
   - "有高血压能喝葡萄酒吗？"
   - "什么是肝硬化？"

2. graph_rag —— 图结构检索（通过知识图谱多跳遍历关联的问答对）
   适用场景: 需要跨实体推理、涉及疾病间关系、药物相互作用、并发症等"关系型查询"。
   特点: 通过知识图谱的实体关联找到相关的问答案例，即使语义差距大也能通过关系路径召回。
   示例:
   - "乙肝大三阳会不会传染？性生活需要注意什么？"（涉及疾病与生活方式的关联）
   - "得了小三阳怎么治疗？"（需要关联乙肝治疗方案）
   - "高血压患者能吃党参吗？"（疾病 × 药物相互作用）

3. combined —— 混合 + 图谱融合（RRF 融合两种检索结果）
   适用场景: 复杂病情咨询，既有事实查询需求又有关系推理需求。
   特点: 取 hybrid_traditional 和 graph_rag 的并集，用 RRF 排序，覆盖率最高但延迟也最高。
   示例:
   - "脂肪肝应该怎样治疗见效呢"（既有治疗方案查询，又有疾病关联推理）

请根据以下标准判断：
- 查询是否涉及"疾病/症状 × X"的交叉关系（如疾病×药物、疾病×生活方式、并发症等）→ graph_rag 或 combined
- 查询是否包含多个医疗实体之间的交互或因果关系 → graph_rag 或 combined
- 查询是否仅为单一医疗知识的解释或指南查询 → hybrid_traditional
- 查询是否明确询问治疗方案+疾病关联 → combined

请只返回以下 JSON 格式，不要包含任何其他说明：
{"strategy": "hybrid_traditional", "reason": "简短说明"}
{"strategy": "graph_rag", "reason": "简短说明"}
{"strategy": "combined", "reason": "简短说明"}
"""


def build_query_text(department: str, title: str, ask: str, original_ask: Optional[str] = None) -> str:
    """构建传给 LLM 的查询文本"""
    parts = [f"【科室】{department}", f"【标题】{title}", f"【描述】{ask}"]
    if original_ask:
        parts.append(f"【原始描述】{original_ask}")
    return "\n".join(parts)


def classify_with_llm(client, query_text: str, model: str = "deepseek-v4-flash") -> dict:
    """调用 LLM 判断路由策略"""
    prompt = ROUTING_INSTRUCTION + f"\n\n请分析以下用户问题：\n{query_text}"

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=200,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    # 提取 JSON
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"LLM 返回格式异常，无法提取 JSON: {raw[:200]}")
    result = json.loads(match.group())

    strategy = result.get("strategy", "").strip()
    if strategy not in ("hybrid_traditional", "graph_rag", "combined"):
        raise ValueError(f"LLM 返回了未知策略: {strategy}")
    return result


def main():
    parser = argparse.ArgumentParser(description="路由策略 Ground Truth 标注工具")
    parser.add_argument(
        "--input",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sampled_100_qa_rewritten.csv"),
        help="输入 CSV 路径（默认 sampled_100_qa.csv）",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出 CSV 路径（默认在输入文件名上加 _labeled）",
    )
    parser.add_argument(
        "--model",
        default="deepseek-v4-flash",
        help="LLM 模型（默认 deepseek-v4-flash）",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="最多处理多少行（0=全部，调试用）",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="从中断的输出文件恢复（追加标注未处理的条目）",
    )
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"❌ 输入文件不存在: {input_path}")
        sys.exit(1)

    output_path = args.output or input_path.replace(".csv", "_labeled.csv")
    if output_path == input_path:
        print("❌ 输出路径不能与输入路径相同")
        sys.exit(1)

    # ---- 初始化 LLM 客户端 ----
    from openai import OpenAI
    from dotenv import load_dotenv

    # 加载 .env 中的 DEEPSEEK_API_KEY
    load_dotenv(override=True)
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ 未设置 DEEPSEEK_API_KEY 环境变量")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    print(f"✅ LLM 客户端初始化完成 (model={args.model})")

    # ---- 读取输入 CSV ----
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        all_rows = list(reader)

    print(f"📄 输入文件: {input_path} ({len(all_rows)} 条)")

    # ---- 恢复支持 ----
    existing_labels = {}  # ask -> strategy
    if args.resume and os.path.exists(args.resume):
        with open(args.resume, "r", encoding="utf-8") as f:
            resume_reader = csv.DictReader(f)
            for row in resume_reader:
                if row.get("expected_routing_strategy"):
                    existing_labels[row.get("ask", "")] = row["expected_routing_strategy"]
        print(f"🔄 从 {args.resume} 恢复 {len(existing_labels)} 条已有标注")

    # 限制行数
    rows_to_process = all_rows[: args.max_rows] if args.max_rows > 0 else all_rows
    total = len(rows_to_process)

    # ---- 逐条标注 ----
    results = []
    stats = {"hybrid_traditional": 0, "graph_rag": 0, "combined": 0}
    cost_est = 0.0

    for i, row in enumerate(rows_to_process):
        ask = row.get("ask", "")
        title = row.get("title", "")
        department = row.get("department", "")
        original_ask = row.get("original_ask", None)

        # 恢复：如果已有标注则直接使用
        if ask in existing_labels:
            strategy = existing_labels[ask]
            row["expected_routing_strategy"] = strategy
            row["routing_label_reason"] = "（从已有标注恢复）"
            results.append(row)
            stats[strategy] = stats.get(strategy, 0) + 1
            continue

        query_text = build_query_text(department, title, ask, original_ask)
        print(f"\r  [{i+1}/{total}] 标注中... {ask[:30]}", end="", flush=True)

        # LLM 调用 + 重试
        max_retries = 2
        result = None
        for attempt in range(max_retries + 1):
            try:
                result = classify_with_llm(client, query_text, model=args.model)
                break
            except Exception as e:
                logger.warning(f"  LLM 调用失败 (第{attempt+1}次): {e}")
                if attempt < max_retries:
                    time.sleep(1)
                else:
                    print(f"\n  ❌ 跳过第 {i+1} 条: {e}")
                    result = {"strategy": "hybrid_traditional", "reason": "LLM调用失败，使用默认值"}

        strategy = result["strategy"]
        reason = result.get("reason", "")

        row["expected_routing_strategy"] = strategy
        row["routing_label_reason"] = reason
        results.append(row)
        stats[strategy] = stats.get(strategy, 0) + 1

        # 估算 tokens（粗略）
        cost_est += len(ROUTING_INSTRUCTION + query_text) / 4

        # 每 5 条保存一次
        if (i + 1) % 5 == 0:
            _save_output(results, output_path, fieldnames)
            print(f" → auto-saved")

    # ---- 保存最终文件 ----
    new_fieldnames = list(fieldnames) + ["expected_routing_strategy", "routing_label_reason"]
    _save_output(results, output_path, new_fieldnames)

    print(f"\n\n✅ 标注完成！")
    print(f"   输出文件: {output_path}")
    print(f"   分布: hybrid_traditional={stats['hybrid_traditional']}, "
          f"graph_rag={stats['graph_rag']}, combined={stats['combined']}")
    print(f"   估算输入 tokens: ~{int(cost_est / 1000)}k")


def _save_output(rows: list, output_path: str, fieldnames: list):
    """写入 CSV（保证字段存在）"""
    for row in rows:
        for col in ("expected_routing_strategy", "routing_label_reason"):
            if col not in row:
                row[col] = ""
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
