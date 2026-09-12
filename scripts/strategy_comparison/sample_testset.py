#!/usr/bin/env python
"""
从原始 CSV 中随机采样 QA 对，并可选调用 LLM 对查询进行语义改写，
使得测试集的查询与数据库不完全一样，更符合实际生产环境测试。

改写后的文件保存为 sampled_{SAMPLE_SIZE}_qa_rewritten.csv，
保留 original_ask 列用于追溯对比。
"""

import os
import sys
import random
import logging
import pandas as pd
from dotenv import load_dotenv

# 加载 .env 中的 DEEPSEEK_API_KEY
load_dotenv(override=True)

# 修复 Windows GBK 编码下的 Unicode 输出问题
if sys.stdout.encoding and sys.stdout.encoding.upper() in ("GBK", "GB2312", "CP936"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

# 输出目录 = 本脚本所在目录
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(OUTPUT_DIR))

# 原始数据
SOURCE_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_内科_plus_重症.csv")
SAMPLE_SIZE = 100
SEED = 42
# 采样输出（原文）
OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"sampled_{SAMPLE_SIZE}_qa.csv")
# 改写后输出
OUTPUT_REWRITTEN_CSV = os.path.join(OUTPUT_DIR, f"sampled_{SAMPLE_SIZE}_qa_rewritten.csv")

# LLM 改写配置
LLM_REWRITE_ENABLED = True  # 设为 False 可跳过改写，仅采样
LLM_MODEL = "deepseek-v4-flash"
LLM_TEMPERATURE = 0.3  # 较低温度确保医学实体准确
LLM_MAX_TOKENS = 512

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("sample_testset")


def _init_llm_client():
    """初始化 OpenAI 兼容客户端（使用环境中的 DEEPSEEK_API_KEY）"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        logger.warning("未检测到 DEEPSEEK_API_KEY 环境变量，跳过 LLM 改写")
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        logger.info("LLM 客户端初始化成功 (base_url=https://api.deepseek.com)")
        return client
    except Exception as e:
        logger.warning(f"LLM 客户端初始化失败: {e}")
        return None


def rewrite_query(llm_client, ask_text: str, idx: int) -> str:
    """
    调用 LLM 对单条查询进行医学语义保留的改写。

    改写原则：
    - 保持疾病名、症状、药物、检查等医学实体完全不变
    - 仅改变措辞、句式、语气（标准化 -> 口语化）
    - 不添加原文没有的医学信息
    - 不遗漏原文的关键医学诉求
    """
    prompt = (
        "你是一名医学数据增强助手。请将以下患者的提问改写成另一种说法，\n"
        "要求：\n"
        "1. 保持所有医学实体完全不变（疾病名、症状、药物、检查、科室等）\n"
        "2. 保持核心医学诉求和意图完全不变\n"
        "3. 改变措辞、句式或语气，使其更像真实患者的自然口语表达\n"
        "4. 不添加原文没有的医学信息或诊断结论\n"
        "5. 只输出改写后的文本，不要任何解释或前缀\n\n"
        f"原文：{ask_text}"
    )

    try:
        resp = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        rewritten = resp.choices[0].message.content.strip()
        # 简单校验：非空且与原文不同
        if not rewritten:
            logger.warning(f"  第 {idx} 条改写结果为空，使用原文")
            return ask_text
        if rewritten == ask_text:
            logger.info(f"  第 {idx} 条改写结果与原文相同，保留")
        return rewritten
    except Exception as e:
        logger.warning(f"  第 {idx} 条改写失败: {e}，使用原文")
        return ask_text


def main():
    # ------------------------------------------------------------------
    #  1. 加载 & 采样
    # ------------------------------------------------------------------
    if not os.path.exists(SOURCE_CSV):
        print("[ERROR] 原始 CSV 不存在: %s" % SOURCE_CSV)
        return

    df = pd.read_csv(SOURCE_CSV)
    total = len(df)
    print("原始数据: %d 条 QA 对" % total)

    if SAMPLE_SIZE >= total:
        print("[WARN] 采样数(%d)大于等于总数(%d)，使用全部数据" % (SAMPLE_SIZE, total))
        sampled = df
    else:
        random.seed(SEED)
        indices = random.sample(range(total), SAMPLE_SIZE)
        sampled = df.iloc[indices].reset_index(drop=True)

    print("采样完成: %d 条" % len(sampled))
    print("  科室分布:")
    print(sampled["department"].value_counts().to_string())

    # ------------------------------------------------------------------
    #  2. 保存原文采样（原有行为保持不变）
    # ------------------------------------------------------------------
    sampled.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print("\n原文采样已保存: %s" % OUTPUT_CSV)

    # ------------------------------------------------------------------
    #  3. LLM 语义改写（新增）
    # ------------------------------------------------------------------
    if not LLM_REWRITE_ENABLED:
        print("\nLLM 改写已禁用 (LLM_REWRITE_ENABLED=False)，跳过改写")
        return

    llm_client = _init_llm_client()
    if llm_client is None:
        print("\nLLM 客户端不可用，跳过改写（仅保存原文采样）")
        print("  提示：请确保 .env 文件中配置了 DEEPSEEK_API_KEY")
        return

    print("\n%s" % ("=" * 55))
    print("  开始 LLM 语义改写 (%s, temperature=%s)" % (LLM_MODEL, LLM_TEMPERATURE))
    print("  共 %d 条查询需要改写" % len(sampled))
    print("%s" % ("=" * 55))

    rewritten_asks = []
    for idx, row in sampled.iterrows():
        ask_text = str(row.get("ask", "")).strip()
        if not ask_text:
            rewritten_asks.append("")
            continue

        rewritten = rewrite_query(llm_client, ask_text, idx)
        rewritten_asks.append(rewritten)

        # 进度显示
        if (idx + 1) % 10 == 0:
            print("  进度: %d/%d" % (idx + 1, len(sampled)))

    # 构造改写后的 DataFrame：保留原字段，ask 替换为改写版，新增 original_ask
    df_rewritten = sampled.copy()
    df_rewritten["ask"] = rewritten_asks
    df_rewritten["original_ask"] = sampled["ask"].values

    # 保存
    df_rewritten.to_csv(OUTPUT_REWRITTEN_CSV, index=False, encoding="utf-8")

    # 统计改写变化
    changed = sum(1 for orig, new in zip(sampled["ask"], rewritten_asks) if orig != new and new)
    print("\n%s" % ("=" * 55))
    print("LLM 改写完成!")
    print("  改写后文件: %s" % OUTPUT_REWRITTEN_CSV)
    print("  改写条数: %d/%d" % (changed, len(sampled)))
    print("  未变化: %d" % (len(sampled) - changed))

    # 展示改写对比示例（前 3 条）
    print("\n  改写对比示例:")
    print("  %s" % ("-" * 55))
    for i in range(min(3, len(sampled))):
        orig = str(sampled.iloc[i]["ask"])
        new = rewritten_asks[i]
        if orig != new:
            print("  [原文]  %s%s" % (orig[:80], "..." if len(orig) > 80 else ""))
            print("  [改写]  %s%s" % (new[:80], "..." if len(new) > 80 else ""))
            print("  %s" % ("-" * 55))

    print("\n使用提示: 将 run_comparison.py 中的测试集路径改为")
    print("  %s" % OUTPUT_REWRITTEN_CSV)
    print("  即可使用改写后的测试集进行策略对比评估。")


if __name__ == "__main__":
    main()
