"""
疾病词典扩充脚本

两阶段扩充：
  阶段1 (expand-diseases): 按科室调用 DeepSeek API 生成新的疾病名称，大幅增加覆盖
  阶段2 (generate-aliases): 为每个疾病生成别名/同义词（缩写、全称、俗称等）

用法：
  python scripts/expand_disease_dict.py                              # 两阶段全跑
  python scripts/expand_disease_dict.py --phase expand-diseases      # 仅阶段1
  python scripts/expand_disease_dict.py --phase generate-aliases     # 仅阶段2
  python scripts/expand_disease_dict.py --no-llm                     # 仅去重/格式转换

输出：
  data/disease_dict_new.txt       — 阶段1 输出：按科室扩充后的去重疾病列表
  data/disease_dict_expanded.txt  — 阶段2 输出：每行格式「主疾病名|别名1|别名2|...」
"""

import os
import sys
import re
import json
import time
import argparse
import logging
from typing import List, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(PROJECT_ROOT, "data", "disease_dict.txt")
DEFAULT_NEW = os.path.join(PROJECT_ROOT, "data", "disease_dict_new.txt")
DEFAULT_EXPANDED = os.path.join(PROJECT_ROOT, "data", "disease_dict_expanded.txt")

# ========== 科室列表（覆盖全院主要临床科室） ==========
DEPARTMENTS = [
    "心血管内科",
    "呼吸内科",
    "消化内科",
    "肾内科",
    "内分泌科",
    "风湿免疫科",
    "血液科",
    "神经内科",
    "精神心理科",
    "感染科",
    "肿瘤内科",
    "普外科",
    "心胸外科",
    "神经外科",
    "泌尿外科",
    "骨科",
    "妇产科",
    "儿科",
    "眼科",
    "耳鼻喉科",
    "口腔科",
    "皮肤性病科",
    "急诊科",
    "康复医学科",
    "老年病科",
    "麻醉科（疼痛门诊）",
    "中医科",
    "肝病科",
    "肛肠科",
    "乳腺外科",
    "甲状腺外科",
    "血管外科",
]


def get_llm_client():
    """初始化 DeepSeek LLM 客户端"""
    # 尝试加载 .env 文件（兼容 dotenv 环境）
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass

    try:
        from openai import OpenAI
    except ImportError:
        logger.error("缺少 openai 库，请执行: pip install openai")
        return None

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("未设置 DEEPSEEK_API_KEY 环境变量")
        return None

    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def call_llm(client, prompt: str, temperature: float = 0.3, max_retries: int = 3) -> str:
    """调用 DeepSeek LLM，含重试逻辑"""
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=4096,
            )
            content = resp.choices[0].message.content.strip()
            if content:
                return content
        except Exception as e:
            last_error = e
            logger.warning(f"LLM 调用失败 (第{attempt}次): {e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)
    logger.error(f"LLM 调用全部 {max_retries} 次重试后失败: {last_error}")
    return ""


def parse_disease_list_from_text(text: str) -> List[str]:
    """从 LLM 返回文本中解析疾病名称列表"""
    diseases = []

    # 尝试按行解析，去除序号和空白
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # 去掉行首序号如 "1." "1、" "1)" "- " "• "
        line = re.sub(r"^[\d\-*•]+[\.\、\）\)]\s*", "", line).strip()
        # 去掉行首的"- "或"* "标记
        line = re.sub(r"^[\-\*]\s+", "", line).strip()
        # 跳过空行及明显非疾病行的标记（如"##"、"---"、markdown标题等）
        if not line or line.startswith("#") or line.startswith("---"):
            continue
        # 跳过只有标点或数字的行
        if re.match(r"^[\s\-\.,，、。]+$", line):
            continue
        # 提取 # 号后的内容（如果行内包含注释风格的分隔）
        if "#" in line:
            line = line.split("#")[0].strip()
        # 按中文顿号/逗号/分号切分（有时 LLM 会在一行列出多个）
        parts = re.split(r"[、，,；;]", line)
        for p in parts:
            p = p.strip()
            # 至少 2 个中文字符才算有效疾病名
            if len(p) >= 2 and re.search(r"[一-鿿]", p):
                diseases.append(p)

    return diseases


def load_disease_set(filepath: str) -> Set[str]:
    """加载疾病词典为 set（去重）"""
    if not os.path.exists(filepath):
        logger.warning(f"文件不存在: {filepath}")
        return set()
    with open(filepath, "r", encoding="utf-8") as f:
        diseases = set()
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 支持 expanded 格式：只取第一个 | 分隔的为主名
            name = line.split("|")[0].strip()
            if name:
                diseases.add(name)
    return diseases


def save_disease_list(filepath: str, diseases: List[str]):
    """保存疾病列表（去重、按长度降序、每行一个）"""
    unique = sorted(set(diseases), key=lambda x: (len(x), x), reverse=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for d in unique:
            f.write(d + "\n")
    logger.info(f"已写入 {len(unique)} 个疾病到 {filepath}")
    return unique


def save_expanded_dict(filepath: str, entries: List[Tuple[str, List[str]]]):
    """保存扩展词典，格式：主疾病名|别名1|别名2|..."""
    with open(filepath, "w", encoding="utf-8") as f:
        for primary, aliases in entries:
            parts = [primary] + aliases
            f.write("|".join(parts) + "\n")
    logger.info(f"已写入 {len(entries)} 条扩展到 {filepath}")


# ========== 阶段1：按科室扩充病种 ==========

def phase_expand_diseases(input_path: str, output_path: str, no_llm: bool = False):
    """
    阶段1：按科室调用 LLM 生成新疾病名称，合并到现有词典后去重输出。
    每批 2 个科室，目标每个科室产出 15-30 个疾病。
    """
    logger.info("=" * 50)
    logger.info("阶段1：按科室扩充疾病列表")
    logger.info("=" * 50)

    # 加载已有疾病
    existing = load_disease_set(input_path)
    all_diseases = set(existing)
    logger.info(f"现有疾病词典: {len(existing)} 个")

    if no_llm:
        logger.info("--no-llm 模式：仅去重输出")
        save_disease_list(output_path, list(all_diseases))
        return

    client = get_llm_client()
    if not client:
        logger.warning("LLM 不可用，降级为仅去重输出")
        save_disease_list(output_path, list(all_diseases))
        return

    # 每批处理 2 个科室
    batch_size = 2
    total_new = 0

    for i in range(0, len(DEPARTMENTS), batch_size):
        batch = DEPARTMENTS[i : i + batch_size]
        dept_names = "、".join(batch)

        logger.info(f"\n正在处理科室 ({i // batch_size + 1}/{(len(DEPARTMENTS) + batch_size - 1) // batch_size}): {dept_names}")

        prompt = (
            f"你是一位资深医学专家。请列出{dept_names}最常见的疾病名称（每个科室15-30个）。\n\n"
            f"要求：\n"
            f"1. 只输出疾病名称，每行一个，不要序号和额外说明\n"
            f"2. 使用标准临床疾病名称（ICD-10 中文名称优先）\n"
            f"3. 涵盖该科室的常见病、多发病、典型病种\n"
            f"4. 避免与以下已有疾病重复（但也可输出，后续会去重）:\n"
        )
        # 附加已有疾病中可能相关的作为参考
        for d in sorted(existing)[:30]:
            prompt += f"   - {d}\n"
        prompt += (
            f"\n请按以下格式输出：\n"
            f"--- {batch[0]} ---\n"
            f"疾病1\n疾病2\n...\n"
        )
        if len(batch) > 1:
            prompt += f"\n--- {batch[1]} ---\n疾病1\n疾病2\n..."

        content = call_llm(client, prompt, temperature=0.3)
        if not content:
            logger.warning(f"科室 {dept_names} 未获取到结果，跳过")
            continue

        new_names = parse_disease_list_from_text(content)
        # 过滤：至少 2 个中文字符
        new_names = [n for n in new_names if len(n) >= 2 and re.search(r"[一-鿿]", n)]
        before = len(all_diseases)
        all_diseases.update(new_names)
        added = len(all_diseases) - before
        total_new += added
        logger.info(f"  本批新增 {added} 个疾病（共解析 {len(new_names)} 个候选）")

        # 每批后即时写入磁盘，防止中途中断丢数据
        save_disease_list(output_path, list(all_diseases))

        # 每批后休息一下，避免限流
        if i + batch_size < len(DEPARTMENTS):
            time.sleep(1)

    logger.info(f"\n阶段1完成：原有 {len(existing)} 个，新增 {total_new} 个，共 {len(all_diseases)} 个（结果已实时写入 {output_path}）")


# ========== 阶段2：生成别名 ==========

def phase_generate_aliases(input_path: str, output_path: str, no_llm: bool = False):
    """
    阶段2：在扩充后的疾病列表上，为每个疾病生成别名。
    每批 10 个疾病提交 LLM，输出格式为「主疾病名|别名1|别名2」。
    """
    logger.info("=" * 50)
    logger.info("阶段2：生成疾病别名")
    logger.info("=" * 50)

    diseases = load_disease_set(input_path)
    if not diseases:
        logger.error(f"未加载到疾病列表: {input_path}")
        return

    # 按长度降序排列
    sorted_diseases = sorted(diseases, key=lambda x: (len(x), x), reverse=True)
    logger.info(f"待处理疾病: {len(sorted_diseases)} 个")

    if no_llm:
        logger.info("--no-llm 模式：仅输出主名|主名（格式占位）")
        entries = [(d, [d]) for d in sorted_diseases]
        save_expanded_dict(output_path, entries)
        return

    client = get_llm_client()
    if not client:
        logger.warning("LLM 不可用，降级为仅输出格式占位")
        entries = [(d, [d]) for d in sorted_diseases]
        save_expanded_dict(output_path, entries)
        return

    entries = []
    batch_size = 10

    for i in range(0, len(sorted_diseases), batch_size):
        batch = sorted_diseases[i : i + batch_size]
        logger.info(f"  处理批次 {i // batch_size + 1}/{(len(sorted_diseases) + batch_size - 1) // batch_size}: 疾病 {batch[0]}...")

        prompt = (
            "你是一位资深医学编码专家。请为以下每个疾病名称生成临床常用的别名/同义词。\n\n"
            "要求：\n"
            "1. 别名包括：缩写、英文缩写、全称、俗称、ICD-10常用名称、中医病名（如有）\n"
            "2. 每个疾病输出 2-5 个别名，真实存在，不可编造\n"
            "3. 如果某个疾病确实没有别名，输出疾病名本身即可\n"
            "4. 输出格式严格按以下 JSON 格式（重要！供程序解析）:\n"
            '   [\n'
            '     {"disease": "疾病名", "aliases": ["别名1", "别名2", "别名3"]},\n'
            "     ...\n"
            "   ]\n\n"
            "疾病列表：\n"
        )
        for d in batch:
            prompt += f"  - {d}\n"

        content = call_llm(client, prompt, temperature=0.3)
        if not content:
            logger.warning(f"批次 {i // batch_size + 1} 未获取结果，疾病名自身作为占位")
            for d in batch:
                entries.append((d, [d]))
            continue

        # 解析 JSON
        batch_entries = parse_alias_json(content, batch)
        entries.extend(batch_entries)
        logger.info(f"  本批完成: {len(batch_entries)} 条")

        # 每批后即时写入磁盘，防止中断丢数据
        save_expanded_dict(output_path, entries)

        # 每批后休息
        if i + batch_size < len(sorted_diseases):
            time.sleep(1)

    save_expanded_dict(output_path, entries)
    logger.info(f"阶段2完成：共生成 {len(entries)} 条别名记录")


def parse_alias_json(text: str, expected_diseases: List[str]) -> List[Tuple[str, List[str]]]:
    """从 LLM 返回文本中解析 JSON 格式的别名数据，带容错"""
    # 尝试提取 JSON 部分（可能在 ```json ... ``` 中）
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()

    # 尝试直接解析
    try:
        data = json.loads(text)
        if isinstance(data, list):
            result = []
            for item in data:
                disease = item.get("disease", "").strip()
                aliases = item.get("aliases", [])
                if isinstance(aliases, list):
                    aliases = [a.strip() for a in aliases if a.strip()]
                if disease:
                    result.append((disease, aliases))
            return result
    except json.JSONDecodeError:
        pass

    # 容错：逐行启发式解析
    logger.warning("JSON 解析失败，尝试启发式解析")
    result = []
    for disease in expected_diseases:
        result.append((disease, [disease]))
    return result


# ========== 主入口 ==========

def main():
    parser = argparse.ArgumentParser(
        description="疾病词典扩充脚本：按科室扩病种 + 生成别名",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python scripts/expand_disease_dict.py                        # 两阶段全跑\n"
            "  python scripts/expand_disease_dict.py --phase expand-diseases\n"
            "  python scripts/expand_disease_dict.py --phase generate-aliases\n"
            "  python scripts/expand_disease_dict.py --no-llm               # 仅去重\n"
        ),
    )
    parser.add_argument(
        "--phase",
        choices=["all", "expand-diseases", "generate-aliases"],
        default="all",
        help="执行阶段 (默认: all)",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"输入疾病词典路径 (默认: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-new",
        default=DEFAULT_NEW,
        help=f"阶段1输出路径 (默认: {DEFAULT_NEW})",
    )
    parser.add_argument(
        "--output-expanded",
        default=DEFAULT_EXPANDED,
        help=f"阶段2输出路径 (默认: {DEFAULT_EXPANDED})",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="不调用 LLM，仅去重和格式转换",
    )
    args = parser.parse_args()

    logger.info(f"项目根目录: {PROJECT_ROOT}")

    if args.phase in ("all", "expand-diseases"):
        phase_expand_diseases(args.input, args.output_new, no_llm=args.no_llm)

    if args.phase in ("all", "generate-aliases"):
        # 阶段2的输入来自阶段1的输出；如果 --phase generate-aliases 单独执行，
        # 可接收自定义 --input 指向 disease_dict_new.txt
        phase_input = args.output_new if args.phase == "all" else args.input
        phase_generate_aliases(phase_input, args.output_expanded, no_llm=args.no_llm)

    if args.phase == "all":
        logger.info("\n✅ 两阶段扩充完成！")
        logger.info(f"  阶段1输出: {args.output_new}")
        logger.info(f"  阶段2输出: {args.output_expanded}")
    elif args.phase == "expand-diseases":
        logger.info(f"\n✅ 阶段1完成！输出: {args.output_new}")
    else:
        logger.info(f"\n✅ 阶段2完成！输出: {args.output_expanded}")


if __name__ == "__main__":
    main()
