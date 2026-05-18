"""
医疗问答数据清洗 + 重症问答对生成
所有清洗工作在 data/cleaning/ 下独立进行，不影响 data/processed/ 的 RAG 索引。
完成后手动将最终 CSV 复制到 data/processed/ 即可。

用法: 
  python scripts/clean_and_generate_qa.py          # 执行阶段一和阶段二
  python scripts/clean_and_generate_qa.py --skip-clean  # 跳过阶段一，只执行阶段二
"""

import csv
import os
import re
import shutil
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

# ── 配置 ──────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
CLEANING_DIR = DATA_DIR / "cleaning"
PROCESSED_DIR = DATA_DIR / "processed"

INPUT_FILE = PROCESSED_DIR / "样例_内科5000-6000.csv"
OUTPUT_FILE = CLEANING_DIR / "cleaned_内科_plus_重症.csv"
FINAL_FILE = PROCESSED_DIR / "cleaned_内科_plus_重症.csv"
COLUMNS = ["department", "title", "ask", "answer"]

MODEL = "deepseek-chat"
MAX_RETRIES = 3
GENERATE_COUNT = 7000

# ── 提示词 ──────────────────────────────────────────

CLEAN_PROMPT = """你是一个医疗数据清洗专家。请对以下医疗问答对进行清洗，要求：

1. 修正语法错误、错别字、不通顺的表述
2. 问题部分保留病人的原始症状和核心关切，删除冗余的寒暄
3. 回答部分重新组织为专业、条理清晰的医学建议：
   - 先给出核心结论
   - 再分点列出具体建议（如：用药、饮食、运动、复查等）
   - 如有必要，补充注意事项
4. 删除结尾的"感谢咨询"、"期望帮助"等模板化客套话
5. 删除回答中的内部标记和无关内容
6. 如果回答有明显医学错误，请修正为正确的医学知识
7. 使语言保持专业但易懂，适合患者阅读
8. 回答必须完整，确保所有医学要点都被保留
9. 回答中严禁使用任何换行符（\\n），所有内容必须写成一个连续段落。用"；"或"。"连接各个要点，不得使用编号列表。

请严格按以下格式输出（不要加任何额外说明和序号）：
[DEPARTMENT]{{department}}[/DEPARTMENT]
[TITLE]{{title}}[/TITLE]
[ASK]{{ask}}[/ASK]
[ANSWER]{{answer}}[/ANSWER]

注意：[/ANSWER] 标记必须在回答完全结束后输出，确保回答内容完整。

【待清洗数据】
科室: {department}
标题: {title}
问题: {ask}
回答: {answer}"""


CRITICAL_DEPT_PROMPT = """你是一位拥有20年临床经验的重症医学科主任医师。请为以下重症科室生成真实的医疗问答对：

【覆盖科室及比例】
- ICU重症监护：30%（呼吸衰竭、多器官衰竭、脓毒症、术后监护等）
- 急诊科：25%（急性胸痛、卒中、严重创伤、中毒、急性腹痛等）
- 神经内科重症：15%（脑梗死、脑出血、癫痫持续状态、重症肌无力危象等）
- 心血管重症：15%（急性心梗、恶性心律失常、心力衰竭、高血压危象等）
- 呼吸内科重症：15%（重症肺炎、ARDS、COPD急性加重、肺栓塞等）

【问题要求】
- 模拟真实患者或家属的口吻，包含具体症状、持续时间、既往病史
- 问题应有紧迫感，体现"重症"特征
- 长度：30-200字

【回答要求】
- 体现三甲医院重症医学专业水准
- 结构：紧急评估 → 可能诊断 → 紧急处理 → 后续建议
- 区分院前处理和入院后处理
- 包含必要的警示信号（什么情况立即拨打120）
- 长度：100-500字
- 回答中严禁使用任何换行符，所有内容必须写成一个连续段落，用"；"或"。"连接要点

请严格按以下CSV格式输出，每行一条，不要序号和额外说明，不要用markdown代码块包裹：
department,title,ask,answer

示例行：
ICU重症监护,重症肺炎患者需要使用呼吸机吗？,我父亲今年68岁因重症肺炎住进ICU已经3天了血氧一直上不去医生说要插管上呼吸机我们很担心想问问呼吸机治疗是否有必要有什么风险？,重症肺炎出现呼吸衰竭时机械通气是挽救生命的关键手段。当常规氧疗无法维持血氧饱和度>90%时就需要考虑气管插管。呼吸机可以保证氧合、减少呼吸肌耗氧、为抗感染治疗争取时间。风险包括呼吸机相关性肺炎和气压伤但ICU会有严格监控。家属应与主管医生充分沟通了解每日脱机评估计划。

现在请生成 {count} 条重症科室问答对："""


# ── 核心函数 ──────────────────────────────────────

def init_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY 环境变量")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def call_deepseek(client: OpenAI, prompt: str, model: str = MODEL, max_tokens: int = 2048) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens,
                timeout=300,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = (attempt + 1) * 5
                print(f"  API 调用失败 (重试 {attempt + 1}/{MAX_RETRIES}，等待 {wait}s): {e}")
                time.sleep(wait)
            else:
                raise


def parse_cleaned_output(raw: str) -> dict:
    """用 XML 风格标记解析清洗后的输出，支持多行回答"""
    result = {"department": "", "title": "", "ask": "", "answer": ""}

    def _extract(tag: str, text: str) -> str:
        pattern = rf"\[{tag}\](.*?)\[/{tag}\]"
        m = re.search(pattern, text, re.DOTALL)
        return m.group(1).strip() if m else ""

    for tag in ["DEPARTMENT", "TITLE", "ASK", "ANSWER"]:
        key = tag.lower()
        result[key] = _extract(tag, raw)

    return result


def is_truncated(cleaned: str, original: str, threshold: float = 0.3) -> bool:
    """检查清洗结果是否被截断"""
    if not cleaned or not original:
        return True
    return len(cleaned) < len(original) * threshold


def count_output_rows(path: Path) -> int:
    """统计输出文件中已有的数据行数（不含表头）"""
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    return max(0, len(rows) - 1)  # 减去表头行


def ensure_output_file_exists(output_path: Path):
    """确保输出文件存在且有表头"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
        print(f"创建输出文件: {output_path}")


def clean_csv(client: OpenAI, input_path: Path, output_path: Path):
    """逐条清洗 CSV 中的问答对，边处理边保存，支持断点续传"""
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    total = len(rows)

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 断点检测 ──
    existing = count_output_rows(output_path)
    if existing > 0 and existing < total:
        print(f"检测到已完成 {existing}/{total} 条，从第 {existing + 1} 条继续...\n")
    elif existing >= total:
        print(f"输出文件已有 {existing} 条（≥ 输入 {total} 条），清洗阶段已完成，跳过。")
        return
    else:
        print(f"共 {total} 条待清洗\n")
        # 首次运行，写入表头
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)

    start_index = existing
    cleaned_count = 0
    fallback_count = 0

    for i, row_data in enumerate(rows, 1):
        # 跳过已处理的行
        if i <= start_index:
            continue

        dept = row_data.get("department", "").strip()
        title = row_data.get("title", "").strip()
        ask = row_data.get("ask", "").strip()
        answer = row_data.get("answer", "").strip()

        if not all([dept, ask, answer]):
            print(f"[{i}/{total}] 跳过空数据")
            continue

        prompt = CLEAN_PROMPT.format(department=dept, title=title, ask=ask, answer=answer)

        try:
            raw = call_deepseek(client, prompt, max_tokens=4096)
            parsed = parse_cleaned_output(raw)

            c_dept = parsed["department"] or dept
            c_title = parsed["title"] or title
            c_ask = parsed["ask"] or ask
            c_answer = parsed["answer"] or answer

            # 强制合并换行
            c_answer = re.sub(r'\n\s*\n', '。', c_answer)
            c_answer = re.sub(r'\n', '；', c_answer)
            c_ask = c_ask.replace('\n', '')

            # 截断检测
            if is_truncated(c_answer, answer):
                print(f"[{i}/{total}] ⚠ 回答疑似截断 (原{len(answer)}字→新{len(c_answer)}字)，回退原文")
                c_answer = answer
                fallback_count += 1

            # 逐条追加写入
            with open(output_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([c_dept, c_title, c_ask, c_answer])

            cleaned_count += 1
            print(f"[{i}/{total}] ✓ {c_title[:30]}... | ans {len(c_answer)}字")

        except Exception as e:
            print(f"[{i}/{total}] ✗ 失败: {e}")
            fallback_count += 1
            with open(output_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([dept, title, ask, answer])

    print(f"\n清洗完成: 本批 {cleaned_count} 条, 累计 {existing + cleaned_count}/{total} (截断回退 {fallback_count})")


def generate_critical_qa(client: OpenAI, output_path: Path, count: int = GENERATE_COUNT):
    """生成重症科室的问答对，追加到文件末尾，支持断点续传"""
    # 确保输出文件存在
    ensure_output_file_exists(output_path)
    
    # ── 断点检测：统计输出文件中已有的数据行 ──
    existing_total = count_output_rows(output_path) if output_path.exists() else 0
    if existing_total >= count:
        print(f"输出文件已有 {existing_total} 条，生成阶段已完成，跳过。")
        return

    print(f"\n{'='*60}")
    print(f"生成目标: {count} 条, 已生成: {existing_total} 条, 待生成: {count - existing_total} 条\n")

    generated = 0
    batch_size = 2
    remaining = count - existing_total
    total_requests = (remaining + batch_size - 1) // batch_size

    for batch_idx in range(total_requests):
        batch_count = min(batch_size, remaining - generated)
        batch_prompt = CRITICAL_DEPT_PROMPT.format(count=batch_count)

        print(f"  ⏳ 批次 {batch_idx + 1}/{total_requests}: 请求 API 生成 {batch_count} 条 (累计 {existing_total + generated}/{count})...", flush=True)

        try:
            raw = call_deepseek(client, batch_prompt, max_tokens=8192)

            lines = raw.strip().split("\n")
            batch_added = 0
            with open(output_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("```") or line.startswith("department"):
                        continue
                    row = next(csv.reader([line]), None)
                    if row and len(row) >= 4:
                        clean_row = [re.sub(r'\n+', '', cell) if cell else '' for cell in row[:4]]
                        writer.writerow(clean_row)
                        batch_added += 1
                        generated += 1

            print(f"  批次 {batch_idx + 1}/{total_requests}: +{batch_added} 条 (累计 {existing_total + generated}/{count})")

        except Exception as e:
            print(f"  批次 {batch_idx + 1} 失败: {e}")

        if existing_total + generated >= count:
            break

    print(f"\n生成完成: 本批 {generated} 条, 累计 {existing_total + generated} 条")


def copy_to_processed():
    """复制最终文件到 processed 目录"""
    if OUTPUT_FILE.exists():
        print(f"\n{'='*60}")
        print(f"复制最终文件到 RAG 索引目录...")
        shutil.copy2(str(OUTPUT_FILE), str(FINAL_FILE))
        print(f"  {OUTPUT_FILE}")
        print(f"  → {FINAL_FILE}")
    else:
        print(f"\n⚠️ 警告: 输出文件 {OUTPUT_FILE} 不存在，跳过复制")


# ── 主流程 ────────────────────────────────────────

def main():
    # 解析命令行参数
    skip_clean = "--skip-clean" in sys.argv
    
    if skip_clean:
        print("⚠️  跳过阶段一（数据清洗），仅执行阶段二（生成重症问答对）\n")
    else:
        print("✓ 执行完整流程（阶段一 + 阶段二）\n")
        print("提示：使用 'python scripts/clean_and_generate_qa.py --skip-clean' 可跳过清洗阶段\n")
    
    # 如果 data/processed/ 下有旧的中间产物，迁移到 cleaning 目录
    old_output = PROCESSED_DIR / "cleaned_内科_plus_重症.csv"
    if old_output.exists() and not OUTPUT_FILE.exists():
        print(f"检测到旧中间产物，迁移: {old_output} → {OUTPUT_FILE}")
        CLEANING_DIR.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_output), str(OUTPUT_FILE))
        print("迁移完成，将从断点继续处理")

    client = init_client()

    if not skip_clean:
        # 阶段一：清洗原始问答数据
        print("=" * 60)
        print("阶段一：清洗原始问答数据")
        print(f"  输入: {INPUT_FILE}")
        print(f"  输出: {OUTPUT_FILE}")
        print("=" * 60)
        
        if not INPUT_FILE.exists():
            print(f"⚠️ 错误: 输入文件不存在: {INPUT_FILE}")
            print("请确认文件路径是否正确，或使用 --skip-clean 跳过此阶段")
            sys.exit(1)
            
        clean_csv(client, INPUT_FILE, OUTPUT_FILE)
    else:
        print("阶段一已跳过")
        # 确保输出文件存在（供阶段二使用）
        ensure_output_file_exists(OUTPUT_FILE)

    # 阶段二：生成重症科室问答对
    print(f"\n{'=' * 60}")
    print("阶段二：生成重症科室问答对")
    print(f"  输出: {OUTPUT_FILE}")
    print("=" * 60)
    generate_critical_qa(client, OUTPUT_FILE, GENERATE_COUNT)

    # 完成后复制到 data/processed/ 供 RAG 索引
    copy_to_processed()
    
    print(f"\n全部完成！可以重启 RAG 后端索引新数据。")


if __name__ == "__main__":
    main()