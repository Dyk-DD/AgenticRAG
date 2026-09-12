#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""清空患者 / 会话 / 问答记录，把系统恢复到"无人使用过"的状态。

只动 qa_history.db 里的业务数据，不碰 Milvus / Neo4j 的索引与图谱
（那些是从历史 QA 构建出来的知识底座，不是访客数据）。

用法:
  python scripts/clear_test_data.py               # 先看有多少，再要求确认
  python scripts/clear_test_data.py --dry-run     # 只看，不删
  python scripts/clear_test_data.py --yes         # 跳过确认（脚本化调用）

容器内执行（scripts/ 在 .dockerignore 里，镜像中没有本文件，需先拷进去。
注意 Git Bash 会把容器路径改写成 Windows 路径，所以加 MSYS_NO_PATHCONV=1）:
  MSYS_NO_PATHCONV=1 docker cp scripts/clear_test_data.py agentic-rag-api:/tmp/
  MSYS_NO_PATHCONV=1 docker compose exec -T -e PYTHONIOENCODING=utf-8 \
      api python /tmp/clear_test_data.py --yes

宿主机执行仅在后端原生运行时才有意义 —— 那种模式下 data/qa_history.db 就是线上库。
容器化部署时库在 rag_state 命名卷里，宿主机上根本看不到，直接跑会另建一个空库。
"""

import os
import sys
import sqlite3
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 从应用自身取路径，避免脚本和线上用的不是同一个库
# （README: 容器里是 /app/data，宿主机上直接跑会落到仓库的 data/ 下）
from app.qa_database import DB_PATH  # noqa: E402

# 删除顺序即外键依赖顺序：qa_records -> sessions -> patients
# （qa_records.session_id -> sessions.id 有外键声明；SQLite 默认不强制，
#  但顺序错了在有 PRAGMA foreign_keys=ON 的环境下会失败）
TABLES = ["qa_records", "sessions", "patients"]


def _counts(conn):
    return {
        t: conn.execute('SELECT COUNT(*) FROM "%s"' % t).fetchone()[0] for t in TABLES
    }


def main():
    parser = argparse.ArgumentParser(description="清空访客业务数据（患者/会话/问答）")
    parser.add_argument("--yes", "-y", action="store_true", help="跳过确认提示")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不删除")
    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        print(f"数据库不存在: {DB_PATH}")
        print("（服务尚未启动过，无需清理）")
        return 0

    conn = sqlite3.connect(DB_PATH)
    try:
        before = _counts(conn)
        print(f"数据库: {DB_PATH}")
        print("当前记录:")
        for t in TABLES:
            print(f"  {t:<14} {before[t]}")

        if not any(before.values()):
            print("\n已经是空的，无需清理。")
            return 0

        if args.dry_run:
            print("\n--dry-run: 未做任何修改。")
            return 0

        if not args.yes:
            # 删除不可逆，且对象是患者数据 —— 默认要求显式确认
            print(
                f"\n将删除以上 {sum(before.values())} 条记录，"
                f"旧患者需重新注册，且无法恢复。"
            )
            if input("确认删除？输入 yes 继续: ").strip().lower() != "yes":
                print("已取消。")
                return 1

        for t in TABLES:
            conn.execute('DELETE FROM "%s"' % t)
        # qa_records 用了 AUTOINCREMENT，不清 sqlite_sequence 的话 id 会接着涨。
        # sqlite_sequence 表在库为空时可能不存在，所以查一下再删。
        has_seq = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type='table' AND name='sqlite_sequence'"
        ).fetchone()[0]
        if has_seq:
            for t in TABLES:
                conn.execute("DELETE FROM sqlite_sequence WHERE name = ?", (t,))
        conn.commit()

        after = _counts(conn)
        print("\n清理后:")
        for t in TABLES:
            print(f"  {t:<14} {after[t]}")
        print("\n完成。旧患者需重新注册。")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
