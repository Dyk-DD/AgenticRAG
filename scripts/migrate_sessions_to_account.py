#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把老患者身份名下的会话与问答记录改挂到一个邮箱账号上。

背景：访问码登录路径已下线（患者ID + 6 位访问码 → 邮箱 + 密码）。老身份
p_xxx 从此没有任何登录方式，它们的会话虽然还在库里，但从界面上够不着。
本脚本把这些会话的 client_id 从老 patient_id 改成目标账号的 id。

它同时兼任「账号找回」：源身份不必是老患者，普通账号也能作为源
（只是那种情况下会把一个账号的数据并到另一个账号，脚本会警告）。

⚠️ 语义记忆会丢：Neo4j 的 Turn 节点不带 client_id（按 session_id 归属，
session_id 不变），所以图谱记忆完好；但 Milvus 的记忆行**带** client_id，
迁移后旧会话的语义记忆对新账号读不出来。Milvus 没有 UPDATE，修它要删了
重新嵌入，标准库脚本做不到。删除功能不受影响（它按 session_id 过滤），
所以不会攒下僵尸行。

⚠️ 本脚本绝不构造 QADatabase()：构造函数里有一次性清理，会在真正的迁移
开始之前就把数据删掉。全程只用裸 sqlite3。

前置条件（缺一个都会停在 check_schema 或目标校验上）：

  1. 新版后端已经跑过一次，库里有了 email / password_salt / password_hash
     三列。注意**建列是懒的**：get_qa_db() 要到第一次有请求碰到数据库时才构造
     QADatabase()，而 /api/health 不碰数据库 —— 所以「重启完容器」并不等于
     「结构已升级」，在有人真正操作过界面之前，库还是旧结构，本脚本会直接拒绝。
  2. 目标账号已经在界面上注册好（--to-email 要求 password_hash 非空，否则
     拒绝迁入）。这条同样要求新后端 + 新前端已经上线。

  于是顺序只能是：上线新版后端 → 发布新前端 → 注册目标账号 → 停 api → 迁移
  → 起 api。不存在「先迁移再上线」这个顺序。

用法:
  # 先看要改多少，不写任何东西
  python scripts/migrate_sessions_to_account.py \
      --from-patient p_xxx --to-email you@example.com --dry-run

  # 正式迁
  python scripts/migrate_sessions_to_account.py \
      --from-patient p_xxx --to-email you@example.com --yes

容器内执行（scripts/ 在 .dockerignore 里，镜像中没有本文件）:

  必须先停掉 api —— 它是唯一并发写者。在跑的容器会让你「改完 sessions、
  它又插进一条带旧 client_id 的轮次」，结果是一轮永久不可见的问答加一个
  说谎的轮次数。而 exec 要求容器在运行，所以这里用 run：
  （注意 Git Bash 会把容器路径改写成 Windows 路径，所以要 MSYS_NO_PATHCONV=1）

  MSYS_NO_PATHCONV=1 docker compose stop api
  MSYS_NO_PATHCONV=1 docker compose run --rm --no-deps \
      -v "E:/AI_Project/Agentic-RAG/scripts:/app/scripts:ro" \
      -e PYTHONIOENCODING=utf-8 api \
      python /app/scripts/migrate_sessions_to_account.py \
      --from-patient p_xxx --to-email you@example.com --dry-run
  MSYS_NO_PATHCONV=1 docker compose start api

宿主机执行只在后端原生运行时才有意义 —— 那种模式下 data/qa_history.db 就是
线上库。容器化部署时库在 rag_state 命名卷里，宿主机上看到的是另一个空库，
直接跑会以「找不到该身份」失败（这是设计如此：宁可大声失败，也不要静默
改错文件）。
"""

import argparse
import os
import sqlite3
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 从应用自身取路径，避免脚本和线上用的不是同一个库
from app.qa_database import DB_PATH  # noqa: E402


def fail(msg: str, code: int = 2) -> int:
    print(f"错误：{msg}", file=sys.stderr)
    return code


def check_schema(conn) -> str:
    """回显一个错误信息，空串表示结构齐备。"""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(patients)")}
    if not cols:
        return "库里没有 patients 表"
    missing = [c for c in ("email", "password_salt", "password_hash") if c not in cols]
    if missing:
        return (
            f"patients 表缺少 {', '.join(missing)} 列 —— 这还是旧版结构。"
            "先启动一次新版后端，让它完成建列迁移，再跑本脚本。"
        )
    return ""


def counts_for(conn, pid: str) -> dict:
    """某个 client_id 名下有多少会话、问答，以及有多少条「孤儿轮次」。

    孤儿轮次 = client_id 为空、但所属会话归这个身份的问答记录。历史上
    record_qa 有过不传 client_id 的版本，会留下这种行：list_sessions 的
    LEFT JOIN 会数上它们，get_session_detail 按 client_id 过滤却看不到。
    """
    sessions = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE client_id = ?", (pid,)
    ).fetchone()[0]
    records = conn.execute(
        "SELECT COUNT(*) FROM qa_records WHERE client_id = ?", (pid,)
    ).fetchone()[0]
    orphans = conn.execute(
        "SELECT COUNT(*) FROM qa_records "
        "WHERE COALESCE(client_id, '') = '' "
        "AND session_id IN (SELECT id FROM sessions WHERE client_id = ?)",
        (pid,),
    ).fetchone()[0]
    return {"sessions": sessions, "records": records, "orphans": orphans}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="把老患者身份名下的会话改挂到一个邮箱账号上"
    )
    parser.add_argument("--from-patient", required=True,
                        help="源身份 id，老身份形如 p_xxxxxxxx")
    parser.add_argument("--to-email", required=True,
                        help="目标账号邮箱（写邮箱而不是 id，便于人工核对）")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写入")
    parser.add_argument("--yes", "-y", action="store_true", help="跳过确认提示")
    parser.add_argument("--adopt-orphan-turns", action="store_true",
                        help="同时认领 client_id 为空的孤儿轮次（默认不动）")
    parser.add_argument("--delete-old", action="store_true",
                        help="迁移后删除源身份行（仍被引用时不会删）。需同时给 --yes")
    args = parser.parse_args()

    src = args.from_patient.strip()
    dst_email = args.to_email.strip().lower()

    if args.delete_old and not args.yes:
        return fail("--delete-old 会删数据，必须同时给 --yes 明确确认")

    if not os.path.exists(DB_PATH):
        print(f"数据库不存在: {DB_PATH}")
        print("（服务尚未启动过，无需迁移）")
        return 0

    conn = sqlite3.connect(DB_PATH)
    try:
        problem = check_schema(conn)
        if problem:
            return fail(problem)

        # ── 校验：源身份 ────────────────────────────────────────────
        row = conn.execute(
            "SELECT id, name, COALESCE(access_hash, '') != '' AS legacy, "
            "COALESCE(password_hash, '') != '' AS account "
            "FROM patients WHERE id = ?",
            (src,),
        ).fetchone()
        if not row:
            return fail(
                f"找不到身份 {src}。"
                "若你确认它存在，多半是跑在了错误的库上 —— 容器化部署时库在 "
                "rag_state 命名卷里，宿主机上的 data/qa_history.db 是另一个空库。"
            )
        src_id, src_name, src_legacy, src_account = row
        print(f"源身份: {src_id}  ({src_name})")
        print(f"        类型: {'老患者（访问码）' if src_legacy else '邮箱账号'}")
        if src_account:
            print("        ⚠️ 源是一个可登录的账号 —— 这是「账号合并 / 找回」，"
                  "不是常规迁移。")

        # ── 校验：目标账号 ──────────────────────────────────────────
        row = conn.execute(
            "SELECT id, name, COALESCE(password_hash, '') != '' "
            "FROM patients WHERE lower(email) = ? AND email != ''",
            (dst_email,),
        ).fetchone()
        if not row:
            return fail(
                f"目标账号 {dst_email} 不存在。先在界面注册这个邮箱，再跑本脚本。"
            )
        dst_id, dst_name, dst_loginable = row
        if not dst_loginable:
            return fail(f"目标 {dst_email}（{dst_id}）没有设置密码，无法登录，拒绝迁入")
        if dst_id == src_id:
            return fail("源身份和目标账号是同一个 id，无需迁移")
        print(f"目标账号: {dst_id}  ({dst_name} <{dst_email}>)")

        # ── 要改什么 ────────────────────────────────────────────────
        before = counts_for(conn, src)
        after = counts_for(conn, dst_id)
        print(f"\n源名下:   {before['sessions']} 会话, {before['records']} 条问答"
              f", {before['orphans']} 条孤儿轮次")
        print(f"目标名下: {after['sessions']} 会话, {after['records']} 条问答")

        # 目标账号名下已有的孤儿轮次同样是 --adopt-orphan-turns 的认领对象，
        # 所以「源名下没东西」不足以直接退出：先跑过一次没带该标志、事后想补认领
        # 时正好撞上这个组合。早退会让它打印「无需迁移」，让人以为孤儿已被认领。
        adoptable = after["orphans"] if args.adopt_orphan_turns else 0
        if before["sessions"] == 0 and before["records"] == 0 and not adoptable:
            print("\n源身份名下没有任何会话或问答，无需迁移。")
            return 0

        if before["orphans"] and not args.adopt_orphan_turns:
            print(f"\n注意：有 {before['orphans']} 条孤儿轮次不会被迁移。"
                  "要一并认领请加 --adopt-orphan-turns。")

        if args.dry_run:
            print("\n--dry-run: 未做任何修改。")
            return 0

        if not args.yes:
            print(
                f"\n将把 {before['sessions']} 个会话与 {before['records']} 条问答"
                f"从 {src_id} 改挂到 {dst_id}。"
            )
            if adoptable:
                print(f"并将认领 {dst_id} 名下 {adoptable} 条孤儿轮次。")
            if input("确认迁移？输入 yes 继续: ").strip().lower() != "yes":
                print("已取消。")
                return 1

        # ── 写入：单事务，写不进去就整体回滚 ────────────────────────
        # isolation_level=None 关掉 Python 的隐式事务管理，好让下面的
        # BEGIN IMMEDIATE 真正生效：立即取写锁，若 api 还在写就快速失败，
        # 而不是先做一半再撞锁。
        conn.isolation_level = None
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("BEGIN IMMEDIATE")
        try:
            cur_s = conn.execute(
                "UPDATE sessions SET client_id = ? WHERE client_id = ?", (dst_id, src_id)
            )
            cur_r = conn.execute(
                "UPDATE qa_records SET client_id = ? WHERE client_id = ?", (dst_id, src_id)
            )

            adopted = 0
            if args.adopt_orphan_turns:
                cur_o = conn.execute(
                    "UPDATE qa_records SET client_id = :dst "
                    "WHERE COALESCE(client_id, '') = '' "
                    "AND session_id IN (SELECT id FROM sessions WHERE client_id = :dst)",
                    {"dst": dst_id},
                )
                adopted = cur_o.rowcount

            # 自检：迁移后，新账号名下的每个会话，其轮次的 client_id 必须一致。
            # 不一致说明迁移期间有并发写入（api 没停），回滚重来。
            #
            # 没给 --adopt-orphan-turns 时必须把孤儿轮次排除在外：它们的 client_id
            # 本来就是空串，且是迁移前就存在的状态，不是本次并发写出来的。不加这个
            # 条件，只要源数据里有孤儿轮次，正常完成的迁移也会被判失败并回滚，而且
            # 报错会指向「容器还在跑」这个错误的方向。并发写入者插入的轮次一定带着
            # 某个身份（src_id 或 dst_id），不会是空串，所以排除空串不会放过真竞态。
            orphan_clause = "" if args.adopt_orphan_turns else "AND COALESCE(q.client_id, '') != ''"
            mismatch = conn.execute(
                "SELECT COUNT(*) FROM qa_records q JOIN sessions s ON s.id = q.session_id "
                f"WHERE s.client_id = ? AND q.client_id != ? {orphan_clause}",
                (dst_id, dst_id),
            ).fetchone()[0]
            if mismatch:
                conn.execute("ROLLBACK")
                return fail(
                    f"自检失败：迁移后有 {mismatch} 条轮次的 client_id 与所属会话不一致，"
                    "已回滚。多半是 api 容器还在运行并写入 —— 停掉它再试。"
                )

            deleted = 0
            if args.delete_old:
                # 只有确实不再被任何行引用时才删，避免留下悬挂的 client_id
                cur_d = conn.execute(
                    "DELETE FROM patients WHERE id = ? "
                    "AND NOT EXISTS (SELECT 1 FROM sessions WHERE client_id = ?) "
                    "AND NOT EXISTS (SELECT 1 FROM qa_records WHERE client_id = ?)",
                    (src_id, src_id, src_id),
                )
                deleted = cur_d.rowcount
        except Exception:
            conn.execute("ROLLBACK")
            raise
        conn.commit()

        print(f"\n完成：{cur_s.rowcount} 个会话、{cur_r.rowcount} 条问答已改挂到 {dst_id}。")
        if adopted:
            print(f"      另外认领了 {adopted} 条孤儿轮次。")
        if args.delete_old:
            print(f"      源身份行{'已删除' if deleted else '未删除（仍被引用）'}。")
        print(f"      身份映射：{src_id} -> {dst_id}")
        print("\n提醒：Neo4j 图谱记忆完好；Milvus 语义记忆对新账号不可见（见本文件顶部说明）。")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
