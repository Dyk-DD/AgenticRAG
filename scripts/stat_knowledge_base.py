"""
知识图谱和向量数据库统计脚本

统计 Neo4j 图谱和 Milvus 向量库的状态，输出到 txt 文件。
用于监控知识库的增长和结构变化。

用法：
  python scripts/stat_knowledge_base.py                              # 默认覆盖写入
  python scripts/stat_knowledge_base.py --append                     # 追加模式
  python scripts/stat_knowledge_base.py --output custom_report.txt   # 自定义输出路径

输出：
  data/knowledge_base_stats.txt  — 知识库统计报告
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Any

from neo4j import GraphDatabase
from pymilvus import MilvusClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 抑制第三方库的详细日志
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
logging.getLogger("pymilvus").setLevel(logging.WARNING)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "data", "knowledge_base_stats.txt")

# ========== 数据库连接配置（统一取自 app/config.py，凭据走环境变量） ==========
sys.path.insert(0, os.path.join(PROJECT_ROOT, "app"))
from config import DEFAULT_CONFIG  # noqa: E402

NEO4J_URI = DEFAULT_CONFIG.neo4j_uri
NEO4J_USER = DEFAULT_CONFIG.neo4j_user
NEO4J_PASSWORD = DEFAULT_CONFIG.neo4j_password
NEO4J_DATABASE = DEFAULT_CONFIG.neo4j_database

MILVUS_URI = f"http://{DEFAULT_CONFIG.milvus_host}:{DEFAULT_CONFIG.milvus_port}"
MILVUS_COLLECTION = DEFAULT_CONFIG.milvus_collection_name


# ========== Neo4j 统计 ==========

def stat_neo4j_node_labels(driver) -> List[Dict[str, Any]]:
    """按标签统计节点数量"""
    query = """
    MATCH (n)
    RETURN labels(n) AS label, count(*) AS count
    ORDER BY count DESC
    """
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(query)
        return [{"label": "/".join(r["label"]), "count": r["count"]} for r in result]


def stat_neo4j_relation_types(driver) -> List[Dict[str, Any]]:
    """按类型统计关系数量"""
    query = """
    MATCH ()-[r]->()
    RETURN type(r) AS type, count(*) AS count
    ORDER BY count DESC
    """
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(query)
        return [{"type": r["type"], "count": r["count"]} for r in result]


def stat_neo4j_disease_top10(driver) -> List[Dict[str, Any]]:
    """疾病关联问答数 TOP 10"""
    query = """
    MATCH (d:Disease)<-[:MENTIONS_DISEASE]-(c:Consultation)
    RETURN d.name AS disease, count(c) AS consultation_count
    ORDER BY consultation_count DESC
    LIMIT 10
    """
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(query)
        return [{"disease": r["disease"], "count": r["consultation_count"]} for r in result]


def stat_neo4j_department_counts(driver) -> List[Dict[str, Any]]:
    """科室关联问答数统计"""
    query = """
    MATCH (d:Department)<-[:BELONGS_TO_DEPT]-(c:Consultation)
    RETURN d.name AS department, count(c) AS consultation_count
    ORDER BY consultation_count DESC
    """
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(query)
        return [{"department": r["department"], "count": r["consultation_count"]} for r in result]


def stat_neo4j_total_counts(driver) -> Dict[str, int]:
    """总节点数和总关系数"""
    query_nodes = "MATCH (n) RETURN count(n) AS total"
    query_rels = "MATCH ()-[r]->() RETURN count(r) AS total"
    with driver.session(database=NEO4J_DATABASE) as session:
        total_nodes = session.run(query_nodes).single()["total"]
        total_rels = session.run(query_rels).single()["total"]
    return {"total_nodes": total_nodes, "total_relations": total_rels}


def collect_neo4j_stats() -> Dict[str, Any]:
    """收集所有 Neo4j 统计信息"""
    stats = {}
    logger.info("正在连接 Neo4j...")
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
        connection_timeout=10,
    )
    try:
        driver.verify_connectivity()
        logger.info("Neo4j 连接成功")

        stats["totals"] = stat_neo4j_total_counts(driver)
        stats["node_labels"] = stat_neo4j_node_labels(driver)
        stats["relation_types"] = stat_neo4j_relation_types(driver)
        stats["disease_top10"] = stat_neo4j_disease_top10(driver)
        stats["department_counts"] = stat_neo4j_department_counts(driver)
        logger.info(
            f"Neo4j 统计完成：{stats['totals']['total_nodes']} 节点, "
            f"{stats['totals']['total_relations']} 关系"
        )
    except Exception as e:
        logger.error(f"Neo4j 统计失败: {e}")
        stats["error"] = str(e)
    finally:
        driver.close()
    return stats


# ========== Milvus 统计 ==========

def collect_milvus_stats() -> Dict[str, Any]:
    """收集 Milvus 向量库统计信息"""
    stats = {}
    logger.info("正在连接 Milvus...")
    try:
        client = MilvusClient(uri=MILVUS_URI)
        # 检查集合是否存在
        if not client.has_collection(MILVUS_COLLECTION):
            logger.warning(f"Milvus 集合 '{MILVUS_COLLECTION}' 不存在")
            return {"error": f"集合 '{MILVUS_COLLECTION}' 不存在"}

        # 集合统计信息
        raw_stats = client.get_collection_stats(MILVUS_COLLECTION)
        stats["collection_name"] = MILVUS_COLLECTION
        stats["row_count"] = raw_stats.get("row_count", 0)
        stats["index_building_progress"] = raw_stats.get("index_building_progress", "N/A")

        # 集合描述信息（获取向量维度等 schema 信息）
        desc = client.describe_collection(MILVUS_COLLECTION)
        for field in desc.get("fields", []):
            if field.get("name") == "vector":
                stats["dimension"] = field.get("params", {}).get("dim", "N/A")
                break
        else:
            stats["dimension"] = "N/A"

        # 集合中的所有集合列表（快速健康检查）
        all_collections = client.list_collections()
        stats["all_collections"] = all_collections

        logger.info(
            f"Milvus 统计完成：{stats['row_count']} 行, "
            f"维度 {stats.get('dimension', 'N/A')}"
        )
    except Exception as e:
        logger.error(f"Milvus 统计失败: {e}")
        stats["error"] = str(e)
    return stats


# ========== 报告格式化与输出 ==========

def format_report(neo4j_stats: Dict[str, Any], milvus_stats: Dict[str, Any]) -> str:
    """格式化为易读的文本报告"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("=" * 56)
    lines.append("          知识库统计报告")
    lines.append(f"  生成时间: {now}")
    lines.append("=" * 56)
    lines.append("")

    # ---- Neo4j 部分 ----
    lines.append("【Neo4j 知识图谱】")
    lines.append("-" * 40)

    if "error" in neo4j_stats:
        lines.append(f"  ⚠️  连接失败: {neo4j_stats['error']}")
    else:
        totals = neo4j_stats.get("totals", {})
        lines.append(f"  总节点数: {totals.get('total_nodes', 0)}")
        lines.append(f"  总关系数: {totals.get('total_relations', 0)}")
        lines.append("")

        # 节点类型分布
        lines.append("  节点类型分布:")
        for item in neo4j_stats.get("node_labels", []):
            lines.append(f"    {item['label']:>20}: {item['count']}")
        lines.append("")

        # 关系类型分布
        lines.append("  关系类型分布:")
        for item in neo4j_stats.get("relation_types", []):
            lines.append(f"    {item['type']:>25}: {item['count']}")
        lines.append("")

        # 疾病 TOP 10
        lines.append("  疾病关联问答 TOP 10:")
        for i, item in enumerate(neo4j_stats.get("disease_top10", []), 1):
            lines.append(f"    {i:>2}. {item['disease']:<20} ({item['count']} 条)")
        lines.append("")

        # 科室分布
        lines.append("  科室问答分布:")
        for item in neo4j_stats.get("department_counts", []):
            lines.append(f"    {item['department']:<16}: {item['count']} 条")

    lines.append("")

    # ---- Milvus 部分 ----
    lines.append("【Milvus 向量数据库】")
    lines.append("-" * 40)

    if "error" in milvus_stats:
        lines.append(f"  ⚠️  连接失败: {milvus_stats['error']}")
    else:
        lines.append(f"  集合名称: {milvus_stats.get('collection_name', 'N/A')}")
        lines.append(f"  数据行数: {milvus_stats.get('row_count', 0)}")
        lines.append(f"  向量维度: {milvus_stats.get('dimension', 'N/A')}")
        progress = milvus_stats.get("index_building_progress", "N/A")
        if isinstance(progress, (int, float)):
            lines.append(f"  索引进度: {progress:.1%}")
        else:
            lines.append(f"  索引进度: {progress}")

        # 其他集合
        all_cols = milvus_stats.get("all_collections", [])
        if all_cols:
            lines.append(f"  全部集合: {', '.join(all_cols)}")

    lines.append("")
    lines.append("=" * 56)
    lines.append(f"报告结束")
    lines.append("=" * 56)

    return "\n".join(lines)


def write_report(report: str, output_path: str, append: bool = False):
    """写入报告到文件"""
    mode = "a" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        if append:
            f.write("\n\n")
        f.write(report + "\n")
    logger.info(f"报告已写入: {output_path}")


# ========== 主入口 ==========

def main():
    parser = argparse.ArgumentParser(
        description="知识图谱和向量数据库统计脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python scripts/stat_knowledge_base.py\n"
            "  python scripts/stat_knowledge_base.py --append\n"
            "  python scripts/stat_knowledge_base.py --output custom_report.txt\n"
        ),
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"输出文件路径 (默认: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="追加模式（默认覆盖写入）",
    )
    args = parser.parse_args()

    logger.info("=" * 40)
    logger.info("开始统计知识库状态")
    logger.info("=" * 40)

    # 收集 Neo4j 统计
    neo4j_stats = collect_neo4j_stats()

    # 收集 Milvus 统计
    milvus_stats = collect_milvus_stats()

    # 生成报告
    report = format_report(neo4j_stats, milvus_stats)

    # 写入文件
    write_report(report, args.output, append=args.append)

    # 控制台也输出一份摘要
    logger.info("\n" + report)


if __name__ == "__main__":
    main()
