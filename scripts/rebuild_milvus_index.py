#!/usr/bin/env python
"""
重建 Milvus 向量索引：
1. 删除已有的 Milvus 集合（clinical_qa_knowledge）
2. 重新构建向量索引（使用更新后的 page_content 编码）
3. 重建完成后初始化检索器

用法:
    conda activate all-in-rag
    python scripts/rebuild_milvus_index.py
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rebuild_milvus")

# 抑制第三方库的详细日志
for lib in ["neo4j", "httpx", "httpcore", "urllib3", "pymilvus", "jieba"]:
    logging.getLogger(lib).setLevel(logging.WARNING)


def main():
    from app.config import DEFAULT_CONFIG
    from app.rag_modules.milvus_index_construction import MilvusIndexConstructionModule

    # 1. 删除 Milvus 集合
    print("=" * 55)
    print("  步骤 1/3: 删除已有 Milvus 集合")
    print("=" * 55)

    index_module = MilvusIndexConstructionModule(
        host=DEFAULT_CONFIG.milvus_host,
        port=DEFAULT_CONFIG.milvus_port,
        collection_name=DEFAULT_CONFIG.milvus_collection_name,
        dimension=DEFAULT_CONFIG.milvus_dimension,
    )

    if index_module.has_collection():
        index_module.delete_collection()
        print("  [OK] 已删除 Milvus 集合: %s" % DEFAULT_CONFIG.milvus_collection_name)
    else:
        print("  [OK] Milvus 集合不存在，无需删除")

    # 2. 重新构建知识库（会重建 Milvus + Neo4j）
    print("\n" + "=" * 55)
    print("  步骤 2/3: 重新构建知识库（读取 CSV + 生成向量 + 写入 Milvus）")
    print("  当前 page_content 编码: 【核心问题】+【患者描述】")
    print("=" * 55)

    # 加载数据
    from app.rag_modules.graph_data_preparation import MedicalDataPreparationModule
    data_module = MedicalDataPreparationModule(
        config=DEFAULT_CONFIG,
        csv_dir=DEFAULT_CONFIG.medical_csv_dir,
        uri=DEFAULT_CONFIG.neo4j_uri,
        user=DEFAULT_CONFIG.neo4j_user,
        password=DEFAULT_CONFIG.neo4j_password,
        llm_client=None,
    )
    data_module.load_csv_data()
    chunks = data_module.build_chunks()

    print("  QA 总数: %d" % len(chunks))
    # 展示第一条 chunk 的 page_content 确认新编码生效
    if chunks:
        print("\n  page_content 编码示例（首条）:")
        print("  %s" % "-" * 50)
        print(chunks[0].page_content)
        print("  %s" % "-" * 50)
        print("  metadata.department: %s" % chunks[0].metadata.get("department"))
        print("  metadata.ask: %s" % chunks[0].metadata.get("ask")[:60] + "...")

    # 重建向量索引
    index_module.build_vector_index(chunks)

    # 3. 验证
    print("\n" + "=" * 55)
    print("  步骤 3/3: 验证索引状态")
    print("=" * 55)
    stats = index_module.get_collection_stats()
    print("  集合: %s" % stats.get("collection_name", "N/A"))
    print("  记录数: %d" % stats.get("row_count", 0))

    # 测试搜索一条
    print("\n  >> 快速检索测试...")
    results = index_module.similarity_search("乙肝大三阳会不会传染", k=3)
    if results:
        print("  检索成功! 返回 %d 条结果" % len(results))
        for i, r in enumerate(results):
            print("    #%d: score=%.4f | dept=%s | title=%s" % (
                i + 1, r["score"],
                r["metadata"].get("department", ""),
                r["metadata"].get("title", "")[:30],
            ))
    else:
        print("  [WARN] 检索无返回结果")

    print("\n" + "=" * 55)
    print("  Milvus 索引重建完成!")
    print("  编码策略: 【核心问题】+【患者描述】（去掉 department）")
    print("=" * 55)


if __name__ == "__main__":
    main()
