from pymilvus import MilvusClient


def clean_broken_collection():
    try:
        # 连接本地 Milvus
        client = MilvusClient(uri="http://localhost:19530")

        collections = [
            "clinical_qa_knowledge",   # 主知识库向量索引
            "conversation_memory",     # 对话记忆向量索引
        ]

        for name in collections:
            if client.has_collection(name):
                client.drop_collection(name)
                print(f"[OK] 已删除集合: {name}")
            else:
                print(f"集合 {name} 不存在，跳过。")

        print("\n如需重建，重新运行 main.py 或重启 API 即可自动构建。")

    except Exception as e:
        print(f"[ERROR] 清理失败: {e}")


if __name__ == "__main__":
    clean_broken_collection()