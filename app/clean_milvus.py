from pymilvus import MilvusClient


def clean_broken_collection():
    try:
        # 连接本地 Milvus
        client = MilvusClient(uri="http://localhost:19530")
        collection_name = "clinical_qa_knowledge"

        # 如果存在则删除
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)
            print(f"✅ 成功删除半成品集合: {collection_name}")
        else:
            print(f"集合 {collection_name} 不存在。")

    except Exception as e:
        print(f"❌ 清理失败: {e}")


if __name__ == "__main__":
    clean_broken_collection()