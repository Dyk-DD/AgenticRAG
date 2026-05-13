from neo4j import GraphDatabase


def clean_neo4j_database():
    """
    一键清空 Neo4j 图数据库中的所有节点和关系
    """
    # 这里的配置项与你的 config.py 保持一致
    uri = "bolt://localhost:8687"
    user = "neo4j"
    password = "changeme"  # 如果你修改过密码，请在这里同步修改

    try:
        print(f"🔗 正在连接 Neo4j 数据库 ({uri})...")
        driver = GraphDatabase.driver(uri, auth=(user, password))

        # 测试连接是否通畅
        driver.verify_connectivity()

        with driver.session() as session:
            print("🧹 开始清理所有图节点和关联边...")
            # DETACH DELETE 会先解除节点的所有关联边，然后再删除节点本身
            session.run("MATCH (n) DETACH DELETE n")

        print("✅ 成功清空 Neo4j 图数据库中的所有数据！")

    except Exception as e:
        print(f"❌ 清理 Neo4j 失败: {e}")
    finally:
        if 'driver' in locals():
            driver.close()


if __name__ == "__main__":
    # ⚠️ 警告：运行此脚本将不可逆地删除 Neo4j 中的所有数据
    confirm = input("⚠️ 警告：这将彻底清空 Neo4j 图数据库中的所有医学数据，是否继续？(y/N): ").strip().lower()
    if confirm == 'y':
        clean_neo4j_database()
    else:
        print("操作已取消。")