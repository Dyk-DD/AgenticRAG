"""临时脚本：清空 Neo4j"""
import os
import sys

from neo4j import GraphDatabase

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
from config import DEFAULT_CONFIG  # noqa: E402

uri = DEFAULT_CONFIG.neo4j_uri
user = DEFAULT_CONFIG.neo4j_user
password = DEFAULT_CONFIG.neo4j_password

driver = GraphDatabase.driver(uri, auth=(user, password))
driver.verify_connectivity()

with driver.session() as session:
    count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    print(f"当前 Neo4j 节点数: {count}")
    session.run("MATCH (n) DETACH DELETE n")
    print("Neo4j 已清空")

driver.close()
