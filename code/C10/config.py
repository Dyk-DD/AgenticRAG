"""
基于图数据库的临床决策辅助RAG系统配置文件
"""

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class GraphRAGConfig:
    """基于图数据库的临床RAG系统配置类"""

    # === 医学数据路径配置 ===
    # 将包含 CSV 问答对的文件夹路径配置在这里
    medical_csv_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "C10", "Data_CSV")

    # === Neo4j数据库配置 ===
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "changeme"
    neo4j_database: str = "neo4j"

    # === Milvus配置 ===
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    # 将集合名称从烹饪(cooking)修改为临床(clinical)
    milvus_collection_name: str = "clinical_qa_knowledge"
    milvus_dimension: int = 768  # BGE-base-zh-v1.5的向量维度

    # === 模型配置 ===
    embedding_model: str = "../models/bge-base-zh-v1.5"
    llm_model: str = "deepseek-chat"

    # === 检索配置（Round-robin策略） ===
    top_k: int = 5

    # === 生成配置 ===
    # 医疗场景需要严谨，建议保持较低温度(0.1)以防大模型产生幻觉
    temperature: float = 0.1
    max_tokens: int = 2048

    # === 图数据处理配置 ===
    # 虽然在医学问答(一条CSV一完整Chunk)中不截断，但保留兼容配置以防后续添加长篇医疗文献
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_graph_depth: int = 2  # 临床图遍历最大深度

    def __post_init__(self):
        """初始化后的处理"""
        # LightRAG使用Round-robin策略，无需权重验证
        pass

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GraphRAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'medical_csv_dir': self.medical_csv_dir,
            'neo4j_uri': self.neo4j_uri,
            'neo4j_user': self.neo4j_user,
            'neo4j_password': self.neo4j_password,
            'neo4j_database': self.neo4j_database,
            'milvus_host': self.milvus_host,
            'milvus_port': self.milvus_port,
            'milvus_collection_name': self.milvus_collection_name,
            'milvus_dimension': self.milvus_dimension,
            'embedding_model': self.embedding_model,
            'llm_model': self.llm_model,
            'top_k': self.top_k,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'max_graph_depth': self.max_graph_depth
        }

# 提供一个默认的实例化对象供全局导入调用
DEFAULT_CONFIG = GraphRAGConfig()