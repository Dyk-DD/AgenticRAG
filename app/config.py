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
    medical_csv_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")

    # === Neo4j数据库配置 ===
    neo4j_uri: str = "bolt://localhost:8687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "changeme"
    neo4j_database: str = "neo4j"

    # === Milvus配置 ===
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    # 修改为临床(clinical)
    milvus_collection_name: str = "clinical_qa_knowledge"
    milvus_dimension: int = 768  # BGE-base-zh-v1.5的向量维度

    # === 模型配置 ===
    embedding_model: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "bge-base-zh-v1.5")
    embedding_dimension: int = 768  # bge-base-zh-v1.5=768, BGE-M3=1024
    #
    # 如需切换至 BGE-M3（需手动下载完整权重 2.27GB）：
    #   1. 设置 embedding_model = "models/bge-m3"
    #   2. 设置 embedding_dimension = 1024
    #   3. 删除 Milvus 集合重建索引（维度不兼容）
    #
    llm_model: str = "deepseek-v4-flash"

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
    max_graph_depth: int = 3  # 临床图遍历最大深度

    # === 记忆模块配置 ===
    memory_enabled: bool = True
    memory_buffer_size: int = 10
    memory_milvus_collection: str = "conversation_memory"
    memory_top_k: int = 3
    memory_summary_threshold: int = 10
    memory_batch_flush_size: int = 5
    memory_max_context_chars: int = 2000

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
            'max_graph_depth': self.max_graph_depth,
            'memory_enabled': self.memory_enabled,
            'memory_buffer_size': self.memory_buffer_size,
            'memory_milvus_collection': self.memory_milvus_collection,
            'memory_top_k': self.memory_top_k,
            'memory_summary_threshold': self.memory_summary_threshold,
            'memory_batch_flush_size': self.memory_batch_flush_size,
            'memory_max_context_chars': self.memory_max_context_chars
        }

# 提供一个默认的实例化对象供全局导入调用
DEFAULT_CONFIG = GraphRAGConfig()