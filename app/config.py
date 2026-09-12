"""
基于图数据库的临床决策辅助RAG系统配置文件
"""

import os
from dataclasses import dataclass
from typing import Dict, Any

from dotenv import load_dotenv

# 在类体求值前加载 .env：本模块的连接类字段是在类定义时读取环境变量的，
# 而 app/api.py 先 `from main import ...` 才 load_dotenv，顺序上依赖不了调用方。
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT_DIR, ".env"), override=False)


def _env(name: str, default: str) -> str:
    """读环境变量，未设置或为空时回落到默认值。"""
    value = os.getenv(name)
    return value if value else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except ValueError:
        return default


@dataclass
class GraphRAGConfig:
    """基于图数据库的临床RAG系统配置类"""

    # === 医学数据路径配置 ===
    # 将包含 CSV 问答对的文件夹路径配置在这里
    medical_csv_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")

    # === Neo4j数据库配置 ===
    # 密码不设默认值：留空即报错，避免仓库里长期残留一份"能用"的凭据
    neo4j_uri: str = _env("NEO4J_URI", "bolt://localhost:8687")
    neo4j_user: str = _env("NEO4J_USER", "neo4j")
    neo4j_password: str = _env("NEO4J_PASSWORD", "")
    neo4j_database: str = _env("NEO4J_DATABASE", "neo4j")

    # === Milvus配置 ===
    milvus_host: str = _env("MILVUS_HOST", "localhost")
    milvus_port: int = _env_int("MILVUS_PORT", 19530)
    # 修改为临床(clinical)
    milvus_collection_name: str = _env("MILVUS_COLLECTION_NAME", "clinical_qa_knowledge")
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
    # ⚠️ 必须用非推理模型，不要改成 deepseek-v4-flash 一类会思考的模型。
    # 推理模型在正文之前先输出 reasoning_content，而思考与正文**共用同一个
    # max_tokens 预算**。本仓库没有任何一处读 reasoning_content，所以预算一旦
    # 被思考吃光，正文就是空字符串，且不抛任何异常。实测 deepseek-v4-flash
    # 回答一个普通医学问题时 reasoning_tokens 就占掉 1308/1453，于是表现为：
    #   - 回答只剩一句免责声明（正文为空，却被当成成功落库）
    #   - 路由与关键词抽取拿不到 JSON，静默降级为规则匹配
    #   - 双层检索因为拿不到关键词而恒返回 0 个结果
    # 这些都是"响亮失败"的反面：功能表面正常，指标全是假的。
    llm_model: str = _env("LLM_MODEL", "deepseek-chat")

    # === 检索配置（Round-robin策略） ===
    top_k: int = 5

    # === 生成配置 ===
    # 医疗场景需要严谨，建议保持较低温度(0.1)以防大模型产生幻觉
    temperature: float = 0.1
    # 生成答案的 token 上限（可用 LLM_MAX_TOKENS 覆盖）。
    # 注意这个值要真正生效，必须由调用方显式传进 GenerationIntegrationModule ——
    # 该类没有 self.config，漏传就会静默退回到构造函数里写死的默认值。
    max_tokens: int = _env_int("LLM_MAX_TOKENS", 4096)

    # === 图数据处理配置 ===
    # 虽然在医学问答(一条CSV一完整Chunk)中不截断，但保留兼容配置以防后续添加长篇医疗文献
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_graph_depth: int = 3  # 临床图遍历最大深度

    # === 三层实体抽取配置 ===
    ner_model_path: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "TCMNER")
    ner_enabled: bool = True
    ner_confidence_threshold: float = 0.5
    disease_dict_path: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "disease_dict_expanded.txt")
    llm_extraction_enabled: bool = True

    # === 记忆模块配置 ===
    memory_enabled: bool = True
    memory_buffer_size: int = 10
    memory_milvus_collection: str = "conversation_memory"
    memory_top_k: int = 3
    memory_summary_threshold: int = 10
    memory_batch_flush_size: int = 5
    memory_max_context_chars: int = 2000

    # 序列化时需要脱敏的字段（评估报告会写出整个配置）
    SECRET_FIELDS = ("neo4j_password",)

    def __post_init__(self):
        """初始化后的处理"""
        # LightRAG使用Round-robin策略，无需权重验证
        if not self.neo4j_password:
            import logging
            logging.getLogger(__name__).warning(
                "NEO4J_PASSWORD 未设置，Neo4j 连接将失败。请在项目根目录的 .env 中配置。"
            )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GraphRAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)

    def to_dict(self, redact_secrets: bool = False) -> Dict[str, Any]:
        """转换为字典。

        ``redact_secrets=True`` 时把凭据字段替换为 ``"***"``。评估报告会把
        本方法的返回值写进 JSON 并落盘，那些调用点必须传 True，
        否则 neo4j 密码会随报告一起外泄。
        """
        data = {
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
        if redact_secrets:
            for field in self.SECRET_FIELDS:
                if data.get(field):
                    data[field] = "***"
        return data

# 提供一个默认的实例化对象供全局导入调用
DEFAULT_CONFIG = GraphRAGConfig()