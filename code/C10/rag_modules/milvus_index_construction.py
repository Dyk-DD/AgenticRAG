"""
临床辅助系统 - Milvus索引构建模块
"""

import logging
import time
from typing import List, Dict, Any, Optional

from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class MilvusIndexConstructionModule:
    """Milvus索引构建模块 - 负责向量化和Milvus索引构建"""

    def __init__(self, 
                 host: str = "localhost", 
                 port: int = 19530,
                 collection_name: str = "clinical_qa_knowledge",
                 dimension: int = 768,
                 model_name: str = "../models/bge-base-zh-v1.5"):
        """
        初始化Milvus索引构建模块

        Args:
            host: Milvus服务器地址
            port: Milvus服务器端口
            collection_name: 集合名称
            dimension: 向量维度
            model_name: 嵌入模型名称
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self.model_name = model_name
        
        self.client = None
        self.embeddings = None
        self.collection_created = False
        
        self._setup_client()
        self._setup_embeddings()
    
    def _safe_truncate(self, text: str, max_length: int) -> str:
        """
        安全截取字符串，处理None值
        Args:
            text: 输入文本
            max_length: 最大长度
        Returns:
            截取后的字符串
        """
        if text is None:
            return ""
        return str(text)[:max_length]
    
    def _setup_client(self):
        """初始化Milvus客户端"""
        try:
            self.client = MilvusClient(
                uri=f"http://{self.host}:{self.port}"
            )
            logger.info(f"已连接到Milvus服务器: {self.host}:{self.port}")
            
            # 测试连接
            collections = self.client.list_collections()
            logger.info(f"连接成功，当前集合: {collections}")
            
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            raise

    def _setup_embeddings(self):
        """初始化嵌入模型，启用 CUDA 加速"""
        logger.info(f"正在初始化嵌入模型: {self.model_name}")

        # 检查是否有可用的 GPU
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            logger.info("检测到 GPU，已启用 CUDA 进行向量嵌入加速")
        else:
            logger.warning("未检测到可用 GPU，将回退至 CPU 运行")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': device},  # 将固定 'cpu' 改为变量 device
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 512  # 增加批处理大小，充分利用显存
            }
        )

        logger.info(f"嵌入模型初始化完成，运行设备: {device}")
    
    def _create_collection_schema(self) -> CollectionSchema:
        """
        创建集合模式
        Returns:
            集合模式对象
        """
        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=150, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=15000),

            # 医学专属元数据字段
            FieldSchema(name="node_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="department", dtype=DataType.VARCHAR, max_length=100),  # 科室
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=300),  # 主诉/标题
            # 👇 新增这两个字段，由于答案可能很长，max_length 设大一点（比如 8000）
            FieldSchema(name="ask", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=4000),
            FieldSchema(name="node_type", dtype=DataType.VARCHAR, max_length=100),  # 节点类型(如: MedicalQA)
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=50),  # 文档类型

            # 块管理字段
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=150),
            FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=100)
        ]

        schema = CollectionSchema(
            fields=fields,
            description="临床决策辅助问答向量集合"
        )
        return schema
    
    def create_collection(self, force_recreate: bool = False) -> bool:
        """
        创建Milvus集合
        Args:
            force_recreate: 是否强制重新创建集合
        
        Returns:
            是否创建成功
        """
        try:
            # 检查集合是否存在
            if self.client.has_collection(self.collection_name):
                if force_recreate:
                    logger.info(f"删除已存在的集合: {self.collection_name}")
                    self.client.drop_collection(self.collection_name)
                else:
                    logger.info(f"集合 {self.collection_name} 已存在")
                    self.collection_created = True
                    return True
            
            # 创建集合
            schema = self._create_collection_schema()
            
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                metric_type="COSINE",  # 使用余弦相似度
                consistency_level="Strong"
            )
            
            logger.info(f"成功创建集合: {self.collection_name}")
            self.collection_created = True
            
            return True
            
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False
    
    def create_index(self) -> bool:
        """
        创建向量索引
        Returns:
            是否创建成功
        """
        try:
            if not self.collection_created:
                raise ValueError("请先创建集合")
            
            # 使用prepare_index_params创建正确的IndexParams对象
            index_params = self.client.prepare_index_params()
            
            # 添加向量字段索引
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",
                metric_type="COSINE",
                params={
                    "M": 16,
                    "efConstruction": 200
                }
            )
            
            self.client.create_index(
                collection_name=self.collection_name,
                index_params=index_params
            )
            
            logger.info("向量索引创建成功")
            return True
            
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            return False

    def build_vector_index(self, chunks: List[Document]) -> bool:
        """
        构建向量索引 (优化为防OOM的分批生成与插入模式)
        """
        logger.info(f"正在构建Milvus向量索引，文档数量: {len(chunks)}...")

        if not chunks:
            raise ValueError("文档块列表不能为空")

        try:
            # 1. 创建集合（如果schema不兼容则强制重新创建）
            if not self.create_collection(force_recreate=True):
                return False

            # 2. 采用双重批处理逻辑防爆显存
            batch_size = 500  # 每次送入模型的文档数量
            total_chunks = len(chunks)

            logger.info("开始分批生成向量并写入Milvus...")

            for i in range(0, total_chunks, batch_size):
                # 截取当前批次的 chunks
                batch_chunks = chunks[i:i + batch_size]

                # 👇 提速核心：为生成向量专门准备截断后的文本 (最多保留前800个字符)
                # BGE 模型只能理解 512 Token，前 800 个字符已经包含了最核心的病状和提问
                embed_texts = [chunk.page_content[:800] for chunk in batch_chunks]

                # 仅对当前批次生成向量
                vectors = self.embeddings.embed_documents(embed_texts)

                # 准备当前批次的插入数据
                entities = []
                for j, (chunk, vector) in enumerate(zip(batch_chunks, vectors)):
                    global_idx = i + j
                    entity = {
                        "id": self._safe_truncate(chunk.metadata.get("chunk_id", f"chunk_{global_idx}"), 150),
                        "vector": vector,
                        # 👇 存入 Milvus 的仍然是完整的长文本 (不受 800 字限制)
                        "text": self._safe_truncate(chunk.page_content, 15000),

                        # 提取医学相关元数据
                        "node_id": self._safe_truncate(chunk.metadata.get("node_id", ""), 100),
                        "department": self._safe_truncate(chunk.metadata.get("department", "未知科室"), 100),
                        "title": self._safe_truncate(chunk.metadata.get("title", ""), 300),
                        # 👇 提取并截断新增的元数据
                        "ask": self._safe_truncate(chunk.metadata.get("ask", ""), 2000),
                        "answer": self._safe_truncate(chunk.metadata.get("answer", ""), 4000),
                        "node_type": self._safe_truncate(chunk.metadata.get("node_type", "MedicalQA"), 100),
                        "doc_type": self._safe_truncate(chunk.metadata.get("doc_type", "qa_pair"), 50),

                        "chunk_id": self._safe_truncate(chunk.metadata.get("chunk_id", f"chunk_{global_idx}"), 150),
                        "parent_id": self._safe_truncate(chunk.metadata.get("parent_id", ""), 100)
                    }
                    entities.append(entity)

                # 插入当前批次到 Milvus
                self.client.insert(
                    collection_name=self.collection_name,
                    data=entities
                )
                logger.info(f"进度: 已处理并插入 {min(i + batch_size, total_chunks)}/{total_chunks} 条数据")

            # 3. 创建索引
            if not self.create_index():
                return False

            # 4. 加载集合到内存
            self.client.load_collection(self.collection_name)
            logger.info("集合已加载到内存")

            # 5. 等待索引构建完成
            logger.info("等待索引构建完成...")
            time.sleep(2)

            logger.info(f"✅ 向量索引构建完成，总计包含 {total_chunks} 个向量")
            return True

        except Exception as e:
            logger.error(f"构建向量索引失败: {e}")
            return False

    def add_documents(self, new_chunks: List[Document]) -> bool:
        """
        向现有索引添加新文档 (分批处理防OOM)
        """
        if not self.collection_created:
            raise ValueError("请先构建向量索引")

        total_chunks = len(new_chunks)
        logger.info(f"正在分批添加 {total_chunks} 个新文档到索引...")

        try:
            batch_size = 500
            for i in range(0, total_chunks, batch_size):
                batch_chunks = new_chunks[i:i + batch_size]
                texts = [chunk.page_content[:800] for chunk in batch_chunks]

                # 分批生成向量
                vectors = self.embeddings.embed_documents(texts)

                entities = []
                for j, (chunk, vector) in enumerate(zip(batch_chunks, vectors)):
                    global_idx = i + j
                    entity = {
                        "id": self._safe_truncate(chunk.metadata.get("chunk_id", f"chunk_new_{global_idx}"), 150),
                        "vector": vector,
                        "text": self._safe_truncate(chunk.page_content, 15000),

                        "node_id": self._safe_truncate(chunk.metadata.get("node_id", ""), 100),
                        "department": self._safe_truncate(chunk.metadata.get("department", "未知科室"), 100),
                        "title": self._safe_truncate(chunk.metadata.get("title", ""), 300),
                        # 👇 提取并截断新增的元数据
                        "ask": self._safe_truncate(chunk.metadata.get("ask", ""), 2000),
                        "answer": self._safe_truncate(chunk.metadata.get("answer", ""), 4000),
                        "node_type": self._safe_truncate(chunk.metadata.get("node_type", "MedicalQA"), 100),
                        "doc_type": self._safe_truncate(chunk.metadata.get("doc_type", "qa_pair"), 50),

                        "chunk_id": self._safe_truncate(chunk.metadata.get("chunk_id", f"chunk_new_{global_idx}"), 150),
                        "parent_id": self._safe_truncate(chunk.metadata.get("parent_id", ""), 100)
                    }
                    entities.append(entity)

                # 插入数据
                self.client.insert(
                    collection_name=self.collection_name,
                    data=entities
                )
                logger.info(f"追加进度: {min(i + batch_size, total_chunks)}/{total_chunks}")

            logger.info("✅ 新文档添加完成")
            return True

        except Exception as e:
            logger.error(f"添加新文档失败: {e}")
            return False
    def similarity_search(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[
        Dict[str, Any]]:
        if not self.collection_created:
            raise ValueError("请先构建或加载向量索引")

        try:
            query_vector = self.embeddings.embed_query(query)

            filter_expr = ""
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        filter_conditions.append(f'{key} == "{value}"')
                    elif isinstance(value, (int, float)):
                        filter_conditions.append(f'{key} == {value}')
                    elif isinstance(value, list):
                        if all(isinstance(v, str) for v in value):
                            value_str = '", "'.join(value)
                            filter_conditions.append(f'{key} in ["{value_str}"]')
                        else:
                            value_str = ', '.join(map(str, value))
                            filter_conditions.append(f'{key} in [{value_str}]')

                if filter_conditions:
                    filter_expr = " and ".join(filter_conditions)

            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 64}
            }

            search_kwargs = {
                "collection_name": self.collection_name,
                "data": [query_vector],
                "anns_field": "vector",
                "limit": k,
                # 调整返回字段匹配新的Schema
                "output_fields": ["text", "node_id", "department", "title",
                                  "ask", "answer", # <-- 增加这两个
                                  "node_type", "doc_type", "chunk_id", "parent_id"],
                "search_params": search_params
            }

            if filter_expr:
                search_kwargs["filter"] = filter_expr

            results = self.client.search(**search_kwargs)

            formatted_results = []
            if results and len(results) > 0:
                for hit in results[0]:
                    result = {
                        "id": hit["id"],
                        "score": hit["distance"],
                        "text": hit["entity"]["text"],
                        "metadata": {
                            "node_id": hit["entity"]["node_id"],
                            "department": hit["entity"]["department"],
                            "title": hit["entity"]["title"],
                            "ask": hit["entity"].get("ask", ""),       # <-- 增加
                            "answer": hit["entity"].get("answer", ""), # <-- 增加
                            "node_type": hit["entity"]["node_type"],
                            "doc_type": hit["entity"]["doc_type"],
                            "chunk_id": hit["entity"]["chunk_id"],
                            "parent_id": hit["entity"]["parent_id"]
                        }
                    }
                    formatted_results.append(result)

            return formatted_results

        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息
        Returns:
            统计信息字典
        """
        try:
            if not self.collection_created:
                return {"error": "集合未创建"}
            
            stats = self.client.get_collection_stats(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "row_count": stats.get("row_count", 0),
                "index_building_progress": stats.get("index_building_progress", 0),
                "stats": stats
            }
            
        except Exception as e:
            logger.error(f"获取集合统计信息失败: {e}")
            return {"error": str(e)}
    
    def delete_collection(self) -> bool:
        """
        删除集合
        Returns:
            是否删除成功
        """
        try:
            if self.client.has_collection(self.collection_name):
                self.client.drop_collection(self.collection_name)
                logger.info(f"集合 {self.collection_name} 已删除")
                self.collection_created = False
                return True
            else:
                logger.info(f"集合 {self.collection_name} 不存在")
                return True
                
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            return False
    
    def has_collection(self) -> bool:
        """
        检查集合是否存在
        Returns:
            集合是否存在
        """
        try:
            return self.client.has_collection(self.collection_name)
        except Exception as e:
            logger.error(f"检查集合存在性失败: {e}")
            return False
    
    def load_collection(self) -> bool:
        """
        加载集合到内存
        Returns:
            是否加载成功
        """
        try:
            if not self.client.has_collection(self.collection_name):
                logger.error(f"集合 {self.collection_name} 不存在")
                return False
            
            self.client.load_collection(self.collection_name)
            self.collection_created = True
            logger.info(f"集合 {self.collection_name} 已加载到内存")
            return True
            
        except Exception as e:
            logger.error(f"加载集合失败: {e}")
            return False
    
    def close(self):
        """关闭连接"""
        if hasattr(self, 'client') and self.client:
            # Milvus客户端不需要显式关闭
            logger.info("Milvus连接已关闭")
    
    def __del__(self):
        """析构函数"""
        self.close() 