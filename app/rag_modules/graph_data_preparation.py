"""
图数据库数据准备模块
"""

import logging
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import os
import re
import hashlib
from neo4j import GraphDatabase
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    """图节点数据结构"""
    node_id: str
    labels: List[str]
    name: str
    properties: Dict[str, Any]

@dataclass
class GraphRelation:
    """图关系数据结构"""
    start_node_id: str
    end_node_id: str
    relation_type: str
    properties: Dict[str, Any]


class MedicalDataPreparationModule:
    def __init__(self, config, csv_dir: str, uri: str, user: str, password: str, database: str = "neo4j"):
        self.config = config
        self.csv_dir = csv_dir
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None

        self.documents = []
        self.chunks = []
        self.qa_pairs = []

        # 1. 加载疾病本地词典并编译正则（O(N) 匹配，替代 O(N×D) 暴力循环）
        disease_dict_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data", "disease_dict.txt")
        self.disease_dict = self._load_disease_dict(disease_dict_path)
        self._disease_pattern = self._build_disease_pattern()
        logger.info(f"成功加载本地疾病词典，共包含 {len(self.disease_dict)} 个疾病条目")

        self._connect()

    def _load_disease_dict(self, filepath: str) -> list:
        "加载本地疾病词典"
        if not os.path.exists(filepath):
            logger.warning(f"未找到疾病词典文件: {filepath}")
            return []
        with open(filepath, 'r', encoding='utf-8') as f:
            return sorted(set([line.strip() for line in f if line.strip()]), key=len, reverse=True)

    def _build_disease_pattern(self):
        "编译疾病词典为单个正则，利用 re 内部 trie 优化实现 O(N) 文本扫描"
        if not self.disease_dict:
            return None
        escaped = [re.escape(d) for d in self.disease_dict]
        return re.compile('|'.join(escaped))

    def _extract_diseases_by_dict(self, title: str, ask: str, answer: str) -> list:
        "编译正则单次扫描提取所有匹配疾病"
        combined_text = f"{title} {ask} {answer}"
        if not self._disease_pattern:
            return []
        matches = self._disease_pattern.findall(combined_text)
        return list(set(matches))
    
    def _connect(self):
        """建立Neo4j连接"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password),
                database=self.database
            )
            logger.info(f"已连接到Neo4j数据库: {self.uri}")
            
            # 测试连接
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_result = result.single()
                if test_result:
                    logger.info("Neo4j连接测试成功")
                    
        except Exception as e:
            logger.error(f"连接Neo4j失败: {e}")
            raise

    def check_graph_needs_update(self) -> bool:
        """通过对比本地CSV数据量与Neo4j元数据，判断是否需要更新图谱"""
        if not self.driver:
            return True
        try:
            with self.driver.session() as session:
                local_count = len(self.qa_pairs)

                # 1. 优先检查系统元数据节点 (记录了之前处理过的CSV总行数)
                meta_res = session.run("MATCH (m:SystemMeta {id:'kb_meta'}) RETURN m.total_csv_lines AS total")
                meta_record = meta_res.single()

                if meta_record and meta_record["total"] >= local_count:
                    logger.info(
                        f"✅ Neo4j 图谱已同步 (基于元数据标记，已处理 {meta_record['total']} 条原始数据)。跳过图谱构建。")
                    return False

                # 2. 如果元数据不存在，降级检查节点数量 (旧逻辑兜底)
                result = session.run("MATCH (c:Consultation) RETURN count(c) as count")
                db_count = result.single()["count"]

                if db_count > 0 and db_count >= local_count:
                    return False
                else:
                    logger.info(f"💡 检测到图谱需同步！本地CSV数据量({local_count})，准备执行图谱构建...")
                    return True
        except Exception as e:
            logger.error(f"检查 Neo4j 图谱状态失败: {e}")
            return True
    
    def close(self):
        """关闭数据库连接"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭")

    def load_csv_data(self):
        """解析 Data_CSV 文件夹及其所有子文件夹下的所有问答对"""
        logger.info(f"正在扫描目录: {self.csv_dir}")

        # 使用 os.walk 递归遍历所有子文件夹
        for root, dirs, files in os.walk(self.csv_dir):
            for file in files:
                if file.endswith('.csv'):
                    path = os.path.join(root, file)
                    logger.info(f"正在读取文件: {path}")
                    try:
                        df = pd.read_csv(path)
                        # 为了防止某些CSV缺少特定列报错，这里加入容错处理
                        for _, row in df.iterrows():
                            self.qa_pairs.append({
                                "department": row.get('department', '未知科室'),
                                "title": row.get('title', ''),
                                "ask": row.get('ask', ''),
                                "answer": row.get('answer', '')
                            })
                    except Exception as e:
                        logger.error(f"读取 CSV 文件 {path} 失败: {e}")

        logger.info(f"递归加载完成，共加载了 {len(self.qa_pairs)} 组医学问答对")

    # 3. 修改写入图数据库的逻辑
    def build_neo4j_graph(self, batch_size: int = 2000):
        """将 CSV 中的医学问答数据批量写入 Neo4j 图数据库"""
        logger.info(f"正在执行词典匹配并分批构建 Neo4j 图谱 (Batch Size: {batch_size})...")
        if not getattr(self, 'qa_pairs', None):
            logger.warning("没有可用的问答数据")
            return

        try:
            with self.driver.session() as session:
                # 建立索引和唯一约束（确保幂等性）
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Department) REQUIRE d.name IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Consultation) REQUIRE c.nodeId IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (dis:Disease) REQUIRE dis.name IS UNIQUE")

            total_pairs = len(self.qa_pairs)
            skipped_count = 0
            success_count = 0

            # 用于暂存在内存中的当前批次数据
            batch_data = []

            for idx, qa in enumerate(self.qa_pairs):
                # 1. 执行词典匹配疾病
                diseases = self._extract_diseases_by_dict(qa['title'], qa['ask'], qa['answer'])

                if not diseases:
                    skipped_count += 1
                else:
                    # 获取哈希ID
                    node_id = self._generate_node_id(qa)

                    # 组装单条数据字典
                    batch_data.append({
                        "node_id": node_id,
                        "department": qa.get('department', '未知科室'),
                        "title": qa.get('title', ''),
                        "ask": qa.get('ask', ''),
                        "answer": qa.get('answer', ''),
                        "diseases": [d.strip() for d in diseases]
                    })
                    success_count += 1

                # 2. 当攒够了 batch_size，或是遍历到最后一条时，执行一次批量写入
                if len(batch_data) >= batch_size or idx == total_pairs - 1:
                    if batch_data:
                        with self.driver.session() as session:
                            # 批量写入1: 使用 UNWIND 极速创建科室、病例及其关联
                            base_cypher = """
                            UNWIND $batch AS data
                            MERGE (d:Department {name: data.department})
                            MERGE (c:Consultation {nodeId: data.node_id})
                            SET c.title = data.title, 
                                c.ask = data.ask, 
                                c.answer = data.answer, 
                                c.department = data.department
                            MERGE (c)-[:BELONGS_TO_DEPT]->(d)
                            """
                            session.run(base_cypher, batch=batch_data)

                            # 批量写入2: 使用 UNWIND 双层循环极速创建疾病节点及关联
                            disease_cypher = """
                            UNWIND $batch AS data
                            UNWIND data.diseases AS disease_name
                            MATCH (c:Consultation {nodeId: data.node_id})
                            MERGE (dis:Disease {name: disease_name})
                            MERGE (c)-[:MENTIONS_DISEASE]->(dis)
                            """
                            session.run(disease_cypher, batch=batch_data)

                        logger.info(
                            f"进度: 已处理 {idx + 1}/{total_pairs}, 成功写入 {success_count} 条, 跳过 {skipped_count} 条")

                        # 清空内存中的当前批次，准备下一次收集
                        batch_data = []

            # 3. 循环结束后，统一写入同步标记（使用上次修复的标记逻辑）
            with self.driver.session() as session:
                session.run("""
                    MERGE (m:SystemMeta {id:'kb_meta'}) 
                    SET m.total_csv_lines = CASE 
                        WHEN m.total_csv_lines IS NULL THEN $total
                        WHEN $total >= m.total_csv_lines THEN $total
                        ELSE m.total_csv_lines + $total
                    END
                    """, total=total_pairs)

            logger.info(f"✅ Neo4j 图谱构建完成！总计成功: {success_count} 条, 因未匹配到疾病跳过: {skipped_count} 条")

        except Exception as e:
            logger.error(f"❌ 批量写入 Neo4j 失败: {e}")

    def build_medical_documents(self) -> List[Document]:
        "构建医学问答文档（保留兼容接口，内部调用合并方法）"
        return self.build_chunks()

    def chunk_documents(self, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
        "封装医学问答块（保留兼容接口，内部调用合并方法）"
        return self.build_chunks()

    def build_chunks(self) -> List[Document]:
        "一次遍历直接生成 chunks：合并 build_medical_documents + chunk_documents，减少全量遍历"
        logger.info("正在构建医学问答块（文档+chunk 合并单次遍历）...")
        chunks = []

        for qa in self.qa_pairs:
            try:
                content = (
                    f"【历史病例_所属科室】：{qa['department']}\n"
                    f"【历史病例_核心问题】：{qa['title']}\n"
                    f"【历史病例_患者主诉/提问】：{qa['ask']}"
                )
                node_id = self._generate_node_id(qa)

                chunk = Document(
                    page_content=content,
                    metadata={
                        "node_id": node_id,
                        "department": qa.get('department', '未知科室'),
                        "title": qa.get('title', ''),
                        "ask": qa.get('ask', ''),
                        "answer": qa.get('answer', ''),
                        "node_type": "MedicalQA",
                        "doc_type": "chunk",
                        "chunk_id": f"{node_id}_full",
                        "parent_id": node_id,
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "chunk_size": len(content),
                        "content_length": len(content),
                    }
                )
                chunks.append(chunk)
            except Exception as e:
                logger.warning(f"构建文档块失败: {e}")
                continue

        self.documents = chunks
        self.chunks = chunks
        logger.info(f"问答块构建完成，共 {len(chunks)} 条完整数据")
        return chunks

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取医学问答数据统计信息

        Returns:
            统计信息字典
        """
        # 1. 基础数据量统计
        stats = {
            'total_qa_pairs': len(getattr(self, 'qa_pairs', [])),  # CSV解析出的问答对总数
            'total_documents': len(self.documents),  # 转换成的文档总数
            'total_chunks': len(self.chunks)  # 封装好的块总数（由于采用1对1策略，通常与documents相等）
        }

        if self.documents:
            # 2. 医学特征分类统计
            departments = {}

            for doc in self.documents:
                # 按科室统计数据分布
                department = doc.metadata.get('department', '未知科室')
                departments[department] = departments.get(department, 0) + 1

            # 3. 汇总长度与分布信息
            stats.update({
                'departments_distribution': departments,  # 科室分布情况
                'total_departments': len(departments),  # 涉及的科室总数
                'avg_content_length': sum(doc.metadata.get('content_length', 0) for doc in self.documents) / len(
                    self.documents),
                'avg_chunk_size': sum(chunk.metadata.get('chunk_size', 0) for chunk in self.chunks) / len(
                    self.chunks) if self.chunks else 0
            })

        return stats

    def _generate_node_id(self, qa: dict) -> str:
        """生成唯一的节点ID，基于内容哈希，确保增量更新时不会主键冲突"""
        unique_str = f"{qa.get('department', '')}_{qa.get('title', '')}_{qa.get('ask', '')}_{qa.get('answer', '')}"
        return "qa_" + hashlib.md5(unique_str.encode('utf-8')).hexdigest()[:12]

    def load_single_csv(self, file_path: str):
        """专门用于增量加载单个新CSV文件"""
        self.qa_pairs = []  # 清空旧数据载体，只处理本次新增的数据
        logger.info(f"正在读取新增数据文件: {file_path}")
        try:
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                self.qa_pairs.append({
                    "department": row.get('department', '未知科室'),
                    "title": row.get('title', ''),
                    "ask": row.get('ask', ''),
                    "answer": row.get('answer', '')
                })
            logger.info(f"增量加载完成，共读取 {len(self.qa_pairs)} 条新问答对")
        except Exception as e:
            logger.error(f"读取新增 CSV 文件失败: {e}")

    def __del__(self):
        """析构函数，确保关闭连接"""
        self.close() 